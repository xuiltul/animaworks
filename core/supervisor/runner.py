# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Child process entry point for Anima subprocess.

Usage:
    python -m core.supervisor.runner \\
        --anima-name sakura \\
        --socket-path ~/.animaworks/run/sockets/sakura.sock \\
        --animas-dir ~/.animaworks/animas \\
        --shared-dir ~/.animaworks/shared
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Union

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.anima import DigitalAnima
from core.memory.streaming_journal import StreamingJournal
from core.schedule_parser import parse_cron_md, parse_schedule, parse_heartbeat_config
from core.schemas import CronTask
from core.supervisor.ipc import IPCServer, IPCRequest, IPCResponse

logger = logging.getLogger(__name__)

_MSG_HEARTBEAT_COOLDOWN_S = 60
_CASCADE_WINDOW_S = 600   # 10 minutes
_CASCADE_THRESHOLD = 4     # max round-trips per pair within window
_DEFAULT_KEEPALIVE_INTERVAL = 30  # fallback keep-alive interval in seconds
_PENDING_WATCHER_POLL_INTERVAL = 3.0
_PENDING_TASK_SUBPROCESS_TIMEOUT = 1800


class _Sentinel:
    """Queue termination marker for keep-alive merge."""

    __slots__ = ()


_SENTINEL = _Sentinel()


# ── AnimaRunner ──────────────────────────────────────────────────

class AnimaRunner:
    """
    Runner for a single Anima in a child process.

    Starts a DigitalAnima instance and exposes it via Unix socket IPC.
    """

    def __init__(
        self,
        anima_name: str,
        socket_path: Path,
        animas_dir: Path,
        shared_dir: Path
    ):
        self.anima_name = anima_name
        self.socket_path = socket_path
        self.animas_dir = animas_dir
        self.shared_dir = shared_dir

        self._anima_dir = animas_dir / anima_name

        self.anima: DigitalAnima | None = None
        self.ipc_server: IPCServer | None = None
        self.inbox_watcher_task: asyncio.Task | None = None
        self.pending_task_watcher_task: asyncio.Task | None = None
        self.scheduler: AsyncIOScheduler | None = None
        self.shutdown_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._started_at = datetime.now()

        # Rate-limit state for inbox watcher
        self._pending_trigger: bool = False
        self._deferred_timer: asyncio.Handle | None = None
        self._last_msg_heartbeat_end: float = 0.0
        self._pair_heartbeat_times: dict[tuple[str, str], list[float]] = {}

        # Overlap prevention flags
        self._heartbeat_running: bool = False
        self._cron_running: set[str] = set()

    async def run(self) -> None:
        """
        Run the anima process.

        Starts IPC server first (creates socket immediately), then
        initializes DigitalAnima (heavy RAG/model loading).
        The parent process can connect to the socket early and poll
        readiness via the ``ping`` method.
        """
        try:
            # Start IPC server first so the socket is created immediately.
            # This allows the parent process to detect the socket quickly
            # while the heavy DigitalAnima initialization proceeds.
            self.ipc_server = IPCServer(
                socket_path=self.socket_path,
                request_handler=self._handle_request
            )
            await self.ipc_server.start()

            logger.info("Initializing Anima: %s", self.anima_name)

            # Initialize DigitalAnima (heavy: RAG indexer, model loading)
            self.anima = DigitalAnima(
                anima_dir=self._anima_dir,
                shared_dir=self.shared_dir
            )
            self.anima.set_on_lock_released(
                lambda: asyncio.ensure_future(self._on_anima_lock_released())
            )

            # Wire on_message_sent callback for WebSocket event emission
            def _on_message_sent(from_name: str, to_name: str, content: str) -> None:
                self._emit_event("anima.interaction", {
                    "from_person": from_name,
                    "to_person": to_name,
                    "type": "message",
                    "summary": content[:200],
                })

            self.anima.set_on_message_sent(_on_message_sent)

            # Crash recovery: check for orphaned streaming journal
            self._recover_streaming_journal()

            self._ready_event.set()

            # Start autonomous scheduler (heartbeat + cron)
            self._setup_scheduler()

            # Start inbox watcher
            self.inbox_watcher_task = asyncio.create_task(
                self._inbox_watcher_loop()
            )

            # Start pending task watcher (picks up animaworks-tool submit)
            self.pending_task_watcher_task = asyncio.create_task(
                self._pending_task_watcher_loop()
            )

            logger.info("Anima process ready: %s", self.anima_name)

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            logger.info("Shutting down: %s", self.anima_name)

        except Exception as e:
            logger.exception("Fatal error in AnimaRunner: %s", e)
            sys.exit(1)

        finally:
            await self._cleanup()

    def _recover_streaming_journal(self) -> None:
        """Recover partial response from an orphaned streaming journal.

        If the previous process crashed during streaming, the journal
        file survives on disk.  Read it, record the partial response in
        conversation memory, and log the crash event.
        """
        if not StreamingJournal.has_orphan(self._anima_dir):
            return

        recovery = StreamingJournal.recover(self._anima_dir)
        if recovery is None:
            return

        logger.warning(
            "Recovered streaming journal for %s: %d chars, %d tool calls, trigger=%s",
            self.anima_name,
            len(recovery.recovered_text),
            len(recovery.tool_calls),
            recovery.trigger,
        )

        # Record recovered text in conversation memory
        if recovery.recovered_text and self.anima:
            try:
                from core.memory.conversation import ConversationMemory
                conv_memory = ConversationMemory(
                    self._anima_dir,
                    self.anima.model_config,
                )
                saved_text = (
                    recovery.recovered_text
                    + "\n[プロセスクラッシュにより応答が中断されました]"
                )
                conv_memory.append_turn("assistant", saved_text)
                conv_memory.save()
                logger.info(
                    "Recovered %d chars into conversation memory for %s",
                    len(recovery.recovered_text),
                    self.anima_name,
                )
            except Exception:
                logger.exception(
                    "Failed to save recovered journal to conversation memory: %s",
                    self.anima_name,
                )

        # Record crash event in activity log
        try:
            from core.memory.activity import ActivityLogger
            activity = ActivityLogger(self._anima_dir)
            activity.log(
                "error",
                summary="プロセスクラッシュにより応答が中断されました",
                meta={
                    "recovered_chars": len(recovery.recovered_text),
                    "trigger": recovery.trigger,
                    "tool_calls": len(recovery.tool_calls),
                    "from_person": recovery.from_person,
                    "started_at": recovery.started_at,
                    "last_event_at": recovery.last_event_at,
                },
            )
        except Exception:
            logger.debug(
                "Failed to log crash recovery to activity log: %s",
                self.anima_name,
                exc_info=True,
            )

    # ── Event Emission ─────────────────────────────────────────────

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Write an event file for the parent process to pick up."""
        import time as _time
        events_dir = self.shared_dir.parent / "run" / "events" / self.anima_name
        events_dir.mkdir(parents=True, exist_ok=True)
        # Use monotonic timestamp for uniqueness
        filename = f"{_time.time_ns()}.json"
        event = {"event": event_type, "data": data}
        tmp = events_dir / f".{filename}"
        tmp.write_text(json.dumps(event, default=str, ensure_ascii=False), encoding="utf-8")
        tmp.rename(events_dir / filename)  # Atomic rename

    # ── Autonomous Scheduler ────────────────────────────────────────

    def _setup_scheduler(self) -> None:
        """Set up and start the autonomous scheduler for heartbeat and cron."""
        if not self.anima:
            return

        try:
            self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
            self._setup_heartbeat()
            self._setup_cron_tasks()
            self.scheduler.start()

            # Wire up hot-reload callback
            self.anima.set_on_schedule_changed(self._reload_schedule)

            job_count = len(self.scheduler.get_jobs())
            logger.info(
                "Scheduler started for %s: %d jobs registered",
                self.anima_name, job_count,
            )
        except Exception:
            logger.exception("Failed to setup scheduler for %s", self.anima_name)
            self.scheduler = None

    def _setup_heartbeat(self) -> None:
        """Register heartbeat job from heartbeat.md."""
        if not self.anima or not self.scheduler:
            return

        config = self.anima.memory.read_heartbeat_config()
        if not config:
            return

        interval, active_start, active_end = parse_heartbeat_config(config)

        self.scheduler.add_job(
            self._heartbeat_tick,
            CronTrigger(
                minute=f"*/{interval}",
                hour=f"{active_start}-{active_end - 1}",
            ),
            id=f"{self.anima_name}_heartbeat",
            name=f"{self.anima_name} heartbeat",
            replace_existing=True,
            misfire_grace_time=300,
            max_instances=1,
        )
        logger.info(
            "Heartbeat registered: %s every %dmin, active %d:00-%d:00",
            self.anima_name, interval, active_start, active_end,
        )

    async def _heartbeat_tick(self) -> None:
        """Execute a scheduled heartbeat."""
        if not self.anima:
            return
        if self._heartbeat_running:
            logger.info("Scheduled heartbeat SKIPPED (already running): %s", self.anima_name)
            return
        self._heartbeat_running = True
        try:
            logger.info("Scheduled heartbeat: %s", self.anima_name)
            result = await self.anima.run_heartbeat()
            # Notify parent for WebSocket broadcast
            self._emit_event("anima.heartbeat", {
                "name": self.anima_name,
                "result": result.model_dump(),
            })
        except Exception:
            logger.exception("Scheduled heartbeat failed: %s", self.anima_name)
        finally:
            self._heartbeat_running = False

    def _setup_cron_tasks(self) -> None:
        """Register cron jobs from cron.md."""
        if not self.anima or not self.scheduler:
            return

        config = self.anima.memory.read_cron_config()
        if not config:
            return

        tasks = parse_cron_md(config)
        for i, task in enumerate(tasks):
            trigger = parse_schedule(task.schedule)
            if not trigger:
                logger.warning(
                    "Could not parse schedule for cron task '%s': '%s'",
                    task.name, task.schedule,
                )
                continue

            self.scheduler.add_job(
                self._cron_tick,
                trigger,
                id=f"{self.anima_name}_cron_{i}",
                name=f"{self.anima_name}: {task.name}",
                args=[task],
                replace_existing=True,
                misfire_grace_time=300,
                max_instances=1,
            )
            logger.info(
                "Cron registered: %s -> %s (%s) [%s]",
                self.anima_name, task.name, task.schedule, task.type,
            )

    async def _cron_tick(self, task: CronTask) -> None:
        """Execute a scheduled cron task."""
        if not self.anima:
            return

        if task.name in self._cron_running:
            logger.info(
                "Scheduled cron SKIPPED (already running): %s -> %s",
                self.anima_name, task.name,
            )
            return

        logger.info("Scheduled cron: %s -> %s [%s]", self.anima_name, task.name, task.type)
        # Run in separate task to avoid blocking other scheduled jobs
        asyncio.create_task(
            self._run_cron_task(task),
            name=f"cron-{self.anima_name}-{task.name}",
        )

    async def _run_cron_task(self, task: CronTask) -> None:
        """Run a single cron task (LLM or command type)."""
        if not self.anima:
            return
        self._cron_running.add(task.name)
        try:
            if task.type == "llm":
                result = await self.anima.run_cron_task(task.name, task.description)
                self._emit_event("anima.cron", {
                    "name": self.anima_name,
                    "task": task.name,
                    "task_type": "llm",
                    "result": result.model_dump(),
                })
            elif task.type == "command":
                result = await self.anima.run_cron_command(
                    task.name,
                    command=task.command,
                    tool=task.tool,
                    args=task.args,
                )
                self._emit_event("anima.cron", {
                    "name": self.anima_name,
                    "task": task.name,
                    "task_type": "command",
                    "result": result,
                })
                # If command produced non-empty output, write to
                # pending.md and trigger a heartbeat so the LLM can
                # review and act on the results.
                stdout = result.get("stdout", "").strip()
                if stdout and result.get("exit_code", 0) == 0:
                    # Check skip_pattern: if stdout matches, suppress heartbeat
                    if task.skip_pattern:
                        try:
                            if re.search(task.skip_pattern, stdout):
                                logger.info(
                                    "Cron command '%s' output matched skip_pattern, "
                                    "suppressing heartbeat for %s",
                                    task.name, self.anima_name,
                                )
                                return
                        except re.error as e:
                            logger.warning(
                                "Invalid skip_pattern '%s' for task '%s': %s — "
                                "continuing without skip",
                                task.skip_pattern, task.name, e,
                            )

                    # NOTE: update_pending() overwrites the file. If multiple
                    # command-type cron tasks run concurrently, only the last
                    # result is retained.
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    pending = (
                        f"## cron結果: {task.name} ({ts})\n\n"
                        f"{stdout}\n"
                    )
                    self.anima.memory.update_pending(pending)
                    logger.info(
                        "Cron command '%s' produced output, triggering heartbeat for %s",
                        task.name, self.anima_name,
                    )
                    asyncio.create_task(
                        self._heartbeat_tick(),
                        name=f"cron-triggered-heartbeat-{self.anima_name}",
                    )
            else:
                logger.warning("Unknown cron type '%s' for task '%s'", task.type, task.name)
        except Exception:
            logger.exception("Cron task failed: %s -> %s", self.anima_name, task.name)
        finally:
            self._cron_running.discard(task.name)

    def _reload_schedule(self, name: str) -> dict[str, Any]:
        """Reload heartbeat and cron schedules from disk (hot-reload callback)."""
        if not self.scheduler:
            return {"error": "Scheduler not running"}

        # Remove all existing jobs
        removed = 0
        for job in self.scheduler.get_jobs():
            job.remove()
            removed += 1

        # Re-setup from current files
        self._setup_heartbeat()
        self._setup_cron_tasks()

        new_jobs = [j.id for j in self.scheduler.get_jobs()]
        logger.info(
            "Schedule reloaded for %s: removed=%d, new_jobs=%s",
            self.anima_name, removed, new_jobs,
        )
        return {"reloaded": name, "removed": removed, "new_jobs": new_jobs}

    # ── IPC Handlers ──────────────────────────────────────────────

    async def _handle_request(
        self, request: IPCRequest
    ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
        """
        Handle incoming IPC request.

        Dispatches to appropriate handler based on method.
        For streaming requests (process_message with stream=True), returns
        an AsyncIterator[IPCResponse] instead of a single IPCResponse.
        """
        try:
            # Check for streaming process_message
            if (
                request.method == "process_message"
                and request.params.get("stream")
            ):
                return self._handle_process_message_stream(request)

            handler = self._get_handler(request.method)
            if not handler:
                return IPCResponse(
                    id=request.id,
                    error={
                        "code": "UNKNOWN_METHOD",
                        "message": f"Unknown method: {request.method}"
                    }
                )

            result = await handler(request.params)
            return IPCResponse(id=request.id, result=result)

        except Exception as e:
            logger.exception("Error handling request %s: %s", request.method, e)
            return IPCResponse(
                id=request.id,
                error={
                    "code": "EXECUTION_ERROR",
                    "message": str(e)
                }
            )

    def _get_handler(self, method: str) -> Callable[..., Awaitable[dict[str, Any]]] | None:
        """Get handler for method."""
        handlers = {
            "process_message": self._handle_process_message,
            "greet": self._handle_greet,
            "run_bootstrap": self._handle_run_bootstrap,
            "run_heartbeat": self._handle_run_heartbeat,
            "run_cron_task": self._handle_run_cron_task,
            "get_status": self._handle_get_status,
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown,
        }
        return handlers.get(method)

    async def _handle_process_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle non-streaming process_message request."""
        if not self.anima:
            raise RuntimeError("Anima not initialized")

        message = params.get("message", "")
        from_person = params.get("from_person", "human")
        images = params.get("images") or None
        attachment_paths = params.get("attachment_paths") or None

        result = await self.anima.process_message(
            message, from_person=from_person,
            images=images, attachment_paths=attachment_paths,
        )

        return {
            "response": result,
            "replied_to": [],
            "notifications": self.anima.drain_notifications(),
        }

    async def _handle_greet(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle greet request (character click greeting)."""
        if not self.anima:
            raise RuntimeError("Anima not initialized")

        return await self.anima.process_greet()

    async def _handle_run_bootstrap(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle run_bootstrap request (background bootstrap execution)."""
        if not self.anima:
            raise RuntimeError("Anima not initialized")

        result = await self.anima.run_bootstrap()
        return {
            "status": "completed",
            "summary": result.summary,
            "duration_ms": result.duration_ms,
        }

    async def _handle_process_message_stream(
        self, request: IPCRequest
    ) -> AsyncIterator[IPCResponse]:
        """Handle streaming process_message request.

        Yields IPCResponse chunks with stream=True, followed by
        a final response with done=True containing the full result.

        Uses an asyncio.Queue to merge Agent SDK stream chunks and
        periodic keep-alive chunks so that the IPC layer's per-chunk
        timeout is reset even during long tool executions.
        """
        if not self.anima:
            yield IPCResponse(
                id=request.id,
                error={
                    "code": "NOT_INITIALIZED",
                    "message": "Anima not initialized"
                }
            )
            return

        # ── Resolve keep-alive interval from config ──────────────
        try:
            from core.config import load_config
            keepalive_interval: int = load_config().server.keepalive_interval
        except Exception:
            keepalive_interval = _DEFAULT_KEEPALIVE_INTERVAL

        message = request.params.get("message", "")
        from_person = request.params.get("from_person", "human")
        images = request.params.get("images") or None
        attachment_paths = request.params.get("attachment_paths") or None
        full_response = ""

        # Track bootstrap state to detect completion
        was_bootstrapping = self.anima.needs_bootstrap

        # ── Queue-based merge of SDK stream + keep-alive ─────────
        queue: asyncio.Queue[IPCResponse | _Sentinel] = asyncio.Queue()
        last_chunk_time = time.monotonic()
        stream_start_time = time.monotonic()

        async def _enqueue(resp: IPCResponse) -> None:
            """Put response on queue and update last-chunk timestamp."""
            nonlocal last_chunk_time
            last_chunk_time = time.monotonic()
            await queue.put(resp)

        async def _stream_producer() -> None:
            """Read Agent SDK stream and enqueue IPCResponse chunks."""
            nonlocal full_response
            try:
                # Emit bootstrap_start if anima needs bootstrap
                if was_bootstrapping:
                    await _enqueue(IPCResponse(
                        id=request.id,
                        stream=True,
                        chunk=json.dumps(
                            {"type": "bootstrap_start"}, ensure_ascii=False,
                        ),
                    ))

                async for chunk in self.anima.process_message_stream(
                    message, from_person=from_person,
                    images=images, attachment_paths=attachment_paths,
                ):
                    event_type = chunk.get("type", "unknown")

                    if event_type == "text_delta":
                        text = chunk.get("text", "")
                        full_response += text
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                    elif event_type == "cycle_done":
                        cycle_result = chunk.get("cycle_result", {})
                        full_response = cycle_result.get(
                            "summary", full_response,
                        )

                        # Emit bootstrap_complete if bootstrap just finished
                        if (
                            was_bootstrapping
                            and not self.anima.needs_bootstrap
                        ):
                            await _enqueue(IPCResponse(
                                id=request.id,
                                stream=True,
                                chunk=json.dumps(
                                    {"type": "bootstrap_complete"},
                                    ensure_ascii=False,
                                ),
                            ))

                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            done=True,
                            result={
                                "response": full_response,
                                "replied_to": [],
                                "cycle_result": cycle_result,
                            },
                        ))
                        return

                    elif event_type == "bootstrap_busy":
                        # Anima is already bootstrapping — forward as-is
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                    elif event_type == "error":
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                    else:
                        # Forward other event types (tool_start, tool_end,
                        # chain_start, etc.) as stream chunks
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(chunk, ensure_ascii=False),
                        ))

                # Stream ended without cycle_done — send final done
                await _enqueue(IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={
                        "response": full_response,
                        "replied_to": [],
                    },
                ))

            except TimeoutError as e:
                logger.error("Timeout in streaming process_message: %s", e)
                await queue.put(IPCResponse(
                    id=request.id,
                    error={
                        "code": "IPC_TIMEOUT",
                        "message": str(e) or "Stream processing timed out",
                    },
                ))
            except Exception as e:
                logger.exception(
                    "Error in streaming process_message: %s", e,
                )
                await queue.put(IPCResponse(
                    id=request.id,
                    error={
                        "code": "STREAM_ERROR",
                        "message": str(e),
                    },
                ))
            finally:
                await queue.put(_SENTINEL)

        async def _keepalive_producer() -> None:
            """Emit keep-alive chunks when Agent SDK stream is silent.

            Stops automatically when *producer_task* finishes (crash or
            normal exit), so that stale keep-alives do not mask a dead
            Agent SDK subprocess.
            """
            try:
                while True:
                    await asyncio.sleep(keepalive_interval)
                    # Stop if producer finished (SENTINEL already queued)
                    if producer_task.done():
                        logger.debug(
                            "Keepalive stopping: producer finished for %s",
                            self.anima_name,
                        )
                        return
                    elapsed_since_chunk = time.monotonic() - last_chunk_time
                    if elapsed_since_chunk >= keepalive_interval:
                        elapsed = round(
                            time.monotonic() - stream_start_time, 1,
                        )
                        logger.debug(
                            "Keep-alive sent for %s (elapsed=%.1fs)",
                            self.anima_name, elapsed,
                        )
                        await _enqueue(IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=json.dumps(
                                {"type": "keepalive", "elapsed_s": elapsed},
                                ensure_ascii=False,
                            ),
                        ))
            except asyncio.CancelledError:
                return

        # Launch producer tasks
        logger.debug(
            "Starting queue-based stream merge for %s (keepalive=%ds)",
            self.anima_name, keepalive_interval,
        )
        producer_task = asyncio.create_task(
            _stream_producer(),
            name=f"stream-producer-{self.anima_name}",
        )
        keepalive_task = asyncio.create_task(
            _keepalive_producer(),
            name=f"keepalive-{self.anima_name}",
        )

        try:
            # ── Main loop: drain queue and yield ─────────────────
            while True:
                item = await queue.get()
                if isinstance(item, _Sentinel):
                    break
                yield item
                # If this was a terminal response, stop immediately
                if item.done or item.error:
                    break
        finally:
            keepalive_task.cancel()
            # Ensure producer finishes; suppress CancelledError
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            logger.debug(
                "Stream merge completed for %s (%.1fs)",
                self.anima_name,
                time.monotonic() - stream_start_time,
            )

    async def _handle_run_heartbeat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle run_heartbeat request."""
        if not self.anima:
            raise RuntimeError("Anima not initialized")

        await self.anima.run_heartbeat()

        return {"status": "completed"}

    async def _handle_run_cron_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle run_cron_task request."""
        if not self.anima:
            raise RuntimeError("Anima not initialized")

        task_name = params.get("task_name")
        task_description = params.get("task_description")

        if not task_name:
            raise ValueError("task_name is required")

        await self.anima.run_cron_task(task_name, task_description)

        return {"status": "completed"}

    async def _handle_get_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle get_status request."""
        if not self.anima:
            raise RuntimeError("Anima not initialized")

        return {
            "status": self.anima._status,
            "current_task": self.anima._current_task or None,
            "needs_bootstrap": self.anima.needs_bootstrap,
            "scheduler_running": self.scheduler.running if self.scheduler else False,
            "scheduler_jobs": len(self.scheduler.get_jobs()) if self.scheduler else 0,
        }

    async def _handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ping request.

        Returns ``status: "initializing"`` while DigitalAnima is loading,
        ``status: "ok"`` once ready.  The parent process polls this to
        confirm readiness.
        """
        uptime = (datetime.now() - self._started_at).total_seconds()
        status = "ok" if self._ready_event.is_set() else "initializing"
        return {
            "status": status,
            "anima": self.anima_name,
            "uptime_sec": round(uptime, 1),
        }

    async def _handle_shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle shutdown request."""
        logger.info("Shutdown requested for %s", self.anima_name)
        self.shutdown_event.set()
        return {"status": "shutting_down"}

    # ── Inbox Rate Limiting ──────────────────────────────────────

    def _is_in_cooldown(self) -> bool:
        """Return True if a message-triggered heartbeat finished too recently."""
        return (time.monotonic() - self._last_msg_heartbeat_end) < _MSG_HEARTBEAT_COOLDOWN_S

    def _check_cascade(self, senders: set[str]) -> bool:
        """Return True if any (anima, sender) pair exceeds cascade threshold."""
        now = time.monotonic()
        for sender in senders:
            keys = [(self.anima_name, sender), (sender, self.anima_name)]
            total = 0
            for k in keys:
                times = self._pair_heartbeat_times.get(k, [])
                # Evict expired entries
                times = [t for t in times if now - t < _CASCADE_WINDOW_S]
                self._pair_heartbeat_times[k] = times
                if not times and k in self._pair_heartbeat_times:
                    del self._pair_heartbeat_times[k]
                total += len(times)
            if total >= _CASCADE_THRESHOLD:
                logger.warning(
                    "CASCADE DETECTED: %s <-> %s (%d round-trips in %ds window). "
                    "Suppressing message-triggered heartbeat.",
                    self.anima_name, sender, total, _CASCADE_WINDOW_S,
                )
                return True
        return False

    def _record_pair_heartbeat(self, senders: set[str]) -> None:
        """Record a heartbeat exchange for cascade tracking."""
        now = time.monotonic()
        for sender in senders:
            key = (self.anima_name, sender)
            self._pair_heartbeat_times.setdefault(key, []).append(now)

    async def _on_anima_lock_released(self) -> None:
        """Check deferred inbox after the anima's lock is released.

        If unread messages exist, schedule a deferred trigger to ensure
        they are processed even when cooldown is still active.
        """
        if not self.anima:
            return
        if not self.anima.messenger.has_unread():
            return
        if self._pending_trigger:
            return
        # Instead of giving up when in cooldown, schedule deferred trigger
        self._schedule_deferred_trigger()

    def _schedule_deferred_trigger(self) -> None:
        """Schedule a deferred heartbeat trigger after cooldown expires.

        Only one timer is maintained.  If a timer is already pending,
        the call is a no-op (the existing timer will fire and re-check
        the inbox).
        """
        if self._deferred_timer is not None:
            return  # already scheduled
        remaining = _MSG_HEARTBEAT_COOLDOWN_S - (
            time.monotonic() - self._last_msg_heartbeat_end
        )
        # If not in cooldown (e.g. lock-only), use a short retry delay
        delay = max(remaining, 2.0)
        loop = asyncio.get_running_loop()
        self._deferred_timer = loop.call_later(
            delay,
            lambda: asyncio.create_task(self._try_deferred_trigger()),
        )
        logger.debug(
            "Deferred trigger scheduled for %s in %.1fs",
            self.anima_name, delay,
        )

    async def _try_deferred_trigger(self) -> None:
        """Attempt to trigger a deferred heartbeat.

        Re-schedules itself if the anima is still blocked by cooldown
        or lock, ensuring messages are never forgotten.
        """
        self._deferred_timer = None
        if not self.anima:
            return
        if not self.anima.messenger.has_unread():
            return
        if self._pending_trigger:
            return
        if self._is_in_cooldown():
            self._schedule_deferred_trigger()
            return
        if self.anima._lock.locked():
            self._schedule_deferred_trigger()
            return
        self._pending_trigger = True
        asyncio.create_task(self._message_triggered_heartbeat())

    async def _message_triggered_heartbeat(self) -> None:
        """Execute a heartbeat triggered by incoming messages."""
        if not self.anima:
            self._pending_trigger = False
            return

        if self._heartbeat_running:
            logger.info("Message-triggered heartbeat SKIPPED (already running): %s", self.anima_name)
            self._pending_trigger = False
            return

        # Peek at inbox senders for cascade detection
        senders = {m.from_person for m in self.anima.messenger.receive()}
        if senders and self._check_cascade(senders):
            self._pending_trigger = False
            return

        self._heartbeat_running = True
        try:
            logger.info("Message-triggered heartbeat: %s", self.anima_name)
            await self.anima.run_heartbeat()
        except Exception:
            logger.exception(
                "Message-triggered heartbeat failed: %s", self.anima_name,
            )
        finally:
            self._heartbeat_running = False
            self._pending_trigger = False
            self._last_msg_heartbeat_end = time.monotonic()
            if senders:
                self._record_pair_heartbeat(senders)

    # ── Pending Task Watcher ──────────────────────────────────────

    async def _pending_task_watcher_loop(self) -> None:
        """Watch state/background_tasks/pending/ for submitted tasks.

        Tasks submitted via ``animaworks-tool submit`` are picked up here
        and executed through BackgroundTaskManager, outside the Anima lock.
        """
        pending_dir = self._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pending task watcher started for %s", self.anima_name)

        while not self.shutdown_event.is_set():
            try:
                for path in sorted(pending_dir.glob("*.json")):
                    try:
                        task_desc = json.loads(path.read_text(encoding="utf-8"))
                        path.unlink()  # Remove immediately to prevent double execution
                        logger.info(
                            "Picked up pending task: id=%s tool=%s subcmd=%s anima=%s",
                            task_desc.get("task_id", "?"),
                            task_desc.get("tool_name", "?"),
                            task_desc.get("subcommand", ""),
                            self.anima_name,
                        )
                        await self._execute_pending_task(task_desc)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Invalid JSON in pending task file: %s", path.name,
                        )
                        path.unlink(missing_ok=True)
                    except Exception:
                        logger.exception(
                            "Error processing pending task file: %s", path.name,
                        )

                await asyncio.sleep(_PENDING_WATCHER_POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "Error in pending task watcher for %s", self.anima_name,
                )
                await asyncio.sleep(_PENDING_WATCHER_POLL_INTERVAL)

        logger.info("Pending task watcher stopped for %s", self.anima_name)

    async def _execute_pending_task(self, task_desc: dict[str, Any]) -> None:
        """Execute a pending task via BackgroundTaskManager.

        Constructs the CLI-equivalent arguments and dispatches through
        ExternalToolDispatcher, running in background (outside _lock).
        """
        if not self.anima:
            logger.warning("Cannot execute pending task: anima not initialized")
            return

        bg_mgr = self.anima.agent.background_manager
        if not bg_mgr:
            logger.warning(
                "Cannot execute pending task: BackgroundTaskManager not available",
            )
            return

        tool_name = task_desc.get("tool_name", "")
        subcommand = task_desc.get("subcommand", "")
        raw_args = task_desc.get("raw_args", [])
        anima_dir = task_desc.get("anima_dir", str(self._anima_dir))

        # Build tool args dict for ExternalToolDispatcher
        # The dispatch expects (tool_name, args_dict) where args_dict
        # contains the raw CLI arguments
        tool_args = {
            "subcommand": subcommand,
            "raw_args": raw_args,
            "anima_dir": anima_dir,
        }

        # Use the task_id from the pending file if available
        task_id = task_desc.get("task_id", "")

        logger.info(
            "Submitting pending task to BackgroundTaskManager: "
            "id=%s tool=%s subcmd=%s",
            task_id, tool_name, subcommand,
        )

        def _dispatch_fn(name: str, args: dict[str, Any]) -> str:
            """Execute the tool via CLI subprocess (same as direct execution)."""
            import os
            import subprocess

            cmd = ["animaworks-tool", name]
            subcmd = args.get("subcommand", "")
            if subcmd:
                cmd.append(subcmd)
            cmd.extend(args.get("raw_args", []))
            # Remove subcommand from raw_args if it's already the first element
            if subcmd and args.get("raw_args") and args["raw_args"][0] == subcmd:
                cmd = ["animaworks-tool", name] + args["raw_args"]
            cmd.append("-j")

            env = {
                **os.environ,
                "ANIMAWORKS_ANIMA_DIR": args.get("anima_dir", ""),
            }

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_PENDING_TASK_SUBPROCESS_TIMEOUT, env=env,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
                raise RuntimeError(f"Tool {name} failed: {error_msg}")
            return result.stdout.strip()

        # Submit to BackgroundTaskManager
        # Use the composite key format for tracking
        composite_name = f"{tool_name}:{subcommand}" if subcommand else tool_name
        bg_mgr.submit(composite_name, tool_args, _dispatch_fn)

    # ── Inbox Watcher ──────────────────────────────────────────────

    async def _inbox_watcher_loop(self) -> None:
        """Poll inbox every 2s; trigger heartbeat on new messages.

        Applies rate limiting to prevent cascade loops between animas
        and cooldown to avoid excessive heartbeat triggers.
        """
        if not self.anima:
            return

        logger.info("Inbox watcher started for %s", self.anima_name)

        while not self.shutdown_event.is_set():
            try:
                if self._pending_trigger:
                    await asyncio.sleep(2.0)
                    continue
                if not self.anima.messenger.has_unread():
                    await asyncio.sleep(2.0)
                    continue
                if self._is_in_cooldown():
                    self._schedule_deferred_trigger()
                    await asyncio.sleep(2.0)
                    continue
                if self.anima._lock.locked():
                    self._schedule_deferred_trigger()
                    await asyncio.sleep(2.0)
                    continue

                self._pending_trigger = True
                asyncio.create_task(self._message_triggered_heartbeat())
                await asyncio.sleep(2.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in inbox watcher for %s: %s", self.anima_name, e,
                )
                await asyncio.sleep(2.0)

        logger.info("Inbox watcher stopped for %s", self.anima_name)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Cancel deferred trigger timer
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None

        # Stop inbox watcher
        if self.inbox_watcher_task:
            self.inbox_watcher_task.cancel()
            try:
                await self.inbox_watcher_task
            except asyncio.CancelledError:
                pass

        # Stop pending task watcher
        if self.pending_task_watcher_task:
            self.pending_task_watcher_task.cancel()
            try:
                await self.pending_task_watcher_task
            except asyncio.CancelledError:
                pass

        # Stop scheduler
        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:
                pass  # Scheduler may not have been started
            logger.info("Scheduler stopped for %s", self.anima_name)

        # Stop IPC server
        if self.ipc_server:
            await self.ipc_server.stop()

        logger.info("Cleanup completed for %s", self.anima_name)


# ── CLI Entry Point ────────────────────────────────────────────────

def setup_logging(anima_name: str, log_dir: Path) -> None:
    """Setup logging for child process with anima-specific log files."""
    from core.logging_config import setup_anima_logging

    setup_anima_logging(
        anima_name=anima_name,
        log_dir=log_dir,
        level="INFO",
        also_to_console=False  # Child processes log to file only
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an Anima in a subprocess"
    )
    parser.add_argument(
        "--anima-name",
        required=True,
        help="Name of the anima to run"
    )
    parser.add_argument(
        "--socket-path",
        required=True,
        type=Path,
        help="Path to Unix socket file"
    )
    parser.add_argument(
        "--animas-dir",
        required=True,
        type=Path,
        help="Path to animas directory"
    )
    parser.add_argument(
        "--shared-dir",
        required=True,
        type=Path,
        help="Path to shared directory"
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        type=Path,
        help="Path to log directory"
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    setup_logging(args.anima_name, args.log_dir)

    runner = AnimaRunner(
        anima_name=args.anima_name,
        socket_path=args.socket_path,
        animas_dir=args.animas_dir,
        shared_dir=args.shared_dir
    )

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
