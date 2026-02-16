"""
Child process entry point for Person subprocess.

Usage:
    python -m core.supervisor.runner \\
        --person-name sakura \\
        --socket-path ~/.animaworks/run/sockets/sakura.sock \\
        --persons-dir ~/.animaworks/persons \\
        --shared-dir ~/.animaworks/shared
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Union

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.person import DigitalPerson
from core.schedule_parser import parse_cron_md, parse_schedule, parse_heartbeat_config
from core.schemas import CronTask
from core.supervisor.ipc import IPCServer, IPCRequest, IPCResponse

logger = logging.getLogger(__name__)

_MSG_HEARTBEAT_COOLDOWN_S = 60
_CASCADE_WINDOW_S = 600   # 10 minutes
_CASCADE_THRESHOLD = 4     # max round-trips per pair within window
_DEFAULT_KEEPALIVE_INTERVAL = 30  # fallback keep-alive interval in seconds


class _Sentinel:
    """Queue termination marker for keep-alive merge."""

    __slots__ = ()


_SENTINEL = _Sentinel()


# ── PersonRunner ──────────────────────────────────────────────────

class PersonRunner:
    """
    Runner for a single Person in a child process.

    Starts a DigitalPerson instance and exposes it via Unix socket IPC.
    """

    def __init__(
        self,
        person_name: str,
        socket_path: Path,
        persons_dir: Path,
        shared_dir: Path
    ):
        self.person_name = person_name
        self.socket_path = socket_path
        self.persons_dir = persons_dir
        self.shared_dir = shared_dir

        self.person: DigitalPerson | None = None
        self.ipc_server: IPCServer | None = None
        self.inbox_watcher_task: asyncio.Task | None = None
        self.scheduler: AsyncIOScheduler | None = None
        self.shutdown_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._started_at = datetime.now()

        # Rate-limit state for inbox watcher
        self._pending_trigger: bool = False
        self._deferred_inbox: bool = False
        self._last_msg_heartbeat_end: float = 0.0
        self._pair_heartbeat_times: dict[tuple[str, str], list[float]] = {}

    async def run(self) -> None:
        """
        Run the person process.

        Starts IPC server first (creates socket immediately), then
        initializes DigitalPerson (heavy RAG/model loading).
        The parent process can connect to the socket early and poll
        readiness via the ``ping`` method.
        """
        try:
            # Start IPC server first so the socket is created immediately.
            # This allows the parent process to detect the socket quickly
            # while the heavy DigitalPerson initialization proceeds.
            self.ipc_server = IPCServer(
                socket_path=self.socket_path,
                request_handler=self._handle_request
            )
            await self.ipc_server.start()

            logger.info("Initializing Person: %s", self.person_name)

            # Initialize DigitalPerson (heavy: RAG indexer, model loading)
            person_dir = self.persons_dir / self.person_name
            self.person = DigitalPerson(
                person_dir=person_dir,
                shared_dir=self.shared_dir
            )
            self.person.set_on_lock_released(
                lambda: asyncio.ensure_future(self._on_person_lock_released())
            )
            self._ready_event.set()

            # Start autonomous scheduler (heartbeat + cron)
            self._setup_scheduler()

            # Start inbox watcher
            self.inbox_watcher_task = asyncio.create_task(
                self._inbox_watcher_loop()
            )

            logger.info("Person process ready: %s", self.person_name)

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            logger.info("Shutting down: %s", self.person_name)

        except Exception as e:
            logger.exception("Fatal error in PersonRunner: %s", e)
            sys.exit(1)

        finally:
            await self._cleanup()

    # ── Event Emission ─────────────────────────────────────────────

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Write an event file for the parent process to pick up."""
        import time as _time
        events_dir = self.shared_dir.parent / "run" / "events" / self.person_name
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
        if not self.person:
            return

        try:
            self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
            self._setup_heartbeat()
            self._setup_cron_tasks()
            self.scheduler.start()

            # Wire up hot-reload callback
            self.person.set_on_schedule_changed(self._reload_schedule)

            job_count = len(self.scheduler.get_jobs())
            logger.info(
                "Scheduler started for %s: %d jobs registered",
                self.person_name, job_count,
            )
        except Exception:
            logger.exception("Failed to setup scheduler for %s", self.person_name)
            self.scheduler = None

    def _setup_heartbeat(self) -> None:
        """Register heartbeat job from heartbeat.md."""
        if not self.person or not self.scheduler:
            return

        config = self.person.memory.read_heartbeat_config()
        if not config:
            return

        interval, active_start, active_end = parse_heartbeat_config(config)

        self.scheduler.add_job(
            self._heartbeat_tick,
            CronTrigger(
                minute=f"*/{interval}",
                hour=f"{active_start}-{active_end - 1}",
            ),
            id=f"{self.person_name}_heartbeat",
            name=f"{self.person_name} heartbeat",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info(
            "Heartbeat registered: %s every %dmin, active %d:00-%d:00",
            self.person_name, interval, active_start, active_end,
        )

    async def _heartbeat_tick(self) -> None:
        """Execute a scheduled heartbeat."""
        if not self.person:
            return
        try:
            logger.info("Scheduled heartbeat: %s", self.person_name)
            result = await self.person.run_heartbeat()
            # Notify parent for WebSocket broadcast
            self._emit_event("person.heartbeat", {
                "name": self.person_name,
                "result": result.model_dump(),
            })
        except Exception:
            logger.exception("Scheduled heartbeat failed: %s", self.person_name)

    def _setup_cron_tasks(self) -> None:
        """Register cron jobs from cron.md."""
        if not self.person or not self.scheduler:
            return

        config = self.person.memory.read_cron_config()
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
                id=f"{self.person_name}_cron_{i}",
                name=f"{self.person_name}: {task.name}",
                args=[task],
                replace_existing=True,
                misfire_grace_time=300,
            )
            logger.info(
                "Cron registered: %s -> %s (%s) [%s]",
                self.person_name, task.name, task.schedule, task.type,
            )

    async def _cron_tick(self, task: CronTask) -> None:
        """Execute a scheduled cron task."""
        if not self.person:
            return

        logger.info("Scheduled cron: %s -> %s [%s]", self.person_name, task.name, task.type)
        # Run in separate task to avoid blocking other scheduled jobs
        asyncio.create_task(
            self._run_cron_task(task),
            name=f"cron-{self.person_name}-{task.name}",
        )

    async def _run_cron_task(self, task: CronTask) -> None:
        """Run a single cron task (LLM or command type)."""
        if not self.person:
            return
        try:
            if task.type == "llm":
                result = await self.person.run_cron_task(task.name, task.description)
                self._emit_event("person.cron", {
                    "name": self.person_name,
                    "task": task.name,
                    "task_type": "llm",
                    "result": result.model_dump(),
                })
            elif task.type == "command":
                result = await self.person.run_cron_command(
                    task.name,
                    command=task.command,
                    tool=task.tool,
                    args=task.args,
                )
                self._emit_event("person.cron", {
                    "name": self.person_name,
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
                        import re as _re
                        if _re.search(task.skip_pattern, stdout):
                            logger.info(
                                "Cron command '%s' output matched skip_pattern, "
                                "suppressing heartbeat for %s",
                                task.name, self.person_name,
                            )
                            return

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    pending = (
                        f"## cron結果: {task.name} ({ts})\n\n"
                        f"{stdout}\n"
                    )
                    self.person.memory.update_pending(pending)
                    logger.info(
                        "Cron command '%s' produced output, triggering heartbeat for %s",
                        task.name, self.person_name,
                    )
                    asyncio.create_task(
                        self._heartbeat_tick(),
                        name=f"cron-triggered-heartbeat-{self.person_name}",
                    )
            else:
                logger.warning("Unknown cron type '%s' for task '%s'", task.type, task.name)
        except Exception:
            logger.exception("Cron task failed: %s -> %s", self.person_name, task.name)

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
            self.person_name, removed, new_jobs,
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
        if not self.person:
            raise RuntimeError("Person not initialized")

        message = params.get("message", "")
        from_person = params.get("from_person", "human")

        result = await self.person.process_message(message, from_person=from_person)

        return {
            "response": result,
            "replied_to": [],
            "notifications": self.person.drain_notifications(),
        }

    async def _handle_greet(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle greet request (character click greeting)."""
        if not self.person:
            raise RuntimeError("Person not initialized")

        return await self.person.process_greet()

    async def _handle_run_bootstrap(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle run_bootstrap request (background bootstrap execution)."""
        if not self.person:
            raise RuntimeError("Person not initialized")

        result = await self.person.run_bootstrap()
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
        if not self.person:
            yield IPCResponse(
                id=request.id,
                error={
                    "code": "NOT_INITIALIZED",
                    "message": "Person not initialized"
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
        full_response = ""

        # Track bootstrap state to detect completion
        was_bootstrapping = self.person.needs_bootstrap

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
                # Emit bootstrap_start if person needs bootstrap
                if was_bootstrapping:
                    await _enqueue(IPCResponse(
                        id=request.id,
                        stream=True,
                        chunk=json.dumps(
                            {"type": "bootstrap_start"}, ensure_ascii=False,
                        ),
                    ))

                async for chunk in self.person.process_message_stream(
                    message, from_person=from_person
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
                            and not self.person.needs_bootstrap
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
                        # Person is already bootstrapping — forward as-is
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
            """Emit keep-alive chunks when Agent SDK stream is silent."""
            try:
                while True:
                    await asyncio.sleep(keepalive_interval)
                    elapsed_since_chunk = time.monotonic() - last_chunk_time
                    if elapsed_since_chunk >= keepalive_interval:
                        elapsed = round(
                            time.monotonic() - stream_start_time, 1,
                        )
                        logger.debug(
                            "Keep-alive sent for %s (elapsed=%.1fs)",
                            self.person_name, elapsed,
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
            self.person_name, keepalive_interval,
        )
        producer_task = asyncio.create_task(
            _stream_producer(),
            name=f"stream-producer-{self.person_name}",
        )
        keepalive_task = asyncio.create_task(
            _keepalive_producer(),
            name=f"keepalive-{self.person_name}",
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
                self.person_name,
                time.monotonic() - stream_start_time,
            )

    async def _handle_run_heartbeat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle run_heartbeat request."""
        if not self.person:
            raise RuntimeError("Person not initialized")

        await self.person.run_heartbeat()

        return {"status": "completed"}

    async def _handle_run_cron_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle run_cron_task request."""
        if not self.person:
            raise RuntimeError("Person not initialized")

        task_name = params.get("task_name")
        task_description = params.get("task_description")

        if not task_name:
            raise ValueError("task_name is required")

        await self.person.run_cron_task(task_name, task_description)

        return {"status": "completed"}

    async def _handle_get_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle get_status request."""
        if not self.person:
            raise RuntimeError("Person not initialized")

        return {
            "status": self.person._status,
            "current_task": self.person._current_task or None,
            "needs_bootstrap": self.person.needs_bootstrap,
            "scheduler_running": self.scheduler.running if self.scheduler else False,
            "scheduler_jobs": len(self.scheduler.get_jobs()) if self.scheduler else 0,
        }

    async def _handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ping request.

        Returns ``status: "initializing"`` while DigitalPerson is loading,
        ``status: "ok"`` once ready.  The parent process polls this to
        confirm readiness.
        """
        uptime = (datetime.now() - self._started_at).total_seconds()
        status = "ok" if self._ready_event.is_set() else "initializing"
        return {
            "status": status,
            "person": self.person_name,
            "uptime_sec": round(uptime, 1),
        }

    async def _handle_shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle shutdown request."""
        logger.info("Shutdown requested for %s", self.person_name)
        self.shutdown_event.set()
        return {"status": "shutting_down"}

    # ── Inbox Rate Limiting ──────────────────────────────────────

    def _is_in_cooldown(self) -> bool:
        """Return True if a message-triggered heartbeat finished too recently."""
        return (time.monotonic() - self._last_msg_heartbeat_end) < _MSG_HEARTBEAT_COOLDOWN_S

    def _check_cascade(self, senders: set[str]) -> bool:
        """Return True if any (person, sender) pair exceeds cascade threshold."""
        now = time.monotonic()
        for sender in senders:
            keys = [(self.person_name, sender), (sender, self.person_name)]
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
                    self.person_name, sender, total, _CASCADE_WINDOW_S,
                )
                return True
        return False

    def _record_pair_heartbeat(self, senders: set[str]) -> None:
        """Record a heartbeat exchange for cascade tracking."""
        now = time.monotonic()
        for sender in senders:
            key = (self.person_name, sender)
            self._pair_heartbeat_times.setdefault(key, []).append(now)

    async def _on_person_lock_released(self) -> None:
        """Check deferred inbox after the person's lock is released."""
        if not self._deferred_inbox:
            return
        self._deferred_inbox = False

        if not self.person:
            return
        if not self.person.messenger.has_unread():
            return
        if self._pending_trigger:
            return
        if self._is_in_cooldown():
            return

        self._pending_trigger = True
        asyncio.create_task(self._message_triggered_heartbeat())

    async def _message_triggered_heartbeat(self) -> None:
        """Execute a heartbeat triggered by incoming messages."""
        if not self.person:
            self._pending_trigger = False
            return

        # Peek at inbox senders for cascade detection
        senders = {m.from_person for m in self.person.messenger.receive()}
        if senders and self._check_cascade(senders):
            self._pending_trigger = False
            return

        try:
            logger.info("Message-triggered heartbeat: %s", self.person_name)
            await self.person.run_heartbeat()
        except Exception:
            logger.exception(
                "Message-triggered heartbeat failed: %s", self.person_name,
            )
        finally:
            self._pending_trigger = False
            self._last_msg_heartbeat_end = time.monotonic()
            if senders:
                self._record_pair_heartbeat(senders)

    # ── Inbox Watcher ──────────────────────────────────────────────

    async def _inbox_watcher_loop(self) -> None:
        """Poll inbox every 2s; trigger heartbeat on new messages.

        Applies rate limiting to prevent cascade loops between persons
        and cooldown to avoid excessive heartbeat triggers.
        """
        if not self.person:
            return

        logger.info("Inbox watcher started for %s", self.person_name)

        while not self.shutdown_event.is_set():
            try:
                if self._pending_trigger:
                    await asyncio.sleep(2.0)
                    continue
                if not self.person.messenger.has_unread():
                    await asyncio.sleep(2.0)
                    continue
                if self._is_in_cooldown():
                    await asyncio.sleep(2.0)
                    continue
                if self.person._lock.locked():
                    self._deferred_inbox = True
                    await asyncio.sleep(2.0)
                    continue

                self._pending_trigger = True
                asyncio.create_task(self._message_triggered_heartbeat())
                await asyncio.sleep(2.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in inbox watcher for %s: %s", self.person_name, e,
                )
                await asyncio.sleep(2.0)

        logger.info("Inbox watcher stopped for %s", self.person_name)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Stop inbox watcher
        if self.inbox_watcher_task:
            self.inbox_watcher_task.cancel()
            try:
                await self.inbox_watcher_task
            except asyncio.CancelledError:
                pass

        # Stop scheduler
        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:
                pass  # Scheduler may not have been started
            logger.info("Scheduler stopped for %s", self.person_name)

        # Stop IPC server
        if self.ipc_server:
            await self.ipc_server.stop()

        logger.info("Cleanup completed for %s", self.person_name)


# ── CLI Entry Point ────────────────────────────────────────────────

def setup_logging(person_name: str, log_dir: Path) -> None:
    """Setup logging for child process with person-specific log files."""
    from core.logging_config import setup_person_logging

    setup_person_logging(
        person_name=person_name,
        log_dir=log_dir,
        level="INFO",
        also_to_console=False  # Child processes log to file only
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Person in a subprocess"
    )
    parser.add_argument(
        "--person-name",
        required=True,
        help="Name of the person to run"
    )
    parser.add_argument(
        "--socket-path",
        required=True,
        type=Path,
        help="Path to Unix socket file"
    )
    parser.add_argument(
        "--persons-dir",
        required=True,
        type=Path,
        help="Path to persons directory"
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

    setup_logging(args.person_name, args.log_dir)

    runner = PersonRunner(
        person_name=args.person_name,
        socket_path=args.socket_path,
        persons_dir=args.persons_dir,
        shared_dir=args.shared_dir
    )

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
