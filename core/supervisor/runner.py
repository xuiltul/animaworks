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
import sys
from pathlib import Path

from core.time_utils import ensure_aware, now_jst

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Union

from core.exceptions import ProcessError, MemoryWriteError, ExecutionError  # noqa: F401
from core.anima import DigitalAnima
from core.memory.streaming_journal import StreamingJournal
from core.supervisor.ipc import IPCServer, IPCRequest, IPCResponse
from core.supervisor.inbox_rate_limiter import InboxRateLimiter
from core.supervisor.pending_executor import PendingTaskExecutor
from core.supervisor.scheduler_manager import SchedulerManager
from core.supervisor.streaming_handler import StreamingIPCHandler

logger = logging.getLogger(__name__)

# ── AnimaRunner ──────────────────────────────────────────────────

class AnimaRunner:
    """
    Runner for a single Anima in a child process.

    Starts a DigitalAnima instance and exposes it via Unix socket IPC.
    Delegates scheduling, inbox rate limiting, streaming, and pending
    task execution to dedicated classes.
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
        self.shutdown_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._started_at = now_jst()

        # Delegate instances (created in run() after anima initialization)
        self._scheduler_mgr: SchedulerManager | None = None
        self._inbox_limiter: InboxRateLimiter | None = None
        self._pending_executor: PendingTaskExecutor | None = None
        self._streaming_handler: StreamingIPCHandler | None = None

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

            # Create delegate instances
            self._scheduler_mgr = SchedulerManager(
                anima=self.anima,
                anima_name=self.anima_name,
                anima_dir=self._anima_dir,
                emit_event=self._emit_event,
            )
            self._inbox_limiter = InboxRateLimiter(
                anima=self.anima,
                anima_name=self.anima_name,
                shutdown_event=self.shutdown_event,
                scheduler_mgr=self._scheduler_mgr,
            )
            self._pending_executor = PendingTaskExecutor(
                anima=self.anima,
                anima_name=self.anima_name,
                anima_dir=self._anima_dir,
                shutdown_event=self.shutdown_event,
            )
            self._streaming_handler = StreamingIPCHandler(
                anima=self.anima,
                anima_name=self.anima_name,
                anima_dir=self._anima_dir,
            )

            self.anima.set_on_lock_released(
                lambda: asyncio.ensure_future(self._inbox_limiter.on_anima_lock_released())
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

            # Clean up stale .tmp files left by interrupted atomic writes
            from core.memory._io import cleanup_tmp_files
            cleanup_tmp_files(self._anima_dir / "state")
            cleanup_tmp_files(self._anima_dir / "knowledge")

            self._ready_event.set()

            # Start autonomous scheduler (heartbeat + cron)
            self._scheduler_mgr.setup()

            # Start inbox watcher
            self.inbox_watcher_task = asyncio.create_task(
                self._inbox_limiter.inbox_watcher_loop()
            )

            # Start pending task watcher (picks up animaworks-tool submit)
            self.pending_task_watcher_task = asyncio.create_task(
                self._pending_executor.watcher_loop()
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
                    + "\n[応答が中断されました]"
                )
                conv_memory.append_turn("assistant", saved_text)
                conv_memory.save()
                StreamingJournal.confirm_recovery(self._anima_dir)
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
        else:
            # No text to save, but still need to clean up the journal
            StreamingJournal.confirm_recovery(self._anima_dir)

        # Record crash event in activity log
        try:
            from core.memory.activity import ActivityLogger
            activity = ActivityLogger(self._anima_dir)
            activity.log(
                "error",
                summary="応答が中断されました（前回セッションの未完了ストリームを回復）",
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

        # Record tool_use events in activity log
        if recovery.tool_calls:
            try:
                from core.memory.activity import ActivityLogger as _AL
                _activity = _AL(self._anima_dir)
                for tc in recovery.tool_calls:
                    _activity.log(
                        "tool_use",
                        summary=f"[recovered] {tc.get('tool', 'unknown')}",
                        tool=tc.get("tool", "unknown"),
                        meta={"recovered": True, **tc},
                    )
            except Exception:
                logger.debug(
                    "Failed to log recovered tool_use events: %s",
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
                and self._streaming_handler
            ):
                return self._streaming_handler.handle_stream(request)

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
        intent = params.get("intent") or ""
        images = params.get("images") or None
        attachment_paths = params.get("attachment_paths") or None

        result = await self.anima.process_message(
            message, from_person=from_person,
            intent=intent,
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

        scheduler = self._scheduler_mgr.scheduler if self._scheduler_mgr else None
        return {
            "status": self.anima._status,
            "current_task": self.anima._current_task or None,
            "needs_bootstrap": self.anima.needs_bootstrap,
            "scheduler_running": scheduler.running if scheduler else False,
            "scheduler_jobs": len(scheduler.get_jobs()) if scheduler else 0,
        }

    async def _handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ping request.

        Returns ``status: "initializing"`` while DigitalAnima is loading,
        ``status: "ok"`` once ready.  The parent process polls this to
        confirm readiness.
        """
        uptime = (now_jst() - ensure_aware(self._started_at)).total_seconds()
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

    # ── Cleanup ───────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Cancel deferred trigger timer
        if self._inbox_limiter:
            self._inbox_limiter.cancel_deferred_timer()

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
        if self._scheduler_mgr:
            self._scheduler_mgr.shutdown()

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
