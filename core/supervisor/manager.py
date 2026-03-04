"""
Process Supervisor - Manages lifecycle of Anima child processes.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as _signal
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from core.time_utils import ensure_aware, now_jst
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.exceptions import (  # noqa: F401
    ProcessError, AnimaNotFoundError, IPCConnectionError, ConfigError, MemoryIOError,
)
from core.supervisor.ipc import IPCResponse
from core.supervisor.process_handle import ProcessHandle, ProcessState
from core.supervisor._mgr_health import HealthMixin
from core.supervisor._mgr_reconcile import ReconcileMixin
from core.supervisor._mgr_scheduler import SchedulerMixin

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────

@dataclass
class RestartPolicy:
    """Process restart policy configuration."""
    max_retries: int = 5                   # Maximum restart attempts
    backoff_base_sec: float = 2.0          # Initial backoff delay
    backoff_max_sec: float = 60.0          # Maximum backoff delay
    reset_after_sec: float = 300.0         # Stable runtime to reset counter


@dataclass
class HealthConfig:
    """Health check configuration."""
    ping_interval_sec: float = 10.0        # Ping interval
    ping_timeout_sec: float = 5.0          # Ping timeout
    max_missed_pings: int = 3              # Consecutive misses before hang
    startup_grace_sec: float = 30.0        # Grace period after startup


@dataclass
class ReconciliationConfig:
    """Reconciliation loop configuration."""
    interval_sec: float = 30.0             # Scan interval


# ── Process Supervisor ─────────────────────────────────────────────

class ProcessSupervisor(HealthMixin, ReconcileMixin, SchedulerMixin):
    """
    Supervisor for managing Anima child processes.

    Responsibilities:
    - Start/stop child processes
    - Health monitoring (ping/pong)
    - Hang detection and recovery (SIGKILL + restart)
    - Auto-restart with exponential backoff
    - Schedule coordination (heartbeat/cron triggers)
    """

    def __init__(
        self,
        animas_dir: Path,
        shared_dir: Path,
        run_dir: Path,
        log_dir: Path | None = None,
        restart_policy: RestartPolicy | None = None,
        health_config: HealthConfig | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
        ws_manager: Any | None = None,
    ):
        self.animas_dir = animas_dir
        self.shared_dir = shared_dir
        self.run_dir = run_dir
        self.log_dir = log_dir
        self.ws_manager = ws_manager

        self.restart_policy = restart_policy or RestartPolicy()
        self.health_config = health_config or HealthConfig()
        self.reconciliation_config = reconciliation_config or ReconciliationConfig()

        self.processes: dict[str, ProcessHandle] = {}
        self._health_check_task: asyncio.Task | None = None
        self._reconciliation_task: asyncio.Task | None = None
        self._inbox_wake_task: asyncio.Task | None = None
        self._shutdown = False
        self.scheduler: AsyncIOScheduler | None = None
        self._scheduler_running: bool = False
        self._restart_counts: dict[str, int] = {}
        self._restarting: set[str] = set()
        self._starting: set[str] = set()
        self._permanently_failed: set[str] = set()
        self._failed_log_times: dict[str, float] = {}
        self._bootstrapping: set[str] = set()
        self._bootstrap_retry_counts: dict[str, int] = {}
        self._bootstrap_max_retries: int = 3

        # Maximum streaming duration before hang detection (seconds).
        # Defaults to 1800s (30 min) to accommodate long tool executions
        # while still catching truly stuck streams.
        self._max_streaming_duration_sec: int = 1800
        try:
            from core.config import load_config
            srv = load_config().server
            self._max_streaming_duration_sec = getattr(
                srv, "max_streaming_duration", 1800,
            )
        except Exception:
            logger.debug("Config load failed for max_streaming_duration", exc_info=True)

        # Callbacks for anima lifecycle events (set by server/app.py)
        self.on_anima_added: Callable[[str], None] | None = None
        self.on_anima_removed: Callable[[str], None] | None = None

    def is_scheduler_running(self) -> bool:
        """Return whether the system scheduler is running."""
        return self._scheduler_running

    # ── Process Lifecycle ─────────────────────────────────────────

    async def start_all(self, anima_names: list[str]) -> None:
        """Start all Anima processes in parallel."""
        logger.info("Starting %d Anima processes (parallel)", len(anima_names))

        # Create socket directory and clean up stale sockets from previous runs
        socket_dir = self.run_dir / "sockets"
        socket_dir.mkdir(parents=True, exist_ok=True)
        for stale_sock in socket_dir.glob("*.sock"):
            try:
                stale_sock.unlink()
                logger.debug("Removed stale socket: %s", stale_sock)
            except OSError as exc:
                logger.warning("Failed to remove stale socket %s: %s", stale_sock, exc)

        # Kill zombie runner processes from a previous server crash
        self._kill_zombie_runners(anima_names)

        # Start all processes in parallel
        if anima_names:
            results = await asyncio.gather(
                *(self.start_anima(name) for name in anima_names),
                return_exceptions=True,
            )
            for name, result in zip(anima_names, results):
                if isinstance(result, Exception):
                    logger.error("Failed to start anima %s: %s", name, result)

        # Start health check loop
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )

        # Start reconciliation loop
        self._reconciliation_task = asyncio.create_task(
            self._reconciliation_loop()
        )

        # Start inbox wake dispatcher
        self._inbox_wake_task = asyncio.create_task(
            self._inbox_wake_dispatcher()
        )

        # Start system scheduler (daily/weekly consolidation)
        self._start_system_scheduler()

        logger.info("All processes started")

    def _kill_zombie_runners(self, anima_names: list[str]) -> None:
        """Detect and kill zombie runner processes from a previous server crash.

        Reads pidfiles under ``run/animas/{name}.pid`` and sends SIGTERM
        to any processes that are still alive.  This ensures a clean slate
        before spawning new runner processes.
        """
        pid_dir = self.run_dir / "animas"
        if not pid_dir.exists():
            return

        for pid_file in pid_dir.glob("*.pid"):
            anima_name = pid_file.stem
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # check if alive
                logger.warning(
                    "Killing zombie runner: %s (pid=%d)", anima_name, pid,
                )
                try:
                    os.kill(pid, _signal.SIGTERM)
                except OSError:
                    pass
                pid_file.unlink(missing_ok=True)
            except (ValueError, ProcessLookupError):
                pid_file.unlink(missing_ok=True)
            except OSError:
                pass

        # Also clean up stale lock files
        for lock_file in pid_dir.glob("*.lock"):
            lock_file.unlink(missing_ok=True)

    async def start_anima(self, anima_name: str) -> None:
        """Start a single Anima process.

        After the process is ready, checks if bootstrap is needed and
        launches it as a background task automatically.
        """
        if anima_name in self.processes:
            logger.warning("Process already exists: %s", anima_name)
            return
        if anima_name in self._starting:
            logger.debug("Start already in progress: %s", anima_name)
            return

        self._starting.add(anima_name)
        try:
            socket_dir = self.run_dir / "sockets"
            socket_dir.mkdir(parents=True, exist_ok=True)
            socket_path = socket_dir / f"{anima_name}.sock"

            handle = ProcessHandle(
                anima_name=anima_name,
                socket_path=socket_path,
                animas_dir=self.animas_dir,
                shared_dir=self.shared_dir,
                log_dir=self.log_dir
            )

            try:
                await handle.start()
                self.processes[anima_name] = handle
                logger.info("Anima process started: %s (PID %s)", anima_name, handle.get_pid())

                # Check if bootstrap is needed and launch in background
                try:
                    status = await self.send_request(
                        anima_name, "get_status", {}, timeout=10.0,
                    )
                    if status.get("needs_bootstrap"):
                        logger.info(
                            "Bootstrap needed for %s, launching background task",
                            anima_name,
                        )
                        asyncio.create_task(self._run_bootstrap(anima_name))
                except Exception as e:
                    logger.warning(
                        "Could not check bootstrap status for %s: %s",
                        anima_name, e,
                    )

            except Exception as e:
                logger.error("Failed to start process %s: %s", anima_name, e)
                raise
        finally:
            self._starting.discard(anima_name)

    def is_bootstrapping(self, anima_name: str) -> bool:
        """Check if an anima is currently bootstrapping."""
        return anima_name in self._bootstrapping

    async def _run_bootstrap(self, anima_name: str) -> None:
        """Run bootstrap for an anima in the background.

        Sends a ``run_bootstrap`` IPC request with a long timeout (600s)
        and broadcasts progress via WebSocket.  Tracks retry counts and
        disables further attempts after ``_bootstrap_max_retries`` failures
        by renaming ``bootstrap.md`` to ``bootstrap.md.failed``.
        """
        # Check retry limit before starting
        retry_count = self._bootstrap_retry_counts.get(anima_name, 0)
        if retry_count >= self._bootstrap_max_retries:
            bootstrap_file = self.animas_dir / anima_name / "bootstrap.md"
            failed_file = bootstrap_file.with_suffix(".md.failed")
            if bootstrap_file.exists():
                bootstrap_file.rename(failed_file)
                logger.error(
                    "Bootstrap retry limit reached for %s (%d/%d). "
                    "Renamed bootstrap.md -> bootstrap.md.failed. "
                    "Manual intervention required.",
                    anima_name, retry_count, self._bootstrap_max_retries,
                )
            else:
                logger.error(
                    "Bootstrap retry limit reached for %s (%d/%d). "
                    "Manual intervention required.",
                    anima_name, retry_count, self._bootstrap_max_retries,
                )
            await self._broadcast_event(
                "anima.bootstrap",
                {"name": anima_name, "status": "max_retries_exceeded"},
            )
            return

        self._bootstrapping.add(anima_name)
        logger.info(
            "Bootstrap started for %s (attempt %d/%d)",
            anima_name, retry_count + 1, self._bootstrap_max_retries,
        )

        # Broadcast bootstrap started
        await self._broadcast_event(
            "anima.bootstrap",
            {"name": anima_name, "status": "started"},
        )

        success = False
        try:
            handle = self.processes.get(anima_name)
            if not handle:
                logger.error("Bootstrap failed: process not found for %s", anima_name)
                await self._broadcast_event(
                    "anima.bootstrap",
                    {"name": anima_name, "status": "failed"},
                )
                return

            response = await handle.send_request(
                "run_bootstrap", {}, timeout=600.0,
            )

            if response.error:
                logger.error(
                    "Bootstrap failed for %s: %s",
                    anima_name, response.error.get("message", "Unknown error"),
                )
                await self._broadcast_event(
                    "anima.bootstrap",
                    {"name": anima_name, "status": "failed"},
                )
                return

            result = response.result or {}
            logger.info(
                "Bootstrap completed for %s (duration_ms=%s)",
                anima_name, result.get("duration_ms", "?"),
            )
            await self._broadcast_event(
                "anima.bootstrap",
                {"name": anima_name, "status": "completed"},
            )
            success = True

        except asyncio.TimeoutError:
            logger.error("Bootstrap timed out for %s (600s)", anima_name)
            await self._broadcast_event(
                "anima.bootstrap",
                {"name": anima_name, "status": "failed"},
            )
        except Exception:
            logger.exception("Bootstrap error for %s", anima_name)
            await self._broadcast_event(
                "anima.bootstrap",
                {"name": anima_name, "status": "failed"},
            )
        finally:
            was_bootstrapping = anima_name in self._bootstrapping
            self._bootstrapping.discard(anima_name)
            if success:
                self._bootstrap_retry_counts.pop(anima_name, None)
            else:
                self._bootstrap_retry_counts[anima_name] = retry_count + 1
            if was_bootstrapping:
                handle = self.processes.get(anima_name)
                if not handle or handle.state != ProcessState.RUNNING:
                    logger.warning(
                        "Bootstrap for %s ended with process not running "
                        "(possible reconciliation interference)",
                        anima_name,
                    )

    async def _broadcast_event(
        self, event_type: str, data: dict[str, Any],
    ) -> None:
        """Broadcast a WebSocket event if ws_manager is available."""
        if self.ws_manager:
            await self.ws_manager.broadcast({"type": event_type, "data": data})

    async def stop_anima(self, anima_name: str) -> None:
        """Stop a single Anima process."""
        handle = self.processes.get(anima_name)
        if not handle:
            logger.warning("Process not found: %s", anima_name)
            return

        await handle.stop(timeout=10.0)
        del self.processes[anima_name]
        logger.info("Anima process stopped: %s", anima_name)

    async def restart_anima(
        self, anima_name: str, *, _reset_counters: bool = True,
    ) -> None:
        """Restart a Anima process.

        Args:
            _reset_counters: When True (default), resets failure tracking
                state.  Internal callers (e.g. ``_handle_process_failure``)
                pass False to preserve the retry counter.
        """
        logger.info("Restarting process: %s", anima_name)

        if _reset_counters:
            self._restart_counts.pop(anima_name, None)
            self._permanently_failed.discard(anima_name)
            self._failed_log_times.pop(anima_name, None)

        # Guard against reconciliation spawning a duplicate process
        # during the window between stop and start.
        self._restarting.add(anima_name)
        try:
            if anima_name in self.processes:
                await self.stop_anima(anima_name)
            await self.start_anima(anima_name)
        finally:
            self._restarting.discard(anima_name)

    async def shutdown_all(self) -> None:
        """Shutdown all processes gracefully."""
        logger.info("Shutting down all processes")
        self._shutdown = True

        # Stop system scheduler
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self._scheduler_running = False
            logger.info("System scheduler stopped")

        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop reconciliation loop
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass

        # Stop inbox wake dispatcher
        if self._inbox_wake_task:
            self._inbox_wake_task.cancel()
            try:
                await self._inbox_wake_task
            except asyncio.CancelledError:
                pass

        # Stop all processes
        tasks = [
            self.stop_anima(name)
            for name in list(self.processes.keys())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All processes shut down")

    # ── IPC ───────────────────────────────────────────────────────

    async def send_request(
        self,
        anima_name: str,
        method: str,
        params: dict[str, Any],
        timeout: float = 60.0,
    ) -> dict:
        """Send IPC request to a Anima process.

        Raises:
            AnimaNotFoundError: If anima not found
            IPCConnectionError: If response contains error
        """
        handle = self.processes.get(anima_name)
        if not handle:
            raise AnimaNotFoundError(f"Anima not found: {anima_name}")

        response = await handle.send_request(method, params, timeout)

        if response.error:
            raise IPCConnectionError(
                f"Request failed: {response.error.get('message', 'Unknown error')}"
            )

        return response.result or {}

    async def send_request_stream(
        self,
        anima_name: str,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[IPCResponse]:
        """Send IPC request to a Anima process and yield streaming responses.

        Raises:
            AnimaNotFoundError: If anima not found
            IPCConnectionError: If response contains error
        """
        handle = self.processes.get(anima_name)
        if not handle:
            raise AnimaNotFoundError(f"Anima not found: {anima_name}")

        async for response in handle.send_request_stream(
            method, params, timeout
        ):
            if response.error:
                raise IPCConnectionError(
                    f"Stream error: {response.error.get('message', 'Unknown error')}"
                )
            yield response

    async def _inbox_wake_dispatcher(self) -> None:
        """Watch ``run/inbox_wake/`` for wake files and trigger process_inbox.

        Files are named after the target anima (e.g. ``run/inbox_wake/sakura``).
        When detected, sends a ``process_inbox`` IPC request to the target and
        deletes the file.  Polls at 0.5s intervals.
        """
        wake_dir = self.run_dir / "inbox_wake"
        wake_dir.mkdir(parents=True, exist_ok=True)

        while not self._shutdown:
            try:
                await asyncio.sleep(0.5)
                if not wake_dir.exists():
                    continue
                for wake_file in wake_dir.iterdir():
                    if wake_file.name.startswith("."):
                        continue
                    target_name = wake_file.name
                    try:
                        wake_file.unlink()
                    except FileNotFoundError:
                        continue
                    except OSError:
                        logger.debug("Failed to remove wake file %s", wake_file, exc_info=True)
                        continue

                    if target_name not in self.processes:
                        logger.debug(
                            "Inbox wake for unknown anima: %s", target_name,
                        )
                        continue

                    try:
                        await self.send_request(
                            target_name, "process_inbox", {}, timeout=30.0,
                        )
                        logger.debug("Inbox wake dispatched: %s", target_name)
                    except Exception:
                        logger.debug(
                            "Failed to dispatch inbox wake for %s",
                            target_name, exc_info=True,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Inbox wake dispatcher error", exc_info=True)
                await asyncio.sleep(1.0)

    async def _poll_anima_events(self) -> None:
        """Read and broadcast event files from child processes."""
        events_base = self.run_dir / "events"
        if not events_base.exists():
            return

        for anima_dir in events_base.iterdir():
            if not anima_dir.is_dir():
                continue
            for event_file in sorted(anima_dir.glob("*.json")):
                try:
                    data = json.loads(event_file.read_text(encoding="utf-8"))
                    event_type = data.get("event", "")
                    event_data = data.get("data", {})
                    await self._broadcast_event(event_type, event_data)
                    event_file.unlink()
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to process event file %s: %s", event_file, e)
                    try:
                        event_file.unlink()
                    except OSError:
                        logger.debug("Failed to remove corrupted event file %s", event_file, exc_info=True)

    # ── Status ───────────────────────────────────────────────────

    def get_process_status(self, anima_name: str) -> dict:
        """Get status of a Anima process."""
        handle = self.processes.get(anima_name)
        if not handle:
            return {"status": "not_found"}

        uptime = (now_jst() - ensure_aware(handle.stats.started_at)).total_seconds()

        return {
            "status": "bootstrapping" if self.is_bootstrapping(anima_name) else handle.state.value,
            "pid": handle.get_pid(),
            "uptime_sec": uptime,
            "restart_count": self._restart_counts.get(anima_name, 0),
            "missed_pings": handle.stats.missed_pings,
            "bootstrapping": self.is_bootstrapping(anima_name),
            "last_ping_at": (
                handle.stats.last_ping_at.isoformat()
                if handle.stats.last_ping_at else None
            ),
        }

    def get_all_status(self) -> dict[str, dict]:
        """Get status of all processes."""
        return {
            name: self.get_process_status(name)
            for name in self.processes
        }
