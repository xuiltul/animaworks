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
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from core.time_utils import ensure_aware, now_jst
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.exceptions import (  # noqa: F401
    ProcessError, AnimaNotFoundError, IPCConnectionError, ConfigError, MemoryIOError,
)
from core.supervisor.ipc import IPCResponse
from core.supervisor.process_handle import ProcessHandle, ProcessState

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

class ProcessSupervisor:
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
        self._shutdown = False
        self.scheduler: AsyncIOScheduler | None = None
        self._scheduler_running: bool = False
        self._restart_counts: dict[str, int] = {}
        self._restarting: set[str] = set()
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

    async def start_all(self, anima_names: list[str]) -> None:
        """
        Start all Anima processes.

        Args:
            anima_names: List of anima names to start
        """
        logger.info("Starting %d Anima processes", len(anima_names))

        # Create socket directory and clean up stale sockets from previous runs
        socket_dir = self.run_dir / "sockets"
        socket_dir.mkdir(parents=True, exist_ok=True)
        for stale_sock in socket_dir.glob("*.sock"):
            try:
                stale_sock.unlink()
                logger.debug("Removed stale socket: %s", stale_sock)
            except OSError as exc:
                logger.warning("Failed to remove stale socket %s: %s", stale_sock, exc)

        # Start each process
        for anima_name in anima_names:
            await self.start_anima(anima_name)

        # Start health check loop
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )

        # Start reconciliation loop
        self._reconciliation_task = asyncio.create_task(
            self._reconciliation_loop()
        )

        # Start system scheduler (daily/weekly consolidation)
        self._start_system_scheduler()

        logger.info("All processes started")

    async def start_anima(self, anima_name: str) -> None:
        """Start a single Anima process.

        After the process is ready, checks if bootstrap is needed and
        launches it as a background task automatically.
        """
        if anima_name in self.processes:
            logger.warning("Process already exists: %s", anima_name)
            return

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
                # Reset retry counter on success
                self._bootstrap_retry_counts.pop(anima_name, None)
            else:
                # Increment retry counter on failure
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

    async def restart_anima(self, anima_name: str) -> None:
        """Restart a Anima process."""
        logger.info("Restarting process: %s", anima_name)

        # Stop existing process
        if anima_name in self.processes:
            await self.stop_anima(anima_name)

        # Start new process
        await self.start_anima(anima_name)

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

        # Stop all processes
        tasks = [
            self.stop_anima(name)
            for name in list(self.processes.keys())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All processes shut down")

    async def send_request(
        self,
        anima_name: str,
        method: str,
        params: dict[str, Any],
        timeout: float = 60.0
    ) -> dict:
        """
        Send IPC request to a Anima process.

        Args:
            anima_name: Target anima name
            method: Method name
            params: Request parameters
            timeout: Timeout in seconds

        Returns:
            Response result dict

        Raises:
            KeyError: If anima not found
            RuntimeError: If process not running
            ValueError: If response contains error
        """
        handle = self.processes.get(anima_name)
        if not handle:
            raise KeyError(f"Anima not found: {anima_name}")

        response = await handle.send_request(method, params, timeout)

        if response.error:
            raise ValueError(
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
        """
        Send IPC request to a Anima process and yield streaming responses.

        Args:
            anima_name: Target anima name
            method: Method name
            params: Request parameters (should include stream=True)
            timeout: Per-chunk timeout in seconds. Resets on each received
                chunk. If None, resolved from config (default 60s).

        Yields:
            IPCResponse objects (chunks and final result)

        Raises:
            KeyError: If anima not found
            RuntimeError: If process not running
        """
        handle = self.processes.get(anima_name)
        if not handle:
            raise KeyError(f"Anima not found: {anima_name}")

        async for response in handle.send_request_stream(
            method, params, timeout
        ):
            if response.error:
                raise ValueError(
                    f"Stream error: {response.error.get('message', 'Unknown error')}"
                )
            yield response

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
                    # Remove corrupted files
                    try:
                        event_file.unlink()
                    except OSError:
                        logger.debug("Failed to remove corrupted event file %s", event_file, exc_info=True)

    async def _health_check_loop(self) -> None:
        """
        Health check loop.

        Periodically pings all processes and handles failures.
        """
        logger.info("Health check loop started")

        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_config.ping_interval_sec)

                # Poll and broadcast child process events
                await self._poll_anima_events()

                # Check all processes in parallel
                checks = [
                    self._check_process_health(anima_name, handle)
                    for anima_name, handle in list(self.processes.items())
                ]
                await asyncio.gather(*checks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop: %s", e)

        logger.info("Health check loop stopped")

    async def _check_process_health(
        self,
        anima_name: str,
        handle: ProcessHandle
    ) -> None:
        """Check health of a single process."""
        # Detect handles stuck in STOPPING state (e.g. after failed shutdown)
        if handle.state == ProcessState.STOPPING:
            if not handle.stopping_since:
                return
            stopping_duration = (
                now_jst() - ensure_aware(handle.stopping_since)
            ).total_seconds()
            if stopping_duration > 30:
                logger.error(
                    "Process stuck in STOPPING state: %s (%.0fs)",
                    anima_name, stopping_duration,
                )
                handle.state = ProcessState.FAILED
                asyncio.create_task(
                    self._handle_process_failure(anima_name, handle)
                )
            return

        # RESTARTING 状態ならヘルスチェックをスキップ
        if handle.state == ProcessState.RESTARTING:
            return

        # During streaming: skip ping (IPC lock held) but still check
        # process liveness and streaming duration timeout.
        if handle.is_streaming:
            # Detect process death during streaming
            if handle.state == ProcessState.FAILED:
                logger.error(
                    "Process FAILED during streaming: %s", anima_name,
                )
                asyncio.create_task(
                    self._handle_process_failure(anima_name, handle)
                )
                return
            if not handle.is_alive():
                logger.error(
                    "Process died during streaming: %s (exit_code=%s)",
                    anima_name, handle.stats.exit_code,
                )
                asyncio.create_task(
                    self._handle_process_failure(anima_name, handle)
                )
                return
            # Streaming duration timeout
            started_at = handle.streaming_started_at
            if started_at is not None:
                streaming_sec = (now_jst() - ensure_aware(started_at)).total_seconds()
                if streaming_sec > self._max_streaming_duration_sec:
                    logger.error(
                        "Streaming timeout for %s (%.0fs > %ds)",
                        anima_name, streaming_sec,
                        self._max_streaming_duration_sec,
                    )
                    asyncio.create_task(
                        self._handle_process_hang(anima_name, handle)
                    )
            return

        # Skip if in startup grace period
        uptime = (now_jst() - ensure_aware(handle.stats.started_at)).total_seconds()
        if uptime < self.health_config.startup_grace_sec:
            logger.debug("Skipping health check for %s (startup grace)", anima_name)
            return

        # Reset restart counter after stable uptime
        if uptime > self.restart_policy.reset_after_sec:
            if self._restart_counts.get(anima_name, 0) > 0:
                self._restart_counts[anima_name] = 0
                logger.info(
                    "Restart counter reset for %s (stable for %.0fs)",
                    anima_name, uptime
                )

        # Direct state check: detect IPC connection loss
        if handle.state == ProcessState.FAILED:
            logger.error(
                "Process in FAILED state (IPC connection lost): %s",
                anima_name
            )
            asyncio.create_task(self._handle_process_failure(anima_name, handle))
            return

        # Check if process is alive
        if not handle.is_alive():
            # Read actual return code from the Popen object (poll() sets it)
            actual_rc = handle.process.returncode if handle.process else None
            handle.stats.exit_code = actual_rc
            logger.error(
                "Process exited unexpectedly: %s (exit_code=%s, signal=%s)",
                anima_name,
                actual_rc,
                -actual_rc if actual_rc is not None and actual_rc < 0 else "N/A",
            )
            asyncio.create_task(self._handle_process_failure(anima_name, handle))
            return

        # Ping process
        success = await handle.ping(timeout=self.health_config.ping_timeout_sec)

        if success:
            # Ping successful
            if handle.stats.missed_pings > 0:
                logger.info("Process recovered: %s", anima_name)
            return

        # Ping failed
        logger.warning(
            "Health check failed: %s (missed=%d/%d)",
            anima_name, handle.stats.missed_pings, self.health_config.max_missed_pings
        )

        # Check if hang threshold exceeded
        if handle.stats.missed_pings >= self.health_config.max_missed_pings:
            logger.error(
                "Process hang detected: %s (PID %s)",
                anima_name, handle.get_pid()
            )
            asyncio.create_task(self._handle_process_hang(anima_name, handle))

    async def _handle_process_failure(
        self,
        anima_name: str,
        handle: ProcessHandle
    ) -> None:
        """Handle process exit/crash.

        Runs as an independent task so the health-check loop is not blocked
        by backoff sleeps.  A per-anima guard prevents duplicate restarts.
        """
        if anima_name in self._restarting:
            return
        self._restarting.add(anima_name)

        # リスタート中であることをハンドルに反映
        handle.state = ProcessState.RESTARTING

        try:
            # Check restart count (supervisor-level, survives handle recreation)
            count = self._restart_counts.get(anima_name, 0)
            if count >= self.restart_policy.max_retries:
                logger.error(
                    "Max restart retries exceeded for %s. Manual intervention required.",
                    anima_name
                )
                handle.state = ProcessState.FAILED
                return

            # Calculate backoff delay
            backoff = min(
                self.restart_policy.backoff_base_sec * (2 ** count),
                self.restart_policy.backoff_max_sec
            )

            logger.info(
                "Scheduling restart for %s (retry %d/%d, delay=%.1fs)",
                anima_name, count + 1, self.restart_policy.max_retries, backoff
            )

            # Wait and restart
            await asyncio.sleep(backoff)

            self._restart_counts[anima_name] = count + 1
            await self.restart_anima(anima_name)

            new_handle = self.processes.get(anima_name)
            if new_handle:
                logger.info(
                    "Process restarted: %s (PID %s, retry=%d/%d)",
                    anima_name, new_handle.get_pid(),
                    count + 1, self.restart_policy.max_retries
                )
            else:
                logger.error(
                    "Restart completed but no handle found: %s", anima_name
                )

        except Exception as e:
            logger.error("Failed to restart %s: %s", anima_name, e)
            handle.state = ProcessState.FAILED
        finally:
            self._restarting.discard(anima_name)

    async def _handle_process_hang(
        self,
        anima_name: str,
        handle: ProcessHandle
    ) -> None:
        """Handle hung process (kill and restart)."""
        logger.warning("Killing hung process: %s", anima_name)

        # Kill process
        await handle.kill()

        # Restart
        await self._handle_process_failure(anima_name, handle)

    # ── Reconciliation ─────────────────────────────────────────────

    @staticmethod
    def read_anima_enabled(anima_dir: Path) -> bool:
        """Read the enabled flag from an anima's status.json.

        Returns True (enabled) when:
        - status.json does not exist (backward compatibility)
        - status.json exists with ``enabled: true``

        Returns False when status.json exists with ``enabled: false``.
        """
        status_file = anima_dir / "status.json"
        if not status_file.exists():
            return True  # Backward compatibility: no file = enabled
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            return bool(data.get("enabled", True))
        except (json.JSONDecodeError, OSError):
            return True  # Treat unreadable file as enabled

    async def _reconciliation_loop(self) -> None:
        """Periodically reconcile desired state (disk) with actual state (processes)."""
        logger.info("Reconciliation loop started (interval=%.0fs)",
                     self.reconciliation_config.interval_sec)

        while not self._shutdown:
            try:
                await asyncio.sleep(self.reconciliation_config.interval_sec)
                await self._reconcile()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Reconciliation failed")

        logger.info("Reconciliation loop stopped")

    async def _reconcile(self) -> None:
        """Scan animas_dir and sync desired state with actual process state."""
        if not self.animas_dir.exists():
            return

        running = set(self.processes.keys())

        # Build desired state from disk
        on_disk: dict[str, bool] = {}  # name -> enabled
        # Anima dirs that have identity.md but no status.json.
        # These are either legacy animas or factory-in-progress.
        # They must NOT be auto-started but must NOT be killed if running.
        on_disk_incomplete: set[str] = set()
        for anima_dir in sorted(self.animas_dir.iterdir()):
            if not anima_dir.is_dir():
                continue
            if not (anima_dir / "identity.md").exists():
                continue
            # status.json is created as the final step of anima_factory.
            # Its absence means creation may still be in progress.
            if not (anima_dir / "status.json").exists():
                on_disk_incomplete.add(anima_dir.name)
                continue
            on_disk[anima_dir.name] = self.read_anima_enabled(anima_dir)

        # enabled + not running → start
        for name, enabled in on_disk.items():
            if enabled and name not in running:
                if name in self._restarting:
                    logger.debug("Reconciliation: skipping %s (restart in progress)", name)
                    continue
                if name in self._bootstrapping:
                    logger.debug("Reconciliation: skipping %s (bootstrap in progress)", name)
                    continue
                logger.info("Reconciliation: starting anima %s", name)
                try:
                    await self.start_anima(name)
                    if self.on_anima_added:
                        self.on_anima_added(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to start %s", name
                    )

        # disabled + running → stop
        for name, enabled in on_disk.items():
            if not enabled and name in running:
                if name in self._bootstrapping:
                    logger.info("Reconciliation: deferring stop for %s (bootstrap in progress)", name)
                    continue
                logger.info(
                    "Reconciliation: stopping anima %s (disabled)", name
                )
                try:
                    await self.stop_anima(name)
                    if self.on_anima_removed:
                        self.on_anima_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s", name
                    )

        # removed from disk + running → stop
        # Protect running animas whose directory exists (identity.md present)
        # even if status.json is missing (legacy or factory-in-progress).
        for name in list(running):
            if name not in on_disk and name not in on_disk_incomplete:
                if name in self._bootstrapping:
                    logger.info("Reconciliation: deferring stop for %s (bootstrap in progress)", name)
                    continue
                logger.info(
                    "Reconciliation: stopping anima %s (removed from disk)",
                    name,
                )
                try:
                    await self.stop_anima(name)
                    if self.on_anima_removed:
                        self.on_anima_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s", name
                    )

        # Check for missing anima assets (fallback generation)
        await self._reconcile_assets()

    async def _reconcile_assets(self) -> None:
        """Check for and generate missing anima assets during reconciliation."""
        try:
            from core.asset_reconciler import find_animas_with_missing_assets, reconcile_anima_assets

            incomplete = find_animas_with_missing_assets(self.animas_dir)
            if not incomplete:
                return

            logger.info(
                "Asset reconciliation: %d anima(s) with missing assets",
                len(incomplete),
            )
            for anima_name, _check in incomplete:
                anima_dir = self.animas_dir / anima_name
                result = await reconcile_anima_assets(anima_dir)
                if not result.get("skipped"):
                    await self._broadcast_event(
                        "anima.assets_updated",
                        {"name": anima_name, "source": "reconciliation"},
                    )
        except Exception:
            logger.exception("Asset reconciliation failed")

    # ── System Scheduler ────────────────────────────────────────

    def _start_system_scheduler(self) -> None:
        """Start the system-level scheduler for consolidation crons."""
        try:
            self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
            self._setup_system_crons()
            self.scheduler.start()
            self._scheduler_running = True
            logger.info("System scheduler started")
        except Exception:
            logger.exception("Failed to start system scheduler")
            self.scheduler = None
            self._scheduler_running = False

    def _setup_system_crons(self) -> None:
        """Register system-wide cron jobs for memory consolidation."""
        if not self.scheduler:
            return

        # Load consolidation config
        try:
            from core.config import load_config
            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            logger.debug("Config load failed for consolidation schedule", exc_info=True)
            consolidation_cfg = None

        # Daily consolidation
        daily_enabled = True
        daily_time = "02:00"
        if consolidation_cfg:
            daily_enabled = getattr(consolidation_cfg, "daily_enabled", True)
            daily_time = getattr(consolidation_cfg, "daily_time", "02:00")

        if daily_enabled:
            hour, minute = (int(x) for x in daily_time.split(":"))
            self.scheduler.add_job(
                self._run_daily_consolidation,
                CronTrigger(hour=hour, minute=minute),
                id="system_daily_consolidation",
                name="System: Daily Consolidation",
                replace_existing=True,
            )
            logger.info("System cron: Daily consolidation at %s JST", daily_time)

        # Weekly integration
        weekly_enabled = True
        weekly_time = "sun:03:00"
        if consolidation_cfg:
            weekly_enabled = getattr(consolidation_cfg, "weekly_enabled", True)
            weekly_time = getattr(consolidation_cfg, "weekly_time", "sun:03:00")

        if weekly_enabled:
            parts = weekly_time.split(":")
            day_of_week = parts[0] if len(parts) == 3 else "sun"
            time_parts = parts[-2:]
            hour, minute = int(time_parts[0]), int(time_parts[1])
            self.scheduler.add_job(
                self._run_weekly_integration,
                CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
                id="system_weekly_integration",
                name="System: Weekly Integration",
                replace_existing=True,
            )
            logger.info("System cron: Weekly integration on %s at %s:%s JST", day_of_week, time_parts[0], time_parts[1])

        # Monthly forgetting
        monthly_enabled = True
        monthly_time = "1:04:00"
        if consolidation_cfg:
            monthly_enabled = getattr(consolidation_cfg, "monthly_enabled", True)
            monthly_time = getattr(consolidation_cfg, "monthly_time", "1:04:00")

        if monthly_enabled:
            parts = monthly_time.split(":")
            day_of_month = int(parts[0]) if len(parts) == 3 else 1
            time_parts = parts[-2:]
            hour, minute = int(time_parts[0]), int(time_parts[1])
            self.scheduler.add_job(
                self._run_monthly_forgetting,
                CronTrigger(day=day_of_month, hour=hour, minute=minute),
                id="system_monthly_forgetting",
                name="System: Monthly Forgetting",
                replace_existing=True,
            )
            logger.info(
                "System cron: Monthly forgetting on day %d at %02d:%02d JST",
                day_of_month, hour, minute,
            )

        # Activity log rotation
        try:
            from core.config.models import ActivityLogConfig

            activity_cfg: ActivityLogConfig | None = None
            try:
                from core.config import load_config as _load_cfg
                _al = getattr(_load_cfg(), "activity_log", None)
                if isinstance(_al, ActivityLogConfig):
                    activity_cfg = _al
            except Exception:
                logger.debug("Config load failed for activity_log rotation schedule", exc_info=True)

            if activity_cfg is None:
                activity_cfg = ActivityLogConfig()

            if activity_cfg.rotation_enabled:
                r_hour, r_minute = (int(x) for x in activity_cfg.rotation_time.split(":"))
                self.scheduler.add_job(
                    self._run_activity_log_rotation,
                    CronTrigger(hour=r_hour, minute=r_minute),
                    id="system_activity_log_rotation",
                    name="System: Activity Log Rotation",
                    replace_existing=True,
                )
                logger.info("System cron: Activity log rotation at %s JST", activity_cfg.rotation_time)
        except Exception:
            logger.debug("Activity log rotation schedule setup failed", exc_info=True)

    def _iter_consolidation_targets(self) -> list[tuple[str, Path]]:
        """Return (anima_name, anima_dir) for all initialized and enabled animas.

        Scans ``self.animas_dir`` on disk so that stopped / crashed animas are
        still included.  Matches the guard pattern used by ``_reconcile()``.
        """
        if not self.animas_dir.exists():
            return []

        targets: list[tuple[str, Path]] = []
        for anima_dir in sorted(self.animas_dir.iterdir()):
            if not anima_dir.is_dir():
                continue
            if not (anima_dir / "identity.md").exists():
                continue
            if not (anima_dir / "status.json").exists():
                continue
            if not self.read_anima_enabled(anima_dir):
                continue
            targets.append((anima_dir.name, anima_dir))
        return targets

    async def _run_daily_consolidation(self) -> None:
        """Run daily consolidation for all animas via IPC.

        Sends ``run_consolidation`` IPC requests to running Anima processes,
        then performs metadata-based post-processing (synaptic downscaling,
        RAG index rebuild) from the supervisor process.
        """
        logger.info("Starting system-wide daily consolidation")

        try:
            from core.config import load_config
            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            logger.debug("Config load failed for daily consolidation", exc_info=True)
            consolidation_cfg = None

        from core.config.models import ConsolidationConfig
        max_turns = ConsolidationConfig().max_turns
        if consolidation_cfg:
            max_turns = getattr(consolidation_cfg, "max_turns", max_turns)

        for anima_name, anima_dir in self._iter_consolidation_targets():
            handle = self.processes.get(anima_name)
            if not handle or handle.state != ProcessState.RUNNING:
                logger.info(
                    "Daily consolidation skipped for %s: process not running",
                    anima_name,
                )
                continue

            try:
                # Anima-driven consolidation via IPC (tool-call loop)
                response = await handle.send_request(
                    "run_consolidation",
                    {"consolidation_type": "daily", "max_turns": max_turns},
                    timeout=600.0,
                )

                if response.error:
                    logger.error(
                        "Daily consolidation IPC error for %s: %s",
                        anima_name, response.error,
                    )
                    continue

                result = response.result or {}
                logger.info(
                    "Daily consolidation for %s: duration_ms=%d",
                    anima_name,
                    result.get("duration_ms", 0),
                )

                # Post-processing: Synaptic downscaling (metadata-based, no LLM)
                try:
                    from core.memory.forgetting import ForgettingEngine
                    forgetter = ForgettingEngine(anima_dir, anima_name)
                    downscaling_result = forgetter.synaptic_downscaling()
                    logger.info(
                        "Synaptic downscaling for %s: %s",
                        anima_name, downscaling_result,
                    )
                except Exception:
                    logger.exception(
                        "Synaptic downscaling failed for anima=%s", anima_name,
                    )

                # Post-processing: Rebuild RAG index
                try:
                    from core.memory.consolidation import ConsolidationEngine
                    engine = ConsolidationEngine(anima_dir, anima_name)
                    engine._rebuild_rag_index()
                except Exception:
                    logger.exception(
                        "RAG index rebuild failed for anima=%s", anima_name,
                    )

                await self._broadcast_event(
                    "system.consolidation",
                    {
                        "anima": anima_name,
                        "type": "daily",
                        "summary": result.get("summary", ""),
                        "duration_ms": result.get("duration_ms", 0),
                    },
                )
            except Exception:
                logger.exception("Daily consolidation failed for %s", anima_name)

    async def _run_weekly_integration(self) -> None:
        """Run weekly integration for all animas via IPC.

        Sends ``run_consolidation`` IPC requests to running Anima processes,
        then performs metadata-based post-processing (neurogenesis reorganization,
        RAG index rebuild) from the supervisor process.
        """
        logger.info("Starting system-wide weekly integration")

        try:
            from core.config import load_config
            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            logger.debug("Config load failed for weekly integration", exc_info=True)
            consolidation_cfg = None

        from core.config.models import ConsolidationConfig as _CC
        max_turns = _CC().max_turns
        if consolidation_cfg:
            max_turns = getattr(consolidation_cfg, "max_turns", max_turns)

        for anima_name, anima_dir in self._iter_consolidation_targets():
            handle = self.processes.get(anima_name)
            if not handle or handle.state != ProcessState.RUNNING:
                logger.info(
                    "Weekly integration skipped for %s: process not running",
                    anima_name,
                )
                continue

            try:
                # Anima-driven consolidation via IPC (tool-call loop)
                response = await handle.send_request(
                    "run_consolidation",
                    {"consolidation_type": "weekly", "max_turns": max_turns},
                    timeout=600.0,
                )

                if response.error:
                    logger.error(
                        "Weekly integration IPC error for %s: %s",
                        anima_name, response.error,
                    )
                    continue

                result = response.result or {}
                logger.info(
                    "Weekly integration for %s: duration_ms=%d",
                    anima_name,
                    result.get("duration_ms", 0),
                )

                # Post-processing: Neurogenesis reorganization (metadata-based)
                try:
                    from core.memory.forgetting import ForgettingEngine
                    forgetter = ForgettingEngine(anima_dir, anima_name)
                    reorg_result = forgetter.neurogenesis_reorganize()
                    logger.info(
                        "Neurogenesis reorganization for %s: %s",
                        anima_name, reorg_result,
                    )
                except Exception:
                    logger.exception(
                        "Neurogenesis reorganization failed for anima=%s",
                        anima_name,
                    )

                # Post-processing: Rebuild RAG index
                try:
                    from core.memory.consolidation import ConsolidationEngine
                    engine = ConsolidationEngine(anima_dir, anima_name)
                    engine._rebuild_rag_index()
                except Exception:
                    logger.exception(
                        "RAG index rebuild failed for anima=%s", anima_name,
                    )

                await self._broadcast_event(
                    "system.consolidation",
                    {
                        "anima": anima_name,
                        "type": "weekly",
                        "summary": result.get("summary", ""),
                        "duration_ms": result.get("duration_ms", 0),
                    },
                )
            except Exception:
                logger.exception("Weekly integration failed for %s", anima_name)

    async def _run_monthly_forgetting(self) -> None:
        """Run monthly forgetting for all animas."""
        logger.info("Starting system-wide monthly forgetting")

        for anima_name, anima_dir in self._iter_consolidation_targets():
            try:
                from core.memory.consolidation import ConsolidationEngine

                engine = ConsolidationEngine(
                    anima_dir=anima_dir,
                    anima_name=anima_name,
                )

                result = await engine.monthly_forget()

                logger.info(
                    "Monthly forgetting for %s: forgotten=%d, archived=%d files",
                    anima_name,
                    result.get("forgotten_chunks", 0),
                    len(result.get("archived_files", [])),
                )

                if not result.get("skipped"):
                    await self._broadcast_event(
                        "system.consolidation",
                        {"anima": anima_name, "type": "monthly_forgetting", "result": result},
                    )
            except Exception:
                logger.exception("Monthly forgetting failed for %s", anima_name)

    async def _run_activity_log_rotation(self) -> None:
        """Run activity log rotation for all animas."""
        logger.info("Starting system-wide activity log rotation")

        try:
            from core.config import load_config
            activity_cfg = getattr(load_config(), "activity_log", None)
        except Exception:
            logger.debug("Config load failed for activity log rotation", exc_info=True)
            activity_cfg = None

        from core.config.models import ActivityLogConfig
        defaults = ActivityLogConfig()
        mode = getattr(activity_cfg, "rotation_mode", defaults.rotation_mode) if activity_cfg else defaults.rotation_mode
        max_size_mb = getattr(activity_cfg, "max_size_mb", defaults.max_size_mb) if activity_cfg else defaults.max_size_mb
        max_age_days = getattr(activity_cfg, "max_age_days", defaults.max_age_days) if activity_cfg else defaults.max_age_days

        try:
            from core.memory.activity import ActivityLogger

            results = ActivityLogger.rotate_all(
                self.animas_dir,
                mode=mode,
                max_size_mb=max_size_mb,
                max_age_days=max_age_days,
            )
            if results:
                total_freed = sum(r.get("freed_bytes", 0) for r in results.values())
                total_deleted = sum(r.get("deleted_files", 0) for r in results.values())
                logger.info(
                    "Activity log rotation complete: %d animas, %d files deleted, %d bytes freed",
                    len(results), total_deleted, total_freed,
                )
            else:
                logger.info("Activity log rotation: no files needed rotation")
        except Exception:
            logger.exception("Activity log rotation failed")

    # ── Status ───────────────────────────────────────────────────

    def get_process_status(self, anima_name: str) -> dict:
        """
        Get status of a Anima process.

        Returns:
            Status dict with state, PID, uptime, etc.
        """
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
