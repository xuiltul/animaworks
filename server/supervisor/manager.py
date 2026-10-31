"""
Process Supervisor - Manages lifecycle of Anima child processes.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.anima.workspace_files import BOOTSTRAP_FILE
from core.exceptions import (  # noqa: F401
    AnimaNotFoundError,
    ConfigError,
    ConfigNotFoundError,
    IPCConnectionError,
    ProcessError,
)
from core.platform.atomic_io import atomic_write_json
from core.platform.process import kill_tree, snapshot_descendants
from core.platform.subprocess_entries import SubprocessEntry
from core.platform.tasks import spawn
from core.runtime.ipc import IPCResponse
from core.time_utils import ensure_aware, now_local
from server.supervisor._mgr_health import HealthMixin
from server.supervisor._mgr_rag_repair import RAGRepairMixin
from server.supervisor._mgr_reconcile import ReconcileMixin
from server.supervisor._mgr_scheduler import SchedulerMixin
from server.supervisor.process_handle import INTERNAL_AUTH_ENV, ProcessHandle, ProcessState
from server.supervisor.restart_state import RestartController

logger = logging.getLogger(__name__)

_RUNNER_CMDLINE_MARKERS = (
    SubprocessEntry.SUPERVISOR_RUNNER.value,
    "core.supervisor.runner",  # legacy name until 2026-11 (S3a)
)


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class RestartPolicy:
    """Process restart policy configuration."""

    max_retries: int = 3  # Consecutive failures before FAILED display
    backoff_base_sec: float = 30.0  # Initial backoff delay
    backoff_max_sec: float = 1800.0  # Maximum backoff delay (30 min)
    reset_after_sec: float = 300.0  # Stable runtime to reset counter


@dataclass
class HealthConfig:
    """Health check configuration."""

    ping_interval_sec: float = 10.0  # Ping interval
    ping_timeout_sec: float = 5.0  # Ping timeout
    max_missed_pings: int = 6  # Consecutive misses before hang
    startup_grace_sec: float = 30.0  # Grace period after startup
    health_check_warmup_seconds: float = 300.0  # Global warmup after startup reaches ready
    runner_warmup_seconds: float = 180.0  # Per-runner warmup after spawn/restart


@dataclass
class ReconciliationConfig:
    """Reconciliation loop configuration."""

    interval_sec: float = 30.0  # Scan interval


# ── Process Supervisor ─────────────────────────────────────────────


class ProcessSupervisor(HealthMixin, RAGRepairMixin, ReconcileMixin, SchedulerMixin):
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
        self._lifecycle_locks: dict[str, asyncio.Lock] = {}
        self._health_check_task: asyncio.Task | None = None
        self._reconciliation_task: asyncio.Task | None = None
        self._event_file_drain_task: asyncio.Task | None = None
        self._event_tasks: dict[str, asyncio.Task] = {}
        self._zombie_reaper_task: asyncio.Task | None = None
        self._shutdown = False
        self.scheduler: AsyncIOScheduler | None = None
        self._scheduler_running: bool = False
        self._restarting: set[str] = set()
        self._restart_worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._starting: set[str] = set()
        self._bootstrapping: set[str] = set()
        self._consolidating: set[str] = set()  # animas currently running daily/weekly consolidation
        self._rag_repairs_in_progress: set[str] = set()
        self._recently_stopped: dict[str, float] = {}  # anima_name → monotonic timestamp
        self._starting_since: dict[str, float] = {}
        self._restart_ctl: RestartController | None = None
        self._bootstrap_retry_counts: dict[str, int] = {}
        self._bootstrap_max_retries: int = 3
        self._bootstrap_retry_delay_s: float = 60.0
        self._bootstrap_retries_file = self.animas_dir / ".bootstrap_retries.json"
        self._load_bootstrap_retries()

        # Bounded drain time for administrative stop/recovery operations; this
        # is not a streaming-duration health kill.
        self._restart_drain_timeout_sec: float = 600.0
        self._stream_drain_timeout_sec: int = 1800
        self._anima_startup_ready_timeout: float = 120.0
        self._anima_stop_timeout: float = 60.0
        self._spawn_timeout_sec: float = 300.0
        try:
            from core.config import load_config

            srv = load_config().server
            self._anima_startup_ready_timeout = float(
                getattr(srv, "anima_startup_ready_timeout", 120),
            )
            self._anima_stop_timeout = float(getattr(srv, "anima_stop_timeout", 60.0))
            self._spawn_timeout_sec = float(getattr(srv, "spawn_timeout", 300))
            self._restart_drain_timeout_sec = max(
                0.0,
                float(getattr(srv, "restart_drain_timeout_seconds", 600)),
            )
            if restart_policy is None:
                retry_count = int(getattr(srv, "supervisor_respawn_max_retries", 3))
                retry_interval = float(getattr(srv, "supervisor_respawn_retry_interval_seconds", 30.0))
                backoff_max = float(getattr(srv, "supervisor_respawn_backoff_max_seconds", 1800.0))
                self.restart_policy.max_retries = max(1, retry_count)
                self.restart_policy.backoff_base_sec = max(0.0, retry_interval)
                self.restart_policy.backoff_max_sec = max(0.0, backoff_max)
            if health_config is None:
                self.health_config.health_check_warmup_seconds = float(getattr(srv, "health_check_warmup_seconds", 300))
                self.health_config.runner_warmup_seconds = float(getattr(srv, "runner_warmup_seconds", 180))
        except (ConfigError, ConfigNotFoundError):
            logger.debug("Config load failed for server process timeouts", exc_info=True)

        self._restart_ctl = RestartController(
            failed_threshold=self.restart_policy.max_retries,
            base_delay_sec=self.restart_policy.backoff_base_sec,
            max_delay_sec=self.restart_policy.backoff_max_sec,
            stable_reset_sec=self.restart_policy.reset_after_sec,
        )

        # Callbacks for anima lifecycle events (set by server/app.py)
        self.on_anima_added: Callable[[str], None] | None = None
        self.on_anima_removed: Callable[[str], None] | None = None

        # Env vars passed to child processes (embed/vector URLs). Set by server/app.py.
        self.child_env_urls: dict[str, str] = {}

        # Internal API auth source, set by server/app.py before animas start.
        # Used to mint a per-anima token for each spawned child.
        self.internal_auth = None

    def is_scheduler_running(self) -> bool:
        """Return whether the system scheduler is running."""
        return self._scheduler_running

    def _load_bootstrap_retries(self) -> None:
        """Load persisted bootstrap retry counts from disk."""
        try:
            if self._bootstrap_retries_file.exists():
                data = json.loads(self._bootstrap_retries_file.read_text())
                if isinstance(data, dict):
                    self._bootstrap_retry_counts = {k: int(v) for k, v in data.items()}
                    logger.info(
                        "Loaded bootstrap retry counts: %s",
                        self._bootstrap_retry_counts,
                    )
        except Exception:
            logger.warning("Failed to load bootstrap retries file", exc_info=True)

    def _save_bootstrap_retries(self) -> None:
        """Persist bootstrap retry counts to disk."""
        try:
            atomic_write_json(
                self._bootstrap_retries_file,
                self._bootstrap_retry_counts,
                indent=2,
                ensure_ascii=True,
                trailing_newline=False,
            )
        except Exception:
            logger.warning("Failed to save bootstrap retries file", exc_info=True)

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
            for name, result in zip(anima_names, results, strict=False):
                if isinstance(result, Exception):
                    logger.error("Failed to start anima %s: %s", name, result)

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Start reconciliation loop
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())

        # Drain fallback event files independently from the health checks.
        self._event_file_drain_task = asyncio.create_task(self._event_file_drain_loop())

        # Start zombie reaper (safety net for orphaned child processes)
        self._zombie_reaper_task = asyncio.create_task(self._zombie_reaper_loop())

        # Start system scheduler (daily/weekly consolidation)
        self._start_system_scheduler()

        logger.info("All processes started")

    def _kill_zombie_runners(self, anima_names: list[str]) -> None:
        """Detect and kill zombie runner processes from a previous server crash.

        Reads pidfiles under ``run/animas/{name}.pid`` and sends SIGTERM
        to any processes that are still alive.  On Windows, waits for each
        process to actually terminate before continuing (otherwise the old
        process may still hold file locks when new processes try to start).
        """
        import psutil as _psutil

        pid_dir = self.run_dir / "animas"
        if not pid_dir.exists():
            return

        for pid_file in pid_dir.glob("*.pid"):
            anima_name = pid_file.stem
            try:
                pid = int(pid_file.read_text().strip())
                # Use psutil to check process liveness (os.kill(pid, 0)
                # does NOT work reliably on Windows — raises OSError).
                try:
                    proc = _psutil.Process(pid)
                    if not proc.is_running():
                        raise _psutil.NoSuchProcess(pid)
                except _psutil.NoSuchProcess:
                    logger.debug("Stale pidfile for %s (pid=%d, already dead)", anima_name, pid)
                    pid_file.unlink(missing_ok=True)
                    continue

                cmdline = proc.cmdline()
                if not any(marker in token for marker in _RUNNER_CMDLINE_MARKERS for token in cmdline):
                    logger.warning(
                        "PID file for %s does not match a known Anima runner (pid=%d); skipping",
                        anima_name,
                        pid,
                    )
                    pid_file.unlink(missing_ok=True)
                    continue

                logger.warning(
                    "Killing zombie runner: %s (pid=%d)",
                    anima_name,
                    pid,
                )
                try:
                    # Snapshot before killing the runner; its children may be reparented after exit.
                    descendants = snapshot_descendants(pid)
                    proc.kill()
                    proc.wait(timeout=5)
                    kill_tree(pid, descendants=descendants, include_root=False)
                    logger.info(
                        "Zombie runner killed and confirmed dead: %s (pid=%d)",
                        anima_name,
                        pid,
                    )
                except _psutil.NoSuchProcess:
                    pass
                except _psutil.TimeoutExpired:
                    logger.error(
                        "Zombie runner did not die within 5s: %s (pid=%d)",
                        anima_name,
                        pid,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to kill zombie runner %s (pid=%d): %s",
                        anima_name,
                        pid,
                        exc,
                    )
                pid_file.unlink(missing_ok=True)
            except ValueError:
                pid_file.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Unexpected error checking zombie %s: %s", anima_name, exc)
                pid_file.unlink(missing_ok=True)

        # Clean up stale lock files (safe now that processes are dead)
        for lock_file in pid_dir.glob("*.lock"):
            try:
                lock_file.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove lock file %s (may still be held)", lock_file)

    async def start_anima(self, anima_name: str) -> None:
        """Start a single Anima process.

        After the process is ready, checks if bootstrap is needed and
        launches it as a background task automatically.

        Serialized with ``stop_anima`` via per-anima lifecycle lock so that
        disable→stop and start cannot race on the same anima.
        """
        lock = self._lifecycle_locks.setdefault(anima_name, asyncio.Lock())
        async with lock:
            # Re-check under lock: shutdown / disable may have raced while we waited.
            if self._shutdown:
                logger.info("skip start during shutdown: %s", anima_name)
                return
            if not self.read_anima_enabled(self.animas_dir / anima_name):
                logger.warning("Refusing to start disabled anima: %s", anima_name)
                return
            if anima_name in self.processes:
                logger.warning("Process already exists: %s", anima_name)
                return
            if anima_name in self._starting:
                logger.debug("Start already in progress: %s", anima_name)
                return

            self._starting.add(anima_name)
            self._starting_since[anima_name] = time.monotonic()
            try:
                socket_dir = self.run_dir / "sockets"
                socket_dir.mkdir(parents=True, exist_ok=True)
                socket_path = socket_dir / f"{anima_name}.sock"

                handle = ProcessHandle(
                    anima_name=anima_name,
                    socket_path=socket_path,
                    animas_dir=self.animas_dir,
                    shared_dir=self.shared_dir,
                    log_dir=self.log_dir,
                    child_env_urls=self.child_env_urls,
                    internal_auth_env=(
                        {INTERNAL_AUTH_ENV: self.internal_auth.token_for_anima(anima_name)}
                        if self.internal_auth is not None
                        else {}
                    ),
                    startup_ready_timeout=self._anima_startup_ready_timeout,
                )

                try:
                    await handle.start()
                    # Re-check after the await: shutdown_all may have taken its
                    # stop snapshot while we were starting. No await between
                    # this check and registration, so the window is closed.
                    if self._shutdown:
                        logger.info("Shutdown during start; stopping %s", anima_name)
                        await handle.stop(
                            timeout=10.0,
                            drain_streams=False,
                            drain_background=False,
                        )
                        return
                    self.processes[anima_name] = handle
                    await self._cancel_event_consumer(anima_name)
                    self._event_tasks[anima_name] = asyncio.create_task(
                        self._consume_events(anima_name, handle),
                        name=f"event-stream-{anima_name}",
                    )
                    if self._restart_ctl is not None:
                        self._restart_ctl.record_started(anima_name)
                    logger.info("Anima process started: %s (PID %s)", anima_name, handle.get_pid())

                    # Check if bootstrap is needed and launch in background
                    try:
                        status = await self.send_request(
                            anima_name,
                            "get_status",
                            {},
                            timeout=10.0,
                        )
                        needs_background = status.get("needs_background_bootstrap")
                        if needs_background is None:
                            needs_background = bool(status.get("needs_bootstrap"))
                        bootstrap_state = status.get("bootstrap_state") or {}
                        if (
                            not needs_background
                            and bootstrap_state.get("state") == "running"
                            and bootstrap_state.get("mode") == "character_sheet"
                            and status.get("status") != "bootstrapping"
                        ):
                            needs_background = True
                        # A failed background setup resumes on the next start;
                        # _run_bootstrap enforces the retry limit.
                        if (
                            not needs_background
                            and bootstrap_state.get("state") == "failed"
                            and bootstrap_state.get("mode") == "character_sheet"
                            and (self.animas_dir / anima_name / BOOTSTRAP_FILE).exists()
                        ):
                            needs_background = True
                        if needs_background:
                            logger.info(
                                "Bootstrap needed for %s, launching background task",
                                anima_name,
                            )
                            spawn(
                                self._run_bootstrap(anima_name),
                                name=f"bootstrap-{anima_name}",
                            )
                    except Exception as e:
                        logger.warning(
                            "Could not check bootstrap status for %s: %s",
                            anima_name,
                            e,
                        )

                except (ProcessError, AnimaNotFoundError):
                    raise
                except Exception as e:
                    raise ProcessError(f"Failed to start process {anima_name}: {e}") from e
            finally:
                self._starting.discard(anima_name)
                self._starting_since.pop(anima_name, None)

    async def _run_bootstrap(self, anima_name: str) -> None:
        """Run bootstrap for an anima in the background.

        Sends a ``run_bootstrap`` IPC request with a long timeout (600s)
        and broadcasts progress via WebSocket.  Tracks retry counts and
        disables further attempts after ``_bootstrap_max_retries`` failures
        by renaming ``BOOTSTRAP.md`` to ``BOOTSTRAP.md.failed``.
        """
        # Check retry limit before starting
        retry_count = self._bootstrap_retry_counts.get(anima_name, 0)
        if retry_count >= self._bootstrap_max_retries:
            bootstrap_file = self.animas_dir / anima_name / BOOTSTRAP_FILE
            failed_file = bootstrap_file.with_suffix(".md.failed")
            if bootstrap_file.exists():
                bootstrap_file.rename(failed_file)
                try:
                    from core.anima.bootstrap_state import mark_bootstrap_failed

                    mark_bootstrap_failed(
                        self.animas_dir / anima_name,
                        "max_retries_exceeded",
                        reason="max_retries_exceeded",
                    )
                except Exception:
                    logger.debug("Failed to mark bootstrap max retries for %s", anima_name, exc_info=True)
                logger.error(
                    "Bootstrap retry limit reached for %s (%d/%d). "
                    "Renamed BOOTSTRAP.md -> BOOTSTRAP.md.failed. "
                    "Manual intervention required.",
                    anima_name,
                    retry_count,
                    self._bootstrap_max_retries,
                )
            else:
                logger.error(
                    "Bootstrap retry limit reached for %s (%d/%d). Manual intervention required.",
                    anima_name,
                    retry_count,
                    self._bootstrap_max_retries,
                )
            await self._broadcast_event(
                "anima.bootstrap",
                {"name": anima_name, "status": "max_retries_exceeded"},
            )
            return

        self._bootstrapping.add(anima_name)
        logger.info(
            "Bootstrap started for %s (attempt %d/%d)",
            anima_name,
            retry_count + 1,
            self._bootstrap_max_retries,
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
                "run_bootstrap",
                {},
                timeout=600.0,
            )

            if response.error:
                logger.error(
                    "Bootstrap failed for %s: %s",
                    anima_name,
                    response.error.get("message", "Unknown error"),
                )
                try:
                    from core.anima.bootstrap_state import mark_bootstrap_failed

                    mark_bootstrap_failed(
                        self.animas_dir / anima_name,
                        response.error.get("message", "Unknown error"),
                    )
                except Exception:
                    logger.debug("Failed to mark bootstrap failure for %s", anima_name, exc_info=True)
                await self._broadcast_event(
                    "anima.bootstrap",
                    {"name": anima_name, "status": "failed"},
                )
                return

            # The bootstrap IPC completed successfully.  Validation outcomes such
            # as ``needs_repair`` should surface to operators, but they are not
            # retryable transport/execution failures and must not consume the
            # retry budget that protects against infinite restart loops.
            success = True

            result = response.result or {}
            bootstrap_status: dict[str, Any] = {}
            try:
                from core.anima.bootstrap_state import finalize_bootstrap_run, get_bootstrap_status

                bootstrap_status = get_bootstrap_status(self.animas_dir / anima_name)
                if bootstrap_status.get("validation_errors") == ["bootstrap_file_left_after_definition"]:
                    bootstrap_status = finalize_bootstrap_run(self.animas_dir / anima_name)
            except Exception:
                logger.debug("Failed to read bootstrap status for %s", anima_name, exc_info=True)

            if bootstrap_status and bootstrap_status.get("needs_repair"):
                logger.error(
                    "Bootstrap finished for %s but needs repair: %s",
                    anima_name,
                    bootstrap_status.get("validation_errors", []),
                )
                await self._broadcast_event(
                    "anima.bootstrap",
                    {
                        "name": anima_name,
                        "status": "needs_repair",
                        "bootstrap_state": bootstrap_status,
                    },
                )
                return

            result_status = str(result.get("status") or result.get("action") or "")
            if (
                bootstrap_status
                and bootstrap_status.get("state") != "completed"
                and result_status
                not in {
                    "completed",
                    "complete",
                    "success",
                }
            ):
                logger.error(
                    "Bootstrap did not complete for %s: result=%s state=%s",
                    anima_name,
                    result_status,
                    bootstrap_status.get("state"),
                )
                await self._broadcast_event(
                    "anima.bootstrap",
                    {
                        "name": anima_name,
                        "status": "failed",
                        "bootstrap_state": bootstrap_status,
                    },
                )
                return

            logger.info(
                "Bootstrap completed for %s (duration_ms=%s)",
                anima_name,
                result.get("duration_ms", "?"),
            )
            await self._broadcast_event(
                "anima.bootstrap",
                {"name": anima_name, "status": "completed", "bootstrap_state": bootstrap_status},
            )
            success = True

        except TimeoutError:
            logger.error("Bootstrap timed out for %s (600s)", anima_name)
            try:
                from core.anima.bootstrap_state import mark_bootstrap_failed

                mark_bootstrap_failed(self.animas_dir / anima_name, "timeout")
            except Exception:
                logger.debug("Failed to mark bootstrap timeout for %s", anima_name, exc_info=True)
            await self._broadcast_event(
                "anima.bootstrap",
                {"name": anima_name, "status": "failed"},
            )
        except Exception:
            logger.exception("Bootstrap error for %s", anima_name)
            try:
                from core.anima.bootstrap_state import mark_bootstrap_failed

                mark_bootstrap_failed(self.animas_dir / anima_name, "supervisor_exception")
            except Exception:
                logger.debug("Failed to mark bootstrap exception for %s", anima_name, exc_info=True)
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
                # The UI tells the user a failed setup is retried; do it, with a
                # pause so a transient error (rate limit, network) can clear.
                # _run_bootstrap stops at _bootstrap_max_retries.
                spawn(
                    self._retry_bootstrap_later(anima_name),
                    name=f"bootstrap-retry-{anima_name}",
                )
            self._save_bootstrap_retries()
            if was_bootstrapping:
                handle = self.processes.get(anima_name)
                if not handle or handle.state != ProcessState.RUNNING:
                    logger.warning(
                        "Bootstrap for %s ended with process not running (possible reconciliation interference)",
                        anima_name,
                    )

    async def _retry_bootstrap_later(self, anima_name: str) -> None:
        """Re-run a failed background bootstrap after a short pause."""
        await asyncio.sleep(self._bootstrap_retry_delay_s)
        anima_dir = self.animas_dir / anima_name
        if anima_name in self._bootstrapping or not (anima_dir / BOOTSTRAP_FILE).exists():
            return
        handle = self.processes.get(anima_name)
        if not handle or handle.state != ProcessState.RUNNING:
            return
        await self._run_bootstrap(anima_name)

    async def _broadcast_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast a WebSocket event if ws_manager is available."""
        if self.ws_manager:
            await self.ws_manager.broadcast({"type": event_type, "data": data})

    async def stop_anima(
        self,
        anima_name: str,
        *,
        drain_streams: bool = True,
        drain_background: bool = True,
        drain_timeout: float | None = None,
        quiesce: bool = False,
    ) -> None:
        """Stop a single Anima process.

        Args:
            drain_streams: When True (default), wait for an in-flight chat
                stream to finish before stopping (protects user responses from
                rolling restarts / RAG repair). Full shutdown passes False for
                a prompt exit.
            drain_background: When True (default), wait for TaskExec lanes to
                become idle. Full shutdown passes False for a prompt exit.
            drain_timeout: Upper bound for each drain. None
                uses the process-handle default. Non-urgent stops (RAG repair)
                pass a longer bound so a response is not cut off mid-turn.
            quiesce: Stop new autonomous work before draining and stopping.
        """
        lock = self._lifecycle_locks.setdefault(anima_name, asyncio.Lock())
        async with lock:
            handle = self.processes.get(anima_name)
            if not handle:
                # Late arrival after a concurrent stop already finished, or
                # stop requested for a process that is not running — both are
                # normal no-ops under concurrent disable/reconcile callers.
                logger.debug("Process not found (no-op stop): %s", anima_name)
                return

            if quiesce:
                await self._quiesce_anima(anima_name, handle)

            await self._cancel_event_consumer(anima_name)
            await handle.stop(
                # Full-server shutdown deliberately retains the short budget
                # so systemd is not held up; ordinary stops/restarts get the
                # configured grace period for in-flight persistence.
                timeout=10.0 if self._shutdown else self._anima_stop_timeout,
                drain_streams=drain_streams,
                drain_background=drain_background,
                drain_timeout=drain_timeout,
            )
            # Only pop if the same handle is still registered — a concurrent
            # start (after this stop finished draining) may have replaced it.
            if self.processes.get(anima_name) is handle:
                self.processes.pop(anima_name, None)
            else:
                logger.warning(
                    "Skip processes.pop for %s: handle replaced during stop",
                    anima_name,
                )
            self._recently_stopped[anima_name] = time.monotonic()
            logger.info("Anima process stopped: %s", anima_name)

    async def _quiesce_anima(self, anima_name: str, handle: ProcessHandle) -> None:
        """Tell an Anima to stop accepting new autonomous work."""
        logger.info("Quiescing new work for Anima: %s", anima_name)
        try:
            response = await handle.send_request("quiesce", {}, timeout=5.0)
            error = getattr(response, "error", None)
            if error:
                logger.warning("Could not quiesce Anima %s: %s", anima_name, error)
                return
            logger.info("Anima quiesced for new autonomous work: %s", anima_name)
        except Exception as exc:
            logger.warning("Could not quiesce Anima %s: %s", anima_name, exc)

    async def _wait_for_processes_to_drain(self, timeout: float) -> None:  # noqa: ASYNC109 -- bounds the coordinated shutdown drain
        """Wait in parallel for all live Animas to finish active work."""
        if timeout <= 0:
            logger.info("Skipping Anima work drain because timeout is 0 seconds")
            return

        handles = {
            name: handle
            for name, handle in self.processes.items()
            if getattr(handle, "process", None) is not None and handle.process.poll() is None
        }
        if not handles:
            logger.info("No live Anima processes to drain")
            return

        logger.info(
            "Starting shutdown work drain: animas=%d timeout=%.0fs",
            len(handles),
            timeout,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        waiting: dict[str, set[str]] = {}
        while handles:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break

            async def inspect(name: str, handle: ProcessHandle) -> tuple[str, set[str]]:
                process = getattr(handle, "process", None)
                if process is None or process.poll() is not None:
                    return name, set()
                try:
                    details = await handle.ping(return_details=True)
                except Exception:
                    logger.warning("Could not inspect Anima work while draining: %s", name, exc_info=True)
                    return name, {"status unavailable"}
                if not isinstance(details, dict):
                    return name, {"status unavailable"}
                work_types: set[str] = set()
                if bool(getattr(handle, "is_streaming", False)):
                    work_types.add("interactive stream")
                if not details.get("success"):
                    work_types.add("status unavailable")
                elif details.get("is_busy"):
                    work_types.add("busy task")
                return name, work_types

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*(inspect(name, handle) for name, handle in handles.items())),
                    timeout=remaining,
                )
            except TimeoutError:
                for name in handles:
                    waiting.setdefault(name, {"status unavailable"})
                break

            next_waiting: dict[str, set[str]] = {}
            for name, work_types in results:
                if not work_types:
                    if name in waiting:
                        logger.info("Anima work drained: anima=%s work=%s", name, ", ".join(sorted(waiting[name])))
                    continue
                next_waiting[name] = work_types
                if waiting.get(name) != work_types:
                    logger.info(
                        "Waiting for Anima work: anima=%s work=%s",
                        name,
                        ", ".join(sorted(work_types)),
                    )
            waiting = next_waiting
            handles = {name: handle for name, handle in handles.items() if name in waiting}
            if not handles:
                logger.info("Shutdown work drain completed")
                return
            await asyncio.sleep(min(1.0, max(0.0, deadline - loop.time())))

        for name, work_types in waiting.items():
            logger.warning(
                "Shutdown work drain timed out: anima=%s work=%s timeout=%.0fs",
                name,
                ", ".join(sorted(work_types)),
                timeout,
            )
        logger.info("Shutdown work drain finished; proceeding with process stop")

    async def restart_anima(
        self,
        anima_name: str,
        *,
        _reset_counters: bool = True,
    ) -> None:
        """Restart a Anima process.

        Args:
            _reset_counters: When True (default), resets failure tracking
                state.  Internal callers (e.g. ``_handle_process_failure``)
                pass False to preserve the retry counter.
        """
        logger.info("Restarting process: %s", anima_name)

        if _reset_counters and not self._shutdown:
            if self._restart_ctl is not None:
                self._restart_ctl.reset(anima_name)
            # A manual restart supersedes a worker waiting in backoff. Cancel
            # and join it before taking the lifecycle guard so it cannot later
            # perform another stop/start for this anima.
            worker_task = self._restart_worker_tasks.pop(anima_name, None)
            if worker_task is not None and worker_task is not asyncio.current_task() and not worker_task.done():
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)

        # Guard against reconciliation spawning a duplicate process
        # during the window between stop and start.
        self._restarting.add(anima_name)
        try:
            if anima_name in self.processes:
                await self.stop_anima(
                    anima_name,
                    quiesce=True,
                )
            await self.start_anima(anima_name)
        finally:
            self._restarting.discard(anima_name)

    async def _mark_process_error(self, anima_name: str, reason: str, handle: ProcessHandle | None = None) -> None:
        """Surface a visible process error even when no live handle remains.

        Broadcasts the restart state only; it does not mutate restart state
        itself (callers use ``record_failure``). If no FAILED record exists
        yet (e.g. an external caller), record one for mutual compatibility.
        """
        # Never mark errors while the server is shutting down.
        if self._shutdown:
            logger.debug("Skip error marking during shutdown: %s", anima_name)
            return
        if handle is None:
            handle = self.processes.get(anima_name)
        if handle is not None:
            handle.state = ProcessState.FAILED
        if self._restart_ctl is not None and not self._restart_ctl.is_failed(anima_name):
            self._restart_ctl.record_failure(anima_name, reason)
        snapshot = self._restart_ctl.snapshot(anima_name) if self._restart_ctl is not None else {}
        await self._broadcast_event(
            "anima.status",
            {"name": anima_name, "status": "error", "error": reason, **snapshot},
        )

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

        # Stop fallback event-file draining and root event subscriptions.
        if self._event_file_drain_task:
            self._event_file_drain_task.cancel()
            await asyncio.gather(self._event_file_drain_task, return_exceptions=True)
        await asyncio.gather(
            *(self._cancel_event_consumer(name) for name in list(self._event_tasks)),
            return_exceptions=True,
        )

        # Stop zombie reaper
        if self._zombie_reaper_task:
            self._zombie_reaper_task.cancel()
            try:
                await self._zombie_reaper_task
            except asyncio.CancelledError:
                pass

        # Quiesce every live Anima before waiting so no new scheduler, inbox,
        # or pending-task work enters while the shared drain budget is used.
        live_processes = [
            (name, handle)
            for name, handle in list(self.processes.items())
            if getattr(handle, "process", None) is not None and handle.process.poll() is None
        ]
        logger.info("Quiescing Animas for server shutdown: count=%d", len(live_processes))
        await asyncio.gather(
            *(self._quiesce_anima(name, handle) for name, handle in live_processes),
            return_exceptions=True,
        )
        await self._wait_for_processes_to_drain(self._restart_drain_timeout_sec)

        # Stop all processes. Full shutdown should exit promptly, so skip
        # interactive-stream and background-lane drains here.
        tasks = [
            self.stop_anima(
                name,
                drain_streams=False,
                drain_background=False,
            )
            for name in list(self.processes.keys())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Barrier: an in-flight start_anima cleans up its own handle under the
        # lifecycle lock when it observes _shutdown. Acquire every lock once so
        # this method does not return while such a cleanup is still running.
        for lock in list(self._lifecycle_locks.values()):
            async with lock:
                pass

        logger.info("All processes shut down")

    # ── IPC ───────────────────────────────────────────────────────

    async def send_request(
        self,
        anima_name: str,
        method: str,
        params: dict[str, Any],
        timeout: float = 60.0,  # noqa: ASYNC109 -- timeout bounds awaited work and is part of this async API
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
            raise IPCConnectionError(f"Request failed: {response.error.get('message', 'Unknown error')}")

        return response.result or {}

    async def send_request_stream(
        self,
        anima_name: str,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,  # noqa: ASYNC109 -- timeout bounds awaited work and is part of this async API
    ) -> AsyncIterator[IPCResponse]:
        """Send IPC request to a Anima process and yield streaming responses.

        Raises:
            AnimaNotFoundError: If anima not found
            IPCConnectionError: If response contains error
        """
        handle = self.processes.get(anima_name)
        if not handle:
            raise AnimaNotFoundError(f"Anima not found: {anima_name}")

        async for response in handle.send_request_stream(method, params, timeout):
            if response.error:
                raise IPCConnectionError(f"Stream error: {response.error.get('message', 'Unknown error')}")
            yield response

    async def _zombie_reaper_loop(self) -> None:
        """Poll only our Anima Popen owners as a periodic safety net.

        Never reap arbitrary children: taking their wait status makes another
        Popen report a fabricated success (ECHILD -> returncode=0), and can
        steal completion from asyncio, SDK or subordinate subprocess owners.
        Popen.poll preserves the real exit status for health reconciliation.
        """
        from core.i18n import t as _t

        while not self._shutdown:
            try:
                await asyncio.sleep(60)
                reaped = 0
                for handle in list(self.processes.values()):
                    process = handle.process
                    if process is None or process.returncode is not None:
                        continue
                    try:
                        if process.poll() is not None:
                            reaped += 1
                    except Exception:
                        logger.debug("Anima process poll failed in zombie reaper", exc_info=True)
                if reaped:
                    logger.info(_t("supervisor.zombie_reaped", count=reaped))
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("zombie reaper error", exc_info=True)

    async def _cancel_event_consumer(self, anima_name: str) -> None:
        """Cancel and join the event subscription task for one process."""
        task = self._event_tasks.pop(anima_name, None)
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _consume_events(self, anima_name: str, handle: ProcessHandle) -> None:
        """Consume pushed root events while *handle* remains registered."""
        current_task = asyncio.current_task()
        try:
            while (
                not self._shutdown and self.processes.get(anima_name) is handle and handle.state == ProcessState.RUNNING
            ):
                try:
                    async with contextlib.aclosing(handle.open_event_stream()) as events:
                        async for event in events:
                            if self.processes.get(anima_name) is not handle:
                                return
                            event_type = event.get("event")
                            event_data = event.get("data", {})
                            if isinstance(event_type, str) and isinstance(event_data, dict):
                                await self._broadcast_event(event_type, event_data)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("Event stream disconnected for %s", anima_name, exc_info=True)

                if (
                    self._shutdown
                    or self.processes.get(anima_name) is not handle
                    or handle.state != ProcessState.RUNNING
                ):
                    return
                await asyncio.sleep(1.0)
        finally:
            if self._event_tasks.get(anima_name) is current_task:
                self._event_tasks.pop(anima_name, None)

    async def _event_file_drain_loop(self) -> None:
        """Drain fallback event files without coupling them to health checks."""
        while not self._shutdown:
            try:
                await self._poll_anima_events()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Event file drain failed", exc_info=True)
            await asyncio.sleep(1.0)

    async def _poll_anima_events(self) -> None:
        """Read and broadcast event files from non-root child processes."""
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
            starting_since = self._starting_since.get(anima_name)
            if starting_since is not None:
                elapsed = time.monotonic() - starting_since
                if elapsed > self._spawn_timeout_sec:
                    reason = f"spawn exceeded timeout ({self._spawn_timeout_sec:.0f}s)"
                    return {
                        "status": "error",
                        "error": reason,
                        "state_since": starting_since,
                        "spawn_started_at": starting_since,
                        "uptime_sec": elapsed,
                    }
                return {
                    "status": "starting",
                    "state_since": starting_since,
                    "spawn_started_at": starting_since,
                    "uptime_sec": elapsed,
                }

            if self._restart_ctl is not None and self._restart_ctl.is_failed(anima_name):
                snapshot = self._restart_ctl.snapshot(anima_name)
                return {
                    "status": "error",
                    "error": snapshot.get("last_error") or "process failed",
                    **snapshot,
                }
            if self._restart_ctl is not None and self._restart_ctl.get(anima_name) is not None:
                snapshot = self._restart_ctl.snapshot(anima_name)
                if snapshot.get("restart_state") == "backoff":
                    # BACKOFF and not running -> restore is in progress
                    return {
                        "status": "restarting",
                        **snapshot,
                    }

            status = {"status": "not_found"}
            try:
                from core.anima.bootstrap_state import get_bootstrap_status

                bootstrap_status = get_bootstrap_status(self.animas_dir / anima_name)
                status.update(
                    {
                        "bootstrap_state": bootstrap_status,
                        "needs_bootstrap": bootstrap_status.get("needs_bootstrap", False),
                        "needs_user_input": bootstrap_status.get("needs_user_input", False),
                        "needs_repair": bootstrap_status.get("needs_repair", False),
                        "needs_background_bootstrap": bootstrap_status.get("needs_background_bootstrap", False),
                    }
                )
            except Exception:
                logger.debug("Failed to read bootstrap status for %s", anima_name, exc_info=True)
            return status

        uptime = (now_local() - ensure_aware(handle.stats.started_at)).total_seconds()
        status_value = "bootstrapping" if self.is_bootstrapping(anima_name) else handle.state.value
        reason = None
        if handle.state in (ProcessState.STARTING, ProcessState.RESTARTING) and uptime > self._spawn_timeout_sec:
            status_value = "error"
            reason = f"spawn exceeded timeout ({self._spawn_timeout_sec:.0f}s)"

        bootstrap_status: dict[str, Any] = {}
        try:
            from core.anima.bootstrap_state import get_bootstrap_status

            bootstrap_status = get_bootstrap_status(self.animas_dir / anima_name)
        except Exception:
            logger.debug("Failed to read bootstrap status for %s", anima_name, exc_info=True)

        subprocesses: list[dict] = []
        try:
            sidecar = self._read_busy_sidecar(anima_name, handle)
            if sidecar:
                subprocesses = [p for p in sidecar.get("processes") or [] if isinstance(p, dict)]
        except Exception:
            logger.debug("Failed to read subprocess info for %s", anima_name, exc_info=True)

        snapshot = self._restart_ctl.snapshot(anima_name) if self._restart_ctl is not None else {}
        return {
            "status": status_value,
            "error": reason if reason else snapshot.get("last_error"),
            "pid": handle.get_pid(),
            "process_count": 1 + len(subprocesses),
            "subprocesses": subprocesses,
            "uptime_sec": uptime,
            **snapshot,
            "missed_pings": handle.stats.missed_pings,
            "last_busy_since": (handle.stats.last_busy_since.isoformat() if handle.stats.last_busy_since else None),
            "bootstrapping": self.is_bootstrapping(anima_name),
            "bootstrap_state": bootstrap_status,
            "needs_bootstrap": bootstrap_status.get("needs_bootstrap", False),
            "needs_user_input": bootstrap_status.get("needs_user_input", False),
            "needs_repair": bootstrap_status.get("needs_repair", False),
            "needs_background_bootstrap": bootstrap_status.get("needs_background_bootstrap", False),
            "last_ping_at": (handle.stats.last_ping_at.isoformat() if handle.stats.last_ping_at else None),
        }

    def get_all_status(self) -> dict[str, dict]:
        """Get status of all processes."""
        names = (
            set(self.processes)
            | set(self._starting_since)
            | (self._restart_ctl.names() if self._restart_ctl is not None else set())
        )
        return {name: self.get_process_status(name) for name in sorted(names)}
