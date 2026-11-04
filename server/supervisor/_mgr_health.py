"""
Health check mixin for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.platform.tasks import spawn
from core.time_utils import ensure_aware, now_local
from server.supervisor._manager_protocols import _HealthMixinHost
from server.supervisor.process_handle import ProcessHandle, ProcessState

logger = logging.getLogger(__name__)


class HealthMixin:
    """Health-check loop, failure handling, and hang detection."""

    def is_bootstrapping(self: _HealthMixinHost, anima_name: str) -> bool:
        """Return True if the anima is currently in bootstrap mode."""
        return anima_name in getattr(self, "_bootstrapping", set())

    def _busy_sidecar_path(self: _HealthMixinHost, anima_name: str) -> Path | None:
        """Return the IPC-independent busy marker path, if run_dir is available."""
        run_dir = getattr(self, "run_dir", None)
        if run_dir is None:
            return None
        return Path(run_dir) / "animas" / f"{anima_name}.busy.json"

    def _read_busy_sidecar(self: _HealthMixinHost, anima_name: str, handle: ProcessHandle) -> dict[str, Any] | None:
        """Read a child-written busy marker for ping-timeout fallback."""
        path = self._busy_sidecar_path(anima_name)
        if path is None or not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or not data.get("is_busy"):
            return None

        marker_pid = data.get("pid")
        current_pid = handle.get_pid()
        if marker_pid is None or current_pid is None:
            return None
        try:
            if int(marker_pid) != int(current_pid):
                return None
        except (TypeError, ValueError):
            return None

        last_progress = data.get("last_progress_at") or data.get("updated_at")
        if last_progress:
            data["last_progress_at"] = last_progress
        return data

    def _health_warmup_reason(self: _HealthMixinHost, anima_name: str, handle: ProcessHandle) -> str | None:
        """Return a reason to suppress unresponsive-runner restarts, if any."""
        try:
            from core.infra import startup_progress

            snapshot = startup_progress.snapshot()
            if snapshot.get("status") == "starting":
                return f"server startup phase={snapshot.get('phase')}"
            ready_at = snapshot.get("ready_at")
            if isinstance(ready_at, int | float) and ready_at > 0:
                elapsed = time.time() - float(ready_at)
                warmup = float(getattr(self.health_config, "health_check_warmup_seconds", 0.0))
                if elapsed < warmup:
                    return f"server startup warmup {elapsed:.0f}s/{warmup:.0f}s"
        except Exception:
            logger.debug("Failed to inspect startup progress for health warmup", exc_info=True)

        uptime = (now_local() - ensure_aware(handle.stats.started_at)).total_seconds()
        runner_warmup = float(getattr(self.health_config, "runner_warmup_seconds", 0.0))
        if uptime < runner_warmup:
            return f"runner warmup {uptime:.0f}s/{runner_warmup:.0f}s"
        return None

    async def _health_check_loop(self: _HealthMixinHost) -> None:
        """Periodically pings all processes and handles failures."""
        logger.info("Health check loop started")

        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_config.ping_interval_sec)

                await self._poll_requested_rag_repairs()

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
        self: _HealthMixinHost,
        anima_name: str,
        handle: ProcessHandle,
    ) -> None:
        """Check health of a single process."""
        # Skip FAILED processes (log at WARNING every 5 minutes). Auto-recovery
        # is driven by the restart worker with exponential backoff.
        if (
            self._restart_ctl is not None
            and self._restart_ctl.is_failed(anima_name)
            and anima_name not in self.processes
        ):
            if self._restart_ctl.should_log_failed(anima_name):
                logger.warning(
                    "Process still in FAILED state: %s (auto-recovery continues, next retry in %.0fs)",
                    anima_name,
                    self._restart_ctl.seconds_until_due(anima_name),
                )
            return

        # Detect handles stuck in STOPPING state (e.g. after failed shutdown)
        if handle.state == ProcessState.STOPPING:
            if self._shutdown or not handle.stopping_since:
                return
            stopping_duration = (now_local() - ensure_aware(handle.stopping_since)).total_seconds()
            if stopping_duration > 30:
                logger.error(
                    "Process stuck in STOPPING state: %s (%.0fs)",
                    anima_name,
                    stopping_duration,
                )
                handle.state = ProcessState.FAILED
                spawn(
                    self._handle_process_failure(anima_name, handle),
                    name=f"process-failure-{anima_name}",
                )
            return

        # RESTARTING 状態ならヘルスチェックをスキップ
        if handle.state == ProcessState.RESTARTING:
            return

        # During streaming the IPC lock is held. Retain process liveness
        # detection, but leave turn-idle timeout enforcement to the engine.
        if handle.is_streaming:
            # Detect process death during streaming
            if handle.state == ProcessState.FAILED:
                logger.error(
                    "Process FAILED during streaming: %s",
                    anima_name,
                )
                spawn(
                    self._handle_process_failure(anima_name, handle),
                    name=f"process-failure-{anima_name}",
                )
                return
            if not handle.is_alive():
                logger.error(
                    "Process died during streaming: %s (exit_code=%s)",
                    anima_name,
                    handle.stats.exit_code,
                )
                spawn(
                    self._handle_process_failure(anima_name, handle),
                    name=f"process-failure-{anima_name}",
                )
                return
            return

        # Direct state check: detect IPC connection loss
        if handle.state == ProcessState.FAILED:
            if handle.is_alive():
                warmup_reason = self._health_warmup_reason(anima_name, handle)
                if warmup_reason:
                    logger.warning(
                        "Suppressing failed-state restart for %s during %s",
                        anima_name,
                        warmup_reason,
                    )
                    return
            logger.error(
                "Process in FAILED state (IPC connection lost): %s",
                anima_name,
            )
            spawn(
                self._handle_process_failure(anima_name, handle),
                name=f"process-failure-{anima_name}",
            )
            return

        # Check if process is alive
        if not handle.is_alive():
            actual_rc = handle.process.returncode if handle.process else None
            handle.stats.exit_code = actual_rc
            logger.error(
                "Process exited unexpectedly: %s (exit_code=%s, signal=%s)",
                anima_name,
                actual_rc,
                -actual_rc if actual_rc is not None and actual_rc < 0 else "N/A",
            )
            spawn(
                self._handle_process_failure(anima_name, handle),
                name=f"process-failure-{anima_name}",
            )
            return

        warmup_reason = self._health_warmup_reason(anima_name, handle)
        if warmup_reason:
            logger.debug("Skipping health check for %s (%s)", anima_name, warmup_reason)
            return

        # Skip if in startup grace period
        uptime = (now_local() - ensure_aware(handle.stats.started_at)).total_seconds()
        if uptime < self.health_config.startup_grace_sec:
            logger.debug("Skipping health check for %s (startup grace)", anima_name)
            return

        # Reset restart state after stable uptime
        if self._restart_ctl is not None:
            self._restart_ctl.record_stable(anima_name, uptime)

        # Ping process
        ping_result = await handle.ping(
            timeout=self.health_config.ping_timeout_sec,
            return_details=True,
        )
        success = bool(ping_result.get("success"))
        is_busy = bool(ping_result.get("is_busy"))

        # Transport error fallback (Windows IPC): don't count as missed ping
        if not success and ping_result.get("transport_error"):
            if handle.is_alive():
                handle.stats.missed_pings = 0
                if not getattr(handle, "_transport_error_logged", False):
                    logger.warning(
                        "IPC transport unavailable for %s — falling back to liveness check only",
                        anima_name,
                    )
                    handle._transport_error_logged = True  # type: ignore[attr-defined]
                return
            # Process dead despite transport error
            actual_rc = handle.process.returncode if handle.process else None
            handle.stats.exit_code = actual_rc
            logger.error(
                "Process dead with transport error: %s (exit_code=%s)",
                anima_name,
                actual_rc,
            )
            spawn(
                self._handle_process_failure(anima_name, handle),
                name=f"process-failure-{anima_name}",
            )
            return

        if success:
            if is_busy:
                handle.stats.missed_pings = 0
                if handle.stats.last_busy_since is None:
                    handle.stats.last_busy_since = now_local()
                return

            if handle.stats.missed_pings > 0 or handle.stats.last_busy_since is not None:
                logger.info("Process recovered: %s", anima_name)
            handle.stats.last_busy_since = None
            return

        # Ping failed
        handle.stats.last_busy_since = None
        logger.warning(
            "Health check failed: %s (missed=%d/%d)",
            anima_name,
            handle.stats.missed_pings,
            self.health_config.max_missed_pings,
        )

        # Check if hang threshold exceeded
        if handle.stats.missed_pings >= self.health_config.max_missed_pings:
            logger.error(
                "Process hang detected: %s (PID %s)",
                anima_name,
                handle.get_pid(),
            )
            spawn(
                self._handle_process_hang(anima_name, handle),
                name=f"process-hang-{anima_name}",
            )

    async def _handle_process_failure(
        self: _HealthMixinHost,
        anima_name: str,
        handle: ProcessHandle,
        reason: str = "",
    ) -> None:
        """Record a failure and (re)ensure the restart worker.

        Runs as an independent task (created by callers).  The per-anima
        ``_restarting`` guard prevents duplicate workers.  Only records the
        failure and marks READY to restart; the backoff wait and single spawn
        attempt live in ``_restart_worker``.
        """
        # Entrance guard: during shutdown the whole failure/restart machinery
        # is a no-op — shutdown_all stops every process anyway, and any state
        # recorded here would only pollute the next server start.
        if self._shutdown:
            logger.info("Skip failure handling during shutdown: %s", anima_name)
            return
        if anima_name in self._restarting:
            return

        # Reserve the per-anima guard synchronously before the first await so
        # concurrent failure notifications cannot record duplicate attempts.
        self._restarting.add(anima_name)
        try:
            # リスタート対象であることをハンドルに反映
            if handle is not None:
                handle.state = ProcessState.RESTARTING

            # Disabled before any retry logic: clean stop, no state pollution.
            if not await asyncio.to_thread(self.read_anima_enabled, self.animas_dir / anima_name):
                logger.info("Skip restart: anima disabled: %s", anima_name)
                if anima_name in self.processes:
                    await self.stop_anima(anima_name)
                if self._restart_ctl is not None:
                    self._restart_ctl.forget(anima_name)
                return

            if self._restart_ctl is not None:
                self._restart_ctl.record_failure(anima_name, reason)
                if self._restart_ctl.is_failed(anima_name):
                    await self._mark_process_error(anima_name, reason, handle)
        finally:
            # Always release this handler's reservation and ensure recovery,
            # including when status broadcast (or another body operation) fails.
            self._restarting.discard(anima_name)
            if not self._shutdown and self._restart_ctl is not None and self._restart_ctl.get(anima_name) is not None:
                self._ensure_restart_worker(anima_name)

    def _ensure_restart_worker(self: _HealthMixinHost, anima_name: str) -> None:
        """Spawn a singleton restart worker for ``anima_name`` if needed."""
        if anima_name in self._restarting:
            return
        self._restarting.add(anima_name)
        self._restart_worker_tasks[anima_name] = asyncio.create_task(self._restart_worker(anima_name))

    async def _restart_worker(self: _HealthMixinHost, anima_name: str) -> None:
        """Single restart worker: await backoff, then one spawn attempt.

        Loops until a spawn succeeds, the anima is disabled, or shutdown.
        Even in FAILED state the worker keeps trying with exponential backoff.
        """
        worker_record = self._restart_ctl.get(anima_name) if self._restart_ctl is not None else None
        try:
            while not self._shutdown:
                # Poll enablement during long exponential-backoff waits so a
                # disabled anima does not retain a worker or retry record.
                while not self._shutdown:
                    if (
                        worker_record is not None
                        and self._restart_ctl is not None
                        and self._restart_ctl.get(anima_name) is not worker_record
                    ):
                        return
                    if not await asyncio.to_thread(self.read_anima_enabled, self.animas_dir / anima_name):
                        logger.info("Restart worker: anima disabled: %s", anima_name)
                        if anima_name in self.processes:
                            await self.stop_anima(anima_name)
                        if self._restart_ctl is not None:
                            self._restart_ctl.forget(anima_name)
                        return
                    if self._restart_ctl is None or self._restart_ctl.is_due(anima_name):
                        break
                    await asyncio.sleep(min(1.0, self._restart_ctl.seconds_until_due(anima_name)))
                if self._shutdown:
                    return

                # Recheck after the final wait before touching the process.
                if not await asyncio.to_thread(self.read_anima_enabled, self.animas_dir / anima_name):
                    logger.info("Restart worker: anima disabled: %s", anima_name)
                    if anima_name in self.processes:
                        await self.stop_anima(anima_name)
                    if self._restart_ctl is not None:
                        self._restart_ctl.forget(anima_name)
                    return

                try:
                    if anima_name in self.processes:
                        await self.stop_anima(anima_name)
                    if (
                        worker_record is not None
                        and self._restart_ctl is not None
                        and self._restart_ctl.get(anima_name) is not worker_record
                    ):
                        return
                    await self.start_anima(anima_name)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    logger.error("Restart attempt failed for %s: %s", anima_name, exc)
                    if self._restart_ctl is not None and (
                        worker_record is None or self._restart_ctl.get(anima_name) is worker_record
                    ):
                        self._restart_ctl.record_failure(anima_name, reason)
                        if self._restart_ctl.is_failed(anima_name):
                            await self._mark_process_error(anima_name, reason)
                    continue

                # Success: mark started and report running.
                if (
                    worker_record is not None
                    and self._restart_ctl is not None
                    and self._restart_ctl.get(anima_name) is not worker_record
                ):
                    return
                if self._restart_ctl is not None:
                    self._restart_ctl.record_started(anima_name)
                await self._broadcast_event(
                    "anima.status",
                    {"name": anima_name, "status": "running"},
                )
                logger.info("Process restarted: %s", anima_name)
                return
        finally:
            self._restarting.discard(anima_name)
            worker_task = self._restart_worker_tasks.get(anima_name)
            if worker_task is asyncio.current_task():
                self._restart_worker_tasks.pop(anima_name, None)

    async def _handle_process_hang(
        self: _HealthMixinHost,
        anima_name: str,
        handle: ProcessHandle,
    ) -> None:
        """Handle hung process (kill and restart)."""
        logger.warning("Killing hung process: %s", anima_name)

        # Kill process
        await handle.kill()

        # Restart
        await self._handle_process_failure(anima_name, handle, reason="hang")

    # ── Reconciliation helper ────────────────────────────────────

    @staticmethod
    def read_anima_enabled(anima_dir: Path) -> bool:
        """Read the enabled flag from an anima's status.json.

        Returns True (enabled) when:
        - status.json does not exist (backward compatibility)
        - status.json exists with ``enabled: true``

        Returns False when status.json exists with ``enabled: false``.
        """
        from core.platform.status_store import read_status

        return bool(read_status(anima_dir).get("enabled", True))
