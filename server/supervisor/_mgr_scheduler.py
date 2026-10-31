"""
System scheduler mixin for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.anima.workspace_files import IDENTITY_FILE
from core.memory.rag.store import CollectionExistence
from core.platform.tasks import spawn
from core.time_utils import get_app_timezone, now_local
from server.supervisor._manager_protocols import _SchedulerMixinHost
from server.supervisor.process_handle import ProcessState

logger = logging.getLogger(__name__)

_ConsolidationOutcome = Literal["ran", "skipped", "timed_out", "failed"]
_ConsolidationWorker = Callable[[str, Path], Awaitable[_ConsolidationOutcome]]


async def _run_bounded(
    items: list[tuple[str, Path]],
    worker: _ConsolidationWorker,
    *,
    limit: int,
    label: str,
) -> list[_ConsolidationOutcome]:
    """Run per-Anima workers concurrently while isolating individual failures."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run_one(anima_name: str, anima_dir: Path) -> _ConsolidationOutcome:
        async with semaphore:
            try:
                return await worker(anima_name, anima_dir)
            except Exception:
                logger.exception("%s worker failed for anima=%s", label, anima_name)
                return "failed"

    results = await asyncio.gather(
        *(run_one(anima_name, anima_dir) for anima_name, anima_dir in items),
        return_exceptions=True,
    )
    outcomes: list[_ConsolidationOutcome] = []
    for (anima_name, _), result in zip(items, results, strict=True):
        if isinstance(result, BaseException):
            logger.error(
                "%s worker failed for anima=%s",
                label,
                anima_name,
                exc_info=(type(result), result, result.__traceback__),
            )
            outcomes.append("failed")
        else:
            outcomes.append(result)
    return outcomes


def _resolve_consolidation_concurrency(consolidation_cfg: object | None, default: int) -> int:
    """Return a positive configured Anima concurrency limit."""
    value = getattr(consolidation_cfg, "max_concurrent_animas", default)
    try:
        return max(1, int(value))
    except (TypeError, ValueError, OverflowError):
        resolved_default = max(1, default)
        logger.warning(
            "Invalid consolidation.max_concurrent_animas=%r; using %d",
            value,
            resolved_default,
        )
        return resolved_default


# ── Marker helpers ──────────────────────────────────────────────────
_MARKER_DIR_NAME = "run"


def _marker_dir(data_dir: Path) -> Path:
    d = data_dir / _MARKER_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_marker(marker_path: Path) -> datetime | None:
    """Read an ISO-8601 timestamp from a marker file."""
    try:
        raw = marker_path.read_text().strip()
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _write_marker(marker_path: Path, ts: datetime | None = None) -> None:
    ts = ts or now_local()
    marker_path.write_text(ts.isoformat())


class SchedulerMixin:
    """System-level cron scheduler for memory consolidation and log rotation."""

    def _start_system_scheduler(self: _SchedulerMixinHost) -> None:
        """Start the system-level scheduler for consolidation crons."""
        try:
            self.scheduler = AsyncIOScheduler(timezone=get_app_timezone())
            self._setup_system_crons()
            self.scheduler.start()
            self._scheduler_running = True
            logger.info("System scheduler started")
            spawn(self._apply_activity_schedule_tick(), name="scheduler-activity-level-startup")
            spawn(self._catchup_missed_jobs(), name="scheduler-catchup-missed-jobs")
        except Exception:
            logger.exception("Failed to start system scheduler")
            self.scheduler = None
            self._scheduler_running = False

    def _setup_system_crons(self: _SchedulerMixinHost) -> None:
        """Register system-wide cron jobs for memory consolidation."""
        if not self.scheduler:
            return

        # Load consolidation config
        try:
            from core.config import load_config
            from core.config.models import ConsolidationConfig

            config = load_config()
            consolidation_cfg = config.consolidation or ConsolidationConfig()
        except Exception:
            logger.debug("Config load failed for consolidation schedule", exc_info=True)
            from core.config.models import ConsolidationConfig

            consolidation_cfg = ConsolidationConfig()

        # Daily consolidation
        daily_enabled = consolidation_cfg.daily_enabled
        daily_time = consolidation_cfg.daily_time

        if daily_enabled:
            hour, minute = (int(x) for x in daily_time.split(":"))
            self.scheduler.add_job(
                self._run_daily_consolidation,
                CronTrigger(hour=hour, minute=minute),
                id="system_daily_consolidation",
                name="System: Daily Consolidation",
                replace_existing=True,
                misfire_grace_time=600,
            )
            logger.info("System cron: Daily consolidation at %s", daily_time)

        # Weekly integration
        weekly_enabled = consolidation_cfg.weekly_enabled
        weekly_time = consolidation_cfg.weekly_time

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
                misfire_grace_time=600,
            )
            logger.info("System cron: Weekly integration on %s at %s:%s", day_of_week, time_parts[0], time_parts[1])

        indexing_enabled = consolidation_cfg.indexing_enabled
        indexing_time = consolidation_cfg.indexing_time

        if indexing_enabled:
            idx_hour, idx_minute = (int(x) for x in indexing_time.split(":"))
            self.scheduler.add_job(
                self._run_daily_indexing,
                CronTrigger(hour=idx_hour, minute=idx_minute),
                id="system_daily_indexing",
                name="System: Daily RAG Indexing",
                replace_existing=True,
                misfire_grace_time=600,
            )
            logger.info("System cron: Daily RAG indexing at %s", indexing_time)

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
                    misfire_grace_time=600,
                )
                logger.info("System cron: Activity log rotation at %s", activity_cfg.rotation_time)
        except Exception:
            logger.debug("Activity log rotation schedule setup failed", exc_info=True)

        # Housekeeping
        try:
            from core.config.models import HousekeepingConfig

            hk_cfg: HousekeepingConfig | None = None
            try:
                from core.config import load_config as _load_hk

                _hk = getattr(_load_hk(), "housekeeping", None)
                if isinstance(_hk, HousekeepingConfig):
                    hk_cfg = _hk
            except Exception:
                logger.debug("Config load failed for housekeeping schedule", exc_info=True)

            if hk_cfg is None:
                hk_cfg = HousekeepingConfig()

            try:
                if hk_cfg.enabled:
                    hk_hour, hk_minute = (int(x) for x in hk_cfg.run_time.split(":"))
                    self.scheduler.add_job(
                        self._run_housekeeping,
                        CronTrigger(hour=hk_hour, minute=hk_minute),
                        id="system_housekeeping",
                        name="System: Housekeeping",
                        replace_existing=True,
                        misfire_grace_time=600,
                    )
                    logger.info("System cron: Housekeeping at %s", hk_cfg.run_time)
            except Exception:
                logger.debug("Daily housekeeping schedule setup failed", exc_info=True)

            try:
                if hk_cfg.workspace_sweep_enabled:
                    self.scheduler.add_job(
                        self._run_workspace_sweep,
                        CronTrigger(minute=0),
                        id="system_workspace_sweep",
                        name="System: Workspace Sweep",
                        replace_existing=True,
                        misfire_grace_time=600,
                    )
                    logger.info("System cron: Workspace sweep every hour")
            except Exception:
                logger.debug("Workspace sweep schedule setup failed", exc_info=True)
        except Exception:
            logger.debug("Housekeeping schedule setup failed", exc_info=True)

        # DM log rotation
        self.scheduler.add_job(
            self._run_dm_log_rotation,
            CronTrigger(hour=4, minute=30),
            id="system_dm_log_rotation",
            name="System: DM Log Rotation",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("System cron: DM log rotation at 04:30")

        self.scheduler.add_job(
            self._apply_activity_schedule_tick,
            CronTrigger(minute="*"),
            id="system_activity_schedule",
            name="System: Activity Schedule",
            replace_existing=True,
            misfire_grace_time=120,
            max_instances=1,
        )
        logger.info("System cron: Activity schedule every minute")

    async def _apply_activity_schedule_tick(self: _SchedulerMixinHost) -> None:
        """Apply the active activity-schedule slot once from the root process."""
        from server.supervisor.activity_schedule import apply_activity_schedule

        try:
            update = await asyncio.to_thread(apply_activity_schedule)
        except Exception:
            logger.exception("Root activity schedule application failed")
            return
        if not update.activity_level_changed:
            return

        logger.info("Activity schedule changed global level to %d%%", update.config.activity_level)
        for name in list(self.processes):
            handle = self.processes.get(name)
            if handle is None:
                continue
            try:
                response = await handle.send_request("reschedule_heartbeat", {}, timeout=10.0)
                if response.error:
                    raise RuntimeError(str(response.error))
            except Exception:
                logger.warning("Failed to send reschedule_heartbeat to %s after activity schedule change", name)

    def _iter_consolidation_targets(self: _SchedulerMixinHost) -> list[tuple[str, Path]]:
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
            if not (anima_dir / IDENTITY_FILE).exists():
                continue
            if not (anima_dir / "status.json").exists():
                continue
            if not self.read_anima_enabled(anima_dir):
                continue
            targets.append((anima_dir.name, anima_dir))
        return targets

    def _get_data_dir(self: _SchedulerMixinHost) -> Path:
        """Return the runtime data directory (``~/.animaworks`` or override)."""
        return self.animas_dir.parent

    @staticmethod
    def _resolve_consolidation_ipc_timeout(
        consolidation_cfg: object | None,
        *,
        consolidation_type: str,
        gate: object | None = None,
    ) -> float:
        """Size consolidation IPC timeout from observed workload.

        The old fixed 1800s cap caused scheduler-side interrupts at the
        30-minute mark.  Daily consolidation now scales with the gate counts
        that triggered the run; weekly uses its own explicit cap.
        """
        from core.config.models import ConsolidationConfig

        defaults = ConsolidationConfig()
        if consolidation_type == "weekly":
            return float(getattr(consolidation_cfg, "weekly_ipc_timeout_seconds", defaults.weekly_ipc_timeout_seconds))

        base = float(getattr(consolidation_cfg, "ipc_timeout_base_seconds", defaults.ipc_timeout_base_seconds))
        per_activity = float(
            getattr(
                consolidation_cfg,
                "ipc_timeout_per_activity_entry_seconds",
                defaults.ipc_timeout_per_activity_entry_seconds,
            )
        )
        per_episode = float(
            getattr(consolidation_cfg, "ipc_timeout_per_episode_seconds", defaults.ipc_timeout_per_episode_seconds)
        )
        max_seconds = float(getattr(consolidation_cfg, "ipc_timeout_max_seconds", defaults.ipc_timeout_max_seconds))
        if gate is None:
            return min(max_seconds, base)
        summary_input_entries = getattr(gate, "summary_input_entries", None)
        activity_entries = (
            getattr(gate, "activity_count", 0) if summary_input_entries is None else summary_input_entries
        )
        estimate = (
            base + float(activity_entries) * per_activity + float(getattr(gate, "episode_count", 0)) * per_episode
        )
        return min(max_seconds, max(base, estimate))

    async def _cancel_consolidation(
        self: _SchedulerMixinHost,
        handle: object,
        anima_name: str,
    ) -> None:
        """Best-effort cancel a still-running anima consolidation after a root timeout."""
        try:
            await handle.send_request("cancel_consolidation", {}, timeout=10.0)
            logger.debug("Cancel request sent for anima=%s consolidation", anima_name)
        except TimeoutError:
            logger.debug("Cancel consolidation timed out for %s", anima_name)
        except Exception:
            logger.debug("Cancel consolidation request failed for %s", anima_name, exc_info=True)

    async def _run_project_archive_consolidations(
        self: _SchedulerMixinHost,
        handle: object,
        anima_name: str,
        anima_dir: Path,
        *,
        consolidation_type: str,
        timeout_s: float,
    ) -> None:
        """Run consolidation once for each non-empty project archive."""
        from core.memory.maintenance.consolidation import list_project_archives

        projects: list[str] = []
        for project in list_project_archives(anima_dir):
            episodes_dir = anima_dir / "episodes" / "projects" / project
            if not any(episodes_dir.glob("*.md")):
                logger.info(
                    "Project consolidation skipped for %s/%s: no episodes",
                    anima_name,
                    project,
                )
                continue
            projects.append(project)
        if not projects:
            return

        consolidating: set[str] = getattr(self, "_consolidating", set())
        consolidating.add(anima_name)
        timed_out = False
        try:
            for project in projects:
                try:
                    response = await handle.send_request(
                        "run_consolidation",
                        {
                            "consolidation_type": consolidation_type,
                            "project": project,
                            "deadline_s": max(60.0, timeout_s - 60),
                        },
                        timeout=timeout_s,
                    )
                except TimeoutError:
                    timed_out = True
                    logger.warning(
                        "consolidation_timeout anima=%s project=%s phase=project type=%s timeout_s=%.0f",
                        anima_name,
                        project,
                        consolidation_type,
                        timeout_s,
                    )
                    await self._cancel_consolidation(handle, anima_name)
                    try:
                        await handle.send_request("interrupt", {}, timeout=10.0)
                    except Exception:
                        logger.debug("Interrupt request after project consolidation timeout failed", exc_info=True)
                    continue
                except Exception:
                    logger.exception("Project consolidation failed for %s/%s", anima_name, project)
                    continue

                if response.error:
                    logger.error(
                        "Project consolidation IPC error for %s/%s: %s",
                        anima_name,
                        project,
                        response.error,
                    )
                else:
                    logger.info("Project consolidation completed for %s/%s", anima_name, project)
        finally:
            if timed_out:
                asyncio.get_running_loop().call_later(120, consolidating.discard, anima_name)
            else:
                consolidating.discard(anima_name)

    async def _run_daily_consolidation(self: _SchedulerMixinHost) -> None:
        """Run daily consolidation for all animas via IPC with bounded concurrency."""
        try:
            from core.config import load_config

            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            logger.debug("Config load failed for daily consolidation", exc_info=True)
            consolidation_cfg = None

        from core.config.models import ConsolidationConfig
        from core.lifecycle.system_consolidation import (
            evaluate_daily_consolidation_gate,
            run_daily_consolidation_post_processing,
            should_skip_inactive_consolidation,
        )
        from core.memory.maintenance.activity_compaction import ActivityCompactionSettings

        defaults = ConsolidationConfig()
        min_entries = defaults.min_episodes_threshold
        backfill_days = defaults.episode_summary_backfill_days
        max_backfill_days = defaults.episode_summary_backfill_max_days_per_run
        max_input_bytes = defaults.episode_summary_max_input_bytes
        if consolidation_cfg:
            min_entries = getattr(consolidation_cfg, "min_episodes_threshold", min_entries)
            backfill_days = getattr(consolidation_cfg, "episode_summary_backfill_days", backfill_days)
            max_backfill_days = getattr(
                consolidation_cfg,
                "episode_summary_backfill_max_days_per_run",
                max_backfill_days,
            )
            max_input_bytes = getattr(consolidation_cfg, "episode_summary_max_input_bytes", max_input_bytes)

        targets = self._iter_consolidation_targets()
        concurrency = _resolve_consolidation_concurrency(consolidation_cfg, defaults.max_concurrent_animas)
        # Reuse one set for all workers; a getattr fallback inside each worker
        # would otherwise create a different set when the host lacks the field.
        consolidating: set[str] = getattr(self, "_consolidating", set())
        started_at = asyncio.get_running_loop().time()
        logger.info(
            "Starting system-wide daily consolidation targets=%d concurrency=%d",
            len(targets),
            concurrency,
        )

        async def run_one(anima_name: str, anima_dir: Path) -> _ConsolidationOutcome:
            inactive = should_skip_inactive_consolidation(anima_dir, anima_name, consolidation_cfg)
            handle = self.processes.get(anima_name)
            if inactive:
                if handle and handle.state == ProcessState.RUNNING:
                    await self._run_project_archive_consolidations(
                        handle,
                        anima_name,
                        anima_dir,
                        consolidation_type="daily",
                        timeout_s=self._resolve_consolidation_ipc_timeout(
                            consolidation_cfg,
                            consolidation_type="daily",
                        ),
                    )
                return "skipped"
            if not handle or handle.state != ProcessState.RUNNING:
                logger.info(
                    "Daily consolidation skipped for %s: process not running",
                    anima_name,
                )
                return "skipped"

            from core.config.helper_models import resolve_helper_model

            episode_summary_model = resolve_helper_model("episode_summary", anima_dir).model
            gate = evaluate_daily_consolidation_gate(
                anima_dir,
                anima_name,
                threshold=min_entries,
                hours=24,
                backfill_days=backfill_days,
                max_backfill_days=max_backfill_days,
                model=episode_summary_model,
                max_input_bytes=max_input_bytes,
                compaction_settings=ActivityCompactionSettings.from_config(consolidation_cfg, defaults),
                exclude_noop_cron=getattr(
                    consolidation_cfg,
                    "episode_summary_exclude_noop_cron",
                    defaults.episode_summary_exclude_noop_cron,
                ),
            )
            if not gate.should_run:
                logger.info(
                    "Daily consolidation skipped for %s: activity=%d episodes=%d threshold=%d backfill_days=%d",
                    anima_name,
                    gate.activity_count,
                    gate.episode_count,
                    gate.threshold,
                    gate.pending_backfill_days,
                )
                await self._run_project_archive_consolidations(
                    handle,
                    anima_name,
                    anima_dir,
                    consolidation_type="daily",
                    timeout_s=self._resolve_consolidation_ipc_timeout(
                        consolidation_cfg,
                        consolidation_type="daily",
                    ),
                )
                return "skipped"

            timeout_s = self._resolve_consolidation_ipc_timeout(
                consolidation_cfg,
                consolidation_type="daily",
                gate=gate,
            )
            result: dict = {}
            timed_out = False
            failed = False
            try:
                consolidating.add(anima_name)
                try:
                    response = await handle.send_request(
                        "run_consolidation",
                        {"consolidation_type": "daily", "deadline_s": max(60.0, timeout_s - 60)},
                        timeout=timeout_s,
                    )
                except TimeoutError:
                    timed_out = True
                    logger.warning(
                        "consolidation_timeout anima=%s phase=phase_a type=daily timeout_s=%.0f",
                        anima_name,
                        timeout_s,
                    )
                    await self._cancel_consolidation(handle, anima_name)
                    try:
                        await handle.send_request("interrupt", {}, timeout=10.0)
                    except Exception:
                        logger.debug("Interrupt request after daily consolidation timeout failed", exc_info=True)
                finally:
                    if timed_out:
                        # Keep protection for 120 seconds after a timed-out IPC.
                        asyncio.get_running_loop().call_later(120, consolidating.discard, anima_name)
                    else:
                        consolidating.discard(anima_name)

                if not timed_out and response.error:
                    logger.error(
                        "Daily consolidation IPC error for %s: %s",
                        anima_name,
                        response.error,
                    )
                    failed = True
                elif not timed_out:
                    result = response.result or {}
                    logger.info(
                        "Daily consolidation for %s: duration_ms=%d",
                        anima_name,
                        result.get("duration_ms", 0),
                    )
            except Exception:
                logger.exception("Daily consolidation failed for %s", anima_name)
                failed = True
            finally:
                if result.get("action") != "skipped":
                    try:
                        await run_daily_consolidation_post_processing(
                            anima_name,
                            anima_dir,
                            consolidation_cfg=consolidation_cfg,
                            model="",
                        )
                    except Exception:
                        logger.exception("Daily consolidation post-processing failed for %s", anima_name)
                        failed = True

                try:
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
                    logger.exception("Daily consolidation event broadcast failed for %s", anima_name)
                    failed = True

            try:
                await self._run_project_archive_consolidations(
                    handle,
                    anima_name,
                    anima_dir,
                    consolidation_type="daily",
                    timeout_s=timeout_s,
                )
            except Exception:
                logger.exception("Daily project archive consolidation failed for %s", anima_name)
                failed = True

            if failed:
                return "failed"
            if timed_out:
                return "timed_out"
            if result.get("action") == "skipped":
                return "skipped"
            return "ran"

        outcomes = await _run_bounded(
            targets,
            run_one,
            limit=concurrency,
            label="Daily consolidation",
        )
        _write_marker(_marker_dir(self._get_data_dir()) / "last_daily_consolidation")
        logger.info(
            "System-wide daily consolidation finished targets=%d ran=%d skipped=%d timed_out=%d failed=%d elapsed_s=%.0f",
            len(targets),
            outcomes.count("ran"),
            outcomes.count("skipped"),
            outcomes.count("timed_out"),
            outcomes.count("failed"),
            asyncio.get_running_loop().time() - started_at,
        )

    async def _run_weekly_integration(self: _SchedulerMixinHost) -> None:
        """Run weekly integration for all animas via IPC with bounded concurrency."""
        try:
            from core.config import load_config

            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            logger.debug("Config load failed for weekly integration", exc_info=True)
            consolidation_cfg = None

        from core.config.models import ConsolidationConfig
        from core.lifecycle.system_consolidation import (
            run_weekly_integration_post_processing,
            should_skip_inactive_consolidation,
        )

        defaults = ConsolidationConfig()

        targets = self._iter_consolidation_targets()
        concurrency = _resolve_consolidation_concurrency(consolidation_cfg, defaults.max_concurrent_animas)
        # See the daily worker: every concurrent mutation must address this same set.
        consolidating: set[str] = getattr(self, "_consolidating", set())
        started_at = asyncio.get_running_loop().time()
        logger.info(
            "Starting system-wide weekly integration targets=%d concurrency=%d",
            len(targets),
            concurrency,
        )

        async def run_one(anima_name: str, anima_dir: Path) -> _ConsolidationOutcome:
            inactive = should_skip_inactive_consolidation(anima_dir, anima_name, consolidation_cfg)
            handle = self.processes.get(anima_name)
            if inactive:
                if handle and handle.state == ProcessState.RUNNING:
                    await self._run_project_archive_consolidations(
                        handle,
                        anima_name,
                        anima_dir,
                        consolidation_type="weekly",
                        timeout_s=self._resolve_consolidation_ipc_timeout(
                            consolidation_cfg,
                            consolidation_type="weekly",
                        ),
                    )
                return "skipped"
            if not handle or handle.state != ProcessState.RUNNING:
                logger.info(
                    "Weekly integration skipped for %s: process not running",
                    anima_name,
                )
                return "skipped"

            timeout_s = self._resolve_consolidation_ipc_timeout(
                consolidation_cfg,
                consolidation_type="weekly",
            )
            result: dict = {}
            timed_out = False
            failed = False
            try:
                consolidating.add(anima_name)
                try:
                    response = await handle.send_request(
                        "run_consolidation",
                        {"consolidation_type": "weekly", "deadline_s": max(60.0, timeout_s - 60)},
                        timeout=timeout_s,
                    )
                except TimeoutError:
                    timed_out = True
                    logger.warning(
                        "consolidation_timeout anima=%s phase=phase_b type=weekly timeout_s=%.0f",
                        anima_name,
                        timeout_s,
                    )
                    await self._cancel_consolidation(handle, anima_name)
                    try:
                        await handle.send_request("interrupt", {}, timeout=10.0)
                    except Exception:
                        logger.debug("Interrupt request after weekly consolidation timeout failed", exc_info=True)
                finally:
                    if timed_out:
                        asyncio.get_running_loop().call_later(120, consolidating.discard, anima_name)
                    else:
                        consolidating.discard(anima_name)

                if not timed_out and response.error:
                    logger.error(
                        "Weekly integration IPC error for %s: %s",
                        anima_name,
                        response.error,
                    )
                    failed = True
                elif not timed_out:
                    result = response.result or {}
                    logger.info(
                        "Weekly integration for %s: duration_ms=%d",
                        anima_name,
                        result.get("duration_ms", 0),
                    )
            except Exception:
                logger.exception("Weekly integration failed for %s", anima_name)
                failed = True
            finally:
                try:
                    await run_weekly_integration_post_processing(
                        anima_name,
                        anima_dir,
                        consolidation_cfg=consolidation_cfg,
                        model="",
                    )
                except Exception:
                    logger.exception("Weekly integration post-processing failed for %s", anima_name)
                    failed = True

                try:
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
                    logger.exception("Weekly integration event broadcast failed for %s", anima_name)
                    failed = True

            try:
                await self._run_project_archive_consolidations(
                    handle,
                    anima_name,
                    anima_dir,
                    consolidation_type="weekly",
                    timeout_s=timeout_s,
                )
            except Exception:
                logger.exception("Weekly project archive consolidation failed for %s", anima_name)
                failed = True

            if failed:
                return "failed"
            if timed_out:
                return "timed_out"
            if result.get("action") == "skipped":
                return "skipped"
            return "ran"

        outcomes = await _run_bounded(
            targets,
            run_one,
            limit=concurrency,
            label="Weekly integration",
        )
        _write_marker(_marker_dir(self._get_data_dir()) / "last_weekly_integration")
        logger.info(
            "System-wide weekly integration finished targets=%d ran=%d skipped=%d timed_out=%d failed=%d elapsed_s=%.0f",
            len(targets),
            outcomes.count("ran"),
            outcomes.count("skipped"),
            outcomes.count("timed_out"),
            outcomes.count("failed"),
            asyncio.get_running_loop().time() - started_at,
        )

    async def _run_daily_indexing(self: _SchedulerMixinHost) -> None:
        """Run daily RAG indexing for all animas.

        Incrementally indexes all memory files (knowledge, episodes,
        procedures, skills, facts) into each anima's per-anima vectordb.
        Also indexes shared collections (common_knowledge, common_skills).
        Runs at 04:00 (configured TZ) as the sole scheduled RAG index update,
        capturing files generated or modified by earlier consolidation jobs.
        """
        logger.info("Starting system-wide daily RAG indexing")

        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.retrieval.bm25 import rebuild_longterm_bm25_index
        except ImportError:
            logger.warning("RAG dependencies not available, skipping daily indexing")
            return

        import gc
        import json

        from core.paths import (
            get_common_knowledge_dir,
            get_common_skills_dir,
        )

        base_dir = self._get_data_dir()

        from core.memory.rag.embedding import get_embedding_e5_prefix_enabled, get_embedding_model_name

        current_model = get_embedding_model_name()
        current_e5_prefix = get_embedding_e5_prefix_enabled()
        global_meta_path = base_dir / "index_meta.json"
        if global_meta_path.is_file():
            from core.i18n import t
            from core.memory.rag.index_signature import index_signature_error

            try:
                meta = json.loads(global_meta_path.read_text(encoding="utf-8"))
                signature_error = index_signature_error(meta, current_model, current_e5_prefix)
            except (json.JSONDecodeError, OSError):
                signature_error = t("rag.signature_unreadable")
            if signature_error:
                logger.warning(t("rag.daily_indexing_blocked", reason=signature_error))
                return

        from core.memory.rag.shared_meta import read_shared_hash, write_shared_hash
        from core.memory.retrieval.rag_search import _compute_dir_hash

        loop = asyncio.get_running_loop()
        total_chunks = 0
        ck_dir = get_common_knowledge_dir()
        cs_dir = get_common_skills_dir()

        shared_sources: list[tuple[str, Path, str, str]] = []
        if ck_dir.is_dir():
            shared_sources.append(("common_knowledge", ck_dir, "*.md", "shared_common_knowledge_hash"))
        if cs_dir.is_dir():
            shared_sources.append(("common_skills", cs_dir, "SKILL.md", "shared_common_skills_hash"))

        for anima_name, anima_dir in self._iter_consolidation_targets():
            try:
                from core.memory.rag.repair import is_repair_locked

                if is_repair_locked(anima_name):
                    logger.warning("Skipping daily RAG indexing for %s: RAG repair lock is held", anima_name)
                    continue

                from core.memory.rag.vector_registry import get_vector_store

                vector_store = get_vector_store(anima_name)
                if vector_store is None:
                    logger.warning("Vector store unavailable for %s, skipping indexing", anima_name)
                    continue
                indexer = MemoryIndexer(vector_store, anima_name, anima_dir)

                memory_types = [
                    ("knowledge", anima_dir / "knowledge"),
                    ("episodes", anima_dir / "episodes"),
                    ("procedures", anima_dir / "procedures"),
                    ("skills", anima_dir / "skills"),
                    ("facts", anima_dir / "facts"),
                ]

                for memory_type, memory_dir in memory_types:
                    if not memory_dir.is_dir():
                        continue
                    result = await loop.run_in_executor(
                        None,
                        indexer.index_directory,
                        memory_dir,
                        memory_type,
                    )
                    total_chunks += result.chunks_indexed

                conv_file = anima_dir / "state" / "conversation.json"
                if conv_file.is_file():
                    chunks = await loop.run_in_executor(
                        None,
                        indexer.index_conversation_summary,
                        anima_dir / "state",
                        anima_name,
                    )
                    total_chunks += chunks

                bm25_result = await loop.run_in_executor(
                    None,
                    rebuild_longterm_bm25_index,
                    anima_dir,
                )
                logger.info(
                    "Daily long-term BM25 indexing for %s complete: documents=%d",
                    anima_name,
                    bm25_result.documents,
                )

                try:
                    from core.memory.facts.entity_index import rebuild_entity_collection

                    ok = await loop.run_in_executor(
                        None,
                        functools.partial(rebuild_entity_collection, anima_dir, vector_store=vector_store),
                    )
                    if not ok:
                        logger.warning("Entity collection rebuild failed during daily indexing for %s", anima_name)
                except Exception:
                    logger.warning(
                        "Entity collection rebuild failed during daily indexing for %s",
                        anima_name,
                        exc_info=True,
                    )

                for label, src_dir, glob, meta_key in shared_sources:
                    if not src_dir.is_dir():
                        continue
                    current_hash = _compute_dir_hash(src_dir, glob)
                    stored_hash = read_shared_hash(anima_dir, meta_key)
                    shared_collection = f"shared_{label}"
                    force = False
                    if current_hash == stored_hash:
                        existence = await loop.run_in_executor(
                            None,
                            vector_store.collection_exists,
                            shared_collection,
                        )
                        if existence is not CollectionExistence.MISSING:
                            continue
                        logger.info(
                            "%s: collection '%s' missing despite tracked hash, forcing re-index",
                            anima_name,
                            shared_collection,
                        )
                        force = True
                    shared_indexer = MemoryIndexer(
                        vector_store,
                        anima_name="shared",
                        anima_dir=base_dir,
                        collection_prefix="shared",
                    )
                    result = await loop.run_in_executor(
                        None,
                        shared_indexer.index_directory,
                        src_dir,
                        label,
                        force,
                    )
                    total_chunks += result.chunks_indexed
                    if result.files_failed == 0:
                        write_shared_hash(anima_dir, meta_key, current_hash)

                logger.info("Daily indexing for %s complete", anima_name)

                del indexer
                gc.collect()

            except Exception:
                logger.exception("Daily indexing failed for anima=%s", anima_name)

        logger.info(
            "System-wide daily RAG indexing complete: %d chunks indexed",
            total_chunks,
        )

        try:
            await self._broadcast_event(
                "system.rag_indexing",
                {"total_chunks": total_chunks},
            )
        except Exception:
            logger.debug("Failed to broadcast rag_indexing event", exc_info=True)

        _write_marker(_marker_dir(self._get_data_dir()) / "last_daily_indexing")

    async def _run_activity_log_rotation(self: _SchedulerMixinHost) -> None:
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
        mode = (
            getattr(activity_cfg, "rotation_mode", defaults.rotation_mode) if activity_cfg else defaults.rotation_mode
        )
        max_size_mb = (
            getattr(activity_cfg, "max_size_mb", defaults.max_size_mb) if activity_cfg else defaults.max_size_mb
        )
        max_age_days = (
            getattr(activity_cfg, "max_age_days", defaults.max_age_days) if activity_cfg else defaults.max_age_days
        )
        max_file_size_mb = (
            getattr(activity_cfg, "max_file_size_mb", defaults.max_file_size_mb)
            if activity_cfg
            else defaults.max_file_size_mb
        )

        try:
            from core.activity.logger import ActivityLogger

            results = ActivityLogger.rotate_all(
                self.animas_dir,
                mode=mode,
                max_size_mb=max_size_mb,
                max_age_days=max_age_days,
                max_file_size_mb=max_file_size_mb,
            )
            if results:
                total_freed = sum(r.get("freed_bytes", 0) for r in results.values())
                total_deleted = sum(r.get("deleted_files", 0) for r in results.values())
                total_bloated = sum(r.get("bloated_rotated_files", 0) for r in results.values())
                logger.info(
                    "Activity log rotation complete: %d animas, %d files deleted, "
                    "%d bloated files rotated, %d bytes freed",
                    len(results),
                    total_deleted,
                    total_bloated,
                    total_freed,
                )
            else:
                logger.info("Activity log rotation: no files needed rotation")
        except Exception:
            logger.exception("Activity log rotation failed")

    def _get_housekeeping_lock(self: _SchedulerMixinHost) -> asyncio.Lock:
        """Return the lazily-created lock guarding housekeeping runs.

        Created on first use so the mixin needs no ``__init__``. Safe under
        asyncio's single-threaded model — there is no await between the
        ``getattr`` check and the assignment.
        """
        lock = getattr(self, "_housekeeping_lock_obj", None)
        if lock is None:
            lock = asyncio.Lock()
            self._housekeeping_lock_obj = lock
        return lock

    async def _run_housekeeping(self: _SchedulerMixinHost) -> None:
        """Run unified housekeeping, guarded against concurrent invocation.

        Both the cron schedule (:222) and the startup catch-up (:1000)
        invoke this; the lock makes the second caller skip rather than run
        cleanup a second time in parallel.
        """
        lock = self._get_housekeeping_lock()
        if lock.locked():
            logger.info("Housekeeping already running; skipping duplicate invocation")
            return
        async with lock:
            await self._run_housekeeping_impl()

    async def _run_workspace_sweep(self: _SchedulerMixinHost) -> None:
        """Run the three frequent workspace cleanup rules under the shared lock."""
        lock = self._get_housekeeping_lock()
        if lock.locked():
            logger.info("Housekeeping already running; skipping workspace sweep")
            return
        async with lock:
            await self._run_housekeeping_impl(
                only={"codex_sessions", "anima_scratch", "shared_worktrees"},
                mark_complete=False,
            )

    async def _run_housekeeping_impl(
        self: _SchedulerMixinHost,
        *,
        only: set[str] | None = None,
        mark_complete: bool = True,
    ) -> None:
        """Run unified housekeeping, optionally restricted to named cleanup rules."""
        logger.info("Starting system-wide housekeeping" if only is None else "Starting workspace sweep")

        try:
            from core.config import load_config
            from core.config.models import HousekeepingConfig, InboxConfig

            cfg = load_config()
            hk_cfg = getattr(cfg, "housekeeping", None)
            if not isinstance(hk_cfg, HousekeepingConfig):
                hk_cfg = HousekeepingConfig()
            inbox_cfg = getattr(cfg, "inbox", None)
            if not isinstance(inbox_cfg, InboxConfig):
                inbox_cfg = InboxConfig()
        except Exception:
            logger.debug("Config load failed for housekeeping", exc_info=True)
            from core.config.models import HousekeepingConfig, InboxConfig

            hk_cfg = HousekeepingConfig()
            inbox_cfg = InboxConfig()

        try:
            from core.memory.maintenance.housekeeping import run_housekeeping

            results = await run_housekeeping(
                self._get_data_dir(),
                housekeeping=hk_cfg,
                inbox=inbox_cfg,
                only=only,
            )
            logger.info("Housekeeping complete: %s", results)
        except Exception:
            logger.exception("Housekeeping failed")

        if mark_complete:
            _write_marker(_marker_dir(self._get_data_dir()) / "last_housekeeping")

    async def _run_dm_log_rotation(self: _SchedulerMixinHost) -> None:
        """Archive old dm_log entries beyond 7 days."""
        logger.info("Starting DM log rotation")
        try:
            from core.tasks.background import rotate_dm_logs

            shared_dir = self._get_data_dir() / "shared"
            result = await rotate_dm_logs(shared_dir)
            if result:
                logger.info("DM log rotation completed: %s", result)
            else:
                logger.debug("DM log rotation: nothing to archive")
        except Exception:
            logger.exception("DM log rotation failed")

    # ── Catch-up for missed scheduled jobs ──────────────────────────

    _CATCHUP_DELAY_SEC = 90

    async def _catchup_missed_jobs(self: _SchedulerMixinHost) -> None:
        """Run after scheduler start to execute any jobs missed while offline.

        Uses marker files in ``~/.animaworks/run/`` to track the last
        successful execution of each scheduled job.  If the expected interval
        has elapsed since the last marker, the job is scheduled for immediate
        (delayed by ``_CATCHUP_DELAY_SEC`` to let all Anima processes boot).
        """
        await asyncio.sleep(self._CATCHUP_DELAY_SEC)

        try:
            from core.config import load_config
            from core.config.models import ConsolidationConfig

            consolidation_cfg = load_config().consolidation or ConsolidationConfig()
        except Exception:
            from core.config.models import ConsolidationConfig

            consolidation_cfg = ConsolidationConfig()

        now = now_local()
        mdir = _marker_dir(self._get_data_dir())

        daily_enabled = consolidation_cfg.daily_enabled
        weekly_enabled = consolidation_cfg.weekly_enabled

        if daily_enabled:
            last = _read_marker(mdir / "last_daily_consolidation")
            if last is None or (now - last) > timedelta(hours=36):
                logger.info(
                    "Catch-up: daily consolidation missed (last=%s), running now",
                    last,
                )
                await self._run_daily_consolidation()

        if weekly_enabled:
            last = _read_marker(mdir / "last_weekly_integration")
            if last is None or (now - last) > timedelta(days=9):
                logger.info(
                    "Catch-up: weekly integration missed (last=%s), running now",
                    last,
                )
                await self._run_weekly_integration()

        indexing_enabled = consolidation_cfg.indexing_enabled
        if indexing_enabled:
            last = _read_marker(mdir / "last_daily_indexing")
            if last is None or (now - last) > timedelta(hours=36):
                logger.info(
                    "Catch-up: daily indexing missed (last=%s), running now",
                    last,
                )
                await self._run_daily_indexing()

        # Housekeeping catch-up
        last = _read_marker(mdir / "last_housekeeping")
        if last is None or (now - last) > timedelta(hours=36):
            logger.info(
                "Catch-up: housekeeping missed (last=%s), running now",
                last,
            )
            await self._run_housekeeping()
