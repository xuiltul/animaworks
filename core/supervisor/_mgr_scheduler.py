"""
System scheduler mixin for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.supervisor.process_handle import ProcessState

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))

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
    ts = ts or datetime.now(_JST)
    marker_path.write_text(ts.isoformat())


class SchedulerMixin:
    """System-level cron scheduler for memory consolidation and log rotation."""

    def _start_system_scheduler(self) -> None:
        """Start the system-level scheduler for consolidation crons."""
        try:
            self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
            self._setup_system_crons()
            self.scheduler.start()
            self._scheduler_running = True
            logger.info("System scheduler started")
            asyncio.ensure_future(self._catchup_missed_jobs())
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

    def _get_data_dir(self) -> Path:
        """Return the runtime data directory (``~/.animaworks`` or override)."""
        return self.animas_dir.parent

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

        _write_marker(_marker_dir(self._get_data_dir()) / "last_daily_consolidation")

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
                    reorg_result = await forgetter.neurogenesis_reorganize()
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

        _write_marker(_marker_dir(self._get_data_dir()) / "last_weekly_integration")

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

        _write_marker(_marker_dir(self._get_data_dir()) / "last_monthly_forgetting")

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

    # ── Catch-up for missed scheduled jobs ──────────────────────────

    _CATCHUP_DELAY_SEC = 90

    async def _catchup_missed_jobs(self) -> None:
        """Run after scheduler start to execute any jobs missed while offline.

        Uses marker files in ``~/.animaworks/run/`` to track the last
        successful execution of each scheduled job.  If the expected interval
        has elapsed since the last marker, the job is scheduled for immediate
        (delayed by ``_CATCHUP_DELAY_SEC`` to let all Anima processes boot).
        """
        await asyncio.sleep(self._CATCHUP_DELAY_SEC)

        try:
            from core.config import load_config
            consolidation_cfg = getattr(load_config(), "consolidation", None)
        except Exception:
            consolidation_cfg = None

        now = datetime.now(_JST)
        mdir = _marker_dir(self._get_data_dir())

        daily_enabled = getattr(consolidation_cfg, "daily_enabled", True) if consolidation_cfg else True
        weekly_enabled = getattr(consolidation_cfg, "weekly_enabled", True) if consolidation_cfg else True
        monthly_enabled = getattr(consolidation_cfg, "monthly_enabled", True) if consolidation_cfg else True

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

        if monthly_enabled:
            last = _read_marker(mdir / "last_monthly_forgetting")
            if last is None or (now - last) > timedelta(days=35):
                logger.info(
                    "Catch-up: monthly forgetting missed (last=%s), running now",
                    last,
                )
                await self._run_monthly_forgetting()
