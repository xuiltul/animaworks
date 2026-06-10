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
from datetime import datetime, timedelta
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.supervisor.process_handle import ProcessState
from core.time_utils import get_app_timezone, now_local

logger = logging.getLogger(__name__)

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

    def _start_system_scheduler(self) -> None:
        """Start the system-level scheduler for consolidation crons."""
        try:
            self.scheduler = AsyncIOScheduler(timezone=get_app_timezone())
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
                misfire_grace_time=600,
            )
            logger.info("System cron: Daily consolidation at %s", daily_time)

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
                misfire_grace_time=600,
            )
            logger.info("System cron: Weekly integration on %s at %s:%s", day_of_week, time_parts[0], time_parts[1])

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
                misfire_grace_time=600,
            )
            logger.info(
                "System cron: Monthly forgetting on day %d at %02d:%02d",
                day_of_month,
                hour,
                minute,
            )

        indexing_enabled = True
        indexing_time = "04:00"
        if consolidation_cfg:
            _ie = getattr(consolidation_cfg, "indexing_enabled", None)
            if isinstance(_ie, bool):
                indexing_enabled = _ie
            _it = getattr(consolidation_cfg, "indexing_time", None)
            if isinstance(_it, str) and ":" in _it:
                indexing_time = _it

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
            logger.debug("Housekeeping schedule setup failed", exc_info=True)

        # DM log rotation (mirrors LifecycleManager registration)
        self.scheduler.add_job(
            self._run_dm_log_rotation,
            CronTrigger(hour=4, minute=30),
            id="system_dm_log_rotation",
            name="System: DM Log Rotation",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("System cron: DM log rotation at 04:30")

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

        Sends ``run_consolidation`` IPC requests to running Anima processes.
        Framework-side post-processing is delegated to the shared lifecycle
        consolidation pipeline and runs even when the IPC phase times out.
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
        from core.lifecycle.system_consolidation import (
            evaluate_daily_consolidation_gate,
            run_daily_consolidation_post_processing,
        )

        defaults = ConsolidationConfig()
        max_turns = ConsolidationConfig().max_turns
        min_entries = defaults.min_episodes_threshold
        model = defaults.llm_model
        if consolidation_cfg:
            max_turns = getattr(consolidation_cfg, "max_turns", max_turns)
            min_entries = getattr(consolidation_cfg, "min_episodes_threshold", min_entries)
            model = getattr(consolidation_cfg, "llm_model", model)

        for anima_name, anima_dir in self._iter_consolidation_targets():
            handle = self.processes.get(anima_name)
            if not handle or handle.state != ProcessState.RUNNING:
                logger.info(
                    "Daily consolidation skipped for %s: process not running",
                    anima_name,
                )
                continue

            gate = evaluate_daily_consolidation_gate(
                anima_dir,
                anima_name,
                threshold=min_entries,
                hours=24,
            )
            if not gate.should_run:
                logger.info(
                    "Daily consolidation skipped for %s: activity=%d episodes=%d carryover=%d threshold=%d",
                    anima_name,
                    gate.activity_count,
                    gate.episode_count,
                    gate.carryover_count,
                    gate.threshold,
                )
                continue

            result: dict = {}
            try:
                _consolidating: set[str] = getattr(self, "_consolidating", set())
                _consolidating.add(anima_name)
                _timed_out = False
                try:
                    response = await handle.send_request(
                        "run_consolidation",
                        {"consolidation_type": "daily", "max_turns": max_turns},
                        timeout=1800.0,
                    )
                except TimeoutError:
                    _timed_out = True
                    logger.warning(
                        "consolidation_timeout anima=%s phase=phase_b type=daily timeout_s=1800",
                        anima_name,
                    )
                    try:
                        await handle.send_request("interrupt", {}, timeout=10.0)
                    except Exception:
                        logger.debug("Interrupt request after daily consolidation timeout failed", exc_info=True)
                finally:
                    if _timed_out:
                        # Grace period: keep protection for 120s after timeout
                        _name_capture = anima_name
                        asyncio.get_running_loop().call_later(120, self._consolidating.discard, _name_capture)
                    else:
                        _consolidating.discard(anima_name)

                if not _timed_out and response.error:
                    logger.error(
                        "Daily consolidation IPC error for %s: %s",
                        anima_name,
                        response.error,
                    )
                elif not _timed_out:
                    result = response.result or {}
                    logger.info(
                        "Daily consolidation for %s: duration_ms=%d",
                        anima_name,
                        result.get("duration_ms", 0),
                    )
            except Exception:
                logger.exception("Daily consolidation failed for %s", anima_name)
            finally:
                await run_daily_consolidation_post_processing(
                    anima_name,
                    anima_dir,
                    consolidation_cfg=consolidation_cfg,
                    model=model,
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

        _write_marker(_marker_dir(self._get_data_dir()) / "last_daily_consolidation")

    async def _run_weekly_integration(self) -> None:
        """Run weekly integration for all animas via IPC.

        Sends ``run_consolidation`` IPC requests to running Anima processes.
        Framework-side post-processing is delegated to the shared lifecycle
        consolidation pipeline and runs even when the IPC phase times out.
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
        from core.lifecycle.system_consolidation import run_weekly_integration_post_processing

        defaults = _CC()
        max_turns = defaults.max_turns
        model = defaults.llm_model
        if consolidation_cfg:
            max_turns = getattr(consolidation_cfg, "max_turns", max_turns)
            model = getattr(consolidation_cfg, "llm_model", model)

        for anima_name, anima_dir in self._iter_consolidation_targets():
            handle = self.processes.get(anima_name)
            if not handle or handle.state != ProcessState.RUNNING:
                logger.info(
                    "Weekly integration skipped for %s: process not running",
                    anima_name,
                )
                continue

            result: dict = {}
            try:
                _consolidating_w: set[str] = getattr(self, "_consolidating", set())
                _consolidating_w.add(anima_name)
                _timed_out_w = False
                try:
                    response = await handle.send_request(
                        "run_consolidation",
                        {"consolidation_type": "weekly", "max_turns": max_turns},
                        timeout=1800.0,
                    )
                except TimeoutError:
                    _timed_out_w = True
                    logger.warning(
                        "consolidation_timeout anima=%s phase=phase_b type=weekly timeout_s=1800",
                        anima_name,
                    )
                    try:
                        await handle.send_request("interrupt", {}, timeout=10.0)
                    except Exception:
                        logger.debug("Interrupt request after weekly consolidation timeout failed", exc_info=True)
                finally:
                    if _timed_out_w:
                        _name_capture_w = anima_name
                        asyncio.get_running_loop().call_later(120, self._consolidating.discard, _name_capture_w)
                    else:
                        _consolidating_w.discard(anima_name)

                if not _timed_out_w and response.error:
                    logger.error(
                        "Weekly integration IPC error for %s: %s",
                        anima_name,
                        response.error,
                    )
                elif not _timed_out_w:
                    result = response.result or {}
                    logger.info(
                        "Weekly integration for %s: duration_ms=%d",
                        anima_name,
                        result.get("duration_ms", 0),
                    )
            except Exception:
                logger.exception("Weekly integration failed for %s", anima_name)
            finally:
                await run_weekly_integration_post_processing(
                    anima_name,
                    anima_dir,
                    consolidation_cfg=consolidation_cfg,
                    model=model,
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

    async def _run_daily_indexing(self) -> None:
        """Run daily RAG indexing for all animas.

        Incrementally indexes all memory files (knowledge, episodes,
        procedures, skills) into each anima's per-anima vectordb.
        Also indexes shared collections (common_knowledge, common_skills).
        Runs at 04:00 (configured TZ), after consolidation (02:00) and
        weekly/monthly jobs (03:00) to capture all generated/modified files.
        """
        logger.info("Starting system-wide daily RAG indexing")

        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store
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

        from core.memory.rag.singleton import get_embedding_model_name

        current_model = get_embedding_model_name()
        global_meta_path = base_dir / "index_meta.json"
        if global_meta_path.is_file():
            try:
                meta = json.loads(global_meta_path.read_text(encoding="utf-8"))
                previous_model = meta.get("embedding_model")
                if previous_model and previous_model != current_model:
                    logger.warning(
                        "Embedding model changed: %s -> %s. "
                        "Skipping daily indexing — run 'animaworks index --full' to rebuild.",
                        previous_model,
                        current_model,
                    )
                    return
            except (json.JSONDecodeError, OSError):
                pass

        from core.memory.rag_search import _compute_dir_hash, _read_shared_hash, _write_shared_hash

        quick_check_timeout = 10.0
        try:
            from core.config import load_config

            quick_check_timeout = float(getattr(load_config().rag, "quick_check_timeout_seconds", 10.0))
        except Exception:
            logger.debug("Config load failed for RAG quick_check timeout", exc_info=True)

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

                from core.memory.rag.sqlite_health import check_anima_vectordb_health_via_worker_or_direct

                health = await loop.run_in_executor(
                    None,
                    functools.partial(
                        check_anima_vectordb_health_via_worker_or_direct,
                        anima_name,
                        timeout_seconds=quick_check_timeout,
                        source="daily_indexing_quick_check",
                    ),
                )
                if health.corrupt:
                    logger.warning(
                        "Skipping daily RAG indexing for %s: quick_check status=%s db=%s",
                        anima_name,
                        health.status,
                        health.db_path,
                    )
                    continue

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
                ]

                for memory_type, memory_dir in memory_types:
                    if not memory_dir.is_dir():
                        continue
                    chunks = await loop.run_in_executor(
                        None,
                        indexer.index_directory,
                        memory_dir,
                        memory_type,
                    )
                    total_chunks += chunks

                conv_file = anima_dir / "state" / "conversation.json"
                if conv_file.is_file():
                    chunks = await loop.run_in_executor(
                        None,
                        indexer.index_conversation_summary,
                        anima_dir / "state",
                        anima_name,
                    )
                    total_chunks += chunks

                meta_path = anima_dir / "index_meta.json"
                for label, src_dir, glob, meta_key in shared_sources:
                    if not src_dir.is_dir():
                        continue
                    current_hash = _compute_dir_hash(src_dir, glob)
                    stored_hash = _read_shared_hash(meta_path, meta_key)
                    shared_collection = f"shared_{label}"
                    force = False
                    if current_hash == stored_hash:
                        try:
                            existing = vector_store.list_collections()
                        except Exception:
                            existing = None
                        if existing is None or shared_collection in existing:
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
                    chunks = await loop.run_in_executor(
                        None,
                        shared_indexer.index_directory,
                        src_dir,
                        label,
                        force,
                    )
                    total_chunks += chunks
                    _write_shared_hash(meta_path, meta_key, current_hash)

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
        mode = (
            getattr(activity_cfg, "rotation_mode", defaults.rotation_mode) if activity_cfg else defaults.rotation_mode
        )
        max_size_mb = (
            getattr(activity_cfg, "max_size_mb", defaults.max_size_mb) if activity_cfg else defaults.max_size_mb
        )
        max_age_days = (
            getattr(activity_cfg, "max_age_days", defaults.max_age_days) if activity_cfg else defaults.max_age_days
        )

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
                    len(results),
                    total_deleted,
                    total_freed,
                )
            else:
                logger.info("Activity log rotation: no files needed rotation")
        except Exception:
            logger.exception("Activity log rotation failed")

    async def _run_housekeeping(self) -> None:
        """Run unified housekeeping for all data types."""
        logger.info("Starting system-wide housekeeping")

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
            from core.memory.housekeeping import run_housekeeping

            results = await run_housekeeping(
                self._get_data_dir(),
                prompt_log_retention_days=hk_cfg.prompt_log_retention_days,
                daemon_log_max_size_mb=hk_cfg.daemon_log_max_size_mb,
                daemon_log_keep_generations=hk_cfg.daemon_log_keep_generations,
                dm_log_archive_retention_days=hk_cfg.dm_log_archive_retention_days,
                cron_log_retention_days=hk_cfg.cron_log_retention_days,
                shortterm_retention_days=hk_cfg.shortterm_retention_days,
                task_results_retention_days=hk_cfg.task_results_retention_days,
                pending_failed_retention_days=hk_cfg.pending_failed_retention_days,
                pending_processing_stale_hours=hk_cfg.pending_processing_stale_hours,
                background_running_stale_hours=hk_cfg.background_running_stale_hours,
                current_state_stale_hours=hk_cfg.current_state_stale_hours,
                taskboard_suppressed_retention_days=hk_cfg.taskboard_suppressed_retention_days,
                inbox_ttl_hours=inbox_cfg.ttl_hours,
                inbox_expired_retention_days=inbox_cfg.expired_retention_days,
                inbox_processed_retention_days=inbox_cfg.processed_retention_days,
                inbox_quarantine_retention_days=inbox_cfg.quarantine_retention_days,
            )
            logger.info("Housekeeping complete: %s", results)
        except Exception:
            logger.exception("Housekeeping failed")

        _write_marker(_marker_dir(self._get_data_dir()) / "last_housekeeping")

    async def _run_dm_log_rotation(self) -> None:
        """Archive old dm_log entries beyond 7 days."""
        logger.info("Starting DM log rotation")
        try:
            from core.background import rotate_dm_logs

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

        now = now_local()
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

        indexing_enabled = getattr(consolidation_cfg, "indexing_enabled", True) if consolidation_cfg else True
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
