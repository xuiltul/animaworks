from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import asyncio
import gc
import json
import logging
from datetime import timedelta
from pathlib import Path

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from core.time_utils import now_local

logger = logging.getLogger("animaworks.lifecycle")


class SystemCronsMixin:
    """Mixin providing system-wide cron tasks (consolidation, indexing, forgetting, etc.)."""

    def _setup_system_crons(self) -> None:
        """Set up system-wide cron tasks for memory consolidation."""
        # Daily RAG indexing: Every day at 04:00
        # Runs after consolidation (02:00) and weekly/monthly jobs (03:00)
        # to catch all generated/modified files as a final sweep
        self.scheduler.add_job(
            self._handle_daily_indexing,
            CronTrigger(hour=4, minute=0),
            id="system_daily_indexing",
            name="System: Daily RAG Indexing",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("System cron: Daily RAG indexing at 04:00")

        # Daily consolidation: Every day at 02:00
        self.scheduler.add_job(
            self._handle_daily_consolidation,
            CronTrigger(hour=2, minute=0),
            id="system_daily_consolidation",
            name="System: Daily Consolidation",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("System cron: Daily consolidation at 02:00")

        # Weekly integration: Every Sunday at 03:00
        self.scheduler.add_job(
            self._handle_weekly_integration,
            CronTrigger(day_of_week="sun", hour=3, minute=0),
            id="system_weekly_integration",
            name="System: Weekly Integration",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("System cron: Weekly integration on Sunday at 03:00")

        # Monthly forgetting: 1st of each month at 03:00
        self.scheduler.add_job(
            self._handle_monthly_forgetting,
            CronTrigger(day=1, hour=3, minute=0),
            id="system_monthly_forgetting",
            name="System: Monthly Forgetting",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("System cron: Monthly forgetting on 1st at 03:00")

        # Daily DM log rotation: Every day at 04:30
        self.scheduler.add_job(
            self._handle_dm_log_rotation,
            CronTrigger(hour=4, minute=30),
            id="system_dm_log_rotation",
            name="System: DM Log Rotation",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("System cron: DM log rotation at 04:30")

    def _schedule_consolidation_retry(self, anima_name: str, max_turns: int) -> None:
        """Schedule a one-shot consolidation retry 3 hours later."""
        retry_time = now_local() + timedelta(hours=3)
        job_id = f"consolidation_retry_{anima_name}"

        self.scheduler.add_job(
            self._run_consolidation_retry,
            DateTrigger(run_date=retry_time),
            id=job_id,
            name=f"Consolidation retry: {anima_name}",
            replace_existing=True,
            kwargs={"anima_name": anima_name, "max_turns": max_turns},
        )
        logger.info("Scheduled consolidation retry for %s at %s", anima_name, retry_time)

    async def _run_consolidation_retry(self, anima_name: str, max_turns: int) -> None:
        """Execute a single consolidation retry. No further retry on failure."""
        anima = self.animas.get(anima_name)
        if anima is None:
            logger.warning("Consolidation retry skipped: anima %s not found", anima_name)
            return
        try:
            result = await anima.run_consolidation(
                consolidation_type="daily",
                max_turns=max_turns,
            )
            logger.info(
                "Consolidation retry for %s completed: duration_ms=%d",
                anima_name,
                result.duration_ms,
            )
            try:
                from core.memory.forgetting import ForgettingEngine

                forgetter = ForgettingEngine(anima.memory.anima_dir, anima_name)
                forgetter.synaptic_downscaling()
            except Exception:
                logger.exception("Synaptic downscaling failed after retry for anima=%s", anima_name)
            try:
                from core.memory.consolidation import ConsolidationEngine

                engine = ConsolidationEngine(anima.memory.anima_dir, anima_name)
                engine._rebuild_rag_index()
            except Exception:
                logger.exception("RAG index rebuild failed after retry for anima=%s", anima_name)
        except Exception:
            logger.exception("Consolidation retry also failed for anima=%s", anima_name)

    async def _handle_daily_indexing(self) -> None:
        """Run daily RAG indexing for all animas.

        Incrementally indexes all memory files (knowledge, episodes,
        procedures, skills, shared_users) so that the RAG database stays
        up-to-date even for files added while the server was stopped.
        Runs at 04:00, after daily consolidation (02:00) and
        weekly/monthly jobs (03:00) to capture all outputs.
        """
        logger.info("Starting system-wide daily RAG indexing")

        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store
        except ImportError:
            logger.warning("RAG dependencies not available, skipping daily indexing")
            return

        from core.memory.rag.singleton import get_embedding_model_name
        from core.memory.rag_search import _compute_dir_hash, _read_shared_hash, _write_shared_hash
        from core.paths import (
            get_common_knowledge_dir,
            get_common_skills_dir,
            get_data_dir,
        )

        base_dir = get_data_dir()
        animas_dir = base_dir / "animas"

        if not animas_dir.is_dir():
            logger.warning("Animas directory not found, skipping daily indexing")
            return

        current_model = get_embedding_model_name()
        global_meta_path = base_dir / "index_meta.json"
        if global_meta_path.is_file():
            try:
                meta = json.loads(global_meta_path.read_text(encoding="utf-8"))
                previous_model = meta.get("embedding_model")
                if previous_model and previous_model != current_model:
                    logger.warning(
                        "Embedding model changed: %s → %s.  "
                        "Skipping daily indexing — run 'animaworks index --full' "
                        "to rebuild.",
                        previous_model,
                        current_model,
                    )
                    return
            except (json.JSONDecodeError, OSError):
                pass

        loop = asyncio.get_running_loop()
        total_chunks = 0
        ck_dir = get_common_knowledge_dir()
        cs_dir = get_common_skills_dir()

        shared_sources: list[tuple[str, Path, str, str]] = []
        if ck_dir.is_dir():
            shared_sources.append(("common_knowledge", ck_dir, "*.md", "shared_common_knowledge_hash"))
        if cs_dir.is_dir():
            shared_sources.append(("common_skills", cs_dir, "SKILL.md", "shared_common_skills_hash"))

        for anima_name, _anima in self.animas.items():
            anima_dir = animas_dir / anima_name
            if not anima_dir.is_dir():
                continue

            try:
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

                meta_path = anima_dir / "index_meta.json"
                for label, src_dir, glob, meta_key in shared_sources:
                    current_hash = _compute_dir_hash(src_dir, glob)
                    stored_hash = _read_shared_hash(meta_path, meta_key)
                    shared_collection = f"shared_{label}"
                    force = False
                    if current_hash == stored_hash:
                        # Verify collection still exists in this anima's
                        # vectordb before short-circuiting (recovery from
                        # wiped/recreated vectordb).
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

        shared_users_dir = base_dir / "shared" / "users"
        if shared_users_dir.is_dir():
            try:
                for anima_name, _anima in self.animas.items():
                    vs = get_vector_store(anima_name)
                    if vs is None:
                        continue
                    su_indexer = MemoryIndexer(vs, "shared", shared_users_dir.parent)
                    chunks = await loop.run_in_executor(
                        None,
                        su_indexer.index_directory,
                        shared_users_dir,
                        "shared_users",
                    )
                    total_chunks += chunks
                    del su_indexer
            except Exception:
                logger.exception("Daily indexing failed for shared_users")

        logger.info(
            "System-wide daily RAG indexing complete: %d chunks indexed",
            total_chunks,
        )

        # Broadcast result via WebSocket
        if self._ws_broadcast:
            await self._ws_broadcast(
                {
                    "type": "system.rag_indexing",
                    "data": {"total_chunks": total_chunks},
                }
            )

    async def _handle_dm_log_rotation(self) -> None:
        """Archive old dm_log entries beyond 7 days."""
        from core.background import rotate_dm_logs
        from core.paths import get_shared_dir

        shared_dir = get_shared_dir()
        try:
            result = await rotate_dm_logs(shared_dir)
            if result:
                logger.info("DM log rotation completed: %s", result)
            else:
                logger.debug("DM log rotation: nothing to archive")
        except Exception:
            logger.exception("DM log rotation failed")
