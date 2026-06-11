from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

from core.anima import DigitalAnima
from core.config.models import load_config
from core.schedule_parser import parse_cron_md as _parse_cron_md
from core.schedule_parser import parse_schedule as _parse_schedule
from core.time_utils import get_app_timezone, now_local

from .inbox_watcher import InboxWatcherMixin
from .rate_limiter import RateLimiterMixin
from .scheduler import SchedulerMixin
from .system_consolidation import SystemConsolidationMixin

logger = logging.getLogger("animaworks.lifecycle")

BroadcastFn = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class LifecycleManager(
    SchedulerMixin,
    InboxWatcherMixin,
    RateLimiterMixin,
    SystemConsolidationMixin,
):
    """Deprecated in-process lifecycle manager for tests and compatibility.

    Production system crons are owned by ``ProcessSupervisor``.  This class
    keeps the public heartbeat/inbox surface import-compatible without
    registering a second copy of system-wide scheduled jobs.
    """

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone=get_app_timezone())
        self.animas: dict[str, DigitalAnima] = {}
        self._ws_broadcast: BroadcastFn | None = None
        self._inbox_watcher_task: asyncio.Task | None = None
        self._pending_triggers: set[str] = set()
        self._deferred_timers: dict[str, asyncio.Handle] = {}
        self._last_msg_heartbeat_end: dict[str, float] = {}
        self._pair_heartbeat_times: dict[tuple[str, str], list[float]] = {}
        self._schedule_mtimes: dict[str, tuple[float, float]] = {}
        hb = load_config().heartbeat
        self._cooldown_s = hb.msg_heartbeat_cooldown_s
        self._cascade_window_s = hb.cascade_window_s
        self._cascade_threshold = hb.cascade_threshold
        self._actionable_intents = hb.actionable_intents

    def set_broadcast(self, fn: BroadcastFn) -> None:
        self._ws_broadcast = fn
        for anima in self.animas.values():
            anima.set_ws_broadcast(fn)

    def register_anima(self, anima: DigitalAnima) -> None:
        self.animas[anima.name] = anima
        anima.set_on_lock_released(lambda n=anima.name: asyncio.ensure_future(self._on_anima_lock_released(n)))
        anima.set_on_schedule_changed(self.reload_anima_schedule)
        if self._ws_broadcast:
            anima.set_ws_broadcast(self._ws_broadcast)
        self._setup_heartbeat(anima)
        self._setup_cron_tasks(anima)
        self._record_schedule_mtimes(anima.name, anima.memory.anima_dir)
        logger.info("Registered '%s' with lifecycle manager", anima.name)

    def unregister_anima(self, name: str) -> None:
        """Remove an anima and all their scheduled jobs."""
        anima = self.animas.pop(name, None)
        if anima:
            anima._session_compactor.cancel_all_for_anima(name)
        self._schedule_mtimes.pop(name, None)
        self._pending_triggers.discard(name)
        timer = self._deferred_timers.pop(name, None)
        if timer:
            timer.cancel()
        for job in self.scheduler.get_jobs():
            if job.id.startswith(f"{name}_") or job.id == f"consolidation_retry_{name}":
                job.remove()
        logger.info("Unregistered '%s' from lifecycle manager", name)

    def reload_anima_schedule(self, name: str) -> dict[str, Any]:
        """Reload heartbeat and cron schedules for an anima from disk."""
        anima = self.animas.get(name)
        if not anima:
            logger.warning("reload_anima_schedule: '%s' not registered", name)
            return {"error": f"Anima '{name}' not registered"}

        hb = load_config().heartbeat
        self._cooldown_s = hb.msg_heartbeat_cooldown_s
        self._cascade_window_s = hb.cascade_window_s
        self._cascade_threshold = hb.cascade_threshold
        self._actionable_intents = hb.actionable_intents

        removed = 0
        for job in self.scheduler.get_jobs():
            if job.id.startswith(f"{name}_"):
                job.remove()
                removed += 1

        self._setup_heartbeat(anima)
        self._setup_cron_tasks(anima)
        self._record_schedule_mtimes(name, anima.memory.anima_dir)

        new_jobs = [j.id for j in self.scheduler.get_jobs() if j.id.startswith(f"{name}_")]
        logger.info(
            "Reloaded schedule for '%s': removed=%d, new_jobs=%s",
            name,
            removed,
            new_jobs,
        )
        return {"reloaded": name, "removed": removed, "new_jobs": new_jobs}

    def start(self) -> None:
        """Start per-Anima heartbeat/cron scheduling and inbox polling."""
        self.scheduler.start()
        self._inbox_watcher_task = asyncio.create_task(self._inbox_watcher_loop())
        logger.info("Lifecycle manager started (scheduler + inbox watcher)")

    def shutdown(self) -> None:
        if self._inbox_watcher_task:
            self._inbox_watcher_task.cancel()
        for timer in self._deferred_timers.values():
            timer.cancel()
        self._deferred_timers.clear()
        for anima in self.animas.values():
            anima._session_compactor.shutdown()
        self.scheduler.shutdown(wait=False)
        logger.info("Lifecycle manager stopped")

    def _schedule_consolidation_retry(self, anima_name: str, max_turns: int) -> None:
        """Schedule a one-shot compatibility consolidation retry."""
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
        """Execute a single compatibility consolidation retry."""
        anima = self.animas.get(anima_name)
        if anima is None:
            logger.warning("Consolidation retry skipped: anima %s not found", anima_name)
            return
        try:
            await anima.run_consolidation(
                consolidation_type="daily",
                max_turns=max_turns,
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
            logger.exception("Consolidation retry failed for anima=%s", anima_name)


__all__ = [
    "BroadcastFn",
    "InboxWatcherMixin",
    "LifecycleManager",
    "RateLimiterMixin",
    "SchedulerMixin",
    "SystemConsolidationMixin",
    "_parse_cron_md",
    "_parse_schedule",
]
