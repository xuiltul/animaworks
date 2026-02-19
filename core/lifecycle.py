from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import asyncio
import logging
import re
import time
from typing import Any, Callable, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.anima import DigitalAnima
from core.schemas import CronTask

logger = logging.getLogger("animaworks.lifecycle")

BroadcastFn = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]

# Minimum seconds between consecutive message-triggered heartbeats
# for the same anima. Prevents cascading loops (A sends to B, B replies
# to A, A replies to B, …).
_MSG_HEARTBEAT_COOLDOWN_S = 60

_CASCADE_WINDOW_S = 600   # 10 minutes
_CASCADE_THRESHOLD = 4     # max round-trips per pair within window

class LifecycleManager:
    """Manages heartbeat and cron for Digital Animas via APScheduler."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
        self.animas: dict[str, DigitalAnima] = {}
        self._ws_broadcast: BroadcastFn | None = None
        self._inbox_watcher_task: asyncio.Task | None = None
        self._pending_triggers: set[str] = set()
        self._deferred_timers: dict[str, asyncio.Handle] = {}
        self._last_msg_heartbeat_end: dict[str, float] = {}
        self._pair_heartbeat_times: dict[tuple[str, str], list[float]] = {}

    def set_broadcast(self, fn: BroadcastFn) -> None:
        self._ws_broadcast = fn
        # Propagate to already-registered animas for bg task notifications
        for anima in self.animas.values():
            anima.set_ws_broadcast(fn)

    def register_anima(self, anima: DigitalAnima) -> None:
        self.animas[anima.name] = anima
        # Wire up lock-release callback for deferred inbox processing
        anima.set_on_lock_released(
            lambda n=anima.name: asyncio.ensure_future(
                self._on_anima_lock_released(n)
            )
        )
        # Wire up schedule-changed callback for hot-reload
        anima.set_on_schedule_changed(self.reload_anima_schedule)
        # Wire up WebSocket broadcast for background task notifications
        if self._ws_broadcast:
            anima.set_ws_broadcast(self._ws_broadcast)
        self._setup_heartbeat(anima)
        self._setup_cron_tasks(anima)
        logger.info("Registered '%s' with lifecycle manager", anima.name)

    def unregister_anima(self, name: str) -> None:
        """Remove an anima and all their scheduled jobs."""
        self.animas.pop(name, None)
        self._pending_triggers.discard(name)
        timer = self._deferred_timers.pop(name, None)
        if timer:
            timer.cancel()
        # Remove all scheduler jobs belonging to this anima
        for job in self.scheduler.get_jobs():
            if job.id.startswith(f"{name}_"):
                job.remove()
        logger.info("Unregistered '%s' from lifecycle manager", name)

    def reload_anima_schedule(self, name: str) -> dict[str, Any]:
        """Reload heartbeat and cron schedules for an anima from disk.

        Called when heartbeat.md or cron.md is modified at runtime.

        Args:
            name: The anima name whose schedule should be reloaded.

        Returns:
            A summary dict with keys ``reloaded``, ``removed``, ``new_jobs``
            (or ``error`` if the anima is not registered).
        """
        anima = self.animas.get(name)
        if not anima:
            logger.warning("reload_anima_schedule: '%s' not registered", name)
            return {"error": f"Anima '{name}' not registered"}

        # Remove existing heartbeat and cron jobs for this anima
        removed = 0
        for job in self.scheduler.get_jobs():
            if job.id.startswith(f"{name}_"):
                job.remove()
                removed += 1

        # Re-setup from current files on disk
        self._setup_heartbeat(anima)
        self._setup_cron_tasks(anima)

        new_jobs = [
            j.id for j in self.scheduler.get_jobs()
            if j.id.startswith(f"{name}_")
        ]
        logger.info(
            "Reloaded schedule for '%s': removed=%d, new_jobs=%s",
            name, removed, new_jobs,
        )
        return {"reloaded": name, "removed": removed, "new_jobs": new_jobs}

    # ── Heartbeat ─────────────────────────────────────────

    def _setup_heartbeat(self, anima: DigitalAnima) -> None:
        config = anima.memory.read_heartbeat_config()

        _HEARTBEAT_INTERVAL = 30  # Fixed system-wide; not configurable per anima

        active_start, active_end = 9, 22
        m = re.search(r"(\d{1,2}):\d{0,2}\s*-\s*(\d{1,2})", config)
        if m:
            active_start, active_end = int(m.group(1)), int(m.group(2))

        self.scheduler.add_job(
            self._heartbeat_wrapper,
            CronTrigger(
                minute=f"*/{_HEARTBEAT_INTERVAL}",
                hour=f"{active_start}-{active_end - 1}",
            ),
            id=f"{anima.name}_heartbeat",
            name=f"{anima.name} heartbeat",
            args=[anima.name],
            replace_existing=True,
        )
        logger.info(
            "Heartbeat '%s': every %dmin, active %d:00-%d:00",
            anima.name,
            _HEARTBEAT_INTERVAL,
            active_start,
            active_end,
        )

    async def _heartbeat_wrapper(self, name: str) -> None:
        anima = self.animas.get(name)
        if not anima:
            return

        logger.info("Heartbeat: %s", name)
        result = await anima.run_heartbeat()
        self._last_msg_heartbeat_end[name] = time.monotonic()
        if self._ws_broadcast:
            await self._ws_broadcast(
                {
                    "type": "anima.heartbeat",
                    "data": {"name": name, "result": result.model_dump()},
                }
            )

    # ── Cron ──────────────────────────────────────────────

    def _setup_cron_tasks(self, anima: DigitalAnima) -> None:
        config = anima.memory.read_cron_config()
        if not config:
            return

        tasks = _parse_cron_md(config)
        for i, task in enumerate(tasks):
            trigger = _parse_schedule(task.schedule)
            if trigger:
                self.scheduler.add_job(
                    self._cron_wrapper,
                    trigger,
                    id=f"{anima.name}_cron_{i}",
                    name=f"{anima.name}: {task.name}",
                    args=[anima.name, task],  # Pass entire CronTask object
                    replace_existing=True,
                )
                logger.info(
                    "Cron '%s': %s (%s) [%s]",
                    anima.name,
                    task.name,
                    task.schedule,
                    task.type,
                )

    async def _cron_wrapper(self, name: str, task: CronTask) -> None:
        """Wrapper for cron task execution (both LLM and command types)."""
        anima = self.animas.get(name)
        if not anima:
            return

        logger.info("Cron: %s -> %s [%s]", name, task.name, task.type)
        # Run cron tasks without awaiting lock — use create_task so
        # multiple simultaneous cron tasks don't block each other.
        asyncio.create_task(
            self._run_cron_and_broadcast(anima, name, task),
            name=f"cron-{name}-{task.name}",
        )

    async def _run_cron_and_broadcast(
        self,
        anima: DigitalAnima,
        name: str,
        task: CronTask,
    ) -> None:
        """Execute a cron task (LLM or command type) and broadcast the result."""
        try:
            if task.type == "llm":
                # LLM-type: invoke agent.run_cycle
                result = await anima.run_cron_task(task.name, task.description)
                broadcast_data = {
                    "type": "anima.cron",
                    "data": {
                        "name": name,
                        "task": task.name,
                        "task_type": "llm",
                        "result": result.model_dump(),
                    },
                }
            elif task.type == "command":
                # Command-type: execute bash/tool directly
                result = await anima.run_cron_command(
                    task.name,
                    command=task.command,
                    tool=task.tool,
                    args=task.args,
                )
                broadcast_data = {
                    "type": "anima.cron",
                    "data": {
                        "name": name,
                        "task": task.name,
                        "task_type": "command",
                        "result": result,
                    },
                }
            else:
                logger.warning(
                    "Unknown cron task type '%s' for %s -> %s",
                    task.type,
                    name,
                    task.name,
                )
                return

            if self._ws_broadcast:
                await self._ws_broadcast(broadcast_data)
        except Exception:
            logger.exception("Cron task failed: %s -> %s", name, task.name)

    # ── Inbox Watcher ──────────────────────────────────────

    def _is_in_cooldown(self, name: str) -> bool:
        """Return True if a message-triggered heartbeat finished too recently."""
        last = self._last_msg_heartbeat_end.get(name, 0.0)
        return (time.monotonic() - last) < _MSG_HEARTBEAT_COOLDOWN_S

    def _check_cascade(self, anima_name: str, senders: set[str]) -> bool:
        """Return True if any (anima, sender) pair exceeds cascade threshold."""
        now = time.monotonic()
        for sender in senders:
            keys = [(anima_name, sender), (sender, anima_name)]
            total = 0
            for k in keys:
                times = self._pair_heartbeat_times.get(k, [])
                # Evict expired entries
                times = [t for t in times if now - t < _CASCADE_WINDOW_S]
                self._pair_heartbeat_times[k] = times
                if not times and k in self._pair_heartbeat_times:
                    del self._pair_heartbeat_times[k]
                total += len(times)
            if total >= _CASCADE_THRESHOLD:
                logger.warning(
                    "CASCADE DETECTED: %s <-> %s (%d round-trips in %ds window). "
                    "Suppressing message-triggered heartbeat.",
                    anima_name, sender, total, _CASCADE_WINDOW_S,
                )
                return True
        return False

    def _record_pair_heartbeat(self, anima_name: str, senders: set[str]) -> None:
        """Record a heartbeat exchange for cascade tracking."""
        now = time.monotonic()
        for sender in senders:
            key = (anima_name, sender)
            self._pair_heartbeat_times.setdefault(key, []).append(now)

    async def _inbox_watcher_loop(self) -> None:
        """Poll inbox dirs every 2s; trigger heartbeat on new messages."""
        logger.info("Inbox watcher started (poll interval: 2s)")
        while True:
            await asyncio.sleep(2)
            for name, anima in self.animas.items():
                if name in self._pending_triggers:
                    continue
                if not anima.messenger.has_unread():
                    continue
                if self._is_in_cooldown(name):
                    self._schedule_deferred_trigger(name)
                    continue
                if anima._lock.locked():
                    self._schedule_deferred_trigger(name)
                    continue
                self._pending_triggers.add(name)
                asyncio.create_task(
                    self._message_triggered_heartbeat(name)
                )

    async def _on_anima_lock_released(self, name: str) -> None:
        """Check deferred inbox after an anima's lock is released.

        If unread messages exist, schedule a deferred trigger to ensure
        they are processed even when cooldown is still active.
        """
        anima = self.animas.get(name)
        if not anima:
            return
        if not anima.messenger.has_unread():
            return
        if name in self._pending_triggers:
            return
        # Instead of giving up when in cooldown, schedule deferred trigger
        self._schedule_deferred_trigger(name)

    def _schedule_deferred_trigger(self, name: str) -> None:
        """Schedule a deferred heartbeat trigger after cooldown expires.

        Only one timer per anima is maintained.  If a timer is already
        pending, the call is a no-op (the existing timer will fire and
        re-check the inbox).
        """
        if name in self._deferred_timers:
            return  # already scheduled
        last = self._last_msg_heartbeat_end.get(name, 0.0)
        remaining = _MSG_HEARTBEAT_COOLDOWN_S - (time.monotonic() - last)
        # If not in cooldown (e.g. lock-only), use a short retry delay
        delay = max(remaining, 2.0)
        loop = asyncio.get_running_loop()
        self._deferred_timers[name] = loop.call_later(
            delay,
            lambda n=name: asyncio.create_task(self._try_deferred_trigger(n)),
        )
        logger.debug(
            "Deferred trigger scheduled for %s in %.1fs", name, delay,
        )

    async def _try_deferred_trigger(self, name: str) -> None:
        """Attempt to trigger a deferred heartbeat.

        Re-schedules itself if the anima is still blocked by cooldown
        or lock, ensuring messages are never forgotten.
        """
        self._deferred_timers.pop(name, None)
        anima = self.animas.get(name)
        if not anima:
            return
        if not anima.messenger.has_unread():
            return
        if name in self._pending_triggers:
            return
        if self._is_in_cooldown(name):
            self._schedule_deferred_trigger(name)
            return
        if anima._lock.locked():
            self._schedule_deferred_trigger(name)
            return
        self._pending_triggers.add(name)
        asyncio.create_task(self._message_triggered_heartbeat(name))

    async def _message_triggered_heartbeat(self, name: str) -> None:
        anima = self.animas.get(name)
        if not anima:
            self._pending_triggers.discard(name)
            return

        # Peek at inbox senders for cascade detection
        senders = {m.from_person for m in anima.messenger.receive()}
        if senders and self._check_cascade(name, senders):
            self._pending_triggers.discard(name)
            return

        try:
            logger.info("Message-triggered heartbeat: %s", name)
            result = await anima.run_heartbeat()
            if self._ws_broadcast:
                await self._ws_broadcast(
                    {
                        "type": "anima.message_heartbeat",
                        "data": {"name": name, "result": result.model_dump()},
                    }
                )
        except Exception:
            logger.exception("Message-triggered heartbeat failed: %s", name)
        finally:
            self._pending_triggers.discard(name)
            self._last_msg_heartbeat_end[name] = time.monotonic()
            if senders:
                self._record_pair_heartbeat(name, senders)

    # ── System Crons ──────────────────────────────────────

    def _setup_system_crons(self) -> None:
        """Set up system-wide cron tasks for memory consolidation."""
        # Daily RAG indexing: Every day at 04:00 JST
        # Runs after consolidation (02:00) and weekly/monthly jobs (03:00)
        # to catch all generated/modified files as a final sweep
        self.scheduler.add_job(
            self._handle_daily_indexing,
            CronTrigger(hour=4, minute=0),
            id="system_daily_indexing",
            name="System: Daily RAG Indexing",
            replace_existing=True,
        )
        logger.info("System cron: Daily RAG indexing at 04:00 JST")

        # Daily consolidation: Every day at 02:00 JST
        self.scheduler.add_job(
            self._handle_daily_consolidation,
            CronTrigger(hour=2, minute=0),
            id="system_daily_consolidation",
            name="System: Daily Consolidation",
            replace_existing=True,
        )
        logger.info("System cron: Daily consolidation at 02:00 JST")

        # Weekly integration: Every Sunday at 03:00 JST
        self.scheduler.add_job(
            self._handle_weekly_integration,
            CronTrigger(day_of_week="sun", hour=3, minute=0),
            id="system_weekly_integration",
            name="System: Weekly Integration",
            replace_existing=True,
        )
        logger.info("System cron: Weekly integration on Sunday at 03:00 JST")

        # Monthly forgetting: 1st of each month at 03:00 JST
        self.scheduler.add_job(
            self._handle_monthly_forgetting,
            CronTrigger(day=1, hour=3, minute=0),
            id="system_monthly_forgetting",
            name="System: Monthly Forgetting",
            replace_existing=True,
        )
        logger.info("System cron: Monthly forgetting on 1st at 03:00 JST")

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
            from core.memory.rag.store import ChromaVectorStore
        except ImportError:
            logger.warning("RAG dependencies not available, skipping daily indexing")
            return

        from core.paths import get_common_skills_dir, get_data_dir

        base_dir = get_data_dir()
        animas_dir = base_dir / "animas"

        if not animas_dir.is_dir():
            logger.warning("Animas directory not found, skipping daily indexing")
            return

        # Check for embedding model change
        import json
        from core.memory.rag.singleton import get_embedding_model_name

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
        vector_store = ChromaVectorStore()
        total_chunks = 0

        # Index each anima
        for anima_name, _anima in self.animas.items():
            anima_dir = animas_dir / anima_name
            if not anima_dir.is_dir():
                continue

            try:
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
                        None, indexer.index_directory, memory_dir, memory_type,
                    )
                    total_chunks += chunks

                logger.info("Daily indexing for %s complete", anima_name)

            except Exception:
                logger.exception(
                    "Daily indexing failed for anima=%s", anima_name
                )

        # Index shared user memories
        shared_users_dir = base_dir / "shared" / "users"
        if shared_users_dir.is_dir():
            try:
                indexer = MemoryIndexer(
                    vector_store, "shared", shared_users_dir.parent
                )
                chunks = await loop.run_in_executor(
                    None, indexer.index_directory, shared_users_dir, "shared_users",
                )
                total_chunks += chunks
            except Exception:
                logger.exception("Daily indexing failed for shared_users")

        # Index shared common_skills
        common_skills_dir = get_common_skills_dir()
        if common_skills_dir.is_dir():
            try:
                shared_indexer = MemoryIndexer(
                    vector_store, "shared", base_dir,
                    collection_prefix="shared",
                )
                chunks = await loop.run_in_executor(
                    None, shared_indexer.index_directory, common_skills_dir, "common_skills",
                )
                total_chunks += chunks
            except Exception:
                logger.exception("Daily indexing failed for common_skills")

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

    async def _handle_daily_consolidation(self) -> None:
        """Run daily consolidation for all animas."""
        logger.info("Starting system-wide daily consolidation")

        # Load consolidation config
        from core.config import load_config
        config = load_config()
        consolidation_cfg = getattr(config, "consolidation", None)

        # Default config if not present
        enabled = True
        model = "anthropic/claude-sonnet-4-20250514"
        min_episodes = 1

        if consolidation_cfg:
            enabled = getattr(consolidation_cfg, "daily_enabled", True)
            model = getattr(consolidation_cfg, "llm_model", model)
            min_episodes = getattr(consolidation_cfg, "min_episodes_threshold", 1)

        if not enabled:
            logger.info("Daily consolidation is disabled in config")
            return

        # Run consolidation for each anima
        for anima_name, anima in self.animas.items():
            try:
                from core.memory.consolidation import ConsolidationEngine

                engine = ConsolidationEngine(
                    anima_dir=anima.memory.anima_dir,
                    anima_name=anima_name,
                )

                result = await engine.daily_consolidate(
                    model=model,
                    min_episodes=min_episodes,
                )

                logger.info(
                    "Daily consolidation for %s: %s",
                    anima_name,
                    result
                )

                # Broadcast result via WebSocket
                if self._ws_broadcast and not result.get("skipped"):
                    await self._ws_broadcast(
                        {
                            "type": "system.consolidation",
                            "data": {
                                "anima": anima_name,
                                "type": "daily",
                                "result": result,
                            },
                        }
                    )

            except Exception:
                logger.exception(
                    "Daily consolidation failed for anima=%s",
                    anima_name
                )

    async def _handle_weekly_integration(self) -> None:
        """Run weekly integration for all animas."""
        logger.info("Starting system-wide weekly integration")

        # Load config
        from core.config import load_config
        config = load_config()
        consolidation_cfg = getattr(config, "consolidation", None)

        # Default config
        enabled = True  # Phase 3 implementation
        model = "anthropic/claude-sonnet-4-20250514"
        duplicate_threshold = 0.85
        episode_retention_days = 30

        if consolidation_cfg:
            enabled = getattr(consolidation_cfg, "weekly_enabled", True)
            model = getattr(consolidation_cfg, "llm_model", model)
            duplicate_threshold = getattr(consolidation_cfg, "duplicate_threshold", 0.85)
            episode_retention_days = getattr(consolidation_cfg, "episode_retention_days", 30)

        if not enabled:
            logger.info("Weekly integration is disabled in config")
            return

        # Run integration for each anima
        for anima_name, anima in self.animas.items():
            try:
                from core.memory.consolidation import ConsolidationEngine

                engine = ConsolidationEngine(
                    anima_dir=anima.memory.anima_dir,
                    anima_name=anima_name,
                )

                result = await engine.weekly_integrate(
                    model=model,
                    duplicate_threshold=duplicate_threshold,
                    episode_retention_days=episode_retention_days,
                )

                logger.info(
                    "Weekly integration for %s: merged=%d compressed=%d",
                    anima_name,
                    len(result.get("knowledge_files_merged", [])),
                    result.get("episodes_compressed", 0)
                )

                # Broadcast result
                if self._ws_broadcast and not result.get("skipped"):
                    await self._ws_broadcast(
                        {
                            "type": "system.consolidation",
                            "data": {
                                "anima": anima_name,
                                "type": "weekly",
                                "result": result,
                            },
                        }
                    )

            except Exception:
                logger.exception(
                    "Weekly integration failed for anima=%s",
                    anima_name
                )

    async def _handle_monthly_forgetting(self) -> None:
        """Run monthly forgetting for all animas."""
        logger.info("Starting system-wide monthly forgetting")

        # Load config
        from core.config import load_config
        config = load_config()
        consolidation_cfg = getattr(config, "consolidation", None)

        # Default config
        enabled = True

        if consolidation_cfg:
            enabled = getattr(consolidation_cfg, "monthly_forgetting_enabled", True)

        if not enabled:
            logger.info("Monthly forgetting is disabled in config")
            return

        # Run forgetting for each anima
        for anima_name, anima in self.animas.items():
            try:
                from core.memory.consolidation import ConsolidationEngine

                engine = ConsolidationEngine(
                    anima_dir=anima.memory.anima_dir,
                    anima_name=anima_name,
                )

                result = await engine.monthly_forget()

                logger.info(
                    "Monthly forgetting for %s: forgotten=%d archived=%d",
                    anima_name,
                    result.get("forgotten_chunks", 0),
                    len(result.get("archived_files", [])),
                )

                # Broadcast result
                if self._ws_broadcast:
                    await self._ws_broadcast(
                        {
                            "type": "system.consolidation",
                            "data": {
                                "anima": anima_name,
                                "type": "monthly_forgetting",
                                "result": result,
                            },
                        }
                    )

            except Exception:
                logger.exception(
                    "Monthly forgetting failed for anima=%s",
                    anima_name
                )

    # ── Lifecycle ─────────────────────────────────────────

    def start(self) -> None:
        self.scheduler.start()
        self._setup_system_crons()
        self._inbox_watcher_task = asyncio.create_task(
            self._inbox_watcher_loop()
        )
        logger.info("Lifecycle manager started (scheduler + inbox watcher + system crons)")

    def shutdown(self) -> None:
        if self._inbox_watcher_task:
            self._inbox_watcher_task.cancel()
        for timer in self._deferred_timers.values():
            timer.cancel()
        self._deferred_timers.clear()
        self.scheduler.shutdown(wait=False)
        logger.info("Lifecycle manager stopped")


# ── Parsing helpers (re-exported from schedule_parser) ────
from core.schedule_parser import (  # noqa: E402
    parse_cron_md as _parse_cron_md,
    parse_schedule as _parse_schedule,
    parse_heartbeat_config as _parse_heartbeat_config,
)