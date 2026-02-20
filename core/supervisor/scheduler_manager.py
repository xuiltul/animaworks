# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""APScheduler management for heartbeat and cron tasks.

Handles registration, execution, overlap prevention, and hot-reload
of heartbeat and cron schedules for a single Anima process.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.schedule_parser import parse_cron_md, parse_schedule, parse_heartbeat_config
from core.schemas import CronTask
from core.time_utils import now_jst

if TYPE_CHECKING:
    from core.anima import DigitalAnima

logger = logging.getLogger(__name__)


class SchedulerManager:
    """APScheduler management: heartbeat/cron registration, execution, reload."""

    def __init__(
        self,
        anima: DigitalAnima,
        anima_name: str,
        anima_dir: Path,
        emit_event: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self._anima = anima
        self._anima_name = anima_name
        self._anima_dir = anima_dir
        self._emit_event = emit_event

        self.scheduler: AsyncIOScheduler | None = None
        self._heartbeat_running: bool = False
        self._cron_running: set[str] = set()

    # ── Public Properties ────────────────────────────────────────

    @property
    def heartbeat_running(self) -> bool:
        """Whether a heartbeat is currently executing (read by InboxRateLimiter)."""
        return self._heartbeat_running

    @heartbeat_running.setter
    def heartbeat_running(self, value: bool) -> None:
        self._heartbeat_running = value

    # ── Setup ────────────────────────────────────────────────────

    def setup(self) -> None:
        """Set up and start the autonomous scheduler for heartbeat and cron."""
        if not self._anima:
            return

        try:
            self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
            self._setup_heartbeat()
            self._setup_cron_tasks()
            self.scheduler.start()

            # Wire up hot-reload callback
            self._anima.set_on_schedule_changed(self.reload_schedule)

            job_count = len(self.scheduler.get_jobs())
            logger.info(
                "Scheduler started for %s: %d jobs registered",
                self._anima_name, job_count,
            )
        except Exception:
            logger.exception("Failed to setup scheduler for %s", self._anima_name)
            self.scheduler = None

    def _setup_heartbeat(self) -> None:
        """Register heartbeat job from heartbeat.md."""
        if not self._anima or not self.scheduler:
            return

        config = self._anima.memory.read_heartbeat_config()
        if not config:
            return

        interval, active_start, active_end = parse_heartbeat_config(config)

        # Determine active hours:
        # 1. heartbeat.md explicit time range (already parsed above)
        # 2. Default: 24h (hour="*")
        m = re.search(r"(\d{1,2}):\d{0,2}\s*-\s*(\d{1,2})", config)
        if m:
            hour_spec = f"{active_start}-{active_end - 1}"
            log_active = f"active {active_start}:00-{active_end}:00"
        else:
            hour_spec = "*"
            log_active = "24h"

        self.scheduler.add_job(
            self.heartbeat_tick,
            CronTrigger(
                minute=f"*/{interval}",
                hour=hour_spec,
            ),
            id=f"{self._anima_name}_heartbeat",
            name=f"{self._anima_name} heartbeat",
            replace_existing=True,
            misfire_grace_time=300,
            max_instances=1,
        )
        logger.info(
            "Heartbeat registered: %s every %dmin, %s",
            self._anima_name, interval, log_active,
        )

    def _setup_cron_tasks(self) -> None:
        """Register cron jobs from cron.md."""
        if not self._anima or not self.scheduler:
            return

        config = self._anima.memory.read_cron_config()
        if not config:
            return

        tasks = parse_cron_md(config)
        for i, task in enumerate(tasks):
            trigger = parse_schedule(task.schedule)
            if not trigger:
                logger.warning(
                    "Could not parse schedule for cron task '%s': '%s'",
                    task.name, task.schedule,
                )
                continue

            self.scheduler.add_job(
                self.cron_tick,
                trigger,
                id=f"{self._anima_name}_cron_{i}",
                name=f"{self._anima_name}: {task.name}",
                args=[task],
                replace_existing=True,
                misfire_grace_time=300,
                max_instances=1,
            )
            logger.info(
                "Cron registered: %s -> %s (%s) [%s]",
                self._anima_name, task.name, task.schedule, task.type,
            )

    # ── Tick Handlers ────────────────────────────────────────────

    async def heartbeat_tick(self) -> None:
        """Execute a scheduled heartbeat."""
        if not self._anima:
            return
        if self._heartbeat_running:
            logger.info("Scheduled heartbeat SKIPPED (already running): %s", self._anima_name)
            return
        self._heartbeat_running = True
        try:
            logger.info("Scheduled heartbeat: %s", self._anima_name)
            result = await self._anima.run_heartbeat()
            # Notify parent for WebSocket broadcast
            self._emit_event("anima.heartbeat", {
                "name": self._anima_name,
                "result": result.model_dump(),
            })
        except Exception:
            logger.exception("Scheduled heartbeat failed: %s", self._anima_name)
        finally:
            self._heartbeat_running = False

    async def cron_tick(self, task: CronTask) -> None:
        """Execute a scheduled cron task."""
        if not self._anima:
            return

        if task.name in self._cron_running:
            logger.info(
                "Scheduled cron SKIPPED (already running): %s -> %s",
                self._anima_name, task.name,
            )
            return

        logger.info("Scheduled cron: %s -> %s [%s]", self._anima_name, task.name, task.type)
        # Run in separate task to avoid blocking other scheduled jobs
        asyncio.create_task(
            self._run_cron_task(task),
            name=f"cron-{self._anima_name}-{task.name}",
        )

    async def _run_cron_task(self, task: CronTask) -> None:
        """Run a single cron task (LLM or command type)."""
        if not self._anima:
            return
        self._cron_running.add(task.name)
        try:
            if task.type == "llm":
                result = await self._anima.run_cron_task(task.name, task.description)
                self._emit_event("anima.cron", {
                    "name": self._anima_name,
                    "task": task.name,
                    "task_type": "llm",
                    "result": result.model_dump(),
                })
            elif task.type == "command":
                result = await self._anima.run_cron_command(
                    task.name,
                    command=task.command,
                    tool=task.tool,
                    args=task.args,
                )
                self._emit_event("anima.cron", {
                    "name": self._anima_name,
                    "task": task.name,
                    "task_type": "command",
                    "result": result,
                })
                # If command produced non-empty output, write to
                # pending.md and trigger a heartbeat so the LLM can
                # review and act on the results.
                stdout = result.get("stdout", "").strip()
                if stdout and result.get("exit_code", 0) == 0:
                    # Check skip_pattern: if stdout matches, suppress heartbeat
                    if task.skip_pattern:
                        try:
                            if re.search(task.skip_pattern, stdout):
                                logger.info(
                                    "Cron command '%s' output matched skip_pattern, "
                                    "suppressing heartbeat for %s",
                                    task.name, self._anima_name,
                                )
                                return
                        except re.error as e:
                            logger.warning(
                                "Invalid skip_pattern '%s' for task '%s': %s — "
                                "continuing without skip",
                                task.skip_pattern, task.name, e,
                            )

                    ts = now_jst().strftime("%Y-%m-%d %H:%M")
                    pending = (
                        f"## cron結果: {task.name} ({ts})\n\n"
                        f"{stdout}\n"
                    )
                    self._anima.memory.update_pending(pending)
                    logger.info(
                        "Cron command '%s' produced output, triggering heartbeat for %s",
                        task.name, self._anima_name,
                    )
                    asyncio.create_task(
                        self.heartbeat_tick(),
                        name=f"cron-triggered-heartbeat-{self._anima_name}",
                    )
            else:
                logger.warning("Unknown cron type '%s' for task '%s'", task.type, task.name)
        except Exception:
            logger.exception("Cron task failed: %s -> %s", self._anima_name, task.name)
        finally:
            self._cron_running.discard(task.name)

    # ── Reload ───────────────────────────────────────────────────

    def reload_schedule(self, name: str) -> dict[str, Any]:
        """Reload heartbeat and cron schedules from disk (hot-reload callback)."""
        if not self.scheduler:
            return {"error": "Scheduler not running"}

        # Remove all existing jobs
        removed = 0
        for job in self.scheduler.get_jobs():
            job.remove()
            removed += 1

        # Re-setup from current files
        self._setup_heartbeat()
        self._setup_cron_tasks()

        new_jobs = [j.id for j in self.scheduler.get_jobs()]
        logger.info(
            "Schedule reloaded for %s: removed=%d, new_jobs=%s",
            self._anima_name, removed, new_jobs,
        )
        return {"reloaded": name, "removed": removed, "new_jobs": new_jobs}

    # ── Cleanup ──────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop the scheduler."""
        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:
                logger.debug("Scheduler shutdown failed for %s (may not have been started)", self._anima_name, exc_info=True)
            logger.info("Scheduler stopped for %s", self._anima_name)
