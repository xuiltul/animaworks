from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import asyncio
import logging
import time
import zlib
from pathlib import Path

from apscheduler.triggers.cron import CronTrigger

from core.anima import DigitalAnima
from core.config.models import load_config
from core.schedule_parser import parse_cron_md as _parse_cron_md
from core.schedule_parser import parse_heartbeat_config
from core.schedule_parser import parse_schedule as _parse_schedule
from core.schemas import CronTask

logger = logging.getLogger("animaworks.lifecycle")


class SchedulerMixin:
    """Mixin providing heartbeat and cron task scheduling and schedule freshness checks."""

    def _record_schedule_mtimes(self, name: str, anima_dir: Path) -> None:
        """Snapshot cron.md and heartbeat.md mtimes for later freshness checks."""
        cron_path = anima_dir / "cron.md"
        hb_path = anima_dir / "heartbeat.md"
        try:
            cron_mt = cron_path.stat().st_mtime if cron_path.is_file() else 0.0
        except OSError:
            cron_mt = 0.0
        try:
            hb_mt = hb_path.stat().st_mtime if hb_path.is_file() else 0.0
        except OSError:
            hb_mt = 0.0
        self._schedule_mtimes[name] = (cron_mt, hb_mt)

    def _check_schedule_freshness(self, name: str) -> bool:
        """Check if cron.md or heartbeat.md changed since last setup.

        If a change is detected, reloads the schedule and returns True.
        Returns False when no change is detected or the anima is unknown.
        """
        anima = self.animas.get(name)
        if not anima:
            return False
        prev = self._schedule_mtimes.get(name)
        if prev is None:
            return False

        anima_dir: Path = anima.memory.anima_dir
        cron_path = anima_dir / "cron.md"
        hb_path = anima_dir / "heartbeat.md"
        try:
            cron_mt = cron_path.stat().st_mtime if cron_path.is_file() else 0.0
        except OSError:
            cron_mt = 0.0
        try:
            hb_mt = hb_path.stat().st_mtime if hb_path.is_file() else 0.0
        except OSError:
            hb_mt = 0.0

        if (cron_mt, hb_mt) != prev:
            logger.info(
                "Schedule file changed for '%s' — reloading (cron: %.0f->%.0f, hb: %.0f->%.0f)",
                name,
                prev[0],
                cron_mt,
                prev[1],
                hb_mt,
            )
            self.reload_anima_schedule(name)
            return True
        return False

    def _setup_heartbeat(self, anima: DigitalAnima) -> None:
        job_id = f"{anima.name}_heartbeat"
        heartbeat_enabled = anima.memory.read_model_config().heartbeat_enabled
        if not heartbeat_enabled:
            if self.scheduler.get_job(job_id) is not None:
                self.scheduler.remove_job(job_id)
            logger.info("Heartbeat disabled for '%s'", anima.name)
            return

        hb_content = anima.memory.read_heartbeat_config()

        # Interval from config.json (not parsed from heartbeat.md)
        app_config = load_config()
        interval = app_config.heartbeat.interval_minutes

        # Fixed offset: crc32(anima_name) % 10 → deterministic 0-9 min spread
        offset = zlib.crc32(anima.name.encode()) % 10

        # Snap to nearest divisor of 60 so cross-hour gaps stay uniform
        _DIVISORS_OF_60 = [5, 6, 10, 12, 15, 20, 30, 60]
        if interval > 0 and 60 % interval != 0:
            snapped = min(_DIVISORS_OF_60, key=lambda d: abs(d - interval))
            logger.info(
                "Heartbeat '%s': interval %d not a divisor of 60, snapped to %d",
                anima.name,
                interval,
                snapped,
            )
            interval = snapped

        # Build minute spec: base slots + offset
        slots = []
        m = offset
        while m < 60:
            slots.append(str(m))
            m += interval
        minute_spec = ",".join(slots)

        # Determine active hours from heartbeat.md
        active_start, active_end = parse_heartbeat_config(hb_content)
        if active_start is not None and active_end is not None:
            hour_spec = f"{active_start}-{active_end - 1}"
            log_active = f"active {active_start}:00-{active_end}:00"
        else:
            hour_spec = "*"
            log_active = "24h"

        self.scheduler.add_job(
            self._heartbeat_wrapper,
            CronTrigger(
                minute=minute_spec,
                hour=hour_spec,
            ),
            id=job_id,
            name=f"{anima.name} heartbeat",
            args=[anima.name],
            replace_existing=True,
        )
        logger.info(
            "Heartbeat '%s': minute=%s (offset=%d, interval=%dmin), %s",
            anima.name,
            minute_spec,
            offset,
            interval,
            log_active,
        )

    async def _heartbeat_wrapper(self, name: str) -> None:
        anima = self.animas.get(name)
        if not anima:
            return

        # Detect schedule file changes (Mode S Write/Edit bypass)
        self._check_schedule_freshness(name)

        # ── Cascade detection for scheduled heartbeats ──
        # NOTE: TOCTOU — inbox is peeked here and re-read inside run_heartbeat().
        # Messages may arrive between the two reads. This is acceptable because
        # the depth limiter at Messenger.send() is the primary defense; this
        # check is a best-effort supplementary layer.
        cascade_suppressed: set[str] = set()
        senders: set[str] = set()
        try:
            inbox_messages = anima.messenger.receive()
            senders = {m.from_person for m in inbox_messages}
            if senders:
                for sender in senders:
                    if self._check_cascade(name, {sender}):
                        cascade_suppressed.add(sender)
        except Exception:
            logger.debug("Cascade check failed for scheduled HB: %s", name, exc_info=True)

        logger.info("Heartbeat: %s (cascade_suppressed=%s)", name, cascade_suppressed or "none")
        result = await anima.run_heartbeat(
            cascade_suppressed_senders=cascade_suppressed or None,
        )
        self._last_msg_heartbeat_end[name] = time.monotonic()

        # Record pair exchanges for processed senders
        try:
            processed_senders = senders - cascade_suppressed if senders else set()
            if processed_senders:
                self._record_pair_heartbeat(name, processed_senders)
        except Exception:
            logger.debug("Pair heartbeat recording failed: %s", name, exc_info=True)

        if self._ws_broadcast:
            await self._ws_broadcast(
                {
                    "type": "anima.heartbeat",
                    "data": {"name": name, "result": result.model_dump()},
                }
            )

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
                    misfire_grace_time=600,
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

        # Detect schedule file changes and skip stale tasks
        if self._check_schedule_freshness(name):
            logger.info(
                "Skipping stale cron '%s' for '%s' (schedule reloaded)",
                task.name,
                name,
            )
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
                skill_kwargs = {"skills": task.skills} if task.skills else {}
                result = await anima.run_cron_task(task.name, task.description, **skill_kwargs)
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
