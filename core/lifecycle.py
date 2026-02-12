from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Callable, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.person import DigitalPerson
from core.schemas import CronTask

logger = logging.getLogger("animaworks.lifecycle")

BroadcastFn = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]

_DAY_MAP = {
    "月曜": "mon",
    "火曜": "tue",
    "水曜": "wed",
    "木曜": "thu",
    "金曜": "fri",
    "土曜": "sat",
    "日曜": "sun",
}


class LifecycleManager:
    """Manages heartbeat and cron for Digital Persons via APScheduler."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
        self.persons: dict[str, DigitalPerson] = {}
        self._ws_broadcast: BroadcastFn | None = None
        self._inbox_watcher_task: asyncio.Task | None = None
        self._pending_triggers: set[str] = set()

    def set_broadcast(self, fn: BroadcastFn) -> None:
        self._ws_broadcast = fn

    def register_person(self, person: DigitalPerson) -> None:
        self.persons[person.name] = person
        self._setup_heartbeat(person)
        self._setup_cron_tasks(person)
        logger.info("Registered '%s' with lifecycle manager", person.name)

    # ── Heartbeat ─────────────────────────────────────────

    def _setup_heartbeat(self, person: DigitalPerson) -> None:
        config = person.memory.read_heartbeat_config()

        interval = 30
        m = re.search(r"(\d+)\s*分", config)
        if m:
            interval = int(m.group(1))

        active_start, active_end = 9, 22
        m = re.search(r"(\d{1,2}):\d{0,2}\s*-\s*(\d{1,2})", config)
        if m:
            active_start, active_end = int(m.group(1)), int(m.group(2))

        self.scheduler.add_job(
            self._heartbeat_wrapper,
            IntervalTrigger(minutes=interval),
            id=f"{person.name}_heartbeat",
            name=f"{person.name} heartbeat",
            args=[person.name, active_start, active_end],
            replace_existing=True,
        )
        logger.info(
            "Heartbeat '%s': every %dmin, active %d:00-%d:00",
            person.name,
            interval,
            active_start,
            active_end,
        )

    async def _heartbeat_wrapper(
        self, name: str, active_start: int, active_end: int
    ) -> None:
        hour = datetime.now().hour
        if not (active_start <= hour < active_end):
            logger.debug("Heartbeat skipped for %s: outside active hours", name)
            return

        person = self.persons.get(name)
        if not person:
            return

        logger.info("Heartbeat: %s", name)
        result = await person.run_heartbeat()
        if self._ws_broadcast:
            await self._ws_broadcast(
                {
                    "type": "person.heartbeat",
                    "data": {"name": name, "result": result.model_dump()},
                }
            )

    # ── Cron ──────────────────────────────────────────────

    def _setup_cron_tasks(self, person: DigitalPerson) -> None:
        config = person.memory.read_cron_config()
        if not config:
            return

        tasks = _parse_cron_md(config)
        for i, task in enumerate(tasks):
            trigger = _parse_schedule(task.schedule)
            if trigger:
                self.scheduler.add_job(
                    self._cron_wrapper,
                    trigger,
                    id=f"{person.name}_cron_{i}",
                    name=f"{person.name}: {task.name}",
                    args=[person.name, task.name, task.description],
                    replace_existing=True,
                )
                logger.info(
                    "Cron '%s': %s (%s)", person.name, task.name, task.schedule
                )

    async def _cron_wrapper(
        self, name: str, task_name: str, description: str
    ) -> None:
        person = self.persons.get(name)
        if not person:
            return

        logger.info("Cron: %s -> %s", name, task_name)
        result = await person.run_cron_task(task_name, description)
        if self._ws_broadcast:
            await self._ws_broadcast(
                {
                    "type": "person.cron",
                    "data": {
                        "name": name,
                        "task": task_name,
                        "result": result.model_dump(),
                    },
                }
            )

    # ── Inbox Watcher ──────────────────────────────────────

    async def _inbox_watcher_loop(self) -> None:
        """Poll inbox dirs every 2s; trigger heartbeat on new messages."""
        logger.info("Inbox watcher started (poll interval: 2s)")
        while True:
            await asyncio.sleep(2)
            for name, person in self.persons.items():
                if name in self._pending_triggers:
                    continue
                if not person.messenger.has_unread():
                    continue
                if person._lock.locked():
                    continue
                self._pending_triggers.add(name)
                asyncio.create_task(
                    self._message_triggered_heartbeat(name)
                )

    async def _message_triggered_heartbeat(self, name: str) -> None:
        person = self.persons.get(name)
        if not person:
            self._pending_triggers.discard(name)
            return
        try:
            logger.info("Message-triggered heartbeat: %s", name)
            result = await person.run_heartbeat()
            if self._ws_broadcast:
                await self._ws_broadcast(
                    {
                        "type": "person.message_heartbeat",
                        "data": {"name": name, "result": result.model_dump()},
                    }
                )
        except Exception:
            logger.exception("Message-triggered heartbeat failed: %s", name)
        finally:
            self._pending_triggers.discard(name)

    # ── Lifecycle ─────────────────────────────────────────

    def start(self) -> None:
        self.scheduler.start()
        self._inbox_watcher_task = asyncio.create_task(
            self._inbox_watcher_loop()
        )
        logger.info("Lifecycle manager started (scheduler + inbox watcher)")

    def shutdown(self) -> None:
        if self._inbox_watcher_task:
            self._inbox_watcher_task.cancel()
        self.scheduler.shutdown(wait=False)
        logger.info("Lifecycle manager stopped")


# ── Parsing helpers ───────────────────────────────────────


def _parse_cron_md(content: str) -> list[CronTask]:
    tasks: list[CronTask] = []
    cur_name = ""
    cur_sched = ""
    cur_desc: list[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if cur_name:
                tasks.append(
                    CronTask(
                        name=cur_name,
                        schedule=cur_sched,
                        description="\n".join(cur_desc).strip(),
                    )
                )
            header = line[3:].strip()
            sm = re.search(r"[（(](.+?)[）)]", header)
            if sm:
                cur_sched = sm.group(1)
                cur_name = header[: header.find("（" if "（" in header else "(")].strip()
            else:
                cur_name = header
                cur_sched = ""
            cur_desc = []
        elif cur_name:
            cur_desc.append(line)

    if cur_name:
        tasks.append(
            CronTask(
                name=cur_name,
                schedule=cur_sched,
                description="\n".join(cur_desc).strip(),
            )
        )
    return tasks


_NTH_DAY_RANGE = {
    1: "1-7",
    2: "8-14",
    3: "15-21",
    4: "22-28",
}


def _parse_schedule(schedule: str) -> CronTrigger | None:
    s = schedule.strip()
    # Remove trailing timezone markers (JST, UTC, etc.)
    s = re.sub(r"\s+[A-Z]{2,4}$", "", s)

    # 毎日 9:00
    m = re.match(r"毎日\s+(\d{1,2}):(\d{2})", s)
    if m:
        return CronTrigger(hour=int(m.group(1)), minute=int(m.group(2)))

    # 平日 9:00
    m = re.match(r"平日\s+(\d{1,2}):(\d{2})", s)
    if m:
        return CronTrigger(
            day_of_week="mon-fri", hour=int(m.group(1)), minute=int(m.group(2))
        )

    # 毎週金曜 17:00
    m = re.match(r"毎週(.+?)\s+(\d{1,2}):(\d{2})", s)
    if m:
        day = _DAY_MAP.get(m.group(1), "fri")
        return CronTrigger(
            day_of_week=day, hour=int(m.group(2)), minute=int(m.group(3))
        )

    # 隔週金曜 17:00
    m = re.match(r"隔週(.+?)\s+(\d{1,2}):(\d{2})", s)
    if m:
        day = _DAY_MAP.get(m.group(1), "fri")
        return CronTrigger(
            day_of_week=day, week="*/2", hour=int(m.group(2)), minute=int(m.group(3))
        )

    # 第2火曜 10:00 (Nth weekday of month)
    m = re.match(r"第(\d)(.+?)\s+(\d{1,2}):(\d{2})", s)
    if m:
        nth = int(m.group(1))
        day = _DAY_MAP.get(m.group(2), "mon")
        day_range = _NTH_DAY_RANGE.get(nth)
        if day_range:
            return CronTrigger(
                day=day_range,
                day_of_week=day,
                hour=int(m.group(3)),
                minute=int(m.group(4)),
            )

    # 毎月1日 9:00
    m = re.match(r"毎月(\d{1,2})日\s+(\d{1,2}):(\d{2})", s)
    if m:
        return CronTrigger(
            day=int(m.group(1)), hour=int(m.group(2)), minute=int(m.group(3))
        )

    # 毎月最終日 18:00
    m = re.match(r"毎月最終日\s+(\d{1,2}):(\d{2})", s)
    if m:
        return CronTrigger(
            day="last", hour=int(m.group(1)), minute=int(m.group(2))
        )

    # Standard cron: */5 * * * *
    if re.match(r"^[\d\*\/\-\,]+(\s+[\d\*\/\-\,]+){4}$", s):
        parts = s.split()
        return CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

    logger.warning("Could not parse schedule: '%s'", s)
    return None
