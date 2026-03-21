# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""APScheduler management for heartbeat and cron tasks.

Handles registration, execution, overlap prevention, and hot-reload
of heartbeat and cron schedules for a single Anima process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import zlib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.config.models import ActivityScheduleEntry, load_config, save_config
from core.i18n import t
from core.schedule_parser import parse_cron_md, parse_heartbeat_config, parse_schedule
from core.schemas import CronTask
from core.time_utils import get_app_timezone, now_local

_INDENTED_SCHEDULE_RE = re.compile(r"^\s+schedule:", re.MULTILINE)
_HEALTH_CHECK_HOURS = 3

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
        self._cron_md_mtime: float = 0.0
        self._heartbeat_md_mtime: float = 0.0
        self._last_schedule_level: int | None = None

        # Polling-based heartbeat state (used when effective_interval > 60)
        self._hb_effective_interval: int = 0
        self._hb_active_start: int | None = None
        self._hb_active_end: int | None = None
        self._hb_first_check_offset: int = 0
        self._hb_first_check_done: bool = False

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
            self.scheduler = AsyncIOScheduler(timezone=get_app_timezone())
            self._setup_heartbeat()
            self._setup_cron_tasks()
            self._setup_cron_health_check()
            self._setup_activity_schedule()
            self.scheduler.start()

            # Apply the correct activity level for the current time on startup
            self._apply_current_schedule_level()

            # Wire up hot-reload callback
            self._anima.set_on_schedule_changed(self.reload_schedule)
            self._record_schedule_mtimes()

            job_count = len(self.scheduler.get_jobs())
            logger.info(
                "Scheduler started for %s: %d jobs registered",
                self._anima_name,
                job_count,
            )
        except Exception:
            logger.exception("Failed to setup scheduler for %s", self._anima_name)
            self.scheduler = None

    def _read_per_anima_interval(self, app_config: Any) -> int:
        """Read heartbeat_interval_minutes from status.json, fallback to global config."""
        try:
            status_path = self._anima_dir / "status.json"
            if status_path.is_file():
                data = json.loads(status_path.read_text(encoding="utf-8"))
                val = data.get("heartbeat_interval_minutes")
                if isinstance(val, (int, float)) and 1 <= val <= 1440:
                    return int(val)
        except (json.JSONDecodeError, OSError, ValueError):
            logger.debug("Failed to read heartbeat_interval_minutes from %s", self._anima_dir)
        return app_config.heartbeat.interval_minutes

    def _setup_heartbeat(self) -> None:
        """Register heartbeat job from heartbeat.md + config.json + activity_level."""
        if not self._anima or not self.scheduler:
            return

        hb_content = self._anima.memory.read_heartbeat_config()
        if not hb_content:
            return

        active_start, active_end = parse_heartbeat_config(hb_content)

        app_config = load_config()
        base_interval = self._read_per_anima_interval(app_config)
        activity_pct = max(10, min(400, app_config.activity_level))
        effective_interval = base_interval / (activity_pct / 100.0)
        effective_interval = max(5.0, effective_interval)
        interval = round(effective_interval)

        # Fixed offset: crc32(anima_name) % 10 → deterministic 0-9 min spread
        offset = zlib.crc32(self._anima_name.encode()) % 10

        # Determine active hours
        if active_start is not None and active_end is not None:
            log_active = f"active {active_start}:00-{active_end}:00"
        else:
            log_active = "24h"

        # CronTrigger minute-slot approach only works when interval divides
        # evenly into 60; otherwise cross-hour gaps become shorter than the
        # intended interval (e.g. interval=43 → slots "9,52" → 43min then
        # 17min gap).  Fall through to polling for non-divisor intervals.
        if interval <= 60 and 60 % interval == 0:
            hour_spec = (
                f"{active_start}-{active_end - 1}" if active_start is not None and active_end is not None else "*"
            )
            slots = []
            m = offset
            while m < 60:
                slots.append(str(m))
                m += interval
            minute_spec = ",".join(slots)

            self.scheduler.add_job(
                self.heartbeat_tick,
                CronTrigger(
                    minute=minute_spec,
                    hour=hour_spec,
                ),
                id=f"{self._anima_name}_heartbeat",
                name=f"{self._anima_name} heartbeat",
                replace_existing=True,
                misfire_grace_time=300,
                max_instances=1,
            )
            logger.info(
                "Heartbeat registered: %s minute=%s (offset=%d, interval=%dmin, activity=%d%%), %s",
                self._anima_name,
                minute_spec,
                offset,
                interval,
                activity_pct,
                log_active,
            )
        else:
            # Polling: check every minute, fire when interval elapsed.
            # Replaces the old IntervalTrigger path which had end_date bugs.
            self._hb_effective_interval = interval
            self._hb_active_start = active_start
            self._hb_active_end = active_end
            self._hb_first_check_offset = offset
            self._hb_first_check_done = False

            self.scheduler.add_job(
                self._heartbeat_check,
                CronTrigger(minute="*"),
                id=f"{self._anima_name}_heartbeat",
                name=f"{self._anima_name} heartbeat",
                replace_existing=True,
                misfire_grace_time=120,
                max_instances=1,
            )
            logger.info(
                "Heartbeat registered (polling): %s every %dmin (offset=%d, activity=%d%%), %s",
                self._anima_name,
                interval,
                offset,
                activity_pct,
                log_active,
            )

    def reschedule_heartbeat(self) -> None:
        """Reschedule heartbeat job with current config (called on activity_level change)."""
        if not self.scheduler:
            return
        job_id = f"{self._anima_name}_heartbeat"
        try:
            self.scheduler.remove_job(job_id)
        except KeyError:
            pass
        self._setup_heartbeat()
        logger.info("Heartbeat rescheduled for %s", self._anima_name)

    # ── Activity Schedule ─────────────────────────────────────

    @staticmethod
    def resolve_scheduled_level(
        schedule: list[ActivityScheduleEntry],
        now_hhmm: str,
    ) -> int | None:
        """Return the activity level for *now_hhmm* (``"HH:MM"``), or None."""
        for entry in schedule:
            if _time_in_range(entry.start, entry.end, now_hhmm):
                return entry.level
        return None

    def _setup_activity_schedule(self) -> None:
        """Register a 1-minute job that checks activity_schedule boundaries."""
        if not self.scheduler:
            return
        app_config = load_config()
        if not app_config.activity_schedule:
            return

        self.scheduler.add_job(
            self._activity_schedule_tick,
            CronTrigger(minute="*"),
            id=f"{self._anima_name}_activity_schedule",
            name=f"{self._anima_name} activity schedule",
            replace_existing=True,
            misfire_grace_time=120,
            max_instances=1,
        )
        logger.info(
            "Activity schedule registered for %s: %d entries",
            self._anima_name,
            len(app_config.activity_schedule),
        )

    def _apply_current_schedule_level(self) -> None:
        """Apply the correct activity level for the current time at startup."""
        app_config = load_config()
        if not app_config.activity_schedule:
            return

        now_hhmm = now_local().strftime("%H:%M")
        target = self.resolve_scheduled_level(app_config.activity_schedule, now_hhmm)
        if target is not None and target != app_config.activity_level:
            app_config.activity_level = target
            save_config(app_config)
            self._last_schedule_level = target
            self.reschedule_heartbeat()
            logger.info(
                "Activity schedule startup: %s set level to %d%% (time=%s)",
                self._anima_name,
                target,
                now_hhmm,
            )
        else:
            self._last_schedule_level = app_config.activity_level

    async def _activity_schedule_tick(self) -> None:
        """Check current time against activity_schedule and switch level if needed."""
        app_config = load_config()
        if not app_config.activity_schedule:
            return

        now_hhmm = now_local().strftime("%H:%M")
        target = self.resolve_scheduled_level(app_config.activity_schedule, now_hhmm)
        if target is None:
            return

        if target != self._last_schedule_level:
            app_config.activity_level = target
            save_config(app_config)
            self._last_schedule_level = target
            self.reschedule_heartbeat()
            logger.info(
                "Activity schedule switch: %s → %d%% (time=%s)",
                self._anima_name,
                target,
                now_hhmm,
            )

    def reload_activity_schedule(self) -> None:
        """Reload the activity schedule job (called after schedule config change)."""
        if not self.scheduler:
            return
        job_id = f"{self._anima_name}_activity_schedule"
        try:
            self.scheduler.remove_job(job_id)
        except KeyError:
            pass
        self._setup_activity_schedule()
        self._apply_current_schedule_level()

    def _setup_cron_tasks(self) -> None:
        """Register cron jobs from cron.md."""
        if not self._anima or not self.scheduler:
            return

        config = self._anima.memory.read_cron_config()
        if not config:
            return

        tasks = parse_cron_md(config)
        registered = 0
        for i, task in enumerate(tasks):
            trigger = parse_schedule(task.schedule)
            if not trigger:
                logger.warning(
                    "Could not parse schedule for cron task '%s': '%s'",
                    task.name,
                    task.schedule,
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
            registered += 1
            logger.info(
                "Cron registered: %s -> %s (%s) [%s]",
                self._anima_name,
                task.name,
                task.schedule,
                task.type,
            )

        self._check_cron_parse_health(config, tasks, registered)

    # ── Cron Health Check ──────────────────────────────────────────

    def _check_cron_parse_health(
        self,
        raw_config: str,
        tasks: list[CronTask],
        registered: int,
    ) -> None:
        """Validate cron parse results and notify the Anima if unhealthy.

        Called at the end of ``_setup_cron_tasks`` (both initial setup and
        hot-reload).  Writes a markdown file to ``background_notifications/``
        which is drained into the next heartbeat or cron context.
        """
        messages: list[str] = []

        if tasks and registered == 0:
            messages.append(t("scheduler.cron_health_no_valid_schedule", task_count=len(tasks)))

        if _INDENTED_SCHEDULE_RE.search(raw_config):
            messages.append(t("scheduler.cron_health_indented_schedule"))

        if not tasks and "schedule:" in raw_config:
            messages.append(t("scheduler.cron_health_unrecognized_schedule"))

        if messages:
            self._write_cron_health_notification("\n\n".join(messages))

    def _setup_cron_health_check(self) -> None:
        """Register a periodic job that checks cron execution health."""
        if not self.scheduler or not self._anima:
            return

        self.scheduler.add_job(
            self._cron_health_tick,
            CronTrigger(minute=0, hour=f"*/{_HEALTH_CHECK_HOURS}"),
            id=f"{self._anima_name}_cron_health",
            name=f"{self._anima_name} cron health check",
            replace_existing=True,
            misfire_grace_time=600,
            max_instances=1,
        )

    async def _cron_health_tick(self) -> None:
        """Compare registered cron jobs against actual execution count."""
        if not self._anima or not self.scheduler:
            return

        try:
            cron_job_prefix = f"{self._anima_name}_cron_"
            cron_jobs = [
                j
                for j in self.scheduler.get_jobs()
                if j.id.startswith(cron_job_prefix) and not j.id.endswith("_health")
            ]
            if not cron_jobs:
                return

            entries = self._anima._activity._load_entries(
                hours=_HEALTH_CHECK_HOURS,
                types=["cron_executed"],
            )

            if len(entries) == 0:
                self._write_cron_health_notification(
                    t(
                        "scheduler.cron_health_no_execution",
                        job_count=len(cron_jobs),
                        hours=_HEALTH_CHECK_HOURS,
                    )
                )
        except Exception:
            logger.debug(
                "Cron health tick failed for %s",
                self._anima_name,
                exc_info=True,
            )

    def _write_cron_health_notification(self, message: str) -> None:
        """Write a cron health warning to ``background_notifications/``."""
        try:
            notif_dir = self._anima_dir / "state" / "background_notifications"
            notif_dir.mkdir(parents=True, exist_ok=True)
            ts = now_local().strftime("%Y%m%d_%H%M%S")
            notif_path = notif_dir / f"cron_health_{ts}.md"
            content = f"# {t('scheduler.cron_health_title')}\n\n{message}\n"
            notif_path.write_text(content, encoding="utf-8")
            logger.warning(
                "[%s] Cron health warning written: %s",
                self._anima_name,
                notif_path.name,
            )
        except Exception:
            logger.debug(
                "Failed to write cron health notification for %s",
                self._anima_name,
                exc_info=True,
            )

    # ── Polling-based Heartbeat ─────────────────────────────────

    def _in_active_hours(self, now: datetime) -> bool:
        """Return True if *now* falls within the configured active hours."""
        if self._hb_active_start is None or self._hb_active_end is None:
            return True
        hour = now.hour
        start, end = self._hb_active_start, self._hb_active_end
        if start < end:
            return start <= hour < end
        # Midnight-crossing (e.g. 22:00 – 06:00)
        return hour >= start or hour < end

    def _get_last_heartbeat_ts(self) -> datetime | None:
        """Return the timestamp of the most recent heartbeat_start from activity_log."""
        try:
            entries = self._anima._activity.recent(
                days=2,
                types=["heartbeat_start"],
                limit=1,
            )
        except Exception:
            logger.debug("Failed to read activity_log for last heartbeat", exc_info=True)
            return None
        if not entries:
            return None
        try:
            return datetime.fromisoformat(entries[-1].ts)
        except (ValueError, TypeError):
            return None

    async def _heartbeat_check(self) -> None:
        """Polling-based heartbeat trigger for intervals > 60 minutes.

        Registered as a CronTrigger(minute="*") job.  Each minute it checks
        whether enough time has elapsed since the last heartbeat (read from
        the activity_log) and whether the current time is within active hours.
        """
        if not self._anima:
            return

        now = now_local()

        if not self._in_active_hours(now):
            return

        last_hb = self._get_last_heartbeat_ts()
        if last_hb is not None:
            if last_hb.tzinfo is None:
                last_hb = last_hb.replace(tzinfo=now.tzinfo)
            elapsed_min = (now - last_hb).total_seconds() / 60.0
            required = self._hb_effective_interval
            if not self._hb_first_check_done:
                required += self._hb_first_check_offset
            if elapsed_min < required:
                return
        else:
            # No heartbeat_start in activity_log (fresh install).
            # Apply offset delay before first heartbeat to spread across Animas.
            if not self._hb_first_check_done and self._hb_first_check_offset > 0:
                # On the very first call _hb_first_check_done is False.
                # Mark it done and skip this minute — the offset will be
                # consumed on subsequent calls via the `required` increase above.
                # For simplicity, just allow immediate fire for fresh installs.
                pass

        self._hb_first_check_done = True
        await self.heartbeat_tick()

    # ── Tick Handlers ────────────────────────────────────────────

    async def heartbeat_tick(self) -> None:
        """Execute a scheduled heartbeat."""
        if not self._anima:
            return
        # Detect schedule file changes (Mode S Write/Edit bypass)
        self._check_schedule_freshness()
        if self._heartbeat_running:
            logger.info("Scheduled heartbeat SKIPPED (already running): %s", self._anima_name)
            return
        self._heartbeat_running = True
        try:
            logger.info("Scheduled heartbeat: %s", self._anima_name)
            result = await self._anima.run_heartbeat()
            # Notify parent for WebSocket broadcast
            self._emit_event(
                "anima.heartbeat",
                {
                    "name": self._anima_name,
                    "result": result.model_dump(),
                },
            )
        except Exception:
            logger.exception("Scheduled heartbeat failed: %s", self._anima_name)
        finally:
            self._heartbeat_running = False

    async def cron_tick(self, task: CronTask) -> None:
        """Execute a scheduled cron task."""
        if not self._anima:
            return

        # Detect schedule file changes and skip stale tasks
        if self._check_schedule_freshness():
            logger.info(
                "Skipping stale cron '%s' for %s (schedule reloaded)",
                task.name,
                self._anima_name,
            )
            return

        if task.name in self._cron_running:
            logger.info(
                "Scheduled cron SKIPPED (already running): %s -> %s",
                self._anima_name,
                task.name,
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
                self._emit_event(
                    "anima.cron",
                    {
                        "name": self._anima_name,
                        "task": task.name,
                        "task_type": "llm",
                        "result": result.model_dump(),
                    },
                )
            elif task.type == "command":
                result = await self._anima.run_cron_command(
                    task.name,
                    command=task.command,
                    tool=task.tool,
                    args=task.args,
                )
                self._emit_event(
                    "anima.cron",
                    {
                        "name": self._anima_name,
                        "task": task.name,
                        "task_type": "command",
                        "result": result,
                    },
                )
                # If command produced non-empty output, run a follow-up
                # cron LLM session so the Anima can review and act on the
                # results with full background context (heartbeat-equivalent).
                stdout = result.get("stdout", "").strip()
                if stdout and result.get("exit_code", 0) == 0:
                    # trigger_heartbeat=False means no follow-up analysis
                    if not task.trigger_heartbeat:
                        logger.info(
                            "Cron command '%s' trigger_heartbeat=False, skipping cron LLM for %s",
                            task.name,
                            self._anima_name,
                        )
                        return

                    # skip_pattern: if stdout matches, suppress follow-up
                    if task.skip_pattern:
                        try:
                            if re.search(task.skip_pattern, stdout):
                                logger.info(
                                    "Cron command '%s' output matched skip_pattern, suppressing cron LLM for %s",
                                    task.name,
                                    self._anima_name,
                                )
                                return
                        except re.error as e:
                            logger.warning(
                                "Invalid skip_pattern '%s' for task '%s': %s — continuing without skip",
                                task.skip_pattern,
                                task.name,
                                e,
                            )

                    logger.info(
                        "Cron command '%s' produced output, running cron LLM for %s",
                        task.name,
                        self._anima_name,
                    )
                    await self._anima.run_cron_task(
                        task.name,
                        task.description or f"cron.mdの「{task.name}」の指示に従って処理してください。",
                        command_output=stdout,
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
        self._setup_cron_health_check()
        self._setup_activity_schedule()
        self._record_schedule_mtimes()

        new_jobs = [j.id for j in self.scheduler.get_jobs()]
        logger.info(
            "Schedule reloaded for %s: removed=%d, new_jobs=%s",
            self._anima_name,
            removed,
            new_jobs,
        )
        return {"reloaded": name, "removed": removed, "new_jobs": new_jobs}

    # ── Schedule Freshness ─────────────────────────────────────

    def _record_schedule_mtimes(self) -> None:
        """Snapshot cron.md and heartbeat.md mtimes for later freshness checks."""
        cron_path = self._anima_dir / "cron.md"
        hb_path = self._anima_dir / "heartbeat.md"
        try:
            self._cron_md_mtime = cron_path.stat().st_mtime if cron_path.is_file() else 0.0
        except OSError:
            self._cron_md_mtime = 0.0
        try:
            self._heartbeat_md_mtime = hb_path.stat().st_mtime if hb_path.is_file() else 0.0
        except OSError:
            self._heartbeat_md_mtime = 0.0

    def _check_schedule_freshness(self) -> bool:
        """Check if cron.md or heartbeat.md changed since last setup.

        If a change is detected, reloads the schedule and returns True.
        Returns False when no change is detected.
        """
        cron_path = self._anima_dir / "cron.md"
        hb_path = self._anima_dir / "heartbeat.md"
        try:
            cron_mtime = cron_path.stat().st_mtime if cron_path.is_file() else 0.0
        except OSError:
            cron_mtime = 0.0
        try:
            hb_mtime = hb_path.stat().st_mtime if hb_path.is_file() else 0.0
        except OSError:
            hb_mtime = 0.0

        if cron_mtime != self._cron_md_mtime or hb_mtime != self._heartbeat_md_mtime:
            logger.info(
                "Schedule file changed for %s (cron mtime %.0f->%.0f, hb mtime %.0f->%.0f), reloading",
                self._anima_name,
                self._cron_md_mtime,
                cron_mtime,
                self._heartbeat_md_mtime,
                hb_mtime,
            )
            self.reload_schedule(self._anima_name)
            return True
        return False

    # ── Cleanup ──────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop the scheduler."""
        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:
                logger.debug(
                    "Scheduler shutdown failed for %s (may not have been started)", self._anima_name, exc_info=True
                )
            logger.info("Scheduler stopped for %s", self._anima_name)


# ── Module helpers ────────────────────────────────────────────────────────


def _time_in_range(start: str, end: str, now: str) -> bool:
    """Check whether *now* (``HH:MM``) falls within [*start*, *end*).

    Handles midnight-crossing ranges (e.g. ``22:00``–``08:00``).
    """
    if start <= end:
        return start <= now < end
    # Wraps past midnight
    return now >= start or now < end
