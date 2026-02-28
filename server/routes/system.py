from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.schedule_parser import parse_cron_md, parse_schedule
from core.time_utils import now_jst

logger = logging.getLogger("animaworks.routes.system")

# ── Frontend Log Setup ────────────────────────────────────────
# Dedicated logger for frontend logs — writes to daily JSONL files.

_frontend_logger: logging.Logger | None = None
_frontend_log_dir: Path | None = None


def _get_frontend_logger() -> logging.Logger:
    """Lazily initialise and return the frontend file logger."""
    global _frontend_logger, _frontend_log_dir

    if _frontend_logger is not None:
        return _frontend_logger

    from core.paths import get_data_dir

    _frontend_log_dir = get_data_dir() / "logs" / "frontend"
    _frontend_log_dir.mkdir(parents=True, exist_ok=True)

    _frontend_logger = logging.getLogger("animaworks.frontend")
    _frontend_logger.setLevel(logging.DEBUG)
    _frontend_logger.propagate = False  # Don't forward to root logger

    # Fixed base filename; TimedRotatingFileHandler appends date suffix on rotation
    # e.g. frontend.jsonl -> frontend.jsonl.20260217
    log_path = _frontend_log_dir / "frontend.jsonl"

    handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=30,  # 30 days retention
        encoding="utf-8",
        utc=False,
    )
    handler.suffix = "%Y%m%d"
    # Raw passthrough: message is already a JSON string
    handler.setFormatter(logging.Formatter("%(message)s"))
    _frontend_logger.addHandler(handler)

    return _frontend_logger


def _collect_cron_last_runs(anima_dir: Path, task_names: set[str]) -> dict[str, str]:
    """Collect latest run timestamp per task from state/cron_logs JSONL files."""
    if not task_names:
        return {}

    log_dir = anima_dir / "state" / "cron_logs"
    if not log_dir.exists():
        return {}

    found: dict[str, str] = {}
    log_files = sorted(log_dir.glob("*.jsonl"), reverse=True)
    for log_file in log_files:
        try:
            lines = log_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            task = str(entry.get("task", "")).strip()
            ts = str(entry.get("timestamp", "")).strip()
            if not task or not ts:
                continue
            if task in task_names and task not in found:
                found[task] = ts

        if len(found) == len(task_names):
            break

    return found


def _parse_cron_jobs(animas_dir: Path, anima_names: list[str]) -> list[dict]:
    """Parse cron.md files and enrich with schedule/next/last execution data."""
    jobs: list[dict] = []
    for name in anima_names:
        cron_path = animas_dir / name / "cron.md"
        if not cron_path.exists():
            continue
        try:
            content = cron_path.read_text(encoding="utf-8")
        except OSError:
            continue
        parsed_tasks = parse_cron_md(content)
        task_names = {task.name for task in parsed_tasks if task.name}
        last_runs = _collect_cron_last_runs(animas_dir / name, task_names)
        now = now_jst()
        for idx, task in enumerate(parsed_tasks):
            trigger = parse_schedule(task.schedule)
            next_run_dt = (
                trigger.get_next_fire_time(None, now)
                if trigger is not None
                else None
            )
            next_run = next_run_dt.isoformat() if next_run_dt else None
            schedule = task.schedule.strip() if task.schedule else ""
            if not schedule:
                m = re.search(r"[（(](.+?)[）)]", task.name)
                if m:
                    schedule = m.group(1).strip()

            jobs.append({
                "id": f"cron-{name}-{idx}",
                "name": task.name,
                "anima": name,
                "type": task.type,
                "schedule": schedule,
                "last_run": last_runs.get(task.name),
                "next_run": next_run,
            })
    return jobs


def create_system_router() -> APIRouter:
    router = APIRouter()

    @router.get("/shared/users")
    async def list_shared_users(request: Request):
        """List registered user names from shared/users/."""
        users_dir = request.app.state.shared_dir / "users"
        if not users_dir.is_dir():
            return []
        return [d.name for d in sorted(users_dir.iterdir()) if d.is_dir()]

    @router.get("/system/status")
    async def system_status(request: Request):
        supervisor = request.app.state.supervisor
        anima_names = request.app.state.anima_names

        # Get all process statuses
        process_statuses = supervisor.get_all_status()

        return {
            "animas": len(anima_names),
            "processes": process_statuses,
            "scheduler_running": supervisor.is_scheduler_running(),
        }

    @router.post("/system/reload")
    async def reload_animas(request: Request):
        """Full sync: add new animas, refresh existing, remove deleted."""
        supervisor = request.app.state.supervisor
        stream_registry = request.app.state.stream_registry
        animas_dir = request.app.state.animas_dir
        current_names = set(request.app.state.anima_names)

        added: list[str] = []
        refreshed: list[str] = []
        removed: list[str] = []
        skipped_busy: list[dict[str, object]] = []

        async def _busy_state(anima_name: str) -> dict[str, object]:
            """Return runtime busy state used to decide reload skip."""
            result: dict[str, object] = {
                "streaming": False,
                "status": "",
                "current_task": "",
                "reasons": [],
            }

            try:
                active_stream = stream_registry.get_active(anima_name)
                if active_stream is not None and active_stream.status == "streaming":
                    result["streaming"] = True
                    cast_reasons = result["reasons"]
                    if isinstance(cast_reasons, list):
                        cast_reasons.append("streaming")
            except Exception:
                logger.debug("Failed to read active stream for %s", anima_name, exc_info=True)

            try:
                status_resp = await supervisor.send_request(
                    anima_name, method="get_status", params={}, timeout=3.0,
                )
                status_text = str(status_resp.get("status", "") or "")
                current_task = str(status_resp.get("current_task", "") or "")
                result["status"] = status_text
                result["current_task"] = current_task
                if status_text and status_text != "idle":
                    cast_reasons = result["reasons"]
                    if isinstance(cast_reasons, list):
                        cast_reasons.append(f"status:{status_text}")
                elif current_task:
                    cast_reasons = result["reasons"]
                    if isinstance(cast_reasons, list):
                        cast_reasons.append("task_running")
            except Exception:
                # If status probe fails (e.g. process transiently unavailable),
                # keep reload behavior unchanged rather than hard-failing.
                logger.debug("Failed to probe runtime status for %s", anima_name, exc_info=True)

            return result

        # Snapshot busy state at reload request time.
        # If an anima is streaming/working at this moment, skip forced restart
        # for this reload cycle even if it becomes idle later.
        busy_snapshot: dict[str, dict[str, object]] = {}
        for name in current_names:
            busy_snapshot[name] = await _busy_state(name)

        # Discover current animas on disk
        from core.supervisor.manager import ProcessSupervisor

        on_disk: set[str] = set()
        if animas_dir.exists():
            for anima_dir in sorted(animas_dir.iterdir()):
                if not anima_dir.is_dir():
                    continue
                if not (anima_dir / "identity.md").exists():
                    continue
                name = anima_dir.name

                # Respect status.json: skip disabled animas
                if not ProcessSupervisor.read_anima_enabled(anima_dir):
                    on_disk.add(name)
                    continue

                on_disk.add(name)

                if name not in current_names:
                    # New anima - start process
                    await supervisor.start_anima(name)
                    request.app.state.anima_names.append(name)
                    added.append(name)
                    logger.info("Hot-loaded anima: %s", name)
                else:
                    busy = busy_snapshot.get(name, {})
                    reasons = busy.get("reasons") if isinstance(busy.get("reasons"), list) else []
                    if reasons:
                        skipped_busy.append({
                            "name": name,
                            "reasons": reasons,
                            "status": busy.get("status", ""),
                            "current_task": busy.get("current_task", ""),
                        })
                        logger.info(
                            "Skipped reload for busy anima: %s (reasons=%s status=%s task=%s)",
                            name, ",".join(str(r) for r in reasons),
                            busy.get("status", ""), busy.get("current_task", ""),
                        )
                        continue
                    # Existing anima — restart to pick up file changes
                    await supervisor.restart_anima(name)
                    refreshed.append(name)
                    logger.info("Refreshed anima: %s", name)

        # Remove animas whose directories no longer exist
        for name in list(current_names):
            if name not in on_disk:
                await supervisor.stop_anima(name)
                request.app.state.anima_names.remove(name)
                removed.append(name)
                logger.info("Unloaded anima: %s", name)

        return {
            "added": added,
            "refreshed": refreshed,
            "skipped_busy": skipped_busy,
            "removed": removed,
            "total": len(request.app.state.anima_names),
        }

    # ── Connections ─────────────────────────────────────────

    @router.get("/system/connections")
    async def system_connections(request: Request):
        """Return WebSocket and process connection info."""
        ws_manager = request.app.state.ws_manager
        supervisor = request.app.state.supervisor

        return {
            "websocket": {
                "connected_clients": (
                    len(ws_manager.active_connections)
                    if hasattr(ws_manager, "active_connections")
                    else 0
                ),
            },
            "processes": {
                name: supervisor.get_process_status(name)
                for name in request.app.state.anima_names
            },
        }

    # ── Scheduler ──────────────────────────────────────────

    @router.get("/system/scheduler")
    async def system_scheduler(request: Request):
        """Return scheduler status and job information."""
        supervisor = request.app.state.supervisor
        animas_dir = request.app.state.animas_dir
        anima_names = request.app.state.anima_names

        # Get configured jobs from cron.md files (for display)
        jobs = _parse_cron_jobs(animas_dir, anima_names)

        # Get system scheduler jobs
        system_jobs = []
        if supervisor.scheduler:
            for job in supervisor.scheduler.get_jobs():
                system_jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "anima": "system",
                    "type": "consolidation",
                    "schedule": str(job.trigger),
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                })

        return {
            "running": supervisor.is_scheduler_running(),
            "system_jobs": system_jobs,
            "anima_jobs": jobs,
        }

    # ── Activity ───────────────────────────────────────────

    @router.get("/activity/recent")
    async def get_recent_activity(
        request: Request,
        hours: int = 48,
        anima: str | None = None,
        offset: int = 0,
        limit: int = 200,
        event_type: str | None = None,
        grouped: bool = False,
        group_limit: int = 50,
        group_offset: int = 0,
    ):
        """Return recent activity events from unified ActivityLogger.

        Reads directly from ``{anima_dir}/activity_log/{date}.jsonl``
        — the single source of truth for all Anima interactions.

        When ``grouped=true``, returns trigger-based groups instead of
        flat events.  Pagination switches to group-based (group_limit,
        group_offset).
        """
        from core.memory.activity import ActivityLogger

        animas_dir = request.app.state.animas_dir
        anima_names = request.app.state.anima_names

        # Parse event_type filter (comma-separated)
        types: list[str] | None = None
        if event_type:
            types = [t.strip() for t in event_type.split(",") if t.strip()]

        target_names = (
            [anima] if anima and anima in anima_names else list(anima_names)
        )

        # Collect entries from all target Animas
        all_entries = []
        for name in target_names:
            anima_dir = animas_dir / name
            if not anima_dir.exists():
                continue
            al = ActivityLogger(anima_dir)
            page = al.recent_page(
                hours=hours,
                limit=0,  # load all to merge across animas
                types=types,
            )
            for entry in page.entries:
                entry._anima_name = name
            all_entries.extend(page.entries)

        # Sort all entries by ts descending (newest first)
        all_entries.sort(key=lambda e: e.ts, reverse=True)

        if grouped:
            # Chronological order for grouping (oldest first)
            chrono = list(reversed(all_entries))
            all_groups = ActivityLogger.group_by_trigger(chrono)
            # Reverse to newest-first for display
            all_groups.reverse()

            group_limit = max(1, min(group_limit, 200))
            group_offset = max(0, group_offset)
            total_groups = len(all_groups)
            total_events = len(all_entries)
            page_groups = all_groups[group_offset:group_offset + group_limit]

            return {
                "groups": page_groups,
                "total_groups": total_groups,
                "total_events": total_events,
                "group_offset": group_offset,
                "group_limit": group_limit,
                "has_more": (group_offset + group_limit) < total_groups,
            }

        # Flat (default) — backward compatible
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        total = len(all_entries)
        page_entries = all_entries[offset:offset + limit]

        return {
            "events": [e.to_api_dict() for e in page_entries],
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total,
        }

    # ── Frontend Log Ingestion ────────────────────────────────

    @router.post("/system/frontend-logs")
    async def receive_frontend_logs(request: Request):
        """Receive a batch of frontend log entries and write to daily JSONL.

        Parses request body directly via ``json.loads`` to accept both
        ``application/json`` and ``text/plain`` Content-Type headers
        (the latter may be sent by ``navigator.sendBeacon`` in some browsers).
        """
        raw = await request.body()
        try:
            entries = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Frontend log: invalid JSON body (%d bytes)", len(raw),
            )
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        if not isinstance(entries, list):
            return JSONResponse({"error": "Expected a JSON array"}, status_code=400)

        if len(entries) > 500:
            return JSONResponse(
                {"error": f"Too many entries: {len(entries)}. Maximum is 500."},
                status_code=400,
            )

        fe_logger = _get_frontend_logger()
        for entry in entries:
            if isinstance(entry, dict):
                fe_logger.info(json.dumps(entry, ensure_ascii=False))
                # Echo to server console for debugging
                logger.debug(
                    "[frontend:%s] %s %s",
                    entry.get("module", "?"),
                    entry.get("level", "?"),
                    entry.get("msg", ""),
                )

        logger.info("Frontend logs received: %d entries", len(entries))
        return {"ok": True, "count": len(entries)}

    # ── Frontend Log Viewer ─────────────────────────────────

    @router.get("/system/frontend-logs")
    async def view_frontend_logs(
        request: Request,
        date: str | None = None,
        level: str | None = None,
        module: str | None = None,
        limit: int = 100,
    ):
        """Read frontend logs from JSONL files with optional filters.

        File layout (TimedRotatingFileHandler with fixed base name):
          - Active file: ``frontend.jsonl`` (today's logs)
          - Rotated files: ``frontend.jsonl.YYYYMMDD`` (past days)
          - Legacy files: ``YYYYMMDD.jsonl`` (pre-migration)

        Query params:
            date: YYYYMMDD (defaults to today)
            level: Filter by log level (DEBUG, INFO, WARN, ERROR)
            module: Filter by module name
            limit: Max entries to return (default 100, max 1000)
        """
        from core.paths import get_data_dir

        limit = max(1, min(limit, 1000))
        target_date = date or now_jst().strftime("%Y%m%d")
        today = now_jst().strftime("%Y%m%d")
        log_dir = get_data_dir() / "logs" / "frontend"

        # Determine which file to read:
        #   today → active file (frontend.jsonl)
        #   past  → rotated file (frontend.jsonl.YYYYMMDD) or legacy (YYYYMMDD.jsonl)
        if target_date == today:
            log_path = log_dir / "frontend.jsonl"
        else:
            log_path = log_dir / f"frontend.jsonl.{target_date}"
            if not log_path.exists():
                # Fallback: legacy date-encoded filename
                log_path = log_dir / f"{target_date}.jsonl"

        if not log_path.exists():
            return {"entries": [], "date": target_date, "total": 0}

        entries: list[dict] = []
        try:
            for line in log_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if level and entry.get("level") != level.upper():
                    continue
                if module and entry.get("module") != module:
                    continue

                entries.append(entry)
        except OSError:
            return JSONResponse(
                {"error": f"Failed to read log file for {target_date}"},
                status_code=500,
            )

        # Return most recent entries first
        entries.reverse()
        total = len(entries)
        entries = entries[:limit]

        return {"entries": entries, "date": target_date, "total": total, "limit": limit}

    # ── Dynamic Log Level ───────────────────────────────────

    @router.get("/system/log-level")
    async def get_log_level(request: Request):
        """Return the current root log level."""
        root = logging.getLogger()
        return {"level": logging.getLevelName(root.level)}

    @router.post("/system/log-level")
    async def set_log_level(request: Request):
        """Change the log level at runtime (no restart required).

        Body: {"level": "DEBUG"} or {"level": "DEBUG", "logger_name": "animaworks.websocket"}
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        new_level = body.get("level", "").upper()
        if new_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            return JSONResponse(
                {"error": f"Invalid level: {body.get('level')}. Use DEBUG/INFO/WARNING/ERROR/CRITICAL"},
                status_code=400,
            )

        logger_name = body.get("logger_name")
        if logger_name:
            target = logging.getLogger(logger_name)
            target.setLevel(getattr(logging, new_level))
            logger.info("Log level changed: %s -> %s", logger_name, new_level)
            return {"logger": logger_name, "level": new_level}
        else:
            root = logging.getLogger()
            root.setLevel(getattr(logging, new_level))
            logger.info("Root log level changed to %s", new_level)
            return {"logger": "root", "level": new_level}

    # ── Health Check ────────────────────────────────────────

    @router.get("/system/health")
    async def health_check():
        """Simple health check endpoint."""
        return {"status": "ok"}

    return router
