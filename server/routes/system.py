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


def _parse_cron_jobs(animas_dir: Path, anima_names: list[str]) -> list[dict]:
    """Parse cron.md files from all animas and return job definitions."""
    jobs: list[dict] = []
    for name in anima_names:
        cron_path = animas_dir / name / "cron.md"
        if not cron_path.exists():
            continue
        try:
            content = cron_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Parse markdown sections: ## Title (schedule info)
        # Skip commented-out sections (inside <!-- --> blocks)
        in_comment = False
        current_title = ""
        current_type = ""
        for line in content.splitlines():
            stripped = line.strip()
            if "<!--" in stripped:
                in_comment = True
            if "-->" in stripped:
                in_comment = False
                continue
            if in_comment:
                continue
            if stripped.startswith("## "):
                current_title = stripped[3:].strip()
                current_type = ""
            elif stripped.startswith("type:"):
                current_type = stripped[5:].strip()
            elif current_title and current_type:
                # Extract schedule from title parentheses
                schedule = ""
                m = re.search(r"[（(](.+?)[）)]", current_title)
                if m:
                    schedule = m.group(1)
                jobs.append({
                    "id": f"cron-{name}-{len(jobs)}",
                    "name": current_title,
                    "anima": name,
                    "type": current_type,
                    "schedule": schedule,
                    "next_run": None,
                })
                current_title = ""
                current_type = ""
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
        animas_dir = request.app.state.animas_dir
        current_names = set(request.app.state.anima_names)

        added: list[str] = []
        refreshed: list[str] = []
        removed: list[str] = []

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
    ):
        """Return recent activity events from unified ActivityLogger.

        Reads directly from ``{anima_dir}/activity_log/{date}.jsonl``
        — the single source of truth for all Anima interactions.
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

        # Apply pagination across merged results
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
