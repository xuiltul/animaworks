from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging

from fastapi import APIRouter, Request

logger = logging.getLogger("animaworks.routes.system")


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
        person_names = request.app.state.person_names

        # Get all process statuses
        process_statuses = supervisor.get_all_status()

        return {
            "persons": len(person_names),
            "processes": process_statuses,
        }

    @router.post("/system/reload")
    async def reload_persons(request: Request):
        """Full sync: add new persons, refresh existing, remove deleted."""
        supervisor = request.app.state.supervisor
        persons_dir = request.app.state.persons_dir
        current_names = set(request.app.state.person_names)

        added: list[str] = []
        refreshed: list[str] = []
        removed: list[str] = []

        # Discover current persons on disk
        on_disk: set[str] = set()
        if persons_dir.exists():
            for person_dir in sorted(persons_dir.iterdir()):
                if not person_dir.is_dir():
                    continue
                if not (person_dir / "identity.md").exists():
                    continue
                name = person_dir.name
                on_disk.add(name)

                if name not in current_names:
                    # New person - start process
                    await supervisor.start_person(name)
                    request.app.state.person_names.append(name)
                    added.append(name)
                    logger.info("Hot-loaded person: %s", name)
                else:
                    # Existing person — restart to pick up file changes
                    await supervisor.restart_person(name)
                    refreshed.append(name)
                    logger.info("Refreshed person: %s", name)

        # Remove persons whose directories no longer exist
        for name in list(current_names):
            if name not in on_disk:
                await supervisor.stop_person(name)
                request.app.state.person_names.remove(name)
                removed.append(name)
                logger.info("Unloaded person: %s", name)

        return {
            "added": added,
            "refreshed": refreshed,
            "removed": removed,
            "total": len(request.app.state.person_names),
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
                for name in request.app.state.person_names
            },
        }

    # ── Scheduler ──────────────────────────────────────────

    @router.get("/system/scheduler")
    async def system_scheduler(request: Request):
        """Return scheduler job information."""
        supervisor = request.app.state.supervisor

        jobs: list[dict] = []
        scheduler = getattr(supervisor, "scheduler", None)
        if scheduler is not None:
            for job in scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": (
                        str(job.next_run_time) if job.next_run_time else None
                    ),
                    "trigger": str(job.trigger),
                })

        return {
            "running": scheduler is not None,
            "jobs": jobs,
        }

    # ── Activity ───────────────────────────────────────────

    @router.get("/activity/recent")
    async def get_recent_activity(
        request: Request, hours: int = 24, person: str | None = None,
    ):
        """Return recent activity events aggregated across persons."""
        from datetime import date as date_type
        from datetime import datetime, timedelta, timezone

        from core.memory.conversation import ConversationMemory
        from core.memory.shortterm import ShortTermMemory

        persons = request.app.state.persons
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events: list[dict] = []

        target_persons = (
            {person: persons[person]}
            if person and person in persons
            else persons
        )

        for name, p in target_persons.items():
            # Short-term memory archives (session history)
            stm = ShortTermMemory(p.person_dir)
            archive_dir = stm._archive_dir
            if archive_dir.exists():
                for json_file in sorted(
                    archive_dir.glob("*.json"), reverse=True,
                ):
                    try:
                        data = json.loads(
                            json_file.read_text(encoding="utf-8"),
                        )
                        ts_str = data.get("timestamp", "")
                        ts = (
                            datetime.fromisoformat(ts_str) if ts_str else None
                        )
                        if ts and ts.replace(tzinfo=timezone.utc) < cutoff:
                            break  # Archive is descending; older entries follow
                        events.append({
                            "type": "session",
                            "persons": [name],
                            "timestamp": ts_str,
                            "summary": (
                                data.get("trigger", "")
                                + ": "
                                + data.get("original_prompt", "")[:80]
                            ),
                            "metadata": {
                                "trigger": data.get("trigger", ""),
                                "turn_count": data.get("turn_count", 0),
                                "context_usage_ratio": data.get(
                                    "context_usage_ratio", 0,
                                ),
                            },
                        })
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue

            # Conversation transcripts (today)
            conv = ConversationMemory(p.person_dir, p.model_config)
            today = date_type.today().isoformat()
            messages = conv.load_transcript(today)
            for msg in messages:
                ts_str = msg.get("timestamp", "")
                events.append({
                    "type": "chat",
                    "persons": [name],
                    "timestamp": ts_str,
                    "summary": (msg.get("content", ""))[:80],
                    "metadata": {"role": msg.get("role", "")},
                })

        # Sort descending by timestamp, cap at 200 items
        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return {"events": events[:200]}

    return router
