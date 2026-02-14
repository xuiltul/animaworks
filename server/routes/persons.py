from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from server.dependencies import get_person

logger = logging.getLogger("animaworks.routes.persons")


def _read_appearance(person_dir: Path) -> dict | None:
    """Read appearance.json from a person directory."""
    path = person_dir / "appearance.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data else None
    except (json.JSONDecodeError, OSError):
        return None


def create_persons_router() -> APIRouter:
    router = APIRouter()

    @router.get("/persons")
    async def list_persons(request: Request):
        supervisor = request.app.state.supervisor
        persons_dir = request.app.state.persons_dir
        person_names = request.app.state.person_names

        result = []
        for name in person_names:
            person_dir = persons_dir / name

            # Get process status
            proc_status = supervisor.get_process_status(name)

            # Read static files
            appearance = _read_appearance(person_dir)

            # Combine data
            data = {
                "name": name,
                "status": proc_status.get("status", "unknown"),
                "pid": proc_status.get("pid"),
                "uptime_sec": proc_status.get("uptime_sec"),
                "appearance": appearance,
            }
            result.append(data)

        return result

    @router.get("/persons/{name}")
    async def get_person_detail(name: str, request: Request):
        supervisor = request.app.state.supervisor
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name

        if not person_dir.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        # Get process status
        proc_status = supervisor.get_process_status(name)

        # Read memory files directly from disk
        from core.memory.manager import MemoryManager

        memory = MemoryManager(person_dir)

        return {
            "status": proc_status,
            "identity": memory.read_identity(),
            "injection": memory.read_injection(),
            "state": memory.read_current_state(),
            "pending": memory.read_pending(),
            "knowledge_files": memory.list_knowledge_files(),
            "episode_files": memory.list_episode_files(),
            "procedure_files": memory.list_procedure_files(),
        }

    @router.post("/persons/{name}/trigger")
    async def trigger_heartbeat(name: str, request: Request):
        supervisor = request.app.state.supervisor

        try:
            # Send IPC request to run heartbeat
            result = await supervisor.send_request(
                person_name=name,
                method="run_heartbeat",
                params={},
                timeout=120.0  # Heartbeat can take longer
            )

            return result

        except KeyError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")
        except ValueError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    # ── Person Config ─────────────────────────────────────

    @router.get("/persons/{name}/config")
    async def get_person_config(name: str, request: Request):
        """Return resolved model configuration for a person."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        from core.config.models import load_config, resolve_person_config

        config = load_config()
        resolved, credential = resolve_person_config(config, name)

        return {
            "person": name,
            "model": resolved.model,
            "execution_mode": resolved.execution_mode,
            "config": resolved.model_dump(),
        }

    return router
