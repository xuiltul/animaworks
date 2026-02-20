from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from core.config.models import AnimaModelConfig, load_config, resolve_anima_config
from core.exceptions import AnimaWorksError
from server.dependencies import get_anima

logger = logging.getLogger("animaworks.routes.animas")


def _read_appearance(anima_dir: Path) -> dict | None:
    """Read appearance.json from an anima directory."""
    path = anima_dir / "appearance.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data else None
    except (json.JSONDecodeError, OSError):
        return None


def create_animas_router() -> APIRouter:
    router = APIRouter()

    @router.get("/animas")
    async def list_animas(request: Request):
        supervisor = request.app.state.supervisor
        animas_dir = request.app.state.animas_dir
        anima_names = request.app.state.anima_names

        config = load_config()

        result = []
        for name in anima_names:
            anima_dir = animas_dir / name

            # Get process status
            proc_status = supervisor.get_process_status(name)

            # Read static files
            appearance = _read_appearance(anima_dir)

            # Read supervisor from config
            anima_cfg = config.animas.get(name, AnimaModelConfig())

            # Resolve model name
            model = None
            try:
                resolved, _ = resolve_anima_config(config, name, anima_dir=anima_dir)
                model = resolved.model
            except Exception:
                logger.debug("Failed to resolve model for anima '%s'", name, exc_info=True)

            # Combine data
            data = {
                "name": name,
                "status": proc_status.get("status", "unknown"),
                "bootstrapping": proc_status.get("bootstrapping", False),
                "pid": proc_status.get("pid"),
                "uptime_sec": proc_status.get("uptime_sec"),
                "appearance": appearance,
                "supervisor": anima_cfg.supervisor,
                "model": model,
            }
            result.append(data)

        return result

    @router.get("/animas/{name}")
    async def get_anima_detail(name: str, request: Request):
        supervisor = request.app.state.supervisor
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name

        if not anima_dir.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Get process status
        proc_status = supervisor.get_process_status(name)

        # Read memory files from disk — parallelised via thread pool
        from core.memory.manager import MemoryManager

        memory = MemoryManager(anima_dir)

        identity, injection, cur_state, pending, k_files, e_files, p_files = (
            await asyncio.gather(
                asyncio.to_thread(memory.read_identity),
                asyncio.to_thread(memory.read_injection),
                asyncio.to_thread(memory.read_current_state),
                asyncio.to_thread(memory.read_pending),
                asyncio.to_thread(memory.list_knowledge_files),
                asyncio.to_thread(memory.list_episode_files),
                asyncio.to_thread(memory.list_procedure_files),
            )
        )

        return {
            "status": proc_status,
            "identity": identity,
            "injection": injection,
            "state": cur_state,
            "pending": pending,
            "knowledge_files": k_files,
            "episode_files": e_files,
            "procedure_files": p_files,
        }

    @router.post("/animas/{name}/trigger")
    async def trigger_heartbeat(name: str, request: Request):
        supervisor = request.app.state.supervisor

        try:
            # Send IPC request to run heartbeat
            result = await supervisor.send_request(
                anima_name=name,
                method="run_heartbeat",
                params={},
                timeout=120.0  # Heartbeat can take longer
            )

            return result

        except KeyError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")
        except ValueError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for heartbeat from anima=%s", name)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                {"error": "Request timed out"}, status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in trigger for anima=%s", name)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                {"error": f"Internal server error: {e}"}, status_code=500,
            )

    # ── Anima Config ─────────────────────────────────────

    @router.get("/animas/{name}/config")
    async def get_anima_config(name: str, request: Request):
        """Return resolved model configuration for an anima."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.config.models import load_config, resolve_anima_config

        config = load_config()
        resolved, credential = resolve_anima_config(config, name, anima_dir=anima_dir)

        return {
            "anima": name,
            "model": resolved.model,
            "execution_mode": resolved.execution_mode,
            "config": resolved.model_dump(),
        }

    # ── Enable / Disable ─────────────────────────────────────

    @router.post("/animas/{name}/enable")
    async def enable_anima(name: str, request: Request):
        """Enable an Anima (set status.json to enabled: true)."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(
                status_code=404, detail=f"Anima '{name}' not found"
            )

        status_file = anima_dir / "status.json"
        status_file.write_text(
            json.dumps({"enabled": True}, indent=2), encoding="utf-8"
        )

        # Start immediately (don't wait for reconciliation)
        supervisor = request.app.state.supervisor
        if name not in supervisor.processes:
            await supervisor.start_anima(name)
            if name not in request.app.state.anima_names:
                request.app.state.anima_names.append(name)

        return {"name": name, "enabled": True}

    @router.post("/animas/{name}/disable")
    async def disable_anima(name: str, request: Request):
        """Disable an Anima (set status.json to enabled: false and stop process)."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(
                status_code=404, detail=f"Anima '{name}' not found"
            )

        status_file = anima_dir / "status.json"
        status_file.write_text(
            json.dumps({"enabled": False}, indent=2), encoding="utf-8"
        )

        # Stop immediately
        supervisor = request.app.state.supervisor
        if name in supervisor.processes:
            await supervisor.stop_anima(name)
            if name in request.app.state.anima_names:
                request.app.state.anima_names.remove(name)

        return {"name": name, "enabled": False}

    # ── Background Tasks ────────────────────────────────────

    @router.get("/animas/{name}/background-tasks")
    async def list_background_tasks(name: str, request: Request):
        """List background tasks for an anima (reads from state dir)."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        bg_dir = anima_dir / "state" / "background_tasks"
        if not bg_dir.exists():
            return {"tasks": []}

        tasks = []
        for path in sorted(bg_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tasks.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        return {"tasks": tasks}

    @router.get("/animas/{name}/background-tasks/{task_id}")
    async def get_background_task(name: str, task_id: str, request: Request):
        """Get a specific background task by ID."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        task_file = anima_dir / "state" / "background_tasks" / f"{task_id}.json"

        if not task_file.exists():
            raise HTTPException(
                status_code=404, detail=f"Background task not found: {task_id}",
            )

        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            return data
        except (json.JSONDecodeError, OSError) as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Start / Stop / Restart ────────────────────────────────

    @router.post("/animas/{name}/start")
    async def start_anima(name: str, request: Request):
        """Start a stopped anima process."""
        supervisor = request.app.state.supervisor
        anima_names = request.app.state.anima_names

        if name not in anima_names:
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        proc_status = supervisor.get_process_status(name)
        current = proc_status.get("status", "unknown")
        if current not in ("not_found", "stopped", "unknown"):
            return {"status": "already_running", "current_status": current}

        await supervisor.start_anima(name)
        return {"status": "started", "name": name}

    @router.post("/animas/{name}/stop")
    async def stop_anima(name: str, request: Request):
        """Stop a specific anima process."""
        supervisor = request.app.state.supervisor
        anima_names = request.app.state.anima_names

        if name not in anima_names:
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        if name not in supervisor.processes:
            return {"status": "already_stopped", "name": name}

        await supervisor.stop_anima(name)
        return {"status": "stopped", "name": name}

    @router.post("/animas/{name}/restart")
    async def restart_anima(name: str, request: Request):
        """Restart a specific anima process."""
        supervisor = request.app.state.supervisor
        anima_names = request.app.state.anima_names

        if name not in anima_names:
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        await supervisor.restart_anima(name)
        proc_status = supervisor.get_process_status(name)
        return {
            "status": "restarted",
            "name": name,
            "pid": proc_status.get("pid"),
        }

    return router
