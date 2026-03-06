from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from core.config.models import load_config, resolve_anima_config

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

            # Resolve all config including supervisor from status.json
            model = None
            role = None
            anima_supervisor = None
            anima_speciality = None
            try:
                resolved, _ = resolve_anima_config(config, name, anima_dir=anima_dir)
                model = resolved.model
                anima_supervisor = resolved.supervisor
                anima_speciality = resolved.speciality
                status_path = anima_dir / "status.json"
                if status_path.exists():
                    status_data = json.loads(status_path.read_text(encoding="utf-8"))
                    role = status_data.get("role")
            except Exception:
                logger.debug("Failed to resolve config for anima '%s'", name, exc_info=True)

            # Combine data
            data = {
                "name": name,
                "status": proc_status.get("status", "unknown"),
                "bootstrapping": proc_status.get("bootstrapping", False),
                "pid": proc_status.get("pid"),
                "uptime_sec": proc_status.get("uptime_sec"),
                "last_busy_since": proc_status.get("last_busy_since"),
                "appearance": appearance,
                "supervisor": anima_supervisor,
                "speciality": anima_speciality,
                "role": role,
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

        identity, injection, cur_state, pending, k_files, e_files, p_files = await asyncio.gather(
            asyncio.to_thread(memory.read_identity),
            asyncio.to_thread(memory.read_injection),
            asyncio.to_thread(memory.read_current_state),
            asyncio.to_thread(memory.read_pending),
            asyncio.to_thread(memory.list_knowledge_files),
            asyncio.to_thread(memory.list_episode_files),
            asyncio.to_thread(memory.list_procedure_files),
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
                timeout=120.0,  # Heartbeat can take longer
            )

            return result

        except KeyError:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Anima not found: {name}") from None
        except ValueError as e:
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=str(e)) from e
        except TimeoutError:
            logger.error("Timeout waiting for heartbeat from anima=%s", name)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"error": "Request timed out"},
                status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in trigger for anima=%s", name)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"error": f"Internal server error: {e}"},
                status_code=500,
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
            raise HTTPException(status_code=404, detail=f"Anima '{name}' not found")

        status_file = anima_dir / "status.json"
        existing = {}
        if status_file.exists():
            try:
                existing = json.loads(status_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        existing["enabled"] = True
        status_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
            raise HTTPException(status_code=404, detail=f"Anima '{name}' not found")

        status_file = anima_dir / "status.json"
        existing = {}
        if status_file.exists():
            try:
                existing = json.loads(status_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        existing["enabled"] = False
        status_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
                status_code=404,
                detail=f"Background task not found: {task_id}",
            )

        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            return data
        except (json.JSONDecodeError, OSError) as e:
            raise HTTPException(status_code=500, detail=str(e)) from None

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

    @router.post("/animas/{name}/interrupt")
    async def interrupt_anima(name: str, request: Request, thread_id: str | None = None):
        """Interrupt the current LLM session without stopping the process.

        Query params:
            thread_id: If provided, only interrupt the specific thread.
                If omitted, interrupt all active threads.
        """
        supervisor = request.app.state.supervisor
        anima_names = request.app.state.anima_names

        if name not in anima_names:
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        if name not in supervisor.processes:
            return {"status": "not_running", "name": name}

        try:
            params: dict[str, Any] = {}
            if thread_id:
                params["thread_id"] = thread_id
            result = await supervisor.send_request(
                anima_name=name,
                method="interrupt",
                params=params,
                timeout=10.0,
            )
            return result
        except TimeoutError:
            return {"status": "timeout", "name": name}
        except Exception as e:
            logger.warning("Failed to interrupt %s: %s", name, e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ── Reload Config ─────────────────────────────────────

    @router.post("/animas/{name}/reload")
    async def reload_anima_config(name: str, request: Request):
        """Hot-reload ModelConfig from status.json without process restart."""
        supervisor = request.app.state.supervisor
        anima_names = request.app.state.anima_names

        if name not in anima_names:
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        try:
            result = await supervisor.send_request(
                anima_name=name,
                method="reload_config",
                params={},
                timeout=10.0,
            )
            return result
        except Exception as e:
            logger.exception("Failed to reload config for anima=%s", name)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/animas/reload-all")
    async def reload_all_anima_configs(request: Request):
        """Hot-reload ModelConfig for all running animas."""
        supervisor = request.app.state.supervisor
        results = {}
        for name, _ in supervisor.processes.items():
            try:
                result = await supervisor.send_request(
                    anima_name=name,
                    method="reload_config",
                    params={},
                    timeout=10.0,
                )
                results[name] = result
            except Exception as e:
                logger.warning("Failed to reload config for %s: %s", name, e)
                results[name] = {"status": "error", "error": str(e)}
        return {"status": "ok", "results": results}

    # ── Org Chart ─────────────────────────────────────────

    @router.get("/org/chart")
    async def get_org_chart(
        request: Request,
        include_disabled: bool = False,
        format: str = "json",
    ):
        """Return the organisation chart as a tree-structured JSON.

        Query params:
            include_disabled: Include disabled animas (default: false)
            format: "json" (default) or "text" for ASCII tree
        """
        supervisor_obj = request.app.state.supervisor
        animas_dir = request.app.state.animas_dir
        anima_names = list(request.app.state.anima_names)

        config = load_config()

        # Optionally include disabled animas
        if include_disabled and animas_dir.exists():
            for anima_dir in sorted(animas_dir.iterdir()):
                if anima_dir.is_dir() and (anima_dir / "identity.md").exists() and anima_dir.name not in anima_names:
                    anima_names.append(anima_dir.name)

        # Build flat lookup: name -> {speciality, supervisor, model, status}
        flat: dict[str, dict] = {}
        for name in anima_names:
            anima_dir = animas_dir / name

            proc_status = supervisor_obj.get_process_status(name)
            model = None
            anima_supervisor = None
            anima_speciality = None
            enabled = name in request.app.state.anima_names
            try:
                resolved, _ = resolve_anima_config(config, name, anima_dir=anima_dir)
                model = resolved.model
                anima_supervisor = resolved.supervisor
                anima_speciality = resolved.speciality
            except Exception:
                pass

            status = proc_status.get("status", "unknown")
            if not enabled:
                status = "disabled"

            flat[name] = {
                "name": name,
                "speciality": anima_speciality,
                "supervisor": anima_supervisor,
                "model": model,
                "status": status,
                "enabled": enabled,
            }

        # Build tree from flat lookup
        def _build_node(name: str) -> dict:
            info = flat[name]
            children_names = sorted(n for n, d in flat.items() if d["supervisor"] == name)
            return {
                "name": name,
                "speciality": info["speciality"],
                "model": info["model"],
                "status": info["status"],
                "enabled": info["enabled"],
                "children": [_build_node(c) for c in children_names],
            }

        # Top-level = animas with no supervisor
        roots = sorted(n for n, d in flat.items() if d["supervisor"] is None)

        from datetime import datetime, timedelta, timezone

        tree = [_build_node(r) for r in roots]

        # Text format: return ASCII tree
        if format == "text":
            from fastapi.responses import PlainTextResponse

            lines: list[str] = []

            def _render(node: dict, prefix: str = "", is_last: bool = True) -> None:
                connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
                status_mark = (
                    "\u2713" if node["status"] == "running" else ("\u2717" if node["status"] == "disabled" else "?")
                )
                label = f"{node['name']} [{node['speciality'] or '?'}] ({status_mark})"
                lines.append(f"{prefix}{connector}{label}")
                child_prefix = prefix + ("    " if is_last else "\u2502   ")
                for i, child in enumerate(node["children"]):
                    _render(child, child_prefix, i == len(node["children"]) - 1)

            lines.append("AnimaWorks Organisation Chart")
            lines.append("=" * 40)
            for i, root in enumerate(tree):
                _render(root, "", i == len(tree) - 1)
            lines.append("")
            lines.append(f"Total: {len(flat)} animas")

            return PlainTextResponse("\n".join(lines))

        return {
            "generated_at": datetime.now(tz=timezone(timedelta(hours=9))).isoformat(),
            "total": len(flat),
            "tree": tree,
            "flat": flat,
        }

    return router
