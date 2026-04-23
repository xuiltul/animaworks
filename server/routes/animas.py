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


def _validate_anima_name(name: str) -> None:
    """Reject names that could escape the animas directory."""
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid anima name")


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
        for name in list(anima_names):
            anima_dir = animas_dir / name

            # Skip animas whose identity.md no longer exists on disk
            if not (anima_dir / "identity.md").exists():
                continue

            # Get process status
            proc_status = supervisor.get_process_status(name)

            # Read static files
            appearance = _read_appearance(anima_dir)

            # Resolve all config including supervisor from status.json
            model = None
            role = None
            department = ""
            title = ""
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
                    department = status_data.get("department", "")
                    title = status_data.get("title", "")
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
                "department": department,
                "title": title,
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

        identity, injection, cur_state, k_files, e_files, p_files = await asyncio.gather(
            asyncio.to_thread(memory.read_identity),
            asyncio.to_thread(memory.read_injection),
            asyncio.to_thread(memory.read_current_state),
            asyncio.to_thread(memory.list_knowledge_files),
            asyncio.to_thread(memory.list_episode_files),
            asyncio.to_thread(memory.list_procedure_files),
        )

        return {
            "status": proc_status,
            "identity": identity,
            "injection": injection,
            "state": cur_state,
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

    # ── Heartbeat / Cron Config ────────────────────────────

    @router.get("/animas/{name}/heartbeat")
    async def get_anima_heartbeat(name: str, request: Request):
        """Return the raw heartbeat.md content for an anima."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        hb_path = anima_dir / "heartbeat.md"
        content = ""
        if hb_path.exists():
            content = await asyncio.to_thread(hb_path.read_text, "utf-8")
        return {"content": content}

    @router.get("/animas/{name}/cron")
    async def get_anima_cron(name: str, request: Request):
        """Return the raw cron.md content for an anima."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        cron_path = anima_dir / "cron.md"
        content = ""
        if cron_path.exists():
            content = await asyncio.to_thread(cron_path.read_text, "utf-8")
        return {"content": content}

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

    # ── Permissions ────────────────────────────────────────

    @router.get("/animas/{name}/permissions")
    async def get_anima_permissions(name: str, request: Request):
        """Return permissions.json for an anima."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        perm_path = anima_dir / "permissions.json"
        if not perm_path.exists():
            return {}
        try:
            data = json.loads(await asyncio.to_thread(perm_path.read_text, "utf-8"))
            return data
        except (json.JSONDecodeError, OSError) as e:
            raise HTTPException(status_code=500, detail=str(e)) from None

    @router.put("/animas/{name}/permissions")
    async def update_anima_permissions(name: str, request: Request):
        """Update permissions.json for an anima."""
        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()

        # Validate: must be a dict
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="permissions must be a JSON object")

        perm_path = anima_dir / "permissions.json"
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        await asyncio.to_thread(perm_path.write_text, content, "utf-8")
        logger.info("Updated permissions.json for anima '%s'", name)
        return {"status": "ok", "name": name}

    # ── Edit Identity / Injection / Model ────────────────────

    @router.put("/animas/{name}/identity")
    async def update_anima_identity(name: str, request: Request):
        """Update the identity.md content for an anima."""
        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        content: str = data.get("content", "")

        identity_path = anima_dir / "identity.md"
        await asyncio.to_thread(identity_path.write_text, content, "utf-8")
        logger.info("Updated identity.md for anima '%s' (%d chars)", name, len(content))
        return {"status": "ok", "name": name, "field": "identity", "length": len(content)}

    @router.put("/animas/{name}/injection")
    async def update_anima_injection(name: str, request: Request):
        """Update the injection.md content for an anima."""
        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        content: str = data.get("content", "")

        injection_path = anima_dir / "injection.md"
        await asyncio.to_thread(injection_path.write_text, content, "utf-8")
        logger.info("Updated injection.md for anima '%s' (%d chars)", name, len(content))
        return {"status": "ok", "name": name, "field": "injection", "length": len(content)}

    @router.put("/animas/{name}/model")
    async def update_anima_model(name: str, request: Request):
        """Update the model setting in status.json for an anima."""
        from core.config.model_config import smart_update_model as _smart_update

        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        model: str = data.get("model", "").strip()
        credential: str = data.get("credential", "").strip() or None

        if not model:
            raise HTTPException(status_code=400, detail="model is required")

        result = await asyncio.to_thread(
            _smart_update,
            anima_dir,
            model=model,
            credential=credential,
        )
        logger.info(
            "Updated model for anima '%s': model=%s credential=%s execution_mode=%s mode_s_auth=%s",
            name,
            result["model"],
            result["credential"],
            result["execution_mode"],
            result.get("mode_s_auth"),
        )
        return {
            "status": "ok",
            "name": name,
            "model": result["model"],
            "credential": result["credential"],
            "execution_mode": result["execution_mode"],
            "mode_s_auth": result.get("mode_s_auth"),
        }

    # ── Aliases ──────────────────────────────────────────────

    @router.get("/animas/{name}/aliases")
    async def get_anima_aliases(name: str, request: Request):
        """Return aliases for an anima from config.json."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        config = load_config()
        anima_cfg = config.animas.get(name)
        aliases = anima_cfg.aliases if anima_cfg else []
        return {"name": name, "aliases": aliases}

    @router.put("/animas/{name}/aliases")
    async def update_anima_aliases(name: str, request: Request):
        """Update aliases for an anima in config.json."""
        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        new_aliases = data.get("aliases", [])
        if not isinstance(new_aliases, list):
            raise HTTPException(status_code=400, detail="aliases must be a list")
        new_aliases = [str(a).strip() for a in new_aliases if str(a).strip()]

        from core.config.io import save_config
        from core.config.schemas import AnimaModelConfig

        config = load_config()
        if name not in config.animas:
            config.animas[name] = AnimaModelConfig()
        config.animas[name].aliases = new_aliases
        await asyncio.to_thread(save_config, config)
        logger.info("Updated aliases for anima '%s': %s", name, new_aliases)
        return {"status": "ok", "name": name, "aliases": new_aliases}

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

    @router.delete("/animas/{name}")
    async def delete_anima(name: str, request: Request):
        """Stop and delete an anima entirely (process + files)."""
        import shutil

        supervisor = request.app.state.supervisor
        animas_dir: Path = request.app.state.animas_dir
        anima_names: list[str] = request.app.state.anima_names
        anima_dir = animas_dir / name

        # Stop the process if running
        if name in supervisor.processes:
            try:
                await supervisor.stop_anima(name)
            except Exception:
                logger.warning("Failed to stop anima '%s' before delete", name, exc_info=True)

        # Remove from anima_names list
        if name in anima_names:
            anima_names.remove(name)

        # Delete directory from disk
        if anima_dir.exists():
            try:
                shutil.rmtree(anima_dir)
                logger.info("Deleted anima directory: %s", name)
            except Exception as exc:
                # If locked files, strip identity.md so it won't appear
                try:
                    (anima_dir / "identity.md").unlink(missing_ok=True)
                    (anima_dir / "status.json").unlink(missing_ok=True)
                except Exception:
                    pass
                logger.warning("Partial delete for '%s': %s", name, exc)
                return {"status": "partial", "name": name, "detail": str(exc)}

        return {"status": "deleted", "name": name}

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
            department = ""
            title = ""
            enabled = name in request.app.state.anima_names
            try:
                resolved, _ = resolve_anima_config(config, name, anima_dir=anima_dir)
                model = resolved.model
                anima_supervisor = resolved.supervisor
                anima_speciality = resolved.speciality
            except Exception:
                pass

            # Read department/title from status.json
            status_path = anima_dir / "status.json"
            if status_path.exists():
                try:
                    sdata = json.loads(status_path.read_text(encoding="utf-8"))
                    department = sdata.get("department", "")
                    title = sdata.get("title", "")
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
                "department": department,
                "title": title,
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
                "department": info.get("department", ""),
                "title": info.get("title", ""),
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

    # ── Team Deploy ─────────────────────────────────────
    # Role → identity / injection content for each team-builder role
    _TEAM_ROLE_CONTENT: dict[str, dict[str, str]] = {
        "secretary": {
            "identity": "あなたは優秀な秘書AIです。丁寧で正確、先回りして行動します。スケジュール管理、メール対応、タスク整理が得意です。",
            "injection": "メール・カレンダー・Slackの確認と管理を行い、上司のスケジュールを最適化してください。重要な予定や期限のリマインドを積極的に行ってください。",
        },
        "customer_support": {
            "identity": "あなたはカスタマーサポート専門のAIです。共感力が高く、問題解決に長けています。顧客満足度を最優先にします。",
            "injection": "顧客からの問い合わせに迅速かつ丁寧に対応してください。FAQを活用し、解決できない場合は適切にエスカレーションしてください。",
        },
        "back_office": {
            "identity": "あなたはバックオフィス業務を担当するAIです。正確で効率的なデータ処理が得意です。",
            "injection": "スプレッドシートの管理、データ入力・集計、書類整理を正確に行ってください。定型業務の自動化を積極的に提案してください。",
        },
        "sales_assist": {
            "identity": "あなたは営業アシスタントAIです。リード管理とフォローアップが得意で、営業チームの生産性向上に貢献します。",
            "injection": "リードの進捗管理、見積もり作成の補助、フォローアップメールの下書き作成を行ってください。商談の状況を把握し、適切なタイミングでのアクションを提案してください。",
        },
        "pr_sns": {
            "identity": "あなたはPR・SNS運用を担当するAIです。トレンドに敏感で、エンゲージメントの高いコンテンツ作成が得意です。",
            "injection": "SNS投稿の企画・下書き作成、プレスリリースの作成補助、ブランディング戦略の提案を行ってください。",
        },
        "recruiter": {
            "identity": "あなたは採用担当AIです。候補者とのコミュニケーションが丁寧で、採用プロセスを効率化します。",
            "injection": "求人票の作成、候補者管理、面接日程の調整、フォローアップメールの作成を行ってください。",
        },
        "accounting": {
            "identity": "あなたは経理アシスタントAIです。数字に正確で、経理業務を効率化します。",
            "injection": "経費集計、請求書チェック、レポート作成を正確に行ってください。異常値を検知した場合は即座に報告してください。",
        },
        "project_manager": {
            "identity": "あなたはプロジェクトマネージャーAIです。進捗管理とボトルネック検出が得意です。",
            "injection": "タスクの進捗追跡、期限管理、ボトルネックの早期検出と報告を行ってください。チームメンバーの負荷バランスにも注意してください。",
        },
        "researcher": {
            "identity": "あなたはリサーチャーAIです。情報収集と分析が得意で、正確で読みやすいレポートを作成します。",
            "injection": "指定されたテーマについて情報を収集・分析し、要点をまとめたレポートを作成してください。情報源の信頼性を常に確認してください。",
        },
        "content_writer": {
            "identity": "あなたはコンテンツライターAIです。読みやすく魅力的な文章を書くことが得意です。",
            "injection": "ブログ記事、メルマガ、案内文などのコンテンツを作成してください。ターゲット読者に合わせたトーンで、SEOも意識して執筆してください。",
        },
    }
    _DEFAULT_ROLE_CONTENT = {
        "identity": "あなたは業務支援AIエージェントです。与えられたタスクを正確かつ効率的に遂行します。",
        "injection": "与えられたタスクに集中し、不明点があれば確認してから行動してください。",
    }

    @router.post("/teams/deploy")
    async def deploy_team(request: Request):
        """Deploy team members as new Animas."""
        from core.anima_factory import create_blank, validate_anima_name

        data = await request.json()
        members_raw: list[dict[str, Any]] = data.get("members", [])
        team_department: str = data.get("department", "").strip()
        team_report_to: str = data.get("reportTo", "").strip()
        logger.info(
            "deploy_team: received %d members (dept=%s, reportTo=%s): %s",
            len(members_raw),
            team_department,
            team_report_to or "(owner)",
            [m.get("displayName") for m in members_raw],
        )

        animas_dir: Path = request.app.state.animas_dir
        created: list[dict[str, str]] = []
        errors: list[str] = []

        # Track the lead member's anima name for supervisor assignment
        lead_anima_name: str | None = None

        for member in members_raw:
            display_name: str = member.get("displayName", "").strip()
            role_id: str = member.get("roleId", "general")
            tools: list[str] = member.get("tools", [])
            member_model: str = member.get("model", "").strip()
            member_credential: str = member.get("credential", "").strip()
            member_title: str = member.get("title", "").strip()
            member_is_lead: bool = member.get("isLead", False)

            if not display_name:
                errors.append("Empty display name — skipped")
                continue

            # Generate a valid anima name (lowercase, alphanumeric)
            name = display_name.lower().replace(" ", "_").replace("-", "_")
            # Strip invalid characters
            name = "".join(c for c in name if c.isascii() and (c.isalnum() or c == "_"))
            if not name or not name[0].isalpha():
                name = f"agent_{name}" if name else f"agent_{len(created) + 1}"

            err = validate_anima_name(name)
            if err:
                name = f"agent_{len(created) + 1}"

            # Reject if name already exists (avoid silent duplicates)
            if (animas_dir / name).exists():
                errors.append(f"{display_name} ({name}): already exists — rename or delete the existing anima first")
                continue

            try:
                anima_dir = create_blank(animas_dir, name)

                # Write role-specific identity.md
                content = _TEAM_ROLE_CONTENT.get(role_id, _DEFAULT_ROLE_CONTENT)
                (anima_dir / "identity.md").write_text(
                    f"# {display_name}\n\n{content['identity']}\n",
                    encoding="utf-8",
                )

                # Write role-specific injection.md
                tool_list = ", ".join(tools) if tools else "なし"
                (anima_dir / "injection.md").write_text(
                    f"{content['injection']}\n\n## 利用可能ツール\n{tool_list}\n",
                    encoding="utf-8",
                )

                # Write status.json with role info, model, org data
                status: dict[str, Any] = {
                    "enabled": True,
                    "role": role_id,
                }

                if team_department:
                    status["department"] = team_department
                if member_title:
                    status["title"] = member_title

                # Supervisor assignment:
                #   Lead → reports to team_report_to (or null = owner)
                #   Non-lead → will be updated later to report to lead
                if member_is_lead:
                    lead_anima_name = name
                    if team_report_to:
                        status["supervisor"] = team_report_to
                    # else: supervisor stays unset → top-level (owner)

                if member_model:
                    status["model"] = member_model
                    if member_credential:
                        status["credential"] = member_credential
                else:
                    try:
                        from core.config.local_llm import apply_local_llm_role_to_status

                        cfg = load_config()
                        apply_local_llm_role_to_status(status, cfg, "general")
                    except Exception:
                        pass

                (anima_dir / "status.json").write_text(
                    json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

                # Register in config.json + auto-create .env Slack token slots
                try:
                    from core.config.anima_registry import register_anima_in_config

                    register_anima_in_config(
                        animas_dir.parent,  # data_dir
                        name,
                        supervisor=status.get("supervisor"),
                    )
                except Exception:
                    logger.debug("register_anima_in_config failed for '%s'", name, exc_info=True)

                created.append({"name": name, "displayName": display_name, "roleId": role_id, "isLead": member_is_lead})
                logger.info("Deployed team member '%s' as anima '%s' (role=%s)", display_name, name, role_id)

            except Exception as exc:
                errors.append(f"{display_name}: {exc}")
                logger.exception("Failed to deploy team member '%s'", display_name)

        # Assign non-lead members' supervisor to the lead
        if lead_anima_name and len(created) > 1:
            for entry in created:
                if entry.get("isLead"):
                    continue
                member_dir = animas_dir / entry["name"]
                status_path = member_dir / "status.json"
                if status_path.exists():
                    try:
                        sdata = json.loads(status_path.read_text(encoding="utf-8"))
                        sdata["supervisor"] = lead_anima_name
                        status_path.write_text(
                            json.dumps(sdata, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    except Exception:
                        logger.warning("Failed to set supervisor for '%s'", entry["name"], exc_info=True)

        # Trigger reconciliation so supervisor picks up new animas
        if created:
            try:
                supervisor = request.app.state.supervisor
                if supervisor and hasattr(supervisor, "reconcile"):
                    await asyncio.get_running_loop().run_in_executor(None, supervisor.reconcile)
            except Exception:
                logger.debug("Reconciliation trigger failed (will auto-detect within 30s)", exc_info=True)

        return {"created": created, "errors": errors}

    return router
