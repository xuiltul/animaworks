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
            company = ""
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
                    company = status_data.get("company", "")
            except Exception:
                logger.debug("Failed to resolve config for anima '%s'", name, exc_info=True)

            # Combine data
            data = {
                "name": name,
                "status": proc_status.get("status", "unknown"),
                "bootstrapping": proc_status.get("bootstrapping", False),
                "bootstrap_state": proc_status.get("bootstrap_state", {}),
                "needs_bootstrap": proc_status.get("needs_bootstrap", False),
                "needs_user_input": proc_status.get("needs_user_input", False),
                "needs_repair": proc_status.get("needs_repair", False),
                "needs_background_bootstrap": proc_status.get("needs_background_bootstrap", False),
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
                "company": company,
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

        # Always call stop_anima (no-op if not running). Under lifecycle lock
        # this waits for an in-flight start, then stops the new process.
        supervisor = request.app.state.supervisor
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
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name

        # Disk existence (same style as enable/disable), not anima_names list.
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Refuse start while disabled; enable API writes enabled=true then starts.
        from core.supervisor.manager import ProcessSupervisor

        if not ProcessSupervisor.read_anima_enabled(anima_dir):
            raise HTTPException(
                status_code=409,
                detail=(f"Anima '{name}' is disabled. Call POST /api/animas/{name}/enable first."),
            )

        proc_status = supervisor.get_process_status(name)
        current = proc_status.get("status", "unknown")
        if current not in ("not_found", "stopped", "unknown"):
            return {"status": "already_running", "current_status": current}

        await supervisor.start_anima(name)
        # Concurrent disable/shutdown may refuse start without raising.
        if name not in supervisor.processes:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Anima '{name}' was not started (disabled or refused). Call POST /api/animas/{name}/enable first."
                ),
            )
        if name not in anima_names:
            anima_names.append(name)
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

        try:
            from core.anima_roster import refresh_anima_roster

            refresh_anima_roster()
        except Exception:
            logger.debug("Failed to refresh anima roster after delete", exc_info=True)

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
            company = ""
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
                    company = sdata.get("company", "")
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
                "company": company,
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
                "company": info.get("company", ""),
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

        companies_meta: dict[str, dict] = {}
        try:
            from core.company import list_companies

            summaries, _ = list_companies(data_dir=animas_dir.parent)
            companies_meta = {s.name: {"display_name": s.display_name} for s in summaries}
        except Exception:
            logger.debug("Failed to list companies for org chart", exc_info=True)

        return {
            "generated_at": datetime.now(tz=timezone(timedelta(hours=9))).isoformat(),
            "total": len(flat),
            "tree": tree,
            "flat": flat,
            "companies": companies_meta,
        }

    return router
