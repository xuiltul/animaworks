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

from core.anima.workspace_files import HEARTBEAT_FILE, IDENTITY_FILE
from core.config.models import load_config, resolve_anima_config
from core.i18n import t

logger = logging.getLogger("animaworks.routes.animas")


def _validate_anima_name(name: str) -> None:
    """Reject names that could escape the animas directory."""
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid anima name")


def _process_handle_is_alive(handle: object) -> bool:
    """Fail closed unless a retained supervisor handle is known to be dead."""
    process = getattr(handle, "process", None)
    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            return poll() is None
        except Exception:
            return True
    state = getattr(getattr(handle, "state", None), "value", None)
    return state not in {"stopped", "failed"}


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


def _read_busy_sidecar(name: str) -> dict | None:
    """Read the IPC-independent busy marker written by the child process.

    Lets clients (e.g. the pixel workspace) show in-progress work immediately
    on page load instead of waiting for the next live event.
    """
    from core.paths import get_data_dir

    path = get_data_dir() / "run" / "animas" / f"{name}.busy.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("is_busy"):
        return None
    return {
        "is_busy": True,
        "busy_since": data.get("busy_since"),
        "last_progress_at": data.get("last_progress_at"),
        "lanes": data.get("lanes") or [],
    }


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

            # Skip animas whose IDENTITY.md no longer exists on disk
            if not (anima_dir / IDENTITY_FILE).exists():
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
            enabled = True
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
                    enabled = bool(status_data.get("enabled", True))
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
                "process_count": proc_status.get("process_count"),
                "subprocesses": proc_status.get("subprocesses") or [],
                "uptime_sec": proc_status.get("uptime_sec"),
                "last_busy_since": proc_status.get("last_busy_since"),
                "busy": _read_busy_sidecar(name),
                "appearance": appearance,
                "supervisor": anima_supervisor,
                "speciality": anima_speciality,
                "role": role,
                "model": model,
                "department": department,
                "title": title,
                "company": company,
                "enabled": enabled,
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

        memory = MemoryManager(anima_dir, read_only=True)

        identity, soul, injection, cur_state, k_files, e_files, p_files = await asyncio.gather(
            asyncio.to_thread(memory.read_identity),
            asyncio.to_thread(memory.read_soul),
            asyncio.to_thread(memory.read_injection),
            asyncio.to_thread(memory.read_current_state),
            asyncio.to_thread(memory.list_knowledge_files),
            asyncio.to_thread(memory.list_episode_files),
            asyncio.to_thread(memory.list_procedure_files),
        )

        return {
            "status": proc_status,
            "identity": identity,
            "soul": soul,
            "injection": injection,
            "state": cur_state,
            "knowledge_files": k_files,
            "episode_files": e_files,
            "procedure_files": p_files,
        }

    @router.get("/animas/{name}/summary")
    async def get_anima_summary(name: str, request: Request):
        """Return only the small set of Anima fields needed by chat selection."""
        _validate_anima_name(name)
        if not name or name in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid anima name")

        animas_dir = request.app.state.animas_dir
        anima_dir = (animas_dir / name).resolve()
        if not anima_dir.is_relative_to(animas_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid anima path")
        if not anima_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        proc_status = request.app.state.supervisor.get_process_status(name)
        status_data: dict[str, Any] = {}
        status_path = anima_dir / "status.json"
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                status_data = loaded
        except (OSError, json.JSONDecodeError):
            logger.debug("Failed to read status.json for anima '%s'", name, exc_info=True)

        busy = _read_busy_sidecar(name)
        return {
            "name": name,
            "display_name": status_data.get("display_name") or name,
            "status": {
                "status": proc_status.get("status", "unknown"),
                "bootstrapping": proc_status.get("bootstrapping", False),
                "needs_bootstrap": proc_status.get("needs_bootstrap", False),
                "needs_user_input": proc_status.get("needs_user_input", False),
                "needs_repair": proc_status.get("needs_repair", False),
                "needs_background_bootstrap": proc_status.get("needs_background_bootstrap", False),
                "active_label": proc_status.get("active_label"),
            },
            "busy": busy,
            "model": status_data.get("model"),
            "role": status_data.get("role"),
            "enabled": bool(status_data.get("enabled", True)),
        }

    @router.get("/animas/{name}/state")
    async def get_anima_state(name: str, request: Request):
        """Read only the current-state document for the chat information pane."""
        _validate_anima_name(name)
        if not name or name in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid anima name")

        animas_dir = request.app.state.animas_dir
        anima_dir = (animas_dir / name).resolve()
        if not anima_dir.is_relative_to(animas_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid anima path")
        if not anima_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.memory.manager import MemoryManager

        memory = MemoryManager(anima_dir, read_only=True)
        current_state = await asyncio.to_thread(memory.read_current_state)
        return {"state": current_state}

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
        """Return the raw HEARTBEAT.md content for an anima."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        hb_path = anima_dir / HEARTBEAT_FILE
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
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
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
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()

        # Validate: must be a dict
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="permissions must be a JSON object")

        from core.anima.settings_store import write_permissions

        await asyncio.to_thread(write_permissions, anima_dir, data)
        logger.info("Updated permissions.json for anima '%s'", name)
        return {"status": "ok", "name": name}

    # ── Edit Identity / Injection / Model ────────────────────

    @router.put("/animas/{name}/identity")
    async def update_anima_identity(name: str, request: Request):
        """Update the IDENTITY.md content for an anima."""
        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        if not isinstance(data, dict) or not isinstance(data.get("content", ""), str):
            raise HTTPException(status_code=400, detail="content must be a string")
        content: str = data.get("content", "")

        from core.anima.settings_store import write_identity

        await asyncio.to_thread(write_identity, anima_dir, content)
        logger.info("Updated IDENTITY.md for anima '%s' (%d chars)", name, len(content))
        return {"status": "ok", "name": name, "field": "identity", "length": len(content)}

    @router.put("/animas/{name}/injection")
    async def update_anima_injection(name: str, request: Request):
        """Update the AGENTS.md content for an anima."""
        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        if not isinstance(data, dict) or not isinstance(data.get("content", ""), str):
            raise HTTPException(status_code=400, detail="content must be a string")
        content: str = data.get("content", "")

        from core.anima.settings_store import write_injection

        await asyncio.to_thread(write_injection, anima_dir, content)
        logger.info("Updated AGENTS.md for anima '%s' (%d chars)", name, len(content))
        return {"status": "ok", "name": name, "field": "injection", "length": len(content)}

    @router.put("/animas/{name}/model")
    async def update_anima_model(name: str, request: Request):
        """Update the model setting in status.json for an anima."""
        from core.config.model_config import smart_update_model as _smart_update

        _validate_anima_name(name)
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")
        raw_model = data.get("model", "")
        raw_credential = data.get("credential", "")
        if not isinstance(raw_model, str) or not isinstance(raw_credential, str):
            raise HTTPException(status_code=400, detail="model and credential must be strings")
        model = raw_model.strip()
        credential = raw_credential.strip() or None

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
        supervisor = getattr(request.app.state, "supervisor", None)
        if supervisor is not None and name in getattr(supervisor, "processes", {}):
            try:
                await supervisor.send_request(name, "reload_config", {}, timeout=10.0)
            except Exception:
                logger.info("Model reload deferred until next start for anima=%s", name, exc_info=True)
        return {
            "status": "ok",
            "name": name,
            "model": result["model"],
            "credential": result["credential"],
            "execution_mode": result["execution_mode"],
            "mode_s_auth": result.get("mode_s_auth"),
            "family_changed": result.get("family_changed", False),
        }

    @router.put("/animas/{name}/background-model")
    async def update_anima_background_model(name: str, request: Request):
        """Update heartbeat/cron model settings in root-owned status.json."""
        _validate_anima_name(name)
        anima_dir = request.app.state.animas_dir / name
        if not anima_dir.is_dir() or not (anima_dir / IDENTITY_FILE).is_file():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")
        data = await request.json()
        model = data.get("model", "") if isinstance(data, dict) else None
        credential = data.get("credential") if isinstance(data, dict) else None
        clear = bool(data.get("clear", False)) if isinstance(data, dict) else False
        if not isinstance(model, str) or (credential is not None and not isinstance(credential, str)):
            raise HTTPException(status_code=400, detail="model and credential must be strings")
        if not clear and not model.strip():
            raise HTTPException(status_code=400, detail="model is required")
        from core.config.model_config import update_status_model

        update: dict[str, Any] = (
            {"background_model": "", "background_credential": ""} if clear else {"background_model": model.strip()}
        )
        if credential is not None and not clear:
            update["background_credential"] = credential.strip()
        await asyncio.to_thread(update_status_model, anima_dir, **update)
        supervisor = getattr(request.app.state, "supervisor", None)
        if supervisor is not None and name in getattr(supervisor, "processes", {}):
            try:
                await supervisor.send_request(name, "reload_config", {}, timeout=10.0)
            except Exception:
                logger.info("Background model reload deferred until next start for anima=%s", name, exc_info=True)
        return {"status": "ok", "name": name, "background_model": "" if clear else model.strip()}

    @router.put("/animas/{name}/role")
    async def update_anima_role(name: str, request: Request):
        """Update role settings and role permission template through root writers."""
        _validate_anima_name(name)
        anima_dir = request.app.state.animas_dir / name
        if not anima_dir.is_dir() or not (anima_dir / IDENTITY_FILE).is_file():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")
        role = data.get("role", "")
        status_only = bool(data.get("status_only", False))
        from core.anima.factory import SHARED_ROLES_DIR, VALID_ROLES, _apply_role_defaults
        from core.anima.settings_store import update_status
        from core.config.local_llm import apply_local_llm_role_to_status
        from core.config.models import load_config

        if not isinstance(role, str) or role not in VALID_ROLES:
            raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
        role_defaults: dict[str, object] = {}
        if not status_only:
            defaults_path = SHARED_ROLES_DIR / role / "defaults.json"
            if defaults_path.is_file():
                try:
                    role_defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    logger.warning("Failed to load role defaults for '%s'", role, exc_info=True)
            config = load_config()
        old_role = "-"

        def apply_role(status: dict[str, object]) -> None:
            nonlocal old_role
            old_role = str(status.get("role", "-"))
            status["role"] = role
            if not status_only:
                for key in ("model", "context_threshold", "conversation_history_threshold"):
                    if key in role_defaults:
                        status[key] = role_defaults[key]
                apply_local_llm_role_to_status(status, config, role)

        try:
            await asyncio.to_thread(update_status, anima_dir, apply_role)
            if not status_only:
                await asyncio.to_thread(_apply_role_defaults, anima_dir, role)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=t("anima.status_json_invalid", name=name)) from exc
        return {"status": "ok", "name": name, "old_role": old_role, "role": role, "status_only": status_only}

    # ── Aliases ──────────────────────────────────────────────

    @router.get("/animas/{name}/aliases")
    async def get_anima_aliases(name: str, request: Request):
        """Return aliases for an anima from config.json."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
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
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        new_aliases = data.get("aliases", [])
        if not isinstance(new_aliases, list):
            raise HTTPException(status_code=400, detail="aliases must be a list")
        new_aliases = [str(a).strip() for a in new_aliases if str(a).strip()]

        from core.config.io import update_config
        from core.config.schemas import AnimaModelConfig

        def update_aliases(config):
            if name not in config.animas:
                config.animas[name] = AnimaModelConfig()
            config.animas[name].aliases = new_aliases
            return config

        await asyncio.to_thread(update_config, update_aliases)
        logger.info("Updated aliases for anima '%s': %s", name, new_aliases)
        return {"status": "ok", "name": name, "aliases": new_aliases}

    # ── Enable / Disable ─────────────────────────────────────

    @router.post("/animas/{name}/enable")
    async def enable_anima(name: str, request: Request):
        """Enable an Anima (set status.json to enabled: true)."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima '{name}' not found")

        from core.anima.settings_store import update_status

        def enable_status(existing: dict[str, Any]) -> None:
            existing["enabled"] = True

        try:
            update_status(anima_dir, enable_status)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=t("anima.status_json_invalid", name=name)) from exc

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
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima '{name}' not found")

        from core.anima.settings_store import update_status

        def disable_status(existing: dict[str, Any]) -> None:
            existing["enabled"] = False

        try:
            update_status(anima_dir, disable_status)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail=t("anima.status_json_invalid", name=name)) from exc

        # Always call stop_anima (no-op if not running). Under lifecycle lock
        # this waits for an in-flight start, then stops the new process.
        supervisor = request.app.state.supervisor
        await supervisor.stop_anima(name)
        if name in request.app.state.anima_names:
            request.app.state.anima_names.remove(name)

        return {"name": name, "enabled": False}

    @router.post("/animas/{name}/rename")
    async def rename_anima(name: str, request: Request):
        """Rename an Anima on root so the CLI never moves owned settings itself."""
        from core.anima.admin import cleanup_rag_collections, rename_dm_logs
        from core.anima.factory import validate_anima_name
        from core.config.anima_registry import rename_anima_in_config
        from core.paths import get_data_dir
        from core.platform.status_store import read_status

        _validate_anima_name(name)
        data = await request.json()
        new_name = data.get("new_name") if isinstance(data, dict) else None
        if not isinstance(new_name, str) or not new_name:
            raise HTTPException(status_code=400, detail="new_name is required")
        if new_name == name:
            raise HTTPException(status_code=400, detail="Old and new names are the same")
        name_error = validate_anima_name(new_name)
        if name_error:
            raise HTTPException(status_code=400, detail=name_error)

        data_dir = get_data_dir()
        animas_dir = request.app.state.animas_dir
        old_dir = animas_dir / name
        new_dir = animas_dir / new_name
        if not old_dir.is_dir() or not (old_dir / IDENTITY_FILE).is_file():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")
        if new_dir.exists():
            raise HTTPException(status_code=409, detail=f"Anima already exists: {new_name}")
        old_enabled = bool(read_status(old_dir).get("enabled", True))

        supervisor = request.app.state.supervisor
        try:
            await supervisor.stop_anima(name)
        except Exception as exc:
            logger.exception("Failed to stop anima before rename: %s", name)
            raise HTTPException(status_code=409, detail=f"Failed to stop anima '{name}': {exc}") from exc

        old_inbox = data_dir / "shared" / "inbox" / name
        new_inbox = data_dir / "shared" / "inbox" / new_name
        inbox_renamed = False
        directory_renamed = False
        try:
            old_dir.rename(new_dir)
            directory_renamed = True
            if old_inbox.exists():
                if new_inbox.exists():
                    raise FileExistsError(f"Inbox already exists: {new_inbox}")
                old_inbox.rename(new_inbox)
                inbox_renamed = True

            run_dir = data_dir / "run"
            for stale in (run_dir / "sockets" / f"{name}.sock", run_dir / "animas" / f"{name}.pid"):
                stale.unlink(missing_ok=True)

            dm_count = await asyncio.to_thread(rename_dm_logs, data_dir / "shared", name, new_name)
            repair_error = ""
            try:
                repair_queued = await asyncio.to_thread(cleanup_rag_collections, new_dir, name, source="server")
            except Exception as exc:
                repair_queued = False
                repair_error = str(exc)
                logger.warning("RAG cleanup failed after anima rename", exc_info=True)

            supervisor_references = rename_anima_in_config(data_dir, name, new_name)
            from core.anima.settings_store import update_status

            status_updated = 0
            for other_dir in sorted(path for path in animas_dir.iterdir() if path.is_dir()):
                if not (other_dir / "status.json").is_file():
                    continue
                changed = False

                def rename_supervisor(status: dict[str, Any]) -> None:
                    nonlocal changed
                    if status.get("supervisor") == name:
                        status["supervisor"] = new_name
                        changed = True

                try:
                    await asyncio.to_thread(update_status, other_dir, rename_supervisor)
                    status_updated += int(changed)
                except Exception:
                    logger.warning("Failed to rename supervisor reference in %s", other_dir.name, exc_info=True)
        except Exception as exc:
            logger.exception("Anima rename failed: %s -> %s", name, new_name)
            if inbox_renamed and new_inbox.exists() and not old_inbox.exists():
                try:
                    new_inbox.rename(old_inbox)
                except OSError:
                    logger.warning("Failed to restore inbox after rename failure", exc_info=True)
            if directory_renamed and new_dir.exists() and not old_dir.exists():
                try:
                    new_dir.rename(old_dir)
                except OSError:
                    logger.warning("Failed to restore anima directory after rename failure", exc_info=True)
            raise HTTPException(status_code=409, detail=f"Failed to rename anima: {exc}") from exc

        restart_ctl = getattr(supervisor, "_restart_ctl", None)
        forget = getattr(restart_ctl, "forget", None)
        if callable(forget):
            forget(name)
        anima_names = request.app.state.anima_names
        if name in anima_names:
            anima_names.remove(name)
        start_error = ""
        if old_enabled:
            if new_name not in anima_names:
                anima_names.append(new_name)
            try:
                await supervisor.start_anima(new_name)
            except Exception as exc:
                start_error = str(exc)
                logger.warning("Anima %s renamed but could not be restarted", new_name, exc_info=True)

        logger.info(
            "Anima renamed via root API: %s -> %s (supervisors=%d status=%d dm_logs=%d repair_queued=%s)",
            name,
            new_name,
            supervisor_references,
            status_updated,
            dm_count,
            repair_queued,
        )
        return {
            "status": "ok",
            "old_name": name,
            "new_name": new_name,
            "supervisor_references_updated": supervisor_references,
            "status_files_updated": status_updated,
            "dm_logs_renamed": dm_count,
            "repair_queued": repair_queued,
            "repair_error": repair_error,
            "start_error": start_error,
            "enabled": old_enabled,
        }

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
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Refuse start while disabled; enable API writes enabled=true then starts.
        from server.supervisor import ProcessSupervisor

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

    @router.post("/animas/{name}/bootstrap/repair")
    async def repair_anima_bootstrap(name: str, request: Request):
        """Recover an Anima whose first-run setup stopped in needs_repair.

        ``action=complete`` keeps the current identity/injection and marks the
        setup finished; ``action=retry`` prepares another bootstrap attempt.
        """
        from core.anima.bootstrap_state import (
            STATE_COMPLETED,
            repair_bootstrap_complete,
            repair_bootstrap_retry,
        )

        _validate_anima_name(name)
        anima_dir = request.app.state.animas_dir / name
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        data = await request.json()
        action = data.get("action") if isinstance(data, dict) else None
        if action not in {"complete", "retry"}:
            raise HTTPException(status_code=400, detail="action must be 'complete' or 'retry'")

        supervisor = request.app.state.supervisor
        retries_file = getattr(supervisor, "_bootstrap_retries_file", None)
        repair = repair_bootstrap_complete if action == "complete" else repair_bootstrap_retry
        status = await asyncio.to_thread(repair, anima_dir, retry_counts_file=retries_file)
        if action == "complete" and status.get("state") != STATE_COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=t("anima.bootstrap_complete_requires_identity", name=name),
            )

        retry_counts = getattr(supervisor, "_bootstrap_retry_counts", None)
        if isinstance(retry_counts, dict):
            retry_counts.pop(name, None)
        if name in request.app.state.anima_names:
            await supervisor.restart_anima(name)
        if action == "complete":
            await supervisor._broadcast_event(  # noqa: SLF001
                "anima.bootstrap",
                {"name": name, "status": "completed", "bootstrap_state": status},
            )
        return {"name": name, "action": action, "bootstrap_state": status}

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
    async def delete_anima(name: str, request: Request, archive: bool = True):
        """Stop and delete an anima entirely (process + files)."""
        _validate_anima_name(name)
        supervisor = request.app.state.supervisor
        animas_dir: Path = request.app.state.animas_dir
        anima_names: list[str] = request.app.state.anima_names
        anima_dir = animas_dir / name
        if not anima_dir.is_dir() or not (anima_dir / IDENTITY_FILE).is_file():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        stop_error: Exception | None = None
        if name in supervisor.processes:
            try:
                await supervisor.stop_anima(name)
            except Exception as exc:
                stop_error = exc
                logger.warning("Failed to stop anima '%s' before delete", name, exc_info=True)

            handle = supervisor.processes.get(name)
            if handle is not None and _process_handle_is_alive(handle):
                detail = f"Could not stop anima '{name}' before delete"
                if stop_error is not None:
                    detail = f"{detail}: {stop_error}"
                raise HTTPException(status_code=409, detail=detail)

        from server.services.anima_admin import delete_anima_files

        result = delete_anima_files(animas_dir.parent, name, archive=archive)
        if result.deleted:
            if name in anima_names:
                anima_names.remove(name)
            try:
                from core.anima.roster import refresh_anima_roster

                refresh_anima_roster()
            except Exception:
                logger.debug("Failed to refresh anima roster after delete", exc_info=True)

        if result.error:
            raise HTTPException(status_code=500, detail=f"Failed to delete anima '{name}': {result.error}")

        return {
            "status": "deleted",
            "name": name,
            "archive_path": str(result.archive_path) if result.archive_path is not None else None,
            "supervisor_warnings": [
                f"Anima '{other_name}' has deleted anima '{name}' as supervisor"
                for other_name in result.supervisor_references
            ],
        }

    # ── Interactive call_human resolve (authenticated UI) ──

    @router.post("/animas/{name}/interactions/{callback_id}/resolve")
    async def resolve_interaction(name: str, callback_id: str, request: Request):
        """Resolve an interactive call_human request from the main chat UI.

        Body: ``{"decision": "approve", "comment": ""}``
        """
        _validate_anima_name(name)

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from None

        decision = (body.get("decision") or "").strip() if isinstance(body, dict) else ""
        comment = (body.get("comment") or "") if isinstance(body, dict) else ""
        if not decision:
            raise HTTPException(status_code=400, detail="Missing decision")

        actor = "web_user"
        user = getattr(request.state, "user", None)
        if user and getattr(user, "username", None):
            actor = user.username

        from core.notification.interactive import get_interaction_router

        rtr = get_interaction_router()
        result = await rtr.resolve(
            callback_id,
            decision=decision,
            actor=actor,
            source="web",
            comment=comment,
        )
        if result is None:
            raise HTTPException(status_code=409, detail="Already resolved or expired")

        return {"status": "ok", "decision": decision}

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
                if anima_dir.is_dir() and (anima_dir / IDENTITY_FILE).exists() and anima_dir.name not in anima_names:
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
                logger.debug("Best-effort operation failed", exc_info=True)

            # Read department/title from status.json
            status_path = anima_dir / "status.json"
            if status_path.exists():
                try:
                    sdata = json.loads(status_path.read_text(encoding="utf-8"))
                    department = sdata.get("department", "")
                    title = sdata.get("title", "")
                    company = sdata.get("company", "")
                except Exception:
                    logger.debug("Best-effort operation failed", exc_info=True)

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
            from core.org.company import list_companies

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
