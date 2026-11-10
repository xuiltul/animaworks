from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, value-free HTTP API for operator vault management."""

import asyncio
import json
import os
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from core.auth.manager import load_auth, verify_password
from core.config.io import find_plaintext_credentials, update_config
from core.config.onepassword import OnePasswordClient
from core.config.vault import SHARED_VAULT_SECTION
from core.config.vault_ops import (
    STATUS_CHANGED,
    STATUS_NEW,
    VaultOpsError,
    VaultOwnershipError,
    apply_sync,
    check,
    delete_value,
    entries_overview,
    last_check_summary,
    plan_sync,
    rollback,
    set_value,
)
from core.config.vault_sources import VaultSourceError, load_sources
from core.hub.members import last_seen, load_members, save_member, token_digest
from core.i18n import t
from core.paths import get_data_dir

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MISSING = object()


def _json(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers={"Cache-Control": "no-store"})


def _error(message: str, status_code: int) -> JSONResponse:
    return _json({"error": message}, status_code)


async def _request_body(request: Request) -> dict[str, Any] | JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return _error(t("vault.invalid_request"), 400)
    if not isinstance(payload, dict):
        return _error(t("vault.invalid_request"), 400)
    return payload


def _data_dir(request: Request) -> Path:
    shared_dir = getattr(request.app.state, "shared_dir", None)
    if shared_dir is not None:
        return Path(shared_dir).parent
    animas_dir = getattr(request.app.state, "animas_dir", None)
    if animas_dir is not None:
        return Path(animas_dir).parent
    return get_data_dir()


def _require_vault_writer(request: Request, password: str | None) -> str:
    """Return the audit actor or reject a write with a localized 403."""
    if not getattr(request.app.state, "setup_complete", True):
        raise HTTPException(status_code=403, detail=t("vault.write_setup"))

    auth_config = load_auth()
    if auth_config.auth_mode == "local_trust":
        return "local_trust"

    user = getattr(request.state, "user", None)
    if (
        getattr(request.state, "auth_via", None) != "session"
        or user is None
        or user.role != "owner"
        or not isinstance(password, str)
        or not isinstance(user.password_hash, str)
    ):
        raise HTTPException(status_code=403, detail=t("vault.write_forbidden"))

    try:
        verified = verify_password(password, user.password_hash)
    except Exception:
        verified = False
    if not verified:
        raise HTTPException(status_code=403, detail=t("vault.write_forbidden"))
    return f"gui:{user.username}"


def _writer_capability(request: Request, auth_config: Any) -> dict[str, Any]:
    if not getattr(request.app.state, "setup_complete", True):
        return {
            "allowed": False,
            "requires_password": auth_config.auth_mode != "local_trust",
            "reason": t("vault.write_setup"),
        }
    if auth_config.auth_mode == "local_trust":
        return {"allowed": True, "requires_password": False, "reason": None}
    user = getattr(request.state, "user", None)
    allowed = getattr(request.state, "auth_via", None) == "session" and bool(user) and user.role == "owner"
    return {
        "allowed": allowed,
        "requires_password": True,
        "reason": None if allowed else t("vault.write_forbidden"),
    }


def _read_authorized(request: Request, auth_config: Any) -> bool:
    if auth_config.auth_mode == "local_trust":
        return True
    user = getattr(request.state, "user", None)
    return bool(user and user.role in {"owner", "admin"}) and getattr(request.state, "auth_via", None) in {
        "session",
        "localhost",
    }


def _op_read_authorized(request: Request, auth_config: Any) -> bool:
    if auth_config.auth_mode == "local_trust":
        return True
    user = getattr(request.state, "user", None)
    return bool(user and user.role == "owner") and getattr(request.state, "auth_via", None) == "session"


async def _writer_actor(request: Request, password: str | None) -> tuple[str | None, JSONResponse | None]:
    try:
        actor = await asyncio.to_thread(_require_vault_writer, request, password)
    except HTTPException as exc:
        return None, _error(str(exc.detail), exc.status_code)
    return actor, None


async def _require_reader(request: Request, *, onepassword: bool = False) -> JSONResponse | None:
    try:
        auth_config = await asyncio.to_thread(load_auth)
    except Exception:
        return _error(t("vault.read_forbidden"), 403)
    authorized = _op_read_authorized(request, auth_config) if onepassword else _read_authorized(request, auth_config)
    if authorized:
        return None
    message = t("vault.op_forbidden") if onepassword else t("vault.read_forbidden")
    return _error(message, 403)


def _validated_keys(payload: dict[str, Any], *, required: bool) -> list[str] | None | JSONResponse:
    raw = payload.get("keys", _MISSING)
    if raw is _MISSING and not required:
        return None
    if not isinstance(raw, list) or any(not isinstance(key, str) or not _KEY_RE.fullmatch(key) for key in raw):
        return _error(t("vault.invalid_request"), 400)
    return raw


def _client(data_dir: Path) -> tuple[OnePasswordClient, str]:
    settings = load_sources(data_dir).onepassword
    client = OnePasswordClient(settings.command, settings.timeout_s, write_command=settings.write_command)
    return client, settings.command[0]


def _op_unavailable(command: str) -> JSONResponse:
    return _error(t("vault.op_not_found", command=command), 503)


def _audit(data_dir: Path, actor: str, action: str, section: str, key: str) -> None:
    path = data_dir / "vault_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at": datetime.now(UTC).isoformat(),
        "actor": actor,
        "action": action,
        "section": section,
        "key": key,
    }
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.fchmod(fd, 0o600)
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        while line:
            written = os.write(fd, line)
            if written <= 0:
                raise OSError("Could not append vault audit record")
            line = line[written:]
    finally:
        os.close(fd)


def _valid_key(key: str) -> bool:
    return bool(_KEY_RE.fullmatch(key))


def _plaintext_config(data_dir: Path) -> list[dict[str, str]]:
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        return []
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    return [{"path": item.path, "vault_key": item.vault_key} for item in find_plaintext_credentials(raw, data_dir)]


def _migrate_config(data_dir: Path) -> list[dict[str, str]]:
    before = _plaintext_config(data_dir)
    update_config(lambda config: None, path=data_dir / "config.json")
    after = _plaintext_config(data_dir)
    remaining = {(item["path"], item["vault_key"]) for item in after}
    return [item for item in before if (item["path"], item["vault_key"]) not in remaining]


def create_vault_router() -> APIRouter:
    router = APIRouter(tags=["vault"])

    @router.get("/vault/overview")
    async def vault_overview(request: Request) -> JSONResponse:
        denied = await _require_reader(request)
        if denied is not None:
            return denied
        try:
            data_dir = _data_dir(request)
            auth_config = await asyncio.to_thread(load_auth)
            sources = load_sources(data_dir)
            client = OnePasswordClient(
                sources.onepassword.command,
                sources.onepassword.timeout_s,
                write_command=sources.onepassword.write_command,
            )
            rows = await asyncio.to_thread(entries_overview, data_dir)
            last_check = last_check_summary(data_dir)
            anima_counts: dict[str, int] = {}
            for entry in rows:
                if entry.section != SHARED_VAULT_SECTION:
                    anima_counts[entry.section] = anima_counts.get(entry.section, 0) + 1

            hub_members: list[dict[str, Any]] = []
            from server.routes.hub import _hub_enabled

            if _hub_enabled():
                shared_dir = Path(getattr(request.app.state, "shared_dir", data_dir / "shared"))
                for member_id, member in sorted(load_members(shared_dir).items()):
                    route = member.message_route or {}
                    is_webhook = route.get("type") == "webhook"
                    hub_members.append(
                        {
                            "id": member_id,
                            "display_name": member.display_name,
                            "enabled": member.enabled,
                            "has_token": bool(member.token_sha256),
                            "last_seen": last_seen(shared_dir, member_id),
                            "webhook_url_key": route.get("url_ref") if is_webhook else None,
                            "webhook_key_key": route.get("key_ref") if is_webhook else None,
                        }
                    )

            return _json(
                {
                    "onepassword": {
                        "available": client.available(),
                        "command": sources.onepassword.command[0],
                        "sources_count": len(sources.entries),
                        "last_check_at": last_check.get("at") if last_check else None,
                    },
                    "entries": [
                        {
                            "section": row.section,
                            "key": row.key,
                            "owner": row.owner,
                            "is_set": row.is_set,
                            "updated_at": row.updated_at,
                            "updated_by": row.updated_by,
                            "last_synced_at": row.last_synced_at,
                            "check": row.last_check,
                        }
                        for row in rows
                        if row.section == SHARED_VAULT_SECTION
                    ],
                    "anima_sections": [
                        {"section": section, "count": count} for section, count in sorted(anima_counts.items())
                    ],
                    "hub_members": hub_members,
                    "plaintext_config": await asyncio.to_thread(_plaintext_config, data_dir),
                    "write": _writer_capability(request, auth_config),
                }
            )
        except Exception:
            return _error(t("vault.invalid_request"), 500)

    @router.post("/vault/check")
    async def vault_check(request: Request) -> JSONResponse:
        denied = await _require_reader(request, onepassword=True)
        if denied is not None:
            return denied
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        keys = _validated_keys(payload, required=False)
        if isinstance(keys, JSONResponse):
            return keys
        try:
            client, command = _client(_data_dir(request))
            if not client.available():
                return _op_unavailable(command)
            results = await asyncio.to_thread(check, _data_dir(request), client, keys)
            return _json({"results": results, "requests": client.request_count})
        except VaultSourceError:
            return _error(t("vault.invalid_request"), 400)
        except Exception:
            return _error(t("vault.op_error"), 500)

    @router.post("/vault/sync/plan")
    async def vault_sync_plan(request: Request) -> JSONResponse:
        denied = await _require_reader(request, onepassword=True)
        if denied is not None:
            return denied
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        keys = _validated_keys(payload, required=False)
        if isinstance(keys, JSONResponse):
            return keys
        try:
            data_dir = _data_dir(request)
            client, command = _client(data_dir)
            if not client.available():
                return _op_unavailable(command)
            plan = await asyncio.to_thread(plan_sync, data_dir, client, keys)
            items = [
                {
                    "key": item.key,
                    "status": item.status,
                    "error": t("vault.op_error") if item.error else None,
                }
                for item in plan
            ]
            return _json({"items": items, "requests": client.request_count})
        except VaultSourceError:
            return _error(t("vault.invalid_request"), 400)
        except Exception:
            return _error(t("vault.op_error"), 500)

    @router.post("/vault/sync/apply")
    async def vault_sync_apply(request: Request) -> JSONResponse:
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        keys = _validated_keys(payload, required=True)
        if isinstance(keys, JSONResponse):
            return keys
        password = payload.get("password")
        if password is not None and not isinstance(password, str):
            return _error(t("vault.invalid_request"), 400)
        actor, denied = await _writer_actor(request, password)
        if denied is not None:
            return denied
        try:
            data_dir = _data_dir(request)
            client, command = _client(data_dir)
            if not client.available():
                return _op_unavailable(command)
            # Always compare against a fresh server-side 1Password read. The
            # caller's preview contains no trusted synchronization state.
            plan = await asyncio.to_thread(plan_sync, data_dir, client, keys)
            applied: list[str] = []
            for item in plan:
                if item.status not in (STATUS_NEW, STATUS_CHANGED):
                    continue
                written = await asyncio.to_thread(apply_sync, data_dir, [item], actor="sync")
                if written:
                    applied.extend(written)
                    await asyncio.to_thread(_audit, data_dir, actor, "sync_apply", SHARED_VAULT_SECTION, item.key)
            skipped = [item.key for item in plan if item.key not in applied]
            return _json({"applied": applied, "skipped": skipped, "requests": client.request_count})
        except VaultSourceError:
            return _error(t("vault.invalid_request"), 400)
        except Exception:
            return _error(t("vault.op_error"), 500)

    @router.put("/vault/shared/{key}")
    async def vault_set_shared(key: str, request: Request) -> JSONResponse:
        if not _valid_key(key):
            return _error(t("vault.invalid_request"), 400)
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        password = payload.get("password")
        value = payload.get("value")
        if password is not None and not isinstance(password, str):
            return _error(t("vault.invalid_request"), 400)
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 64 * 1024:
            return _error(t("vault.invalid_request"), 400)
        actor, denied = await _writer_actor(request, password)
        if denied is not None:
            return denied
        data_dir = _data_dir(request)
        try:
            await asyncio.to_thread(set_value, data_dir, SHARED_VAULT_SECTION, key, value, actor=actor)
            await asyncio.to_thread(_audit, data_dir, actor, "set", SHARED_VAULT_SECTION, key)
            return _json({"key": key, "stored": True})
        except VaultOwnershipError:
            return _error(t("vault.managed_store", name=key), 409)
        except VaultOpsError:
            return _error(t("vault.invalid_request"), 400)
        except Exception:
            return _error(t("vault.invalid_request"), 500)

    @router.post("/vault/shared/{key}/delete")
    async def vault_delete_shared(key: str, request: Request) -> JSONResponse:
        if not _valid_key(key):
            return _error(t("vault.invalid_request"), 400)
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        password = payload.get("password")
        if password is not None and not isinstance(password, str):
            return _error(t("vault.invalid_request"), 400)
        actor, denied = await _writer_actor(request, password)
        if denied is not None:
            return denied
        data_dir = _data_dir(request)
        try:
            deleted = await asyncio.to_thread(delete_value, data_dir, SHARED_VAULT_SECTION, key, actor=actor)
            if not deleted:
                return _error(t("vault.invalid_request"), 404)
            await asyncio.to_thread(_audit, data_dir, actor, "delete", SHARED_VAULT_SECTION, key)
            return _json({"key": key, "deleted": True})
        except VaultOwnershipError:
            return _error(t("vault.managed_delete", name=key), 409)
        except VaultOpsError:
            return _error(t("vault.invalid_request"), 400)
        except Exception:
            return _error(t("vault.invalid_request"), 500)

    @router.post("/vault/shared/{key}/rollback")
    async def vault_rollback_shared(key: str, request: Request) -> JSONResponse:
        if not _valid_key(key):
            return _error(t("vault.invalid_request"), 400)
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        password = payload.get("password")
        if password is not None and not isinstance(password, str):
            return _error(t("vault.invalid_request"), 400)
        actor, denied = await _writer_actor(request, password)
        if denied is not None:
            return denied
        data_dir = _data_dir(request)
        try:
            restored = await asyncio.to_thread(rollback, data_dir, key, actor=actor)
            if not restored:
                return _error(t("vault.rollback_missing", name=key), 404)
            await asyncio.to_thread(_audit, data_dir, actor, "rollback", SHARED_VAULT_SECTION, key)
            return _json({"key": key, "rolled_back": True})
        except VaultOpsError:
            return _error(t("vault.invalid_request"), 400)
        except Exception:
            return _error(t("vault.invalid_request"), 500)

    @router.post("/vault/hub-members/{member_id}/reissue-token")
    async def vault_reissue_hub_token(member_id: str, request: Request) -> JSONResponse:
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        password = payload.get("password")
        if password is not None and not isinstance(password, str):
            return _error(t("vault.invalid_request"), 400)
        actor, denied = await _writer_actor(request, password)
        if denied is not None:
            return denied
        try:
            from server.routes.hub import _hub_enabled

            if not _hub_enabled():
                return _error(t("vault.invalid_request"), 404)
            shared_dir = Path(getattr(request.app.state, "shared_dir", _data_dir(request) / "shared"))
            members = await asyncio.to_thread(load_members, shared_dir)
            member = members.get(member_id)
            if member is None:
                return _error(t("vault.invalid_request"), 404)
            token = secrets.token_urlsafe(32)
            member.token_sha256 = token_digest(token)
            await asyncio.to_thread(save_member, shared_dir, member)
            await asyncio.to_thread(_audit, _data_dir(request), actor, "reissue_token", "hub_members", member_id)
            return _json({"token": token})
        except Exception:
            return _error(t("vault.invalid_request"), 500)

    @router.post("/vault/migrate-config")
    async def vault_migrate_config(request: Request) -> JSONResponse:
        payload = await _request_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        password = payload.get("password")
        if password is not None and not isinstance(password, str):
            return _error(t("vault.invalid_request"), 400)
        actor, denied = await _writer_actor(request, password)
        if denied is not None:
            return denied
        data_dir = _data_dir(request)
        try:
            moved = await asyncio.to_thread(_migrate_config, data_dir)
            await asyncio.to_thread(_audit, data_dir, actor, "migrate_config", "config", "credentials")
            return _json({"moved": moved})
        except PermissionError:
            return _error(t("vault.migrate_server_running"), 409)
        except Exception:
            return _error(t("vault.invalid_request"), 500)

    return router
