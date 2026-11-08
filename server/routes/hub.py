from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only HTTP API for hub member avatars."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from core.config.models import load_config
from core.hub.avatar import avatar_path
from core.hub.members import MEMBER_ID_RE, load_members
from core.paths import get_shared_dir


def _shared_dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "shared_dir", None) or get_shared_dir())


def _hub_enabled() -> bool:
    try:
        return bool(load_config().hub.enabled)
    except Exception:
        return False


def create_hub_router() -> APIRouter:
    router = APIRouter()

    @router.get("/hub/members")
    def list_hub_members(request: Request) -> list[dict[str, object]]:
        if not _hub_enabled():
            return []
        shared_dir = _shared_dir(request)
        result: list[dict[str, object]] = []
        for member_id, member in sorted(load_members(shared_dir).items()):
            path = avatar_path(shared_dir, member_id)
            avatar_url: str | None = None
            try:
                if path.is_file():
                    avatar_url = f"/api/hub/members/{member_id}/avatar?v={int(path.stat().st_mtime)}"
            except OSError:
                avatar_url = None
            result.append(
                {
                    "id": member.id,
                    "display_name": member.display_name,
                    "enabled": member.enabled,
                    "avatar_url": avatar_url,
                }
            )
        return result

    @router.get("/hub/members/{member_id}/avatar")
    def hub_member_avatar(member_id: str, request: Request) -> FileResponse:
        if not MEMBER_ID_RE.fullmatch(member_id) or not _hub_enabled():
            raise HTTPException(status_code=404, detail="not_found")
        path = avatar_path(_shared_dir(request), member_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not_found")
        return FileResponse(path, media_type="image/png")

    return router
