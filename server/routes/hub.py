from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only HTTP API for the hub roster and hub page metadata."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from core.config.models import load_config
from core.hub.avatar import avatar_path
from core.hub.members import MEMBER_ID_RE, last_seen, load_members
from core.hub.silence import is_overdue
from core.paths import get_shared_dir
from server.routes.channels import _human_unread, _request_username


def _shared_dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "shared_dir", None) or get_shared_dir())


def _hub_config() -> object | None:
    try:
        return load_config().hub
    except Exception:
        return None


def _hub_enabled() -> bool:
    config = _hub_config()
    return bool(getattr(config, "enabled", False))


def create_hub_router() -> APIRouter:
    router = APIRouter()

    @router.get("/hub/info")
    def hub_info(request: Request) -> dict[str, object]:
        config = _hub_config()
        if config is None or not bool(getattr(config, "enabled", False)):
            return {"enabled": False}
        board = str(getattr(config, "board", "") or "hub")
        unread = 0
        try:
            unread = int(_human_unread(_shared_dir(request), board, _request_username(request)))
        except Exception:
            unread = 0
        return {
            "enabled": True,
            "board": board,
            "company": str(getattr(config, "company", "") or ""),
            "timeline_enabled": bool(getattr(config, "timeline_enabled", False)),
            "unread": unread,
        }

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
            route = member.message_route if isinstance(member.message_route, dict) else None
            result.append(
                {
                    "id": member.id,
                    "display_name": member.display_name,
                    "kind": member.kind,
                    "enabled": member.enabled,
                    "avatar_url": avatar_url,
                    "route_type": route.get("type") if route is not None else None,
                    "expect_every": member.expect_every,
                    "last_seen": last_seen(shared_dir, member_id),
                    "overdue": is_overdue(shared_dir, member),
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
