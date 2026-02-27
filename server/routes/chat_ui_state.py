from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Dashboard chat UI state persistence routes."""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.auth.manager import load_auth
from core.paths import get_shared_dir

logger = logging.getLogger("animaworks.routes.chat_ui_state")

_STATE_FILE_NAME = "chat_ui_state.json"
_STATE_VERSION = 1


class ChatUiStateUpdateRequest(BaseModel):
    """Chat UI state payload from frontend."""

    state: dict[str, Any] = Field(default_factory=dict)


def _resolve_username(request: Request) -> str:
    """Resolve current username with local_trust fallback."""
    user = getattr(request.state, "user", None)
    if user and getattr(user, "username", None):
        return user.username

    auth = load_auth()
    if auth.owner and auth.owner.username:
        return auth.owner.username
    raise HTTPException(status_code=401, detail="Not authenticated")


def _state_path_for_user(username: str):
    return get_shared_dir() / "users" / username / _STATE_FILE_NAME


def _default_state() -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "active_anima": None,
        "anima_tabs": [],
        "thread_state": {},
    }


def create_chat_ui_state_router() -> APIRouter:
    router = APIRouter(tags=["chat-ui-state"])

    @router.get("/chat/ui-state")
    async def get_chat_ui_state(request: Request):
        """Get persisted dashboard chat UI state for current user."""
        username = _resolve_username(request)
        state_path = _state_path_for_user(username)
        if not state_path.exists():
            return {"state": _default_state()}

        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("Failed to read chat ui state for %s: %s", username, err)
            return {"state": _default_state()}

        if not isinstance(data, dict):
            return {"state": _default_state()}

        merged = _default_state()
        merged.update(data)
        return {"state": merged}

    @router.put("/chat/ui-state")
    async def put_chat_ui_state(body: ChatUiStateUpdateRequest, request: Request):
        """Persist dashboard chat UI state for current user."""
        username = _resolve_username(request)
        state_path = _state_path_for_user(username)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        merged = _default_state()
        if isinstance(body.state, dict):
            merged.update(body.state)
        merged["version"] = _STATE_VERSION

        try:
            state_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as err:
            logger.error("Failed to save chat ui state for %s: %s", username, err)
            return {"status": "error"}

        return {"status": "ok"}

    return router
