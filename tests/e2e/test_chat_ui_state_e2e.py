# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for dashboard chat UI state persistence API."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.auth.models import AuthConfig, AuthUser
from server.routes.chat_ui_state import create_chat_ui_state_router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(create_chat_ui_state_router(), prefix="/api")
    return app


def _auth_owner(username: str) -> AuthConfig:
    return AuthConfig(owner=AuthUser(username=username, role="owner"))


async def test_local_trust_owner_fallback_persists_per_user(tmp_path):
    app = _make_app()
    transport = ASGITransport(app=app)
    payload = {
        "state": {
            "active_anima": "sora",
            "anima_tabs": [{"name": "sora", "unread_star": False}],
            "thread_state": {
                "sora": {
                    "active_thread_id": "default",
                    "threads": [{"id": "default", "label": "メイン", "unread": False}],
                },
            },
        },
    }

    with patch("server.routes.chat_ui_state.get_shared_dir", return_value=tmp_path), patch(
        "server.routes.chat_ui_state.load_auth",
        return_value=_auth_owner("owner_user"),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            put_resp = await client.put("/api/chat/ui-state", json=payload)
            assert put_resp.status_code == 200

            get_resp = await client.get("/api/chat/ui-state")
            assert get_resp.status_code == 200
            state = get_resp.json()["state"]
            assert state["active_anima"] == "sora"

    expected_path = tmp_path / "users" / "owner_user" / "chat_ui_state.json"
    assert expected_path.exists()
