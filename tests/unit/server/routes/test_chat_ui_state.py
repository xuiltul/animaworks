# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server/routes/chat_ui_state.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from server.routes.chat_ui_state import create_chat_ui_state_router


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request: Request, call_next):
        request.state.user = SimpleNamespace(username="alice")
        return await call_next(request)

    app.include_router(create_chat_ui_state_router(), prefix="/api")
    return app


async def test_get_returns_default_when_file_missing(tmp_path):
    app = _make_app()
    transport = ASGITransport(app=app)
    with patch("server.routes.chat_ui_state.get_shared_dir", return_value=tmp_path):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/chat/ui-state")
    assert resp.status_code == 200
    data = resp.json()["state"]
    assert data["version"] == 1
    assert data["active_anima"] is None
    assert data["anima_tabs"] == []
    assert data["thread_state"] == {}


async def test_put_and_get_roundtrip(tmp_path):
    app = _make_app()
    transport = ASGITransport(app=app)
    payload = {
        "state": {
            "active_anima": "mika",
            "anima_tabs": [{"name": "mika", "unread_star": True}],
            "thread_state": {
                "mika": {
                    "active_thread_id": "default",
                    "threads": [{"id": "default", "label": "メイン", "unread": True}],
                },
            },
        },
    }

    with patch("server.routes.chat_ui_state.get_shared_dir", return_value=tmp_path):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            put_resp = await client.put("/api/chat/ui-state", json=payload)
            assert put_resp.status_code == 200
            assert put_resp.json()["status"] == "ok"

            get_resp = await client.get("/api/chat/ui-state")
            assert get_resp.status_code == 200
            state = get_resp.json()["state"]
            assert state["active_anima"] == "mika"
            assert state["anima_tabs"][0]["name"] == "mika"
            assert state["thread_state"]["mika"]["threads"][0]["unread"] is True


async def test_returns_401_when_no_authenticated_user(tmp_path):
    app = FastAPI()
    app.include_router(create_chat_ui_state_router(), prefix="/api")
    transport = ASGITransport(app=app)

    class _NoOwner:
        owner = None

    with patch("server.routes.chat_ui_state.get_shared_dir", return_value=tmp_path), patch(
        "server.routes.chat_ui_state.load_auth",
        return_value=_NoOwner(),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/chat/ui-state")
            assert resp.status_code == 401
