"""Unit tests for server/routes/internal.py — Internal API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app():
    from fastapi import FastAPI
    from server.routes.internal import create_internal_router

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_internal_router()
    app.include_router(router, prefix="/api")
    return app


# ── POST /internal/message-sent ──────────────────────────


class TestInternalMessageSent:
    async def test_message_sent_broadcasts(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={
                    "from_person": "alice",
                    "to_person": "bob",
                    "content": "Hello Bob",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        ws = app.state.ws_manager
        ws.broadcast.assert_awaited_once()
        call_data = ws.broadcast.call_args[0][0]
        assert call_data["type"] == "anima.interaction"
        assert call_data["data"]["from_person"] == "alice"
        assert call_data["data"]["to_person"] == "bob"

    async def test_message_sent_no_anima_match(self):
        """Non-managed anima as sender should not crash."""
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={
                    "from_person": "external",
                    "to_person": "alice",
                    "content": "Hi",
                },
            )
        assert resp.status_code == 200

    async def test_message_sent_truncates_content(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        long_content = "x" * 500
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={
                    "from_person": "alice",
                    "to_person": "bob",
                    "content": long_content,
                },
            )
        assert resp.status_code == 200
        ws = app.state.ws_manager
        call_data = ws.broadcast.call_args[0][0]
        # The summary should be truncated to 200 chars
        assert len(call_data["data"]["summary"]) <= 200

    async def test_message_sent_missing_fields(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={"from_person": "alice"},
            )
        # Pydantic validation error
        assert resp.status_code == 422

    async def test_message_sent_default_content(self):
        """Content field has a default of empty string."""
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={"from_person": "alice", "to_person": "bob"},
            )
        assert resp.status_code == 200
