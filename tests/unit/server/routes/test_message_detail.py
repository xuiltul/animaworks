"""Unit tests for GET /api/messages/{message_id} endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(shared_dir: Path):
    from fastapi import FastAPI
    from server.routes.internal import create_internal_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_internal_router()
    app.include_router(router, prefix="/api")
    return app


def _create_message_file(shared_dir: Path, to_person: str, message_id: str, *, processed: bool = True) -> dict:
    """Helper: create a message JSON file in the inbox directory."""
    if processed:
        target_dir = shared_dir / "inbox" / to_person / "processed"
    else:
        target_dir = shared_dir / "inbox" / to_person
    target_dir.mkdir(parents=True, exist_ok=True)

    msg_data = {
        "id": message_id,
        "thread_id": message_id,
        "reply_to": "",
        "from_person": "sakura",
        "to_person": to_person,
        "type": "message",
        "content": "## Test Report\n\nThis is a **test message** with markdown.",
        "attachments": [],
        "timestamp": "2026-02-17T12:15:15.353847",
    }
    filepath = target_dir / f"{message_id}.json"
    filepath.write_text(json.dumps(msg_data, ensure_ascii=False), encoding="utf-8")
    return msg_data


class TestGetMessage:
    async def test_get_processed_message(self, tmp_path):
        """Should return full message JSON for a processed message."""
        msg_data = _create_message_file(tmp_path, "kotoha", "20260217_121515_353828")
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/20260217_121515_353828")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "20260217_121515_353828"
        assert data["from_person"] == "sakura"
        assert data["to_person"] == "kotoha"
        assert "## Test Report" in data["content"]

    async def test_get_unread_message(self, tmp_path):
        """Should find messages in unread inbox (not just processed)."""
        msg_data = _create_message_file(tmp_path, "kotoha", "20260217_130000_000001", processed=False)
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/20260217_130000_000001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "20260217_130000_000001"

    async def test_message_not_found(self, tmp_path):
        """Should return 404 for non-existent message."""
        (tmp_path / "inbox" / "kotoha" / "processed").mkdir(parents=True)
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/nonexistent_id")
        assert resp.status_code == 404

    async def test_no_inbox_dir(self, tmp_path):
        """Should return 404 when inbox directory doesn't exist."""
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/20260217_121515_353828")
        assert resp.status_code == 404

    async def test_path_traversal_slash(self, tmp_path):
        """Should reject message_id with path separators."""
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/../../../etc/passwd")
        assert resp.status_code != 200

    async def test_path_traversal_dotdot(self, tmp_path):
        """Should reject message_id containing '..'."""
        (tmp_path / "inbox" / "kotoha" / "processed").mkdir(parents=True)
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/..%2F..%2Fetc%2Fpasswd")
        # The URL-encoded path should either be rejected or not find anything
        assert resp.status_code in (400, 404)

    async def test_processed_preferred_over_unread(self, tmp_path):
        """When same message exists in both processed and unread, processed should be found."""
        _create_message_file(tmp_path, "kotoha", "20260217_121000_000001", processed=True)
        _create_message_file(tmp_path, "kotoha", "20260217_121000_000001", processed=False)
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/20260217_121000_000001")
        assert resp.status_code == 200

    async def test_message_id_broadcast_includes_id(self, tmp_path):
        """POST /internal/message-sent should include message_id in broadcast."""
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={
                    "from_person": "sakura",
                    "to_person": "kotoha",
                    "content": "Hello",
                    "message_id": "20260217_121515_353828",
                },
            )
        assert resp.status_code == 200
        ws = app.state.ws_manager
        ws.broadcast.assert_awaited_once()
        call_data = ws.broadcast.call_args[0][0]
        assert call_data["data"]["message_id"] == "20260217_121515_353828"

    async def test_search_across_multiple_animas(self, tmp_path):
        """Should search across all anima inboxes to find the message."""
        # Message in sakura's inbox, not kotoha's
        _create_message_file(tmp_path, "sakura", "20260217_140000_000001")
        # Also create empty kotoha inbox
        (tmp_path / "inbox" / "kotoha" / "processed").mkdir(parents=True)
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/messages/20260217_140000_000001")
        assert resp.status_code == 200
        assert resp.json()["to_person"] == "sakura"
