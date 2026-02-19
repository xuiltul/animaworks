# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Board WebUI API — channel and DM endpoints.

Tests the full flow through the FastAPI app with real filesystem
operations (no mocks for Messenger).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _create_app(shared_dir: Path):
    """Create a test FastAPI app with channels router and real filesystem."""
    from fastapi import FastAPI
    from server.routes.channels import create_channels_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_channels_router()
    app.include_router(router, prefix="/api")
    return app


def _seed_channels(shared_dir: Path) -> None:
    """Create initial channel files like animaworks init does."""
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    (channels_dir / "general.jsonl").touch()
    (channels_dir / "ops.jsonl").touch()
    (shared_dir / "dm_logs").mkdir(parents=True, exist_ok=True)


# ── Full Channel Flow ────────────────────────────────────


class TestE2EChannelFlow:
    """Tests the complete channel lifecycle: list → post → read → mentions."""

    async def test_full_channel_lifecycle(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. List channels — should have general and ops
            resp = await client.get("/api/channels")
            assert resp.status_code == 200
            channels = resp.json()
            names = [c["name"] for c in channels]
            assert "general" in names
            assert "ops" in names

            # 2. Post to general channel
            resp = await client.post(
                "/api/channels/general",
                json={"text": "Hello team!", "from_name": "taka"},
            )
            assert resp.status_code == 200

            resp = await client.post(
                "/api/channels/general",
                json={"text": "@sakura please check", "from_name": "taka"},
            )
            assert resp.status_code == 200

            resp = await client.post(
                "/api/channels/general",
                json={"text": "Acknowledged", "from_name": "anima_user"},
            )
            assert resp.status_code == 200

            # 3. Read channel messages
            resp = await client.get("/api/channels/general")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 3
            assert data["messages"][0]["text"] == "Hello team!"
            assert data["messages"][0]["source"] == "human"

            # 4. Check mentions
            resp = await client.get("/api/channels/general/mentions/sakura")
            assert resp.status_code == 200
            mentions = resp.json()
            assert mentions["count"] == 1
            assert "@sakura" in mentions["mentions"][0]["text"]

            # 5. Verify channel list updated counts
            resp = await client.get("/api/channels")
            channels = resp.json()
            general = next(c for c in channels if c["name"] == "general")
            assert general["message_count"] == 3

    async def test_websocket_broadcast_on_post(self, tmp_path: Path):
        """Verify that posting triggers WebSocket broadcast."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/channels/general",
                json={"text": "WS test message", "from_name": "taka"},
            )

        ws = app.state.ws_manager
        ws.broadcast.assert_awaited_once()
        event = ws.broadcast.call_args[0][0]
        assert event["type"] == "board.post"
        assert event["data"]["channel"] == "general"
        assert event["data"]["text"] == "WS test message"
        assert event["data"]["source"] == "human"
        assert event["data"]["from"] == "taka"

    async def test_human_source_verification(self, tmp_path: Path):
        """Posts via API always have source=human."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/channels/general",
                json={"text": "human post"},
            )
            resp = await client.get("/api/channels/general")

        messages = resp.json()["messages"]
        assert len(messages) == 1
        assert messages[0]["source"] == "human"
        assert messages[0]["from"] == "human"  # default from_name


# ── DM History Flow ──────────────────────────────────────


class TestE2EDMFlow:
    """Tests DM history listing and retrieval."""

    async def test_dm_history_with_messenger(self, tmp_path: Path):
        """Verify DM history reading works with real JSONL files."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        # Simulate DM conversation by writing JSONL directly
        dm_file = shared_dir / "dm_logs" / "alice-bob.jsonl"
        entries = [
            {"ts": "2026-02-17T10:00:00", "from": "alice", "text": "Hi Bob", "source": "anima"},
            {"ts": "2026-02-17T10:01:00", "from": "bob", "text": "Hi Alice", "source": "anima"},
            {"ts": "2026-02-17T10:02:00", "from": "alice", "text": "How are you?", "source": "anima"},
        ]
        dm_file.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
            encoding="utf-8",
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. List DM pairs
            resp = await client.get("/api/dm")
            assert resp.status_code == 200
            pairs = resp.json()
            assert len(pairs) == 1
            assert pairs[0]["pair"] == "alice-bob"
            assert pairs[0]["participants"] == ["alice", "bob"]
            assert pairs[0]["message_count"] == 3

            # 2. Get DM history
            resp = await client.get("/api/dm/alice-bob")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 3
            assert data["messages"][0]["from"] == "alice"
            assert data["messages"][2]["text"] == "How are you?"

            # 3. Get DM history with limit
            resp = await client.get("/api/dm/alice-bob?limit=2")
            data = resp.json()
            assert len(data["messages"]) == 2
            # Should return the LAST 2 messages
            assert data["messages"][0]["from"] == "bob"
            assert data["messages"][1]["from"] == "alice"


# ── Error Cases ──────────────────────────────────────────


class TestE2EErrorCases:
    async def test_post_to_nonexistent_channel(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/nonexistent",
                json={"text": "test"},
            )
        assert resp.status_code == 404

    async def test_get_nonexistent_dm_pair(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-charlie")
        assert resp.status_code == 404

    async def test_get_nonexistent_channel(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels/deleted")
        assert resp.status_code == 404


# ── Multi-Channel Isolation ──────────────────────────────


class TestE2EChannelIsolation:
    async def test_channels_are_isolated(self, tmp_path: Path):
        """Messages in one channel don't appear in another."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/channels/general",
                json={"text": "general msg"},
            )
            await client.post(
                "/api/channels/ops",
                json={"text": "ops msg"},
            )

            general_resp = await client.get("/api/channels/general")
            ops_resp = await client.get("/api/channels/ops")

        general_msgs = general_resp.json()["messages"]
        ops_msgs = ops_resp.json()["messages"]

        assert len(general_msgs) == 1
        assert general_msgs[0]["text"] == "general msg"
        assert len(ops_msgs) == 1
        assert ops_msgs[0]["text"] == "ops msg"
