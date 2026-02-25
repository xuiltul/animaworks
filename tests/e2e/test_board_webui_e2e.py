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
from unittest.mock import AsyncMock, MagicMock, patch

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

    @patch("core.config.models.load_config", side_effect=Exception("no config"))
    async def test_dm_history_with_messenger(self, _mock_cfg: MagicMock, tmp_path: Path):
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


# ── Reverse Pagination ───────────────────────────────────


class TestE2EReversePagination:
    """Tests reverse pagination: offset=0 returns newest N messages."""

    async def test_newest_messages_returned_first(self, tmp_path: Path):
        """Post 10 messages, then verify offset=0 returns the newest 5."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for i in range(10):
                await client.post(
                    "/api/channels/general",
                    json={"text": f"message-{i}", "from_name": "taka"},
                )

            resp = await client.get("/api/channels/general?limit=5&offset=0")
            data = resp.json()

        assert data["total"] == 10
        assert len(data["messages"]) == 5
        assert data["has_more"] is True
        texts = [m["text"] for m in data["messages"]]
        assert texts == [f"message-{i}" for i in range(5, 10)]

    async def test_full_traversal_with_real_posts(self, tmp_path: Path):
        """Post 75 messages, traverse all pages, verify all recovered."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        channel_file = shared_dir / "channels" / "general.jsonl"
        entries = [
            json.dumps({
                "ts": f"2026-02-25T{h:02d}:{m:02d}:00",
                "from": "sakura",
                "text": f"msg-{i}",
                "source": "anima",
            })
            for i, (h, m) in enumerate(
                [(h, m) for h in range(24) for m in range(0, 60, 1)][:75]
            )
        ]
        channel_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        all_texts: list[str] = []
        offset = 0
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            while True:
                resp = await client.get(
                    f"/api/channels/general?limit=50&offset={offset}"
                )
                data = resp.json()
                page_texts = [m["text"] for m in data["messages"]]
                all_texts = page_texts + all_texts
                offset += len(data["messages"])
                if not data["has_more"]:
                    break

        assert len(all_texts) == 75
        assert all_texts == [f"msg-{i}" for i in range(75)]

    async def test_three_page_traversal(self, tmp_path: Path):
        """Post 125 messages (3 pages), traverse all, verify none lost."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        channel_file = shared_dir / "channels" / "general.jsonl"
        entries = [
            json.dumps({
                "ts": f"2026-02-25T{i // 60:02d}:{i % 60:02d}:00",
                "from": "sakura",
                "text": f"msg-{i}",
                "source": "anima",
            })
            for i in range(125)
        ]
        channel_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        all_texts: list[str] = []
        offset = 0
        pages = 0
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            while True:
                resp = await client.get(
                    f"/api/channels/general?limit=50&offset={offset}"
                )
                data = resp.json()
                page_texts = [m["text"] for m in data["messages"]]
                all_texts = page_texts + all_texts
                offset += len(data["messages"])
                pages += 1
                if not data["has_more"]:
                    break

        assert pages == 3, f"Expected 3 pages, got {pages}"
        assert len(all_texts) == 125
        assert all_texts == [f"msg-{i}" for i in range(125)]

    async def test_small_channel_single_page(self, tmp_path: Path):
        """Channel with < limit messages returns all with has_more=false."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for i in range(3):
                await client.post(
                    "/api/channels/general",
                    json={"text": f"short-{i}", "from_name": "taka"},
                )

            resp = await client.get("/api/channels/general?limit=50&offset=0")
            data = resp.json()

        assert data["total"] == 3
        assert len(data["messages"]) == 3
        assert data["has_more"] is False
        texts = [m["text"] for m in data["messages"]]
        assert texts == ["short-0", "short-1", "short-2"]

    async def test_pagination_preserves_chronological_order(self, tmp_path: Path):
        """Each page's messages are in chronological order (old → new)."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _seed_channels(shared_dir)

        channel_file = shared_dir / "channels" / "general.jsonl"
        entries = [
            json.dumps({
                "ts": f"2026-02-25T{i:02d}:00:00",
                "from": "sakura",
                "text": f"msg-{i}",
                "source": "anima",
            })
            for i in range(20)
        ]
        channel_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Page 1: newest 10
            resp1 = await client.get("/api/channels/general?limit=10&offset=0")
            # Page 2: older 10
            resp2 = await client.get("/api/channels/general?limit=10&offset=10")

        page1 = resp1.json()["messages"]
        page2 = resp2.json()["messages"]

        page1_ts = [m["ts"] for m in page1]
        page2_ts = [m["ts"] for m in page2]
        assert page1_ts == sorted(page1_ts)
        assert page2_ts == sorted(page2_ts)
        assert page2_ts[-1] < page1_ts[0]


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
