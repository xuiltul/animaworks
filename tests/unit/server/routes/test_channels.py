"""Unit tests for server/routes/channels.py — Board channel and DM API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(shared_dir: Path):
    from fastapi import FastAPI
    from server.routes.channels import create_channels_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_channels_router()
    app.include_router(router, prefix="/api")
    return app


def _write_channel(shared_dir: Path, name: str, entries: list[dict]) -> None:
    """Write JSONL entries to a channel file."""
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    filepath = channels_dir / f"{name}.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dm(shared_dir: Path, pair: str, entries: list[dict]) -> None:
    """Write JSONL entries to a DM log file."""
    dm_dir = shared_dir / "dm_logs"
    dm_dir.mkdir(parents=True, exist_ok=True)
    filepath = dm_dir / f"{pair}.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_activity_log(
    data_dir: Path, anima_name: str, date_str: str, entries: list[dict],
) -> None:
    """Write JSONL entries to an Anima's activity_log directory.

    Args:
        data_dir: Root data directory (parent of both ``shared/`` and ``animas/``).
        anima_name: Name of the Anima.
        date_str: ISO date string (e.g. "2026-02-18").
        entries: List of activity entry dicts.
    """
    log_dir = data_dir / "animas" / anima_name / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / f"{date_str}.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _today() -> str:
    """Return today's date as ISO string for activity_log filenames."""
    return date.today().isoformat()


# ── GET /api/channels ────────────────────────────────────


class TestListChannels:
    async def test_empty_when_no_channels_dir(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_lists_channels_with_metadata(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [
            {"ts": "2026-02-17T10:00:00", "from": "sakura", "text": "Hello", "source": "anima"},
            {"ts": "2026-02-17T11:00:00", "from": "taka", "text": "Hi", "source": "human"},
        ])
        _write_channel(shared_dir, "ops", [
            {"ts": "2026-02-17T09:00:00", "from": "mio", "text": "Server OK", "source": "anima"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        general = next(c for c in data if c["name"] == "general")
        assert general["message_count"] == 2
        assert general["last_post_ts"] == "2026-02-17T11:00:00"

        ops = next(c for c in data if c["name"] == "ops")
        assert ops["message_count"] == 1


# ── GET /api/channels/{name} ────────────────────────────


class TestGetChannelMessages:
    async def test_returns_messages(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": f"2026-02-17T{h:02d}:00:00", "from": "sakura", "text": f"msg{h}", "source": "anima"}
            for h in range(5)
        ]
        _write_channel(shared_dir, "general", entries)

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels/general?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["channel"] == "general"
        assert len(data["messages"]) == 3
        assert data["total"] == 5
        assert data["has_more"] is True

    async def test_offset_pagination(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": f"2026-02-17T{h:02d}:00:00", "from": "sakura", "text": f"msg{h}", "source": "anima"}
            for h in range(5)
        ]
        _write_channel(shared_dir, "general", entries)

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels/general?limit=2&offset=3")
        data = resp.json()
        assert len(data["messages"]) == 2
        assert data["offset"] == 3

    async def test_404_for_nonexistent_channel(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels/nonexistent")
        assert resp.status_code == 404

    async def test_400_for_invalid_channel_name(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Uppercase name violates ^[a-z][a-z0-9_-]{0,30}$
            resp = await client.get("/api/channels/INVALID")
        assert resp.status_code == 400

    async def test_empty_channel_returns_empty_list(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels/general")
        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == []


# ── POST /api/channels/{name} ───────────────────────────


class TestPostToChannel:
    async def test_human_post_success(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        # Create empty channel file
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={"text": "Hello from human", "from_name": "taka"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify message was written to channel file
        channel_file = shared_dir / "channels" / "general.jsonl"
        lines = channel_file.read_text(encoding="utf-8").strip().splitlines()
        # Filter out empty lines from the initial empty write
        lines = [l for l in lines if l.strip()]
        assert len(lines) >= 1
        last = json.loads(lines[-1])
        assert last["text"] == "Hello from human"
        assert last["source"] == "human"
        assert last["from"] == "taka"

    async def test_post_broadcasts_websocket_event(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/channels/general",
                json={"text": "broadcast test"},
            )

        ws = app.state.ws_manager
        ws.broadcast.assert_awaited_once()
        call_data = ws.broadcast.call_args[0][0]
        assert call_data["type"] == "board.post"
        assert call_data["data"]["channel"] == "general"
        assert call_data["data"]["text"] == "broadcast test"
        assert call_data["data"]["source"] == "human"

    async def test_post_to_nonexistent_channel_returns_404(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/nonexistent",
                json={"text": "test"},
            )
        assert resp.status_code == 404

    async def test_post_default_from_name(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={"text": "no from_name"},
            )
        assert resp.status_code == 200

        ws = app.state.ws_manager
        call_data = ws.broadcast.call_args[0][0]
        assert call_data["data"]["from"] == "human"

    async def test_post_missing_text_returns_422(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={},
            )
        assert resp.status_code == 422

    async def test_post_400_for_invalid_channel_name(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/INVALID_NAME",
                json={"text": "test"},
            )
        assert resp.status_code == 400


# ── GET /api/channels/{name}/mentions/{anima} ────────────


class TestGetChannelMentions:
    async def test_returns_mentions(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [
            {"ts": "2026-02-17T10:00:00", "from": "mio", "text": "@sakura check this", "source": "anima"},
            {"ts": "2026-02-17T11:00:00", "from": "kotoha", "text": "No mention here", "source": "anima"},
            {"ts": "2026-02-17T12:00:00", "from": "taka", "text": "@sakura urgent", "source": "human"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels/general/mentions/sakura")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert all("@sakura" in m["text"] for m in data["mentions"])

    async def test_no_mentions_returns_empty(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [
            {"ts": "2026-02-17T10:00:00", "from": "mio", "text": "Hello", "source": "anima"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/channels/general/mentions/sakura")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ── GET /api/dm ──────────────────────────────────────────


class TestListDMPairs:
    async def test_empty_when_no_dm_dir(self, tmp_path: Path):
        """No animas dir and no dm_logs dir returns empty list."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_lists_dm_pairs(self, tmp_path: Path):
        """Legacy dm_logs entries are picked up by the fallback path."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_dm(shared_dir, "alice-bob", [
            {"ts": "2026-02-17T10:00:00", "from": "alice", "text": "Hi Bob", "source": "anima"},
            {"ts": "2026-02-17T11:00:00", "from": "bob", "text": "Hi Alice", "source": "anima"},
        ])
        _write_dm(shared_dir, "mio-sakura", [
            {"ts": "2026-02-17T09:00:00", "from": "mio", "text": "Report", "source": "anima"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        ab = next(p for p in data if p["pair"] == "alice-bob")
        assert ab["message_count"] == 2
        assert ab["participants"] == ["alice", "bob"]
        assert ab["last_message_ts"] == "2026-02-17T11:00:00"

    async def test_lists_dm_pairs_from_activity_log(self, tmp_path: Path):
        """Activity log entries (primary source) are aggregated into DM pairs."""
        data_dir = tmp_path
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        today = _today()

        # Alice sent 2 DMs to Bob
        _write_activity_log(data_dir, "alice", today, [
            {"ts": f"{today}T10:00:00", "type": "dm_sent", "content": "Hello Bob", "from": "alice", "to": "bob"},
            {"ts": f"{today}T10:05:00", "type": "dm_sent", "content": "Are you there?", "from": "alice", "to": "bob"},
        ])
        # Bob received them and sent a reply
        _write_activity_log(data_dir, "bob", today, [
            {"ts": f"{today}T10:00:00", "type": "dm_received", "content": "Hello Bob", "from": "alice", "to": "bob"},
            {"ts": f"{today}T10:10:00", "type": "dm_sent", "content": "Yes I am here", "from": "bob", "to": "alice"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

        pair = data[0]
        assert pair["pair"] == "alice-bob"
        assert set(pair["participants"]) == {"alice", "bob"}
        # Total count: 2 from alice's log + 2 from bob's log = 4
        assert pair["message_count"] == 4
        assert pair["last_message_ts"] == f"{today}T10:10:00"

    async def test_merges_activity_log_and_legacy_dm_logs(self, tmp_path: Path):
        """Activity log entries and legacy dm_logs are merged into a single pair."""
        data_dir = tmp_path
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        today = _today()

        # Activity log: 1 entry from alice
        _write_activity_log(data_dir, "alice", today, [
            {"ts": f"{today}T12:00:00", "type": "dm_sent", "content": "New message", "from": "alice", "to": "bob"},
        ])

        # Legacy dm_logs: 2 older entries
        _write_dm(shared_dir, "alice-bob", [
            {"ts": "2026-01-15T10:00:00", "from": "alice", "text": "Old msg 1", "source": "anima"},
            {"ts": "2026-01-15T11:00:00", "from": "bob", "text": "Old msg 2", "source": "anima"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

        pair = data[0]
        assert pair["pair"] == "alice-bob"
        # 1 from activity_log + 2 from legacy dm_logs = 3
        assert pair["message_count"] == 3
        # The activity_log entry is newer
        assert pair["last_message_ts"] == f"{today}T12:00:00"

    async def test_empty_when_no_data_sources(self, tmp_path: Path):
        """No animas dir and no dm_logs dir returns empty list."""
        data_dir = tmp_path
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        # Do NOT create animas/ or dm_logs/

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")
        assert resp.status_code == 200
        assert resp.json() == []


# ── GET /api/dm/{pair} ───────────────────────────────────


class TestGetDMHistory:
    async def test_returns_dm_messages(self, tmp_path: Path):
        """Legacy dm_logs are returned via the fallback path."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": f"2026-02-17T{h:02d}:00:00", "from": "alice" if h % 2 == 0 else "bob", "text": f"msg{h}", "source": "anima"}
            for h in range(10)
        ]
        _write_dm(shared_dir, "alice-bob", entries)

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "alice-bob"
        assert len(data["messages"]) == 5
        assert data["total"] == 10

    async def test_returns_dm_from_activity_log(self, tmp_path: Path):
        """Activity log entries are returned as the primary DM source."""
        data_dir = tmp_path
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        today = _today()

        _write_activity_log(data_dir, "alice", today, [
            {"ts": f"{today}T10:00:00", "type": "dm_sent", "content": "Hello Bob", "from": "alice", "to": "bob"},
            {"ts": f"{today}T10:05:00", "type": "dm_sent", "content": "How are you?", "from": "alice", "to": "bob"},
        ])
        _write_activity_log(data_dir, "bob", today, [
            {"ts": f"{today}T10:10:00", "type": "dm_sent", "content": "I am fine", "from": "bob", "to": "alice"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "alice-bob"
        assert data["total"] == 3
        assert len(data["messages"]) == 3

        # Messages should be sorted chronologically
        timestamps = [m["ts"] for m in data["messages"]]
        assert timestamps == sorted(timestamps)

        # Verify content
        texts = [m["text"] for m in data["messages"]]
        assert "Hello Bob" in texts
        assert "How are you?" in texts
        assert "I am fine" in texts

    async def test_deduplicates_entries(self, tmp_path: Path):
        """Same message in both participants' activity logs appears only once."""
        data_dir = tmp_path
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        today = _today()

        # Alice's log has dm_sent
        _write_activity_log(data_dir, "alice", today, [
            {"ts": f"{today}T10:00:00", "type": "dm_sent", "content": "Hello Bob", "from": "alice", "to": "bob"},
        ])
        # Bob's log has dm_received for the same message (same ts + content)
        _write_activity_log(data_dir, "bob", today, [
            {"ts": f"{today}T10:00:00", "type": "dm_received", "content": "Hello Bob", "from": "alice", "to": "bob"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=50")
        assert resp.status_code == 200
        data = resp.json()
        # Deduplication by "ts|content" key should yield 1 unique message
        assert data["total"] == 1
        assert len(data["messages"]) == 1
        assert data["messages"][0]["text"] == "Hello Bob"

    async def test_falls_back_to_legacy_dm_logs(self, tmp_path: Path):
        """When no activity_log exists, legacy dm_logs are returned."""
        data_dir = tmp_path
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        # No animas dir at all

        _write_dm(shared_dir, "alice-bob", [
            {"ts": "2026-01-15T10:00:00", "from": "alice", "text": "Legacy hello", "source": "anima"},
            {"ts": "2026-01-15T11:00:00", "from": "bob", "text": "Legacy reply", "source": "anima"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "alice-bob"
        assert data["total"] == 2
        assert data["messages"][0]["text"] == "Legacy hello"

    async def test_404_when_no_data_anywhere(self, tmp_path: Path):
        """No activity_log and no dm_logs for a pair returns 404."""
        data_dir = tmp_path
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        # Create animas dir but no activity_log for alice/bob
        (data_dir / "animas" / "alice").mkdir(parents=True)
        (data_dir / "animas" / "bob").mkdir(parents=True)
        # Create empty dm_logs dir
        (shared_dir / "dm_logs").mkdir()

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob")
        assert resp.status_code == 404

    async def test_404_for_nonexistent_dm(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "dm_logs").mkdir()

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-charlie")
        assert resp.status_code == 404

    async def test_returns_all_when_limit_exceeds_total(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_dm(shared_dir, "alice-bob", [
            {"ts": "2026-02-17T10:00:00", "from": "alice", "text": "Hi", "source": "anima"},
        ])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=100")
        data = resp.json()
        assert len(data["messages"]) == 1
        assert data["total"] == 1

    async def test_400_for_invalid_dm_pair_name(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Dots and uppercase violate the safe name regex
            resp = await client.get("/api/dm/..secret")
        assert resp.status_code == 400
