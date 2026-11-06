# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for DM visibility and WebSocket event emission.

Verifies that:
1. DM API endpoints read from per-Anima activity_log (unified activity log)
2. DM history returns messages recorded in activity logs
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from core.activity.logger import ActivityLogger
from core.time_utils import now_jst

# ── Helpers ────────────────────────────────────────────────────


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


def _setup_data_dir(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create an isolated data directory with shared/ and animas/ subdirectories.

    Returns (data_dir, shared_dir, animas_dir).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shared_dir = data_dir / "shared"
    shared_dir.mkdir()
    (shared_dir / "channels").mkdir()
    (shared_dir / "inbox").mkdir()
    animas_dir = data_dir / "animas"
    animas_dir.mkdir()
    return data_dir, shared_dir, animas_dir


def _make_anima_dir(animas_dir: Path, name: str) -> Path:
    """Create a minimal anima directory with activity_log subdirectory."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "activity_log").mkdir(exist_ok=True)
    return anima_dir


# ── Test 1: DM API returns messages from activity_log ──────────


class TestDMApiReturnsMessagesFromActivityLog:
    """Full integration: ActivityLogger.log() -> GET /api/dm endpoints."""

    @patch("core.config.models.load_config", side_effect=Exception("no config"))
    async def test_dm_list_includes_pair_from_activity_log(
        self,
        _mock_cfg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """GET /api/dm lists pairs discovered from per-Anima activity_log."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        # Create two Animas with activity_log entries
        alice_dir = _make_anima_dir(animas_dir, "alice")
        bob_dir = _make_anima_dir(animas_dir, "bob")

        # Alice sent a DM to Bob
        alice_activity = ActivityLogger(alice_dir)
        alice_activity.log(
            "dm_sent",
            content="Hello Bob!",
            from_person="alice",
            to_person="bob",
        )

        # Bob received the DM from Alice
        bob_activity = ActivityLogger(bob_dir)
        bob_activity.log(
            "dm_received",
            content="Hello Bob!",
            from_person="alice",
            to_person="bob",
        )

        # Bob replied
        bob_activity.log(
            "dm_sent",
            content="Hi Alice!",
            from_person="bob",
            to_person="alice",
        )
        alice_activity.log(
            "dm_received",
            content="Hi Alice!",
            from_person="bob",
            to_person="alice",
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")

        assert resp.status_code == 200
        pairs = resp.json()
        assert len(pairs) >= 1

        # The pair key should be alphabetically sorted: alice-bob
        ab = next((p for p in pairs if p["pair"] == "alice-bob"), None)
        assert ab is not None, f"Expected alice-bob pair, got: {[p['pair'] for p in pairs]}"
        assert ab["participants"] == ["alice", "bob"]
        assert ab["message_count"] >= 2  # At least the messages we logged

    async def test_dm_history_returns_messages_from_activity_log(
        self,
        tmp_path: Path,
    ) -> None:
        """GET /api/dm/{pair} returns messages from activity_log entries."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        alice_dir = _make_anima_dir(animas_dir, "alice")
        bob_dir = _make_anima_dir(animas_dir, "bob")

        # Log a conversation via ActivityLogger
        alice_activity = ActivityLogger(alice_dir)
        alice_activity.log(
            "dm_sent",
            content="How is the project going?",
            from_person="alice",
            to_person="bob",
        )

        bob_activity = ActivityLogger(bob_dir)
        bob_activity.log(
            "dm_received",
            content="How is the project going?",
            from_person="alice",
            to_person="bob",
        )
        bob_activity.log(
            "dm_sent",
            content="Going well, thanks for asking!",
            from_person="bob",
            to_person="alice",
        )

        alice_activity.log(
            "dm_received",
            content="Going well, thanks for asking!",
            from_person="bob",
            to_person="alice",
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob")

        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "alice-bob"
        assert data["participants"] == ["alice", "bob"]

        messages = data["messages"]
        assert len(messages) >= 2

        # Check message content and source
        texts = [m["text"] for m in messages]
        assert "How is the project going?" in texts
        assert "Going well, thanks for asking!" in texts

        # All messages should be from activity_log source
        for msg in messages:
            assert msg["source"] == "activity_log"

    async def test_dm_history_deduplicates_same_content(
        self,
        tmp_path: Path,
    ) -> None:
        """Messages logged by both sender and receiver are deduplicated."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        alice_dir = _make_anima_dir(animas_dir, "alice")
        bob_dir = _make_anima_dir(animas_dir, "bob")

        # Both alice and bob log the same message (dm_sent / dm_received)
        # with identical timestamp and content
        fixed_ts = now_jst().isoformat()

        # Write directly to activity_log JSONL for precise timestamp control
        date_str = now_jst().strftime("%Y-%m-%d")
        alice_log = alice_dir / "activity_log" / f"{date_str}.jsonl"
        bob_log = bob_dir / "activity_log" / f"{date_str}.jsonl"

        entry_alice = {
            "ts": fixed_ts,
            "type": "dm_sent",
            "content": "Exact same message",
            "from": "alice",
            "to": "bob",
        }
        entry_bob = {
            "ts": fixed_ts,
            "type": "dm_received",
            "content": "Exact same message",
            "from": "alice",
            "to": "bob",
        }

        alice_log.write_text(
            json.dumps(entry_alice, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        bob_log.write_text(
            json.dumps(entry_bob, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob")

        assert resp.status_code == 200
        data = resp.json()
        messages = data["messages"]

        # The dedup key is "ts|content", so identical entries should be merged
        exact_matches = [m for m in messages if m["text"] == "Exact same message"]
        assert len(exact_matches) == 1, f"Expected 1 deduplicated message, got {len(exact_matches)}: {exact_matches}"
