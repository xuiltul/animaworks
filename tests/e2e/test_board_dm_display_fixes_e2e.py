# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Board DM display fixes — dedup, garbage pair filter, alias.

Tests the fixes for:
1. get_dm_history() reads only message_sent (+ dm_sent alias), eliminating
   structural duplicates from sender+receiver both logging the same message.
2. list_dm_pairs() filters pairs where both participants must be valid Anima
   names (from config), and excludes count=0 pairs.
3. Legacy dm_sent entries still match via alias in ActivityLogger.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

logger = logging.getLogger(__name__)


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

    Returns:
        Tuple of (data_dir, shared_dir, animas_dir).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shared_dir = data_dir / "shared"
    shared_dir.mkdir()
    (shared_dir / "channels").mkdir()
    (shared_dir / "dm_logs").mkdir()
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


def _write_activity_log(
    data_dir: Path,
    anima_name: str,
    date_str: str,
    entries: list[dict],
) -> None:
    """Write JSONL entries to anima activity_log/{date}.jsonl."""
    log_path = data_dir / "animas" / anima_name / "activity_log" / f"{date_str}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dm(shared_dir: Path, pair: str, entries: list[dict]) -> None:
    """Write JSONL entries to shared/dm_logs/{pair}.jsonl."""
    dm_dir = shared_dir / "dm_logs"
    dm_dir.mkdir(parents=True, exist_ok=True)
    filepath = dm_dir / f"{pair}.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Test 1: DM history no duplicates (millisecond difference) ──


class TestDMHistoryNoDuplicates:
    """get_dm_history reads only message_sent, eliminating structural duplicates."""

    async def test_dm_history_no_duplicates_with_millisecond_difference(
        self, tmp_path: Path
    ) -> None:
        """With fix (only message_sent), same logical message appears once.

        Bug scenario: alice sends a message, recorded in alice's log as
        message_sent at T10:00:00.839520 and in bob's log as message_received
        at T10:00:00.851259. Without the fix both were read → duplicate.
        With the fix we only read message_sent → exactly one.
        """
        data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)
        _make_anima_dir(animas_dir, "alice")
        _make_anima_dir(animas_dir, "bob")
        date_str = date.today().isoformat()

        msg_text = "Hello Bob, how are you?"
        ts_sender = "2026-02-25T10:00:00.839520"
        ts_receiver = "2026-02-25T10:00:00.851259"

        _write_activity_log(
            data_dir,
            "alice",
            date_str,
            [
                {
                    "ts": ts_sender,
                    "type": "message_sent",
                    "content": msg_text,
                    "from": "alice",
                    "to": "bob",
                },
            ],
        )
        _write_activity_log(
            data_dir,
            "bob",
            date_str,
            [
                {
                    "ts": ts_receiver,
                    "type": "message_received",
                    "content": msg_text,
                    "from": "alice",
                    "to": "bob",
                },
            ],
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1, (
            f"Expected 1 message (no duplicate), got total={data['total']}: "
            f"{data['messages']}"
        )
        assert data["messages"][0]["text"] == msg_text


# ── Test 2: DM pairs excludes garbage pairs ──────────────────────


class TestDMPairsExcludesGarbage:
    """list_dm_pairs filters pairs where both participants are valid Anima names."""

    async def test_dm_pairs_excludes_garbage_pairs(self, tmp_path: Path) -> None:
        """Valid pair (alice-bob) appears; garbage (.animaworks-sumire, kotoha-skills)
        pairs are filtered out when config.animas contains only alice and bob.
        """
        data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)
        _make_anima_dir(animas_dir, "alice")
        _make_anima_dir(animas_dir, "bob")
        date_str = date.today().isoformat()

        # Valid pair: alice sends to bob
        # Garbage: .animaworks-sumire in from field (alice's log has message_received)
        _write_activity_log(
            data_dir,
            "alice",
            date_str,
            [
                {
                    "ts": "2026-02-25T10:00:00",
                    "type": "message_sent",
                    "content": "Hi Bob",
                    "from": "alice",
                    "to": "bob",
                },
                {
                    "ts": "2026-02-25T10:01:00",
                    "type": "message_received",
                    "content": "Spam",
                    "from": ".animaworks-sumire",
                    "to": "alice",
                },
            ],
        )

        # Garbage: kotoha-skills in from field (bob's log)
        _write_activity_log(
            data_dir,
            "bob",
            date_str,
            [
                {
                    "ts": "2026-02-25T10:02:00",
                    "type": "message_received",
                    "content": "Skills noise",
                    "from": "kotoha-skills",
                    "to": "bob",
                },
            ],
        )

        mock_config = MagicMock()
        mock_config.animas = {"alice": MagicMock(), "bob": MagicMock()}

        with patch("core.config.models.load_config", return_value=mock_config):
            app = _create_app(shared_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/api/dm")

        assert resp.status_code == 200
        pairs = resp.json()
        pair_names = [p["pair"] for p in pairs]
        assert "alice-bob" in pair_names, f"Valid pair missing. Got: {pair_names}"
        assert ".animaworks-sumire-alice" not in pair_names
        assert "kotoha-skills-bob" not in pair_names


# ── Test 3: DM pairs excludes zero-count ──────────────────────────


class TestDMPairsExcludesZeroCount:
    """list_dm_pairs excludes pairs with zero messages."""

    async def test_dm_pairs_excludes_zero_count(self, tmp_path: Path) -> None:
        """A dm_logs file with 0 lines produces a pair with count=0, which is excluded."""
        data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)
        (shared_dir / "dm_logs").mkdir(parents=True, exist_ok=True)

        # Empty dm_logs file for alice-bob
        (shared_dir / "dm_logs" / "alice-bob.jsonl").write_text("", encoding="utf-8")

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")

        assert resp.status_code == 200
        pairs = resp.json()
        pair_names = [p["pair"] for p in pairs]
        assert "alice-bob" not in pair_names, (
            f"Zero-count pair should be excluded. Got: {pairs}"
        )


# ── Test 4: Activity log + legacy merge without duplication ──────


class TestDMHistoryMergeWithoutDuplication:
    """get_dm_history merges activity_log and legacy dm_logs without duplicating."""

    async def test_dm_history_combines_activity_log_and_legacy_without_duplication(
        self, tmp_path: Path
    ) -> None:
        """Same ts+content in both sources appears once; unique legacy entry also included."""
        data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)
        _make_anima_dir(animas_dir, "alice")
        _make_anima_dir(animas_dir, "bob")
        date_str = date.today().isoformat()

        shared_ts = "2026-02-25T10:00:00"
        shared_content = "Shared message in both sources"
        legacy_only_ts = "2026-02-24T15:00:00"
        legacy_only_content = "Older legacy-only message"

        _write_activity_log(
            data_dir,
            "alice",
            date_str,
            [
                {
                    "ts": shared_ts,
                    "type": "message_sent",
                    "content": shared_content,
                    "from": "alice",
                    "to": "bob",
                },
            ],
        )

        _write_dm(
            shared_dir,
            "alice-bob",
            [
                {"ts": legacy_only_ts, "from": "alice", "text": legacy_only_content},
                {"ts": shared_ts, "from": "alice", "text": shared_content},
            ],
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=100")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2, (
            f"Expected 2 unique messages, got total={data['total']}: "
            f"{data['messages']}"
        )
        texts = [m.get("text") or m.get("content", "") for m in data["messages"]]
        assert shared_content in texts
        assert legacy_only_content in texts
        assert texts.count(shared_content) == 1, "Shared message must not be duplicated"


# ── Test 5: Legacy dm_sent alias ───────────────────────────────


class TestLegacyDmSentAlias:
    """Legacy dm_sent entries still match via ActivityLogger alias."""

    async def test_legacy_dm_sent_entries_still_match_via_alias(
        self, tmp_path: Path
    ) -> None:
        """Activity log entries with old dm_sent type are returned by get_dm_history.

        ActivityLogger resolves dm_sent → message_sent via _EVENT_TYPE_ALIASES,
        so requesting types=["message_sent"] matches dm_sent entries.
        """
        data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)
        _make_anima_dir(animas_dir, "alice")
        _make_anima_dir(animas_dir, "bob")
        date_str = date.today().isoformat()

        msg_text = "Legacy dm_sent still works"
        _write_activity_log(
            data_dir,
            "alice",
            date_str,
            [
                {
                    "ts": "2026-02-25T11:00:00",
                    "type": "dm_sent",
                    "content": msg_text,
                    "from": "alice",
                    "to": "bob",
                },
            ],
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["messages"][0]["text"] == msg_text
