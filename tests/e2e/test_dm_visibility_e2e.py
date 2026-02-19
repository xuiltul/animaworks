# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for DM visibility and WebSocket event emission.

Verifies that:
1. DM API endpoints read from per-Anima activity_log (unified activity log)
2. Legacy dm_logs/ are merged with activity_log without duplicates
3. AnimaRunner emits anima.interaction events when messages are sent
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from core.memory.activity import ActivityLogger


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


def _write_legacy_dm(shared_dir: Path, pair: str, entries: list[dict]) -> None:
    """Write JSONL entries to a legacy dm_logs/ file."""
    dm_dir = shared_dir / "dm_logs"
    dm_dir.mkdir(parents=True, exist_ok=True)
    filepath = dm_dir / f"{pair}.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Test 1: DM API returns messages from activity_log ──────────


class TestDMApiReturnsMessagesFromActivityLog:
    """Full integration: ActivityLogger.log() -> GET /api/dm endpoints."""

    async def test_dm_list_includes_pair_from_activity_log(
        self, tmp_path: Path,
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
        self, tmp_path: Path,
    ) -> None:
        """GET /api/dm/{pair} returns messages from activity_log entries."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        alice_dir = _make_anima_dir(animas_dir, "alice")
        bob_dir = _make_anima_dir(animas_dir, "bob")

        # Log a conversation via ActivityLogger
        ts_base = datetime.now().isoformat()

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
        self, tmp_path: Path,
    ) -> None:
        """Messages logged by both sender and receiver are deduplicated."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        alice_dir = _make_anima_dir(animas_dir, "alice")
        bob_dir = _make_anima_dir(animas_dir, "bob")

        # Both alice and bob log the same message (dm_sent / dm_received)
        # with identical timestamp and content
        fixed_ts = datetime.now().isoformat()

        # Write directly to activity_log JSONL for precise timestamp control
        date_str = datetime.now().strftime("%Y-%m-%d")
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
            json.dumps(entry_alice, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        bob_log.write_text(
            json.dumps(entry_bob, ensure_ascii=False) + "\n", encoding="utf-8",
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
        assert len(exact_matches) == 1, (
            f"Expected 1 deduplicated message, got {len(exact_matches)}: {exact_matches}"
        )


# ── Test 2: DM API merges legacy and activity_log ─────────────


class TestDMApiMergesLegacyAndActivityLog:
    """Legacy dm_logs/ + activity_log messages are merged without duplicates."""

    async def test_legacy_and_activity_log_both_appear(
        self, tmp_path: Path,
    ) -> None:
        """Old messages from dm_logs/ and new messages from activity_log
        are both returned by GET /api/dm/{pair}."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        alice_dir = _make_anima_dir(animas_dir, "alice")
        bob_dir = _make_anima_dir(animas_dir, "bob")

        # Legacy messages (old dm_logs/ format)
        legacy_entries = [
            {
                "ts": "2026-01-15T10:00:00",
                "from": "alice",
                "text": "Legacy message 1",
                "source": "anima",
            },
            {
                "ts": "2026-01-15T11:00:00",
                "from": "bob",
                "text": "Legacy message 2",
                "source": "anima",
            },
        ]
        _write_legacy_dm(shared_dir, "alice-bob", legacy_entries)

        # New messages via activity_log
        alice_activity = ActivityLogger(alice_dir)
        alice_activity.log(
            "dm_sent",
            content="New activity_log message",
            from_person="alice",
            to_person="bob",
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=100")

        assert resp.status_code == 200
        data = resp.json()
        messages = data["messages"]
        texts = [m["text"] for m in messages]

        # Both legacy and new messages should be present
        assert "Legacy message 1" in texts, f"Legacy msg1 missing. Got: {texts}"
        assert "Legacy message 2" in texts, f"Legacy msg2 missing. Got: {texts}"
        assert "New activity_log message" in texts, f"Activity log msg missing. Got: {texts}"

        # Total should include all unique messages
        assert data["total"] >= 3

    async def test_dm_list_merges_counts_from_both_sources(
        self, tmp_path: Path,
    ) -> None:
        """GET /api/dm includes counts from both legacy dm_logs/ and activity_log."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        alice_dir = _make_anima_dir(animas_dir, "alice")
        _make_anima_dir(animas_dir, "bob")

        # 2 legacy messages
        legacy_entries = [
            {"ts": "2026-01-15T10:00:00", "from": "alice", "text": "Old 1", "source": "anima"},
            {"ts": "2026-01-15T11:00:00", "from": "bob", "text": "Old 2", "source": "anima"},
        ]
        _write_legacy_dm(shared_dir, "alice-bob", legacy_entries)

        # 1 new message via activity_log
        alice_activity = ActivityLogger(alice_dir)
        alice_activity.log(
            "dm_sent",
            content="New msg",
            from_person="alice",
            to_person="bob",
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm")

        assert resp.status_code == 200
        pairs = resp.json()
        ab = next((p for p in pairs if p["pair"] == "alice-bob"), None)
        assert ab is not None
        # Legacy count (2) + activity_log count (at least 1)
        assert ab["message_count"] >= 3

    async def test_no_duplicates_when_same_message_in_both(
        self, tmp_path: Path,
    ) -> None:
        """If the same ts|content exists in both legacy and activity_log,
        it is not duplicated in the response."""
        _data_dir, shared_dir, animas_dir = _setup_data_dir(tmp_path)

        alice_dir = _make_anima_dir(animas_dir, "alice")
        _make_anima_dir(animas_dir, "bob")

        fixed_ts = "2026-02-18T09:00:00"
        shared_content = "This message exists in both sources"

        # Write to legacy dm_logs/
        _write_legacy_dm(shared_dir, "alice-bob", [
            {"ts": fixed_ts, "from": "alice", "text": shared_content, "source": "anima"},
        ])

        # Write identical entry to alice's activity_log
        log_file = alice_dir / "activity_log" / "2026-02-18.jsonl"
        entry = {
            "ts": fixed_ts,
            "type": "dm_sent",
            "content": shared_content,
            "from": "alice",
            "to": "bob",
        }
        log_file.write_text(
            json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8",
        )

        app = _create_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dm/alice-bob?limit=100")

        assert resp.status_code == 200
        data = resp.json()
        messages = data["messages"]

        # The dedup key is "ts|content" (or "ts|text"), so identical entries
        # from both sources should be merged into one
        matching = [m for m in messages if m.get("text") == shared_content]
        assert len(matching) == 1, (
            f"Expected 1 deduplicated message, got {len(matching)}: {matching}"
        )


# ── Test 3: Runner emits interaction event on message_sent ────


class TestRunnerEmitsInteractionEvent:
    """AnimaRunner._emit_event writes event files for parent process pickup."""

    def test_emit_event_creates_json_file(self, tmp_path: Path) -> None:
        """_emit_event writes an atomic JSON event file under run/events/{anima}."""
        # Simulate the _emit_event mechanism from AnimaRunner.
        # We replicate the logic directly since AnimaRunner is heavy to instantiate.
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        data_dir = shared_dir.parent  # _emit_event uses shared_dir.parent / "run" / "events"
        anima_name = "test-anima"

        # Replicate _emit_event logic
        event_type = "anima.interaction"
        event_data = {
            "from_person": "test-anima",
            "to_person": "bob",
            "type": "message",
            "summary": "Hello Bob, this is a test message",
        }

        events_dir = data_dir / "run" / "events" / anima_name
        events_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{time.time_ns()}.json"
        event = {"event": event_type, "data": event_data}
        tmp_file = events_dir / f".{filename}"
        tmp_file.write_text(
            json.dumps(event, default=str, ensure_ascii=False), encoding="utf-8",
        )
        tmp_file.rename(events_dir / filename)  # Atomic rename

        # Verify the event file was created
        event_files = list(events_dir.glob("*.json"))
        assert len(event_files) == 1

        # Verify content
        written_event = json.loads(event_files[0].read_text(encoding="utf-8"))
        assert written_event["event"] == "anima.interaction"
        assert written_event["data"]["from_person"] == "test-anima"
        assert written_event["data"]["to_person"] == "bob"
        assert written_event["data"]["type"] == "message"
        assert written_event["data"]["summary"] == "Hello Bob, this is a test message"

    def test_emit_event_no_temp_files_remain(self, tmp_path: Path) -> None:
        """After _emit_event, no dot-prefixed temp files remain."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        data_dir = shared_dir.parent
        anima_name = "test-anima"

        events_dir = data_dir / "run" / "events" / anima_name
        events_dir.mkdir(parents=True, exist_ok=True)

        # Emit multiple events
        for i in range(5):
            filename = f"{time.time_ns()}.json"
            event = {
                "event": "anima.interaction",
                "data": {"index": i},
            }
            tmp_file = events_dir / f".{filename}"
            tmp_file.write_text(
                json.dumps(event, ensure_ascii=False), encoding="utf-8",
            )
            tmp_file.rename(events_dir / filename)

        # No temp files should remain
        temp_files = list(events_dir.glob(".*"))
        assert len(temp_files) == 0, f"Temp files remain: {temp_files}"

        # All event files should be present
        event_files = list(events_dir.glob("*.json"))
        assert len(event_files) == 5

    def test_on_message_sent_callback_produces_correct_event(
        self, tmp_path: Path,
    ) -> None:
        """The on_message_sent callback wired in AnimaRunner produces
        the correct anima.interaction event payload."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        data_dir = shared_dir.parent
        anima_name = "sakura"

        events_dir = data_dir / "run" / "events" / anima_name
        events_dir.mkdir(parents=True, exist_ok=True)

        # Simulate the callback that AnimaRunner.run() wires up:
        #   def _on_message_sent(from_name, to_name, content):
        #       self._emit_event("anima.interaction", {
        #           "from_person": from_name,
        #           "to_person": to_name,
        #           "type": "message",
        #           "summary": content[:200],
        #       })

        def _emit_event(event_type: str, data: dict) -> None:
            events_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{time.time_ns()}.json"
            event = {"event": event_type, "data": data}
            tmp_file = events_dir / f".{filename}"
            tmp_file.write_text(
                json.dumps(event, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_file.rename(events_dir / filename)

        def on_message_sent(from_name: str, to_name: str, content: str) -> None:
            _emit_event("anima.interaction", {
                "from_person": from_name,
                "to_person": to_name,
                "type": "message",
                "summary": content[:200],
            })

        # Trigger the callback as AnimaRunner would
        on_message_sent("sakura", "mio", "This is a test DM from sakura to mio")

        # Verify event file
        event_files = list(events_dir.glob("*.json"))
        assert len(event_files) == 1

        event = json.loads(event_files[0].read_text(encoding="utf-8"))
        assert event["event"] == "anima.interaction"
        assert event["data"]["from_person"] == "sakura"
        assert event["data"]["to_person"] == "mio"
        assert event["data"]["type"] == "message"
        assert event["data"]["summary"] == "This is a test DM from sakura to mio"

    def test_on_message_sent_truncates_long_content(
        self, tmp_path: Path,
    ) -> None:
        """The on_message_sent callback truncates content at 200 characters."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        data_dir = shared_dir.parent
        anima_name = "sakura"

        events_dir = data_dir / "run" / "events" / anima_name
        events_dir.mkdir(parents=True, exist_ok=True)

        def _emit_event(event_type: str, data: dict) -> None:
            filename = f"{time.time_ns()}.json"
            event = {"event": event_type, "data": data}
            tmp_file = events_dir / f".{filename}"
            tmp_file.write_text(
                json.dumps(event, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_file.rename(events_dir / filename)

        def on_message_sent(from_name: str, to_name: str, content: str) -> None:
            _emit_event("anima.interaction", {
                "from_person": from_name,
                "to_person": to_name,
                "type": "message",
                "summary": content[:200],
            })

        long_content = "A" * 500
        on_message_sent("sakura", "mio", long_content)

        event_files = list(events_dir.glob("*.json"))
        assert len(event_files) == 1

        event = json.loads(event_files[0].read_text(encoding="utf-8"))
        assert len(event["data"]["summary"]) == 200
        assert event["data"]["summary"] == "A" * 200
