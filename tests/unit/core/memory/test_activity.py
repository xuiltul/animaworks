"""Unit tests for core/memory/activity.py — ActivityLogger."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def logger(anima_dir: Path) -> ActivityLogger:
    return ActivityLogger(anima_dir)


# ── Basic tests ──────────────────────────────────────────


class TestLogBasic:
    def test_log_creates_file(self, logger: ActivityLogger, anima_dir: Path):
        logger.log("message_received", content="hello")
        today = date.today().isoformat()
        path = anima_dir / "activity_log" / f"{today}.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["type"] == "message_received"
        assert data["content"] == "hello"

    def test_log_returns_entry(self, logger: ActivityLogger):
        entry = logger.log("dm_sent", content="hi", to_person="alice")
        assert isinstance(entry, ActivityEntry)
        assert entry.type == "dm_sent"
        assert entry.content == "hi"
        assert entry.to_person == "alice"
        assert entry.ts  # non-empty timestamp

    def test_log_appends_multiple(self, logger: ActivityLogger, anima_dir: Path):
        logger.log("message_received", content="first")
        logger.log("response_sent", content="second")
        logger.log("tool_use", tool="web_search")
        today = date.today().isoformat()
        path = anima_dir / "activity_log" / f"{today}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_log_omits_empty_fields(self, logger: ActivityLogger, anima_dir: Path):
        logger.log("heartbeat_start")
        today = date.today().isoformat()
        path = anima_dir / "activity_log" / f"{today}.jsonl"
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert "content" not in data
        assert "summary" not in data
        assert "from_person" not in data
        assert "to_person" not in data
        assert "channel" not in data
        assert "tool" not in data
        assert "via" not in data
        assert "meta" not in data
        assert "ts" in data
        assert "type" in data


# ── ActivityEntry ────────────────────────────────────────


class TestActivityEntry:
    def test_entry_to_dict_minimal(self):
        entry = ActivityEntry(ts="2026-02-17T10:00:00", type="heartbeat_start")
        d = entry.to_dict()
        assert d == {"ts": "2026-02-17T10:00:00", "type": "heartbeat_start"}

    def test_entry_to_dict_full(self):
        entry = ActivityEntry(
            ts="2026-02-17T10:00:00",
            type="dm_sent",
            content="Hello there",
            summary="Greeting",
            from_person="alice",
            to_person="bob",
            channel="general",
            tool="slack",
            via="slack",
            meta={"priority": "high"},
        )
        d = entry.to_dict()
        assert d["ts"] == "2026-02-17T10:00:00"
        assert d["type"] == "dm_sent"
        assert d["content"] == "Hello there"
        assert d["summary"] == "Greeting"
        assert d["from_person"] == "alice"
        assert d["to_person"] == "bob"
        assert d["channel"] == "general"
        assert d["tool"] == "slack"
        assert d["via"] == "slack"
        assert d["meta"] == {"priority": "high"}
        assert len(d) == 10


# ── recent() ─────────────────────────────────────────────


class TestRecent:
    def test_recent_returns_chronological(self, logger: ActivityLogger):
        logger.log("message_received", content="first")
        logger.log("response_sent", content="second")
        logger.log("tool_use", content="third")
        entries = logger.recent(days=1)
        assert len(entries) == 3
        assert entries[0].content == "first"
        assert entries[1].content == "second"
        assert entries[2].content == "third"
        # timestamps should be non-decreasing
        for i in range(len(entries) - 1):
            assert entries[i].ts <= entries[i + 1].ts

    def test_recent_limit(self, logger: ActivityLogger):
        for i in range(10):
            logger.log("message_received", content=f"msg-{i}")
        entries = logger.recent(days=1, limit=3)
        assert len(entries) == 3
        # Should keep the most recent 3
        assert entries[0].content == "msg-7"
        assert entries[1].content == "msg-8"
        assert entries[2].content == "msg-9"

    def test_recent_type_filter(self, logger: ActivityLogger):
        logger.log("message_received", content="msg")
        logger.log("tool_use", tool="search")
        logger.log("response_sent", content="reply")
        logger.log("tool_use", tool="read")
        entries = logger.recent(days=1, types=["tool_use"])
        assert len(entries) == 2
        assert all(e.type == "tool_use" for e in entries)

    def test_recent_involving_filter(self, logger: ActivityLogger):
        logger.log("dm_sent", from_person="alice", to_person="bob")
        logger.log("dm_sent", from_person="charlie", to_person="dave")
        logger.log("channel_post", from_person="eve", channel="alice")
        entries = logger.recent(days=1, involving="alice")
        assert len(entries) == 2
        types_found = {(e.from_person, e.to_person, e.channel) for e in entries}
        assert ("alice", "bob", "") in types_found
        assert ("eve", "", "alice") in types_found

    def test_recent_multiple_days(self, logger: ActivityLogger, anima_dir: Path):
        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = date.today()
        # Write yesterday's log manually
        from datetime import timedelta

        yesterday = today - timedelta(days=1)
        yesterday_entry = {
            "ts": f"{yesterday.isoformat()}T08:00:00",
            "type": "heartbeat_start",
            "content": "yesterday-event",
        }
        (log_dir / f"{yesterday.isoformat()}.jsonl").write_text(
            json.dumps(yesterday_entry, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Write today's entry via API
        logger.log("message_received", content="today-event")
        entries = logger.recent(days=2)
        assert len(entries) == 2
        assert entries[0].content == "yesterday-event"
        assert entries[1].content == "today-event"

    def test_recent_empty_dir(self, logger: ActivityLogger):
        entries = logger.recent(days=7)
        assert entries == []


# ── format_for_priming() ─────────────────────────────────


class TestFormatForPriming:
    def test_format_empty_entries(self, logger: ActivityLogger):
        result = logger.format_for_priming([])
        assert result == ""

    def test_format_basic(self, logger: ActivityLogger):
        entries = [
            ActivityEntry(
                ts="2026-02-17T10:30:00",
                type="message_received",
                content="Hello",
                from_person="user",
            ),
            ActivityEntry(
                ts="2026-02-17T10:31:00",
                type="response_sent",
                content="Hi there",
            ),
        ]
        result = logger.format_for_priming(entries)
        assert "10:30" in result
        assert "10:31" in result
        assert "message_received" in result
        assert "response_sent" in result
        assert "Hello" in result
        assert "Hi there" in result
        assert "from:user" in result

    def test_format_budget_truncation(self, logger: ActivityLogger):
        entries = []
        for i in range(100):
            entries.append(ActivityEntry(
                ts=f"2026-02-17T10:{i % 60:02d}:00",
                type="message_received",
                content=f"Message number {i} with some padding text to take up space",
            ))
        # Very small budget
        result = logger.format_for_priming(entries, budget_tokens=50)
        # Should be truncated, not all 100 entries
        lines = result.strip().splitlines()
        assert len(lines) < 100
        # Newest entries should be preserved (last entries in chronological input)
        assert "Message number 99" in result

    def test_format_long_content_truncated(self, logger: ActivityLogger):
        long_text = "x" * 300
        entries = [
            ActivityEntry(
                ts="2026-02-17T10:00:00",
                type="message_received",
                content=long_text,
            ),
        ]
        result = logger.format_for_priming(entries, budget_tokens=5000)
        assert "..." in result
        # The original 300-char content should be cut to 200 + "..."
        assert long_text not in result


# ── Error resilience ─────────────────────────────────────


class TestErrorResilience:
    def test_log_handles_io_error(self, logger: ActivityLogger, anima_dir: Path):
        # Make log_dir a file instead of a directory to cause mkdir to fail
        log_dir = anima_dir / "activity_log"
        log_dir.parent.mkdir(parents=True, exist_ok=True)
        log_dir.write_text("not a directory", encoding="utf-8")
        # Should not raise
        entry = logger.log("test_event", content="should not crash")
        assert isinstance(entry, ActivityEntry)
        assert entry.type == "test_event"

    def test_recent_handles_malformed_jsonl(
        self, logger: ActivityLogger, anima_dir: Path
    ):
        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        path = log_dir / f"{today}.jsonl"
        lines = [
            json.dumps({"ts": "2026-02-17T10:00:00", "type": "ok", "content": "good"}),
            "this is not valid json{{{",
            "",
            json.dumps({"ts": "2026-02-17T10:01:00", "type": "ok", "content": "also good"}),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        entries = logger.recent(days=1)
        assert len(entries) == 2
        assert entries[0].content == "good"
        assert entries[1].content == "also good"
