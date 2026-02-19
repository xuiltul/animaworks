"""Unit tests for core/memory/activity.py — ASCII labels, pointer, and pagination."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger, ActivityPage


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    (d / "activity_log").mkdir(parents=True)
    return d


@pytest.fixture
def activity_logger(anima_dir: Path) -> ActivityLogger:
    return ActivityLogger(anima_dir)


# ── ASCII label mapping ──────────────────────────────────


class TestFormatEntryUsesAsciiLabels:
    """_format_entry() must produce the correct ASCII label for every type."""

    _TYPE_TO_LABEL: dict[str, str] = {
        "message_received": "MSG<",
        "response_sent": "MSG>",
        "channel_read": "CH.R",
        "channel_post": "CH.W",
        "dm_received": "DM<",
        "dm_sent": "DM>",
        "human_notify": "NTFY",
        "tool_use": "TOOL",
        "heartbeat_start": "HB",
        "heartbeat_end": "HB",
        "cron_executed": "CRON",
        "memory_write": "MEM",
        "error": "ERR",
    }

    @pytest.mark.parametrize(
        "event_type,expected_label",
        list(_TYPE_TO_LABEL.items()),
        ids=list(_TYPE_TO_LABEL.keys()),
    )
    def test_format_entry_uses_ascii_labels(
        self, event_type: str, expected_label: str
    ) -> None:
        entry = ActivityEntry(
            ts="2026-02-18T10:00:00",
            type=event_type,
            content="test content",
        )
        line = ActivityLogger._format_entry(entry)
        assert expected_label in line, (
            f"Expected label '{expected_label}' for type '{event_type}', "
            f"got line: {line}"
        )


# ── Truncation + pointer ─────────────────────────────────


class TestTruncationPointer:
    def test_truncated_entry_has_pointer(self) -> None:
        """Content > 200 chars should produce a pointer to the JSONL file."""
        long_content = "A" * 250
        entry = ActivityEntry(
            ts="2026-02-18T10:30:00",
            type="message_received",
            content=long_content,
        )
        line = ActivityLogger._format_entry(entry)
        assert "-> activity_log/2026-02-18.jsonl" in line
        # The line should NOT end with just bare "..."
        assert not line.rstrip().endswith("...")

    def test_short_entry_no_pointer(self) -> None:
        """Content < 200 chars should NOT include a pointer."""
        entry = ActivityEntry(
            ts="2026-02-18T10:30:00",
            type="message_received",
            content="short message",
        )
        line = ActivityLogger._format_entry(entry)
        assert "-> activity_log/" not in line


# ── Line numbers ──────────────────────────────────────────


class TestLineNumbers:
    def test_line_number_assigned_by_recent(
        self, activity_logger: ActivityLogger
    ) -> None:
        """recent() must assign sequential _line_number to each entry."""
        activity_logger.log("message_received", content="first")
        activity_logger.log("response_sent", content="second")
        activity_logger.log("tool_use", content="third")

        entries = activity_logger.recent(days=1)
        assert len(entries) == 3
        line_numbers = [e._line_number for e in entries]
        assert line_numbers == [1, 2, 3]

    def test_line_number_not_in_jsonl(self) -> None:
        """_line_number is internal and must NOT appear in to_dict() output."""
        entry = ActivityEntry(
            ts="2026-02-18T10:00:00",
            type="message_received",
            content="test",
        )
        entry._line_number = 5
        d = entry.to_dict()
        assert "_line_number" not in d


# ── format_for_priming basics ─────────────────────────────


class TestFormatForPrimingBasics:
    def test_format_basic_single_entries(
        self, activity_logger: ActivityLogger
    ) -> None:
        """Single-entry groups should render with ASCII labels."""
        entries = [
            ActivityEntry(
                ts="2026-02-18T10:00:00",
                type="message_received",
                content="Hello",
                from_person="user",
            ),
            ActivityEntry(
                ts="2026-02-18T10:01:00",
                type="response_sent",
                content="Hi there",
            ),
        ]
        result = activity_logger.format_for_priming(entries)
        assert "MSG<" in result
        assert "MSG>" in result

    def test_format_budget_truncation(
        self, activity_logger: ActivityLogger
    ) -> None:
        """A small budget must limit the output, dropping oldest entries."""
        entries = []
        for i in range(50):
            entries.append(
                ActivityEntry(
                    ts=f"2026-02-18T{10 + i // 60:02d}:{i % 60:02d}:00",
                    type="message_received",
                    content=f"Message number {i} with padding text to occupy space",
                )
            )
        result = activity_logger.format_for_priming(entries, budget_tokens=50)
        lines = [ln for ln in result.strip().splitlines() if ln.strip()]
        # Budget of 50 tokens ~ 200 chars; should NOT contain all 50 entries
        assert len(lines) < 50
        # The output should be non-empty
        assert len(lines) > 0


# ── Helper for writing test JSONL ────────────────────────


def _write_activity(anima_dir: Path, entries: list[dict]) -> None:
    """Write test activity entries to {anima_dir}/activity_log/{date}.jsonl."""
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, list[dict]] = {}
    for entry in entries:
        date_str = entry["ts"][:10]
        by_date.setdefault(date_str, []).append(entry)
    for date_str, date_entries in by_date.items():
        path = log_dir / f"{date_str}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for e in date_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ── recent_page() and _load_entries() ────────────────────


class TestRecentPage:
    """Tests for ActivityLogger.recent_page()."""

    def test_basic_pagination(self, tmp_path: Path) -> None:
        """Write 10 entries, request page of 5 — total=10, has_more=True."""
        anima_dir = tmp_path / "test-anima"
        now = datetime.now()
        entries = [
            {"ts": (now - timedelta(seconds=10 - i)).isoformat(), "type": "tool_use", "summary": f"Entry {i}", "content": ""}
            for i in range(10)
        ]
        _write_activity(anima_dir, entries)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1, limit=5)

        assert isinstance(page, ActivityPage)
        assert page.total == 10
        assert len(page.entries) == 5
        assert page.has_more is True
        assert page.offset == 0
        assert page.limit == 5

    def test_offset(self, tmp_path: Path) -> None:
        """Write 10 entries, request offset=5, limit=5 — older 5 entries returned."""
        anima_dir = tmp_path / "test-anima"
        now = datetime.now()
        entries = [
            {"ts": (now - timedelta(seconds=10 - i)).isoformat(), "type": "tool_use", "summary": f"Entry {i}", "content": ""}
            for i in range(10)
        ]
        _write_activity(anima_dir, entries)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1, offset=5, limit=5)

        assert page.total == 10
        assert len(page.entries) == 5
        assert page.has_more is False
        assert page.offset == 5

    def test_hours_filter(self, tmp_path: Path) -> None:
        """Only entries within the last N hours are included."""
        anima_dir = tmp_path / "test-anima"
        now = datetime.now()
        entries = [
            # Recent entry (30 min ago)
            {"ts": (now - timedelta(minutes=30)).isoformat(), "type": "heartbeat_start", "summary": "Recent", "content": ""},
            # Old entry (3 hours ago)
            {"ts": (now - timedelta(hours=3)).isoformat(), "type": "heartbeat_start", "summary": "Old", "content": ""},
        ]
        _write_activity(anima_dir, entries)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(hours=1, limit=10)

        assert page.total == 1
        assert page.entries[0].summary == "Recent"

    def test_limit_zero_returns_all(self, tmp_path: Path) -> None:
        """limit=0 should return all entries."""
        anima_dir = tmp_path / "test-anima"
        now = datetime.now()
        entries = [
            {"ts": (now - timedelta(seconds=i)).isoformat(), "type": "tool_use", "summary": f"Entry {i}", "content": ""}
            for i in range(15)
        ]
        _write_activity(anima_dir, entries)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1, limit=0)

        assert len(page.entries) == 15
        assert page.total == 15
        assert page.has_more is False

    def test_types_filter(self, tmp_path: Path) -> None:
        """Only matching event types are returned."""
        anima_dir = tmp_path / "test-anima"
        now = datetime.now()
        entries = [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "HB", "content": ""},
            {"ts": now.isoformat(), "type": "cron_executed", "summary": "Cron", "content": ""},
            {"ts": now.isoformat(), "type": "tool_use", "summary": "Tool", "content": ""},
        ]
        _write_activity(anima_dir, entries)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1, types=["heartbeat_start"])

        assert page.total == 1
        assert page.entries[0].type == "heartbeat_start"

    def test_involving_filter(self, tmp_path: Path) -> None:
        """Only entries involving the specified name are returned."""
        anima_dir = tmp_path / "test-anima"
        now = datetime.now()
        entries = [
            {"ts": now.isoformat(), "type": "dm_sent", "summary": "msg1", "content": "", "from": "alice", "to": "bob"},
            {"ts": now.isoformat(), "type": "dm_sent", "summary": "msg2", "content": "", "from": "charlie", "to": "dave"},
        ]
        _write_activity(anima_dir, entries)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1, involving="bob")

        assert page.total == 1
        assert page.entries[0].summary == "msg1"

    def test_entries_newest_first(self, tmp_path: Path) -> None:
        """recent_page() returns entries in newest-first order."""
        anima_dir = tmp_path / "test-anima"
        now = datetime.now()
        entries = [
            {"ts": (now - timedelta(hours=2)).isoformat(), "type": "heartbeat_start", "summary": "Old", "content": ""},
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "New", "content": ""},
        ]
        _write_activity(anima_dir, entries)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1, limit=10)

        assert page.entries[0].summary == "New"
        assert page.entries[1].summary == "Old"

    def test_empty_log(self, tmp_path: Path) -> None:
        """No activity_log directory at all returns empty page."""
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1)

        assert page.total == 0
        assert len(page.entries) == 0
        assert page.has_more is False

    def test_limit_clamped_to_500(self, tmp_path: Path) -> None:
        """Limit should be capped at 500."""
        anima_dir = tmp_path / "test-anima"
        (anima_dir / "activity_log").mkdir(parents=True)

        al = ActivityLogger(anima_dir)
        page = al.recent_page(days=1, limit=9999)

        assert page.limit == 500


class TestToApiDict:
    """Tests for ActivityEntry.to_api_dict()."""

    def test_all_fields_present(self) -> None:
        """to_api_dict should include all required fields."""
        entry = ActivityEntry(
            ts="2026-02-18T10:30:00",
            type="heartbeat_start",
            content="巡回中",
            summary="定期巡回",
            from_person="system",
            to_person="alice",
            channel="ops",
            tool="",
            via="internal",
            meta={"duration_ms": 150},
        )
        d = entry.to_api_dict("alice")

        assert d["id"] == "alice:2026-02-18T10:30:00:heartbeat_start:0"
        assert d["ts"] == "2026-02-18T10:30:00"
        assert d["type"] == "heartbeat_start"
        assert d["anima"] == "alice"
        assert d["summary"] == "定期巡回"
        assert d["content"] == "巡回中"
        assert d["from_person"] == "system"
        assert d["to_person"] == "alice"
        assert d["channel"] == "ops"
        assert d["tool"] == ""
        assert d["via"] == "internal"
        assert d["meta"] == {"duration_ms": 150}

    def test_uses_internal_anima_name(self) -> None:
        """to_api_dict without explicit name uses _anima_name."""
        entry = ActivityEntry(
            ts="2026-02-18T10:30:00",
            type="tool_use",
        )
        entry._anima_name = "bob"
        d = entry.to_api_dict()

        assert d["anima"] == "bob"
        assert d["id"] == "bob:2026-02-18T10:30:00:tool_use:0"

    def test_explicit_name_overrides_internal(self) -> None:
        """Explicit anima_name argument overrides _anima_name."""
        entry = ActivityEntry(
            ts="2026-02-18T10:30:00",
            type="tool_use",
        )
        entry._anima_name = "bob"
        d = entry.to_api_dict("alice")

        assert d["anima"] == "alice"


class TestLoadEntries:
    """Tests for ActivityLogger._load_entries()."""

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed JSONL lines are silently skipped."""
        anima_dir = tmp_path / "test-anima"
        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True)

        now = datetime.now()
        good = json.dumps({"ts": now.isoformat(), "type": "heartbeat_start", "summary": "Good", "content": ""})
        content = "not json\n" + good + "\n{bad\n"
        (log_dir / f"{now.strftime('%Y-%m-%d')}.jsonl").write_text(content, encoding="utf-8")

        al = ActivityLogger(anima_dir)
        entries = al._load_entries(days=1)

        assert len(entries) == 1
        assert entries[0].summary == "Good"

    def test_from_to_field_mapping(self, tmp_path: Path) -> None:
        """JSONL 'from'/'to' keys are mapped to from_person/to_person."""
        anima_dir = tmp_path / "test-anima"
        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True)

        now = datetime.now()
        entry = json.dumps({"ts": now.isoformat(), "type": "dm_sent", "from": "alice", "to": "bob", "summary": "hi"})
        (log_dir / f"{now.strftime('%Y-%m-%d')}.jsonl").write_text(entry + "\n", encoding="utf-8")

        al = ActivityLogger(anima_dir)
        entries = al._load_entries(days=1)

        assert len(entries) == 1
        assert entries[0].from_person == "alice"
        assert entries[0].to_person == "bob"

    def test_multi_day_scan(self, tmp_path: Path) -> None:
        """Entries across multiple date files are all loaded."""
        anima_dir = tmp_path / "test-anima"
        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True)

        now = datetime.now()
        yesterday = now - timedelta(days=1)
        e1 = json.dumps({"ts": yesterday.isoformat(), "type": "heartbeat_start", "summary": "Yesterday"})
        e2 = json.dumps({"ts": now.isoformat(), "type": "heartbeat_start", "summary": "Today"})
        (log_dir / f"{yesterday.strftime('%Y-%m-%d')}.jsonl").write_text(e1 + "\n", encoding="utf-8")
        (log_dir / f"{now.strftime('%Y-%m-%d')}.jsonl").write_text(e2 + "\n", encoding="utf-8")

        al = ActivityLogger(anima_dir)
        entries = al._load_entries(days=2)

        assert len(entries) == 2
        summaries = [e.summary for e in entries]
        assert "Yesterday" in summaries
        assert "Today" in summaries
