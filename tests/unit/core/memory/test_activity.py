"""Unit tests for core/memory/activity.py — Phase 1: ASCII labels + pointer."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger


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
