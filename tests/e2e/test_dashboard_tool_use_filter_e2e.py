# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for dashboard tool_use filtering via ActivityLogger live events.

Verifies that:
- Visible tools (delegate_task, update_task, etc.) emit live event files
- Non-visible tools (search_memory, Read, Bash, etc.) do NOT emit live events
- Non-tool_use event types in _LIVE_EVENT_TYPES still emit
- All tool_use entries are written to activity_log JSONL regardless of visibility
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityLogger
from core.time_utils import now_jst

# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture()
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "testanima"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def activity_logger(anima_dir: Path) -> ActivityLogger:
    return ActivityLogger(anima_dir)


# ── Helpers ───────────────────────────────────────────────


def _read_event_files(data_dir: Path, anima_name: str = "testanima") -> list[dict]:
    """Read all ta_*.json event files for the anima."""
    event_dir = data_dir / "run" / "events" / anima_name
    if not event_dir.exists():
        return []
    files = sorted(event_dir.glob("ta_*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _read_activity_entries(anima_dir: Path) -> list[dict]:
    """Read all JSONL entries from today's activity log."""
    log_dir = anima_dir / "activity_log"
    date_str = now_jst().strftime("%Y-%m-%d")
    log_file = log_dir / f"{date_str}.jsonl"
    if not log_file.exists():
        return []
    entries = []
    for line in log_file.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


# ── Test 1: Visible tool emits live event ─────────────────


class TestVisibleToolEmitsLiveEvent:
    """ActivityLogger visible tool emits live event file."""

    def test_delegate_task_emits_event_file(
        self, activity_logger: ActivityLogger, anima_dir: Path, tmp_path: Path
    ) -> None:
        """Log tool_use with tool=delegate_task and verify event file is created."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_use",
                tool="delegate_task",
                summary="Delegated task to bob",
                content="task details",
            )

        events = _read_event_files(tmp_path)
        assert len(events) == 1
        assert events[0]["event"] == "anima.tool_activity"
        assert events[0]["data"]["type"] == "tool_use"
        assert events[0]["data"]["tool"] == "delegate_task"
        assert events[0]["data"]["name"] == "testanima"


# ── Test 2: Non-visible tool does NOT emit ────────────────


class TestNonVisibleToolDoesNotEmit:
    """ActivityLogger non-visible tool does NOT emit live event."""

    def test_search_memory_does_not_emit(self, activity_logger: ActivityLogger, tmp_path: Path) -> None:
        """Log tool_use with tool=search_memory and verify NO event file is created."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_use",
                tool="search_memory",
                summary="Searched knowledge",
                content="query",
            )

        events = _read_event_files(tmp_path)
        assert len(events) == 0


# ── Test 3: All visible tools emit ────────────────────────


class TestAllVisibleToolsEmit:
    """Each tool in VISIBLE_TOOL_NAMES emits a live event."""

    @pytest.mark.parametrize(
        "tool_name",
        [
            "delegate_task",
            "update_task",
            "backlog_task",
            "submit_tasks",
            "call_human",
            "post_channel",
            "send_message",
        ],
    )
    def test_visible_tool_emits(
        self,
        activity_logger: ActivityLogger,
        tmp_path: Path,
        tool_name: str,
    ) -> None:
        """Each visible tool should emit a live event file."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_use",
                tool=tool_name,
                summary=f"Used {tool_name}",
            )

        events = _read_event_files(tmp_path)
        assert len(events) == 1, f"Expected 1 event for {tool_name}, got {len(events)}"
        assert events[0]["data"]["tool"] == tool_name


# ── Test 4: Non-visible tools do NOT emit ─────────────────


class TestNonVisibleToolsDoNotEmit:
    """Common internal tools do NOT emit live events."""

    @pytest.mark.parametrize(
        "tool_name",
        [
            "search_memory",
            "read_memory_file",
            "write_memory_file",
            "archive_memory_file",
            "Read",
            "Write",
            "Bash",
            "web_search",
            "slack",
            "chatwork",
            "gmail",
            "github",
        ],
    )
    def test_non_visible_tool_does_not_emit(
        self,
        activity_logger: ActivityLogger,
        tmp_path: Path,
        tool_name: str,
    ) -> None:
        """Each non-visible tool should NOT create an event file."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_use",
                tool=tool_name,
                summary=f"Used {tool_name}",
            )

        events = _read_event_files(tmp_path)
        assert len(events) == 0, f"Expected 0 events for {tool_name}, got {len(events)}"


# ── Test 5: Non-tool_use events still emit ─────────────────


class TestNonToolUseEventsStillEmit:
    """Event types in _LIVE_EVENT_TYPES still emit live events."""

    @pytest.mark.parametrize(
        "event_type,kwargs",
        [
            ("message_sent", {"content": "Hi", "to_person": "bob", "summary": "→ bob: Hi"}),
            ("response_sent", {"content": "Reply", "to_person": "user", "summary": "Reply"}),
            ("channel_post", {"channel": "general", "content": "Post", "summary": "Post"}),
            ("task_created", {"summary": "New task"}),
            ("task_updated", {"summary": "Task done"}),
        ],
    )
    def test_live_event_type_emits(
        self,
        activity_logger: ActivityLogger,
        tmp_path: Path,
        event_type: str,
        kwargs: dict,
    ) -> None:
        """message_sent, response_sent, channel_post, task_created, task_updated emit."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(event_type, **kwargs)

        events = _read_event_files(tmp_path)
        assert len(events) == 1, f"Expected 1 event for {event_type}, got {len(events)}"
        assert events[0]["data"]["type"] == event_type


# ── Test 6: Tool_use still logged to activity_log ──────────


class TestToolUseStillLoggedToActivityLog:
    """Non-visible tool_use MUST still be written to activity_log JSONL."""

    def test_search_memory_logged_to_activity_log(
        self, activity_logger: ActivityLogger, anima_dir: Path, tmp_path: Path
    ) -> None:
        """Non-visible tool_use (search_memory) is written to activity_log even though no live event."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_use",
                tool="search_memory",
                summary="Searched knowledge",
                content="query",
            )

        entries = _read_activity_entries(anima_dir)
        tool_entries = [e for e in entries if e.get("type") == "tool_use"]
        assert len(tool_entries) == 1
        assert tool_entries[0]["tool"] == "search_memory"

    def test_read_memory_file_logged_to_activity_log(
        self, activity_logger: ActivityLogger, anima_dir: Path, tmp_path: Path
    ) -> None:
        """Non-visible tool_use (read_memory_file) is written to activity_log."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_use",
                tool="read_memory_file",
                summary="Read knowledge/foo.md",
            )

        entries = _read_activity_entries(anima_dir)
        tool_entries = [e for e in entries if e.get("type") == "tool_use"]
        assert len(tool_entries) == 1
        assert tool_entries[0]["tool"] == "read_memory_file"
