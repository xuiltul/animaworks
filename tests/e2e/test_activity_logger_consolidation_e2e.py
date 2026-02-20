# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests: ActivityLogger consolidation — verify activity logging still works.

These tests create minimal DigitalAnima and ToolHandler instances (bypassing
their heavy __init__ via ``object.__new__``) and exercise the actual
``ActivityLogger`` read/write paths to confirm the refactoring preserved
functional correctness.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.activity import ActivityLogger


# ── DigitalAnima activity logging ────────────────────────────


class TestAnimaActivityLogging:
    """Verify DigitalAnima uses self._activity for logging."""

    @staticmethod
    def _make_anima(anima_dir: Path):
        """Create a minimal DigitalAnima bypassing heavy __init__."""
        from core.anima import DigitalAnima

        anima = object.__new__(DigitalAnima)
        anima.anima_dir = anima_dir
        anima.name = anima_dir.name
        anima._activity = ActivityLogger(anima_dir)

        # Minimal fields needed by methods under test
        anima._lock = asyncio.Lock()
        anima._user_waiting = asyncio.Event()
        anima._status = "idle"
        anima._current_task = ""
        anima._last_heartbeat = None
        anima._last_activity = None
        anima._on_lock_released = None
        anima._heartbeat_stream_queue = None
        anima._heartbeat_context = ""
        anima._last_greet_at = None
        anima._last_greet_text = None
        anima._last_greet_emotion = "neutral"
        anima._GREET_COOLDOWN = 3600
        anima._HEARTBEAT_HISTORY_N = 3
        anima._RECENT_REFLECTIONS_N = 3
        anima._ws_broadcast = None
        anima.memory = MagicMock()
        anima.model_config = MagicMock()
        anima.messenger = MagicMock()
        anima.agent = MagicMock()
        anima.agent.background_manager = None
        return anima

    def test_activity_instance_is_set(self, tmp_path):
        """self._activity is set and is an ActivityLogger."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)
        assert isinstance(anima._activity, ActivityLogger)
        assert anima._activity.anima_dir == anima_dir

    def test_activity_log_writes_to_file(self, tmp_path):
        """self._activity.log() writes JSONL entries to activity_log/."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)

        anima._activity.log(
            "message_received", content="hello", from_person="user",
        )

        log_dir = anima_dir / "activity_log"
        assert log_dir.is_dir()
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "message_received"
        assert entry["content"] == "hello"
        assert entry["from"] == "user"

    def test_multiple_log_calls_append(self, tmp_path):
        """Multiple self._activity.log() calls append to the same day file."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)

        anima._activity.log("message_received", content="msg1", from_person="alice")
        anima._activity.log("response_sent", content="reply1", to_person="alice")
        anima._activity.log("tool_use", tool="search_memory", summary="query=test")

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["type"] == "message_received"
        assert json.loads(lines[1])["type"] == "response_sent"
        assert json.loads(lines[2])["type"] == "tool_use"

    def test_heartbeat_history_loads_via_activity(self, tmp_path):
        """_load_heartbeat_history uses self._activity.recent()."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)
        # Set up needed memory mock for legacy fallback
        anima.memory.load_recent_heartbeat_summary = MagicMock(return_value="")

        # Write a heartbeat_end entry
        anima._activity.log("heartbeat_end", summary="Check complete")

        result = anima._load_heartbeat_history()
        assert "Check complete" in result

    def test_heartbeat_history_returns_empty_when_no_entries(self, tmp_path):
        """_load_heartbeat_history returns '' when no entries exist."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)
        anima.memory.load_recent_heartbeat_summary = MagicMock(return_value="")

        result = anima._load_heartbeat_history()
        assert result == ""

    def test_recent_reflections_loads_via_activity(self, tmp_path):
        """_load_recent_reflections uses self._activity.recent()."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)

        # Write a heartbeat_reflection entry
        anima._activity.log("heartbeat_reflection", content="Learned something new")

        result = anima._load_recent_reflections()
        assert "Learned something new" in result

    def test_recent_reflections_returns_empty_when_no_entries(self, tmp_path):
        """_load_recent_reflections returns '' when no entries exist."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)

        result = anima._load_recent_reflections()
        assert result == ""

    def test_shared_activity_instance_across_calls(self, tmp_path):
        """The same ActivityLogger instance is reused across multiple method calls."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima(anima_dir)

        activity_id = id(anima._activity)
        anima._activity.log("message_received", content="msg1")
        assert id(anima._activity) == activity_id
        anima._activity.log("response_sent", content="reply1")
        assert id(anima._activity) == activity_id


# ── ToolHandler activity logging ─────────────────────────────


class TestToolHandlerActivityLogging:
    """Verify ToolHandler uses self._activity for logging."""

    @staticmethod
    def _make_handler(anima_dir: Path):
        """Create a minimal ToolHandler bypassing heavy __init__."""
        from core.tooling.handler import ToolHandler

        handler = object.__new__(ToolHandler)
        handler._anima_dir = anima_dir
        handler._anima_name = anima_dir.name
        handler._activity = ActivityLogger(anima_dir)
        handler._memory = MagicMock()
        handler._messenger = None
        handler._on_message_sent = None
        handler._on_schedule_changed = None
        handler._human_notifier = None
        handler._background_manager = None
        handler._pending_notifications = []
        handler._replied_to = set()
        handler._session_id = "test12345678"
        handler._external = MagicMock()
        handler._dispatch = {}
        return handler

    def test_activity_instance_is_set(self, tmp_path):
        """self._activity is set and is an ActivityLogger."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        handler = self._make_handler(anima_dir)
        assert isinstance(handler._activity, ActivityLogger)
        assert handler._activity.anima_dir == anima_dir

    def test_log_tool_activity_writes(self, tmp_path):
        """_log_tool_activity writes to activity log via self._activity."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        handler = self._make_handler(anima_dir)

        handler._log_tool_activity("search_memory", {"query": "test"})

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "tool_use"
        assert entry["tool"] == "search_memory"

    def test_log_tool_activity_channel_post(self, tmp_path):
        """_log_tool_activity logs channel_post with correct fields."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        handler = self._make_handler(anima_dir)

        handler._log_tool_activity(
            "post_channel", {"text": "Hello team!", "channel": "general"},
        )

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["type"] == "channel_post"
        assert entry["content"] == "Hello team!"
        assert entry["channel"] == "general"

    def test_log_tool_activity_read_channel(self, tmp_path):
        """_log_tool_activity logs channel_read with correct fields."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        handler = self._make_handler(anima_dir)

        handler._log_tool_activity("read_channel", {"channel": "ops", "limit": 10})

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["type"] == "channel_read"
        assert entry["channel"] == "ops"

    def test_log_tool_activity_call_human(self, tmp_path):
        """_log_tool_activity logs human_notify with correct fields."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        handler = self._make_handler(anima_dir)

        handler._log_tool_activity(
            "call_human", {"body": "Urgent: server down"},
        )

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["type"] == "human_notify"
        assert "Urgent: server down" in entry["content"]
        assert entry["via"] == "configured_channels"

    def test_shared_activity_instance_across_calls(self, tmp_path):
        """The same ActivityLogger instance is reused across tool calls."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        handler = self._make_handler(anima_dir)

        activity_id = id(handler._activity)
        handler._log_tool_activity("search_memory", {"query": "a"})
        assert id(handler._activity) == activity_id
        handler._log_tool_activity("read_channel", {"channel": "general"})
        assert id(handler._activity) == activity_id


# ── Cross-component consistency ──────────────────────────────


class TestCrossComponentConsistency:
    """Verify that activity logs written by different components are compatible."""

    def test_anima_and_handler_write_to_same_directory(self, tmp_path):
        """Both DigitalAnima and ToolHandler _activity instances share the same log dir."""
        from core.anima import DigitalAnima
        from core.tooling.handler import ToolHandler

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        # Create both with the same anima_dir
        anima = object.__new__(DigitalAnima)
        anima.anima_dir = anima_dir
        anima.name = anima_dir.name
        anima._activity = ActivityLogger(anima_dir)

        handler = object.__new__(ToolHandler)
        handler._anima_dir = anima_dir
        handler._activity = ActivityLogger(anima_dir)

        # Both should write to the same log directory
        assert anima._activity._log_dir == handler._activity._log_dir

    def test_entries_from_both_components_in_same_file(self, tmp_path):
        """Entries from DigitalAnima and ToolHandler end up in the same JSONL file."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        activity_anima = ActivityLogger(anima_dir)
        activity_handler = ActivityLogger(anima_dir)

        activity_anima.log("message_received", content="hello")
        activity_handler.log("tool_use", tool="search_memory")

        log_dir = anima_dir / "activity_log"
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        types = [json.loads(line)["type"] for line in lines]
        assert "message_received" in types
        assert "tool_use" in types
