# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard visualization event enhancements.

Covers:
- New event types in _LIVE_EVENT_TYPES (message_sent, response_sent, etc.)
- Enhanced _emit_live_event payload (content, from_person, to_person, channel)
- Summary field population for message_sent and response_sent
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityLogger


@pytest.fixture()
def activity_logger(tmp_path: Path) -> ActivityLogger:
    anima_dir = tmp_path / "animas" / "testanima"
    anima_dir.mkdir(parents=True)
    return ActivityLogger(anima_dir)


def _read_event_files(tmp_path: Path) -> list[dict]:
    event_dir = tmp_path / "run" / "events" / "testanima"
    if not event_dir.exists():
        return []
    files = sorted(event_dir.glob("ta_*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


class TestNewLiveEventTypes:
    """New event types added to _LIVE_EVENT_TYPES should emit event files."""

    def test_message_sent_emits(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "message_sent",
                content="Hello from testanima",
                to_person="other",
                summary="→ other: Hello from testanima",
            )
        events = _read_event_files(tmp_path)
        assert len(events) == 1
        assert events[0]["data"]["type"] == "message_sent"

    def test_response_sent_emits(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "response_sent",
                content="Here is my response",
                to_person="human",
                summary="Here is my response",
            )
        events = _read_event_files(tmp_path)
        assert len(events) == 1
        assert events[0]["data"]["type"] == "response_sent"

    def test_channel_post_emits(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "channel_post",
                channel="general",
                content="Board post content",
                summary="Board post",
            )
        events = _read_event_files(tmp_path)
        assert len(events) == 1
        assert events[0]["data"]["type"] == "channel_post"

    def test_task_created_emits(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("task_created", summary="New task created")
        events = _read_event_files(tmp_path)
        assert len(events) == 1
        assert events[0]["data"]["type"] == "task_created"

    def test_task_updated_emits(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("task_updated", summary="Task done")
        events = _read_event_files(tmp_path)
        assert len(events) == 1
        assert events[0]["data"]["type"] == "task_updated"

    def test_message_received_still_does_not_emit(
        self, activity_logger: ActivityLogger, tmp_path: Path
    ):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("message_received", content="hi")
        events = _read_event_files(tmp_path)
        assert len(events) == 0


class TestEnhancedLiveEventPayload:
    """_emit_live_event payload should include content, from_person, to_person, channel."""

    def test_message_sent_payload_has_extended_fields(
        self, activity_logger: ActivityLogger, tmp_path: Path
    ):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "message_sent",
                content="Full message content here",
                to_person="alice",
                summary="→ alice: Full message content",
                meta={"intent": "report"},
            )
        events = _read_event_files(tmp_path)
        assert len(events) == 1
        data = events[0]["data"]
        assert data["to_person"] == "alice"
        assert "Full message" in data["content"]
        assert data["summary"] == "→ alice: Full message content"

    def test_response_sent_payload_has_content(
        self, activity_logger: ActivityLogger, tmp_path: Path
    ):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "response_sent",
                content="Detailed response text",
                to_person="user",
                channel="chat",
                summary="Detailed response text",
            )
        events = _read_event_files(tmp_path)
        data = events[0]["data"]
        assert data["to_person"] == "user"
        assert data["channel"] == "chat"
        assert "Detailed response" in data["content"]

    def test_channel_post_payload_has_channel(
        self, activity_logger: ActivityLogger, tmp_path: Path
    ):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "channel_post",
                channel="dev",
                content="Important update",
                summary="Update",
            )
        events = _read_event_files(tmp_path)
        data = events[0]["data"]
        assert data["channel"] == "dev"

    def test_content_truncated_to_200(
        self, activity_logger: ActivityLogger, tmp_path: Path
    ):
        long_content = "x" * 500
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "message_sent",
                content=long_content,
                to_person="bob",
                summary="→ bob: " + long_content[:80],
            )
        events = _read_event_files(tmp_path)
        data = events[0]["data"]
        assert len(data["content"]) <= 200
