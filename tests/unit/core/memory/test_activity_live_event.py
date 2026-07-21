from __future__ import annotations

"""Tests for ActivityLogger._emit_live_event — live event broadcast."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityLogger


@pytest.fixture()
def activity_logger(tmp_path: Path) -> ActivityLogger:
    ActivityLogger._live_rate_limiter.reset()
    anima_dir = tmp_path / "animas" / "testanima"
    anima_dir.mkdir(parents=True)
    return ActivityLogger(anima_dir)


class TestEmitLiveEvent:
    def test_visible_tool_use_emits_event_file(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("tool_use", tool="delegate_task", summary="task delegation")

        event_dir = tmp_path / "run" / "events" / "testanima"
        files = list(event_dir.glob("ta_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["event"] == "anima.tool_activity"
        assert data["data"]["name"] == "testanima"
        assert data["data"]["type"] == "tool_use"
        assert data["data"]["tool"] == "delegate_task"

    def test_any_tool_use_emits(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("tool_use", tool="Read", summary="/foo/bar.py")

        event_dir = tmp_path / "run" / "events" / "testanima"
        files = list(event_dir.glob("ta_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))["data"]
        assert data["kind"] == "tool_use"
        assert data["tool"] == "Read"

    def test_tool_result_emits_normalized_payload(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_result",
                tool="Bash",
                summary="x" * 250,
                meta={"result_status": "fail"},
            )

        event_dir = tmp_path / "run" / "events" / "testanima"
        files = list(event_dir.glob("ta_*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))["data"]
        assert data["kind"] == "tool_result"
        assert data["is_error"] is True
        assert len(data["summary"]) == 200

    def test_rate_limit_is_per_anima_and_reports_drops(self, activity_logger: ActivityLogger, tmp_path: Path):
        other_dir = tmp_path / "animas" / "other"
        other_dir.mkdir(parents=True)
        other_logger = ActivityLogger(other_dir)

        with (
            patch("core.memory.activity.get_data_dir", return_value=tmp_path),
            patch("core.memory.activity.time.monotonic", return_value=100.0),
        ):
            for index in range(7):
                activity_logger.log("tool_use", tool="Read", summary=str(index))
            other_logger.log("tool_use", tool="Read", summary="other")

        event_dir = tmp_path / "run" / "events" / "testanima"
        assert len(list(event_dir.glob("ta_*.json"))) == 5
        assert len(list((tmp_path / "run" / "events" / "other").glob("ta_*.json"))) == 1

        with (
            patch("core.memory.activity.get_data_dir", return_value=tmp_path),
            patch("core.memory.activity.time.monotonic", return_value=101.0),
        ):
            activity_logger.log("tool_result", tool="Read", content="ok")

        events = [json.loads(path.read_text(encoding="utf-8"))["data"] for path in event_dir.glob("ta_*.json")]
        assert len(events) == 6
        assert next(event for event in events if event["kind"] == "tool_result")["dropped"] == 2

    def test_inbox_processing_start_emits(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("inbox_processing_start", summary="processing")

        event_dir = tmp_path / "run" / "events" / "testanima"
        files = list(event_dir.glob("ta_*.json"))
        assert len(files) == 1

    def test_non_live_event_does_not_emit(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("message_received", content="hello")

        event_dir = tmp_path / "run" / "events" / "testanima"
        assert not event_dir.exists() or len(list(event_dir.glob("ta_*.json"))) == 0

    def test_heartbeat_start_does_not_emit(self, activity_logger: ActivityLogger, tmp_path: Path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log("heartbeat_start", summary="heartbeat")

        event_dir = tmp_path / "run" / "events" / "testanima"
        assert not event_dir.exists() or len(list(event_dir.glob("ta_*.json"))) == 0

    def test_emit_failure_does_not_break_log(self, activity_logger: ActivityLogger, tmp_path: Path):
        """_emit_live_event failure should not prevent the log entry from being written."""
        with patch("core.memory.activity.get_data_dir", side_effect=OSError("disk full")):
            entry = activity_logger.log("tool_use", tool="Bash", summary="ls")

        assert entry.tool == "Bash"
        log_dir = activity_logger._log_dir
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
