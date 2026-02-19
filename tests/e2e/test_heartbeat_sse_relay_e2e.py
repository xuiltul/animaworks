# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for heartbeat SSE relay.

Validates:
  - SSE event types (heartbeat_relay_start, heartbeat_relay, heartbeat_relay_done)
    are correctly formatted by _handle_chunk()
  - Full relay flow: relay events precede normal chat events
"""
from __future__ import annotations

import json

import pytest

from server.routes.chat import _format_sse, _handle_chunk

pytestmark = pytest.mark.e2e


# ── SSE Event Formatting ─────────────────────────────────


class TestHeartbeatRelaySSEEvents:
    """Test that _handle_chunk correctly formats heartbeat relay events."""

    def test_heartbeat_relay_start(self) -> None:
        """heartbeat_relay_start chunk produces correct SSE frame."""
        chunk = {
            "type": "heartbeat_relay_start",
            "message": "ハートビート処理中（kotohaからのメッセージを確認中）",
        }
        frame, text = _handle_chunk(chunk)
        assert frame is not None
        assert "heartbeat_relay_start" in frame
        assert text == ""

        # Parse the SSE data
        data_line = frame.split("data: ", 1)[1].split("\n")[0]
        data = json.loads(data_line)
        assert data["message"] == "ハートビート処理中（kotohaからのメッセージを確認中）"

    def test_heartbeat_relay_text(self) -> None:
        """heartbeat_relay chunk produces correct SSE frame with text."""
        chunk = {
            "type": "heartbeat_relay",
            "text": "メッセージを確認しています",
        }
        frame, text = _handle_chunk(chunk)
        assert frame is not None
        assert "heartbeat_relay" in frame
        # heartbeat_relay returns the text for accumulation
        assert text == "メッセージを確認しています"

        data_line = frame.split("data: ", 1)[1].split("\n")[0]
        data = json.loads(data_line)
        assert data["text"] == "メッセージを確認しています"

    def test_heartbeat_relay_empty_text(self) -> None:
        """heartbeat_relay with empty text still produces valid SSE frame."""
        chunk = {"type": "heartbeat_relay", "text": ""}
        frame, text = _handle_chunk(chunk)
        assert frame is not None
        assert text == ""

    def test_heartbeat_relay_done(self) -> None:
        """heartbeat_relay_done chunk produces correct SSE frame."""
        chunk = {"type": "heartbeat_relay_done"}
        frame, text = _handle_chunk(chunk)
        assert frame is not None
        assert "heartbeat_relay_done" in frame
        assert text == ""

        data_line = frame.split("data: ", 1)[1].split("\n")[0]
        data = json.loads(data_line)
        assert data == {}

    def test_heartbeat_relay_start_default_message(self) -> None:
        """heartbeat_relay_start without message key uses default."""
        chunk = {"type": "heartbeat_relay_start"}
        frame, _ = _handle_chunk(chunk)
        assert frame is not None
        data_line = frame.split("data: ", 1)[1].split("\n")[0]
        data = json.loads(data_line)
        assert data["message"] == "処理中です"


class TestHeartbeatRelaySSEFormat:
    """Test SSE frame format compliance."""

    def test_relay_start_sse_format(self) -> None:
        """SSE frame follows 'event: ...\\ndata: ...\\n\\n' format."""
        chunk = {"type": "heartbeat_relay_start", "message": "処理中"}
        frame, _ = _handle_chunk(chunk)
        lines = frame.split("\n")
        assert lines[0].startswith("event: heartbeat_relay_start")
        assert lines[1].startswith("data: ")
        assert lines[2] == ""
        assert lines[3] == ""

    def test_relay_sse_format(self) -> None:
        """heartbeat_relay SSE frame follows correct format."""
        chunk = {"type": "heartbeat_relay", "text": "chunk"}
        frame, _ = _handle_chunk(chunk)
        lines = frame.split("\n")
        assert lines[0].startswith("event: heartbeat_relay")
        assert not lines[0].startswith("event: heartbeat_relay_")
        assert lines[1].startswith("data: ")

    def test_relay_done_sse_format(self) -> None:
        """heartbeat_relay_done SSE frame follows correct format."""
        chunk = {"type": "heartbeat_relay_done"}
        frame, _ = _handle_chunk(chunk)
        lines = frame.split("\n")
        assert lines[0] == "event: heartbeat_relay_done"
        assert lines[1].startswith("data: ")


class TestHeartbeatRelayWithOtherEvents:
    """Test that heartbeat relay events coexist with existing event types."""

    def test_text_delta_unaffected(self) -> None:
        """text_delta events still work correctly alongside relay events."""
        chunk = {"type": "text_delta", "text": "Hello"}
        frame, text = _handle_chunk(chunk)
        assert frame is not None
        assert "text_delta" in frame
        assert text == ""

    def test_cycle_done_unaffected(self) -> None:
        """cycle_done events still work correctly."""
        chunk = {
            "type": "cycle_done",
            "cycle_result": {"summary": "Test response"},
        }
        frame, text = _handle_chunk(chunk)
        assert frame is not None
        assert "done" in frame
        assert text == "Test response"

    def test_bootstrap_busy_unaffected(self) -> None:
        """bootstrap_busy events still work correctly."""
        chunk = {"type": "bootstrap_busy", "message": "初期化中"}
        frame, _ = _handle_chunk(chunk)
        assert frame is not None
        assert "bootstrap" in frame

    def test_full_relay_sequence(self) -> None:
        """Simulate a full relay sequence followed by normal events."""
        chunks = [
            {"type": "heartbeat_relay_start", "message": "HB処理中"},
            {"type": "heartbeat_relay", "text": "Checking "},
            {"type": "heartbeat_relay", "text": "messages..."},
            {"type": "heartbeat_relay_done"},
            {"type": "text_delta", "text": "こんにちは"},
            {"type": "cycle_done", "cycle_result": {"summary": "こんにちは"}},
        ]

        frames = []
        for chunk in chunks:
            frame, _ = _handle_chunk(chunk)
            if frame:
                frames.append(frame)

        # All chunks produced frames
        assert len(frames) == 6

        # Verify event types in order
        event_names = []
        for frame in frames:
            event_line = frame.split("\n")[0]
            event_name = event_line.replace("event: ", "")
            event_names.append(event_name)

        assert event_names == [
            "heartbeat_relay_start",
            "heartbeat_relay",
            "heartbeat_relay",
            "heartbeat_relay_done",
            "text_delta",
            "done",
        ]
