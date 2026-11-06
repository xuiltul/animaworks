"""Unit tests for thinking_text handling in SSE done events."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from server.routes.chat import _chunk_to_event, _handle_chunk

_CLOSE_TAG = chr(60) + "/" + "think" + chr(62)  #  response
_OPEN_TAG = chr(60) + "think" + chr(62)  #  thinking


def _sse_data(sse_str: str) -> dict:
    data_line = [line for line in sse_str.split("\n") if line.startswith("data:")][0]
    return json.loads(data_line[len("data:") :])


class TestDoneEventThinkingText:
    """Verify that thinking_text is stripped from done SSE and replaced with thinking_summary."""

    def _make_cycle_done_chunk(self, thinking_text: str = "", summary: str = "response"):
        return {
            "type": "cycle_done",
            "cycle_result": {
                "trigger": "chat",
                "action": "responded",
                "summary": summary,
                "thinking_text": thinking_text,
                "duration_ms": 100,
                "context_usage_ratio": 0.5,
                "total_turns": 1,
                "tool_call_records": [{"name": "test_tool"}],
                "usage": None,
            },
        }

    def test_thinking_text_removed_from_handle_chunk(self):
        chunk = self._make_cycle_done_chunk(thinking_text="secret thinking")
        sse_str, _ = _handle_chunk(chunk)
        assert sse_str is not None
        data = _sse_data(sse_str)
        assert "thinking_text" not in data
        assert data.get("thinking_summary") == "secret thinking"

    def test_thinking_summary_truncated_to_5000(self):
        long_thinking = "x" * 8000
        chunk = self._make_cycle_done_chunk(thinking_text=long_thinking)
        sse_str, _ = _handle_chunk(chunk)
        data = _sse_data(sse_str)
        assert len(data["thinking_summary"]) == 5000

    def test_empty_thinking_text_gives_null_summary(self):
        chunk = self._make_cycle_done_chunk(thinking_text="")
        sse_str, _ = _handle_chunk(chunk)
        data = _sse_data(sse_str)
        assert data["thinking_summary"] is None

    def test_tool_call_records_removed(self):
        chunk = self._make_cycle_done_chunk(thinking_text="think")
        sse_str, _ = _handle_chunk(chunk)
        data = _sse_data(sse_str)
        assert "tool_call_records" not in data

    def test_chunk_to_event_thinking_text_removed(self):
        chunk = self._make_cycle_done_chunk(thinking_text="secret")
        result = _chunk_to_event(chunk)
        assert result is not None
        event_name, payload = result
        assert event_name == "done"
        assert "thinking_text" not in payload
        assert payload.get("thinking_summary") == "secret"

    def test_chunk_to_event_tool_records_removed(self):
        chunk = self._make_cycle_done_chunk(thinking_text="")
        result = _chunk_to_event(chunk)
        assert result is not None
        _, payload = result
        assert "tool_call_records" not in payload

    # ── Defensive strip_thinking_tags in done event ──────

    def test_handle_chunk_strips_leaked_think_tags_from_summary(self):
        """If summary still contains  thinking... response, strip and move to thinking_summary."""
        chunk = self._make_cycle_done_chunk(
            thinking_text="",
            summary=f"{_OPEN_TAG}leaked reasoning{_CLOSE_TAG}clean response",
        )
        sse_str, clean = _handle_chunk(chunk)
        data = _sse_data(sse_str)
        assert data["summary"] == "clean response"
        assert _OPEN_TAG not in data["summary"]
        assert data["thinking_summary"] == "leaked reasoning"
        assert clean == "clean response"

    def test_handle_chunk_preserves_body_on_orphan_close_tag(self):
        """vLLM pattern: orphan  response with no  thinking.
        Body before the stray close tag must be PRESERVED (regression f93221ae),
        not discarded as if it were leaked thinking."""
        chunk = self._make_cycle_done_chunk(
            thinking_text="",
            summary=f"reasoning content {_CLOSE_TAG}\n\nactual response",
        )
        sse_str, clean = _handle_chunk(chunk)
        data = _sse_data(sse_str)
        assert _CLOSE_TAG not in data["summary"]
        assert "reasoning content" in data["summary"]
        assert "actual response" in data["summary"]
        assert data["thinking_summary"] is None
        assert "reasoning content" in clean and "actual response" in clean

    def test_chunk_to_event_preserves_body_on_orphan_close_tag(self):
        """WebSocket path also preserves the body when  response appears without  thinking."""
        chunk = self._make_cycle_done_chunk(
            thinking_text="",
            summary=f"pre answer {_CLOSE_TAG} post answer",
        )
        result = _chunk_to_event(chunk)
        assert result is not None
        _, payload = result
        assert _CLOSE_TAG not in payload["summary"]
        assert "pre answer" in payload["summary"]
        assert "post answer" in payload["summary"]
        assert payload["thinking_summary"] is None

    def test_chunk_to_event_strips_leaked_think_tags(self):
        """WebSocket path also strips leaked think tags from summary."""
        chunk = self._make_cycle_done_chunk(
            thinking_text="",
            summary=f"{_OPEN_TAG}ws leaked{_CLOSE_TAG}ws clean",
        )
        result = _chunk_to_event(chunk)
        assert result is not None
        _, payload = result
        assert payload["summary"] == "ws clean"
        assert payload["thinking_summary"] == "ws leaked"

    def test_handle_chunk_preserves_existing_thinking_text(self):
        """If thinking_text is already set, leaked tags don't overwrite it."""
        chunk = self._make_cycle_done_chunk(
            thinking_text="proper thinking from safety net",
            summary=f"{_OPEN_TAG}leaked{_CLOSE_TAG}response",
        )
        sse_str, _ = _handle_chunk(chunk)
        data = _sse_data(sse_str)
        assert data["summary"] == "response"
        assert data["thinking_summary"] == "proper thinking from safety net"
