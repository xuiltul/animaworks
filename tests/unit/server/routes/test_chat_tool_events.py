"""Unit tests for the tool_start SSE payload (argument preview)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from server.routes.chat import _chunk_to_event


class TestToolStartInputSummary:

    def test_input_is_summarised(self):
        event, payload = _chunk_to_event(
            {
                "type": "tool_start",
                "tool_name": "Bash",
                "tool_id": "t1",
                "input": {"command": "ls -la"},
            }
        )
        assert event == "tool_start"
        assert payload["input_summary"] == "ls -la"

    def test_unknown_tool_gets_generic_summary(self):
        _, payload = _chunk_to_event(
            {
                "type": "tool_start",
                "tool_name": "some_mcp_tool",
                "tool_id": "t2",
                "input": {"channel": "ops"},
            }
        )
        assert payload["input_summary"] == "channel=ops"

    def test_no_input_leaves_payload_unchanged(self):
        _, payload = _chunk_to_event(
            {"type": "tool_start", "tool_name": "Bash", "tool_id": "t3"}
        )
        assert payload == {"tool_name": "Bash", "tool_id": "t3"}

    def test_summary_is_clipped(self):
        _, payload = _chunk_to_event(
            {
                "type": "tool_start",
                "tool_name": "Bash",
                "tool_id": "t4",
                "input": {"command": "x" * 500},
            }
        )
        assert len(payload["input_summary"]) <= 200
