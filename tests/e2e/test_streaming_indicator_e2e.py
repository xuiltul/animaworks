# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for streaming loading indicator improvements.

Tests the SSE streaming flow to verify:
1. tool_start events are properly sent through SSE
2. tool_end events are properly sent through SSE
3. The done event includes summary
4. Multiple tool calls in sequence produce correct SSE event ordering
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry


# ── Test Helpers ─────────────────────────────────────────────

ANIMA_NAME = "test-anima"


def _make_test_app():
    """Create a test FastAPI app with mock supervisor."""
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()

    supervisor = MagicMock()
    supervisor.processes = {ANIMA_NAME: True}
    supervisor.is_bootstrapping.return_value = False
    app.state.supervisor = supervisor

    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


def _ipc_resp(*, done=False, result=None, chunk=None):
    """Create a mock IPC response."""
    resp = MagicMock()
    resp.done = done
    resp.result = result
    resp.chunk = chunk
    return resp


def _parse_sse_events(body: str) -> list[dict]:
    """Parse SSE body into list of {event, data} dicts."""
    events = []
    current_event = "message"
    for line in body.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append({"event": current_event, "data": data})
            except json.JSONDecodeError:
                pass
    return events


async def _stream_request(app, message="テスト"):
    """Send a streaming chat request and return parsed SSE events."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/animas/{ANIMA_NAME}/chat/stream",
            json={"message": message, "from_person": "human"},
        )
    return resp, _parse_sse_events(resp.text)


# ── E2E: Tool call SSE events ──────────────────────────────


class TestToolCallSSEEvents:
    """E2E: Verify tool_start and tool_end SSE events flow correctly."""

    async def test_tool_start_event_sent(self):
        """tool_start SSE event should contain tool_name."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_end", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "text_delta", "text": "検索結果です。"})
            )
            yield _ipc_resp(
                done=True,
                result={
                    "response": "検索結果です。",
                    "cycle_result": {"summary": "検索結果です。"},
                },
            )

        app.state.supervisor.send_request_stream = mock_stream

        resp, events = await _stream_request(app, "検索して")

        assert resp.status_code == 200
        tool_starts = [e for e in events if e["event"] == "tool_start"]
        assert len(tool_starts) >= 1, "tool_start event not found in SSE stream"
        assert tool_starts[0]["data"]["tool_name"] == "web_search"

    async def test_tool_end_event_sent(self):
        """tool_end SSE event should be sent after tool execution."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_end", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                done=True,
                result={
                    "response": "完了",
                    "cycle_result": {"summary": "完了"},
                },
            )

        app.state.supervisor.send_request_stream = mock_stream

        resp, events = await _stream_request(app)

        assert resp.status_code == 200
        tool_ends = [e for e in events if e["event"] == "tool_end"]
        assert len(tool_ends) >= 1, "tool_end event not found in SSE stream"

    async def test_event_ordering_tool_then_text(self):
        """Events should arrive in order: tool_start -> tool_end -> text_delta -> done."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_end", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "text_delta", "text": "結果です。"})
            )
            yield _ipc_resp(
                done=True,
                result={
                    "response": "結果です。",
                    "cycle_result": {"summary": "結果です。"},
                },
            )

        app.state.supervisor.send_request_stream = mock_stream

        resp, events = await _stream_request(app)
        event_names = [e["event"] for e in events]

        # tool_start must come before tool_end
        ts_idx = event_names.index("tool_start")
        te_idx = event_names.index("tool_end")
        assert ts_idx < te_idx, "tool_start should come before tool_end"

        # tool_end must come before text_delta
        td_idx = event_names.index("text_delta")
        assert te_idx < td_idx, "tool_end should come before text_delta"

        # done must be last
        done_idx = event_names.index("done")
        assert done_idx == len(event_names) - 1, "done should be the last event"

    async def test_multiple_tool_calls_produce_multiple_events(self):
        """Multiple sequential tool calls should each produce start/end events."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_end", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_start", "tool_name": "read_file", "tool_id": "t2"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_end", "tool_name": "read_file", "tool_id": "t2"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "text_delta", "text": "2つのツールを使いました。"})
            )
            yield _ipc_resp(
                done=True,
                result={
                    "response": "2つのツールを使いました。",
                    "cycle_result": {"summary": "2つのツールを使いました。"},
                },
            )

        app.state.supervisor.send_request_stream = mock_stream

        resp, events = await _stream_request(app)

        tool_starts = [e for e in events if e["event"] == "tool_start"]
        tool_ends = [e for e in events if e["event"] == "tool_end"]

        assert len(tool_starts) == 2, f"Expected 2 tool_start events, got {len(tool_starts)}"
        assert len(tool_ends) == 2, f"Expected 2 tool_end events, got {len(tool_ends)}"
        assert tool_starts[0]["data"]["tool_name"] == "web_search"
        assert tool_starts[1]["data"]["tool_name"] == "read_file"

    async def test_done_event_has_summary(self):
        """The done event should contain a summary field."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_end", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "text_delta", "text": "検索完了。"})
            )
            yield _ipc_resp(
                done=True,
                result={
                    "response": "検索完了。",
                    "cycle_result": {"summary": "検索完了。"},
                },
            )

        app.state.supervisor.send_request_stream = mock_stream

        resp, events = await _stream_request(app)

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1, "Expected exactly 1 done event"
        assert "summary" in done_events[0]["data"], "done event should contain summary"
