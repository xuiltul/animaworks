# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for workspace emotion tag handling and status notification.

Tests the complete SSE streaming flow to verify:
1. The done event provides clean summary + emotion for frontend consumption
2. text_delta events stream raw text (frontend strips emotion tags)
3. anima.status WebSocket events are emitted with correct structure
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry


# ── Test Helpers ─────────────────────────────────────────────


def _make_test_app():
    """Create a test FastAPI app with mock supervisor."""
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()
    app.state.supervisor = MagicMock()
    app.state.supervisor.is_bootstrapping = MagicMock(return_value=False)

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


# ── E2E: Emotion tag in streaming flow ──────────────────────


class TestWorkspaceEmotionStreamE2E:
    """E2E: Full streaming flow with emotion tags, verifying the done event
    contract that workspace app.js relies on after the fix."""

    async def test_done_event_has_clean_summary_for_workspace(self):
        """The done event must provide summary without emotion tags,
        so that workspace app.js can use data.summary to override
        the streaming text (which may contain raw tags)."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                chunk=json.dumps({"type": "text_delta", "text": "わぁ、楽しそう！"})
            )
            yield _ipc_resp(
                chunk=json.dumps({
                    "type": "text_delta",
                    "text": '\n<!-- emotion: {"emotion": "smile"} -->',
                })
            )
            yield _ipc_resp(
                done=True,
                result={
                    "response": 'わぁ、楽しそう！\n<!-- emotion: {"emotion": "smile"} -->',
                    "cycle_result": {
                        "summary": 'わぁ、楽しそう！\n<!-- emotion: {"emotion": "smile"} -->',
                    },
                },
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat/stream",
                json={"message": "何か楽しいことない？", "from_person": "user"},
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)

        # Verify text_delta events contain raw text (including emotion tag)
        deltas = [e for e in events if e["event"] == "text_delta"]
        assert len(deltas) == 2
        all_text = "".join(d["data"]["text"] for d in deltas)
        assert "<!-- emotion" in all_text  # raw tag present during streaming

        # Verify done event has clean summary and emotion
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        done_data = done_events[0]["data"]
        assert done_data["summary"] == "わぁ、楽しそう！"
        assert done_data["emotion"] == "smile"
        assert "<!-- emotion" not in done_data["summary"]

    async def test_streaming_with_multiple_text_deltas_and_tools(self):
        """Test complex streaming with tools + text + emotion tag."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "tool_end", "tool_id": "t1", "tool_name": "web_search"})
            )
            yield _ipc_resp(
                chunk=json.dumps({"type": "text_delta", "text": "検索結果をまとめます。"})
            )
            yield _ipc_resp(
                done=True,
                result={
                    "response": '検索結果をまとめます。\n<!-- emotion: {"emotion": "thinking"} -->',
                    "cycle_result": {
                        "summary": '検索結果をまとめます。\n<!-- emotion: {"emotion": "thinking"} -->',
                    },
                },
            )

        app.state.supervisor.processes = {"kotoha": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/kotoha/chat/stream",
                json={"message": "最新ニュースを調べて"},
            )

        events = _parse_sse_events(resp.text)
        done_data = [e for e in events if e["event"] == "done"][0]["data"]
        assert done_data["summary"] == "検索結果をまとめます。"
        assert done_data["emotion"] == "thinking"

    async def test_no_emotion_tag_produces_neutral(self):
        """When LLM doesn't include emotion tag, done event has neutral."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                done=True,
                result={
                    "response": "シンプルな返事です。",
                    "cycle_result": {"summary": "シンプルな返事です。"},
                },
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat/stream",
                json={"message": "テスト"},
            )

        events = _parse_sse_events(resp.text)
        done_data = [e for e in events if e["event"] == "done"][0]["data"]
        assert done_data["summary"] == "シンプルな返事です。"
        assert done_data["emotion"] == "neutral"


# ── E2E: Status notification events ─────────────────────────


class TestWorkspaceStatusEventsE2E:
    """E2E: Verify anima.status WebSocket events emitted during chat,
    testing the structure the frontend deduplication relies on."""

    async def test_stream_emits_thinking_then_idle(self):
        """A streaming chat should emit exactly thinking → idle status events."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                done=True,
                result={
                    "response": "Response",
                    "cycle_result": {"summary": "Response"},
                },
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/sakura/chat/stream",
                json={"message": "Hello"},
            )

        # Collect all anima.status broadcasts
        ws = app.state.ws_manager
        status_events = []
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else {}
            if isinstance(payload, dict) and payload.get("type") == "anima.status":
                status_events.append(payload["data"])

        assert len(status_events) == 2
        assert status_events[0]["name"] == "sakura"
        assert status_events[0]["status"] == "thinking"
        assert status_events[1]["name"] == "sakura"
        assert status_events[1]["status"] == "idle"

    async def test_status_events_have_name_and_status_fields(self):
        """Each anima.status event must have name and status for dedup."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                done=True,
                result={
                    "response": "OK",
                    "cycle_result": {"summary": "OK"},
                },
            )

        app.state.supervisor.processes = {"kotoha": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/kotoha/chat/stream",
                json={"message": "Hi"},
            )

        ws = app.state.ws_manager
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else {}
            if isinstance(payload, dict) and payload.get("type") == "anima.status":
                data = payload["data"]
                assert "name" in data, "anima.status event must have 'name'"
                assert "status" in data, "anima.status event must have 'status'"

    async def test_non_streaming_chat_also_emits_status_pair(self):
        """Non-streaming /chat endpoint should also emit thinking → idle."""
        app = _make_test_app()

        app.state.supervisor.send_request = AsyncMock(return_value={
            "response": "Answer\n<!-- emotion: {\"emotion\": \"smile\"} -->"
        })

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat",
                json={"message": "Question"},
            )

        assert resp.status_code == 200

        ws = app.state.ws_manager
        status_events = []
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else {}
            if isinstance(payload, dict) and payload.get("type") == "anima.status":
                status_events.append(payload["data"])

        assert len(status_events) == 2
        assert status_events[0]["status"] == "thinking"
        assert status_events[1]["status"] == "idle"
