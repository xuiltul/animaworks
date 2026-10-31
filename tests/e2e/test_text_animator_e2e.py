# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for TextAnimator FE streaming smoothing.

Verifies:
1. SSE pipeline delivers multiple rapid text_delta events for TextAnimator buffering
2. Done event carries summary for TextAnimator flush
3. TextAnimator class structure in render-utils.js
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry

# ── Paths ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RENDER_UTILS = _PROJECT_ROOT / "server" / "static" / "shared" / "chat" / "render-utils.js"

# ── Helpers ──────────────────────────────────────────────────


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
    resp = MagicMock()
    resp.done = done
    resp.result = result
    resp.chunk = chunk
    return resp


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    current_event = "message"
    for line in body.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                events.append({"event": current_event, "data": json.loads(line[6:])})
            except json.JSONDecodeError:
                pass
    return events


# ── SSE Pipeline for TextAnimator ────────────────────────────


class TestTextAnimatorSSEPipeline:
    """Verify SSE text_delta events are delivered correctly for TextAnimator consumption."""

    async def test_rapid_text_deltas_emitted(self):
        """Multiple rapid text_delta events are sent for animator to buffer and smooth."""
        app = _make_test_app()
        full_text = "Hello world! This is a test."

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "Hello "}))
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "world! "}))
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "This is a test."}))
            yield _ipc_resp(
                done=True,
                result={"response": full_text, "cycle_result": {"summary": full_text}},
            )

        app.state.supervisor.processes = {"test-anima": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/animas/test-anima/chat/stream",
                json={"message": "hi", "from_person": "user"},
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        text_deltas = [e for e in events if e["event"] == "text_delta"]
        assert len(text_deltas) == 3, f"Expected 3 text_delta events, got {len(text_deltas)}"
        full = "".join(e["data"].get("text", "") for e in text_deltas)
        assert full == full_text

    async def test_done_event_carries_summary_for_flush(self):
        """Done event includes summary text for TextAnimator.flush() to display immediately."""
        app = _make_test_app()
        summary = "Final complete response"

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "streaming..."}))
            yield _ipc_resp(
                done=True,
                result={"response": summary, "cycle_result": {"summary": summary}},
            )

        app.state.supervisor.processes = {"test-anima": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/animas/test-anima/chat/stream",
                json={"message": "test", "from_person": "user"},
            )

        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"].get("summary") == summary

    async def test_many_small_deltas_for_animator_buffering(self):
        """Many small deltas simulate bursty API delivery that TextAnimator smooths."""
        app = _make_test_app()
        words = ["The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog."]
        full_text = "".join(words)

        async def mock_stream(*args, **kwargs):
            for w in words:
                yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": w}))
            yield _ipc_resp(
                done=True,
                result={"response": full_text, "cycle_result": {"summary": full_text}},
            )

        app.state.supervisor.processes = {"test-anima": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/animas/test-anima/chat/stream",
                json={"message": "test", "from_person": "user"},
            )

        events = _parse_sse_events(resp.text)
        text_deltas = [e for e in events if e["event"] == "text_delta"]
        assert len(text_deltas) == len(words)


# ── TextAnimator JS Class Structure ──────────────────────────


class TestTextAnimatorJSStructure:
    """Verify TextAnimator class internal structure in render-utils.js."""

    @pytest.fixture()
    def src(self) -> str:
        return _RENDER_UTILS.read_text()

    def test_constructor_accepts_char_interval(self, src: str):
        assert "charIntervalMs" in src

    def test_constructor_accepts_on_update(self, src: str):
        assert "onUpdate" in src

    def test_buffer_initialized_empty(self, src: str):
        assert 'this._buffer = ""' in src

    def test_display_len_initialized_zero(self, src: str):
        assert "this._displayLen = 0" in src

    def test_step_calculates_chars_to_add(self, src: str):
        assert "charsToAdd" in src

    def test_flush_sets_display_len_to_buffer_length(self, src: str):
        assert "this._displayLen = this._buffer.length" in src

    def test_cancel_tick_on_stop(self, src: str):
        assert "cancelAnimationFrame" in src

    def test_schedule_tick_guards_duplicate(self, src: str):
        assert "if (this._rafId != null) return" in src
