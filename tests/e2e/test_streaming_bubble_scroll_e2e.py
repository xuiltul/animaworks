# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for chat bubble scroll behavior during SSE streaming.

Verifies:
1. SSE endpoint delivers multi-line text_delta events correctly
2. chat.js uses rAF + scrollIntoView for streaming updates
3. renderSimpleMarkdown converts newlines to <br> tags
4. CSS allows multi-line rendering (white-space: pre-wrap)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry

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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_source(relpath: str) -> str:
    return (_project_root() / relpath).read_text(encoding="utf-8")


# ── SSE streaming with multi-line text ───────────────────────


class TestStreamingMultiLineSSE:
    """Verify SSE streaming endpoint delivers multi-line text correctly."""

    async def test_multiline_text_delta_events_streamed(self):
        """Multi-line text delivered as text_delta with newlines preserved."""
        app = _make_test_app()
        multiline = "こんにちは！\n\n今日はいい天気ですね。\n散歩に行きませんか？"

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "こんにちは！\n\n"}))
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "今日はいい天気ですね。\n"}))
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "散歩に行きませんか？"}))
            yield _ipc_resp(
                done=True,
                result={"response": multiline, "cycle_result": {"summary": multiline}},
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat/stream",
                json={"message": "今日の天気は？", "from_person": "user"},
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        deltas = [e for e in events if e["event"] == "text_delta"]
        assert len(deltas) == 3
        all_text = "".join(d["data"]["text"] for d in deltas)
        assert all_text.count("\n") >= 3, "Newlines must be preserved in streamed text"

    async def test_done_event_preserves_multiline_summary(self):
        """Done event summary preserves multi-line content."""
        app = _make_test_app()
        summary = "行を1つ目\n行を2つ目\n行を3つ目"

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": summary}))
            yield _ipc_resp(
                done=True,
                result={"response": summary, "cycle_result": {"summary": summary}},
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat/stream", json={"message": "テスト"},
            )

        done_events = [e for e in _parse_sse_events(resp.text) if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["summary"].count("\n") == 2

    async def test_streaming_long_response_with_paragraphs(self):
        """Long streaming response with multiple paragraphs delivered correctly."""
        app = _make_test_app()
        paragraphs = [
            "最初の段落です。これは導入文になります。",
            "2番目の段落です。ここでは詳細を説明します。",
            "3番目の段落です。さらに補足情報を追加します。",
            "最後の段落です。まとめとして結論を述べます。",
        ]
        full_text = "\n\n".join(paragraphs)

        async def mock_stream(*args, **kwargs):
            for i, para in enumerate(paragraphs):
                prefix = "\n\n" if i > 0 else ""
                yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": prefix + para}))
            yield _ipc_resp(
                done=True,
                result={"response": full_text, "cycle_result": {"summary": full_text}},
            )

        app.state.supervisor.processes = {"kotoha": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/kotoha/chat/stream", json={"message": "詳しく教えて"},
            )

        deltas = [e for e in _parse_sse_events(resp.text) if e["event"] == "text_delta"]
        assert len(deltas) == 4
        assert "".join(d["data"]["text"] for d in deltas) == full_text


# ── chat.js rAF + scrollIntoView pattern ─────────────────────


class TestAppJsScrollPattern:
    """Verify app.js contains streaming scroll patterns.

    After workspace chat.js deletion, streaming scroll is handled
    by app.js using scrollTop-based scrolling and rAF-throttled updates.
    """

    @pytest.fixture()
    def source(self) -> str:
        return _read_source("server/static/workspace/modules/app.js")

    def test_update_streaming_bubble_exists(self, source):
        assert "function updateStreamingBubble" in source

    def test_schedule_streaming_update_exists(self, source):
        assert "function scheduleStreamingUpdate" in source

    def test_schedule_streaming_update_uses_raf(self, source):
        assert "_convRafPending" in source
        assert "requestAnimationFrame" in source

    def test_update_streaming_bubble_scrolls(self, source):
        """updateStreamingBubble scrolls the messages container."""
        idx = source.index("function updateStreamingBubble")
        end_marker = source.find("\n// ──", idx + 1)
        if end_marker == -1:
            end_marker = source.find("\nfunction ", idx + 100)
        func_body = source[idx:end_marker]
        assert "scrollTop" in func_body or "scrollIntoView" in func_body


# ── CSS multi-line rendering ─────────────────────────────────


class TestChatBubbleCSSMultiLine:
    """Verify workspace CSS allows multi-line chat bubble rendering."""

    @pytest.fixture()
    def css(self) -> str:
        return _read_source("server/static/workspace/style.css")

    def test_chat_bubble_has_pre_wrap(self, css):
        assert "white-space: pre-wrap" in css

    def test_chat_bubble_has_word_break(self, css):
        assert "word-break: break-word" in css

    def test_streaming_bubble_has_min_height(self, css):
        assert "min-height" in css


# ── Markdown line break rendering ────────────────────────────


class TestMarkdownLineBreakRendering:
    """Verify utils.js renderSimpleMarkdown handles line breaks."""

    def test_render_simple_markdown_exists(self):
        assert "export function renderSimpleMarkdown" in _read_source(
            "server/static/workspace/modules/utils.js"
        )

    def test_newline_to_br_conversion(self):
        assert "<br>" in _read_source("server/static/workspace/modules/utils.js")

    def test_app_js_uses_render_simple_markdown(self):
        assert "renderSimpleMarkdown" in _read_source(
            "server/static/workspace/modules/app.js"
        )


# ── Streaming + scroll integration ──────────────────────────


class TestStreamingScrollIntegration:
    """Integration: full SSE pipeline delivers content for rAF scroll."""

    async def test_text_delta_events_interleaved_with_tool_events(self):
        """Tool events interleaved with text_delta — all text preserved."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": "調べてみます。\n"}))
            yield _ipc_resp(chunk=json.dumps({
                "type": "tool_start", "tool_name": "web_search", "tool_id": "t1",
            }))
            yield _ipc_resp(chunk=json.dumps({
                "type": "tool_end", "tool_id": "t1", "tool_name": "web_search",
            }))
            yield _ipc_resp(chunk=json.dumps({
                "type": "text_delta", "text": "\n検索結果:\n- 項目1\n- 項目2\n- 項目3",
            }))
            full = "調べてみます。\n\n検索結果:\n- 項目1\n- 項目2\n- 項目3"
            yield _ipc_resp(
                done=True,
                result={"response": full, "cycle_result": {"summary": full}},
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat/stream",
                json={"message": "最新ニュースを調べて"},
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        deltas = [e for e in events if e["event"] == "text_delta"]
        assert len(deltas) == 2
        all_text = "".join(d["data"]["text"] for d in deltas)
        assert all_text.count("\n") >= 5

        assert len([e for e in events if e["event"] == "tool_start"]) == 1
        assert len([e for e in events if e["event"] == "tool_end"]) == 1

    async def test_empty_then_multiline_streaming(self):
        """Incremental line-by-line streaming produces correct events."""
        app = _make_test_app()
        lines = [
            "1行目: はじめまして。\n",
            "2行目: 質問にお答えします。\n",
            "3行目: 以下のとおりです。\n",
            "4行目: まず第一に...\n",
            "5行目: 次に...\n",
            "6行目: 最後に...",
        ]

        async def mock_stream(*args, **kwargs):
            for line in lines:
                yield _ipc_resp(chunk=json.dumps({"type": "text_delta", "text": line}))
            full = "".join(lines)
            yield _ipc_resp(
                done=True,
                result={"response": full, "cycle_result": {"summary": full}},
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat/stream", json={"message": "詳細に回答して"},
            )

        deltas = [e for e in _parse_sse_events(resp.text) if e["event"] == "text_delta"]
        assert len(deltas) == 6

        accumulated = "".join(d["data"]["text"] for d in deltas)
        assert accumulated.count("\n") == 5

    async def test_sse_content_type_is_event_stream(self):
        """Streaming endpoint returns content-type: text/event-stream."""
        app = _make_test_app()

        async def mock_stream(*args, **kwargs):
            yield _ipc_resp(
                done=True,
                result={"response": "OK", "cycle_result": {"summary": "OK"}},
            )

        app.state.supervisor.processes = {"sakura": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/sakura/chat/stream", json={"message": "テスト"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
