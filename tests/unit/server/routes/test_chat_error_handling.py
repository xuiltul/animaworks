"""Unit tests for error handling in server/routes/chat.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.supervisor.ipc import IPCResponse
from server.stream_registry import StreamRegistry


def _make_test_app(supervisor=None):
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.animas = {}
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()
    app.state.supervisor = supervisor or MagicMock()
    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


# ── Non-streaming error handling ─────────────────────────────────


def _make_supervisor(**overrides):
    """Create a supervisor mock with is_bootstrapping=False and processes={"alice"}."""
    sup = MagicMock()
    sup.is_bootstrapping = MagicMock(return_value=False)
    sup.processes = {"alice"}
    for key, value in overrides.items():
        setattr(sup, key, value)
    return sup


class TestChatErrorHandling:
    async def test_chat_runtime_error_returns_500(self):
        sup = _make_supervisor(
            send_request=AsyncMock(side_effect=RuntimeError("process not running")),
        )
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/chat", json={"message": "hi"})
        assert resp.status_code == 500

    async def test_chat_timeout_returns_504(self):
        sup = _make_supervisor(
            send_request=AsyncMock(side_effect=asyncio.TimeoutError()),
        )
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/chat", json={"message": "hi"})
        assert resp.status_code == 504

    async def test_greet_timeout_returns_504(self):
        sup = _make_supervisor(
            send_request=AsyncMock(side_effect=asyncio.TimeoutError()),
        )
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/greet")
        assert resp.status_code == 504

    async def test_greet_runtime_error_returns_500(self):
        sup = _make_supervisor(
            send_request=AsyncMock(side_effect=RuntimeError("IPC not connected")),
        )
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/greet")
        assert resp.status_code == 500


# ── Streaming error codes ────────────────────────────────────────


def _parse_sse_events(body: str) -> list[dict]:
    """Parse SSE body into list of {event, data} dicts."""
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[7:].strip()
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        if data_lines:
            try:
                events.append({
                    "event": event_name,
                    "data": json.loads("\n".join(data_lines)),
                })
            except json.JSONDecodeError:
                pass
    return events


class TestStreamErrorCodes:
    async def test_stream_anima_not_found_has_code(self):
        async def _raise_key_error(*args, **kwargs):
            raise KeyError("alice")
            yield  # noqa: unreachable

        sup = _make_supervisor()
        sup.send_request_stream = _raise_key_error
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )
        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert error_events[0]["data"]["code"] == "ANIMA_NOT_FOUND"

    async def test_stream_generic_error_has_code(self):
        async def _raise_generic(*args, **kwargs):
            raise ValueError("something broke")
            yield  # noqa: unreachable

        sup = _make_supervisor()
        sup.send_request_stream = _raise_generic
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )
        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        # ValueError is now caught specifically as IPC_ERROR (more descriptive)
        assert error_events[0]["data"]["code"] == "IPC_ERROR"

    async def test_stream_timeout_has_code(self):
        async def _raise_timeout(*args, **kwargs):
            raise TimeoutError("timed out")
            yield  # noqa: unreachable

        sup = _make_supervisor()
        sup.send_request_stream = _raise_timeout
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )
        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert error_events[0]["data"]["code"] == "IPC_TIMEOUT"

    async def test_stream_error_chunk_propagates_code(self):
        """Error chunks from IPC with 'code' field should be propagated to SSE."""
        async def _stream_with_error_chunk(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({
                    "type": "error",
                    "code": "LLM_ERROR",
                    "message": "Model overloaded",
                }),
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream_with_error_chunk
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )
        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert error_events[0]["data"]["code"] == "LLM_ERROR"


# ── Keep-alive handling ─────────────────────────────────────────


class TestStreamKeepalive:
    @patch("core.config.load_config")
    async def test_keepalive_chunk_becomes_sse_comment(self, mock_load_config):
        """Keep-alive chunks should be emitted as SSE comments, not events."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream_with_keepalive(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({"type": "keepalive", "elapsed_s": 30}),
            )
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "hello", "cycle_result": {"summary": "hello"}},
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream_with_keepalive
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        body = resp.text
        # SSE comment line is present
        assert ": keepalive" in body
        # Keep-alive must NOT appear as a named SSE event
        assert "event: keepalive" not in body
        # Normal done event is still emitted
        assert "event: done" in body

    @patch("core.config.load_config")
    async def test_keepalive_not_rendered_as_event(self, mock_load_config):
        """Parsed SSE events should never contain a keep-alive entry."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream_keepalive_only(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({"type": "keepalive", "elapsed_s": 10}),
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream_keepalive_only
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        events = _parse_sse_events(resp.text)
        keepalive_events = [e for e in events if e.get("event") == "keepalive"]
        assert keepalive_events == []
