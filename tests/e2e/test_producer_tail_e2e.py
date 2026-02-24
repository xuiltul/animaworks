"""E2E tests for producer task / SSE-IPC separation.

Tests the full chat_stream endpoint with producer/tail architecture:
- Producer runs as a background asyncio task consuming IPC
- Tail yields SSE frames from StreamRegistry
- SSE client disconnect does NOT kill producer
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.supervisor.ipc import IPCResponse
from server.stream_registry import StreamRegistry


# ── Helpers ──────────────────────────────────────────────


def _make_supervisor(**overrides):
    sup = MagicMock()
    sup.is_bootstrapping = MagicMock(return_value=False)
    sup.processes = {"alice"}
    for key, value in overrides.items():
        setattr(sup, key, value)
    return sup


def _make_test_app(supervisor=None):
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.animas = {}
    ws = MagicMock()
    ws.broadcast = AsyncMock()
    ws.broadcast_notification = AsyncMock()
    app.state.ws_manager = ws
    app.state.stream_registry = StreamRegistry()
    app.state.supervisor = supervisor or MagicMock()
    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


def _parse_sse_events(body: str) -> list[dict]:
    """Parse SSE text into a list of event dicts."""
    events = []
    current: dict = {}
    for line in body.split("\n"):
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            try:
                current["data"] = json.loads(line[6:])
            except json.JSONDecodeError:
                current["data"] = line[6:]
        elif line == "" and current:
            events.append(current)
            current = {}
    return events


# ── Producer/Tail Architecture ────────────────────────────


class TestProducerTailArchitecture:
    """Verify the producer/tail separation works end-to-end."""

    @patch("core.config.load_config")
    async def test_normal_stream_produces_stream_start_and_done(self, mock_load_config):
        """A normal IPC stream should produce stream_start and done events."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "hello"}),
            )
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "hello", "cycle_result": {"summary": "hello"}},
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        events = _parse_sse_events(resp.text)
        event_types = [e["event"] for e in events]

        assert "stream_start" in event_types
        assert "text_delta" in event_types
        assert "done" in event_types

    @patch("core.config.load_config")
    async def test_producer_sets_done_flag_on_normal_completion(self, mock_load_config):
        """On normal completion, mark_complete(done=True) should be called."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "ok", "cycle_result": {"summary": "ok"}},
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream
        app = _make_test_app(supervisor=sup)
        registry = app.state.stream_registry

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        # Find the stream that was created
        events = _parse_sse_events(resp.text)
        stream_start = next(e for e in events if e["event"] == "stream_start")
        response_id = stream_start["data"]["response_id"]

        stream = registry.get(response_id)
        assert stream is not None
        assert stream.complete is True
        assert stream.done is True

    @patch("core.config.load_config")
    async def test_producer_task_registered_in_registry(self, mock_load_config):
        """Producer task should be registered via set_producer_task."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "ok", "cycle_result": {"summary": "ok"}},
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream
        app = _make_test_app(supervisor=sup)
        registry = app.state.stream_registry

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        events = _parse_sse_events(resp.text)
        stream_start = next(e for e in events if e["event"] == "stream_start")
        response_id = stream_start["data"]["response_id"]

        stream = registry.get(response_id)
        assert stream is not None
        assert stream.producer_task is not None
        assert stream.producer_task.done()  # Should be finished

    @patch("core.config.load_config")
    async def test_error_events_buffered_in_registry(self, mock_load_config):
        """IPC errors should add error events to StreamRegistry."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            raise RuntimeError("Process restarting")
            yield  # make it an async generator  # noqa: RUF100

        sup = _make_supervisor()
        sup.send_request_stream = _stream
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
        assert error_events[-1]["data"]["code"] == "ANIMA_RESTARTING"

    @patch("core.config.load_config")
    async def test_timeout_error_event(self, mock_load_config):
        """IPC timeout should produce an error event."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            raise TimeoutError()
            yield  # noqa: RUF100

        sup = _make_supervisor()
        sup.send_request_stream = _stream
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
        assert error_events[-1]["data"]["code"] == "IPC_TIMEOUT"

    @patch("core.config.load_config")
    async def test_done_false_on_error(self, mock_load_config):
        """On error, mark_complete(done=False) should be called."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            raise RuntimeError("Not connected")
            yield  # noqa: RUF100

        sup = _make_supervisor()
        sup.send_request_stream = _stream
        app = _make_test_app(supervisor=sup)
        registry = app.state.stream_registry

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        events = _parse_sse_events(resp.text)
        stream_start = next(e for e in events if e["event"] == "stream_start")
        response_id = stream_start["data"]["response_id"]

        stream = registry.get(response_id)
        assert stream is not None
        assert stream.complete is True
        assert stream.done is False

    @patch("core.config.load_config")
    async def test_websocket_status_emitted(self, mock_load_config):
        """Producer should emit thinking/idle WebSocket events via ws_manager."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "ok", "cycle_result": {"summary": "ok"}},
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream
        app = _make_test_app(supervisor=sup)
        ws = app.state.ws_manager

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        # Wait briefly for producer task to finish
        await asyncio.sleep(0.1)

        # Check WebSocket broadcasts
        broadcast_calls = ws.broadcast.call_args_list
        broadcast_data = [call[0][0] for call in broadcast_calls]

        # Should have thinking and idle status
        thinking = [d for d in broadcast_data if d.get("type") == "anima.status" and d.get("data", {}).get("status") == "thinking"]
        idle = [d for d in broadcast_data if d.get("type") == "anima.status" and d.get("data", {}).get("status") == "idle"]
        assert len(thinking) >= 1
        assert len(idle) >= 1

    @patch("core.config.load_config")
    async def test_resume_after_producer_completes(self, mock_load_config):
        """Resume endpoint should work after producer has completed."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "hello"}),
            )
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "hello", "cycle_result": {"summary": "hello"}},
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream
        app = _make_test_app(supervisor=sup)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First: normal stream
            resp1 = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )
            events1 = _parse_sse_events(resp1.text)
            stream_start = next(e for e in events1 if e["event"] == "stream_start")
            response_id = stream_start["data"]["response_id"]

            # Resume: get buffered events
            resp2 = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "", "resume": response_id, "last_event_id": ""},
            )

        events2 = _parse_sse_events(resp2.text)
        # Should replay all events
        assert len(events2) >= len(events1)

    @patch("core.config.load_config")
    async def test_bootstrap_side_effects_via_ws_manager(self, mock_load_config):
        """Bootstrap chunks should trigger WebSocket broadcasts via ws_manager."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({"type": "bootstrap_start"}),
            )
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({"type": "bootstrap_complete"}),
            )
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "ok", "cycle_result": {"summary": "ok"}},
            )

        sup = _make_supervisor()
        sup.send_request_stream = _stream
        app = _make_test_app(supervisor=sup)
        ws = app.state.ws_manager

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hi"},
            )

        await asyncio.sleep(0.1)

        broadcast_data = [call[0][0] for call in ws.broadcast.call_args_list]
        bootstrap_events = [d for d in broadcast_data if d.get("type") == "anima.bootstrap"]
        assert len(bootstrap_events) >= 2

    @patch("core.config.load_config")
    async def test_stream_incomplete_produces_error_event(self, mock_load_config):
        """Stream ending without done flag should produce STREAM_INCOMPLETE error."""
        mock_config = MagicMock()
        mock_config.server.ipc_stream_timeout = 60
        mock_load_config.return_value = mock_config

        async def _stream(*args, **kwargs):
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "partial"}),
            )
            # No done response — stream ends

        sup = _make_supervisor()
        sup.send_request_stream = _stream
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
        assert error_events[-1]["data"]["code"] == "STREAM_INCOMPLETE"


# ── App shutdown ─────────────────────────────────────────


class TestAppShutdown:
    """Verify cancel_all_producers is called on server shutdown."""

    async def test_cancel_all_producers_available(self):
        """StreamRegistry should have cancel_all_producers method."""
        reg = StreamRegistry()
        assert hasattr(reg, "cancel_all_producers")
        reg.cancel_all_producers()  # Should not raise with no streams
