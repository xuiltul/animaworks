# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E test for bootstrap notification feature.

Tests the full API response when sending a message to an Anima
that is currently bootstrapping.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry


def _make_test_app_with_bootstrap():
    """Build a test FastAPI app with a mock bootstrapping anima."""
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router
    from core.supervisor.ipc import IPCResponse

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()

    supervisor = MagicMock()
    supervisor.processes = {"alice"}
    # Anima is NOT currently bootstrapping (so the guard doesn't block the stream)
    supervisor.is_bootstrapping = MagicMock(return_value=False)

    async def _send_request_stream(
        anima_name, method, params, timeout=120.0,
    ):
        """Simulate IPC stream from a bootstrapping anima."""
        if anima_name == "alice":
            # Emit bootstrap_start
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps(
                    {"type": "bootstrap_start"}, ensure_ascii=False,
                ),
            )
            # Emit text delta
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps(
                    {"type": "text_delta", "text": "Hello, I am Alice!"},
                    ensure_ascii=False,
                ),
            )
            # Emit bootstrap_complete
            yield IPCResponse(
                id="test",
                stream=True,
                chunk=json.dumps(
                    {"type": "bootstrap_complete"}, ensure_ascii=False,
                ),
            )
            # Final done
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={
                    "response": "Hello, I am Alice!",
                    "replied_to": [],
                    "cycle_result": {"summary": "Hello, I am Alice!"},
                },
            )

    supervisor.send_request_stream = _send_request_stream
    app.state.supervisor = supervisor

    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


def _make_test_app_bootstrap_busy():
    """Build a test FastAPI app where the anima is busy bootstrapping.

    The chat route checks supervisor.is_bootstrapping() before streaming.
    When True, it returns a "busy" bootstrap SSE event immediately without
    consulting the IPC stream.
    """
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()

    supervisor = MagicMock()
    supervisor.processes = {"alice"}
    # Anima IS currently bootstrapping — triggers the guard path
    supervisor.is_bootstrapping = MagicMock(return_value=True)

    app.state.supervisor = supervisor

    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


# ── Tests ──────────────────────────────────────────────────────


class TestBootstrapStreamE2E:
    """Test bootstrap events in the SSE stream."""

    async def test_bootstrap_events_in_stream(self):
        """When an anima is bootstrapping, the SSE stream should include
        bootstrap start and complete events."""
        app = _make_test_app_with_bootstrap()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hello"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        assert "event: bootstrap" in body
        assert '"started"' in body
        assert '"completed"' in body
        assert "event: text_delta" in body
        assert "event: done" in body

    async def test_bootstrap_busy_response(self):
        """When an anima is already bootstrapping and the lock is held,
        the SSE stream should include a bootstrap busy event."""
        app = _make_test_app_bootstrap_busy()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hello"},
            )

        assert resp.status_code == 200
        body = resp.text
        assert "event: bootstrap" in body
        assert '"busy"' in body
        # The route guard returns this specific Japanese message
        assert "作成中" in body or "初期化中" in body

    async def test_websocket_bootstrap_broadcast(self):
        """WebSocket broadcast should be called with anima.bootstrap event.

        The _handle_chunk function uses asyncio.ensure_future to emit
        broadcast events, so we must allow the event loop to process
        those fire-and-forget coroutines before inspecting the mock.
        """
        app = _make_test_app_with_bootstrap()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hello"},
            )

        # Allow fire-and-forget emit() tasks to complete
        await asyncio.sleep(0.05)

        ws = app.state.ws_manager
        # Should have broadcast calls — at least anima.status + anima.bootstrap
        assert ws.broadcast.await_count >= 2

        # Inspect broadcast calls for anima.bootstrap events
        broadcast_calls = [
            call.args[0] for call in ws.broadcast.call_args_list
        ]
        bootstrap_events = [
            c for c in broadcast_calls
            if c.get("type") == "anima.bootstrap"
        ]
        assert len(bootstrap_events) >= 1
        statuses = {e["data"]["status"] for e in bootstrap_events}
        assert "started" in statuses or "completed" in statuses
