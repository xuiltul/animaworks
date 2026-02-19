"""Unit tests for server/routes/chat.py — Chat API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry


def _make_test_app(animas: dict | None = None, supervisor: MagicMock | None = None):
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    animas = animas or {}
    app.state.animas = animas
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()
    if supervisor is not None:
        supervisor.is_bootstrapping = MagicMock(return_value=False)
        app.state.supervisor = supervisor
    else:
        # Build a supervisor mock that delegates to anima mocks
        sup = MagicMock()
        sup.processes = set(animas.keys())
        sup.is_bootstrapping = MagicMock(return_value=False)

        async def _send_request(anima_name, method, params, timeout=60.0):
            if anima_name not in animas:
                raise KeyError(anima_name)
            p = animas[anima_name]
            if method == "process_message":
                result = await p.process_message(
                    params.get("message", ""),
                    from_person=params.get("from_person", "human"),
                )
                return {"response": result, "replied_to": []}
            if method == "greet":
                return await p.process_greet()
            raise ValueError(f"Unknown method: {method}")

        async def _send_request_stream(anima_name, method, params, timeout=120.0):
            if anima_name not in animas:
                raise KeyError(anima_name)
            p = animas[anima_name]
            from core.supervisor.ipc import IPCResponse
            import json as _json
            async for chunk in p.process_message_stream(
                params.get("message", ""),
                from_person=params.get("from_person", "human"),
            ):
                event_type = chunk.get("type", "unknown")
                if event_type == "cycle_done":
                    cycle_result = chunk.get("cycle_result", {})
                    yield IPCResponse(
                        id="test",
                        stream=True,
                        done=True,
                        result={
                            "response": cycle_result.get("summary", ""),
                            "replied_to": [],
                            "cycle_result": cycle_result,
                        },
                    )
                    return
                yield IPCResponse(
                    id="test",
                    stream=True,
                    chunk=_json.dumps(chunk, ensure_ascii=False),
                )

        sup.send_request = _send_request
        sup.send_request_stream = _send_request_stream
        app.state.supervisor = sup
    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


def _make_mock_anima(name: str = "alice"):
    dp = MagicMock()
    dp.name = name
    dp.process_message = AsyncMock(return_value="Hello from Alice")
    return dp


# ── POST /animas/{name}/chat ────────────────────────────


class TestChat:
    async def test_chat_success(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello from Alice"
        assert data["anima"] == "alice"

    async def test_chat_with_from_anima(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi", "from_person": "bob"},
            )
        assert resp.status_code == 200
        alice.process_message.assert_awaited_once_with("Hi", from_person="bob")

    async def test_chat_default_from_anima(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )
        alice.process_message.assert_awaited_once_with("Hi", from_person="human")

    async def test_chat_anima_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    async def test_chat_broadcasts_status(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )

        ws = app.state.ws_manager
        # 2 broadcasts: thinking status + idle status
        assert ws.broadcast.await_count >= 2
        broadcast_types = [
            call[0][0]["type"] for call in ws.broadcast.call_args_list
        ]
        assert "anima.status" in broadcast_types
        assert "chat.response" not in broadcast_types

    async def test_chat_missing_message_field(self):
        app = _make_test_app({"alice": _make_mock_anima()})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={},
            )
        # Pydantic validation error
        assert resp.status_code == 422


# ── POST /animas/{name}/chat/stream ─────────────────────


class TestChatStream:
    async def test_stream_anima_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    async def test_stream_success(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human"):
            yield {"type": "text_delta", "text": "Hello"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Hello"}}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: text_delta" in body
        assert "event: done" in body

    async def test_stream_tool_events(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human"):
            yield {"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"}
            yield {"type": "tool_end", "tool_id": "t1", "tool_name": "web_search"}
            yield {"type": "chain_start", "chain": 2}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Done"}}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: tool_start" in body
        assert "event: tool_end" in body
        assert "event: chain_start" in body

    async def test_stream_error_event(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human"):
            yield {"type": "error", "message": "Something went wrong"}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: error" in body

    async def test_stream_exception_handling(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human"):
            raise RuntimeError("unexpected error")

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: error" in body
        assert "Internal server error" in body


# ── POST /animas/{name}/greet ──────────────────────────


class TestGreet:
    async def test_greet_success(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(return_value={
            "response": "こんにちは！待機中です。",
            "emotion": "smile",
            "cached": False,
        })
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/greet")

        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "こんにちは！待機中です。"
        assert data["emotion"] == "smile"
        assert data["cached"] is False
        assert data["anima"] == "alice"

    async def test_greet_cached(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(return_value={
            "response": "Hi!",
            "emotion": "neutral",
            "cached": True,
        })
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/greet")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is True

    async def test_greet_anima_not_found(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(side_effect=KeyError("bob"))
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/bob/greet")

        assert resp.status_code == 404

    async def test_greet_ipc_sends_correct_method(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(return_value={
            "response": "Hi", "emotion": "neutral", "cached": False,
        })
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/animas/alice/greet")

        supervisor.send_request.assert_awaited_once_with(
            anima_name="alice",
            method="greet",
            params={},
            timeout=60.0,
        )
