"""Unit tests for server/routes/chat.py — Chat API endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(persons: dict | None = None):
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.persons = persons or {}
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


def _make_mock_person(name: str = "alice"):
    person = MagicMock()
    person.name = name
    person.process_message = AsyncMock(return_value="Hello from Alice")
    return person


# ── POST /persons/{name}/chat ────────────────────────────


class TestChat:
    async def test_chat_success(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/alice/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello from Alice"
        assert data["person"] == "alice"

    async def test_chat_with_from_person(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/alice/chat",
                json={"message": "Hi", "from_person": "bob"},
            )
        assert resp.status_code == 200
        alice.process_message.assert_awaited_once_with("Hi", from_person="bob")

    async def test_chat_default_from_person(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/alice/chat",
                json={"message": "Hi"},
            )
        alice.process_message.assert_awaited_once_with("Hi", from_person="human")

    async def test_chat_person_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/nobody/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_chat_broadcasts_status(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/persons/alice/chat",
                json={"message": "Hi"},
            )

        ws = app.state.ws_manager
        # At least 3 broadcasts: thinking status, idle status, chat.response
        assert ws.broadcast.await_count >= 3

    async def test_chat_missing_message_field(self):
        app = _make_test_app({"alice": _make_mock_person()})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/alice/chat",
                json={},
            )
        # Pydantic validation error
        assert resp.status_code == 422


# ── POST /persons/{name}/chat/stream ─────────────────────


class TestChatStream:
    async def test_stream_person_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/nobody/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_stream_success(self):
        alice = _make_mock_person("alice")

        async def mock_stream(msg, from_person="human"):
            yield {"type": "text_delta", "text": "Hello"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Hello"}}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/alice/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: text_delta" in body
        assert "event: done" in body

    async def test_stream_tool_events(self):
        alice = _make_mock_person("alice")

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
                "/api/persons/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: tool_start" in body
        assert "event: tool_end" in body
        assert "event: chain_start" in body

    async def test_stream_error_event(self):
        alice = _make_mock_person("alice")

        async def mock_stream(msg, from_person="human"):
            yield {"type": "error", "message": "Something went wrong"}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: error" in body

    async def test_stream_exception_handling(self):
        alice = _make_mock_person("alice")

        async def mock_stream(msg, from_person="human"):
            raise RuntimeError("unexpected error")

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/persons/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: error" in body
        assert "Internal server error" in body
