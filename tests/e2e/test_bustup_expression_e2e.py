# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the bustup expression system.

Tests the complete flow from chat SSE streaming through emotion extraction
to the done event containing the emotion field.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry


# ── Test Helpers ─────────────────────────────────────────────


def _make_test_app_with_supervisor():
    """Create a test FastAPI app with a mock supervisor for IPC streaming."""
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


def _make_ipc_response(*, done: bool = False, result: dict | None = None, chunk: str | None = None):
    """Create a mock IPC response object."""
    resp = MagicMock()
    resp.done = done
    resp.result = result
    resp.chunk = chunk
    return resp


# ── E2E Tests ────────────────────────────────────────────────


class TestBustupExpressionE2E:
    """E2E tests: chat message -> LLM response with emotion -> SSE stream -> emotion in done event."""

    async def test_emotion_extracted_from_stream_response(self):
        """Test that emotion metadata is extracted from LLM response and included in SSE done event."""
        app = _make_test_app_with_supervisor()

        async def mock_stream(*args, **kwargs):
            # Text delta chunks
            yield _make_ipc_response(
                chunk=json.dumps({"type": "text_delta", "text": "I'm glad "})
            )
            yield _make_ipc_response(
                chunk=json.dumps({"type": "text_delta", "text": "to help!"})
            )
            # Done with emotion metadata in the response
            yield _make_ipc_response(
                done=True,
                result={
                    "response": 'I\'m glad to help!\n<!-- emotion: {"emotion": "smile"} -->',
                    "cycle_result": {
                        "summary": 'I\'m glad to help!\n<!-- emotion: {"emotion": "smile"} -->',
                    },
                },
            )

        app.state.supervisor.processes = {"alice": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Can you help me?", "from_person": "tester"},
            )

        assert resp.status_code == 200
        body = resp.text

        # Verify text_delta events exist
        assert "event: text_delta" in body

        # Verify done event contains emotion
        assert "event: done" in body

        # Parse the done event data
        for line in body.split("\n"):
            if line.startswith("data: ") and "emotion" in line:
                data = json.loads(line[6:])
                assert data["emotion"] == "smile"
                # Verify metadata was stripped from summary
                assert "<!-- emotion" not in data.get("summary", "")
                break
        else:
            pytest.fail("No done event with emotion found in SSE stream")

    async def test_neutral_emotion_when_no_metadata(self):
        """Test that neutral emotion is returned when LLM does not include metadata."""
        app = _make_test_app_with_supervisor()

        async def mock_stream(*args, **kwargs):
            yield _make_ipc_response(
                done=True,
                result={
                    "response": "Here is your answer.",
                    "cycle_result": {"summary": "Here is your answer."},
                },
            )

        app.state.supervisor.processes = {"alice": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "What is 2+2?"},
            )

        body = resp.text
        for line in body.split("\n"):
            if line.startswith("data: ") and "emotion" in line:
                data = json.loads(line[6:])
                assert data["emotion"] == "neutral"
                break
        else:
            pytest.fail("No done event with emotion found")

    async def test_troubled_emotion_in_stream(self):
        """Test that 'troubled' emotion is correctly passed through."""
        app = _make_test_app_with_supervisor()

        async def mock_stream(*args, **kwargs):
            yield _make_ipc_response(
                done=True,
                result={
                    "response": 'That is a difficult problem.\n<!-- emotion: {"emotion": "troubled"} -->',
                    "cycle_result": {
                        "summary": 'That is a difficult problem.\n<!-- emotion: {"emotion": "troubled"} -->',
                    },
                },
            )

        app.state.supervisor.processes = {"alice": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "This is hard"},
            )

        body = resp.text
        for line in body.split("\n"):
            if line.startswith("data: ") and "emotion" in line:
                data = json.loads(line[6:])
                assert data["emotion"] == "troubled"
                assert "<!-- emotion" not in data.get("summary", "")
                break

    async def test_invalid_emotion_falls_back_to_neutral(self):
        """Test that invalid emotion names fall back to neutral."""
        app = _make_test_app_with_supervisor()

        async def mock_stream(*args, **kwargs):
            yield _make_ipc_response(
                done=True,
                result={
                    "response": 'Response\n<!-- emotion: {"emotion": "angry"} -->',
                    "cycle_result": {
                        "summary": 'Response\n<!-- emotion: {"emotion": "angry"} -->',
                    },
                },
            )

        app.state.supervisor.processes = {"alice": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Test"},
            )

        body = resp.text
        for line in body.split("\n"):
            if line.startswith("data: ") and "emotion" in line:
                data = json.loads(line[6:])
                assert data["emotion"] == "neutral"
                break

    async def test_no_chat_response_broadcast_on_stream(self):
        """Test that chat.response WebSocket broadcast is NOT emitted during streaming."""
        app = _make_test_app_with_supervisor()

        async def mock_stream(*args, **kwargs):
            yield _make_ipc_response(
                done=True,
                result={
                    "response": 'Hello!\n<!-- emotion: {"emotion": "smile"} -->',
                    "cycle_result": {
                        "summary": 'Hello!\n<!-- emotion: {"emotion": "smile"} -->',
                    },
                },
            )

        app.state.supervisor.processes = {"alice": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )

        # Verify no chat.response broadcast was sent
        ws = app.state.ws_manager
        for call in ws.broadcast.call_args_list:
            payload = call[0][0] if call[0] else call[1].get("message", {})
            if isinstance(payload, dict):
                assert payload.get("type") != "chat.response", \
                    "chat.response should not be broadcast during streaming"

    @pytest.mark.parametrize("emotion", [
        "smile", "laugh", "troubled", "surprised", "thinking", "embarrassed",
    ])
    async def test_all_valid_emotions_pass_through(self, emotion: str):
        """Test that all valid emotion types are correctly extracted and forwarded."""
        app = _make_test_app_with_supervisor()

        async def mock_stream(*args, **kwargs):
            yield _make_ipc_response(
                done=True,
                result={
                    "response": f'Response\n<!-- emotion: {{"emotion": "{emotion}"}} -->',
                    "cycle_result": {
                        "summary": f'Response\n<!-- emotion: {{"emotion": "{emotion}"}} -->',
                    },
                },
            )

        app.state.supervisor.processes = {"alice": True}
        app.state.supervisor.send_request_stream = mock_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Test"},
            )

        body = resp.text
        for line in body.split("\n"):
            if line.startswith("data: ") and "emotion" in line:
                data = json.loads(line[6:])
                assert data["emotion"] == emotion
                break
        else:
            pytest.fail(f"No done event with emotion '{emotion}' found")
