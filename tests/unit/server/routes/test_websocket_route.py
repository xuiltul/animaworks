"""Unit tests for server/routes/websocket_route.py — WebSocket route."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from server.routes.websocket_route import create_websocket_router
from server.websocket import WebSocketManager


class TestWebSocketRoute:
    def test_websocket_connect_and_disconnect(self):
        app = FastAPI()
        ws_manager = WebSocketManager()
        app.state.ws_manager = ws_manager
        router = create_websocket_router()
        app.include_router(router)

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            assert len(ws_manager.active_connections) == 1

        # After disconnect
        assert len(ws_manager.active_connections) == 0

    def test_websocket_receives_messages(self):
        app = FastAPI()
        ws_manager = WebSocketManager()
        app.state.ws_manager = ws_manager
        router = create_websocket_router()
        app.include_router(router)

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # Send a message (the route just receives and discards)
            ws.send_text("ping")


class TestWebSocketRouteExceptionHandling:
    """Tests for WebSocket endpoint exception handling."""

    async def test_normal_disconnect_calls_disconnect(self):
        """When receive_text() raises WebSocketDisconnect, disconnect() should be called."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect(code=1000))

        ws_manager = MagicMock()
        ws_manager.connect = AsyncMock()
        ws_manager.handle_client_message = AsyncMock()
        ws_manager.disconnect = MagicMock()

        # Build the app with the real router
        app = FastAPI()
        app.state.ws_manager = ws_manager
        router = create_websocket_router()
        app.include_router(router)

        # Get the endpoint function from the router
        endpoint = router.routes[0].endpoint

        # Mock the app attribute on ws so ws.app.state.ws_manager works
        ws.app = app

        await endpoint(ws)

        ws_manager.connect.assert_awaited_once_with(ws)
        ws_manager.disconnect.assert_called_once_with(ws)

    async def test_unexpected_exception_calls_disconnect(self):
        """When receive_text() raises RuntimeError, disconnect() should still be called."""
        ws = AsyncMock()
        ws.receive_text = AsyncMock(side_effect=RuntimeError("unexpected failure"))

        ws_manager = MagicMock()
        ws_manager.connect = AsyncMock()
        ws_manager.handle_client_message = AsyncMock()
        ws_manager.disconnect = MagicMock()

        app = FastAPI()
        app.state.ws_manager = ws_manager
        router = create_websocket_router()
        app.include_router(router)

        endpoint = router.routes[0].endpoint
        ws.app = app

        await endpoint(ws)

        ws_manager.connect.assert_awaited_once_with(ws)
        ws_manager.disconnect.assert_called_once_with(ws)

    async def test_client_message_forwarded_to_handler(self):
        """When receive_text() returns data, handle_client_message() should be called."""
        ws = AsyncMock()
        # First call returns data, second call raises disconnect
        ws.receive_text = AsyncMock(
            side_effect=['{"type": "pong"}', WebSocketDisconnect(code=1000)]
        )

        ws_manager = MagicMock()
        ws_manager.connect = AsyncMock()
        ws_manager.handle_client_message = AsyncMock()
        ws_manager.disconnect = MagicMock()

        app = FastAPI()
        app.state.ws_manager = ws_manager
        router = create_websocket_router()
        app.include_router(router)

        endpoint = router.routes[0].endpoint
        ws.app = app

        await endpoint(ws)

        ws_manager.handle_client_message.assert_awaited_once_with(ws, '{"type": "pong"}')
        ws_manager.disconnect.assert_called_once_with(ws)
