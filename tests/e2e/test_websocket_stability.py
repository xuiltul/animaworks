# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for WebSocket stability improvements.

Tests the actual WebSocket endpoint through a real FastAPI app using
Starlette's TestClient. External dependencies (supervisor, config, etc.)
are mocked, but the WebSocket endpoint, manager, and message handling
run against real code.

Covers:
  - Connection lifecycle (connect, disconnect, cleanup)
  - Pong message handling and _last_pong tracking
  - Multiple simultaneous client connections
  - Notification queue flush on connect
  - Notification queue cap enforcement
  - Broadcast delivery to connected clients
  - Disconnect cleanup of _last_pong tracking state
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from server.routes.websocket_route import create_websocket_router
from server.websocket import WebSocketManager


# ── Fixtures ────────────────────────────────────────────────────


def _mock_local_trust_auth():
    """Return a mock auth config with auth_mode='local_trust'."""
    auth = MagicMock()
    auth.auth_mode = "local_trust"
    return auth


@pytest.fixture(autouse=True)
def _bypass_ws_auth():
    """Bypass WebSocket auth for all tests by mocking load_auth to return local_trust."""
    with patch(
        "server.routes.websocket_route.load_auth",
        side_effect=_mock_local_trust_auth,
    ):
        yield


@pytest.fixture
def ws_app() -> tuple[FastAPI, WebSocketManager]:
    """Create a minimal FastAPI app with only the WebSocket endpoint.

    Returns a (app, ws_manager) tuple so tests can inspect manager state.
    """
    app = FastAPI()
    ws_manager = WebSocketManager()
    app.state.ws_manager = ws_manager
    app.include_router(create_websocket_router())
    return app, ws_manager


# ── Connection Lifecycle ────────────────────────────────────────


class TestWebSocketConnectionLifecycle:
    """Verify connection registration and teardown through the real endpoint."""

    def test_connect_registers_in_active_connections(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Opening a WebSocket connection adds it to active_connections."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as _ws:
                assert len(ws_manager.active_connections) == 1

    def test_disconnect_removes_from_active_connections(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Closing a WebSocket connection removes it from active_connections."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws"):
                assert len(ws_manager.active_connections) == 1
            # After the context manager exits, the connection is closed
            assert len(ws_manager.active_connections) == 0

    def test_disconnect_cleans_up_last_pong_entry(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Disconnecting should remove the connection's _last_pong entry."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws"):
                assert len(ws_manager._last_pong) == 1
            # After disconnect, the tracking entry should be removed
            assert len(ws_manager._last_pong) == 0

    def test_connect_initialises_last_pong_timestamp(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """New connections get an initial _last_pong timestamp on connect."""
        app, ws_manager = ws_app
        before = time.time()
        with TestClient(app) as client:
            with client.websocket_connect("/ws"):
                assert len(ws_manager._last_pong) == 1
                pong_ts = next(iter(ws_manager._last_pong.values()))
                assert pong_ts >= before


# ── Multiple Clients ────────────────────────────────────────────


class TestWebSocketMultipleClients:
    """Verify correct tracking when multiple clients connect simultaneously."""

    def test_multiple_clients_all_tracked(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Each concurrent WebSocket connection is tracked independently."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws"):
                with client.websocket_connect("/ws"):
                    assert len(ws_manager.active_connections) == 2
                    assert len(ws_manager._last_pong) == 2
                # Inner connection closed
                assert len(ws_manager.active_connections) == 1
                assert len(ws_manager._last_pong) == 1
            # Both closed
            assert len(ws_manager.active_connections) == 0
            assert len(ws_manager._last_pong) == 0

    def test_each_client_has_unique_pong_entry(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Each connection gets its own _last_pong key (id-based)."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws"):
                with client.websocket_connect("/ws"):
                    pong_keys = list(ws_manager._last_pong.keys())
                    assert len(pong_keys) == 2
                    assert pong_keys[0] != pong_keys[1]


# ── Client Pong Handling ────────────────────────────────────────


class TestWebSocketPongHandling:
    """Verify that pong messages from clients update tracking state."""

    def test_pong_message_updates_last_pong_timestamp(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Sending a pong message from the client updates _last_pong."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                conn = ws_manager.active_connections[0]
                initial_pong = ws_manager._last_pong[id(conn)]

                # Small delay so the updated timestamp is measurably later
                time.sleep(0.05)

                ws.send_json({"type": "pong"})
                # Give the server-side handler a moment to process
                time.sleep(0.05)

                updated_pong = ws_manager._last_pong[id(conn)]
                assert updated_pong > initial_pong

    def test_non_pong_message_does_not_update_timestamp(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Sending a non-pong message should not change _last_pong."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                conn = ws_manager.active_connections[0]
                initial_pong = ws_manager._last_pong[id(conn)]

                ws.send_json({"type": "some_other_message", "data": "hello"})
                time.sleep(0.05)

                assert ws_manager._last_pong[id(conn)] == initial_pong

    def test_malformed_json_does_not_crash_endpoint(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Sending invalid JSON should not crash the WebSocket endpoint."""
        app, ws_manager = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text("this is not json")
                time.sleep(0.05)
                # Connection should still be alive
                assert len(ws_manager.active_connections) == 1


# ── Notification Queue Flush ────────────────────────────────────


class TestWebSocketNotificationQueueFlush:
    """Verify that queued notifications are flushed to newly connected clients."""

    def test_queued_notifications_flushed_on_connect(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Notifications queued before any client connects are sent on connect.

        broadcast_notification queues two events per call:
        anima.proactive_message (for chat) and anima.notification (for toast).
        """
        app, ws_manager = ws_app

        # Queue notifications while no clients are connected
        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(ws_manager.broadcast_notification({
            "anima": "test-anima",
            "subject": "Queued Alert",
            "body": "This was queued while offline",
        }))
        loop.close()

        # 2 events queued: proactive_message + notification
        assert len(ws_manager._notification_queue) == 2

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # The queued events should be flushed immediately
                data1 = ws.receive_json()
                assert data1["type"] == "anima.proactive_message"
                assert data1["data"]["subject"] == "Queued Alert"

                data2 = ws.receive_json()
                assert data2["type"] == "anima.notification"
                assert data2["data"]["subject"] == "Queued Alert"
                assert data2["data"]["body"] == "This was queued while offline"

                # Queue should be empty after flush
                assert len(ws_manager._notification_queue) == 0

    def test_multiple_queued_notifications_flushed_in_order(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """Multiple queued notifications are flushed in FIFO order.

        Each broadcast_notification call queues 2 events (proactive_message
        + notification), so 3 calls produce 6 queued events.
        """
        app, ws_manager = ws_app

        import asyncio

        loop = asyncio.new_event_loop()
        for i in range(3):
            loop.run_until_complete(ws_manager.broadcast_notification({
                "anima": "test-anima",
                "subject": f"Alert {i}",
                "body": f"Notification {i}",
            }))
        loop.close()

        # 3 calls x 2 events each = 6 queued events
        assert len(ws_manager._notification_queue) == 6

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                for i in range(3):
                    # Each notification produces a proactive_message then a notification
                    proactive = ws.receive_json()
                    assert proactive["type"] == "anima.proactive_message"
                    assert proactive["data"]["subject"] == f"Alert {i}"

                    notif = ws.receive_json()
                    assert notif["type"] == "anima.notification"
                    assert notif["data"]["subject"] == f"Alert {i}"

                assert len(ws_manager._notification_queue) == 0

    def test_queue_empty_after_flush(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """After flushing, the notification queue is empty."""
        app, ws_manager = ws_app

        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(ws_manager.broadcast_notification({
            "anima": "test-anima",
            "subject": "Flush me",
            "body": "Should be drained",
        }))
        loop.close()

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                _ = ws.receive_json()
                assert ws_manager._notification_queue == []


# ── Notification Queue Cap ──────────────────────────────────────


class TestWebSocketNotificationQueueCap:
    """Verify _MAX_QUEUE_SIZE is enforced to prevent unbounded growth."""

    def test_queue_cap_drops_oldest_when_exceeded(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """When queue exceeds _MAX_QUEUE_SIZE, the oldest entry is dropped."""
        app, ws_manager = ws_app
        max_size = ws_manager._MAX_QUEUE_SIZE

        import asyncio

        loop = asyncio.new_event_loop()
        # Queue more than the maximum
        for i in range(max_size + 5):
            loop.run_until_complete(ws_manager.broadcast_notification({
                "anima": "test-anima",
                "subject": f"Notification {i}",
            }))
        loop.close()

        assert len(ws_manager._notification_queue) == max_size

        # Each broadcast_notification queues 2 events (proactive_message + notification).
        # First 25 calls fill the queue to 50. Each subsequent call adds 2 and trims 2,
        # so 30 more calls drop the 30 oldest pairs. First remaining is pair for #30.
        first_queued = ws_manager._notification_queue[0]
        assert first_queued["data"]["subject"] == "Notification 30"

        # The newest should be the last one enqueued
        last_queued = ws_manager._notification_queue[-1]
        assert last_queued["data"]["subject"] == f"Notification {max_size + 4}"


# ── Broadcast Delivery ──────────────────────────────────────────


class TestWebSocketBroadcastDelivery:
    """Verify broadcast messages reach connected clients through the endpoint."""

    def test_broadcast_notification_received_by_connected_client(
        self, ws_app: tuple[FastAPI, WebSocketManager],
    ) -> None:
        """A notification broadcast while a client is connected is received."""
        app, ws_manager = ws_app

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # Broadcast a notification from another thread since the
                # WebSocket endpoint's event loop is managed by TestClient
                import asyncio
                import threading

                def _broadcast() -> None:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        ws_manager.broadcast({"type": "test.event", "data": "hello"})
                    )
                    loop.close()

                t = threading.Thread(target=_broadcast)
                t.start()
                t.join(timeout=5.0)

                data = ws.receive_json()
                assert data["type"] == "test.event"
                assert data["data"] == "hello"
