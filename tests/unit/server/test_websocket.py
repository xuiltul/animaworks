"""Unit tests for server/websocket.py — WebSocketManager."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.websocket import WebSocketManager


class TestWebSocketManager:
    """Tests for WebSocketManager."""

    def test_init_empty(self):
        mgr = WebSocketManager()
        assert mgr.active_connections == []

    async def test_connect(self):
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        ws.accept.assert_awaited_once()
        assert ws in mgr.active_connections

    async def test_connect_multiple(self):
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect(ws1)
        await mgr.connect(ws2)

        assert len(mgr.active_connections) == 2

    def test_disconnect_existing(self):
        mgr = WebSocketManager()
        ws = MagicMock()
        mgr.active_connections.append(ws)

        mgr.disconnect(ws)
        assert ws not in mgr.active_connections

    def test_disconnect_nonexistent(self):
        mgr = WebSocketManager()
        ws = MagicMock()

        # Should not raise
        mgr.disconnect(ws)
        assert len(mgr.active_connections) == 0

    async def test_broadcast_no_connections(self):
        mgr = WebSocketManager()
        # Should not raise
        await mgr.broadcast({"type": "test"})

    async def test_broadcast_sends_to_all(self):
        mgr = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        mgr.active_connections = [ws1, ws2]

        await mgr.broadcast({"type": "test", "data": "hello"})

        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()

        # Verify JSON content
        sent = ws1.send_text.call_args[0][0]
        import json
        data = json.loads(sent)
        assert data["type"] == "test"
        assert data["data"] == "hello"

    async def test_broadcast_removes_disconnected(self):
        mgr = WebSocketManager()
        ws_ok = AsyncMock()
        ws_broken = AsyncMock()
        ws_broken.send_text.side_effect = Exception("connection lost")

        mgr.active_connections = [ws_ok, ws_broken]

        await mgr.broadcast({"type": "test"})

        # Broken connection should be removed
        assert ws_broken not in mgr.active_connections
        assert ws_ok in mgr.active_connections

    async def test_broadcast_ensures_ascii_false(self):
        mgr = WebSocketManager()
        ws = AsyncMock()
        mgr.active_connections = [ws]

        await mgr.broadcast({"message": "日本語テスト"})

        sent = ws.send_text.call_args[0][0]
        assert "日本語テスト" in sent

    async def test_broadcast_uses_default_str(self):
        """Non-serializable values should use str() as default."""
        from pathlib import Path

        mgr = WebSocketManager()
        ws = AsyncMock()
        mgr.active_connections = [ws]

        await mgr.broadcast({"path": Path("/tmp/test")})

        ws.send_text.assert_awaited_once()


class TestWebSocketManagerHeartbeat:
    """Tests for WebSocketManager heartbeat features."""

    async def test_connect_sets_last_pong(self):
        """After connect(), _last_pong[id(ws)] should be set to approximately current time."""
        mgr = WebSocketManager()
        ws = AsyncMock()

        before = time.time()
        await mgr.connect(ws)
        after = time.time()

        ws_id = id(ws)
        assert ws_id in mgr._last_pong
        assert before <= mgr._last_pong[ws_id] <= after

    async def test_disconnect_cleans_last_pong(self):
        """After disconnect(), _last_pong should not contain the ws id."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        await mgr.connect(ws)

        ws_id = id(ws)
        assert ws_id in mgr._last_pong

        mgr.disconnect(ws)
        assert ws_id not in mgr._last_pong

    async def test_handle_client_message_pong(self):
        """Sending {"type": "pong"} should update _last_pong."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws_id = id(ws)
        mgr._last_pong[ws_id] = 1000.0  # old timestamp

        await mgr.handle_client_message(ws, json.dumps({"type": "pong"}))

        assert mgr._last_pong[ws_id] > 1000.0
        assert mgr._last_pong[ws_id] == pytest.approx(time.time(), abs=2.0)

    async def test_handle_client_message_non_json(self):
        """Non-JSON data should not raise."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws_id = id(ws)
        mgr._last_pong[ws_id] = 1000.0

        # Should not raise
        await mgr.handle_client_message(ws, "this is not json")

        # _last_pong should remain unchanged
        assert mgr._last_pong[ws_id] == 1000.0

    async def test_handle_client_message_non_pong(self):
        """JSON with type != "pong" should not update _last_pong."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        ws_id = id(ws)
        mgr._last_pong[ws_id] = 1000.0

        await mgr.handle_client_message(ws, json.dumps({"type": "hello"}))

        assert mgr._last_pong[ws_id] == 1000.0

    async def test_start_heartbeat_creates_task(self):
        """After start_heartbeat(), _heartbeat_task should not be None."""
        mgr = WebSocketManager()
        assert mgr._heartbeat_task is None

        await mgr.start_heartbeat()

        assert mgr._heartbeat_task is not None
        assert isinstance(mgr._heartbeat_task, asyncio.Task)

        # Cleanup
        await mgr.stop_heartbeat()

    async def test_stop_heartbeat_cancels_task(self):
        """After stop_heartbeat(), _heartbeat_task should be None."""
        mgr = WebSocketManager()
        await mgr.start_heartbeat()
        assert mgr._heartbeat_task is not None

        await mgr.stop_heartbeat()

        assert mgr._heartbeat_task is None

    async def test_heartbeat_loop_sends_ping(self):
        """After one heartbeat interval, all connections should receive a ping message."""
        mgr = WebSocketManager()
        ws = AsyncMock()
        mgr.active_connections = [ws]
        mgr._last_pong[id(ws)] = time.time()

        # Patch sleep to run once then cancel
        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError()

        with patch("server.websocket.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await mgr._heartbeat_loop()

        ws.send_text.assert_awaited_once()
        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["type"] == "ping"
        assert "ts" in sent

    async def test_heartbeat_loop_removes_stale(self):
        """Connections with expired _last_pong should be disconnected."""
        mgr = WebSocketManager()
        ws_stale = AsyncMock()
        ws_fresh = AsyncMock()
        mgr.active_connections = [ws_stale, ws_fresh]
        # Stale: last pong was 120 seconds ago (exceeds _HEARTBEAT_TIMEOUT of 60s)
        mgr._last_pong[id(ws_stale)] = time.time() - 120
        mgr._last_pong[id(ws_fresh)] = time.time()

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError()

        with patch("server.websocket.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await mgr._heartbeat_loop()

        # Stale connection should be removed
        assert ws_stale not in mgr.active_connections
        assert id(ws_stale) not in mgr._last_pong

        # Fresh connection should still be present and received a ping
        assert ws_fresh in mgr.active_connections
        ws_fresh.send_text.assert_awaited_once()

    async def test_heartbeat_loop_handles_send_failure(self):
        """If send_text raises during ping, the connection should be removed."""
        mgr = WebSocketManager()
        ws_broken = AsyncMock()
        ws_broken.send_text.side_effect = Exception("connection reset")
        mgr.active_connections = [ws_broken]
        mgr._last_pong[id(ws_broken)] = time.time()

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError()

        with patch("server.websocket.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await mgr._heartbeat_loop()

        # Broken connection should be removed
        assert ws_broken not in mgr.active_connections
        assert id(ws_broken) not in mgr._last_pong
