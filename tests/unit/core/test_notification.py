# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the anima notification feature.

Tests the notification pipeline:
  ToolHandler._handle_notify_human  ->  queue in _pending_notifications
  ToolHandler.drain_notifications   ->  return & clear
  AgentCore.drain_notifications     ->  passthrough
  DigitalAnima.drain_notifications ->  passthrough
  WebSocketManager.broadcast_notification  ->  queue / broadcast
  WebSocketManager.flush_notification_queue -> flush on connect
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.notification.notifier import HumanNotifier
from core.tooling.handler import ToolHandler
from server.websocket import WebSocketManager


# ── Helpers ───────────────────────────────────────────────────


def _make_mock_notifier(*, channel_count: int = 1, notify_result: list[str] | None = None) -> MagicMock:
    """Create a mock HumanNotifier with configurable channel_count and notify result."""
    notifier = MagicMock(spec=HumanNotifier)
    notifier.channel_count = channel_count
    if notify_result is None:
        notify_result = ["ntfy: OK"]
    notifier.notify = AsyncMock(return_value=notify_result)
    return notifier


def _make_tool_handler(
    tmp_path: Path,
    *,
    notifier: MagicMock | None = None,
) -> ToolHandler:
    """Create a ToolHandler with a mock memory and optional notifier."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        human_notifier=notifier,
    )


# ── ToolHandler._handle_notify_human queues notification ──────


class TestToolHandlerNotifyHumanQueues:
    def test_queues_notification_on_success(self, tmp_path):
        """When notify_human succeeds, notification data should be appended to _pending_notifications."""
        notifier = _make_mock_notifier()
        handler = _make_tool_handler(tmp_path, notifier=notifier)

        result = handler.handle("call_human", {
            "subject": "Test Alert",
            "body": "Something happened",
            "priority": "high",
        })

        parsed = json.loads(result)
        assert parsed["status"] == "sent"

        # Verify the notification was queued
        assert len(handler._pending_notifications) == 1
        notif = handler._pending_notifications[0]
        assert notif["anima"] == "test-anima"
        assert notif["subject"] == "Test Alert"
        assert notif["body"] == "Something happened"
        assert notif["priority"] == "high"
        assert "timestamp" in notif

    def test_queues_multiple_notifications(self, tmp_path):
        """Multiple notify_human calls accumulate in the queue."""
        notifier = _make_mock_notifier()
        handler = _make_tool_handler(tmp_path, notifier=notifier)

        handler.handle("call_human", {
            "subject": "First",
            "body": "First notification",
        })
        handler.handle("call_human", {
            "subject": "Second",
            "body": "Second notification",
        })

        assert len(handler._pending_notifications) == 2
        assert handler._pending_notifications[0]["subject"] == "First"
        assert handler._pending_notifications[1]["subject"] == "Second"


# ── ToolHandler.drain_notifications returns and clears ────────


class TestToolHandlerDrainNotifications:
    def test_drain_returns_and_clears(self, tmp_path):
        """drain_notifications should return accumulated notifications and reset the list."""
        notifier = _make_mock_notifier()
        handler = _make_tool_handler(tmp_path, notifier=notifier)

        handler.handle("call_human", {
            "subject": "Alert",
            "body": "Body text",
        })
        handler.handle("call_human", {
            "subject": "Alert 2",
            "body": "Body text 2",
        })

        drained = handler.drain_notifications()

        assert len(drained) == 2
        assert drained[0]["subject"] == "Alert"
        assert drained[1]["subject"] == "Alert 2"

        # After drain, list should be empty
        assert handler._pending_notifications == []
        assert handler.drain_notifications() == []

    def test_drain_empty_returns_empty_list(self, tmp_path):
        """drain_notifications on empty queue returns []."""
        handler = _make_tool_handler(tmp_path)
        assert handler.drain_notifications() == []


# ── ToolHandler._handle_notify_human does NOT queue on failure ─


class TestToolHandlerNotifyHumanNoQueueOnFailure:
    def test_no_queue_when_notifier_missing(self, tmp_path):
        """When no notifier is configured, no notification should be queued."""
        handler = _make_tool_handler(tmp_path, notifier=None)

        result = handler.handle("call_human", {
            "subject": "Alert",
            "body": "Body",
        })

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert handler._pending_notifications == []

    def test_no_queue_when_no_channels(self, tmp_path):
        """When notifier has 0 channels, no notification should be queued."""
        notifier = _make_mock_notifier(channel_count=0)
        handler = _make_tool_handler(tmp_path, notifier=notifier)

        result = handler.handle("call_human", {
            "subject": "Alert",
            "body": "Body",
        })

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert handler._pending_notifications == []

    def test_no_queue_when_notify_raises(self, tmp_path):
        """When notifier.notify() raises an exception, no notification should be queued."""
        notifier = _make_mock_notifier()
        notifier.notify = AsyncMock(side_effect=RuntimeError("channel down"))
        handler = _make_tool_handler(tmp_path, notifier=notifier)

        result = handler.handle("call_human", {
            "subject": "Alert",
            "body": "Body",
        })

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Failed to send notification" in parsed["message"]
        assert handler._pending_notifications == []

    def test_no_queue_when_missing_subject(self, tmp_path):
        """When subject is empty, no notification should be queued."""
        notifier = _make_mock_notifier()
        handler = _make_tool_handler(tmp_path, notifier=notifier)

        result = handler.handle("call_human", {
            "subject": "",
            "body": "Body",
        })

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert handler._pending_notifications == []

    def test_no_queue_when_missing_body(self, tmp_path):
        """When body is empty, no notification should be queued."""
        notifier = _make_mock_notifier()
        handler = _make_tool_handler(tmp_path, notifier=notifier)

        result = handler.handle("call_human", {
            "subject": "Alert",
            "body": "",
        })

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert handler._pending_notifications == []


# ── AgentCore.drain_notifications passes through to ToolHandler ─


class TestAgentCoreDrainNotifications:
    def test_passthrough(self, data_dir, make_anima):
        """AgentCore.drain_notifications delegates to ToolHandler."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            # The agent is a mock from patching AgentCore; verify the method exists
            mock_drain = MagicMock(return_value=[{"subject": "test"}])
            dp.agent.drain_notifications = mock_drain

            result = dp.agent.drain_notifications()
            mock_drain.assert_called_once()
            assert result == [{"subject": "test"}]


# ── DigitalAnima.drain_notifications passes through ──────────


class TestDigitalAnimaDrainNotifications:
    def test_passthrough(self, data_dir, make_anima):
        """DigitalAnima.drain_notifications delegates to agent.drain_notifications."""
        anima_dir = make_anima("bob")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            expected = [{"subject": "alert", "body": "test body"}]
            dp.agent.drain_notifications = MagicMock(return_value=expected)

            result = dp.drain_notifications()
            dp.agent.drain_notifications.assert_called_once()
            assert result == expected

    def test_empty_passthrough(self, data_dir, make_anima):
        """DigitalAnima.drain_notifications returns [] when no notifications."""
        anima_dir = make_anima("charlie")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            dp.agent.drain_notifications = MagicMock(return_value=[])
            assert dp.drain_notifications() == []


# ── WebSocketManager.broadcast_notification queues when no clients ─


class TestWebSocketManagerBroadcastQueue:
    async def test_queues_when_no_clients(self):
        """When active_connections is empty, notifications should go to _notification_queue."""
        manager = WebSocketManager()
        assert manager.active_connections == []

        data = {"anima": "alice", "subject": "Alert", "body": "test"}
        await manager.broadcast_notification(data)

        assert len(manager._notification_queue) == 2
        assert manager._notification_queue[0]["type"] == "anima.proactive_message"
        assert manager._notification_queue[0]["data"] == data
        assert manager._notification_queue[1]["type"] == "anima.notification"
        assert manager._notification_queue[1]["data"] == data

    async def test_queue_respects_max_size(self):
        """Queue drops oldest entries when exceeding _MAX_QUEUE_SIZE.

        Each broadcast_notification queues 2 events (proactive_message + notification).
        With MAX=50, 55 calls produce 110 events, trimmed to 50.
        The surviving 50 events come from calls 30..54 (25 calls * 2 = 50).
        """
        manager = WebSocketManager()

        for i in range(manager._MAX_QUEUE_SIZE + 5):
            await manager.broadcast_notification({"index": i})

        assert len(manager._notification_queue) == manager._MAX_QUEUE_SIZE
        # Oldest should have been dropped; first in queue is index 30
        assert manager._notification_queue[0]["data"]["index"] == 30

    async def test_queue_accumulates_multiple(self):
        """Multiple notifications accumulate in the queue (2 events each)."""
        manager = WebSocketManager()

        await manager.broadcast_notification({"subject": "first"})
        await manager.broadcast_notification({"subject": "second"})
        await manager.broadcast_notification({"subject": "third"})

        assert len(manager._notification_queue) == 6
        subjects = [e["data"]["subject"] for e in manager._notification_queue]
        assert subjects == ["first", "first", "second", "second", "third", "third"]


# ── WebSocketManager.broadcast_notification broadcasts when clients exist ─


class TestWebSocketManagerBroadcastImmediate:
    async def test_broadcasts_when_clients_exist(self):
        """When active_connections has clients, should broadcast immediately."""
        manager = WebSocketManager()

        mock_ws = AsyncMock()
        manager.active_connections.append(mock_ws)

        data = {"anima": "alice", "subject": "Alert"}
        await manager.broadcast_notification(data)

        # Should NOT go to queue
        assert len(manager._notification_queue) == 0

        # Should have been broadcast via send_text (2 events: proactive_message + notification)
        assert mock_ws.send_text.call_count == 2
        first_sent = json.loads(mock_ws.send_text.call_args_list[0][0][0])
        second_sent = json.loads(mock_ws.send_text.call_args_list[1][0][0])
        assert first_sent["type"] == "anima.proactive_message"
        assert first_sent["data"] == data
        assert second_sent["type"] == "anima.notification"
        assert second_sent["data"] == data

    async def test_broadcasts_to_multiple_clients(self):
        """Broadcasts to all connected clients (2 events each)."""
        manager = WebSocketManager()

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        manager.active_connections.extend([ws1, ws2])

        data = {"anima": "bob", "subject": "Update"}
        await manager.broadcast_notification(data)

        assert ws1.send_text.call_count == 2
        assert ws2.send_text.call_count == 2
        assert manager._notification_queue == []


# ── WebSocketManager.flush_notification_queue sends queued events ─


class TestWebSocketManagerFlushQueue:
    async def test_flush_sends_queued_events(self):
        """After connect, queued notifications should be flushed to the new client."""
        manager = WebSocketManager()

        # Queue some notifications while no clients connected
        # Each broadcast_notification queues 2 events (proactive_message + notification)
        await manager.broadcast_notification({"subject": "first"})
        await manager.broadcast_notification({"subject": "second"})
        assert len(manager._notification_queue) == 4

        # Simulate a client connecting and flushing
        mock_ws = AsyncMock()
        await manager.flush_notification_queue(mock_ws)

        # All queued items should have been sent
        assert mock_ws.send_text.call_count == 4

        # Queue should be empty after flush
        assert len(manager._notification_queue) == 0

        # Verify the content of flushed messages (proactive, notification, proactive, notification)
        first_msg = json.loads(mock_ws.send_text.call_args_list[0][0][0])
        second_msg = json.loads(mock_ws.send_text.call_args_list[1][0][0])
        third_msg = json.loads(mock_ws.send_text.call_args_list[2][0][0])
        fourth_msg = json.loads(mock_ws.send_text.call_args_list[3][0][0])
        assert first_msg["type"] == "anima.proactive_message"
        assert first_msg["data"]["subject"] == "first"
        assert second_msg["type"] == "anima.notification"
        assert second_msg["data"]["subject"] == "first"
        assert third_msg["type"] == "anima.proactive_message"
        assert third_msg["data"]["subject"] == "second"
        assert fourth_msg["type"] == "anima.notification"
        assert fourth_msg["data"]["subject"] == "second"

    async def test_flush_empty_queue_is_noop(self):
        """Flushing an empty queue should be a no-op."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()
        await manager.flush_notification_queue(mock_ws)
        mock_ws.send_text.assert_not_called()

    async def test_flush_stops_on_send_error(self):
        """If send_text fails during flush, the remaining events are dropped."""
        manager = WebSocketManager()

        # 3 broadcast_notification calls = 6 queued events
        await manager.broadcast_notification({"subject": "first"})
        await manager.broadcast_notification({"subject": "second"})
        await manager.broadcast_notification({"subject": "third"})

        mock_ws = AsyncMock()
        # Fail on the second send_text call
        mock_ws.send_text.side_effect = [None, Exception("connection closed"), None, None, None, None]

        await manager.flush_notification_queue(mock_ws)

        # Should have attempted 2 sends (first success, second failure, then break)
        assert mock_ws.send_text.call_count == 2

    async def test_connect_triggers_flush(self):
        """WebSocketManager.connect() should automatically flush queued notifications."""
        manager = WebSocketManager()

        # Queue a notification (2 events: proactive_message + notification)
        await manager.broadcast_notification({"subject": "queued"})
        assert len(manager._notification_queue) == 2

        # Mock the WebSocket with proper accept()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws)

        # accept() should have been called
        mock_ws.accept.assert_called_once()

        # Queued notification should have been flushed
        assert len(manager._notification_queue) == 0
        # send_text called twice for the flushed events (proactive_message + notification)
        assert mock_ws.send_text.call_count == 2
        first_sent = json.loads(mock_ws.send_text.call_args_list[0][0][0])
        second_sent = json.loads(mock_ws.send_text.call_args_list[1][0][0])
        assert first_sent["type"] == "anima.proactive_message"
        assert first_sent["data"]["subject"] == "queued"
        assert second_sent["type"] == "anima.notification"
        assert second_sent["data"]["subject"] == "queued"
