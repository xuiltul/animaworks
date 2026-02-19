# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the notification broadcast pipeline.

Tests the full flow from process_message_stream yielding notification_sent
events through to WebSocket broadcast and queue lifecycle.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.websocket import WebSocketManager


# ── E2E: notification_sent event in stream ────────────────────


class TestNotificationSentEventInStream:
    """Verify that process_message_stream yields notification_sent events
    when the agent uses call_human during a cycle."""

    async def test_notification_sent_appears_in_stream(self, data_dir, make_anima):
        """When call_human is called during streaming, notification_sent events
        should appear in the process_message_stream output."""
        anima_dir = make_anima("stream-notif-test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.compress_if_needed = AsyncMock()
            MockConv.return_value.finalize_session = AsyncMock(return_value=False)
            MockConv.return_value.build_chat_prompt.return_value = "prompt"
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            # Simulate notification data that would be queued by ToolHandler
            pending_notifications = [
                {
                    "anima": "stream-notif-test",
                    "subject": "Task Complete",
                    "body": "The daily report has been generated",
                    "priority": "normal",
                    "timestamp": "2026-02-16T10:00:00",
                },
                {
                    "anima": "stream-notif-test",
                    "subject": "Error Alert",
                    "body": "Database connection timeout detected",
                    "priority": "high",
                    "timestamp": "2026-02-16T10:01:00",
                },
            ]

            # drain_notifications returns the queued notifications once,
            # then returns empty on subsequent calls
            dp.agent.drain_notifications = MagicMock(
                side_effect=[pending_notifications, []]
            )

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "text_delta", "text": "Processing your request..."}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "summary": "Processing your request...",
                        "trigger": "message:human",
                        "action": "responded",
                        "duration_ms": 500,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            # Collect all stream events
            events = []
            async for chunk in dp.process_message_stream("Check status", from_person="human"):
                events.append(chunk)

            # Extract event types
            event_types = [e.get("type") for e in events]

            # Should contain text_delta, notification_sent (x2), and cycle_done
            assert "text_delta" in event_types
            assert "cycle_done" in event_types
            assert event_types.count("notification_sent") == 2

            # Verify notification_sent events appear before cycle_done
            notif_indices = [i for i, e in enumerate(events) if e.get("type") == "notification_sent"]
            done_index = next(i for i, e in enumerate(events) if e.get("type") == "cycle_done")
            for idx in notif_indices:
                assert idx < done_index, "notification_sent must appear before cycle_done"

            # Verify notification data content
            notif_events = [e for e in events if e.get("type") == "notification_sent"]
            assert notif_events[0]["data"]["subject"] == "Task Complete"
            assert notif_events[1]["data"]["subject"] == "Error Alert"
            assert notif_events[1]["data"]["priority"] == "high"

    async def test_no_notification_events_when_none_queued(self, data_dir, make_anima):
        """When no notifications are queued, no notification_sent events appear."""
        anima_dir = make_anima("no-notif-test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.compress_if_needed = AsyncMock()
            MockConv.return_value.finalize_session = AsyncMock(return_value=False)
            MockConv.return_value.build_chat_prompt.return_value = "prompt"
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            # No notifications queued
            dp.agent.drain_notifications = MagicMock(return_value=[])

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "text_delta", "text": "Hello"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "Hello"},
                }

            dp.agent.run_cycle_streaming = mock_stream

            events = []
            async for chunk in dp.process_message_stream("Hi"):
                events.append(chunk)

            event_types = [e.get("type") for e in events]
            assert "notification_sent" not in event_types
            assert "cycle_done" in event_types


# ── E2E: WebSocket notification queue lifecycle ───────────────


class TestWebSocketNotificationQueueLifecycle:
    """Test the full queue lifecycle: queue when offline -> connect -> flush."""

    async def test_queue_then_connect_then_flush(self):
        """Notifications queued when no clients are connected are flushed
        when a new client connects."""
        manager = WebSocketManager()

        # Phase 1: Queue notifications while no clients connected
        assert len(manager.active_connections) == 0

        await manager.broadcast_notification({
            "anima": "alice",
            "subject": "Offline Alert 1",
            "body": "First notification while offline",
            "priority": "normal",
            "timestamp": "2026-02-16T10:00:00",
        })
        await manager.broadcast_notification({
            "anima": "alice",
            "subject": "Offline Alert 2",
            "body": "Second notification while offline",
            "priority": "high",
            "timestamp": "2026-02-16T10:05:00",
        })

        # Each broadcast_notification produces 2 events: proactive_message + notification
        assert len(manager._notification_queue) == 4

        # Phase 2: Client connects
        mock_ws = AsyncMock()
        await manager.connect(mock_ws)

        # Phase 3: Queued notifications should have been flushed
        assert len(manager._notification_queue) == 0
        assert mock_ws.accept.call_count == 1

        # Verify flushed content (2 events per broadcast_notification × 2 calls = 4)
        assert mock_ws.send_text.call_count == 4

        msg1 = json.loads(mock_ws.send_text.call_args_list[0][0][0])
        msg2 = json.loads(mock_ws.send_text.call_args_list[1][0][0])
        msg3 = json.loads(mock_ws.send_text.call_args_list[2][0][0])
        msg4 = json.loads(mock_ws.send_text.call_args_list[3][0][0])

        assert msg1["type"] == "anima.proactive_message"
        assert msg1["data"]["subject"] == "Offline Alert 1"

        assert msg2["type"] == "anima.notification"
        assert msg2["data"]["subject"] == "Offline Alert 1"

        assert msg3["type"] == "anima.proactive_message"
        assert msg3["data"]["subject"] == "Offline Alert 2"

        assert msg4["type"] == "anima.notification"
        assert msg4["data"]["subject"] == "Offline Alert 2"

    async def test_new_notifications_broadcast_after_connect(self):
        """After a client connects (and flush completes), new notifications
        are broadcast immediately rather than queued."""
        manager = WebSocketManager()

        # Queue one notification offline (produces 2 events)
        await manager.broadcast_notification({"subject": "offline"})
        assert len(manager._notification_queue) == 2

        # Connect a client
        mock_ws = AsyncMock()
        await manager.connect(mock_ws)

        # Queue should be flushed
        assert len(manager._notification_queue) == 0

        # Reset send_text tracking for clarity
        mock_ws.send_text.reset_mock()

        # New notification should broadcast immediately (2 events), not queue
        await manager.broadcast_notification({"subject": "online"})
        assert len(manager._notification_queue) == 0
        assert mock_ws.send_text.call_count == 2

        sent1 = json.loads(mock_ws.send_text.call_args_list[0][0][0])
        sent2 = json.loads(mock_ws.send_text.call_args_list[1][0][0])
        assert sent1["type"] == "anima.proactive_message"
        assert sent1["data"]["subject"] == "online"
        assert sent2["type"] == "anima.notification"
        assert sent2["data"]["subject"] == "online"

    async def test_queue_persists_across_disconnect_reconnect(self):
        """If a client disconnects and notifications arrive before
        reconnection, they are queued and flushed on the next connect."""
        manager = WebSocketManager()

        # Phase 1: Connect a client
        ws1 = AsyncMock()
        await manager.connect(ws1)
        assert len(manager.active_connections) == 1

        # Phase 2: Disconnect
        manager.disconnect(ws1)
        assert len(manager.active_connections) == 0

        # Phase 3: Notifications arrive while disconnected (2 events queued)
        await manager.broadcast_notification({"subject": "while-disconnected"})
        assert len(manager._notification_queue) == 2

        # Phase 4: New client connects
        ws2 = AsyncMock()
        await manager.connect(ws2)

        # Queued notifications should be flushed to ws2
        assert len(manager._notification_queue) == 0
        # ws2 receives 2 events (proactive_message + notification) from flush
        assert ws2.send_text.call_count == 2
        flushed1 = json.loads(ws2.send_text.call_args_list[0][0][0])
        flushed2 = json.loads(ws2.send_text.call_args_list[1][0][0])
        assert flushed1["type"] == "anima.proactive_message"
        assert flushed1["data"]["subject"] == "while-disconnected"
        assert flushed2["type"] == "anima.notification"
        assert flushed2["data"]["subject"] == "while-disconnected"

    async def test_full_pipeline_stream_to_websocket(self, data_dir, make_anima):
        """Full pipeline: agent call_human -> stream event -> WebSocket broadcast.

        This tests the integration between DigitalAnima stream events and
        the WebSocket manager, simulating what the server chat route does.
        """
        anima_dir = make_anima("pipeline-test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.compress_if_needed = AsyncMock()
            MockConv.return_value.finalize_session = AsyncMock(return_value=False)
            MockConv.return_value.build_chat_prompt.return_value = "prompt"
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            # Simulate one pending notification
            dp.agent.drain_notifications = MagicMock(
                return_value=[{
                    "anima": "pipeline-test",
                    "subject": "Pipeline Test",
                    "body": "Full pipeline verification",
                    "priority": "normal",
                    "timestamp": "2026-02-16T12:00:00",
                }]
            )

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "text_delta", "text": "Done"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "Done"},
                }

            dp.agent.run_cycle_streaming = mock_stream

            # Set up a WebSocket manager with a connected client
            ws_manager = WebSocketManager()
            mock_ws = AsyncMock()
            await ws_manager.connect(mock_ws)
            mock_ws.send_text.reset_mock()  # Clear the accept-related calls

            # Simulate the server-side stream processing (as chat.py does)
            async for chunk in dp.process_message_stream("test"):
                if chunk.get("type") == "notification_sent":
                    # This is what _handle_chunk + emit_notification does
                    await ws_manager.broadcast_notification(chunk["data"])

            # Verify both events were broadcast to the WebSocket client
            assert mock_ws.send_text.call_count == 2
            proactive_msg = json.loads(mock_ws.send_text.call_args_list[0][0][0])
            notif_msg = json.loads(mock_ws.send_text.call_args_list[1][0][0])
            assert proactive_msg["type"] == "anima.proactive_message"
            assert proactive_msg["data"]["subject"] == "Pipeline Test"
            assert proactive_msg["data"]["anima"] == "pipeline-test"
            assert notif_msg["type"] == "anima.notification"
            assert notif_msg["data"]["subject"] == "Pipeline Test"
