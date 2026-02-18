# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for heartbeat SSE relay in core/anima.py.

Validates:
  - _heartbeat_stream_queue is created when lock is held during process_message_stream
  - heartbeat chunks are written to queue when it exists
  - heartbeat_relay_start / heartbeat_relay / heartbeat_relay_done events are yielded
  - Normal flow when lock is NOT held (no relay events)
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.messenger import InboxItem
from core.schemas import CycleResult


# ── Helpers ───────────────────────────────────────────────


def _make_anima(tmp_path: Path) -> Any:
    """Create a minimal DigitalAnima with mocked dependencies."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True)
    (shared_dir / "users").mkdir(parents=True)

    # Create minimal required files
    (anima_dir / "identity.md").write_text("Test identity", encoding="utf-8")
    (anima_dir / "injection.md").write_text("Test injection", encoding="utf-8")

    with (
        patch("core.anima.MemoryManager") as MockMemory,
        patch("core.anima.Messenger") as MockMessenger,
        patch("core.anima.AgentCore") as MockAgent,
    ):
        mock_memory = MockMemory.return_value
        mock_memory.read_model_config.return_value = MagicMock(
            model="claude-sonnet-4-20250514",
            context_threshold=0.8,
        )

        mock_agent = MockAgent.return_value
        mock_agent.background_manager = None
        mock_agent.drain_notifications.return_value = []

        from core.anima import DigitalAnima
        anima = DigitalAnima(anima_dir, shared_dir)

    return anima


async def _fake_streaming_chunks() -> AsyncGenerator[dict, None]:
    """Simulate agent.run_cycle_streaming output."""
    yield {"type": "text_delta", "text": "Hello "}
    yield {"type": "text_delta", "text": "world"}
    yield {
        "type": "cycle_done",
        "cycle_result": {
            "trigger": "message:human",
            "action": "responded",
            "summary": "Hello world",
            "duration_ms": 100,
            "context_usage_ratio": 0.1,
            "session_chained": False,
            "total_turns": 1,
        },
    }


# ── Tests ─────────────────────────────────────────────────


class TestHeartbeatRelayInit:
    """Test that heartbeat relay instance variables are initialized."""

    def test_queue_initially_none(self, tmp_path: Path) -> None:
        anima = _make_anima(tmp_path)
        assert anima._heartbeat_stream_queue is None

    def test_context_initially_empty(self, tmp_path: Path) -> None:
        anima = _make_anima(tmp_path)
        assert anima._heartbeat_context == ""


class TestHeartbeatRelayDuringStream:
    """Test heartbeat relay events when lock is held."""

    @pytest.mark.asyncio
    async def test_relay_events_when_lock_held(self, tmp_path: Path) -> None:
        """When lock is held, process_message_stream yields relay events."""
        anima = _make_anima(tmp_path)
        anima._heartbeat_context = "テスト処理中"

        # Pre-acquire the lock to simulate heartbeat holding it
        await anima._lock.acquire()

        # Set up a task that:
        # 1. Writes chunks to the queue once it's created
        # 2. Releases the lock after sending sentinel
        async def _simulate_heartbeat() -> None:
            # Wait for queue to be created
            for _ in range(50):
                if anima._heartbeat_stream_queue is not None:
                    break
                await asyncio.sleep(0.01)

            queue = anima._heartbeat_stream_queue
            assert queue is not None
            await queue.put({"text": "Heartbeat "})
            await queue.put({"text": "output"})
            await queue.put(None)  # sentinel
            # Small delay then release lock
            await asyncio.sleep(0.05)
            anima._lock.release()

        heartbeat_task = asyncio.create_task(_simulate_heartbeat())

        # Mock agent streaming for the actual message processing
        anima.agent.run_cycle_streaming = MagicMock(
            return_value=_fake_streaming_chunks()
        )

        # Mock ConversationMemory
        with patch("core.anima.ConversationMemory") as MockConv:
            mock_conv = MockConv.return_value
            mock_conv.compress_if_needed = AsyncMock()
            mock_conv.build_chat_prompt.return_value = "test prompt"
            mock_conv.append_turn = MagicMock()
            mock_conv.save = MagicMock()
            mock_conv.finalize_session = AsyncMock()

            collected: list[dict] = []
            async for chunk in anima.process_message_stream("hello", "human"):
                collected.append(chunk)

        await heartbeat_task

        # Verify relay events
        event_types = [c["type"] for c in collected]
        assert "heartbeat_relay_start" in event_types
        assert "heartbeat_relay" in event_types
        assert "heartbeat_relay_done" in event_types

        # Verify relay_start contains context message
        start_event = next(c for c in collected if c["type"] == "heartbeat_relay_start")
        assert "テスト処理中" in start_event["message"]

        # Verify relay chunks contain text
        relay_events = [c for c in collected if c["type"] == "heartbeat_relay"]
        relay_text = "".join(c["text"] for c in relay_events)
        assert "Heartbeat " in relay_text
        assert "output" in relay_text

        # Verify normal streaming events follow after relay
        assert "text_delta" in event_types
        assert "cycle_done" in event_types

        # Verify relay events come before normal events
        relay_done_idx = event_types.index("heartbeat_relay_done")
        text_delta_idx = event_types.index("text_delta")
        assert relay_done_idx < text_delta_idx

    @pytest.mark.asyncio
    async def test_no_relay_when_lock_free(self, tmp_path: Path) -> None:
        """When lock is NOT held, no relay events are yielded."""
        anima = _make_anima(tmp_path)

        # Lock is NOT held — normal flow
        anima.agent.run_cycle_streaming = MagicMock(
            return_value=_fake_streaming_chunks()
        )

        with patch("core.anima.ConversationMemory") as MockConv:
            mock_conv = MockConv.return_value
            mock_conv.compress_if_needed = AsyncMock()
            mock_conv.build_chat_prompt.return_value = "test prompt"
            mock_conv.append_turn = MagicMock()
            mock_conv.save = MagicMock()
            mock_conv.finalize_session = AsyncMock()

            collected: list[dict] = []
            async for chunk in anima.process_message_stream("hello", "human"):
                collected.append(chunk)

        event_types = [c["type"] for c in collected]

        # No relay events
        assert "heartbeat_relay_start" not in event_types
        assert "heartbeat_relay" not in event_types
        assert "heartbeat_relay_done" not in event_types

        # Normal events are present
        assert "text_delta" in event_types
        assert "cycle_done" in event_types

    @pytest.mark.asyncio
    async def test_queue_cleaned_up_after_relay(self, tmp_path: Path) -> None:
        """_heartbeat_stream_queue is set to None after relay completes."""
        anima = _make_anima(tmp_path)

        await anima._lock.acquire()

        async def _release_quickly() -> None:
            for _ in range(50):
                if anima._heartbeat_stream_queue is not None:
                    break
                await asyncio.sleep(0.01)
            queue = anima._heartbeat_stream_queue
            if queue:
                await queue.put(None)
            await asyncio.sleep(0.05)
            anima._lock.release()

        release_task = asyncio.create_task(_release_quickly())

        anima.agent.run_cycle_streaming = MagicMock(
            return_value=_fake_streaming_chunks()
        )

        with patch("core.anima.ConversationMemory") as MockConv:
            mock_conv = MockConv.return_value
            mock_conv.compress_if_needed = AsyncMock()
            mock_conv.build_chat_prompt.return_value = "test prompt"
            mock_conv.append_turn = MagicMock()
            mock_conv.save = MagicMock()
            mock_conv.finalize_session = AsyncMock()

            async for _ in anima.process_message_stream("hello", "human"):
                pass

        await release_task
        assert anima._heartbeat_stream_queue is None


class TestHeartbeatStreamingMode:
    """Test that run_heartbeat uses streaming and writes to queue."""

    @pytest.mark.asyncio
    async def test_heartbeat_writes_to_queue(self, tmp_path: Path) -> None:
        """When _heartbeat_stream_queue exists, heartbeat writes chunks to it."""
        anima = _make_anima(tmp_path)

        # Set up a queue before heartbeat runs
        queue: asyncio.Queue = asyncio.Queue()
        anima._heartbeat_stream_queue = queue

        async def _fake_hb_streaming(prompt: str, trigger: str = "heartbeat") -> AsyncGenerator[dict, None]:
            yield {"type": "text_delta", "text": "HB chunk"}
            yield {
                "type": "cycle_done",
                "cycle_result": {
                    "trigger": "heartbeat",
                    "action": "responded",
                    "summary": "HB chunk",
                    "duration_ms": 50,
                    "context_usage_ratio": 0.05,
                    "session_chained": False,
                    "total_turns": 1,
                },
            }

        anima.agent.run_cycle_streaming = MagicMock(
            return_value=_fake_hb_streaming("test", "heartbeat")
        )
        anima.agent.reset_reply_tracking = MagicMock()
        anima.agent.replied_to = set()
        anima.memory.read_heartbeat_config.return_value = None
        anima.messenger.has_unread.return_value = False

        with patch("core.anima.load_prompt", return_value="test prompt"):
            result = await anima.run_heartbeat()

        assert result.trigger == "heartbeat"

        # Collect everything from the queue
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

        # Should have the text_delta chunk + sentinel (None)
        assert len(items) == 2
        assert items[0]["text"] == "HB chunk"
        assert items[1] is None

    @pytest.mark.asyncio
    async def test_heartbeat_sets_context_with_senders(self, tmp_path: Path) -> None:
        """Heartbeat context includes sender names when processing messages."""
        anima = _make_anima(tmp_path)

        async def _fake_hb_streaming(prompt: str, trigger: str = "heartbeat") -> AsyncGenerator[dict, None]:
            yield {
                "type": "cycle_done",
                "cycle_result": {
                    "trigger": "heartbeat",
                    "action": "responded",
                    "summary": "done",
                    "duration_ms": 10,
                    "context_usage_ratio": 0.01,
                    "session_chained": False,
                    "total_turns": 1,
                },
            }

        anima.agent.run_cycle_streaming = MagicMock(
            return_value=_fake_hb_streaming("test", "heartbeat")
        )
        anima.agent.reset_reply_tracking = MagicMock()
        anima.agent.replied_to = set()
        anima.memory.read_heartbeat_config.return_value = None

        # Simulate unread messages
        mock_msg = MagicMock()
        mock_msg.from_person = "kotoha"
        mock_msg.content = "Hello!"
        anima.messenger.has_unread.return_value = True
        anima.messenger.receive_with_paths.return_value = [InboxItem(msg=mock_msg, path=Path("/fake/msg.json"))]
        anima.messenger.archive_paths.return_value = 1

        # Capture context during execution
        captured_context = None

        original_streaming = anima.agent.run_cycle_streaming

        def _capture_context(*args: Any, **kwargs: Any) -> Any:
            nonlocal captured_context
            captured_context = anima._heartbeat_context
            return original_streaming(*args, **kwargs)

        anima.agent.run_cycle_streaming = _capture_context

        with patch("core.anima.load_prompt", return_value="test prompt"):
            await anima.run_heartbeat()

        assert captured_context is not None
        assert "kotoha" in captured_context
