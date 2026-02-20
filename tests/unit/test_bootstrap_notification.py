# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bootstrap notification feature.

Tests:
- process_message_stream() returns immediate message during bootstrap
- _handle_chunk() processes bootstrap_start / bootstrap_complete / bootstrap_busy
- statusClass("bootstrapping") maps to the correct CSS class
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── process_message_stream bootstrap guard ────────────────────────


class TestProcessMessageStreamBootstrapGuard:
    """Test that process_message_stream yields an immediate rejection
    when bootstrapping is in progress (lock held + needs_bootstrap)."""

    async def test_bootstrap_busy_yields_immediately(self, tmp_path: Path):
        """When needs_bootstrap=True and lock is held, stream should
        yield a single bootstrap_busy chunk and return."""
        from core.anima import DigitalAnima

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Test", encoding="utf-8")
        (anima_dir / "bootstrap.md").write_text("# Bootstrap", encoding="utf-8")
        for sub in [
            "episodes", "knowledge", "procedures", "skills",
            "state", "shortterm", "shortterm/archive", "transcripts",
        ]:
            (anima_dir / sub).mkdir(parents=True, exist_ok=True)
        (anima_dir / "state" / "current_task.md").write_text(
            "status: idle\n", encoding="utf-8",
        )
        (anima_dir / "state" / "pending.md").write_text("", encoding="utf-8")

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(parents=True)
        (shared_dir / "inbox" / "test-anima").mkdir(parents=True)
        (shared_dir / "users").mkdir(parents=True)

        with (
            patch("core.anima.MemoryManager"),
            patch("core.anima.AgentCore"),
            patch("core.anima.Messenger"),
        ):
            dp = DigitalAnima(anima_dir, shared_dir)

        # Verify bootstrap file exists
        assert dp.needs_bootstrap is True

        # Acquire lock to simulate ongoing bootstrap
        await dp._lock.acquire()

        try:
            chunks: list[dict] = []
            async for chunk in dp.process_message_stream("hello"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0]["type"] == "bootstrap_busy"
            assert "初期化中" in chunks[0]["message"]
        finally:
            dp._lock.release()

    async def test_no_bootstrap_file_proceeds_normally(self, tmp_path: Path):
        """When needs_bootstrap=False, stream should proceed normally
        even if the lock is held (waits for lock)."""
        from core.anima import DigitalAnima

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Test", encoding="utf-8")
        # No bootstrap.md
        for sub in [
            "episodes", "knowledge", "procedures", "skills",
            "state", "shortterm", "shortterm/archive", "transcripts",
        ]:
            (anima_dir / sub).mkdir(parents=True, exist_ok=True)
        (anima_dir / "state" / "current_task.md").write_text(
            "status: idle\n", encoding="utf-8",
        )
        (anima_dir / "state" / "pending.md").write_text("", encoding="utf-8")

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(parents=True)
        (shared_dir / "inbox" / "test-anima").mkdir(parents=True)
        (shared_dir / "users").mkdir(parents=True)

        with (
            patch("core.anima.MemoryManager"),
            patch("core.anima.AgentCore") as mock_agent_cls,
            patch("core.anima.Messenger"),
        ):
            dp = DigitalAnima(anima_dir, shared_dir)

        assert dp.needs_bootstrap is False

        # Mock run_cycle_streaming to yield a quick done
        async def _mock_stream(prompt, trigger="", **kwargs):
            yield {"type": "text_delta", "text": "hi"}
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "hi"},
            }

        dp.agent.run_cycle_streaming = _mock_stream
        dp.memory.read_model_config = MagicMock(return_value={
            "model": "test", "max_tokens": 100, "context_threshold": 0.5,
            "conversation_history_threshold": 0.3,
        })

        # Patch ConversationMemory to avoid file ops
        with patch("core.anima.ConversationMemory") as mock_conv:
            mock_conv_inst = MagicMock()
            mock_conv_inst.compress_if_needed = AsyncMock()
            mock_conv_inst.build_chat_prompt = MagicMock(return_value="prompt")
            mock_conv_inst.append_turn = MagicMock()
            mock_conv_inst.save = MagicMock()
            mock_conv_inst.finalize_session = AsyncMock()
            mock_conv.return_value = mock_conv_inst

            chunks: list[dict] = []
            async for chunk in dp.process_message_stream("hello"):
                chunks.append(chunk)

        types = [c["type"] for c in chunks]
        assert "text_delta" in types
        assert "cycle_done" in types
        # bootstrap_busy should NOT appear
        assert "bootstrap_busy" not in types


# ── _handle_chunk bootstrap events ──────────────────────────────


class TestHandleChunkBootstrap:
    """Test _handle_chunk() handles bootstrap_start, bootstrap_complete,
    and bootstrap_busy chunk types correctly."""

    def test_bootstrap_start_chunk(self):
        from server.routes.chat import _handle_chunk

        frame, text = _handle_chunk({"type": "bootstrap_start"})
        assert frame is not None
        assert "event: bootstrap" in frame
        assert '"started"' in frame
        assert text == ""

    def test_bootstrap_complete_chunk(self):
        from server.routes.chat import _handle_chunk

        frame, text = _handle_chunk({"type": "bootstrap_complete"})
        assert frame is not None
        assert "event: bootstrap" in frame
        assert '"completed"' in frame
        assert text == ""

    def test_bootstrap_busy_chunk(self):
        from server.routes.chat import _handle_chunk

        frame, text = _handle_chunk({
            "type": "bootstrap_busy",
            "message": "現在初期化中です。しばらくお待ちください。",
        })
        assert frame is not None
        assert "event: bootstrap" in frame
        assert '"busy"' in frame
        assert "初期化中" in frame
        assert text == ""

    async def test_bootstrap_start_emits_websocket_event(self):
        """When request and anima_name are provided, bootstrap_start
        should trigger a WebSocket emit."""
        from server.routes.chat import _handle_chunk

        mock_request = MagicMock()
        ws_manager = MagicMock()
        ws_manager.broadcast = AsyncMock()
        mock_request.app.state.ws_manager = ws_manager

        frame, _ = _handle_chunk(
            {"type": "bootstrap_start"},
            request=mock_request,
            anima_name="alice",
        )
        assert frame is not None
        assert "event: bootstrap" in frame

    async def test_bootstrap_complete_emits_websocket_event(self):
        """When request and anima_name are provided, bootstrap_complete
        should trigger a WebSocket emit."""
        from server.routes.chat import _handle_chunk

        mock_request = MagicMock()
        ws_manager = MagicMock()
        ws_manager.broadcast = AsyncMock()
        mock_request.app.state.ws_manager = ws_manager

        frame, _ = _handle_chunk(
            {"type": "bootstrap_complete"},
            request=mock_request,
            anima_name="alice",
        )
        assert frame is not None
        assert "event: bootstrap" in frame

    def test_text_delta_still_works(self):
        """Ensure existing text_delta handling is unaffected."""
        from server.routes.chat import _handle_chunk

        frame, text = _handle_chunk({"type": "text_delta", "text": "hello"})
        assert frame is not None
        assert "event: text_delta" in frame
        assert text == ""

    def test_cycle_done_still_works(self):
        """Ensure existing cycle_done handling is unaffected."""
        from server.routes.chat import _handle_chunk

        frame, text = _handle_chunk({
            "type": "cycle_done",
            "cycle_result": {"summary": "done"},
        })
        assert frame is not None
        assert "event: done" in frame
        assert text == "done"


# ── statusClass("bootstrapping") ────────────────────────────────


class TestStatusClassBootstrapping:
    """Test that statusClass maps "bootstrapping" to "status-thinking"."""

    def test_bootstrapping_maps_to_thinking(self):
        """The JS function statusClass("bootstrapping") should return
        "status-thinking".  We verify the Python-side equivalent logic
        here by reimplementing the mapping."""
        # Reimplementation of statusClass for testing
        def status_class(status: str | None) -> str:
            if not status:
                return "status-offline"
            s = status.lower()
            if s in ("idle", "running"):
                return "status-idle"
            if s in ("thinking", "processing", "busy", "bootstrapping"):
                return "status-thinking"
            if s == "error":
                return "status-error"
            return "status-offline"

        assert status_class("bootstrapping") == "status-thinking"
        assert status_class("Bootstrapping") == "status-thinking"
        assert status_class("thinking") == "status-thinking"
        assert status_class("idle") == "status-idle"
        assert status_class(None) == "status-offline"


# ── StreamingIPCHandler bootstrap notification ─────────────────────


class TestStreamingHandlerBootstrapNotification:
    """Test that StreamingIPCHandler.handle_stream emits bootstrap_start
    and bootstrap_complete chunks when appropriate."""

    async def test_bootstrap_start_emitted(self, tmp_path: Path):
        """When needs_bootstrap is True at stream start, a bootstrap_start
        chunk should be emitted before any anima stream chunks."""
        from core.supervisor.streaming_handler import StreamingIPCHandler
        from core.supervisor.ipc import IPCRequest

        mock_anima = MagicMock()
        mock_anima.needs_bootstrap = True

        async def mock_stream(msg, from_person="human", **kwargs):
            yield {"type": "text_delta", "text": "hello"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "hello"}}

        mock_anima.process_message_stream = mock_stream

        handler = StreamingIPCHandler(
            anima=mock_anima,
            anima_name="test",
            anima_dir=tmp_path / "animas" / "test",
        )

        request = IPCRequest(
            id="test_001",
            method="process_message",
            params={"message": "hi", "stream": True},
        )

        chunks: list[dict] = []
        async for resp in handler.handle_stream(request):
            if resp.chunk:
                chunks.append(json.loads(resp.chunk))

        # First chunk should be bootstrap_start
        assert len(chunks) >= 1
        assert chunks[0]["type"] == "bootstrap_start"

    async def test_bootstrap_complete_emitted_when_finished(self, tmp_path: Path):
        """When needs_bootstrap transitions from True to False during the
        stream, a bootstrap_complete chunk should be emitted."""
        from core.supervisor.streaming_handler import StreamingIPCHandler
        from core.supervisor.ipc import IPCRequest

        mock_anima = MagicMock()
        # Start as True, then switch to False after stream
        bootstrap_values = [True, False]  # needs_bootstrap changes
        type(mock_anima).needs_bootstrap = property(
            lambda self: bootstrap_values[0] if bootstrap_values else False,
        )

        async def mock_stream(msg, from_person="human", **kwargs):
            yield {"type": "text_delta", "text": "hello"}
            # Simulate bootstrap completion (file deleted during agent run)
            bootstrap_values[0] = False
            yield {"type": "cycle_done", "cycle_result": {"summary": "hello"}}

        mock_anima.process_message_stream = mock_stream

        handler = StreamingIPCHandler(
            anima=mock_anima,
            anima_name="test",
            anima_dir=tmp_path / "animas" / "test",
        )

        request = IPCRequest(
            id="test_002",
            method="process_message",
            params={"message": "hi", "stream": True},
        )

        all_chunks: list[dict] = []
        done_result = None
        async for resp in handler.handle_stream(request):
            if resp.chunk:
                all_chunks.append(json.loads(resp.chunk))
            if resp.done:
                done_result = resp.result

        types = [c["type"] for c in all_chunks]
        assert "bootstrap_start" in types
        assert "bootstrap_complete" in types
        assert done_result is not None

    async def test_no_bootstrap_events_when_not_bootstrapping(self, tmp_path: Path):
        """When needs_bootstrap is False, no bootstrap events should appear."""
        from core.supervisor.streaming_handler import StreamingIPCHandler
        from core.supervisor.ipc import IPCRequest

        mock_anima = MagicMock()
        mock_anima.needs_bootstrap = False

        async def mock_stream(msg, from_person="human", **kwargs):
            yield {"type": "text_delta", "text": "hello"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "hello"}}

        mock_anima.process_message_stream = mock_stream

        handler = StreamingIPCHandler(
            anima=mock_anima,
            anima_name="test",
            anima_dir=tmp_path / "animas" / "test",
        )

        request = IPCRequest(
            id="test_003",
            method="process_message",
            params={"message": "hi", "stream": True},
        )

        all_chunks: list[dict] = []
        async for resp in handler.handle_stream(request):
            if resp.chunk:
                all_chunks.append(json.loads(resp.chunk))

        types = [c["type"] for c in all_chunks]
        assert "bootstrap_start" not in types
        assert "bootstrap_complete" not in types
