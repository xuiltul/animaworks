"""Unit tests for the interrupt mechanism (Phase 2 of webui-stop-button)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── BaseExecutor interrupt_event ─────────────────────────


class TestBaseExecutorInterrupt:
    """Test interrupt_event on BaseExecutor."""

    def test_check_interrupted_no_event(self):
        """_check_interrupted returns False when no event is set."""
        from core.execution.base import BaseExecutor

        class DummyExecutor(BaseExecutor):
            async def execute(self, *a, **kw):
                pass

        executor = DummyExecutor(
            model_config=MagicMock(),
            anima_dir=Path("/tmp/fake"),
        )
        assert executor._check_interrupted() is False

    def test_check_interrupted_event_not_set(self):
        """_check_interrupted returns False when event exists but not set."""
        from core.execution.base import BaseExecutor

        class DummyExecutor(BaseExecutor):
            async def execute(self, *a, **kw):
                pass

        event = asyncio.Event()
        executor = DummyExecutor(
            model_config=MagicMock(),
            anima_dir=Path("/tmp/fake"),
            interrupt_event=event,
        )
        assert executor._check_interrupted() is False

    def test_check_interrupted_event_set(self):
        """_check_interrupted returns True when event is set."""
        from core.execution.base import BaseExecutor

        class DummyExecutor(BaseExecutor):
            async def execute(self, *a, **kw):
                pass

        event = asyncio.Event()
        event.set()
        executor = DummyExecutor(
            model_config=MagicMock(),
            anima_dir=Path("/tmp/fake"),
            interrupt_event=event,
        )
        assert executor._check_interrupted() is True


# ── DigitalAnima interrupt ───────────────────────────────


class TestDigitalAnimaInterrupt:
    """Test DigitalAnima.interrupt() and _interrupt_events lifecycle."""

    @pytest.fixture
    def anima_dir(self, tmp_path):
        """Create minimal anima directory structure."""
        d = tmp_path / "test-anima"
        d.mkdir()
        (d / "IDENTITY.md").write_text("# Test Anima", encoding="utf-8")
        (d / "AGENTS.md").write_text("test injection", encoding="utf-8")
        (d / "status.json").write_text(
            '{"enabled": true, "model": "claude-sonnet-4-6"}',
            encoding="utf-8",
        )
        (d / "state").mkdir()
        (d / "episodes").mkdir()
        (d / "knowledge").mkdir()
        (d / "procedures").mkdir()
        (d / "skills").mkdir()
        (d / "shortterm").mkdir()
        (d / "shortterm" / "chat").mkdir()
        (d / "shortterm" / "heartbeat").mkdir()
        (d / "activity_log").mkdir()
        (d / "assets").mkdir()
        return d

    @pytest.fixture
    def shared_dir(self, tmp_path):
        d = tmp_path / "shared"
        d.mkdir()
        (d / "channels").mkdir()
        (d / "users").mkdir()
        (d / "dm_logs").mkdir()
        return d

    @patch("core.anima.digital_anima.AgentCore")
    @patch("core.anima.digital_anima.MemoryManager")
    @patch("core.anima.digital_anima.Messenger")
    async def test_interrupt_sets_event(self, mock_messenger_cls, mock_mm_cls, mock_agent_cls, anima_dir, shared_dir):
        """interrupt() should set all _interrupt_events."""
        mock_mm_cls.return_value.read_model_config.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()
        mock_agent_cls.return_value.set_on_message_sent = MagicMock()
        mock_agent_cls.return_value.set_on_schedule_changed = MagicMock()
        mock_agent_cls.return_value.set_interrupt_event = MagicMock()
        mock_agent_cls.return_value.background_manager = None
        mock_agent_cls.return_value._tool_handler = MagicMock()

        from core.anima.digital_anima import DigitalAnima

        anima = DigitalAnima(anima_dir, shared_dir)

        evt = anima._get_interrupt_event("default")
        assert not evt.is_set()
        result = await anima.interrupt()
        assert evt.is_set()
        assert result["status"] == "interrupted"

    @patch("core.anima.digital_anima.AgentCore")
    @patch("core.anima.digital_anima.MemoryManager")
    @patch("core.anima.digital_anima.Messenger")
    async def test_interrupt_event_cleared_on_process_message(
        self, mock_messenger_cls, mock_mm_cls, mock_agent_cls, anima_dir, shared_dir
    ):
        """process_message should clear the interrupt event for the thread."""
        mock_mm_cls.return_value.read_model_config.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_agent.set_on_message_sent = MagicMock()
        mock_agent.set_on_schedule_changed = MagicMock()
        mock_agent.set_interrupt_event = MagicMock()
        mock_agent.background_manager = None
        mock_agent._tool_handler = MagicMock()
        mock_agent.execution_mode = "a"

        async def mock_stream(*_args, **_kwargs):
            yield {
                "type": "cycle_done",
                "cycle_result": {
                    "trigger": "message:human",
                    "action": "responded",
                    "summary": "test",
                    "session_type": "chat",
                    "thread_id": "default",
                    "tool_call_records": [],
                },
            }

        mock_agent.run_cycle_streaming = mock_stream
        mock_agent_cls.return_value = mock_agent

        from core.anima.digital_anima import DigitalAnima

        anima = DigitalAnima(anima_dir, shared_dir)

        evt = anima._get_interrupt_event("default")
        evt.set()
        assert evt.is_set()

        # process_message should clear it (even though it may fail due to mocking,
        # the clear() happens at the very start)
        try:
            await anima.process_message("test", from_person="human")
        except Exception:
            pass

        assert not evt.is_set()


# ── AgentCore interrupt_event propagation ────────────────


class TestAgentCoreInterruptPropagation:
    """Test that AgentCore propagates interrupt_event to executors."""

    def test_set_interrupt_event(self):
        from core.agent.agent_core import AgentCore

        agent = MagicMock(spec=AgentCore)
        agent._interrupt_event = None
        agent._executor = MagicMock()
        agent._executor._interrupt_event = None

        # Call the real method
        event = asyncio.Event()
        AgentCore.set_interrupt_event(agent, event)

        assert agent._interrupt_event is event
        assert agent._executor._interrupt_event is event
