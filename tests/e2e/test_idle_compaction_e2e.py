# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for session idle auto-compaction."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.lifecycle import LifecycleManager
from core.schemas import CycleResult
from core.session_compactor import SessionCompactor, run_idle_compaction
from core.tooling.handler import active_session_type

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def anima(data_dir: Path, make_anima):
    """Create a DigitalAnima with mocked agent (execution_mode 'a')."""
    anima_dir = make_anima("test-anima", execution_mode="a", model="claude-sonnet-4-6")
    shared_dir = data_dir / "shared"
    for d in [
        anima_dir / "activity_log",
    ]:
        d.mkdir(parents=True, exist_ok=True)
    with patch("core.anima.AgentCore") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.background_manager = None
        mock_agent.execution_mode = "a"
        mock_agent._tool_handler = MagicMock()
        mock_agent.model_config = MagicMock()
        mock_agent_cls.return_value = mock_agent
        from core.anima import DigitalAnima

        return DigitalAnima(anima_dir, shared_dir)


# ── Test 1: Timer scheduled after process_message and fires compaction ─────


@pytest.mark.asyncio
async def test_timer_scheduled_after_process_message_and_fires_compaction(anima, data_dir: Path, make_anima) -> None:
    """Message processing triggers timer scheduling; timer fires compaction."""
    # Patch load_config so SessionCompactor gets very short idle (0.01 min ≈ 0.6s)
    mock_config = MagicMock()
    mock_config.heartbeat.idle_compaction_minutes = 0.01
    with patch("core.config.models.load_config", return_value=mock_config):
        anima_dir = make_anima("test-anima", execution_mode="a", model="claude-sonnet-4-6")
        shared_dir = data_dir / "shared"
        with patch("core.anima.AgentCore") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.background_manager = None
            mock_agent.execution_mode = "a"
            mock_agent._tool_handler = MagicMock()
            mock_agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)
            mock_agent.model_config = MagicMock()
            mock_agent_cls.return_value = mock_agent
            from core.anima import DigitalAnima

            anima = DigitalAnima(anima_dir, shared_dir)

    # Mock run_cycle to return quickly
    cycle_result = CycleResult(
        trigger="message:human",
        action="respond",
        summary="Test response",
        duration_ms=10,
        tool_call_records=[],
    )
    anima.agent.run_cycle = AsyncMock(return_value=cycle_result)

    # Mock ConversationMemory.compress_if_needed to avoid LLM calls.
    # Patch must extend through timer fire (run_idle_compaction uses it).
    compress_called = asyncio.Event()

    async def mock_compress():
        compress_called.set()
        return False

    mock_conv = MagicMock()
    mock_conv.compress_if_needed = AsyncMock(side_effect=mock_compress)
    mock_conv.build_structured_messages = MagicMock(return_value=[])
    mock_conv.append_turn = MagicMock()
    mock_conv.save = MagicMock()
    mock_conv.write_transcript = MagicMock()
    mock_conv.load = MagicMock(return_value=MagicMock(turns=[], compressed_summary=""))
    mock_conv.finalize_if_session_ended = AsyncMock()

    mock_conv_cls = MagicMock(return_value=mock_conv)

    # Patch at source; both process_message and run_idle_compaction import from here
    with patch("core.memory.conversation.ConversationMemory", mock_conv_cls):
        # Run process_message
        result = await anima.process_message("Hello", from_person="human")

        assert result == "Test response"

        # Verify timer is registered
        key = ("test-anima", "default")
        assert key in anima._session_compactor._timers

        # Fast-forward: wait for timer to fire (0.01 min = 0.6 s)
        await asyncio.sleep(1.0)

        # Timer callback creates task for run_idle_compaction; give it time
        await asyncio.sleep(0.5)

        # Verify compaction ran (compress_if_needed was called by run_idle_compaction)
        assert compress_called.is_set()


# ── Test 2: Timer cancelled when new message arrives ───────────────────────


@pytest.mark.asyncio
async def test_timer_cancelled_when_new_message_arrives() -> None:
    """Schedule a timer; cancel() removes it (simulating new message arrival)."""
    compactor = SessionCompactor(idle_minutes=10.0)
    callback = MagicMock()

    compactor.schedule("alice", "default", callback)
    assert ("alice", "default") in compactor._timers

    compactor.cancel("alice", "default")
    assert ("alice", "default") not in compactor._timers


# ── Test 3: Lifecycle shutdown cancels all compaction timers ───────────────


@pytest.mark.asyncio
async def test_lifecycle_shutdown_cancels_all_compaction_timers(anima) -> None:
    """Register animas, schedule timers, shutdown lifecycle; all timers cancelled."""
    lifecycle = LifecycleManager()
    lifecycle.register_anima(anima)

    # Schedule timers for the anima (need event loop for schedule)
    anima._session_compactor.schedule("test-anima", "default", lambda: None)
    anima._session_compactor.schedule("test-anima", "thread-2", lambda: None)

    assert ("test-anima", "default") in anima._session_compactor._timers
    assert ("test-anima", "thread-2") in anima._session_compactor._timers

    # Patch scheduler.shutdown to avoid SchedulerNotRunningError
    with patch.object(lifecycle.scheduler, "shutdown"):
        lifecycle.shutdown()

    assert len(anima._session_compactor._timers) == 0


# ── Test 4: Config idle_compaction_minutes changes timer delay ──────────────


def test_config_idle_compaction_minutes_sets_timer_delay() -> None:
    """SessionCompactor uses idle_compaction_minutes from config."""
    custom_minutes = 5.5
    compactor = SessionCompactor(idle_minutes=custom_minutes)
    assert compactor._idle_minutes == custom_minutes


# ── Test 5: Activity log records idle_compaction event ────────────────────


@pytest.mark.asyncio
async def test_activity_log_records_idle_compaction_event(anima) -> None:
    """Run compaction; activity_log contains 'idle_compaction' event."""
    # Mock ConversationMemory to avoid LLM
    with patch(
        "core.session_compactor._compact_mode_a",
        new_callable=AsyncMock,
        return_value=True,
    ):
        await run_idle_compaction(anima, "default")

    # Find today's activity log
    from core.time_utils import now_jst

    date_str = now_jst().strftime("%Y-%m-%d")
    log_dir = anima.anima_dir / "activity_log"
    log_path = log_dir / f"{date_str}.jsonl"

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    entries = [json.loads(line) for line in lines if line]
    types = [e.get("type") for e in entries]
    assert "idle_compaction" in types
