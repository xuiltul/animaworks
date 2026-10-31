from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for heartbeat/conversation concurrency feature.

Tests cover the lock separation, status slot system, concurrent lock
acquisition, session file separation, streaming journal separation,
short-term memory separation, board fanout context variable, and
replied_to session separation.
"""

import asyncio
import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from core.anima.digital_anima import DigitalAnima
    from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def anima(tmp_path: Path) -> DigitalAnima:
    """Create a minimal DigitalAnima with mocked internals."""
    anima_dir = tmp_path / "animas" / "test-anima"
    shared_dir = tmp_path / "shared"

    # Create minimum required directory structure
    for d in [
        anima_dir / "state",
        anima_dir / "episodes",
        anima_dir / "knowledge",
        anima_dir / "procedures",
        anima_dir / "skills",
        anima_dir / "shortterm",
        anima_dir / "activity_log",
        shared_dir / "inbox" / "test-anima",
        shared_dir / "channels",
        shared_dir / "users",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Write minimal required files
    (anima_dir / "IDENTITY.md").write_text("# Test Anima", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("Test injection", encoding="utf-8")
    (anima_dir / "status.json").write_text(
        '{"enabled": true, "role": "general", "model": "claude-sonnet-4-6"}',
        encoding="utf-8",
    )

    def _mock_agent() -> MagicMock:
        mock_agent = MagicMock()
        mock_agent.background_manager = None
        mock_agent.execution_mode = "s"
        mock_agent._tool_handler = MagicMock()
        return mock_agent

    with patch("core.anima.digital_anima.AgentCore") as mock_agent_cls:
        mock_agent_cls.side_effect = [_mock_agent(), _mock_agent(), _mock_agent()]

        from core.anima.digital_anima import DigitalAnima

        return DigitalAnima(anima_dir, shared_dir)


@pytest.fixture
def handler(tmp_path: Path) -> ToolHandler:
    """Create a minimal ToolHandler for testing replied_to."""
    anima_dir = tmp_path / "test-anima"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    with patch("core.tooling.handler.MemoryManager"), patch("core.tooling.handler.Messenger"):
        from core.tooling.handler import ToolHandler

        return ToolHandler(
            anima_dir=anima_dir,
            memory=MagicMock(),
            messenger=MagicMock(),
        )


# ── Test 1: Lock separation ──────────────────────────────────────


@pytest.mark.unit
class TestLockSeparation:
    """Verify that conversation and background locks are independent."""

    def test_anima_has_separate_locks(self, anima: DigitalAnima) -> None:
        """DigitalAnima should have _conversation_locks dict and _background_lock."""
        assert hasattr(anima, "_conversation_locks")
        assert hasattr(anima, "_background_lock")
        assert hasattr(anima, "_agent_session_lock")
        assert hasattr(anima, "_agent_session_locks")
        assert anima._get_thread_lock("default") is not anima._background_lock
        assert anima._agent_session_lock is not anima._background_lock
        assert anima._agent_session_lock is not anima._get_thread_lock("default")
        assert anima._agent_session_lock is anima._agent_session_locks["chat"]
        assert anima._agent_session_locks["chat"] is not anima._agent_session_locks["background"]
        assert anima._agent_session_locks["chat"] is not anima._agent_session_locks["inbox"]
        assert anima._agent_session_locks["background"] is not anima._agent_session_locks["inbox"]

    def test_anima_has_separate_lane_agents(self, anima: DigitalAnima) -> None:
        """DigitalAnima should isolate AgentCore instances per execution lane."""
        assert anima.agent is anima._agent_for_lane("chat")
        assert anima._agent_for_lane("chat") is not anima._agent_for_lane("background")
        assert anima._agent_for_lane("chat") is not anima._agent_for_lane("inbox")
        assert anima._agent_for_lane("background") is not anima._agent_for_lane("inbox")

    def test_callbacks_are_wired_to_all_lane_agents(self, anima: DigitalAnima) -> None:
        """Public callback setters should propagate to every lane agent."""
        on_message = MagicMock()
        on_schedule = MagicMock()

        anima.set_on_message_sent(on_message)
        anima.set_on_schedule_changed(on_schedule)

        for agent in anima._iter_lane_agents():
            agent.set_on_message_sent.assert_called_with(on_message)
            agent.set_on_schedule_changed.assert_called_with(on_schedule)

    def test_reload_config_updates_all_lane_agents(self, anima: DigitalAnima) -> None:
        """Config reload should not leave background/inbox agents stale."""
        from core.schemas import ModelConfig

        new_config = ModelConfig(model="claude-sonnet-4-6")
        anima.memory.read_model_config = MagicMock(return_value=new_config)

        result = anima.reload_config()

        assert result["model"] == "claude-sonnet-4-6"
        for agent in anima._iter_lane_agents():
            agent.update_model_config.assert_called_with(new_config)

    def test_no_legacy_lock(self, anima: DigitalAnima) -> None:
        """DigitalAnima should NOT have the old single _lock."""
        assert not hasattr(anima, "_lock")

    def test_no_user_waiting(self, anima: DigitalAnima) -> None:
        """DigitalAnima should NOT have _user_waiting event."""
        assert not hasattr(anima, "_user_waiting")

    def test_no_heartbeat_stream_queue(self, anima: DigitalAnima) -> None:
        """DigitalAnima should NOT have _heartbeat_stream_queue."""
        assert not hasattr(anima, "_heartbeat_stream_queue")


# ── Test 2: Status slots ─────────────────────────────────────────


@pytest.mark.unit
class TestStatusSlots:
    """Verify status slot system works correctly."""

    def test_initial_status_idle(self, anima: DigitalAnima) -> None:
        """Status slots should start with inbox and background as idle."""
        assert anima._status_slots["inbox"] == "idle"
        assert anima._status_slots["background"] == "idle"
        assert anima._task_slots["inbox"] == ""
        assert anima._task_slots["background"] == ""

    def test_primary_status_conversation_priority(
        self,
        anima: DigitalAnima,
    ) -> None:
        """primary_status should prefer conversation over background."""
        anima._status_slots["conversation:default"] = "thinking"
        anima._status_slots["background"] = "checking"
        assert anima.primary_status == "thinking"

    def test_primary_status_background_fallback(
        self,
        anima: DigitalAnima,
    ) -> None:
        """primary_status should return background when no conversation is active."""
        anima._status_slots["background"] = "checking"
        assert anima.primary_status == "checking"

    def test_primary_status_both_idle(self, anima: DigitalAnima) -> None:
        """primary_status should return idle when both are idle."""
        assert anima.primary_status == "idle"

    def test_status_property_returns_anima_status(
        self,
        anima: DigitalAnima,
    ) -> None:
        """status property should return AnimaStatus with primary values."""
        anima._status_slots["background"] = "checking"
        s = anima.status
        assert s.status == "checking"
        assert s.name == anima.name


# ── Test 3: Concurrent lock acquisition ──────────────────────────


@pytest.mark.unit
class TestConcurrentLockAcquisition:
    """Verify that conversation and background locks can be held simultaneously."""

    async def test_both_locks_can_be_held(
        self,
        anima: DigitalAnima,
    ) -> None:
        """Both locks should be acquirable at the same time."""
        async with anima._get_thread_lock("default"), anima._background_lock:
            assert anima._get_thread_lock("default").locked()
            assert anima._background_lock.locked()

    async def test_conversation_does_not_block_background(
        self,
        anima: DigitalAnima,
    ) -> None:
        """Acquiring conversation lock should not prevent background lock acquisition."""
        acquired = False
        # This should NOT block
        async with anima._get_thread_lock("default"), anima._background_lock:
            acquired = True
        assert acquired

    async def test_background_does_not_block_conversation(
        self,
        anima: DigitalAnima,
    ) -> None:
        """Acquiring background lock should not prevent conversation lock acquisition."""
        acquired = False
        async with anima._background_lock, anima._get_thread_lock("default"):
            acquired = True
        assert acquired

    async def test_chat_agent_session_lock_serializes_chat_agent_state(
        self,
        anima: DigitalAnima,
    ) -> None:
        """The chat lane preparation lock should admit one chat session at a time."""
        holder_entered = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with anima._agent_session_lock:
                holder_entered.set()
                await release_holder.wait()

        holder_task = asyncio.create_task(holder())
        await holder_entered.wait()

        waiter = asyncio.create_task(anima._agent_session_lock.acquire())
        await asyncio.sleep(0)
        assert waiter.done() is False

        release_holder.set()
        await holder_task
        await asyncio.wait_for(waiter, timeout=1.0)
        anima._agent_session_lock.release()

    async def test_background_agent_session_lock_does_not_block_chat(
        self,
        anima: DigitalAnima,
    ) -> None:
        """A TaskExec/background session lock must not block chat session setup."""
        async with anima._agent_session_context("background"):
            waiter = asyncio.create_task(anima._agent_session_locks["chat"].acquire())
            await asyncio.wait_for(waiter, timeout=1.0)
            assert waiter.done()
            anima._agent_session_locks["chat"].release()

    async def test_chat_agent_session_lock_does_not_block_background(
        self,
        anima: DigitalAnima,
    ) -> None:
        """A chat session lock must not block TaskExec/background session setup."""
        async with anima._agent_session_context("chat"):
            waiter = asyncio.create_task(anima._agent_session_locks["background"].acquire())
            await asyncio.wait_for(waiter, timeout=1.0)
            assert waiter.done()
            anima._agent_session_locks["background"].release()

    def test_agent_entrypoints_use_session_lock(self) -> None:
        """Chat/background/inbox entrypoints should guard AgentCore mutable state."""
        from core.anima.inbox import InboxMixin
        from core.anima.lifecycle import LifecycleMixin
        from core.anima.messaging import MessagingMixin
        from core.tasks.pending_executor import PendingTaskExecutor

        def guarded(obj, *helpers) -> bool:
            sources = [inspect.getsource(obj)]
            for helper in helpers:
                if helper is not None:
                    sources.append(inspect.getsource(helper))
            combined = "\n".join(sources)
            return "_agent_session_lock" in combined or "_agent_session_context" in combined

        assert guarded(MessagingMixin.process_message, MessagingMixin.process_message_stream)
        assert guarded(MessagingMixin.process_message_stream)
        # run_heartbeat routes the agent cycle through _run_heartbeat_agent_session
        assert guarded(
            LifecycleMixin.run_heartbeat,
            getattr(LifecycleMixin, "_run_heartbeat_agent_session", None),
        )
        assert guarded(LifecycleMixin._run_daily_consolidation)
        assert guarded(LifecycleMixin._run_weekly_consolidation)
        assert guarded(LifecycleMixin.run_cron_task)
        assert guarded(LifecycleMixin.run_cron_command)
        assert guarded(InboxMixin.process_inbox_message)
        assert guarded(PendingTaskExecutor._run_llm_task)


# ── Test 4: Session file separation ──────────────────────────────


@pytest.mark.unit
class TestSessionFilesSeparation:
    """Verify session ID files are correctly separated by session type."""

    def test_session_file_chat(self) -> None:
        from core.execution.engines.claude._sdk_session import _session_file

        assert _session_file("chat") == "current_session_chat.json"

    def test_session_file_heartbeat(self) -> None:
        from core.execution.engines.claude._sdk_session import _session_file

        assert _session_file("heartbeat") == "current_session_heartbeat.json"

    def test_load_save_independent(self, tmp_path: Path) -> None:
        """Chat and heartbeat session IDs should not interfere."""
        from core.execution.engines.claude._sdk_session import _load_session_id, _save_session_id

        anima_dir = tmp_path / "test-anima"
        (anima_dir / "state").mkdir(parents=True)

        _save_session_id(anima_dir, "chat-session-123", session_type="chat")
        _save_session_id(anima_dir, "hb-session-456", session_type="heartbeat")

        assert _load_session_id(anima_dir, session_type="chat") == "chat-session-123"
        assert _load_session_id(anima_dir, session_type="heartbeat") == "hb-session-456"


# ── Test 5: Streaming journal separation ─────────────────────────


@pytest.mark.unit
class TestStreamingJournalSeparation:
    """Verify streaming journals use separate files per session type."""

    def test_journal_file_names(self, tmp_path: Path) -> None:
        from core.memory.conversation.streaming_journal import StreamingJournal

        anima_dir = tmp_path / "test-anima"
        (anima_dir / "shortterm").mkdir(parents=True)

        chat_j = StreamingJournal(anima_dir, session_type="chat")
        hb_j = StreamingJournal(anima_dir, session_type="heartbeat")

        assert "streaming_journal_chat.jsonl" in str(chat_j._journal_path)
        assert "streaming_journal_heartbeat.jsonl" in str(hb_j._journal_path)
        assert chat_j._journal_path != hb_j._journal_path


# ── Test 6: ShortTermMemory separation ───────────────────────────


@pytest.mark.unit
class TestShortTermMemorySeparation:
    """Verify ShortTermMemory uses separate directories per session type."""

    def test_separate_directories(self, tmp_path: Path) -> None:
        from core.memory.conversation.shortterm import ShortTermMemory

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()

        chat_stm = ShortTermMemory(anima_dir, session_type="chat")
        hb_stm = ShortTermMemory(anima_dir, session_type="heartbeat")

        assert chat_stm.shortterm_dir != hb_stm.shortterm_dir
        assert "chat" in str(chat_stm.shortterm_dir)
        assert "heartbeat" in str(hb_stm.shortterm_dir)
        assert chat_stm.shortterm_dir.exists()
        assert hb_stm.shortterm_dir.exists()


# ── Test 7: suppress_board_fanout context variable ───────────────


@pytest.mark.unit
class TestSuppressBoardFanoutContextVar:
    """Verify suppress_board_fanout uses contextvars."""

    def test_default_is_false(self) -> None:
        from core.tooling.handler import suppress_board_fanout

        assert suppress_board_fanout.get() is False

    def test_set_and_reset(self) -> None:
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(True)
        assert suppress_board_fanout.get() is True
        suppress_board_fanout.reset(token)
        assert suppress_board_fanout.get() is False


# ── Test 8: replied_to session separation ────────────────────────


@pytest.mark.unit
class TestRepliedToSessionSeparation:
    """Verify replied_to tracks sessions independently."""

    def test_replied_to_dict_structure(self, handler: ToolHandler) -> None:
        """_replied_to should be a dict with isolated runtime session keys."""
        assert isinstance(handler._replied_to, dict)
        assert "chat" in handler._replied_to
        assert "heartbeat" in handler._replied_to
        assert "cron" in handler._replied_to
        assert "task" in handler._replied_to
        assert "inbox" in handler._replied_to

    def test_replied_to_property_returns_union(
        self,
        handler: ToolHandler,
    ) -> None:
        handler._replied_to["chat"].add("alice")
        handler._replied_to["heartbeat"].add("bob")
        assert handler.replied_to == {"alice", "bob"}

    def test_reset_specific_session(self, handler: ToolHandler) -> None:
        handler._replied_to["chat"].add("alice")
        handler._replied_to["heartbeat"].add("bob")
        handler.reset_replied_to(session_type="chat")
        assert handler._replied_to["chat"] == set()
        assert handler._replied_to["heartbeat"] == {"bob"}

    def test_set_active_session_type(self, handler: ToolHandler) -> None:
        from core.tooling.handler import active_session_type

        token = handler.set_active_session_type("heartbeat")
        assert active_session_type.get() == "heartbeat"
        active_session_type.reset(token)
        assert active_session_type.get() == "unknown"

    def test_active_session_type_is_contextvar(self) -> None:
        """active_session_type should be a ContextVar with a neutral default."""
        from core.tooling.handler import active_session_type

        assert active_session_type.get() == "unknown"
