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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def anima(tmp_path: Path) -> "DigitalAnima":
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
    (anima_dir / "identity.md").write_text("# Test Anima", encoding="utf-8")
    (anima_dir / "injection.md").write_text("Test injection", encoding="utf-8")
    (anima_dir / "status.json").write_text(
        '{"enabled": true, "role": "general", "model": "claude-sonnet-4-6"}',
        encoding="utf-8",
    )

    with patch("core.anima.AgentCore") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.background_manager = None
        mock_agent.execution_mode = "s"
        mock_agent._tool_handler = MagicMock()
        mock_agent_cls.return_value = mock_agent

        from core.anima import DigitalAnima
        return DigitalAnima(anima_dir, shared_dir)


@pytest.fixture
def handler(tmp_path: Path) -> "ToolHandler":
    """Create a minimal ToolHandler for testing replied_to."""
    anima_dir = tmp_path / "test-anima"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    with patch("core.tooling.handler.MemoryManager"):
        with patch("core.tooling.handler.Messenger"):
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

    def test_anima_has_separate_locks(self, anima: "DigitalAnima") -> None:
        """DigitalAnima should have _conversation_locks dict and _background_lock."""
        assert hasattr(anima, "_conversation_locks")
        assert hasattr(anima, "_background_lock")
        assert anima._get_thread_lock("default") is not anima._background_lock

    def test_no_legacy_lock(self, anima: "DigitalAnima") -> None:
        """DigitalAnima should NOT have the old single _lock."""
        assert not hasattr(anima, "_lock")

    def test_no_user_waiting(self, anima: "DigitalAnima") -> None:
        """DigitalAnima should NOT have _user_waiting event."""
        assert not hasattr(anima, "_user_waiting")

    def test_no_heartbeat_stream_queue(self, anima: "DigitalAnima") -> None:
        """DigitalAnima should NOT have _heartbeat_stream_queue."""
        assert not hasattr(anima, "_heartbeat_stream_queue")


# ── Test 2: Status slots ─────────────────────────────────────────


@pytest.mark.unit
class TestStatusSlots:
    """Verify status slot system works correctly."""

    def test_initial_status_idle(self, anima: "DigitalAnima") -> None:
        """All status slots should start as idle."""
        assert anima._status_slots == {"conversation": "idle", "inbox": "idle", "background": "idle"}
        assert anima._task_slots == {"conversation": "", "inbox": "", "background": ""}

    def test_primary_status_conversation_priority(
        self, anima: "DigitalAnima",
    ) -> None:
        """primary_status should prefer conversation over background."""
        anima._status_slots["conversation"] = "thinking"
        anima._status_slots["background"] = "checking"
        assert anima.primary_status == "thinking"

    def test_primary_status_background_fallback(
        self, anima: "DigitalAnima",
    ) -> None:
        """primary_status should return background when conversation is idle."""
        anima._status_slots["conversation"] = "idle"
        anima._status_slots["background"] = "checking"
        assert anima.primary_status == "checking"

    def test_primary_status_both_idle(self, anima: "DigitalAnima") -> None:
        """primary_status should return idle when both are idle."""
        assert anima.primary_status == "idle"

    def test_status_property_returns_anima_status(
        self, anima: "DigitalAnima",
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
        self, anima: "DigitalAnima",
    ) -> None:
        """Both locks should be acquirable at the same time."""
        async with anima._get_thread_lock("default"):
            async with anima._background_lock:
                assert anima._get_thread_lock("default").locked()
                assert anima._background_lock.locked()

    async def test_conversation_does_not_block_background(
        self, anima: "DigitalAnima",
    ) -> None:
        """Acquiring conversation lock should not prevent background lock acquisition."""
        acquired = False
        async with anima._get_thread_lock("default"):
            # This should NOT block
            async with anima._background_lock:
                acquired = True
        assert acquired

    async def test_background_does_not_block_conversation(
        self, anima: "DigitalAnima",
    ) -> None:
        """Acquiring background lock should not prevent conversation lock acquisition."""
        acquired = False
        async with anima._background_lock:
            async with anima._get_thread_lock("default"):
                acquired = True
        assert acquired


# ── Test 4: Session file separation ──────────────────────────────


@pytest.mark.unit
class TestSessionFilesSeparation:
    """Verify session ID files are correctly separated by session type."""

    def test_session_file_chat(self) -> None:
        from core.execution.agent_sdk import _session_file
        assert _session_file("chat") == "current_session_chat.json"

    def test_session_file_heartbeat(self) -> None:
        from core.execution.agent_sdk import _session_file
        assert _session_file("heartbeat") == "current_session_heartbeat.json"

    def test_load_save_independent(self, tmp_path: Path) -> None:
        """Chat and heartbeat session IDs should not interfere."""
        from core.execution.agent_sdk import _load_session_id, _save_session_id

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
        from core.memory.streaming_journal import StreamingJournal

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
        from core.memory.shortterm import ShortTermMemory

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

    def test_replied_to_dict_structure(self, handler: "ToolHandler") -> None:
        """_replied_to should be a dict with chat and background keys."""
        assert isinstance(handler._replied_to, dict)
        assert "chat" in handler._replied_to
        assert "background" in handler._replied_to

    def test_replied_to_property_returns_union(
        self, handler: "ToolHandler",
    ) -> None:
        handler._replied_to["chat"].add("alice")
        handler._replied_to["background"].add("bob")
        assert handler.replied_to == {"alice", "bob"}

    def test_reset_specific_session(self, handler: "ToolHandler") -> None:
        handler._replied_to["chat"].add("alice")
        handler._replied_to["background"].add("bob")
        handler.reset_replied_to(session_type="chat")
        assert handler._replied_to["chat"] == set()
        assert handler._replied_to["background"] == {"bob"}

    def test_set_active_session_type(self, handler: "ToolHandler") -> None:
        from core.tooling.handler import active_session_type
        token = handler.set_active_session_type("background")
        assert active_session_type.get() == "background"
        active_session_type.reset(token)
        assert active_session_type.get() == "chat"

    def test_active_session_type_is_contextvar(self) -> None:
        """active_session_type should be a ContextVar with default 'chat'."""
        from core.tooling.handler import active_session_type
        assert active_session_type.get() == "chat"
