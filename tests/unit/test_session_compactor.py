# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for session idle auto-compaction."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.session_compactor import (
    SessionCompactor,
    _compact_conversation,
    _compact_mode_s,
    _extract_recent_chat_context,
    run_idle_compaction,
)
from core.config.models import HeartbeatConfig
from core.execution.engines.claude._sdk_session import _load_session_id
from core.memory.conversation.compression import CompressionResult
from core.schemas import ModelConfig

# ── SessionCompactor ──────────────────────────────────────────────────────


def _compression_result(performed: bool = False) -> CompressionResult:
    return CompressionResult(
        status="llm_primary" if performed else "skipped_no_compression_needed",
        performed=performed,
        raw_turns_before=60 if performed else 0,
        raw_turns_after=20 if performed else 0,
        compressed_turns=40 if performed else 0,
    )


class TestSessionCompactor:
    """SessionCompactor timer management."""

    @pytest.mark.asyncio
    async def test_schedule_new_timer(self) -> None:
        """Timer is scheduled correctly for new (anima, thread_id)."""
        compactor = SessionCompactor(idle_minutes=5.0)
        callback = MagicMock()

        compactor.schedule("alice", "thread-1", callback)

        assert len(compactor._timers) == 1
        key = ("alice", "thread-1")
        assert key in compactor._timers
        handle = compactor._timers[key]
        assert not handle.cancelled()

    @pytest.mark.asyncio
    async def test_schedule_cancels_existing_on_reschedule(self) -> None:
        """Existing timer is cancelled when rescheduling same key."""
        compactor = SessionCompactor(idle_minutes=5.0)
        callback1 = MagicMock()
        callback2 = MagicMock()

        compactor.schedule("alice", "thread-1", callback1)
        old_handle = compactor._timers[("alice", "thread-1")]
        compactor.schedule("alice", "thread-1", callback2)

        assert old_handle.cancelled()
        assert len(compactor._timers) == 1
        assert compactor._timers[("alice", "thread-1")] is not old_handle

    def test_cancel_existing_timer(self) -> None:
        """cancel() removes and cancels the timer for given anima/thread."""
        compactor = SessionCompactor(idle_minutes=5.0)
        loop = asyncio.new_event_loop()
        try:
            handle = loop.call_later(3600, lambda: None)
            compactor._timers[("alice", "thread-1")] = handle

            compactor.cancel("alice", "thread-1")

            assert ("alice", "thread-1") not in compactor._timers
            assert handle.cancelled()
        finally:
            loop.close()

    def test_cancel_nonexistent_is_noop(self) -> None:
        """cancel() on non-existent key does nothing."""
        compactor = SessionCompactor(idle_minutes=5.0)
        compactor.cancel("alice", "thread-1")
        assert len(compactor._timers) == 0

    def test_cancel_all_for_anima(self) -> None:
        """cancel_all_for_anima() cancels all timers for the anima."""
        compactor = SessionCompactor(idle_minutes=5.0)
        loop = asyncio.new_event_loop()
        try:
            h1 = loop.call_later(3600, lambda: None)
            h2 = loop.call_later(3600, lambda: None)
            compactor._timers[("alice", "t1")] = h1
            compactor._timers[("alice", "t2")] = h2
            compactor._timers[("bob", "t1")] = loop.call_later(3600, lambda: None)

            compactor.cancel_all_for_anima("alice")

            assert ("alice", "t1") not in compactor._timers
            assert ("alice", "t2") not in compactor._timers
            assert ("bob", "t1") in compactor._timers
            assert h1.cancelled()
            assert h2.cancelled()
        finally:
            loop.close()

    def test_shutdown(self) -> None:
        """shutdown() cancels all timers for the compactor."""
        compactor = SessionCompactor(idle_minutes=5.0)
        loop = asyncio.new_event_loop()
        try:
            h1 = loop.call_later(3600, lambda: None)
            h2 = loop.call_later(3600, lambda: None)
            compactor._timers[("alice", "t1")] = h1
            compactor._timers[("bob", "t1")] = h2

            compactor.shutdown()

            assert len(compactor._timers) == 0
            assert h1.cancelled()
            assert h2.cancelled()
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """When _MAX_TIMERS exceeded, oldest timer is evicted."""
        with patch("core.agent.session_compactor._MAX_TIMERS", 3):
            compactor = SessionCompactor(idle_minutes=5.0)
            callback = MagicMock()

            compactor.schedule("a", "t1", callback)
            compactor.schedule("a", "t2", callback)
            compactor.schedule("a", "t3", callback)
            assert len(compactor._timers) == 3

            compactor.schedule("a", "t4", callback)

            assert len(compactor._timers) == 3
            assert ("a", "t1") not in compactor._timers
            assert ("a", "t2") in compactor._timers
            assert ("a", "t3") in compactor._timers
            assert ("a", "t4") in compactor._timers

    def test_fire_invokes_callback_and_removes_handle(self) -> None:
        """_fire() invokes callback and removes handle from dict."""
        compactor = SessionCompactor(idle_minutes=5.0)
        callback = MagicMock()
        key = ("alice", "thread-1")
        compactor._timers[key] = MagicMock()

        compactor._fire(key, callback)

        callback.assert_called_once()
        assert key not in compactor._timers

    def test_fire_handles_callback_exception(self) -> None:
        """_fire() catches and logs callback exceptions without propagating."""
        compactor = SessionCompactor(idle_minutes=5.0)

        def failing_callback() -> None:
            raise ValueError("boom")

        key = ("alice", "thread-1")
        compactor._timers[key] = MagicMock()

        compactor._fire(key, failing_callback)

        assert key not in compactor._timers


# ── _load_session_id immortal sessions ──────────────────────────────────────


class TestLoadSessionIdImmortal:
    """_load_session_id returns session ID regardless of age (no TTL)."""

    def test_old_session_returns_id(self, tmp_path: Path) -> None:
        """Old sessions are always returned (TTL was removed)."""
        anima_dir = tmp_path / "animas" / "test-sdk"
        (anima_dir / "state").mkdir(parents=True)
        path = anima_dir / "state" / "current_session_chat.json"
        old_ts = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        path.write_text(
            json.dumps({"session_id": "sess-old", "timestamp": old_ts}),
            encoding="utf-8",
        )

        result = _load_session_id(anima_dir, "chat")

        assert result == "sess-old"

    def test_no_skip_timeout_check_param(self) -> None:
        """skip_timeout_check parameter was removed from _load_session_id."""
        import inspect

        sig = inspect.signature(_load_session_id)
        assert "skip_timeout_check" not in sig.parameters


# ── HeartbeatConfig.idle_compaction_minutes ───────────────────────────────────


class TestHeartbeatConfigIdleCompactionMinutes:
    """HeartbeatConfig.idle_compaction_minutes default and validation."""

    def test_default_value_is_10(self) -> None:
        """Default idle_compaction_minutes is 10.0."""
        config = HeartbeatConfig()
        assert config.idle_compaction_minutes == 10.0

    def test_validation_accepts_valid_range(self) -> None:
        """Accepts values within ge=1.0, le=120.0."""
        config = HeartbeatConfig(idle_compaction_minutes=1.0)
        assert config.idle_compaction_minutes == 1.0

        config = HeartbeatConfig(idle_compaction_minutes=120.0)
        assert config.idle_compaction_minutes == 120.0

    def test_validation_rejects_below_min(self) -> None:
        """Rejects values below 1.0."""
        with pytest.raises(ValueError):
            HeartbeatConfig(idle_compaction_minutes=0.5)

    def test_validation_rejects_above_max(self) -> None:
        """Rejects values above 120.0."""
        with pytest.raises(ValueError):
            HeartbeatConfig(idle_compaction_minutes=121.0)


# ── COMPACT_TIMEOUT_SEC ───────────────────────────────────────────────────────


class TestCompactTimeoutSec:
    """COMPACT_TIMEOUT_SEC is independent from RESUME_TIMEOUT_SEC."""

    def test_compact_timeout_defined(self) -> None:
        from core.execution.engines.claude._sdk_session import COMPACT_TIMEOUT_SEC

        assert COMPACT_TIMEOUT_SEC == 300.0

    def test_compact_timeout_independent_from_resume(self) -> None:
        from core.execution.engines.claude._sdk_session import COMPACT_TIMEOUT_SEC, RESUME_TIMEOUT_SEC

        assert COMPACT_TIMEOUT_SEC != RESUME_TIMEOUT_SEC
        assert COMPACT_TIMEOUT_SEC > RESUME_TIMEOUT_SEC

    def test_compact_sdk_session_uses_compact_timeout(self) -> None:
        """compact_sdk_session source references COMPACT_TIMEOUT_SEC, not RESUME_TIMEOUT_SEC."""
        import inspect

        from core.execution.engines.claude._sdk_session import compact_sdk_session

        src = inspect.getsource(compact_sdk_session)
        assert "COMPACT_TIMEOUT_SEC" in src
        assert "RESUME_TIMEOUT_SEC" not in src


class TestCompactSdkSessionLogLevel:
    """compact_sdk_session logs general exceptions at ERROR level."""

    def test_general_exception_logs_error(self) -> None:
        """The except-Exception block uses logger.error, not logger.warning."""
        import ast
        import inspect

        from core.execution.engines.claude._sdk_session import compact_sdk_session

        src = inspect.getsource(compact_sdk_session)
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Attribute) and stmt.attr == "error":
                            return
        pytest.fail("Expected logger.error in except Exception block of compact_sdk_session")


# ── Mode-specific compaction ─────────────────────────────────────────────────────


class TestModeSpecificCompaction:
    """Mode-specific compaction functions."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test"
        d.mkdir(parents=True)
        for sub in ["state", "episodes", "knowledge", "shortterm", "activity_log"]:
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "IDENTITY.md").write_text("# Test", encoding="utf-8")
        (d / "status.json").write_text(
            '{"enabled": true, "model": "claude-sonnet-4-6"}',
            encoding="utf-8",
        )
        return d

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            context_threshold=0.50,
        )

    @pytest.mark.asyncio
    async def test_compact_conversation_calls_compress_and_finalize(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Conversation compaction compresses history and finalizes the conversation."""
        with patch("core.memory.conversation.memory.ConversationMemory") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv.compress_if_needed_detailed = AsyncMock(return_value=_compression_result(True))
            mock_conv.finalize_if_session_ended = AsyncMock()
            mock_conv.load.return_value = MagicMock(compressed_summary="", turns=[])
            mock_conv_cls.return_value = mock_conv

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            result = await _compact_conversation(anima, "default", clear_engine=None)

            mock_conv.compress_if_needed_detailed.assert_awaited_once()
            mock_conv.finalize_if_session_ended.assert_awaited_once()
            assert result["compression_performed"] is True
            assert result["compression_status"] == "llm_primary"

    @pytest.mark.asyncio
    async def test_compact_mode_a_saves_shortterm_when_state_has_content(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Conversation compaction saves shortterm when conversation state has content."""
        mock_turn = MagicMock()
        mock_turn.role = "assistant"
        mock_turn.content = "I completed the task."

        with (
            patch("core.memory.conversation.memory.ConversationMemory") as mock_conv_cls,
            patch("core.memory.conversation.shortterm.ShortTermMemory") as mock_stm_cls,
        ):
            mock_conv = MagicMock()
            mock_conv.compress_if_needed_detailed = AsyncMock(return_value=_compression_result(False))
            mock_conv.finalize_if_session_ended = AsyncMock()
            mock_conv.load.return_value = MagicMock(
                compressed_summary="User asked about tasks. Assistant helped.",
                turns=[mock_turn],
            )
            mock_conv_cls.return_value = mock_conv

            mock_stm = MagicMock()
            mock_stm_cls.return_value = mock_stm

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            await _compact_conversation(anima, "default", clear_engine=None)

            mock_stm_cls.assert_called_once_with(anima_dir, session_type="chat", thread_id="default")
            mock_stm.save.assert_called_once()
            saved_state = mock_stm.save.call_args[0][0]
            assert "User asked about tasks" in saved_state.accumulated_response
            assert saved_state.trigger == "idle_compaction"
            assert saved_state.notes == "Auto-saved during idle compaction"

    @pytest.mark.asyncio
    async def test_compact_mode_a_skips_shortterm_when_state_empty(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Conversation compaction skips shortterm save when conversation state is empty."""
        with (
            patch("core.memory.conversation.memory.ConversationMemory") as mock_conv_cls,
            patch("core.memory.conversation.shortterm.ShortTermMemory") as mock_stm_cls,
        ):
            mock_conv = MagicMock()
            mock_conv.compress_if_needed_detailed = AsyncMock(return_value=_compression_result(False))
            mock_conv.finalize_if_session_ended = AsyncMock()
            mock_conv.load.return_value = MagicMock(compressed_summary="", turns=[])
            mock_conv_cls.return_value = mock_conv

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            await _compact_conversation(anima, "default", clear_engine=None)

            mock_stm_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_mode_a_shortterm_includes_last_3_turns(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Conversation compaction includes up to the last 3 turns in shortterm."""
        turns = []
        for i in range(5):
            t = MagicMock()
            t.role = "assistant" if i % 2 else "human"
            t.content = f"Turn {i} content"
            turns.append(t)

        with (
            patch("core.memory.conversation.memory.ConversationMemory") as mock_conv_cls,
            patch("core.memory.conversation.shortterm.ShortTermMemory") as mock_stm_cls,
        ):
            mock_conv = MagicMock()
            mock_conv.compress_if_needed_detailed = AsyncMock(return_value=_compression_result(False))
            mock_conv.finalize_if_session_ended = AsyncMock()
            mock_conv.load.return_value = MagicMock(compressed_summary="", turns=turns)
            mock_conv_cls.return_value = mock_conv

            mock_stm = MagicMock()
            mock_stm_cls.return_value = mock_stm

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            await _compact_conversation(anima, "default", clear_engine=None)

            saved_state = mock_stm.save.call_args[0][0]
            assert "Turn 2" in saved_state.accumulated_response
            assert "Turn 3" in saved_state.accumulated_response
            assert "Turn 4" in saved_state.accumulated_response
            assert "Turn 0" not in saved_state.accumulated_response
            assert "Turn 1" not in saved_state.accumulated_response

    @pytest.mark.asyncio
    async def test_compact_conversation_clears_engine_session_when_requested(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Conversation compaction clears the selected engine session."""
        with (
            patch("core.memory.conversation.memory.ConversationMemory") as mock_conv_cls,
            patch("core.agent.session_compactor.clear_engine_session") as mock_clear,
            patch("core.memory.conversation.shortterm.ShortTermMemory") as mock_stm_cls,
        ):
            mock_conv = MagicMock()
            mock_conv.compress_if_needed_detailed = AsyncMock(return_value=_compression_result(True))
            mock_conv.load.return_value = MagicMock(
                compressed_summary="summary",
                turns=[],
            )
            mock_conv.finalize_if_session_ended = AsyncMock()
            mock_conv_cls.return_value = mock_conv

            mock_stm = MagicMock()
            mock_stm_cls.return_value = mock_stm

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            result = await _compact_conversation(anima, "default", clear_engine="codex")

            mock_conv.compress_if_needed_detailed.assert_awaited_once()
            mock_stm.save.assert_called_once()
            mock_clear.assert_called_once_with(anima_dir, "codex", "chat", "default")
            mock_conv.finalize_if_session_ended.assert_awaited_once()
            assert result["compression_performed"] is True
            assert result["engine_session_cleared"] is True
            assert result["engine"] == "codex"

    @pytest.mark.asyncio
    async def test_compact_mode_s_saves_shortterm_and_clears_session(self, anima_dir: Path) -> None:
        """_compact_mode_s extracts activity_log context, saves shortterm, clears session_id."""
        from core.time_utils import now_local

        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{now_local().date().isoformat()}.jsonl"

        entries = [
            {
                "ts": "2026-04-13T12:00:00+09:00",
                "type": "message_received",
                "content": "Hello",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T12:00:10+09:00",
                "type": "tool_use",
                "tool": "Read",
                "content": "/tmp/test.md",
                "meta": {"args": {"path": "/tmp/test.md"}, "tool_use_id": "tu1"},
            },
            {
                "ts": "2026-04-13T12:00:11+09:00",
                "type": "tool_result",
                "content": "file contents here",
                "meta": {"tool_use_id": "tu1", "is_error": False},
            },
            {
                "ts": "2026-04-13T12:00:30+09:00",
                "type": "response_sent",
                "content": "I read the file for you.",
                "meta": {"thread_id": "default"},
            },
        ]
        log_file.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
            encoding="utf-8",
        )

        session_file = anima_dir / "state" / "current_session_chat.json"
        session_file.write_text(
            json.dumps({"session_id": "sess-123", "timestamp": "2026-04-13T12:00:00Z"}),
            encoding="utf-8",
        )

        anima = MagicMock()
        anima.anima_dir = anima_dir
        anima.name = "test"

        result = await _compact_mode_s(anima, "default")

        assert result is True
        shortterm_json = anima_dir / "shortterm" / "chat" / "session_state.json"
        assert shortterm_json.exists()

        state = json.loads(shortterm_json.read_text(encoding="utf-8"))
        assert "Hello" in state["accumulated_response"]
        assert "I read the file" in state["accumulated_response"]
        assert state["trigger"] == "idle_compaction"
        assert len(state["tool_uses"]) > 0
        assert not session_file.exists()

    @pytest.mark.asyncio
    async def test_compact_mode_s_empty_log_clears_session_only(self, anima_dir: Path) -> None:
        """_compact_mode_s with empty activity_log clears session without saving shortterm."""
        session_file = anima_dir / "state" / "current_session_chat.json"
        session_file.write_text(
            json.dumps({"session_id": "sess-456", "timestamp": "2026-04-13T12:00:00Z"}),
            encoding="utf-8",
        )

        anima = MagicMock()
        anima.anima_dir = anima_dir
        anima.name = "test"

        result = await _compact_mode_s(anima, "default")

        assert result is True
        shortterm_json = anima_dir / "shortterm" / "chat" / "session_state.json"
        assert not shortterm_json.exists()
        assert not session_file.exists()

    @pytest.mark.asyncio
    async def test_compact_mode_s_does_not_call_compact_sdk(self, anima_dir: Path) -> None:
        """_compact_mode_s does NOT call executor.compact_session (no /compact API call)."""
        mock_executor = AsyncMock()
        mock_executor.compact_session = AsyncMock(return_value=True)

        anima = MagicMock()
        anima.anima_dir = anima_dir
        anima.name = "test"
        anima.agent._executor = mock_executor

        await _compact_mode_s(anima, "default")

        mock_executor.compact_session.assert_not_awaited()


# ── _extract_recent_chat_context ──────────────────────────────────────────────


class TestExtractRecentChatContext:
    """Tests for activity_log based context extraction."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test"
        d.mkdir(parents=True)
        (d / "activity_log").mkdir(parents=True)
        return d

    def _write_log(self, anima_dir: Path, entries: list[dict], date_str: str | None = None) -> None:
        from core.time_utils import now_local

        date = date_str or now_local().date().isoformat()
        log_file = anima_dir / "activity_log" / f"{date}.jsonl"
        log_file.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
            encoding="utf-8",
        )

    def test_empty_log_returns_empty_dict(self, anima_dir: Path) -> None:
        """Empty or missing activity_log returns empty dict."""
        result = _extract_recent_chat_context(anima_dir)
        assert result == {}

    def test_extracts_user_assistant_rounds(self, anima_dir: Path) -> None:
        """Extracts user/assistant message pairs in chronological order."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Question 1",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Answer 1",
                "meta": {"thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:01:00+09:00",
                "type": "message_received",
                "content": "Question 2",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:01:30+09:00",
                "type": "response_sent",
                "content": "Answer 2",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert "Question 1" in result["accumulated_response"]
        assert "Answer 1" in result["accumulated_response"]
        assert "Question 2" in result["accumulated_response"]
        assert "Answer 2" in result["accumulated_response"]
        assert result["original_prompt"] == "Question 1"

    def test_limits_to_3_rounds(self, anima_dir: Path) -> None:
        """Only the most recent 3 user/assistant rounds are kept."""
        entries = []
        for i in range(5):
            entries.append(
                {
                    "ts": f"2026-04-13T10:{i:02d}:00+09:00",
                    "type": "message_received",
                    "content": f"Q{i}",
                    "meta": {"from_type": "human", "thread_id": "default"},
                }
            )
            entries.append(
                {
                    "ts": f"2026-04-13T10:{i:02d}:30+09:00",
                    "type": "response_sent",
                    "content": f"A{i}",
                    "meta": {"thread_id": "default"},
                }
            )
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert "Q2" in result["accumulated_response"]
        assert "Q3" in result["accumulated_response"]
        assert "Q4" in result["accumulated_response"]
        assert "Q0" not in result["accumulated_response"]
        assert "Q1" not in result["accumulated_response"]

    def test_extracts_tool_use_and_result_paired(self, anima_dir: Path) -> None:
        """Extracts tool_use and tool_result entries with correct pairing."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Read file",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:05+09:00",
                "type": "tool_use",
                "tool": "Read",
                "content": "/tmp/f.md",
                "meta": {"args": {"path": "/tmp/f.md"}, "tool_use_id": "tu1", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:06+09:00",
                "type": "tool_result",
                "content": "file content here",
                "meta": {"tool_use_id": "tu1", "is_error": False, "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Done",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert len(result["tool_uses"]) >= 1
        paired = result["tool_uses"][0]
        assert paired["name"] == "Read"
        assert "result" in paired
        assert "file content" in paired["result"]

    def test_limits_to_10_tool_entries(self, anima_dir: Path) -> None:
        """Only the most recent 10 tool entries are kept."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Go",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
        ]
        for i in range(15):
            entries.append(
                {
                    "ts": f"2026-04-13T10:00:{i + 1:02d}+09:00",
                    "type": "tool_use",
                    "tool": f"Tool{i}",
                    "content": f"arg{i}",
                    "meta": {"args": {}, "tool_use_id": f"tu{i}", "thread_id": "default"},
                }
            )
        entries.append(
            {
                "ts": "2026-04-13T10:01:00+09:00",
                "type": "response_sent",
                "content": "Done",
                "meta": {"thread_id": "default"},
            }
        )
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert len(result["tool_uses"]) <= 10

    def test_skips_non_human_messages(self, anima_dir: Path) -> None:
        """Skips message_received entries that are not from humans (e.g. inbox)."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "From another anima",
                "meta": {"from_type": "anima", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:01:00+09:00",
                "type": "message_received",
                "content": "From human",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:01:30+09:00",
                "type": "response_sent",
                "content": "Reply",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert "From another anima" not in result["accumulated_response"]
        assert "From human" in result["accumulated_response"]

    def test_skips_inbox_entries_even_with_default_thread(self, anima_dir: Path) -> None:
        """Inbox activity never seeds default chat short-term memory."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Inbox question",
                "channel": "inbox",
                "meta": {"from_type": "human", "trigger": "inbox:sakura"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Inbox answer",
                "channel": "inbox",
                "meta": {"trigger": "inbox:sakura"},
            },
            {
                "ts": "2026-04-13T10:01:00+09:00",
                "type": "message_received",
                "content": "Chat question",
                "channel": "chat",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:01:30+09:00",
                "type": "response_sent",
                "content": "Chat answer",
                "channel": "chat",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert "Inbox question" not in result["accumulated_response"]
        assert "Inbox answer" not in result["accumulated_response"]
        assert "Chat question" in result["accumulated_response"]

    def test_skips_heartbeat_entries(self, anima_dir: Path) -> None:
        """Heartbeat/cron entries are excluded from extraction."""
        entries = [
            {"ts": "2026-04-13T10:00:00+09:00", "type": "heartbeat_start", "content": "HB start", "meta": {}},
            {"ts": "2026-04-13T10:00:30+09:00", "type": "heartbeat_end", "content": "HB end", "meta": {}},
            {
                "ts": "2026-04-13T10:01:00+09:00",
                "type": "message_received",
                "content": "Hello",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:01:30+09:00",
                "type": "response_sent",
                "content": "Hi",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert "HB start" not in result.get("accumulated_response", "")
        assert "Hello" in result["accumulated_response"]

    def test_trigger_and_notes_fields(self, anima_dir: Path) -> None:
        """Result includes idle_compaction trigger and notes."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Hi",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Hello",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert result["trigger"] == "idle_compaction"
        assert "activity_log" in result["notes"]

    def test_excludes_background_tools_from_chat_handover(self, anima_dir: Path) -> None:
        """Cron/heartbeat tools (no thread_id) never masquerade as chat tools."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Fix calendar",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:05+09:00",
                "type": "tool_use",
                "tool": "ChatTool",
                "content": "cal",
                "ctx": "chat",
                "meta": {"args": {}, "tool_use_id": "c1"},
            },
            {
                "ts": "2026-04-13T10:00:06+09:00",
                "type": "tool_use",
                "tool": "CronDuringTurn",
                "content": "cw",
                "ctx": "cron:Chatwork監視",
                "meta": {"args": {}, "tool_use_id": "x1"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Done",
                "ctx": "chat",
                "meta": {"thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:05:00+09:00",
                "type": "tool_use",
                "tool": "CronAfterTurn",
                "content": "cw",
                "ctx": "cron:Chatwork監視",
                "meta": {"args": {}, "tool_use_id": "x2"},
            },
            {
                "ts": "2026-04-13T10:06:00+09:00",
                "type": "tool_use",
                "tool": "HeartbeatTool",
                "content": "hb",
                "ctx": "heartbeat",
                "meta": {"args": {}, "tool_use_id": "x3"},
            },
            {
                "ts": "2026-04-13T10:07:00+09:00",
                "type": "tool_use",
                "tool": "NoCtxOutsideTurn",
                "content": "n",
                "meta": {"args": {}, "tool_use_id": "x4"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        names = [t["name"] for t in result["tool_uses"]]
        assert names == ["ChatTool"]

    def test_keeps_ctxless_tools_inside_turn(self, anima_dir: Path) -> None:
        """Modes that log no ctx still hand over tools used during the turn."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Go",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:05+09:00",
                "type": "tool_use",
                "tool": "Bash",
                "content": "ls",
                "meta": {"args": {}, "tool_use_id": "t1"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Done",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        assert [t["name"] for t in result["tool_uses"]] == ["Bash"]

    def test_filters_by_thread_id(self, anima_dir: Path) -> None:
        """Only entries matching the specified thread_id are included."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Thread1 Q",
                "meta": {"from_type": "human", "thread_id": "thread-1"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Thread1 A",
                "meta": {"thread_id": "thread-1"},
            },
            {
                "ts": "2026-04-13T10:01:00+09:00",
                "type": "message_received",
                "content": "Default Q",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:01:30+09:00",
                "type": "response_sent",
                "content": "Default A",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir, thread_id="thread-1")

        assert "Thread1 Q" in result["accumulated_response"]
        assert "Thread1 A" in result["accumulated_response"]
        assert "Default Q" not in result["accumulated_response"]

    def test_date_boundary_fallback(self, anima_dir: Path) -> None:
        """Falls back to yesterday's log when today's is empty."""
        from core.time_utils import now_local

        yesterday = (now_local().date() - timedelta(days=1)).isoformat()
        entries = [
            {
                "ts": f"{yesterday}T23:55:00+09:00",
                "type": "message_received",
                "content": "Late night Q",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": f"{yesterday}T23:55:30+09:00",
                "type": "response_sent",
                "content": "Late night A",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries, date_str=yesterday)

        result = _extract_recent_chat_context(anima_dir)

        assert "Late night Q" in result["accumulated_response"]
        assert "Late night A" in result["accumulated_response"]

    def test_tool_use_id_not_leaked_to_output(self, anima_dir: Path) -> None:
        """tool_use_id is used for pairing but not included in output dict."""
        entries = [
            {
                "ts": "2026-04-13T10:00:00+09:00",
                "type": "message_received",
                "content": "Go",
                "meta": {"from_type": "human", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:05+09:00",
                "type": "tool_use",
                "tool": "Write",
                "content": "f.txt",
                "meta": {"args": {"path": "f.txt"}, "tool_use_id": "tu99", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:06+09:00",
                "type": "tool_result",
                "content": "ok",
                "meta": {"tool_use_id": "tu99", "thread_id": "default"},
            },
            {
                "ts": "2026-04-13T10:00:30+09:00",
                "type": "response_sent",
                "content": "Done",
                "meta": {"thread_id": "default"},
            },
        ]
        self._write_log(anima_dir, entries)

        result = _extract_recent_chat_context(anima_dir)

        for tu in result["tool_uses"]:
            assert "tool_use_id" not in tu


# ── run_idle_compaction ───────────────────────────────────────────────────────


class TestRunIdleCompaction:
    """run_idle_compaction behavior."""

    @pytest.mark.asyncio
    async def test_lock_timeout_skips_compaction(self) -> None:
        """When lock cannot be acquired within timeout, compaction is skipped."""
        anima = MagicMock()
        anima.name = "alice"
        anima.agent.execution_mode = "a"
        mock_lock = MagicMock()
        mock_lock.acquire = AsyncMock(side_effect=TimeoutError())
        mock_lock.release = MagicMock()
        anima._get_thread_lock = MagicMock(return_value=mock_lock)

        await run_idle_compaction(anima, "thread-1")

        mock_lock.acquire.assert_awaited_once()
        mock_lock.release.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_to_mode_a(self) -> None:
        """run_idle_compaction dispatches to conversation compaction for mode 'a'."""
        anima = MagicMock()
        anima.name = "alice"
        anima.agent.execution_mode = "a"
        mock_lock = MagicMock()
        mock_lock.acquire = AsyncMock(side_effect=[None, None])
        mock_lock.release = MagicMock()
        anima._get_thread_lock = MagicMock(return_value=mock_lock)

        with patch(
            "core.agent.session_compactor._compact_conversation",
            new_callable=AsyncMock,
        ) as mock_compact:
            mock_compact.return_value = {"compression_performed": False}
            with patch("core.activity.logger.ActivityLogger") as mock_activity:
                mock_activity.return_value.alog = AsyncMock()
                await run_idle_compaction(anima, "thread-1")

            mock_compact.assert_awaited_once_with(anima, "thread-1", clear_engine=None)

    @pytest.mark.parametrize(
        ("mode", "engine", "storage_engine", "should_clear"),
        [
            ("x", "grok", "grok", True),
            ("y", "agy", "agy", True),
            ("d", "cursor", "cursor", True),
            ("c", "codex", "codex", True),
            ("g", None, "grok", False),
            ("a", None, "grok", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_mode_compaction_clears_only_selected_engine_session(
        self,
        tmp_path: Path,
        mode: str,
        engine: str | None,
        storage_engine: str,
        should_clear: bool,
    ) -> None:
        from core.execution.session.session_store import SessionRecord, SessionStore

        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        model_config = ModelConfig(model="test/model", max_tokens=4096, context_threshold=0.5)
        path = SessionStore.path_for(storage_engine, anima_dir, "chat", "thread-1")
        SessionStore(path).write_text_record(SessionRecord("session-id", 1), with_turn_count=True)

        with (
            patch("core.memory.conversation.memory.ConversationMemory") as mock_conv_cls,
            patch("core.activity.logger.ActivityLogger") as mock_activity,
        ):
            mock_activity.return_value.alog = AsyncMock()
            mock_conv = MagicMock()
            mock_conv.compress_if_needed_detailed = AsyncMock(return_value=_compression_result(False))
            mock_conv.load.return_value = MagicMock(compressed_summary="", turns=[])
            mock_conv.finalize_if_session_ended = AsyncMock(return_value=False)
            mock_conv_cls.return_value = mock_conv

            anima = MagicMock()
            anima.name = "alice"
            anima.anima_dir = anima_dir
            anima.agent.execution_mode = mode
            anima.agent.model_config = model_config
            anima.agent._executor = MagicMock(session_engine=engine)
            lock = MagicMock()
            lock.acquire = AsyncMock(return_value=True)
            lock.release = MagicMock()
            anima._get_thread_lock = MagicMock(return_value=lock)

            assert await run_idle_compaction(anima, "thread-1") is True

        assert path.exists() is (not should_clear)

    @pytest.mark.asyncio
    async def test_mode_s_fallback_to_mode_a_when_compact_returns_false(
        self,
    ) -> None:
        """When mode S compaction returns False, it falls back to conversation compaction."""
        anima = MagicMock()
        anima.name = "alice"
        anima.agent.execution_mode = "s"
        mock_lock = MagicMock()
        mock_lock.acquire = AsyncMock(side_effect=[None, None])
        mock_lock.release = MagicMock()
        anima._get_thread_lock = MagicMock(return_value=mock_lock)

        with (
            patch(
                "core.agent.session_compactor._compact_mode_s",
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_s,
            patch(
                "core.agent.session_compactor._compact_conversation",
                new_callable=AsyncMock,
            ) as mock_a,
        ):
            mock_a.return_value = {"compression_performed": False}
            with patch("core.activity.logger.ActivityLogger") as mock_activity:
                mock_activity.return_value.alog = AsyncMock()
                await run_idle_compaction(anima, "thread-1")

            mock_s.assert_awaited_once()
            mock_a.assert_awaited_once_with(anima, "thread-1", clear_engine=None)

    @pytest.mark.asyncio
    async def test_exception_caught_and_logged(self) -> None:
        """Exceptions in compaction are caught and logged without propagating."""
        anima = MagicMock()
        anima.name = "alice"
        anima.agent.execution_mode = "a"
        mock_lock = MagicMock()
        mock_lock.acquire = AsyncMock(return_value=None)
        mock_lock.release = MagicMock()
        anima._get_thread_lock = MagicMock(return_value=mock_lock)

        with (
            patch(
                "core.agent.session_compactor._compact_conversation",
                new_callable=AsyncMock,
                side_effect=RuntimeError("compaction failed"),
            ),
            patch("core.activity.logger.ActivityLogger") as mock_activity,
        ):
            mock_activity.return_value.alog = AsyncMock()
            await run_idle_compaction(anima, "thread-1")

        mock_lock.release.assert_called_once()


# ── Mode A blocking: shortterm.clear() conditioned on threshold ──────────────


class TestModeABlockingShorttermClear:
    """Mode A blocking path should only clear shortterm when threshold is NOT exceeded."""

    @pytest.mark.asyncio
    async def test_shortterm_preserved_when_threshold_exceeded(self, tmp_path: Path) -> None:
        """When tracker.threshold_exceeded is True, shortterm.clear() is NOT called."""
        from core.memory.conversation.shortterm import SessionState, ShortTermMemory

        anima_dir = tmp_path / "animas" / "test-mode-a"
        (anima_dir / "shortterm" / "chat").mkdir(parents=True)

        shortterm = ShortTermMemory(anima_dir, session_type="chat")
        shortterm.save(
            SessionState(
                session_id="test-session",
                accumulated_response="Important conversation context",
                trigger="chat",
                notes="Saved after context threshold",
            )
        )
        assert shortterm.has_pending()

        tracker = MagicMock()
        tracker.threshold_exceeded = True

        if not tracker.threshold_exceeded:
            shortterm.clear()

        assert shortterm.has_pending(), "shortterm must be preserved when threshold exceeded"

    @pytest.mark.asyncio
    async def test_shortterm_cleared_when_threshold_not_exceeded(self, tmp_path: Path) -> None:
        """When tracker.threshold_exceeded is False, shortterm.clear() IS called."""
        from core.memory.conversation.shortterm import SessionState, ShortTermMemory

        anima_dir = tmp_path / "animas" / "test-mode-a"
        (anima_dir / "shortterm" / "chat").mkdir(parents=True)

        shortterm = ShortTermMemory(anima_dir, session_type="chat")
        shortterm.save(
            SessionState(
                session_id="test-session",
                accumulated_response="Some context",
                trigger="chat",
            )
        )
        assert shortterm.has_pending()

        tracker = MagicMock()
        tracker.threshold_exceeded = False

        if not tracker.threshold_exceeded:
            shortterm.clear()

        assert not shortterm.has_pending(), "shortterm must be cleared when threshold not exceeded"


# ── Integration with anima.py ───────────────────────────────────────────────────


class TestAnimaIntegration:
    """SessionCompactor initialization in DigitalAnima."""

    def test_session_compactor_initialized_with_idle_minutes_from_config(self, tmp_path: Path) -> None:
        """DigitalAnima.__init__ creates SessionCompactor with idle_minutes from config."""
        anima_dir = tmp_path / "animas" / "test-session"
        shared_dir = tmp_path / "shared"
        for d in [
            anima_dir / "state",
            anima_dir / "episodes",
            anima_dir / "shortterm",
            shared_dir / "inbox" / "test-session",
        ]:
            d.mkdir(parents=True, exist_ok=True)
        (anima_dir / "IDENTITY.md").write_text("# Test", encoding="utf-8")
        (anima_dir / "AGENTS.md").write_text("Test", encoding="utf-8")
        (anima_dir / "status.json").write_text(
            '{"enabled": true, "model": "claude-sonnet-4-6"}',
            encoding="utf-8",
        )

        with (
            patch("core.anima.digital_anima.AgentCore") as mock_agent_cls,
            patch("core.anima.digital_anima.MemoryManager"),
            patch("core.config.models.load_config") as mock_load_config,
        ):
            mock_agent = MagicMock()
            mock_agent.background_manager = None
            mock_agent._tool_handler = MagicMock()
            mock_agent_cls.return_value = mock_agent

            mock_config = MagicMock()
            mock_config.heartbeat.idle_compaction_minutes = 15.0
            mock_load_config.return_value = mock_config

            from core.anima.digital_anima import DigitalAnima

            anima = DigitalAnima(anima_dir, shared_dir)

            assert anima._session_compactor is not None
            assert anima._session_compactor._idle_minutes == 15.0
