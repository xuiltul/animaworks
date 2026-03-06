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

from core.config.models import HeartbeatConfig
from core.execution._sdk_session import (
    SESSION_RESUME_TIMEOUT_MIN,
    _load_session_id,
)
from core.execution.codex_sdk import CodexSDKExecutor
from core.schemas import ModelConfig
from core.session_compactor import (
    SessionCompactor,
    _compact_mode_a,
    _compact_mode_b,
    _compact_mode_c,
    _compact_mode_s,
    run_idle_compaction,
)

# ── SessionCompactor ──────────────────────────────────────────────────────


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
        with patch("core.session_compactor._MAX_TIMERS", 3):
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


# ── _load_session_id skip_timeout_check ──────────────────────────────────────


class TestLoadSessionIdSkipTimeoutCheck:
    """_load_session_id with skip_timeout_check parameter."""

    def test_skip_timeout_check_true_returns_old_session(self, tmp_path: Path) -> None:
        """When skip_timeout_check=True, session_id returned even when older than threshold."""
        anima_dir = tmp_path / "animas" / "test-sdk"
        (anima_dir / "state").mkdir(parents=True)
        path = anima_dir / "state" / "current_session_chat.json"
        old_ts = (datetime.now(UTC) - timedelta(minutes=SESSION_RESUME_TIMEOUT_MIN + 10)).isoformat()
        path.write_text(
            json.dumps({"session_id": "sess-old", "timestamp": old_ts}),
            encoding="utf-8",
        )

        result = _load_session_id(anima_dir, "chat", skip_timeout_check=True)

        assert result == "sess-old"

    def test_skip_timeout_check_false_returns_none_for_old_session(self, tmp_path: Path) -> None:
        """When skip_timeout_check=False (default), old session returns None."""
        anima_dir = tmp_path / "animas" / "test-sdk"
        (anima_dir / "state").mkdir(parents=True)
        path = anima_dir / "state" / "current_session_chat.json"
        old_ts = (datetime.now(UTC) - timedelta(minutes=SESSION_RESUME_TIMEOUT_MIN + 5)).isoformat()
        path.write_text(
            json.dumps({"session_id": "sess-old", "timestamp": old_ts}),
            encoding="utf-8",
        )

        result = _load_session_id(anima_dir, "chat", skip_timeout_check=False)

        assert result is None


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


# ── CodexSDKExecutor.discard_thread ───────────────────────────────────────────


class TestCodexSDKExecutorDiscardThread:
    """CodexSDKExecutor.discard_thread deletes thread file."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test-codex"
        d.mkdir(parents=True)
        (d / "shortterm" / "chat").mkdir(parents=True)
        (d / "shortterm" / "heartbeat").mkdir(parents=True)
        return d

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            model="codex/o4-mini",
            max_tokens=4096,
            credential="openai",
            api_key="test-key",
        )

    def test_discard_thread_deletes_file(self, anima_dir: Path, model_config: ModelConfig) -> None:
        """discard_thread() removes the thread ID file."""
        thread_file = anima_dir / "shortterm" / "chat" / "codex_thread_id.txt"
        thread_file.parent.mkdir(parents=True, exist_ok=True)
        thread_file.write_text("thread-abc-123", encoding="utf-8")
        assert thread_file.exists()

        executor = CodexSDKExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
        )
        executor.discard_thread(session_type="chat")

        assert not thread_file.exists()

    def test_discard_thread_with_custom_thread_id(self, anima_dir: Path, model_config: ModelConfig) -> None:
        """discard_thread() with chat_thread_id deletes per-thread file."""
        thread_dir = anima_dir / "shortterm" / "chat" / "my-thread"
        thread_dir.mkdir(parents=True)
        thread_file = thread_dir / "codex_thread_id.txt"
        thread_file.write_text("thread-xyz", encoding="utf-8")
        assert thread_file.exists()

        executor = CodexSDKExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
        )
        executor.discard_thread(session_type="chat", chat_thread_id="my-thread")

        assert not thread_file.exists()


# ── Mode-specific compaction ─────────────────────────────────────────────────────


class TestModeSpecificCompaction:
    """Mode-specific compaction functions."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test"
        d.mkdir(parents=True)
        for sub in ["state", "episodes", "knowledge", "shortterm", "activity_log"]:
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "identity.md").write_text("# Test", encoding="utf-8")
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
    async def test_compact_mode_a_calls_compress_and_finalize(self, anima_dir: Path, model_config: ModelConfig) -> None:
        """_compact_mode_a calls compress_if_needed and finalize_if_session_ended."""
        with patch("core.memory.conversation.ConversationMemory") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv.compress_if_needed = AsyncMock(return_value=True)
            mock_conv.finalize_if_session_ended = AsyncMock()
            mock_conv_cls.return_value = mock_conv

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            result = await _compact_mode_a(anima, "default")

            mock_conv.compress_if_needed.assert_awaited_once()
            mock_conv.finalize_if_session_ended.assert_awaited_once()
            assert result is True

    @pytest.mark.asyncio
    async def test_compact_mode_b_delegates_to_mode_a(self, anima_dir: Path, model_config: ModelConfig) -> None:
        """_compact_mode_b delegates to _compact_mode_a."""
        with patch("core.session_compactor._compact_mode_a", new_callable=AsyncMock) as mock_a:
            mock_a.return_value = True

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            result = await _compact_mode_b(anima, "default")

            mock_a.assert_awaited_once_with(anima, "default")
            assert result is True

    @pytest.mark.asyncio
    async def test_compact_mode_c_calls_compress_shortterm_clear_finalize(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """_compact_mode_c calls compress, shortterm save, clear_thread_id, finalize."""
        with (
            patch("core.memory.conversation.ConversationMemory") as mock_conv_cls,
            patch("core.execution.codex_sdk._clear_thread_id") as mock_clear,
            patch("core.memory.shortterm.ShortTermMemory") as mock_stm_cls,
        ):
            mock_conv = MagicMock()
            mock_conv.compress_if_needed = AsyncMock()
            mock_conv.load.return_value = MagicMock(
                compressed_summary="",
                turns=[],
            )
            mock_conv.finalize_if_session_ended = AsyncMock()
            mock_conv_cls.return_value = mock_conv

            mock_stm = MagicMock()
            mock_stm_cls.return_value = mock_stm

            anima = MagicMock()
            anima.anima_dir = anima_dir
            anima.agent.model_config = model_config

            result = await _compact_mode_c(anima, "default")

            mock_conv.compress_if_needed.assert_awaited_once()
            mock_stm.save.assert_called_once()
            mock_clear.assert_called_once_with(anima_dir, "chat", "default")
            mock_conv.finalize_if_session_ended.assert_awaited_once()
            assert result is True

    @pytest.mark.asyncio
    async def test_compact_mode_s_delegates_to_executor_compact_session(self, anima_dir: Path) -> None:
        """_compact_mode_s calls executor.compact_session when available."""
        mock_executor = AsyncMock()
        mock_executor.compact_session = AsyncMock(return_value=True)

        anima = MagicMock()
        anima.anima_dir = anima_dir
        anima.agent._executor = mock_executor

        result = await _compact_mode_s(anima, "default")

        mock_executor.compact_session.assert_awaited_once_with(
            anima_dir=anima_dir,
            session_type="chat",
            thread_id="default",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_compact_mode_s_returns_false_when_no_compact_session(self, anima_dir: Path) -> None:
        """_compact_mode_s returns False when executor has no compact_session."""
        mock_executor = MagicMock(spec=[])  # no compact_session attr

        anima = MagicMock()
        anima.agent._executor = mock_executor

        result = await _compact_mode_s(anima, "default")

        assert result is False


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
        """run_idle_compaction dispatches to _compact_mode_a for mode 'a'."""
        anima = MagicMock()
        anima.name = "alice"
        anima.agent.execution_mode = "a"
        mock_lock = MagicMock()
        mock_lock.acquire = AsyncMock(side_effect=[None, None])
        mock_lock.release = MagicMock()
        anima._get_thread_lock = MagicMock(return_value=mock_lock)

        with patch(
            "core.session_compactor._compact_mode_a",
            new_callable=AsyncMock,
        ) as mock_compact:
            with patch("core.memory.activity.ActivityLogger"):
                await run_idle_compaction(anima, "thread-1")

            mock_compact.assert_awaited_once_with(anima, "thread-1")

    @pytest.mark.asyncio
    async def test_mode_s_fallback_to_mode_a_when_compact_returns_false(
        self,
    ) -> None:
        """When mode S and _compact_mode_s returns False, falls back to _compact_mode_a."""
        anima = MagicMock()
        anima.name = "alice"
        anima.agent.execution_mode = "s"
        mock_lock = MagicMock()
        mock_lock.acquire = AsyncMock(side_effect=[None, None])
        mock_lock.release = MagicMock()
        anima._get_thread_lock = MagicMock(return_value=mock_lock)

        with (
            patch(
                "core.session_compactor._compact_mode_s",
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_s,
            patch(
                "core.session_compactor._compact_mode_a",
                new_callable=AsyncMock,
            ) as mock_a,
        ):
            with patch("core.memory.activity.ActivityLogger"):
                await run_idle_compaction(anima, "thread-1")

            mock_s.assert_awaited_once()
            mock_a.assert_awaited_once_with(anima, "thread-1")

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
                "core.session_compactor._compact_mode_a",
                new_callable=AsyncMock,
                side_effect=RuntimeError("compaction failed"),
            ),
            patch("core.memory.activity.ActivityLogger"),
        ):
            await run_idle_compaction(anima, "thread-1")

        mock_lock.release.assert_called_once()


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
        (anima_dir / "identity.md").write_text("# Test", encoding="utf-8")
        (anima_dir / "injection.md").write_text("Test", encoding="utf-8")
        (anima_dir / "status.json").write_text(
            '{"enabled": true, "model": "claude-sonnet-4-6"}',
            encoding="utf-8",
        )

        with (
            patch("core.anima.AgentCore") as mock_agent_cls,
            patch("core.anima.MemoryManager"),
            patch("core.config.models.load_config") as mock_load_config,
        ):
            mock_agent = MagicMock()
            mock_agent.background_manager = None
            mock_agent._tool_handler = MagicMock()
            mock_agent_cls.return_value = mock_agent

            mock_config = MagicMock()
            mock_config.heartbeat.idle_compaction_minutes = 15.0
            mock_load_config.return_value = mock_config

            from core.anima import DigitalAnima

            anima = DigitalAnima(anima_dir, shared_dir)

            assert anima._session_compactor is not None
            assert anima._session_compactor._idle_minutes == 15.0
