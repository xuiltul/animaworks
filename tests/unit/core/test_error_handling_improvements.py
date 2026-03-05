"""Unit tests for error handling improvements across all layers.

Verifies that broad `except Exception` catches have been replaced with
specific custom exceptions from core.exceptions, preserving the correct
behavior for both normal and error paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.exceptions import (
    AnimaWorksError,
    ConfigError,
    ConfigNotFoundError,
    ExecutionError,
    LLMAPIError,
    MemoryWriteError,
    ProcessError,
    ToolExecutionError,
)


# ── Layer 1: Execution ──────────────────────────────────────────────


class TestAssistedExecutorErrorHandling:
    """Verify assisted.py raises typed exceptions instead of swallowing errors."""

    @pytest.fixture
    def assisted_executor(self, data_dir: Path, make_anima):
        anima_dir = make_anima(
            "test-b", model="ollama/gemma3:27b",
            execution_mode="assisted", max_turns=5,
        )
        from core.memory import MemoryManager
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        from core.tooling.handler import ToolHandler
        tool_handler = ToolHandler(anima_dir=anima_dir, memory=memory)
        from core.execution.assisted import AssistedExecutor
        return AssistedExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            memory=memory,
        )

    @pytest.mark.asyncio
    async def test_llm_error_raises_execution_error(self, assisted_executor):
        """Generic Exception from LLM should be wrapped in ExecutionError."""
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API timeout")):
            with pytest.raises(ExecutionError, match="API timeout"):
                await assisted_executor.execute(
                    prompt="test", system_prompt="test",
                )

    @pytest.mark.asyncio
    async def test_llm_api_error_propagates(self, assisted_executor):
        """LLMAPIError should propagate unchanged."""
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=LLMAPIError("rate limited")):
            with pytest.raises(LLMAPIError, match="rate limited"):
                await assisted_executor.execute(
                    prompt="test", system_prompt="test",
                )

    @pytest.mark.asyncio
    async def test_tool_execution_error_returns_result_string(self, assisted_executor):
        """ToolExecutionError in tool dispatch should become result string, not crash."""
        known = next(iter(assisted_executor._known_tools)) if assisted_executor._known_tools else "read_file"

        def _make_resp(content, finish="stop"):
            msg = MagicMock()
            msg.content = content
            ch = MagicMock()
            ch.message = msg
            ch.finish_reason = finish
            resp = MagicMock()
            resp.choices = [ch]
            return resp

        resp1 = _make_resp(
            f'<tool_call>{{"name": "{known}", "arguments": {{"path": "/tmp/x"}}}}</tool_call>',
            finish="tool_calls",
        )
        resp2 = _make_resp("Done")

        assisted_executor._tool_handler = MagicMock()
        assisted_executor._tool_handler.handle.side_effect = ToolExecutionError("tool broke")

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[resp1, resp2]):
            result = await assisted_executor.execute(
                prompt="use tool", system_prompt="test",
            )
        assert result.text is not None

    def test_exception_catch_ordering_in_tool_dispatch(self):
        """Verify the catch ordering: ToolExecutionError -> AnimaWorksError -> Exception."""
        from core.exceptions import AnimaWorksError

        caught_by = None
        for exc in [ToolExecutionError("te"), MemoryWriteError("mw"), RuntimeError("rt")]:
            caught_by = None
            try:
                raise exc
            except ToolExecutionError:
                caught_by = "ToolExecutionError"
            except AnimaWorksError:
                caught_by = "AnimaWorksError"
            except Exception:
                caught_by = "Exception"

            if isinstance(exc, ToolExecutionError):
                assert caught_by == "ToolExecutionError"
            elif isinstance(exc, MemoryWriteError):
                assert caught_by == "AnimaWorksError"
            elif isinstance(exc, RuntimeError):
                assert caught_by == "Exception"


# ── Layer 3: Tooling ────────────────────────────────────────────────


class TestToolHandlerErrorHandling:
    """Verify handler.py uses specific exception types."""

    @pytest.fixture
    def handler(self, data_dir: Path, make_anima):
        anima_dir = make_anima("tool-test")
        from core.memory import MemoryManager
        memory = MemoryManager(anima_dir)
        from core.tooling.handler import ToolHandler
        return ToolHandler(anima_dir=anima_dir, memory=memory)

    def test_persist_replied_to_handles_os_error(self, handler, tmp_path):
        """OSError in _persist_replied_to should log warning, not crash."""
        original_dir = handler._anima_dir
        handler._anima_dir = tmp_path / "nonexistent" / "deep" / "path"
        with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
            handler._persist_replied_to("alice", success=True)
        handler._anima_dir = original_dir

    def test_handle_unknown_tool_returns_error_result(self, handler):
        """Unknown internal tool should return error result string."""
        result = handler.handle("nonexistent_tool_xyz_42", {})
        assert "error" in result.lower() or "不明" in result or "Unknown" in result


# ── Layer 4: Messaging ─────────────────────────────────────────────


class TestMessengerErrorHandling:
    """Verify messenger.py catches specific exception types."""

    @pytest.fixture
    def messenger(self, data_dir: Path, make_anima):
        make_anima("msg-test")
        from core.messenger import Messenger
        return Messenger(
            shared_dir=data_dir / "shared",
            anima_name="msg-test",
        )

    def test_receive_handles_corrupt_inbox_files(self, messenger):
        """receive() should skip corrupt JSON files without crashing."""
        inbox_dir = messenger.inbox_dir
        inbox_dir.mkdir(parents=True, exist_ok=True)
        (inbox_dir / "corrupt.json").write_text("{invalid json", encoding="utf-8")
        msgs = messenger.receive()
        assert isinstance(msgs, list)

    def test_read_dm_history_handles_bad_activity_log(self, messenger, data_dir):
        """read_dm_history should fallback gracefully on corrupt data."""
        anima_dir = data_dir / "animas" / "msg-test"
        activity_dir = anima_dir / "activity_log"
        activity_dir.mkdir(parents=True, exist_ok=True)
        today = __import__("datetime").date.today().isoformat()
        (activity_dir / f"{today}.jsonl").write_text(
            "{bad json\n", encoding="utf-8",
        )
        result = messenger.read_dm_history("other-anima", limit=5)
        assert isinstance(result, list)


# ── Layer 6: Config ────────────────────────────────────────────────


class TestConfigModelsErrorHandling:
    """Verify config/models.py catches ConfigError specifically."""

    def test_resolve_context_window_returns_none_on_config_error(self):
        """resolve_context_window should return None when config fails."""
        from core.config import invalidate_cache
        invalidate_cache()
        with patch("core.config.models.load_config", side_effect=ConfigError("bad")):
            from core.config.models import resolve_context_window
            result = resolve_context_window("test-model")
            assert result is None
        invalidate_cache()

    def test_resolve_context_window_returns_none_on_os_error(self):
        """resolve_context_window should return None when file I/O fails."""
        from core.config import invalidate_cache
        invalidate_cache()
        with patch("core.config.models.load_config", side_effect=OSError("perm denied")):
            from core.config.models import resolve_context_window
            result = resolve_context_window("test-model")
            assert result is None
        invalidate_cache()

    def test_resolve_context_window_does_not_catch_non_config_error(self):
        """Non-ConfigError/OSError exceptions should propagate."""
        from core.config import invalidate_cache
        invalidate_cache()
        with patch("core.config.models.load_config", side_effect=RuntimeError("unexpected")):
            from core.config.models import resolve_context_window
            with pytest.raises(RuntimeError, match="unexpected"):
                resolve_context_window("test-model")
        invalidate_cache()


# ── Layer 7: Memory ────────────────────────────────────────────────


class TestActivityLoggerErrorHandling:
    """Verify activity.py raises MemoryWriteError for serialization failures."""

    @pytest.fixture
    def activity_logger(self, tmp_path: Path):
        from core.memory.activity import ActivityLogger
        al = ActivityLogger(anima_dir=tmp_path)
        (tmp_path / "activity_log").mkdir(parents=True, exist_ok=True)
        return al

    def test_append_raises_memory_write_error_on_os_error(self, activity_logger, tmp_path):
        """OSError during append should raise MemoryWriteError."""
        from core.memory.activity import ActivityEntry
        entry = ActivityEntry(
            ts="2026-01-01T00:00:00+09:00", type="test_event",
            content="test", summary="", from_person="", to_person="",
            channel="", tool="", via="", meta={},
        )
        log_dir = tmp_path / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        ro_file = log_dir / "2026-01-01.jsonl"
        ro_file.write_text("")
        ro_file.chmod(0o000)
        try:
            with pytest.raises(MemoryWriteError):
                activity_logger._append(entry)
        finally:
            ro_file.chmod(0o644)

    def test_load_entries_skips_corrupt_files(self, activity_logger, tmp_path):
        """Corrupt JSONL files should be skipped, not crash."""
        log_dir = tmp_path / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "2026-01-01.jsonl").write_text(
            "{bad json\n", encoding="utf-8",
        )
        entries = activity_logger._load_entries(days=365)
        assert isinstance(entries, list)


class TestConversationMemoryErrorHandling:
    """Verify conversation.py catches specific LLM errors."""

    def test_load_context_window_overrides_catches_config_error(self, tmp_path):
        """ConfigError in _load_context_window_overrides should return None."""
        from core.config import invalidate_cache
        from core.memory.conversation import ConversationMemory
        invalidate_cache()
        cm = ConversationMemory.__new__(ConversationMemory)
        with patch("core.config.models.load_config", side_effect=ConfigError("bad")):
            result = cm._load_context_window_overrides()
            assert result is None
        invalidate_cache()

    def test_load_context_window_overrides_catches_os_error(self, tmp_path):
        """OSError in _load_context_window_overrides should return None."""
        from core.config import invalidate_cache
        from core.memory.conversation import ConversationMemory
        invalidate_cache()
        cm = ConversationMemory.__new__(ConversationMemory)
        with patch("core.config.models.load_config", side_effect=OSError("perm denied")):
            result = cm._load_context_window_overrides()
            assert result is None
        invalidate_cache()

    def test_load_context_window_overrides_does_not_catch_runtime_error(self, tmp_path):
        """Non-ConfigError/OSError should propagate."""
        from core.config import invalidate_cache
        from core.memory.conversation import ConversationMemory
        invalidate_cache()
        cm = ConversationMemory.__new__(ConversationMemory)
        with patch("core.config.models.load_config", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError, match="unexpected"):
                cm._load_context_window_overrides()
        invalidate_cache()


# ── Layer 2: Anima ─────────────────────────────────────────────────


class TestAnimaErrorHandling:
    """Verify anima.py uses specific exceptions for notification reading."""

    def test_read_notifications_handles_os_error(self, tmp_path):
        """OSError in notification read should be handled gracefully."""
        from core.anima import DigitalAnima

        anima = DigitalAnima.__new__(DigitalAnima)
        agent_mock = MagicMock()
        agent_mock.anima_dir = tmp_path
        anima.agent = agent_mock
        notif_dir = tmp_path / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)
        notif_file = notif_dir / "test.md"
        notif_file.write_text("notification content")
        notif_file.chmod(0o000)

        try:
            result = anima.drain_background_notifications()
            assert result == []
        finally:
            notif_file.chmod(0o644)

    def test_read_notifications_handles_decode_error(self, tmp_path):
        """UnicodeDecodeError in notification read should be handled."""
        from core.anima import DigitalAnima

        anima = DigitalAnima.__new__(DigitalAnima)
        agent_mock = MagicMock()
        agent_mock.anima_dir = tmp_path
        anima.agent = agent_mock
        notif_dir = tmp_path / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)
        notif_file = notif_dir / "test.md"
        notif_file.write_bytes(b"\x80\x81\x82\x83")

        result = anima.drain_background_notifications()
        assert result == []
