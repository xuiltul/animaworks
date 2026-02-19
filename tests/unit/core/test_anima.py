"""Unit tests for core/anima.py — DigitalAnima entity."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.schemas import CycleResult, AnimaStatus


# ── Helpers ───────────────────────────────────────────────


def _make_cycle_result(**kwargs) -> CycleResult:
    defaults = dict(trigger="test", action="responded", summary="done", duration_ms=100)
    defaults.update(kwargs)
    return CycleResult(**defaults)


# ── DigitalAnima construction ────────────────────────────


class TestDigitalAnimaInit:
    def test_init(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMessenger:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            assert dp.name == "alice"
            assert dp.anima_dir == anima_dir
            assert dp._status == "idle"
            assert dp._current_task == ""
            assert dp._last_heartbeat is None
            assert dp._last_activity is None


class TestDigitalAnimaStatus:
    def test_status_property(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMessenger:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMessenger.return_value.unread_count.return_value = 5
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            status = dp.status
            assert isinstance(status, AnimaStatus)
            assert status.name == "alice"
            assert status.status == "idle"
            assert status.pending_messages == 5


class TestNeedsBootstrap:
    def test_needs_bootstrap_true(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        (anima_dir / "bootstrap.md").write_text("bootstrap", encoding="utf-8")

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            assert dp.needs_bootstrap is True

    def test_needs_bootstrap_false(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        # Ensure no bootstrap.md
        bp = anima_dir / "bootstrap.md"
        if bp.exists():
            bp.unlink()

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            assert dp.needs_bootstrap is False


# ── Callbacks ─────────────────────────────────────────────


class TestCallbacks:
    def test_set_on_message_sent(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            fn = MagicMock()
            dp.set_on_message_sent(fn)
            dp.agent.set_on_message_sent.assert_called_once_with(fn)

    def test_set_on_lock_released(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            fn = MagicMock()
            dp.set_on_lock_released(fn)
            assert dp._on_lock_released is fn


class TestNotifyLockReleased:
    def test_calls_callback(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            fn = MagicMock()
            dp._on_lock_released = fn
            dp._notify_lock_released()
            fn.assert_called_once()

    def test_no_callback(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp._on_lock_released = None
            dp._notify_lock_released()  # should not raise

    def test_exception_in_callback_is_caught(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp._on_lock_released = MagicMock(side_effect=RuntimeError("boom"))
            dp._notify_lock_released()  # should not raise


# ── process_message ───────────────────────────────────────


class TestProcessMessage:
    async def test_process_message_returns_summary(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
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
            dp.agent.run_cycle = AsyncMock(return_value=_make_cycle_result(summary="Hello!"))

            result = await dp.process_message("Hi", from_person="human")
            assert result == "Hello!"
            assert dp._status == "idle"

    async def test_status_transitions(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
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

            observed_statuses = []

            async def mock_run_cycle(prompt, trigger="manual", **kwargs):
                observed_statuses.append(dp._status)
                return _make_cycle_result()

            dp.agent.run_cycle = mock_run_cycle
            await dp.process_message("test")
            assert "thinking" in observed_statuses
            assert dp._status == "idle"

    async def test_exception_resets_status(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.compress_if_needed = AsyncMock()
            MockConv.return_value.finalize_session = AsyncMock(return_value=False)
            MockConv.return_value.build_chat_prompt.return_value = "prompt"

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(side_effect=RuntimeError("fail"))

            with pytest.raises(RuntimeError):
                await dp.process_message("test")
            assert dp._status == "idle"


# ── run_heartbeat ─────────────────────────────────────────


class TestRunHeartbeat:
    async def test_run_heartbeat(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMsg.return_value.has_unread.return_value = False

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(return_value=_make_cycle_result())
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            result = await dp.run_heartbeat()
            assert isinstance(result, CycleResult)
            assert dp._last_heartbeat is not None
            assert dp._status == "idle"


# ── run_cron_task ─────────────────────────────────────────


class TestRunCronTask:
    async def test_run_cron_task(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.load_prompt", return_value="cron prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(return_value=_make_cycle_result())

            result = await dp.run_cron_task("daily_report", "Generate report")
            assert isinstance(result, CycleResult)
            assert dp._status == "idle"


# ── process_greet ────────────────────────────────────────


class TestProcessGreet:
    async def test_greet_returns_response(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(
                return_value=_make_cycle_result(
                    summary='こんにちは！今は待機中です。<!-- emotion: {"emotion": "smile"} -->'
                )
            )

            result = await dp.process_greet()
            assert result["response"] == "こんにちは！今は待機中です。"
            assert result["emotion"] == "smile"
            assert result["cached"] is False

    async def test_greet_caches_response(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(
                return_value=_make_cycle_result(summary="Hello!")
            )

            # First call
            result1 = await dp.process_greet()
            assert result1["cached"] is False

            # Second call within cooldown
            result2 = await dp.process_greet()
            assert result2["cached"] is True
            assert result2["response"] == result1["response"]
            # LLM should only be called once
            assert dp.agent.run_cycle.await_count == 1

    async def test_greet_cache_expires(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(
                return_value=_make_cycle_result(summary="Hello!")
            )

            # First call
            await dp.process_greet()

            # Simulate cache expiry (must exceed _GREET_COOLDOWN of 3600s)
            dp._last_greet_at = time.time() - 3601

            # Second call after expiry
            result = await dp.process_greet()
            assert result["cached"] is False
            assert dp.agent.run_cycle.await_count == 2

    async def test_greet_records_visit_and_assistant_turns(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(
                return_value=_make_cycle_result(summary="Hi there!")
            )

            await dp.process_greet()

            # Should record visit marker (system) + assistant turn
            from unittest.mock import call
            MockConv.return_value.append_turn.assert_has_calls([
                call("system", "[デスクを訪問]"),
                call("assistant", "Hi there!"),
            ])
            assert MockConv.return_value.append_turn.call_count == 2
            assert MockConv.return_value.save.call_count == 2

    async def test_greet_restores_previous_status(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            # Set pre-existing status
            dp._status = "working"
            dp._current_task = "Report generation"

            observed_statuses = []

            async def mock_run_cycle(prompt, trigger="manual", **kwargs):
                observed_statuses.append(dp._status)
                return _make_cycle_result(summary="Hello!")

            dp.agent.run_cycle = mock_run_cycle

            await dp.process_greet()

            # During greet, status should be "greeting"
            assert "greeting" in observed_statuses
            # After greet, status should be restored
            assert dp._status == "working"
            assert dp._current_task == "Report generation"

    async def test_greet_emotion_fallback(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            # No emotion tag in response
            dp.agent.run_cycle = AsyncMock(
                return_value=_make_cycle_result(summary="Plain greeting")
            )

            result = await dp.process_greet()
            assert result["emotion"] == "neutral"

    async def test_greet_exception_restores_status(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp._status = "working"
            dp._current_task = "Some task"
            dp.agent.run_cycle = AsyncMock(side_effect=RuntimeError("fail"))

            with pytest.raises(RuntimeError):
                await dp.process_greet()

            # Status should be restored even after exception
            assert dp._status == "working"
            assert dp._current_task == "Some task"


# ── Conversation data-loss fix tests ─────────────────────


class TestProcessMessageConversationSave:
    """Tests that process_message pre-saves user input and handles errors."""

    async def test_user_input_saved_before_agent_execution(
        self, data_dir, make_anima,
    ):
        """append_turn('human', ...) and save() must be called BEFORE agent.run_cycle()."""
        anima_dir = make_anima("alice")
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

            # Track call order via a shared list
            call_order: list[str] = []
            MockConv.return_value.append_turn.side_effect = (
                lambda role, text, **kwargs: call_order.append(f"append_turn:{role}")
            )
            MockConv.return_value.save.side_effect = (
                lambda: call_order.append("save")
            )

            async def mock_run_cycle(prompt, trigger="manual", **kwargs):
                call_order.append("run_cycle")
                return _make_cycle_result(summary="OK")

            dp.agent.run_cycle = mock_run_cycle

            await dp.process_message("Hello", from_person="human")

            # Verify pre-save ordering: human turn + save happen before run_cycle
            assert call_order.index("append_turn:human") < call_order.index("run_cycle")
            assert call_order.index("save") < call_order.index("run_cycle")

    async def test_error_saves_user_input_and_error_marker(
        self, data_dir, make_anima,
    ):
        """When agent.run_cycle() raises, both user input and error marker are saved."""
        anima_dir = make_anima("alice")
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
            dp.agent.run_cycle = AsyncMock(side_effect=RuntimeError("boom"))

            with pytest.raises(RuntimeError):
                await dp.process_message("Hi there", from_person="human")

            # Verify human turn was recorded
            append_calls = MockConv.return_value.append_turn.call_args_list
            assert any(
                c.args == ("human", "Hi there") or c.kwargs == {}
                and c.args == ("human", "Hi there")
                for c in append_calls
            ), f"Expected append_turn('human', 'Hi there'), got {append_calls}"

            # Verify error marker was recorded
            assert any(
                c.args == ("assistant", "[ERROR: エージェント実行中にエラーが発生しました]")
                for c in append_calls
            ), f"Expected error marker append_turn call, got {append_calls}"

            # save() called at least twice: pre-save + error save
            assert MockConv.return_value.save.call_count >= 2

    async def test_success_saves_both_turns(self, data_dir, make_anima):
        """On success, both human and assistant turns are saved (regression test)."""
        anima_dir = make_anima("alice")
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
            dp.agent.run_cycle = AsyncMock(
                return_value=_make_cycle_result(summary="Great answer")
            )

            result = await dp.process_message("Question", from_person="human")
            assert result == "Great answer"

            append_calls = MockConv.return_value.append_turn.call_args_list
            roles = [c.args[0] for c in append_calls]
            contents = [c.args[1] for c in append_calls]

            assert ("human", "Question") == (roles[0], contents[0])
            assert ("assistant", "Great answer") == (roles[1], contents[1])

            # save() called twice: pre-save + success save
            assert MockConv.return_value.save.call_count == 2


class TestProcessMessageStreamConversationSave:
    """Tests that process_message_stream pre-saves user input and handles errors."""

    async def test_stream_user_input_saved_before_streaming(
        self, data_dir, make_anima,
    ):
        """append_turn('human', ...) and save() called before streaming starts."""
        anima_dir = make_anima("alice")
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

            call_order: list[str] = []
            MockConv.return_value.append_turn.side_effect = (
                lambda role, text, **kwargs: call_order.append(f"append_turn:{role}")
            )
            MockConv.return_value.save.side_effect = (
                lambda: call_order.append("save")
            )

            async def mock_stream(prompt, trigger="manual", **kwargs):
                call_order.append("stream_start")
                yield {"type": "text_delta", "text": "Hello"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "Hello"},
                }

            dp.agent.run_cycle_streaming = mock_stream

            chunks = []
            async for chunk in dp.process_message_stream("Hi", from_person="human"):
                chunks.append(chunk)

            # Pre-save must happen before streaming begins
            assert call_order.index("append_turn:human") < call_order.index("stream_start")
            assert call_order.index("save") < call_order.index("stream_start")

    async def test_stream_error_saves_partial_response(
        self, data_dir, make_anima,
    ):
        """When streaming raises after some text_delta chunks, partial response + error marker is saved."""
        anima_dir = make_anima("alice")
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

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "text_delta", "text": "partial "}
                yield {"type": "text_delta", "text": "response"}
                raise RuntimeError("stream failed")

            dp.agent.run_cycle_streaming = mock_stream

            chunks = []
            async for chunk in dp.process_message_stream("Hi", from_person="human"):
                chunks.append(chunk)

            # Verify error event was yielded
            assert any(c.get("type") == "error" for c in chunks)

            # Verify the assistant error turn includes partial response + error marker
            append_calls = MockConv.return_value.append_turn.call_args_list
            assistant_calls = [c for c in append_calls if c.args[0] == "assistant"]
            assert len(assistant_calls) == 1
            error_content = assistant_calls[0].args[1]
            assert "partial response" in error_content
            assert "[応答が中断されました]" in error_content

            # save() called at least twice: pre-save + error save
            assert MockConv.return_value.save.call_count >= 2

    async def test_stream_error_without_partial_saves_error_only(
        self, data_dir, make_anima,
    ):
        """When streaming raises immediately (no text_delta), error marker alone is saved."""
        anima_dir = make_anima("alice")
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

            async def mock_stream(prompt, trigger="manual", **kwargs):
                raise RuntimeError("immediate failure")
                yield  # noqa: unreachable — makes this an async generator

            dp.agent.run_cycle_streaming = mock_stream

            chunks = []
            async for chunk in dp.process_message_stream("Hi", from_person="human"):
                chunks.append(chunk)

            # Verify error event was yielded
            assert any(c.get("type") == "error" for c in chunks)

            # Verify assistant turn has error marker only (no partial response prefix)
            append_calls = MockConv.return_value.append_turn.call_args_list
            assistant_calls = [c for c in append_calls if c.args[0] == "assistant"]
            assert len(assistant_calls) == 1
            error_content = assistant_calls[0].args[1]
            assert error_content == "[応答が中断されました]"

    async def test_stream_success_saves_normally(self, data_dir, make_anima):
        """Normal streaming save still works (cycle_done path)."""
        anima_dir = make_anima("alice")
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

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "text_delta", "text": "Full "}
                yield {"type": "text_delta", "text": "response"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "Full response"},
                }

            dp.agent.run_cycle_streaming = mock_stream

            chunks = []
            async for chunk in dp.process_message_stream("Hi", from_person="human"):
                chunks.append(chunk)

            # Verify cycle_done was yielded
            assert any(c.get("type") == "cycle_done" for c in chunks)

            # Verify both human and assistant turns saved
            append_calls = MockConv.return_value.append_turn.call_args_list
            roles = [c.args[0] for c in append_calls]
            assert roles == ["human", "assistant"]

            # Human turn pre-saved, assistant turn saved on cycle_done
            assert append_calls[0].args == ("human", "Hi")
            assert append_calls[1].args == ("assistant", "Full response")

            # save() called twice: pre-save + cycle_done save
            assert MockConv.return_value.save.call_count == 2


class TestProcessGreetConversationSave:
    """Tests that process_greet saves error markers on failure."""

    async def test_greet_error_saves_error_marker(self, data_dir, make_anima):
        """When agent.run_cycle() raises during greet, visit marker + error marker are saved."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="greet prompt"):
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockConv.return_value.append_turn = MagicMock()
            MockConv.return_value.save = MagicMock()

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.run_cycle = AsyncMock(side_effect=RuntimeError("greet failed"))

            with pytest.raises(RuntimeError):
                await dp.process_greet()

            # Verify visit marker + error marker were saved
            from unittest.mock import call
            MockConv.return_value.append_turn.assert_has_calls([
                call("system", "[デスクを訪問]"),
                call("assistant", "[ERROR: 挨拶生成中にエラーが発生しました]"),
            ])
            assert MockConv.return_value.append_turn.call_count == 2
            assert MockConv.return_value.save.call_count == 2
