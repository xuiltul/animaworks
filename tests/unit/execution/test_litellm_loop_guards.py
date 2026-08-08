"""Fault-injection tests for Mode A loop guards (retry / empty / runaway)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from core.exceptions import LLMAPIError
from core.execution.reminder import (
    msg_empty_response,
    msg_tool_loop_halt,
    msg_tool_loop_warning,
)
from core.schemas import ModelConfig
from core.tooling.handler import ToolHandler
from tests.helpers.mocks import make_litellm_response, make_tool_call


class _RateLimitError(Exception):
    status_code = 429


class _AuthError(Exception):
    status_code = 401


@pytest.fixture(autouse=True)
def _fast_backoff():
    """Zero backoff so retry tests do not sleep."""
    with patch("core.execution.litellm_loop.decorrelated_jitter", return_value=0.0):
        yield


@pytest.fixture(autouse=True)
def _mock_rate_guard():
    guard = MagicMock()
    guard.blocked_remaining.return_value = 0.0
    guard.config.default_block_seconds = 60
    with patch("core.execution.litellm_loop.get_rate_guard", return_value=guard):
        yield guard


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    (d / "identity.md").write_text("# Test", encoding="utf-8")
    for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm"]:
        (d / sub).mkdir(exist_ok=True)
    return d


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-4o",
        api_key="sk-test",
        max_tokens=1024,
        context_threshold=0.50,
        max_chains=2,
    )


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    from core.memory import MemoryManager

    m = MagicMock(spec=MemoryManager)
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    m.anima_dir = anima_dir
    return m


@pytest.fixture
def tool_handler(anima_dir: Path, memory: MagicMock) -> ToolHandler:
    return ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])


def _make_executor(model_config, anima_dir, tool_handler, memory, interrupt_event=None):
    from core.execution.litellm_loop import LiteLLMExecutor

    return LiteLLMExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
        interrupt_event=interrupt_event,
    )


@pytest.fixture
def executor(model_config, anima_dir, tool_handler, memory):
    return _make_executor(model_config, anima_dir, tool_handler, memory)


def _messages_of_call(mock: AsyncMock, index: int) -> list[dict]:
    return mock.call_args_list[index].kwargs["messages"]


# ── In-loop retry ────────────────────────────────────────────


class TestInLoopRetry:
    async def test_transient_429_mid_loop_preserves_work(self, executor):
        """A 429 after a tool call is retried and accumulated work survives."""
        tc = make_tool_call("search_memory", {"query": "test"})
        resp_tool = make_litellm_response(content="", tool_calls=[tc])
        resp_final = make_litellm_response(content="Final answer", tool_calls=None)

        mock = AsyncMock(side_effect=[resp_tool, _RateLimitError("too many requests"), resp_final])
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test prompt", system_prompt="sys")

        assert mock.call_count == 3
        assert "Final answer" in result.text
        assert len(result.tool_call_records) == 1

    async def test_auth_error_raises_immediately(self, executor):
        mock = AsyncMock(side_effect=_AuthError("invalid api key"))
        with patch("litellm.acompletion", mock), pytest.raises(LLMAPIError):
            await executor.execute("test", system_prompt="sys")
        assert mock.call_count == 1

    async def test_rate_error_reports_to_guard(self, executor, _mock_rate_guard):
        resp_final = make_litellm_response(content="ok", tool_calls=None)
        mock = AsyncMock(side_effect=[_RateLimitError("too many requests"), resp_final])
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert "ok" in result.text
        _mock_rate_guard.report_block.assert_called_once()
        assert _mock_rate_guard.report_block.call_args.args[0] == "openai:api"

    async def test_retry_budget_exhausted_raises(self, executor):
        mock = AsyncMock(side_effect=_RateLimitError("too many requests"))
        with patch("litellm.acompletion", mock), pytest.raises(LLMAPIError):
            await executor.execute("test", system_prompt="sys")
        # 1 initial + MAX_LLM_RETRIES(3)
        assert mock.call_count == 4

    async def test_inner_litellm_retries_disabled(self, executor):
        """num_retries=0 on wrapped calls — in-loop retry is the single authority."""
        resp = make_litellm_response(content="ok", tool_calls=None)
        mock = AsyncMock(side_effect=[resp])
        with patch("litellm.acompletion", mock):
            await executor.execute("test", system_prompt="sys")
        assert mock.call_args_list[0].kwargs.get("num_retries") == 0

    async def test_interrupt_during_retry_backoff(self, model_config, anima_dir, tool_handler, memory):
        interrupt_event = asyncio.Event()
        executor = _make_executor(model_config, anima_dir, tool_handler, memory, interrupt_event=interrupt_event)

        async def _fail_and_interrupt(**kwargs):
            interrupt_event.set()
            raise _RateLimitError("too many requests")

        mock = AsyncMock(side_effect=_fail_and_interrupt)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert result.text == "[Session interrupted by user]"
        assert mock.call_count == 1


# ── Empty-response recovery ──────────────────────────────────


class TestEmptyResponseRecovery:
    async def test_reprompts_then_gives_up(self, executor):
        empty = make_litellm_response(content="", tool_calls=None)
        mock = AsyncMock(side_effect=[empty, empty, empty])
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert mock.call_count == 3
        assert result.text == "Unable to generate a final response. Please try again."
        assert result.truncated is True

    async def test_recovers_after_reprompt(self, executor):
        empty = make_litellm_response(content="", tool_calls=None)
        final = make_litellm_response(content="Recovered", tool_calls=None)
        mock = AsyncMock(side_effect=[empty, final])
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert mock.call_count == 2
        assert "Recovered" in result.text
        assert result.truncated is False
        # The reprompt reminder was injected into the 2nd call's history
        reprompt = msg_empty_response()
        assert any(m.get("role") == "user" and reprompt in str(m.get("content")) for m in _messages_of_call(mock, 1))

    async def test_thinking_only_response_counts_as_empty(self, executor):
        thinking_only = make_litellm_response(content="<think>pondering...</think>", tool_calls=None)
        final = make_litellm_response(content="Answer", tool_calls=None)
        mock = AsyncMock(side_effect=[thinking_only, final])
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert mock.call_count == 2
        assert "Answer" in result.text


# ── Runaway guard ────────────────────────────────────────────


class TestRunawayGuard:
    def _identical_tool_resp(self):
        tc = make_tool_call("search_memory", {"query": "same"})
        return make_litellm_response(content="", tool_calls=[tc])

    async def test_warn_reminder_at_three_consecutive(self, executor):
        responses = [self._identical_tool_resp() for _ in range(3)]
        responses.append(make_litellm_response(content="Done", tool_calls=None))
        mock = AsyncMock(side_effect=responses)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert "Done" in result.text
        warning = msg_tool_loop_warning(tool_names="search_memory", count=3)
        assert any(warning in str(m.get("content")) for m in _messages_of_call(mock, 3))

    async def test_halt_at_five_consecutive_forces_finalization(self, executor):
        responses = [self._identical_tool_resp() for _ in range(5)]
        responses.append(make_litellm_response(content="Summary of work", tool_calls=None))
        mock = AsyncMock(side_effect=responses)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert mock.call_count == 6
        assert "Summary of work" in result.text
        # Final call: tools stripped, halt reminder present
        last_kwargs = mock.call_args_list[5].kwargs
        assert "tools" not in last_kwargs
        halt_msg = msg_tool_loop_halt(count=5)
        assert any(halt_msg in str(m.get("content")) for m in last_kwargs["messages"])
        # Only 4 tool executions happened (the 5th batch was blocked)
        assert len(result.tool_call_records) == 4

    async def test_tool_call_after_halt_finalizes_immediately(self, executor):
        """If the model keeps calling tools after HALT (e.g. Bedrock keeps
        toolConfig), the loop finalizes instead of burning iterations."""
        responses = [self._identical_tool_resp() for _ in range(7)]
        mock = AsyncMock(side_effect=responses)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        # 5 identical (halt at 5th, blocked) + 1 post-halt tool call → finalize
        assert mock.call_count == 6
        assert result.truncated is True
        assert len(result.tool_call_records) == 4

    async def test_varied_calls_do_not_trigger_guard(self, executor):
        responses = []
        for i in range(4):
            tc = make_tool_call("search_memory", {"query": f"q{i}"})
            responses.append(make_litellm_response(content="", tool_calls=[tc]))
        responses.append(make_litellm_response(content="Done", tool_calls=None))
        mock = AsyncMock(side_effect=responses)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert "Done" in result.text
        warning_fragment = msg_tool_loop_warning(tool_names="search_memory", count=3)
        for i in range(5):
            assert not any(warning_fragment in str(m.get("content")) for m in _messages_of_call(mock, i))

    async def test_runaway_grace_turn_is_tool_free(self, executor):
        responses = [self._identical_tool_resp() for _ in range(5)]
        responses.append(make_litellm_response(content="Runaway summary", tool_calls=None))
        mock = AsyncMock(side_effect=responses)
        with patch("litellm.acompletion", mock):
            result = await executor.execute(
                "test",
                system_prompt="sys",
            )
        assert result.text == "Runaway summary"
        assert result.truncated is True
        assert mock.call_count == 6
        assert "tools" not in mock.call_args_list[-1].kwargs


# ── Streaming: retry only before first yield ─────────────────


class TestStreamingRetry:
    @pytest.fixture
    def ollama_executor(self, anima_dir, tool_handler, memory):
        config = ModelConfig(
            model="ollama/qwen3:8b",
            max_tokens=1024,
            context_threshold=0.50,
            max_chains=2,
        )
        return _make_executor(config, anima_dir, tool_handler, memory)

    async def _collect(self, gen):
        return [event async for event in gen]

    async def test_iteration0_transient_error_is_retried(self, ollama_executor):
        from core.prompt.context import ContextTracker

        final = make_litellm_response(content="Streamed answer", tool_calls=None)
        mock = AsyncMock(side_effect=[_RateLimitError("too many requests"), final])
        with (
            patch("litellm.acompletion", mock),
            patch("core.execution._litellm_streaming.decorrelated_jitter", return_value=0.0),
        ):
            events = await self._collect(
                ollama_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=ContextTracker(model="ollama/qwen3:8b"),
                )
            )
        assert mock.call_count == 2
        done = [e for e in events if e["type"] == "done"]
        assert done and "Streamed answer" in done[0]["full_text"]

    async def test_error_after_first_yield_is_not_retried(self, ollama_executor):
        from core.exceptions import StreamDisconnectedError
        from core.prompt.context import ContextTracker

        tc = make_tool_call("search_memory", {"query": "x"})
        resp_tool = make_litellm_response(content="working on it", tool_calls=[tc])
        err = _RateLimitError("too many requests")
        mock = AsyncMock(side_effect=[resp_tool, err, err, err, err])
        with (
            patch("litellm.acompletion", mock),
            patch("core.execution._litellm_streaming.decorrelated_jitter", return_value=0.0),
            pytest.raises(StreamDisconnectedError),
        ):
            await self._collect(
                ollama_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=ContextTracker(model="ollama/qwen3:8b"),
                )
            )
        # iteration 0 succeeded, iteration 1 failed once — no retry attempts
        # (pre-existing fail-fast behavior preserved after partial yield)
        assert mock.call_count == 2
