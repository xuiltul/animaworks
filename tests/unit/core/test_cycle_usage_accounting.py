"""Per-attempt billing and failure propagation at the executor/cycle boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.base import ExecutionResult, StreamDisconnectedError, TokenUsage
from core.prompt.builder import BuildResult
from tests.unit.core.test_agent import _make_agent


@pytest.fixture
def cycle(tmp_path):
    agent = _make_agent(tmp_path, model="codex/test-model", resolved_mode="C")
    agent._run_priming = AsyncMock(return_value=("", ""))
    agent._preflight_size_check = AsyncMock(return_value=("system", "prompt", False))
    agent._load_stream_retry_config = lambda: {"checkpoint_enabled": False, "retry_max": 0, "retry_delay_s": 0}
    agent._executor.supports_streaming = True
    with (
        patch("core._agent_cycle.build_system_prompt", return_value=BuildResult(system_prompt="system")),
        patch("core._agent_cycle._save_prompt_log"),
        patch("core._agent_cycle._save_prompt_log_end"),
        patch("core._agent_cycle._log_session_token_usage") as log,
    ):
        yield agent, log


def _done(usage, *, emitted=False):
    return {
        "type": "done",
        "full_text": "finished",
        "result_message": SimpleNamespace(num_turns=2),
        "usage": usage,
        "usage_already_emitted": emitted,
    }


async def test_stream_usage_deltas_are_not_counted_again_at_done(cycle):
    agent, log = cycle

    async def stream(*args, **kwargs):
        yield {"type": "usage", "usage": {"input_tokens": 100, "cache_read_tokens": 80}}
        yield {"type": "usage", "usage": {"input_tokens": 50, "output_tokens": 10}}
        yield _done({"input_tokens": 150, "cache_read_tokens": 80, "output_tokens": 10}, emitted=True)

    agent._executor.execute_streaming = stream
    events = [e async for e in agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")]
    usage = events[-1]["cycle_result"]["usage"]
    assert usage["input_tokens"] == 150
    assert usage["cache_read_tokens"] == 80
    log.assert_called_once()
    assert log.call_args.kwargs["usage"] == usage
    assert log.call_args.kwargs["turns"] == 2
    assert not any(e["type"] == "usage" for e in events)


@pytest.mark.parametrize("via_exception", [False, True])
async def test_terminal_error_preserves_usage_and_completed_tools(cycle, via_exception):
    agent, log = cycle

    async def stream(*args, **kwargs):
        event = {
            "type": "error",
            "terminal": True,
            "message": "API Error: ConnectionRefused",
            "reason": "network",
            "usage": {"input_tokens": 20},
            "tool_call_records": [
                {
                    "tool_name": "send_message",
                    "tool_id": "already-sent",
                    "input_summary": "report",
                    "result_summary": "sent",
                    "is_error": False,
                }
            ],
        }
        if via_exception:
            exc = RuntimeError(event["message"])
            exc.usage = event["usage"]
            exc.tool_call_records = event["tool_call_records"]
            raise exc
        yield event

    agent._executor.execute_streaming = stream
    agent.model_config.fallback_models = ["s:claude-test"]
    with patch("core.execution.fallback_activity.runtime_fallback_config", return_value=None) as fallback:
        events = [e async for e in agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")]
    result = events[-1]["cycle_result"]
    assert result["action"] == "error"
    assert result["usage"]["input_tokens"] == 20
    assert result["tool_call_records"][0]["tool_id"] == "already-sent"
    assert fallback.call_args.kwargs["partial_execution"] is True
    log.assert_called_once()


@pytest.mark.parametrize("emitted", [False, True])
async def test_cancelled_stream_bills_observed_usage_once_and_propagates(cycle, emitted):
    agent, log = cycle

    async def stream(*args, **kwargs):
        if emitted:
            yield {"type": "usage", "usage": {"input_tokens": 100}}
        exc = asyncio.CancelledError()
        exc.usage = {"input_tokens": 100}
        exc.usage_already_emitted = emitted
        raise exc

    agent._executor.execute_streaming = stream
    with pytest.raises(asyncio.CancelledError):
        _ = [e async for e in agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")]
    log.assert_called_once()
    assert log.call_args.kwargs["usage"]["input_tokens"] == 100


async def test_stream_close_flushes_usage(cycle):
    agent, log = cycle

    async def stream(*args, **kwargs):
        yield {"type": "usage", "usage": {"input_tokens": 25}}
        yield {"type": "text_delta", "text": "partial"}
        yield _done({"input_tokens": 25}, emitted=True)

    agent._executor.execute_streaming = stream
    iterator = agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")
    assert (await anext(iterator))["type"] == "text_delta"
    await iterator.aclose()
    log.assert_called_once()
    assert log.call_args.kwargs["usage"]["input_tokens"] == 25


async def test_fallback_usage_is_billed_under_each_actual_provider(cycle):
    agent, log = cycle
    agent.model_config.fallback_models = ["s:claude-test"]
    fallback = agent.model_config.model_copy(update={"model": "claude-test", "resolved_mode": "S"})

    async def primary_stream(*args, **kwargs):
        yield {"type": "error", "terminal": True, "message": "quota exhausted", "usage": {"input_tokens": 12}}

    async def fallback_stream(*args, **kwargs):
        yield _done({"input_tokens": 5, "cache_read_tokens": 60})

    agent._executor.execute_streaming = primary_stream
    other_executor = MagicMock()
    other_executor.execute_streaming = fallback_stream
    agent._create_executor = MagicMock(return_value=other_executor)
    with patch("core.execution.fallback_activity.runtime_fallback_config", return_value=fallback):
        events = [e async for e in agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")]
    assert events[-1]["cycle_result"]["action"] == "responded"
    assert len(log.call_args_list) == 2
    first, second = (call.kwargs for call in log.call_args_list)
    assert (first["model"], first["mode"], first["usage"]["input_tokens"]) == ("codex/test-model", "c", 12)
    assert (second["model"], second["mode"], second["usage"]["input_tokens"]) == ("claude-test", "s", 5)


async def test_wrapped_api_failure_routes_without_repeating_stream_retry_budget(cycle):
    agent, log = cycle
    agent.model_config.fallback_models = ["s:claude-test"]
    fallback = agent.model_config.model_copy(update={"model": "claude-test", "resolved_mode": "S"})
    attempts = []

    async def primary_stream(*args, **kwargs):
        attempts.append("primary")
        yield {"type": "usage", "usage": {"input_tokens": 1}}
        raise StreamDisconnectedError("A-stream error") from ConnectionError("Connection refused")

    async def fallback_stream(*args, **kwargs):
        attempts.append("fallback")
        yield _done({"input_tokens": 2})

    agent._executor.execute_streaming = primary_stream
    other_executor = MagicMock()
    other_executor.execute_streaming = fallback_stream
    agent._create_executor = MagicMock(return_value=other_executor)
    with patch("core.execution.fallback_activity.runtime_fallback_config", return_value=fallback) as route:
        events = [e async for e in agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")]
    assert attempts == ["primary", "fallback"]
    assert route.call_args.kwargs["reason"] == "network"
    assert events[-1]["cycle_result"]["action"] == "responded"
    assert log.call_count == 2


async def test_wrapped_content_policy_error_never_retries_or_switches(cycle):
    agent, _ = cycle
    attempts = []

    async def stream(*args, **kwargs):
        attempts.append("primary")
        raise StreamDisconnectedError("A-stream error") from RuntimeError("content_filter")
        yield  # pragma: no cover

    agent._executor.execute_streaming = stream
    events = [e async for e in agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")]
    assert attempts == ["primary"]
    assert events[-1]["cycle_result"]["action"] == "error"
    assert events[-1]["cycle_result"]["reason"] == "content_policy"


@pytest.mark.parametrize("work_type", ["tool_start", "tool_end", "text_delta"])
@pytest.mark.parametrize("as_model", [False, True])
async def test_outer_fallback_preserves_stream_started_guard_without_final_tool_records(cycle, work_type, as_model):
    from core.execution.fallback_activity import run_with_model_fallback
    from core.schemas import CycleResult

    agent, _ = cycle
    agent.model_config.fallback_models = ["s:claude-test"]
    fallback = agent.model_config.model_copy(update={"model": "claude-test", "resolved_mode": "S"})
    invocations = []

    async def stream(*args, **kwargs):
        yield {"type": work_type, "tool_name": "send_message", "tool_id": "sent-once", "text": "partial reply"}
        yield {"type": "error", "terminal": True, "message": "API Error: ConnectionRefused", "reason": "network"}

    agent._executor.execute_streaming = stream

    async def run(config):
        invocations.append(config.model)
        events = [event async for event in agent._run_cycle_streaming_inner("prompt", trigger="heartbeat")]
        result = events[-1]["cycle_result"]
        assert result["tool_call_records"] == []
        assert result["fallback_safe"] is False
        return CycleResult.model_validate(result) if as_model else result

    with (
        patch("core.execution.fallback_activity.resolve_effective_model_config", return_value=fallback) as resolve,
        patch("core.execution.fallback_activity.report_capacity_block") as block,
    ):
        result = await run_with_model_fallback(
            run,
            activity=MagicMock(),
            primary_config=agent.model_config,
            active_config=agent.model_config,
            channel="heartbeat",
        )

    assert invocations == [agent.model_config.model]
    assert (result.action if as_model else result["action"]) == "error"
    resolve.assert_not_called()
    block.assert_not_called()


def test_legacy_cycle_result_without_fallback_safe_keeps_tool_replay_guard():
    from core.execution.fallback_activity import has_partial_execution
    from core.schemas import CycleResult

    legacy = CycleResult.model_validate({"trigger": "heartbeat", "action": "error"})
    assert legacy.fallback_safe is True
    assert not has_partial_execution(legacy)
    legacy.tool_call_records = [{"tool_name": "send_message", "tool_id": "sent"}]
    assert has_partial_execution(legacy)


async def test_codex_blocking_failure_is_not_reported_as_success(cycle):
    agent, log = cycle
    agent._executor.execute = AsyncMock(
        return_value=ExecutionResult(
            text="network unavailable", error=True, reason="network", usage=TokenUsage(input_tokens=30)
        )
    )
    result = await agent._run_cycle_inner_scoped("prompt", "heartbeat")
    assert result.action == "error"
    assert result.stop_kind == "stream_error"
    assert result.reason == "network"
    assert log.call_args.kwargs["usage"]["input_tokens"] == 30


async def test_codex_blocking_cancel_retains_usage(cycle):
    agent, log = cycle
    exc = asyncio.CancelledError()
    exc.usage = {"input_tokens": 41}
    agent._executor.execute = AsyncMock(side_effect=exc)
    with pytest.raises(asyncio.CancelledError):
        await agent._run_cycle_inner_scoped("prompt", "heartbeat")
    log.assert_called_once()
    assert log.call_args.kwargs["usage"]["input_tokens"] == 41
