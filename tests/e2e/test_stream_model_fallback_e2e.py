# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E: a terminal LLM error swaps the model in-flight on every streaming route.

Task exec, the declaration probe, heartbeat and inbox all stream through
``run_cycle_streaming``; the swap lives there so none of them can burn a
rate-limited primary.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

from core.prompt.builder import BuildResult
from tests.helpers.mocks import MockResultMessage, patch_agent_sdk_streaming


def _terminal_quota_executor():
    """Executor that reports a terminal quota error, like Codex at its limit."""

    async def execute_streaming(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "type": "error",
            "terminal": True,
            "reason": "quota_exhausted",
            "message": "You've hit your usage limit.",
        }

    class _MockExecutor:
        supports_streaming = True

        def __init__(self) -> None:
            self.execute_streaming = execute_streaming

    return _MockExecutor()


def _success_executor(text: str):
    async def execute_streaming(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "text_delta", "text": text}
        yield {
            "type": "done",
            "full_text": text,
            "result_message": MockResultMessage(
                usage={"input_tokens": 10, "output_tokens": 5},
                num_turns=1,
            ),
            "replied_to_from_transcript": set(),
        }

    class _MockExecutor:
        supports_streaming = True

        def __init__(self) -> None:
            self.execute_streaming = execute_streaming

    return _MockExecutor()


async def test_terminal_quota_error_swaps_to_fallback_model(make_agent_core, monkeypatch):
    with patch_agent_sdk_streaming():
        agent = make_agent_core(name="fallback-stream-e2e", model="claude-sonnet-4-6")
        agent._sdk_available = True

    agent.model_config = agent.model_config.model_copy(update={"fallback_models": ["x:grok/grok-4.5"]})
    fallback_cfg = agent.model_config.model_copy(
        update={"model": "grok/grok-4.5", "execution_mode": "x", "resolved_mode": "X"},
    )
    agent._executor = _terminal_quota_executor()
    swapped = _success_executor("recovered on the fallback")

    monkeypatch.setattr(agent, "_run_priming", AsyncMock(return_value=("", "")))
    monkeypatch.setattr(agent, "_create_executor", lambda cfg=None: swapped)
    monkeypatch.setattr(
        "core._agent_cycle.build_system_prompt",
        lambda *args, **kwargs: BuildResult(system_prompt="mock system prompt"),
    )
    monkeypatch.setattr("core._agent_cycle.inject_shortterm", lambda sp, st: sp)
    # The primary is "blocked": preflight is a no-op here so the in-flight swap
    # is what gets exercised.
    monkeypatch.setattr(
        "core.execution.fallback_activity.preflight_fallback_config",
        lambda anima_dir, cfg, **kwargs: cfg,
    )
    monkeypatch.setattr(
        "core.execution.fallback_activity.runtime_fallback_config",
        lambda *args, **kwargs: fallback_cfg,
    )

    events = [chunk async for chunk in agent.run_cycle_streaming("do the task", trigger="task:t-1")]
    types = [e["type"] for e in events]

    assert "retry_start" in types
    assert [e for e in events if e["type"] == "retry_start"][0]["fallback_model"] == "grok/grok-4.5"
    # The suppressed terminal error must not reach the caller: task exec would
    # mark the task failed on it.
    assert "error" not in types
    cycle_done = [e for e in events if e["type"] == "cycle_done"][0]["cycle_result"]
    assert cycle_done["action"] == "responded"
    assert "recovered on the fallback" in cycle_done["summary"]


async def test_terminal_error_surfaces_when_no_fallback_available(make_agent_core, monkeypatch):
    with patch_agent_sdk_streaming():
        agent = make_agent_core(name="fallback-stream-none-e2e", model="claude-sonnet-4-6")
        agent._sdk_available = True

    agent.model_config = agent.model_config.model_copy(update={"fallback_models": ["x:grok/grok-4.5"]})
    agent._executor = _terminal_quota_executor()

    monkeypatch.setattr(agent, "_run_priming", AsyncMock(return_value=("", "")))
    monkeypatch.setattr(
        "core._agent_cycle.build_system_prompt",
        lambda *args, **kwargs: BuildResult(system_prompt="mock system prompt"),
    )
    monkeypatch.setattr("core._agent_cycle.inject_shortterm", lambda sp, st: sp)
    monkeypatch.setattr(
        "core.execution.fallback_activity.preflight_fallback_config",
        lambda anima_dir, cfg, **kwargs: cfg,
    )
    monkeypatch.setattr(
        "core.execution.fallback_activity.runtime_fallback_config",
        lambda *args, **kwargs: None,
    )

    events = [chunk async for chunk in agent.run_cycle_streaming("do the task", trigger="task:t-2")]

    errors = [e for e in events if e["type"] == "error"]
    assert errors and errors[0]["terminal"] is True
    cycle_done = [e for e in events if e["type"] == "cycle_done"][0]["cycle_result"]
    assert cycle_done["action"] == "error"


async def test_partial_tool_work_is_not_replayed_on_another_engine(make_agent_core, monkeypatch):
    from unittest.mock import MagicMock

    with patch_agent_sdk_streaming():
        agent = make_agent_core(name="fallback-partial-e2e", model="claude-sonnet-4-6")
    agent.model_config = agent.model_config.model_copy(update={"fallback_models": ["x:grok/grok-4.5"]})
    deliveries = []

    async def partial_stream(*args, **kwargs):
        yield {"type": "tool_start", "tool_name": "send_message", "tool_id": "sent-once"}
        deliveries.append("report")
        yield {
            "type": "error",
            "terminal": True,
            "reason": "quota_exhausted",
            "message": "You've hit your usage limit.",
        }

    agent._executor = _terminal_quota_executor()
    agent._executor.execute_streaming = partial_stream
    create_executor = MagicMock(return_value=_success_executor("would replay the original request"))
    monkeypatch.setattr(agent, "_create_executor", create_executor)
    monkeypatch.setattr(agent, "_run_priming", AsyncMock(return_value=("", "")))
    monkeypatch.setattr(
        "core._agent_cycle.build_system_prompt",
        lambda *args, **kwargs: BuildResult(system_prompt="mock system prompt"),
    )
    monkeypatch.setattr("core._agent_cycle.inject_shortterm", lambda sp, st: sp)
    monkeypatch.setattr("core.execution.fallback_activity.preflight_fallback_config", lambda anima_dir, cfg, **kw: cfg)

    events = [chunk async for chunk in agent.run_cycle_streaming("send the report", trigger="task:delivery")]
    create_executor.assert_not_called()
    assert deliveries == ["report"]
    assert not any(event["type"] == "retry_start" for event in events)
    assert any(event["type"] == "error" for event in events)
