from __future__ import annotations

import asyncio
import types

import pytest

from core._agent_cycle import CycleMixin as _CycleBase


class _DummyToolHandler:
    def bind_runtime_session(self, ctx):  # noqa: D401 - test stub
        self.ctx = ctx

    def set_active_session_type(self, session_type):
        return None


class _DummyAgent(_CycleBase):
    """Minimal stand-in exposing only what run_cycle_streaming touches."""

    def __init__(self, inner_chunks):
        self._tool_handler = _DummyToolHandler()
        self._lock = asyncio.Lock()
        self._inner_chunks = inner_chunks
        self.model_config = None

    def _get_agent_lock(self, thread_id):
        return self._lock

    def _check_monthly_token_budget(self, *, trigger, model_config):
        return None

    async def _run_cycle_streaming_inner(self, *args, **kwargs):
        for chunk in self._inner_chunks:
            yield chunk


async def _collect(agent, thread_id):
    from core._agent_cycle import CycleMixin

    run = types.MethodType(CycleMixin.run_cycle_streaming, agent)
    return [chunk async for chunk in run("prompt", "message:owner", thread_id=thread_id)]


@pytest.mark.asyncio
async def test_streaming_cycle_result_thread_id_follows_request_thread():
    """CycleResult.thread_id defaults to "default"; the wrapper must still stamp the real thread.

    Regression: non-default chat threads were rejected by the chat session guard,
    so the assistant reply vanished from the UI.
    """
    from core.schemas import CycleResult

    inner = CycleResult(trigger="message:owner", action="respond", summary="hi").model_dump(mode="json")
    assert inner["thread_id"] == "default"

    agent = _DummyAgent([{"type": "cycle_done", "cycle_result": inner}])
    chunks = await _collect(agent, "c43aec1d")

    result = chunks[-1]["cycle_result"]
    assert result["thread_id"] == "c43aec1d"
    assert result["session_type"] == "chat"
    assert result["request_id"]
