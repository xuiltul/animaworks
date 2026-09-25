"""Task-only same-session context compaction in the streaming cycle."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.prompt.builder import BuildResult
from tests.unit.core.test_agent import _make_agent


@pytest.fixture
def task_cycle(tmp_path):
    agent = _make_agent(tmp_path, model="claude-sonnet-4-6", resolved_mode="S")
    agent.model_config.task_compaction_tokens = 100
    agent.model_config.task_compaction_max = 6
    agent._run_priming = AsyncMock(return_value=("", ""))
    agent._preflight_size_check = AsyncMock(return_value=("system", "prompt", False))
    agent._load_stream_retry_config = lambda: {
        "checkpoint_enabled": False,
        "retry_max": 0,
        "retry_delay_s": 0,
    }
    agent._executor.supports_streaming = True
    agent._executor.compact_session_by_id = AsyncMock(return_value=True)
    with (
        patch("core._agent_cycle.build_system_prompt", return_value=BuildResult(system_prompt="system")),
        patch("core._agent_cycle._save_prompt_log"),
        patch("core._agent_cycle._save_prompt_log_end"),
        patch("core._agent_cycle._log_session_token_usage"),
    ):
        yield agent


def _stream_responses(agent, responses):
    calls = []

    async def stream(*args, **kwargs):
        tracker = args[2]
        calls.append({"args": args, "kwargs": kwargs.copy()})
        response = responses[len(calls) - 1]
        if "tokens" in response:
            tracker.update_from_message_start({"input_tokens": response["tokens"]})
            yield {"type": "context_update", "input_tokens": response["tokens"]}
        yield {"type": "text_delta", "text": response["text"]}
        yield {
            "type": "done",
            "full_text": response["text"],
            "result_message": SimpleNamespace(
                session_id=response.get("session_id", "sdk-task-session"),
                num_turns=1,
            ),
            "task_compact_requested": response.get("compact", False),
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

    agent._executor.execute_streaming = stream
    return calls


@pytest.mark.asyncio
async def test_task_below_threshold_finishes_in_one_attempt(task_cycle):
    calls = _stream_responses(task_cycle, [{"text": "finished", "tokens": 99}])

    events = [event async for event in task_cycle._run_cycle_streaming_inner("original task", trigger="task:task-1")]

    assert len(calls) == 1
    task_cycle._executor.compact_session_by_id.assert_not_awaited()
    assert events[-1]["cycle_result"]["summary"] == "finished"


@pytest.mark.asyncio
async def test_task_compacts_then_resumes_same_session_and_accumulates_text(task_cycle):
    calls = _stream_responses(
        task_cycle,
        [
            {"text": "before compaction", "compact": True, "session_id": "sdk-session-1", "tokens": 120},
            {"text": "after compaction", "tokens": 30},
        ],
    )

    events = [event async for event in task_cycle._run_cycle_streaming_inner("original task", trigger="task:task-2")]

    assert len(calls) == 2
    task_cycle._executor.compact_session_by_id.assert_awaited_once()
    assert calls[1]["kwargs"]["resume_session_id"] == "sdk-session-1"
    continuation_prompt = calls[1]["args"][1]
    assert "original task" in continuation_prompt
    assert "繰り返さない" in continuation_prompt
    assert events[-1]["cycle_result"]["summary"] == "before compaction\nafter compaction"

    activity_dir = task_cycle.anima_dir / "activity_log"
    entries = [line for path in activity_dir.glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
    assert any('"type": "task_compacted"' in line and '"success": true' in line for line in entries)
    assert any('"type": "task_compacted_after"' in line and '"tokens_after": 30' in line for line in entries)


@pytest.mark.asyncio
async def test_task_compaction_stops_at_configured_maximum(task_cycle):
    task_cycle.model_config.task_compaction_max = 1
    calls = _stream_responses(
        task_cycle,
        [
            {"text": "first", "compact": True, "session_id": "sdk-session-1"},
            {"text": "second", "compact": True, "session_id": "sdk-session-1"},
        ],
    )

    events = [event async for event in task_cycle._run_cycle_streaming_inner("task", trigger="task:task-3")]

    assert len(calls) == 2
    task_cycle._executor.compact_session_by_id.assert_awaited_once()
    assert events[-1]["cycle_result"]["summary"] == "first\nsecond"


@pytest.mark.asyncio
async def test_compact_failure_still_resumes_with_continuation_prompt(task_cycle):
    task_cycle._executor.compact_session_by_id = AsyncMock(return_value=False)
    calls = _stream_responses(
        task_cycle,
        [
            {"text": "work already done", "compact": True, "session_id": "sdk-session-2"},
            {"text": "continued", "tokens": 25},
        ],
    )

    events = [event async for event in task_cycle._run_cycle_streaming_inner("original task", trigger="task:task-4")]

    assert len(calls) == 2
    assert calls[1]["kwargs"]["resume_session_id"] == "sdk-session-2"
    assert "original task" in calls[1]["args"][1]
    assert events[-1]["cycle_result"]["summary"] == "work already done\ncontinued"


@pytest.mark.parametrize(
    ("trigger", "task_tokens"),
    [("chat", 100), ("heartbeat", 100), ("task:task-5", 0)],
)
@pytest.mark.asyncio
async def test_non_task_triggers_and_zero_limit_do_not_compact(task_cycle, trigger: str, task_tokens: int):
    task_cycle.model_config.task_compaction_tokens = task_tokens
    calls = _stream_responses(
        task_cycle,
        [{"text": "finished", "compact": True, "session_id": "sdk-session-3"}],
    )

    events = [event async for event in task_cycle._run_cycle_streaming_inner("task", trigger=trigger)]

    assert len(calls) == 1
    assert "resume_session_id" not in calls[0]["kwargs"]
    task_cycle._executor.compact_session_by_id.assert_not_awaited()
    assert events[-1]["cycle_result"]["summary"] == "finished"
