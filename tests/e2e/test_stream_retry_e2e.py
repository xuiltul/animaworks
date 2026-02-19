# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for stream checkpoint retry feature.

Verifies the full flow of stream disconnection detection, checkpoint
persistence, retry with rebuilt prompt, and cleanup on success.

Tests run in mock mode (``--mock``) without real API keys.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.agent import AgentCore
from core.execution.agent_sdk import StreamDisconnectedError
from core.memory.shortterm import ShortTermMemory, StreamCheckpoint
from core.prompt.builder import BuildResult

from tests.helpers.mocks import (
    MockAssistantMessage,
    MockResultMessage,
    MockStreamEvent,
    MockTextBlock,
    patch_agent_sdk_streaming,
)


# ── Helper: mock executor factories ────────────────────────


def _make_streaming_executor(
    *,
    fail_count: int = 0,
    text_deltas: list[str] | None = None,
    partial_text: str = "",
):
    """Build a mock executor whose ``execute_streaming`` fails *fail_count*
    times with ``StreamDisconnectedError``, then succeeds.

    Args:
        fail_count: How many consecutive calls should raise
            StreamDisconnectedError before succeeding.
        text_deltas: Text chunks yielded on the successful call.
        partial_text: Partial text carried by the error.
    """
    if text_deltas is None:
        text_deltas = ["Hello ", "from ", "retry!"]

    full_text = "".join(text_deltas)
    call_count = 0

    async def execute_streaming(
        system_prompt: str,
        prompt: str,
        tracker: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        nonlocal call_count
        call_count += 1

        if call_count <= fail_count:
            raise StreamDisconnectedError(
                f"Mock disconnect #{call_count}",
                partial_text=partial_text,
            )

        # Successful stream
        for delta in text_deltas:
            yield {
                "type": "text_delta",
                "text": delta,
            }
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": MockResultMessage(
                usage={"input_tokens": 500, "output_tokens": 100},
                num_turns=1,
            ),
            "replied_to_from_transcript": set(),
        }

    class _MockExecutor:
        supports_streaming = True

        def __init__(self) -> None:
            self.execute_streaming = execute_streaming
            self.call_count_getter = lambda: call_count

    return _MockExecutor()


def _make_always_failing_executor():
    """Build a mock executor that always raises StreamDisconnectedError."""

    async def execute_streaming(
        system_prompt: str,
        prompt: str,
        tracker: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise StreamDisconnectedError(
            "Persistent disconnect",
            partial_text="partial output",
        )
        # Make this a generator
        yield  # pragma: no cover

    class _MockExecutor:
        supports_streaming = True

        def __init__(self) -> None:
            self.execute_streaming = execute_streaming

    return _MockExecutor()


# ── Tests ──────────────────────────────────────────────────


class TestStreamRetryFullFlow:
    """Test the complete stream retry lifecycle: disconnect -> retry -> success."""

    async def test_stream_retry_full_flow(self, make_agent_core, monkeypatch):
        """Disconnect on first call, succeed on second.

        Verifies:
        - retry_start event is yielded
        - cycle_done event is yielded after successful retry
        - Checkpoint file is cleared after success
        """
        with patch_agent_sdk_streaming():
            agent = make_agent_core(
                name="retry-e2e",
                model="claude-sonnet-4-20250514",
            )
            agent._sdk_available = True

        # Replace executor with our mock that fails once then succeeds
        mock_exec = _make_streaming_executor(fail_count=1, partial_text="partial")
        agent._executor = mock_exec

        # Mock priming to avoid real memory search
        monkeypatch.setattr(agent, "_run_priming", AsyncMock(return_value=""))

        # Mock retry config: fast retry, max 2 retries
        monkeypatch.setattr(
            agent,
            "_load_stream_retry_config",
            lambda: {
                "checkpoint_enabled": True,
                "retry_max": 2,
                "retry_delay_s": 0.01,
            },
        )

        # Mock prompt building to avoid filesystem reads
        monkeypatch.setattr(
            "core.agent.build_system_prompt",
            lambda *args, **kwargs: BuildResult(system_prompt="mock system prompt"),
        )
        monkeypatch.setattr(
            "core.agent.inject_shortterm",
            lambda sp, st: sp,
        )

        # Collect all streamed events
        events: list[dict[str, Any]] = []
        async for chunk in agent.run_cycle_streaming("Test prompt", trigger="test"):
            events.append(chunk)

        event_types = [e["type"] for e in events]

        # Must have a retry_start event (the first call disconnected)
        assert "retry_start" in event_types, (
            f"Expected retry_start in events, got: {event_types}"
        )

        # Must have cycle_done at the end (retry succeeded)
        assert "cycle_done" in event_types, (
            f"Expected cycle_done in events, got: {event_types}"
        )

        # The retry_start should indicate retry=1
        retry_events = [e for e in events if e["type"] == "retry_start"]
        assert retry_events[0]["retry"] == 1
        assert retry_events[0]["max_retries"] == 2

        # cycle_done should contain a valid cycle_result
        cycle_done = [e for e in events if e["type"] == "cycle_done"][0]
        assert "cycle_result" in cycle_done
        assert cycle_done["cycle_result"]["action"] == "responded"

        # Checkpoint file should be cleared after success
        shortterm = ShortTermMemory(agent.anima_dir)
        assert shortterm.load_checkpoint() is None, (
            "Checkpoint should be cleared after successful retry"
        )


class TestStreamRetryMaxExceeded:
    """Test that exceeding max retries produces an error event."""

    async def test_stream_retry_max_exceeded(self, make_agent_core, monkeypatch):
        """All calls fail -> error event after max_retries.

        Verifies:
        - After max_retries, an error event is yielded
        - The error message mentions max retries / exhaustion
        """
        with patch_agent_sdk_streaming():
            agent = make_agent_core(
                name="retry-max",
                model="claude-sonnet-4-20250514",
            )
            agent._sdk_available = True

        # Replace executor with one that always fails
        agent._executor = _make_always_failing_executor()

        # Mock priming
        monkeypatch.setattr(agent, "_run_priming", AsyncMock(return_value=""))

        # Mock retry config: max 2 retries, fast delay
        monkeypatch.setattr(
            agent,
            "_load_stream_retry_config",
            lambda: {
                "checkpoint_enabled": True,
                "retry_max": 2,
                "retry_delay_s": 0.01,
            },
        )

        monkeypatch.setattr(
            "core.agent.build_system_prompt",
            lambda *args, **kwargs: BuildResult(system_prompt="mock system prompt"),
        )
        monkeypatch.setattr(
            "core.agent.inject_shortterm",
            lambda sp, st: sp,
        )

        events: list[dict[str, Any]] = []
        async for chunk in agent.run_cycle_streaming("Test prompt", trigger="test"):
            events.append(chunk)

        event_types = [e["type"] for e in events]

        # Should have retry_start events for each retry attempt
        retry_events = [e for e in events if e["type"] == "retry_start"]
        assert len(retry_events) == 2, (
            f"Expected 2 retry_start events, got {len(retry_events)}"
        )

        # Should have an error event after exhausting retries
        assert "error" in event_types, (
            f"Expected error event after max retries, got: {event_types}"
        )

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1

        # Error message should mention max retries
        error_msg = error_events[0]["message"]
        assert "リトライ" in error_msg or "max" in error_msg.lower() or "回" in error_msg, (
            f"Error message should mention max retries, got: {error_msg}"
        )

        # cycle_done should still be yielded (with whatever was accumulated)
        assert "cycle_done" in event_types, (
            "cycle_done should be yielded even after retry exhaustion"
        )


class TestCheckpointClearedOnSuccess:
    """Test that stream_checkpoint.json is removed after a successful completion."""

    async def test_checkpoint_cleared_on_success(self, make_agent_core, monkeypatch):
        """No disconnect: verify checkpoint file does not linger."""
        with patch_agent_sdk_streaming():
            agent = make_agent_core(
                name="retry-clean",
                model="claude-sonnet-4-20250514",
            )
            agent._sdk_available = True

        # Use a successful executor (no failures)
        mock_exec = _make_streaming_executor(fail_count=0)
        agent._executor = mock_exec

        monkeypatch.setattr(agent, "_run_priming", AsyncMock(return_value=""))
        monkeypatch.setattr(
            agent,
            "_load_stream_retry_config",
            lambda: {
                "checkpoint_enabled": True,
                "retry_max": 3,
                "retry_delay_s": 0.01,
            },
        )
        monkeypatch.setattr(
            "core.agent.build_system_prompt",
            lambda *args, **kwargs: BuildResult(system_prompt="mock system prompt"),
        )
        monkeypatch.setattr(
            "core.agent.inject_shortterm",
            lambda sp, st: sp,
        )

        # Consume all events
        events: list[dict[str, Any]] = []
        async for chunk in agent.run_cycle_streaming("Hello", trigger="test"):
            events.append(chunk)

        event_types = [e["type"] for e in events]

        # No retry events should be emitted
        assert "retry_start" not in event_types, (
            "No retries expected on clean execution"
        )

        # cycle_done must be present
        assert "cycle_done" in event_types

        # The checkpoint file must not exist
        checkpoint_path = agent.anima_dir / "shortterm" / "stream_checkpoint.json"
        assert not checkpoint_path.exists(), (
            f"stream_checkpoint.json should not exist after success, "
            f"but found at {checkpoint_path}"
        )

    async def test_checkpoint_saved_during_tool_end_events(
        self, make_agent_core, monkeypatch
    ):
        """Verify checkpoint is persisted when tool_end events are emitted,
        then cleared on successful completion."""

        async def execute_streaming_with_tool(
            system_prompt: str,
            prompt: str,
            tracker: Any,
            **kwargs: Any,
        ) -> AsyncGenerator[dict[str, Any], None]:
            """Executor that emits a tool_start/tool_end pair, then completes."""
            yield {
                "type": "tool_start",
                "tool_name": "Read",
                "tool_id": "tool_001",
            }
            yield {
                "type": "tool_end",
                "tool_name": "Read",
                "tool_id": "tool_001",
            }
            yield {
                "type": "text_delta",
                "text": "File contents read.",
            }
            yield {
                "type": "done",
                "full_text": "File contents read.",
                "result_message": MockResultMessage(
                    usage={"input_tokens": 500, "output_tokens": 100},
                ),
                "replied_to_from_transcript": set(),
            }

        class _ToolExecutor:
            supports_streaming = True
            execute_streaming = staticmethod(execute_streaming_with_tool)

        with patch_agent_sdk_streaming():
            agent = make_agent_core(
                name="retry-tool",
                model="claude-sonnet-4-20250514",
            )
            agent._sdk_available = True

        agent._executor = _ToolExecutor()

        monkeypatch.setattr(agent, "_run_priming", AsyncMock(return_value=""))
        monkeypatch.setattr(
            agent,
            "_load_stream_retry_config",
            lambda: {
                "checkpoint_enabled": True,
                "retry_max": 3,
                "retry_delay_s": 0.01,
            },
        )
        monkeypatch.setattr(
            "core.agent.build_system_prompt",
            lambda *args, **kwargs: BuildResult(system_prompt="mock system prompt"),
        )
        monkeypatch.setattr(
            "core.agent.inject_shortterm",
            lambda sp, st: sp,
        )

        events: list[dict[str, Any]] = []
        checkpoint_existed_during_stream = False

        async for chunk in agent.run_cycle_streaming("Read a file", trigger="test"):
            events.append(chunk)
            # Check if checkpoint exists right after tool_end is yielded
            if chunk["type"] == "tool_end":
                shortterm = ShortTermMemory(agent.anima_dir)
                cp = shortterm.load_checkpoint()
                if cp is not None:
                    checkpoint_existed_during_stream = True

        # Checkpoint should have been saved during the tool_end event
        assert checkpoint_existed_during_stream, (
            "Checkpoint should be saved after tool_end events"
        )

        # But after completion, checkpoint should be cleared
        shortterm = ShortTermMemory(agent.anima_dir)
        assert shortterm.load_checkpoint() is None, (
            "Checkpoint should be cleared after successful completion"
        )
