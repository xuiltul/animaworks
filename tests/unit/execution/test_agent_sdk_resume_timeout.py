"""Tests for agent_sdk.py resume timeout guard and clear_session_ids public wrapper.

Covers:
  - RESUME_TIMEOUT_SEC constant is defined
  - asyncio.wait_for is used when resuming (session_id_to_resume is set)
  - TimeoutError causes _clear_session_id to be called and fallback to fresh session
  - clear_session_ids() public wrapper clears both 'chat' and 'heartbeat' types
  - _clear_session_id() file deletion logic
  - _load_session_id() / _save_session_id() persistence
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import ModelConfig


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-6",
        api_key="sk-test",
        max_turns=5,
        context_threshold=0.50,
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    (d / "state").mkdir(parents=True)
    return d


# ── Session persistence helpers ───────────────────────────────


class TestClearSessionId:
    """_clear_session_id() removes the session file."""

    def test_removes_existing_file(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _clear_session_id, _save_session_id

        _save_session_id(anima_dir, "sess-001", "chat")
        path = anima_dir / "state" / "current_session_chat.json"
        assert path.exists()

        _clear_session_id(anima_dir, "chat")
        assert not path.exists()

    def test_noop_when_no_file(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _clear_session_id

        # Should not raise
        _clear_session_id(anima_dir, "chat")
        _clear_session_id(anima_dir, "heartbeat")

    def test_clears_heartbeat_session(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _clear_session_id, _save_session_id

        _save_session_id(anima_dir, "sess-hb", "heartbeat")
        path = anima_dir / "state" / "current_session_heartbeat.json"
        assert path.exists()

        _clear_session_id(anima_dir, "heartbeat")
        assert not path.exists()


class TestClearSessionIds:
    """clear_session_ids() public wrapper clears both chat and heartbeat."""

    def test_clears_both_types(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import (
            clear_session_ids,
            _save_session_id,
        )

        _save_session_id(anima_dir, "sess-chat", "chat")
        _save_session_id(anima_dir, "sess-hb", "heartbeat")

        chat_path = anima_dir / "state" / "current_session_chat.json"
        hb_path = anima_dir / "state" / "current_session_heartbeat.json"
        assert chat_path.exists()
        assert hb_path.exists()

        clear_session_ids(anima_dir)

        assert not chat_path.exists(), "chat session file should be removed"
        assert not hb_path.exists(), "heartbeat session file should be removed"

    def test_noop_when_no_files(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import clear_session_ids

        # Should not raise even when no files exist
        clear_session_ids(anima_dir)

    def test_clears_only_chat_if_only_chat_exists(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import (
            clear_session_ids,
            _save_session_id,
        )

        _save_session_id(anima_dir, "sess-chat", "chat")
        chat_path = anima_dir / "state" / "current_session_chat.json"
        assert chat_path.exists()

        # Should not raise when heartbeat file is missing
        clear_session_ids(anima_dir)
        assert not chat_path.exists()


class TestResumeTimeoutConstant:
    """RESUME_TIMEOUT_SEC is defined with a reasonable value."""

    def test_resume_timeout_defined(self) -> None:
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

        assert RESUME_TIMEOUT_SEC == 15.0

    def test_resume_timeout_is_positive(self) -> None:
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

        assert RESUME_TIMEOUT_SEC > 0


# ── Resume timeout guard in execute_streaming ─────────────────


@contextmanager
def _patch_sdk_for_streaming(messages: list[Any]):
    """Patch claude_agent_sdk for streaming tests with custom message sequence."""
    from tests.helpers.mocks import (
        MockAssistantMessage,
        MockClaudeSDKClient,
        MockResultMessage,
        MockStreamEvent,
        MockTextBlock,
        MockToolResultBlock,
        MockUserMessage,
        MockSystemMessage,
    )

    def _client_factory(**kwargs: Any) -> MockClaudeSDKClient:
        return MockClaudeSDKClient(messages=messages, **kwargs)

    mock_module = MagicMock()
    mock_module.ClaudeSDKClient = _client_factory
    mock_module.AssistantMessage = MockAssistantMessage
    mock_module.ResultMessage = MockResultMessage
    mock_module.TextBlock = MockTextBlock
    mock_module.ToolUseBlock = MagicMock
    mock_module.ToolResultBlock = MockToolResultBlock
    mock_module.UserMessage = MockUserMessage
    mock_module.SystemMessage = MockSystemMessage
    mock_module.ClaudeAgentOptions = MagicMock
    mock_module.HookMatcher = MagicMock
    mock_module.ClaudeSDKError = Exception
    mock_module.ProcessError = Exception

    mock_types = MagicMock()
    mock_types.StreamEvent = MockStreamEvent
    mock_types.HookContext = MagicMock
    mock_types.HookInput = MagicMock
    mock_types.PreToolUseHookSpecificOutput = MagicMock
    mock_types.SyncHookJSONOutput = MagicMock
    mock_module.types = mock_types

    saved: dict[str, Any] = {}
    for key in ["claude_agent_sdk", "claude_agent_sdk.types"]:
        saved[key] = sys.modules.get(key)
        sys.modules[key] = mock_types if key == "claude_agent_sdk.types" else mock_module

    try:
        yield mock_module
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


class TestResumeTimeoutGuard:
    """asyncio.wait_for is applied to first-event receive during session resume."""

    @pytest.mark.asyncio
    async def test_wait_for_called_with_resume_timeout_when_resuming(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        """When a session_id is present, asyncio.wait_for wraps first-event receive."""
        from core.execution.agent_sdk import _save_session_id
        from tests.helpers.mocks import MockResultMessage, MockStreamEvent, MockAssistantMessage, MockTextBlock

        # Persist a session ID so execute_streaming takes the resume path
        _save_session_id(anima_dir, "stale-session-001", "chat")

        messages = [
            MockStreamEvent({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hello"},
                "index": 0,
            }),
            MockAssistantMessage([MockTextBlock("hello")]),
            MockResultMessage(usage={"input_tokens": 100, "output_tokens": 50}),
        ]

        wait_for_calls: list[dict] = []
        original_wait_for = asyncio.wait_for

        async def _spy_wait_for(coro, timeout=None, **kwargs):
            wait_for_calls.append({"timeout": timeout})
            return await original_wait_for(coro, timeout=timeout, **kwargs)

        with _patch_sdk_for_streaming(messages):
            from core.execution.agent_sdk import AgentSDKExecutor
            from core.prompt.context import ContextTracker

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with patch("asyncio.wait_for", side_effect=_spy_wait_for):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    events.append(event)

        # asyncio.wait_for should have been called with RESUME_TIMEOUT_SEC
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC
        timeout_values = [c["timeout"] for c in wait_for_calls]
        assert RESUME_TIMEOUT_SEC in timeout_values, (
            f"Expected wait_for to be called with {RESUME_TIMEOUT_SEC}s timeout, "
            f"but got timeouts: {timeout_values}"
        )

    @pytest.mark.asyncio
    async def test_no_wait_for_when_no_session_to_resume(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        """When no session ID exists, asyncio.wait_for is NOT called for resume guard."""
        from tests.helpers.mocks import MockResultMessage, MockStreamEvent, MockAssistantMessage, MockTextBlock

        # No session file → fresh session path (no resume)
        messages = [
            MockStreamEvent({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "fresh"},
                "index": 0,
            }),
            MockAssistantMessage([MockTextBlock("fresh")]),
            MockResultMessage(usage={"input_tokens": 50, "output_tokens": 20}),
        ]

        wait_for_calls: list[dict] = []
        original_wait_for = asyncio.wait_for

        async def _spy_wait_for(coro, timeout=None, **kwargs):
            wait_for_calls.append({"timeout": timeout})
            return await original_wait_for(coro, timeout=timeout, **kwargs)

        with _patch_sdk_for_streaming(messages):
            from core.execution.agent_sdk import AgentSDKExecutor
            from core.prompt.context import ContextTracker

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with patch("asyncio.wait_for", side_effect=_spy_wait_for):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    events.append(event)

        # No wait_for should have been called for resume timeout
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC
        resume_timeout_calls = [c for c in wait_for_calls if c["timeout"] == RESUME_TIMEOUT_SEC]
        assert resume_timeout_calls == [], (
            "Expected no wait_for resume guard on fresh session, "
            f"but got: {resume_timeout_calls}"
        )

    @pytest.mark.asyncio
    async def test_clear_session_id_called_on_resume_timeout(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        """When resume times out, _clear_session_id is called and falls back to
        fresh session."""
        from core.execution.agent_sdk import _save_session_id
        from tests.helpers.mocks import MockResultMessage, MockStreamEvent, MockAssistantMessage, MockTextBlock

        _save_session_id(anima_dir, "stale-session-for-timeout", "chat")

        # After timeout, fresh session produces events normally
        fresh_messages = [
            MockStreamEvent({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "recovered"},
                "index": 0,
            }),
            MockAssistantMessage([MockTextBlock("recovered")]),
            MockResultMessage(usage={"input_tokens": 100, "output_tokens": 50}),
        ]

        clear_calls: list[str] = []
        original_clear = None

        def _spy_clear(anima_dir_arg, session_type):
            clear_calls.append(session_type)
            if original_clear:
                original_clear(anima_dir_arg, session_type)

        call_count = [0]

        async def _timeout_on_first_then_succeed(coro, timeout=None, **kwargs):
            """Raise TimeoutError on the first call (resume), succeed thereafter."""
            from core.execution.agent_sdk import RESUME_TIMEOUT_SEC
            if timeout == RESUME_TIMEOUT_SEC and call_count[0] == 0:
                call_count[0] += 1
                raise asyncio.TimeoutError("resume timed out")
            return await coro

        with _patch_sdk_for_streaming(fresh_messages):
            from core.execution.agent_sdk import AgentSDKExecutor, _clear_session_id
            from core.prompt.context import ContextTracker

            original_clear = _clear_session_id

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with (
                patch("asyncio.wait_for", side_effect=_timeout_on_first_then_succeed),
                patch("core.execution._sdk_session._clear_session_id", side_effect=_spy_clear),
            ):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    events.append(event)

        # _clear_session_id should have been called for the chat session type
        assert len(clear_calls) >= 1, (
            "Expected _clear_session_id to be called on resume timeout, "
            f"but got: {clear_calls}"
        )
        assert "chat" in clear_calls, (
            f"Expected 'chat' session to be cleared, got: {clear_calls}"
        )
