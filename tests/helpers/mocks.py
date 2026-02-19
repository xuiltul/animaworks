# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Mock factories for LLM API calls.

Provides helpers to mock litellm.acompletion, claude_agent_sdk.query,
and anthropic.AsyncAnthropic for isolated testing.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


# ── litellm mocks ─────────────────────────────────────────


def make_litellm_response(
    content: str = "Mock response",
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Create a mock ``litellm.acompletion`` response object."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    # model_dump() is called when appending assistant message to history
    dump = {"role": "assistant", "content": content}
    if tool_calls:
        dump["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    mock_message.model_dump.return_value = dump

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return mock_response


def make_tool_call(
    name: str,
    arguments: dict[str, Any],
    call_id: str = "call_001",
) -> MagicMock:
    """Create a mock tool_call object matching LiteLLM format."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@contextmanager
def patch_litellm(*responses: MagicMock):
    """Patch ``litellm.acompletion`` with a sequence of mock responses.

    Injects a mock ``litellm`` module into ``sys.modules`` so that lazy
    ``import litellm`` in production code resolves without the real package.

    Usage::

        with patch_litellm(resp1, resp2, resp3):
            result = await agent.run_cycle("hello")
    """
    mock_fn = AsyncMock(side_effect=list(responses))

    mock_module = MagicMock()
    mock_module.acompletion = mock_fn
    mock_module.token_counter = MagicMock(return_value=100)

    saved = sys.modules.get("litellm")
    try:
        sys.modules["litellm"] = mock_module
        yield mock_fn
    finally:
        if saved is None:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = saved


# ── claude_agent_sdk mocks ────────────────────────────────


class MockTextBlock:
    """Mock for ``claude_agent_sdk.TextBlock``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class MockToolUseBlock:
    """Mock for ``claude_agent_sdk.ToolUseBlock``."""

    def __init__(self, name: str, input: dict, id: str = "tu_001") -> None:
        self.name = name
        self.input = input
        self.id = id
        self.type = "tool_use"


class MockAssistantMessage:
    """Mock for ``claude_agent_sdk.AssistantMessage``."""

    def __init__(self, content: list[Any]) -> None:
        self.content = content


class MockResultMessage:
    """Mock for ``claude_agent_sdk.ResultMessage``."""

    def __init__(
        self,
        *,
        usage: dict[str, int] | None = None,
        num_turns: int = 1,
        session_id: str = "test-session-001",
    ) -> None:
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 200}
        self.num_turns = num_turns
        self.session_id = session_id


class MockStreamEvent:
    """Mock for ``claude_agent_sdk.types.StreamEvent``."""

    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event


@contextmanager
def patch_agent_sdk(
    response_text: str = "Mock A1 response.",
    tool_uses: list[MockToolUseBlock] | None = None,
    usage: dict[str, int] | None = None,
    num_turns: int = 1,
):
    """Patch ``claude_agent_sdk.query`` with an async generator mock.

    Injects the mock module into ``sys.modules`` so lazy imports work.
    """
    content_blocks: list[Any] = []
    if tool_uses:
        content_blocks.extend(tool_uses)
    content_blocks.append(MockTextBlock(response_text))

    assistant_msg = MockAssistantMessage(content_blocks)
    result_msg = MockResultMessage(
        usage=usage, num_turns=num_turns,
    )

    async def mock_query(prompt, options):
        yield assistant_msg
        yield result_msg

    # Build mock module
    mock_module = MagicMock()
    mock_module.query = mock_query
    mock_module.AssistantMessage = MockAssistantMessage
    mock_module.ResultMessage = MockResultMessage
    mock_module.TextBlock = MockTextBlock
    mock_module.ToolUseBlock = MockToolUseBlock
    mock_module.ClaudeAgentOptions = MagicMock
    mock_module.HookMatcher = MagicMock

    # Types submodule
    mock_types = MagicMock()
    mock_types.HookContext = MagicMock
    mock_types.HookInput = MagicMock
    mock_types.PostToolUseHookSpecificOutput = MagicMock
    mock_types.SyncHookJSONOutput = MagicMock
    mock_types.StreamEvent = MockStreamEvent
    mock_module.types = mock_types

    saved_modules: dict[str, Any] = {}
    keys_to_patch = ["claude_agent_sdk", "claude_agent_sdk.types"]

    try:
        for key in keys_to_patch:
            saved_modules[key] = sys.modules.get(key)
            if key == "claude_agent_sdk.types":
                sys.modules[key] = mock_types
            else:
                sys.modules[key] = mock_module
        yield mock_module
    finally:
        for key in keys_to_patch:
            if saved_modules[key] is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = saved_modules[key]


@contextmanager
def patch_agent_sdk_streaming(
    text_deltas: list[str] | None = None,
    usage: dict[str, int] | None = None,
    num_turns: int = 1,
):
    """Patch ``claude_agent_sdk.query`` for streaming tests.

    Yields ``StreamEvent`` objects simulating real streaming behavior.
    """
    if text_deltas is None:
        text_deltas = ["Mock ", "streaming ", "response."]

    full_text = "".join(text_deltas)
    result_msg = MockResultMessage(usage=usage, num_turns=num_turns)

    async def mock_query(prompt, options):
        # Yield text deltas as StreamEvents
        for delta in text_deltas:
            yield MockStreamEvent({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": delta},
                "index": 0,
            })
        # Yield assistant message with full text
        yield MockAssistantMessage([MockTextBlock(full_text)])
        # Yield result
        yield result_msg

    mock_module = MagicMock()
    mock_module.query = mock_query
    mock_module.AssistantMessage = MockAssistantMessage
    mock_module.ResultMessage = MockResultMessage
    mock_module.TextBlock = MockTextBlock
    mock_module.ToolUseBlock = MockToolUseBlock
    mock_module.ClaudeAgentOptions = MagicMock
    mock_module.HookMatcher = MagicMock

    mock_types = MagicMock()
    mock_types.StreamEvent = MockStreamEvent
    mock_types.HookContext = MagicMock
    mock_types.HookInput = MagicMock
    mock_types.PostToolUseHookSpecificOutput = MagicMock
    mock_types.SyncHookJSONOutput = MagicMock
    mock_module.types = mock_types

    saved_modules: dict[str, Any] = {}
    keys_to_patch = ["claude_agent_sdk", "claude_agent_sdk.types"]

    try:
        for key in keys_to_patch:
            saved_modules[key] = sys.modules.get(key)
            if key == "claude_agent_sdk.types":
                sys.modules[key] = mock_types
            else:
                sys.modules[key] = mock_module
        yield mock_module
    finally:
        for key in keys_to_patch:
            if saved_modules[key] is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = saved_modules[key]


# ── conversation compression mocks ────────────────────────


@contextmanager
def patch_anthropic_compression(summary_text: str = "Compressed summary."):
    """Patch ``litellm.acompletion`` for conversation compression tests.

    Now uses litellm instead of anthropic.AsyncAnthropic.
    """
    mock_response = make_litellm_response(content=summary_text)
    mock_acompletion = AsyncMock(return_value=mock_response)

    saved = sys.modules.get("litellm")
    if saved is None:
        mock_mod = MagicMock()
        mock_mod.acompletion = mock_acompletion
        sys.modules["litellm"] = mock_mod

    try:
        with patch("litellm.acompletion", mock_acompletion):
            yield mock_acompletion
    finally:
        if saved is None:
            sys.modules.pop("litellm", None)
