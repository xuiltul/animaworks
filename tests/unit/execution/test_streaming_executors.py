"""Tests for streaming execution across all executor types.

Covers StreamDisconnectedError shared behavior, BaseExecutor default
streaming, AssistedExecutor (Mode B) iteration-level streaming,
LiteLLMExecutor (Mode A2) token-level and iteration-level streaming,
and AnthropicFallbackExecutor (Mode A1 Fallback) token-level streaming.
All LLM calls are mocked — no real API calls are made.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.base import (
    BaseExecutor,
    ExecutionResult,
    StreamDisconnectedError,
)
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory


# ── Helpers ──────────────────────────────────────────────────


class ConcreteExecutor(BaseExecutor):
    """Minimal concrete subclass for testing BaseExecutor streaming defaults."""

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(text=f"response: {prompt}")


async def _collect_events(async_gen) -> list[dict[str, Any]]:
    """Collect all events from an async generator into a list."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


def _make_llm_response(content: str = "hello") -> MagicMock:
    """Build a fake LiteLLM response object with message content."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── StreamDisconnectedError ───────────────────────────────────


class TestStreamDisconnectedErrorAttributes:
    """StreamDisconnectedError has partial_text attribute."""

    def test_carries_partial_text(self) -> None:
        err = StreamDisconnectedError(
            "stream lost",
            partial_text="accumulated output",
        )
        assert err.partial_text == "accumulated output"
        assert str(err) == "stream lost"

    def test_is_exception(self) -> None:
        err = StreamDisconnectedError("test")
        assert isinstance(err, Exception)


class TestStreamDisconnectedErrorDefaultPartial:
    """Default partial_text is an empty string."""

    def test_default_empty_partial_text(self) -> None:
        err = StreamDisconnectedError("disconnected")
        assert err.partial_text == ""

    def test_default_message(self) -> None:
        err = StreamDisconnectedError()
        assert str(err) == "Stream disconnected"
        assert err.partial_text == ""


class TestStreamDisconnectedBackwardCompatImport:
    """Importing StreamDisconnectedError from agent_sdk still works."""

    def test_import_from_agent_sdk(self) -> None:
        from core.execution.agent_sdk import StreamDisconnectedError as SDE
        assert SDE is StreamDisconnectedError

    def test_in_agent_sdk_all(self) -> None:
        from core.execution import agent_sdk
        assert "StreamDisconnectedError" in agent_sdk.__all__


# ── BaseExecutor streaming defaults ───────────────────────────


class TestBaseSupportsStreamingDefaultTrue:
    """BaseExecutor.supports_streaming returns True now."""

    def test_supports_streaming_is_true(self, tmp_path: Path) -> None:
        config = ModelConfig(model="test-model")
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        assert executor.supports_streaming is True


class TestBaseDefaultStreamingYieldsTextAndDone:
    """Default execute_streaming yields text_delta + done from execute() result."""

    @pytest.mark.asyncio
    async def test_default_streaming_events(self, tmp_path: Path) -> None:
        config = ModelConfig(model="test-model")
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        tracker = MagicMock(spec=ContextTracker)

        events = await _collect_events(
            executor.execute_streaming(
                system_prompt="sys",
                prompt="hello",
                tracker=tracker,
            )
        )

        assert len(events) == 2

        # First event: text_delta with the full response
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "response: hello"

        # Second event: done with full_text and result_message
        assert events[1]["type"] == "done"
        assert events[1]["full_text"] == "response: hello"
        assert events[1]["result_message"] is None


# ── AssistedExecutor (Mode B) streaming ───────────────────────


@pytest.fixture
def assisted_executor(tmp_path: Path):
    """Build an AssistedExecutor with mocked dependencies."""
    from core.execution.assisted import AssistedExecutor

    config = ModelConfig(model="ollama/test-model", max_turns=5)
    tool_handler = MagicMock()
    tool_handler._human_notifier = None
    memory = MagicMock()

    # Patch _build_tool_schemas to avoid importing real tool modules
    with patch.object(AssistedExecutor, "_build_tool_schemas", return_value=[]):
        executor = AssistedExecutor(
            model_config=config,
            anima_dir=tmp_path,
            tool_handler=tool_handler,
            memory=memory,
        )
    return executor


class TestModeBStreamingNoToolCall:
    """Single iteration, no tool call yields text_delta + done."""

    @pytest.mark.asyncio
    async def test_no_tool_call(self, assisted_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # Mock _call_llm to return a simple text response
        assisted_executor._call_llm = AsyncMock(
            return_value=_make_llm_response("This is the final answer.")
        )

        events = await _collect_events(
            assisted_executor.execute_streaming(
                system_prompt="You are a helpful assistant.",
                prompt="What is 2+2?",
                tracker=tracker,
            )
        )

        # Should have text_delta and done
        types = [e["type"] for e in events]
        assert "text_delta" in types
        assert "done" in types

        # text_delta should contain the response
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0]["text"] == "This is the final answer."

        # done event should have full_text
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["full_text"] == "This is the final answer."
        assert done_events[0]["result_message"] is None


class TestModeBStreamingWithToolCall:
    """Tool call yields text_delta + tool_start + tool_end + text_delta + done."""

    @pytest.mark.asyncio
    async def test_tool_call_then_final(self, assisted_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # First call: LLM returns a tool call embedded in text
        tool_response = _make_llm_response(
            'Let me search for that.\n```json\n{"tool": "web_search", "arguments": {"query": "test"}}\n```'
        )
        # Second call: LLM returns final text (no tool call)
        final_response = _make_llm_response("The answer is 42.")

        assisted_executor._call_llm = AsyncMock(
            side_effect=[tool_response, final_response]
        )
        # Add web_search to known tools so it passes validation
        assisted_executor._known_tools = {"web_search"}
        # Mock tool execution
        assisted_executor._tool_handler.handle = MagicMock(
            return_value="Search results: found 42"
        )

        events = await _collect_events(
            assisted_executor.execute_streaming(
                system_prompt="sys",
                prompt="Search for the answer",
                tracker=tracker,
            )
        )

        types = [e["type"] for e in events]

        # Expected event sequence:
        # 1. text_delta (narrative from first iteration: "Let me search for that.")
        # 2. tool_start (web_search)
        # 3. tool_end (web_search)
        # 4. text_delta (final answer: "The answer is 42.")
        # 5. done
        assert "tool_start" in types
        assert "tool_end" in types
        assert "done" in types

        # Verify tool_start event
        tool_starts = [e for e in events if e["type"] == "tool_start"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "web_search"
        assert "tool_id" in tool_starts[0]

        # Verify tool_end event
        tool_ends = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_ends) == 1
        assert tool_ends[0]["tool_name"] == "web_search"
        assert tool_ends[0]["tool_id"] == tool_starts[0]["tool_id"]

        # Verify done event
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert "The answer is 42." in done[0]["full_text"]

        # Verify the narrative text from the first iteration was yielded
        text_events = [e for e in events if e["type"] == "text_delta"]
        all_text = " ".join(e["text"] for e in text_events)
        assert "Let me search for that." in all_text

        # Verify tool_handler.handle was called
        assisted_executor._tool_handler.handle.assert_called_once_with(
            "web_search", {"query": "test"},
        )


class TestModeBStreamingErrorRaisesStreamDisconnected:
    """API error during streaming raises StreamDisconnectedError."""

    @pytest.mark.asyncio
    async def test_api_error_raises_stream_disconnected(
        self, assisted_executor,
    ) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # _call_llm raises an exception
        assisted_executor._call_llm = AsyncMock(
            side_effect=RuntimeError("API connection refused")
        )

        with pytest.raises(StreamDisconnectedError) as exc_info:
            await _collect_events(
                assisted_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Do something",
                    tracker=tracker,
                )
            )

        err = exc_info.value
        # partial_text should be empty since no iterations completed
        assert err.partial_text == ""
        assert "Mode B stream error" in str(err)
        assert "API connection refused" in str(err)

    @pytest.mark.asyncio
    async def test_error_after_partial_response(self, assisted_executor) -> None:
        """Error after one successful iteration preserves partial text."""
        tracker = MagicMock(spec=ContextTracker)

        # First call succeeds with a tool call
        tool_response = _make_llm_response(
            'Thinking...\n```json\n{"tool": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
        )
        # Second call fails
        assisted_executor._call_llm = AsyncMock(
            side_effect=[tool_response, RuntimeError("timeout")]
        )
        assisted_executor._known_tools = {"read_file"}
        assisted_executor._tool_handler.handle = MagicMock(
            return_value="file contents"
        )

        with pytest.raises(StreamDisconnectedError) as exc_info:
            await _collect_events(
                assisted_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Read the file",
                    tracker=tracker,
                )
            )

        err = exc_info.value
        # The narrative from the first iteration should be in partial_text
        assert "Thinking..." in err.partial_text


# ── A2 LiteLLMExecutor streaming ─────────────────────────────


class FakeStreamChunk:
    """Simulate a single streaming chunk from litellm.acompletion(stream=True)."""

    def __init__(
        self,
        text: str | None = None,
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Any | None = None,
    ) -> None:
        delta = MagicMock()
        delta.content = text
        delta.tool_calls = tool_calls

        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason

        self.choices = [choice]
        self.usage = usage


class FakeUsage:
    """Simulate a usage object from litellm streaming."""

    def __init__(
        self,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeDelta:
    """Simulate a tool_call delta fragment."""

    def __init__(
        self,
        index: int,
        id: str | None = None,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = id
        func = MagicMock()
        func.name = name
        func.arguments = arguments
        self.function = func


async def _fake_async_stream(chunks: list[Any]):
    """Create an async iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


def _make_litellm_a2_response(
    content: str = "hello",
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Build a mock litellm.acompletion response (non-streaming)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": (
            [
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
            if tool_calls
            else None
        ),
    }

    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return resp


def _make_mock_tool_call(
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


@pytest.fixture
def litellm_executor(tmp_path: Path):
    """Build a LiteLLMExecutor with mocked dependencies."""
    from core.execution.litellm_loop import LiteLLMExecutor

    config = ModelConfig(
        model="openai/gpt-4o",
        api_key="sk-test",
        max_tokens=1024,
        max_turns=5,
    )
    tool_handler = MagicMock()
    tool_handler._human_notifier = None
    memory = MagicMock()

    # Create minimal anima dir structure
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    for sub in ["skills", "state"]:
        (anima_dir / sub).mkdir(exist_ok=True)

    executor = LiteLLMExecutor(
        model_config=config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )
    return executor


@pytest.fixture
def ollama_executor(tmp_path: Path):
    """Build a LiteLLMExecutor configured for Ollama (iteration-level)."""
    from core.execution.litellm_loop import LiteLLMExecutor

    config = ModelConfig(
        model="ollama/llama3.2",
        max_tokens=2048,
        max_turns=5,
    )
    tool_handler = MagicMock()
    tool_handler._human_notifier = None
    memory = MagicMock()

    anima_dir = tmp_path / "animas" / "test-ollama"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    for sub in ["skills", "state"]:
        (anima_dir / sub).mkdir(exist_ok=True)

    executor = LiteLLMExecutor(
        model_config=config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )
    return executor


class TestIsOllamaModelDetection:
    """_is_ollama_model returns True for ollama/ and ollama_chat/ prefixes."""

    def test_ollama_prefix(self, tmp_path: Path) -> None:
        from core.execution.litellm_loop import LiteLLMExecutor

        config = ModelConfig(model="ollama/llama3.2")
        th = MagicMock()
        th._human_notifier = None
        anima_dir = tmp_path / "test"
        anima_dir.mkdir()
        (anima_dir / "permissions.md").write_text("")
        (anima_dir / "skills").mkdir()
        ex = LiteLLMExecutor(
            model_config=config, anima_dir=anima_dir,
            tool_handler=th, tool_registry=[], memory=MagicMock(),
        )
        assert ex._is_ollama_model is True

    def test_ollama_chat_prefix(self, tmp_path: Path) -> None:
        from core.execution.litellm_loop import LiteLLMExecutor

        config = ModelConfig(model="ollama_chat/glm-4")
        th = MagicMock()
        th._human_notifier = None
        anima_dir = tmp_path / "test"
        anima_dir.mkdir()
        (anima_dir / "permissions.md").write_text("")
        (anima_dir / "skills").mkdir()
        ex = LiteLLMExecutor(
            model_config=config, anima_dir=anima_dir,
            tool_handler=th, tool_registry=[], memory=MagicMock(),
        )
        assert ex._is_ollama_model is True

    def test_openai_prefix_is_not_ollama(self, litellm_executor) -> None:
        assert litellm_executor._is_ollama_model is False

    def test_anthropic_prefix_is_not_ollama(self, tmp_path: Path) -> None:
        from core.execution.litellm_loop import LiteLLMExecutor

        config = ModelConfig(model="anthropic/claude-sonnet-4-20250514")
        th = MagicMock()
        th._human_notifier = None
        anima_dir = tmp_path / "test"
        anima_dir.mkdir()
        (anima_dir / "permissions.md").write_text("")
        (anima_dir / "skills").mkdir()
        ex = LiteLLMExecutor(
            model_config=config, anima_dir=anima_dir,
            tool_handler=th, tool_registry=[], memory=MagicMock(),
        )
        assert ex._is_ollama_model is False


class TestA2TokenLevelTextOnly:
    """Token-level streaming with text-only response (no tool calls)."""

    async def test_yields_text_deltas_and_done(self, litellm_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # Create streaming chunks: two text deltas + finish_reason=stop
        chunks = [
            FakeStreamChunk(text="Hello "),
            FakeStreamChunk(text="world!"),
            FakeStreamChunk(
                finish_reason="stop",
                usage=FakeUsage(prompt_tokens=50, completion_tokens=10),
            ),
        ]

        mock_acompletion = AsyncMock(return_value=_fake_async_stream(chunks))

        with patch("litellm.acompletion", mock_acompletion), \
             patch.object(litellm_executor, "_preflight_clamp", return_value={}):
            events = await _collect_events(
                litellm_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Hi",
                    tracker=tracker,
                )
            )

        types = [e["type"] for e in events]
        assert "text_delta" in types
        assert "done" in types

        # Verify text deltas
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 2
        assert text_events[0]["text"] == "Hello "
        assert text_events[1]["text"] == "world!"

        # Verify done event
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert done[0]["full_text"] == "Hello world!"
        assert done[0]["result_message"] is None


class TestA2TokenLevelWithToolCall:
    """Token-level streaming with tool call deltas."""

    async def test_yields_tool_start_and_tool_end(self, litellm_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # First LLM call: yields text + tool_call chunks
        iter1_chunks = [
            FakeStreamChunk(text="Let me search."),
            FakeStreamChunk(
                tool_calls=[FakeDelta(index=0, id="call_1", name="search_memory")],
            ),
            FakeStreamChunk(
                tool_calls=[FakeDelta(index=0, arguments='{"query":')],
            ),
            FakeStreamChunk(
                tool_calls=[FakeDelta(index=0, arguments=' "test"}')],
            ),
            FakeStreamChunk(finish_reason="tool_calls"),
        ]

        # Second LLM call: final text only
        iter2_chunks = [
            FakeStreamChunk(text="Found it!"),
            FakeStreamChunk(finish_reason="stop"),
        ]

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _fake_async_stream(iter1_chunks)
            return _fake_async_stream(iter2_chunks)

        async def mock_process_tool_calls(parsed_calls, messages, tools, active_categories):
            """Mock _process_streaming_tool_calls as an async generator yielding tool_end events."""
            for tc in parsed_calls:
                yield {
                    "type": "tool_end",
                    "tool_id": tc["id"],
                    "tool_name": tc["name"],
                }

        with patch("litellm.acompletion", side_effect=mock_acompletion), \
             patch.object(litellm_executor, "_preflight_clamp", return_value={}), \
             patch.object(
                 litellm_executor, "_process_streaming_tool_calls",
                 mock_process_tool_calls,
             ):
            events = await _collect_events(
                litellm_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Search for something",
                    tracker=tracker,
                )
            )

        types = [e["type"] for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert "done" in types

        # Verify tool_start
        tool_starts = [e for e in events if e["type"] == "tool_start"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "search_memory"
        assert tool_starts[0]["tool_id"] == "call_1"

        # Verify tool_end
        tool_ends = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_ends) == 1
        assert tool_ends[0]["tool_name"] == "search_memory"
        assert tool_ends[0]["tool_id"] == "call_1"

        # Verify done
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert "Found it!" in done[0]["full_text"]


class TestA2TokenLevelErrorRaisesStreamDisconnected:
    """API error during token-level streaming raises StreamDisconnectedError."""

    async def test_error_raises_stream_disconnected(
        self, litellm_executor,
    ) -> None:
        tracker = MagicMock(spec=ContextTracker)

        mock_acompletion = AsyncMock(
            side_effect=RuntimeError("Connection refused"),
        )

        with pytest.raises(StreamDisconnectedError) as exc_info:
            with patch("litellm.acompletion", mock_acompletion), \
                 patch.object(litellm_executor, "_preflight_clamp", return_value={}):
                await _collect_events(
                    litellm_executor.execute_streaming(
                        system_prompt="sys",
                        prompt="Hi",
                        tracker=tracker,
                    )
                )

        err = exc_info.value
        assert err.partial_text == ""
        assert "stream error" in str(err).lower()
        assert "Connection refused" in str(err)


class TestA2IterationLevelTextOnly:
    """Iteration-level streaming (Ollama) with text-only response."""

    async def test_yields_text_delta_and_done(self, ollama_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        resp = _make_litellm_a2_response(
            content="Ollama says hello!",
            tool_calls=None,
            prompt_tokens=50,
            completion_tokens=20,
        )

        mock_acompletion = AsyncMock(return_value=resp)

        with patch("litellm.acompletion", mock_acompletion), \
             patch.object(ollama_executor, "_preflight_clamp", return_value={}):
            events = await _collect_events(
                ollama_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Hello",
                    tracker=tracker,
                )
            )

        types = [e["type"] for e in events]
        assert "text_delta" in types
        assert "done" in types

        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0]["text"] == "Ollama says hello!"

        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert done[0]["full_text"] == "Ollama says hello!"


class TestA2IterationLevelWithToolCall:
    """Iteration-level streaming (Ollama) with tool calls."""

    async def test_yields_tool_events(self, ollama_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # First response: tool call
        tc = _make_mock_tool_call(
            "search_memory", {"query": "test"}, "call_olm_1",
        )
        resp_tool = _make_litellm_a2_response(
            content="Searching...",
            tool_calls=[tc],
        )

        # Second response: final text
        resp_final = _make_litellm_a2_response(
            content="Found it!",
            tool_calls=None,
        )

        mock_acompletion = AsyncMock(
            side_effect=[resp_tool, resp_final],
        )

        # Mock tool execution
        async def mock_execute_tool_call(tc, fn_args):
            return {"role": "tool", "tool_call_id": tc.id, "content": "result"}

        with patch("litellm.acompletion", mock_acompletion), \
             patch.object(ollama_executor, "_preflight_clamp", return_value={}), \
             patch.object(
                 ollama_executor, "_execute_tool_call",
                 side_effect=mock_execute_tool_call,
             ):
            events = await _collect_events(
                ollama_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Search for data",
                    tracker=tracker,
                )
            )

        types = [e["type"] for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert "done" in types

        # Verify tool events
        tool_starts = [e for e in events if e["type"] == "tool_start"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "search_memory"

        tool_ends = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_ends) == 1
        assert tool_ends[0]["tool_name"] == "search_memory"

        # Verify done
        done = [e for e in events if e["type"] == "done"]
        assert done[0]["full_text"] == "Searching...\nFound it!"


class TestA2DispatchToTokenLevel:
    """Non-Ollama model routes to token-level streaming."""

    async def test_non_ollama_uses_token_level(self, litellm_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # Spy on _stream_token_level
        token_called = False
        original_token = litellm_executor._stream_token_level

        async def spy_token(*args, **kwargs):
            nonlocal token_called
            token_called = True
            async for event in original_token(*args, **kwargs):
                yield event

        chunks = [
            FakeStreamChunk(text="test"),
            FakeStreamChunk(finish_reason="stop"),
        ]

        with patch("litellm.acompletion", AsyncMock(return_value=_fake_async_stream(chunks))), \
             patch.object(litellm_executor, "_preflight_clamp", return_value={}), \
             patch.object(litellm_executor, "_stream_token_level", spy_token):
            await _collect_events(
                litellm_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                )
            )

        assert token_called is True


class TestA2DispatchToIterationLevel:
    """Ollama model routes to iteration-level streaming."""

    async def test_ollama_uses_iteration_level(self, ollama_executor) -> None:
        tracker = MagicMock(spec=ContextTracker)

        iteration_called = False
        original_iter = ollama_executor._stream_iteration_level

        async def spy_iter(*args, **kwargs):
            nonlocal iteration_called
            iteration_called = True
            async for event in original_iter(*args, **kwargs):
                yield event

        resp = _make_litellm_a2_response(content="ok", tool_calls=None)

        with patch("litellm.acompletion", AsyncMock(return_value=resp)), \
             patch.object(ollama_executor, "_preflight_clamp", return_value={}), \
             patch.object(ollama_executor, "_stream_iteration_level", spy_iter):
            await _collect_events(
                ollama_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                )
            )

        assert iteration_called is True


# ── A1 Fallback AnthropicFallbackExecutor streaming ──────────


class FakeContentBlock:
    """Simulate an Anthropic API content block."""

    def __init__(
        self,
        type: str,
        text: str = "",
        name: str = "",
        id: str = "",
        input: dict | None = None,
    ) -> None:
        self.type = type
        self.text = text
        self.name = name
        self.id = id
        self.input = input or {}


class FakeStreamEvent:
    """Simulate an Anthropic streaming event."""

    def __init__(
        self,
        type: str,
        text: str = "",
        content_block: Any = None,
    ) -> None:
        self.type = type
        self.text = text
        self.content_block = content_block


class FakeFinalMessage:
    """Simulate the final message from Anthropic stream."""

    def __init__(
        self,
        content: list[Any],
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        self.content = content
        self.usage = MagicMock(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class FakeAnthropicStream:
    """Simulate an Anthropic messages.stream() async context manager.

    Provides both the async iterator (for event in stream) and
    get_final_message() to match the Anthropic SDK interface.
    """

    def __init__(
        self,
        events: list[FakeStreamEvent],
        final_message: FakeFinalMessage,
    ) -> None:
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for event in self._events:
            yield event

    async def get_final_message(self) -> FakeFinalMessage:
        return self._final_message


@pytest.fixture
def anthropic_fallback_executor(tmp_path: Path):
    """Build an AnthropicFallbackExecutor with mocked dependencies."""
    from core.execution.anthropic_fallback import AnthropicFallbackExecutor

    config = ModelConfig(
        model="claude-sonnet-4-20250514",
        api_key="sk-test-anthropic",
        max_tokens=4096,
        max_turns=5,
    )
    tool_handler = MagicMock()
    tool_handler._human_notifier = None
    tool_handler.handle = MagicMock(return_value="tool result")
    memory = MagicMock()

    anima_dir = tmp_path / "animas" / "test-fallback"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    for sub in ["skills", "state"]:
        (anima_dir / sub).mkdir(exist_ok=True)

    executor = AnthropicFallbackExecutor(
        model_config=config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )
    return executor


class TestA1FallbackStreamingTextOnly:
    """Anthropic fallback streaming with text-only response."""

    async def test_yields_text_deltas_and_done(
        self, anthropic_fallback_executor,
    ) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # Build stream events
        stream_events = [
            FakeStreamEvent(type="text", text="Hello "),
            FakeStreamEvent(type="text", text="from Claude!"),
        ]

        # Final message: text only, no tool_use blocks
        final_msg = FakeFinalMessage(
            content=[FakeContentBlock(type="text", text="Hello from Claude!")],
        )

        fake_stream = FakeAnthropicStream(stream_events, final_msg)

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=fake_stream)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch.object(
                 anthropic_fallback_executor, "_build_tools",
                 return_value=[],
             ):
            events = await _collect_events(
                anthropic_fallback_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Hello",
                    tracker=tracker,
                )
            )

        types = [e["type"] for e in events]
        assert "text_delta" in types
        assert "done" in types

        # Verify text deltas
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 2
        assert text_events[0]["text"] == "Hello "
        assert text_events[1]["text"] == "from Claude!"

        # Verify done
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert done[0]["full_text"] == "Hello from Claude!"
        assert done[0]["result_message"] is None


class TestA1FallbackStreamingWithToolCall:
    """Anthropic fallback streaming with tool_use blocks."""

    async def test_yields_tool_events(
        self, anthropic_fallback_executor,
    ) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # First iteration: text + tool_use
        tool_block = FakeContentBlock(
            type="tool_use",
            name="search_memory",
            id="toolu_001",
            input={"query": "test"},
        )
        iter1_events = [
            FakeStreamEvent(type="text", text="Let me search."),
            FakeStreamEvent(
                type="content_block_start",
                content_block=tool_block,
            ),
        ]
        iter1_final = FakeFinalMessage(
            content=[
                FakeContentBlock(type="text", text="Let me search."),
                tool_block,
            ],
        )

        # Second iteration: text only (final)
        iter2_events = [
            FakeStreamEvent(type="text", text="Found it!"),
        ]
        iter2_final = FakeFinalMessage(
            content=[FakeContentBlock(type="text", text="Found it!")],
        )

        call_count = 0

        def make_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeAnthropicStream(iter1_events, iter1_final)
            return FakeAnthropicStream(iter2_events, iter2_final)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(side_effect=make_stream)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch.object(
                 anthropic_fallback_executor, "_build_tools",
                 return_value=[],
             ):
            events = await _collect_events(
                anthropic_fallback_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Search for data",
                    tracker=tracker,
                )
            )

        types = [e["type"] for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert "done" in types

        # Verify tool_start
        tool_starts = [e for e in events if e["type"] == "tool_start"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "search_memory"
        assert tool_starts[0]["tool_id"] == "toolu_001"

        # Verify tool_end
        tool_ends = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_ends) == 1
        assert tool_ends[0]["tool_name"] == "search_memory"
        assert tool_ends[0]["tool_id"] == "toolu_001"

        # Verify tool_handler was called
        anthropic_fallback_executor._tool_handler.handle.assert_called_once_with(
            "search_memory", {"query": "test"},
        )

        # Verify done
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert "Found it!" in done[0]["full_text"]


class TestA1FallbackStreamingError:
    """API error during Anthropic fallback streaming raises StreamDisconnectedError."""

    async def test_error_raises_stream_disconnected(
        self, anthropic_fallback_executor,
    ) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # Make the stream context manager raise on __aenter__
        class FailingStream:
            async def __aenter__(self):
                raise ConnectionError("API unreachable")

            async def __aexit__(self, *args):
                return False

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=FailingStream())

        with pytest.raises(StreamDisconnectedError) as exc_info:
            with patch("anthropic.AsyncAnthropic", return_value=mock_client), \
                 patch.object(
                     anthropic_fallback_executor, "_build_tools",
                     return_value=[],
                 ):
                await _collect_events(
                    anthropic_fallback_executor.execute_streaming(
                        system_prompt="sys",
                        prompt="Hi",
                        tracker=tracker,
                    )
                )

        err = exc_info.value
        assert err.partial_text == ""
        assert "stream error" in str(err).lower()
        assert "API unreachable" in str(err)


class TestA1FallbackStreamingToolErrorResilience:
    """Tool handler errors don't abort the stream — tool_end is still yielded."""

    async def test_tool_error_yields_tool_end(
        self, anthropic_fallback_executor,
    ) -> None:
        tracker = MagicMock(spec=ContextTracker)

        # First iteration: tool_use that will fail
        tool_block = FakeContentBlock(
            type="tool_use",
            name="search_memory",
            id="toolu_err",
            input={"query": "fail"},
        )
        iter1_events = [
            FakeStreamEvent(
                type="content_block_start",
                content_block=tool_block,
            ),
        ]
        iter1_final = FakeFinalMessage(
            content=[tool_block],
        )

        # Second iteration: final text
        iter2_events = [
            FakeStreamEvent(type="text", text="Recovered."),
        ]
        iter2_final = FakeFinalMessage(
            content=[FakeContentBlock(type="text", text="Recovered.")],
        )

        call_count = 0

        def make_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeAnthropicStream(iter1_events, iter1_final)
            return FakeAnthropicStream(iter2_events, iter2_final)

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(side_effect=make_stream)

        # Make tool_handler.handle raise an exception
        anthropic_fallback_executor._tool_handler.handle = MagicMock(
            side_effect=RuntimeError("Tool crashed!"),
        )

        with patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch.object(
                 anthropic_fallback_executor, "_build_tools",
                 return_value=[],
             ):
            events = await _collect_events(
                anthropic_fallback_executor.execute_streaming(
                    system_prompt="sys",
                    prompt="Try something",
                    tracker=tracker,
                )
            )

        types = [e["type"] for e in events]

        # tool_start should be yielded for the tool_use block
        assert "tool_start" in types

        # tool_end should still be yielded even though tool execution failed
        assert "tool_end" in types

        # done should still be yielded
        assert "done" in types

        # Verify tool_end
        tool_ends = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_ends) == 1
        assert tool_ends[0]["tool_name"] == "search_memory"
        assert tool_ends[0]["tool_id"] == "toolu_err"
