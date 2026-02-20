# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Mode B context overflow protection.

Validates preflight check, tool output truncation, timeout, and num_ctx.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.assisted import AssistedExecutor, _truncate_tool_output
from core.memory import MemoryManager
from core.tooling.handler import ToolHandler


def _make_llm_response(content: str) -> MagicMock:
    """Create a mock LiteLLM response object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestModeBPreflightProtection:
    """Test preflight context window checks in Mode B."""

    @pytest.fixture
    def assisted_executor(self, data_dir: Path, make_anima):
        """Create an AssistedExecutor with mocked dependencies."""
        anima_dir = make_anima(
            "test-preflight",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
            max_turns=5,
        )
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
        )
        return AssistedExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            memory=memory,
        )

    async def test_preflight_clamps_max_tokens(self, assisted_executor):
        """When input tokens are near context limit, max_tokens is clamped."""
        mock_response = _make_llm_response("応答テスト")

        captured_kwargs: list[dict[str, Any]] = []

        async def capture_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return mock_response

        with (
            patch("litellm.acompletion", side_effect=capture_acompletion),
            patch("litellm.token_counter", return_value=127_000),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            result = await assisted_executor.execute(
                prompt="test",
                system_prompt="sys",
            )

        assert len(captured_kwargs) >= 1
        called_max_tokens = captured_kwargs[0]["max_tokens"]
        # Context window for gemma3 is 128000. With 127000 input tokens,
        # available = 128000 - 127000 = 1000. Clamped = 1000 - 128 = 872.
        # This should be less than the configured max_tokens (1024 from test defaults).
        assert called_max_tokens < 1024
        assert called_max_tokens == 872
        assert "応答テスト" in result.text

    async def test_preflight_rejects_too_large_prompt(self, assisted_executor):
        """When prompt exceeds context window, execute returns error text."""
        with (
            patch("litellm.acompletion", new_callable=AsyncMock),
            patch("litellm.token_counter", return_value=128_000),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            result = await assisted_executor.execute(
                prompt="test",
                system_prompt="sys",
            )

        assert "Error" in result.text or "too large" in result.text

    async def test_normal_prompt_passes_preflight(self, assisted_executor):
        """Small prompt passes preflight and LLM responds normally."""
        mock_response = _make_llm_response("正常な応答です")

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            result = await assisted_executor.execute(
                prompt="test",
                system_prompt="sys",
            )

        assert "正常な応答です" in result.text


class TestModeBToolOutputTruncation:
    """Test tool output truncation in Mode B."""

    @pytest.fixture
    def assisted_executor(self, data_dir: Path, make_anima):
        """Create an AssistedExecutor with mocked dependencies."""
        anima_dir = make_anima(
            "test-truncation",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
            max_turns=5,
        )
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
        )
        return AssistedExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            memory=memory,
        )

    async def test_large_tool_output_is_truncated(self, assisted_executor):
        """Tool returning >4096 bytes has its output truncated with marker."""
        tool_call_response = _make_llm_response(
            '検索します。\n\n```json\n{"tool": "search_memory", "arguments": {"query": "test"}}\n```'
        )
        final_response = _make_llm_response("完了しました。")

        captured_messages_list: list[list[dict[str, Any]]] = []
        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages_list.append(kwargs.get("messages", []))
            if call_count == 1:
                return tool_call_response
            return final_response

        # Tool returns a very large string (10000 chars > 4096 bytes)
        large_output = "A" * 10_000

        with (
            patch("litellm.acompletion", side_effect=mock_acompletion),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
            patch.object(
                assisted_executor._tool_handler,
                "handle",
                return_value=large_output,
            ),
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            await assisted_executor.execute(
                prompt="テスト",
                system_prompt="テスト",
            )

        assert call_count == 2
        # Second call should contain the truncated tool result
        second_call_messages = captured_messages_list[1]
        tool_result_msg = second_call_messages[-1]
        assert tool_result_msg["role"] == "user"
        assert "出力切り捨て" in tool_result_msg["content"]

    async def test_small_tool_output_not_truncated(self, assisted_executor):
        """Tool returning <4096 bytes is not truncated."""
        tool_call_response = _make_llm_response(
            '検索します。\n\n```json\n{"tool": "search_memory", "arguments": {"query": "test"}}\n```'
        )
        final_response = _make_llm_response("完了しました。")

        captured_messages_list: list[list[dict[str, Any]]] = []
        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages_list.append(kwargs.get("messages", []))
            if call_count == 1:
                return tool_call_response
            return final_response

        # Tool returns a small string (< 4096 bytes)
        small_output = "検索結果: テストデータ"

        with (
            patch("litellm.acompletion", side_effect=mock_acompletion),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
            patch.object(
                assisted_executor._tool_handler,
                "handle",
                return_value=small_output,
            ),
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            await assisted_executor.execute(
                prompt="テスト",
                system_prompt="テスト",
            )

        assert call_count == 2
        second_call_messages = captured_messages_list[1]
        tool_result_msg = second_call_messages[-1]
        assert tool_result_msg["role"] == "user"
        assert "出力切り捨て" not in tool_result_msg["content"]
        assert "検索結果: テストデータ" in tool_result_msg["content"]

    def test_truncate_function_directly(self):
        """Direct unit test for _truncate_tool_output helper."""
        # Large input
        large = "X" * 10_000
        truncated = _truncate_tool_output(large)
        assert "出力切り捨て" in truncated
        assert len(truncated.encode("utf-8")) < len(large.encode("utf-8"))

        # Small input
        small = "hello"
        result = _truncate_tool_output(small)
        assert result == small
        assert "出力切り捨て" not in result


class TestModeBTimeout:
    """Test LLM timeout and Ollama num_ctx in Mode B."""

    @pytest.fixture
    def ollama_executor(self, data_dir: Path, make_anima):
        """Create an AssistedExecutor for an Ollama model."""
        anima_dir = make_anima(
            "test-timeout-ollama",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
            max_turns=5,
        )
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
        )
        return AssistedExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            memory=memory,
        )

    @pytest.fixture
    def api_executor(self, data_dir: Path, make_anima):
        """Create an AssistedExecutor for a non-Ollama API model."""
        anima_dir = make_anima(
            "test-timeout-api",
            model="openai/gpt-4o",
            execution_mode="assisted",
            max_turns=5,
        )
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
        )
        return AssistedExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            memory=memory,
        )

    async def test_llm_call_includes_timeout_ollama(self, ollama_executor):
        """Verify litellm.acompletion is called with timeout for Ollama model.

        When llm_timeout is None (auto-detect), ollama/ models get 300s.
        """
        mock_response = _make_llm_response("テスト応答")
        # Override llm_timeout to None to trigger auto-detection
        ollama_executor._model_config.llm_timeout = None

        captured_kwargs: list[dict[str, Any]] = []

        async def capture_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return mock_response

        with (
            patch("litellm.acompletion", side_effect=capture_acompletion),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            await ollama_executor.execute(
                prompt="test",
                system_prompt="sys",
            )

        assert len(captured_kwargs) >= 1
        assert "timeout" in captured_kwargs[0]
        # Ollama models auto-detect to 300s timeout when llm_timeout is None
        assert captured_kwargs[0]["timeout"] == 300

    async def test_llm_call_includes_timeout_api(self, api_executor):
        """Verify litellm.acompletion is called with timeout for API model."""
        mock_response = _make_llm_response("テスト応答")

        captured_kwargs: list[dict[str, Any]] = []

        async def capture_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return mock_response

        with (
            patch("litellm.acompletion", side_effect=capture_acompletion),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            await api_executor.execute(
                prompt="test",
                system_prompt="sys",
            )

        assert len(captured_kwargs) >= 1
        assert "timeout" in captured_kwargs[0]
        # API models default to 600s timeout
        assert captured_kwargs[0]["timeout"] == 600

    async def test_ollama_model_includes_num_ctx(self, ollama_executor):
        """For ollama/ model, num_ctx is passed to acompletion."""
        mock_response = _make_llm_response("テスト応答")

        captured_kwargs: list[dict[str, Any]] = []

        async def capture_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return mock_response

        with (
            patch("litellm.acompletion", side_effect=capture_acompletion),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            await ollama_executor.execute(
                prompt="test",
                system_prompt="sys",
            )

        assert len(captured_kwargs) >= 1
        assert "num_ctx" in captured_kwargs[0]
        # gemma3 context window is 128000
        assert captured_kwargs[0]["num_ctx"] == 128_000

    async def test_non_ollama_model_no_num_ctx(self, api_executor):
        """For non-ollama model, num_ctx is NOT passed to acompletion."""
        mock_response = _make_llm_response("テスト応答")

        captured_kwargs: list[dict[str, Any]] = []

        async def capture_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return mock_response

        with (
            patch("litellm.acompletion", side_effect=capture_acompletion),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            await api_executor.execute(
                prompt="test",
                system_prompt="sys",
            )

        assert len(captured_kwargs) >= 1
        assert "num_ctx" not in captured_kwargs[0]


class TestModeBStreamingProtection:
    """Test preflight and truncation in execute_streaming() path."""

    @pytest.fixture
    def assisted_executor(self, data_dir: Path, make_anima):
        """Create an AssistedExecutor with mocked dependencies."""
        anima_dir = make_anima(
            "test-streaming-prot",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
            max_turns=5,
        )
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
        )
        return AssistedExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            memory=memory,
        )

    async def test_streaming_preflight_rejects_too_large(self, assisted_executor):
        """Streaming: when prompt fills context window, yields error text_delta."""
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="ollama/gemma3:27b")

        with (
            patch("litellm.acompletion", new_callable=AsyncMock),
            patch("litellm.token_counter", return_value=128_000),
            patch("core.config.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            events = []
            async for event in assisted_executor.execute_streaming(
                system_prompt="sys",
                prompt="test",
                tracker=tracker,
            ):
                events.append(event)

        # Should have a text_delta with error and a done event
        text_events = [e for e in events if e.get("type") == "text_delta"]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(text_events) >= 1
        assert "Error" in text_events[0]["text"] or "too large" in text_events[0]["text"]
        assert len(done_events) == 1

    async def test_streaming_tool_output_truncated(self, assisted_executor):
        """Streaming: large tool output is truncated before injection."""
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="ollama/gemma3:27b")

        tool_call_response = _make_llm_response(
            '```json\n{"tool": "search_memory", "arguments": {"query": "test"}}\n```'
        )
        final_response = _make_llm_response("完了しました。")

        captured_messages_list: list[list[dict[str, Any]]] = []
        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages_list.append(kwargs.get("messages", []))
            if call_count == 1:
                return tool_call_response
            return final_response

        large_output = "B" * 10_000

        with (
            patch("litellm.acompletion", side_effect=mock_acompletion),
            patch("litellm.token_counter", return_value=500),
            patch("core.config.load_config") as mock_config,
            patch.object(
                assisted_executor._tool_handler,
                "handle",
                return_value=large_output,
            ),
        ):
            mock_config.return_value = MagicMock(model_context_windows=None)
            events = []
            async for event in assisted_executor.execute_streaming(
                system_prompt="テスト",
                prompt="テスト",
                tracker=tracker,
            ):
                events.append(event)

        assert call_count == 2
        # Second call messages should have truncated tool output
        second_call_messages = captured_messages_list[1]
        tool_result_msg = second_call_messages[-1]
        assert tool_result_msg["role"] == "user"
        assert "出力切り捨て" in tool_result_msg["content"]

        # Should have done event
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
