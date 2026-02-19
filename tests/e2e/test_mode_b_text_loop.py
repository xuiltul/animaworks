# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Mode B text-based tool loop.

Validates the full flow: system prompt construction -> LLM mock ->
tool call extraction -> tool execution -> result injection -> final response.
No real API calls are made.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.assisted import AssistedExecutor
from core.memory import MemoryManager
from core.tooling.handler import ToolHandler


def _make_llm_response(content: str):
    """Create a mock LiteLLM response object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestModeBTextLoop:
    """Test the full Mode B text-based tool loop."""

    @pytest.fixture
    def assisted_executor(self, data_dir: Path, make_anima):
        """Create an AssistedExecutor with mocked dependencies."""
        anima_dir = make_anima(
            "test-b",
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

    async def test_simple_response_no_tool_call(self, assisted_executor):
        """LLM returns plain text -> no tool loop, just return."""
        mock_response = _make_llm_response("こんにちは！元気ですか？")

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await assisted_executor.execute(
                prompt="こんにちは",
                system_prompt="あなたはテスト用Animaです。",
            )

        assert "こんにちは" in result.text
        assert "元気ですか" in result.text

    async def test_single_tool_call_then_response(self, assisted_executor):
        """LLM calls a tool, gets result, then responds."""
        tool_call_response = _make_llm_response(
            '検索してみます。\n\n```json\n{"tool": "search_memory", "arguments": {"query": "テスト"}}\n```'
        )
        final_response = _make_llm_response(
            "検索結果によると、テストデータがありませんでした。"
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_call_response
            return final_response

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await assisted_executor.execute(
                prompt="テストについて教えて",
                system_prompt="あなたはテスト用Animaです。",
            )

        assert call_count == 2
        assert "テストデータがありませんでした" in result.text

    async def test_unknown_tool_gets_error_message(self, assisted_executor):
        """LLM calls unknown tool -> error injected -> LLM responds."""
        unknown_tool_response = _make_llm_response(
            '```json\n{"tool": "nonexistent_tool", "arguments": {}}\n```'
        )
        final_response = _make_llm_response(
            "すみません、そのツールは利用できません。"
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return unknown_tool_response
            return final_response

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await assisted_executor.execute(
                prompt="何か調べて",
                system_prompt="テスト",
            )

        assert call_count == 2
        # Should have received error about unknown tool
        assert "利用できません" in result.text

    async def test_max_turns_limit(self, assisted_executor):
        """Tool calls exhaust max_turns -> returns accumulated text."""
        # Every response is a tool call -> should eventually hit max_turns
        tool_call = _make_llm_response(
            '考え中...\n```json\n{"tool": "search_memory", "arguments": {"query": "loop"}}\n```'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=tool_call):
            result = await assisted_executor.execute(
                prompt="ループテスト",
                system_prompt="テスト",
            )

        # Should have accumulated the "考え中..." narrative text
        assert result.text  # Not empty

    async def test_system_prompt_includes_tool_spec(self, assisted_executor):
        """System prompt should include tool specification text."""
        captured_messages = []

        async def capture_acompletion(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _make_llm_response("テスト応答")

        with patch("litellm.acompletion", side_effect=capture_acompletion):
            await assisted_executor.execute(
                prompt="テスト",
                system_prompt="ベースプロンプト",
            )

        assert len(captured_messages) == 1
        system_msg = captured_messages[0][0]
        assert system_msg["role"] == "system"
        # Should contain both the base prompt and the tool spec
        assert "ベースプロンプト" in system_msg["content"]
        assert "利用可能なツール" in system_msg["content"]
        assert "search_memory" in system_msg["content"]

    async def test_tool_result_injected_as_user_message(self, assisted_executor):
        """After tool execution, result is injected as user message."""
        tool_call_response = _make_llm_response(
            '```json\n{"tool": "search_memory", "arguments": {"query": "test"}}\n```'
        )
        final_response = _make_llm_response("完了しました。")

        captured_messages_list = []
        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages_list.append(kwargs.get("messages", []))
            if call_count == 1:
                return tool_call_response
            return final_response

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            await assisted_executor.execute(
                prompt="テスト",
                system_prompt="テスト",
            )

        # Second call should have tool result injected
        assert len(captured_messages_list) == 2
        second_call_messages = captured_messages_list[1]
        # Should have: system, user(original), assistant(tool call), user(tool result)
        assert len(second_call_messages) == 4
        tool_result_msg = second_call_messages[3]
        assert tool_result_msg["role"] == "user"
        assert "ツール実行結果:" in tool_result_msg["content"]

    async def test_llm_api_error_returns_error_text(self, assisted_executor):
        """LiteLLM error -> returns error message."""
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API timeout")):
            result = await assisted_executor.execute(
                prompt="テスト",
                system_prompt="テスト",
            )

        assert "[LLM API Error:" in result.text
        assert "API timeout" in result.text


class TestModeBModeRouting:
    """Test that Mode B is correctly routed through AgentCore."""

    def test_mode_b_executor_is_assisted(self, make_agent_core):
        """Mode B anima creates AssistedExecutor."""
        agent = make_agent_core(
            name="test-mode-b",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )
        assert isinstance(agent._executor, AssistedExecutor)

    def test_mode_b_executor_has_tool_handler(self, make_agent_core):
        """AssistedExecutor should have a ToolHandler."""
        agent = make_agent_core(
            name="test-mode-b-handler",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )
        executor = agent._executor
        assert isinstance(executor, AssistedExecutor)
        assert executor._tool_handler is not None

    def test_mode_b_executor_knows_tools(self, make_agent_core):
        """AssistedExecutor should have a known_tools whitelist."""
        agent = make_agent_core(
            name="test-mode-b-tools",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )
        executor = agent._executor
        assert isinstance(executor, AssistedExecutor)
        assert "search_memory" in executor._known_tools
        assert "read_file" in executor._known_tools
        assert "write_memory_file" in executor._known_tools
