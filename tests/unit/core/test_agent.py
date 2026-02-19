"""Unit tests for core/agent.py — AgentCore orchestrator."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.prompt.builder import BuildResult
from core.schemas import CycleResult, ModelConfig


# ── Helper to construct AgentCore with mocked dependencies ─


def _make_agent(
    anima_dir: Path,
    model: str = "claude-sonnet-4-20250514",
    execution_mode: str | None = None,
    resolved_mode: str | None = None,
):
    """Create AgentCore with all external dependencies mocked."""
    mc = ModelConfig(
        model=model,
        execution_mode=execution_mode,
        resolved_mode=resolved_mode,
        api_key="test-key",
    )
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.anima_dir = anima_dir
    messenger = MagicMock()

    with patch("core.agent.ToolHandler"), \
         patch("core.agent.AgentCore._check_sdk", return_value=False), \
         patch("core.agent.AgentCore._init_tool_registry", return_value=[]), \
         patch("core.agent.AgentCore._discover_personal_tools", return_value={}), \
         patch("core.agent.AgentCore._create_executor") as mock_create:
        mock_executor = MagicMock()
        mock_create.return_value = mock_executor
        from core.agent import AgentCore
        agent = AgentCore(anima_dir, memory, mc, messenger)
        agent._executor = mock_executor
    return agent


# ── Mode resolution ───────────────────────────────────────


class TestResolveExecutionMode:
    def test_assisted_mode(self, tmp_path):
        agent = _make_agent(tmp_path, resolved_mode="B")
        assert agent._resolve_execution_mode() == "b"

    def test_auto_non_claude_model(self, tmp_path):
        agent = _make_agent(tmp_path, model="openai/gpt-4o")
        assert agent._resolve_execution_mode() == "a2"

    def test_auto_claude_without_sdk(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-20250514")
        agent._sdk_available = False
        assert agent._resolve_execution_mode() == "a2"

    def test_auto_claude_with_sdk(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-20250514")
        agent._sdk_available = True
        assert agent._resolve_execution_mode() == "a1"

    def test_autonomous_explicit_non_claude(self, tmp_path):
        agent = _make_agent(tmp_path, model="openai/gpt-4o", execution_mode="autonomous")
        assert agent._resolve_execution_mode() == "a2"


class TestIsClaude:
    def test_claude_prefix(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-20250514")
        assert agent._is_claude_model() is True

    def test_anthropic_prefix(self, tmp_path):
        agent = _make_agent(tmp_path, model="anthropic/claude-haiku-3.5")
        assert agent._is_claude_model() is True

    def test_non_claude(self, tmp_path):
        agent = _make_agent(tmp_path, model="openai/gpt-4o")
        assert agent._is_claude_model() is False


# ── Callbacks / reply tracking ────────────────────────────


class TestCallbacksAndReply:
    def test_set_on_message_sent(self, tmp_path):
        agent = _make_agent(tmp_path)
        fn = MagicMock()
        agent.set_on_message_sent(fn)

    def test_reset_reply_tracking(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.reset_reply_tracking()
        agent._tool_handler.reset_replied_to.assert_called_once()

    def test_replied_to(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._tool_handler.replied_to = {"bob", "charlie"}
        assert agent.replied_to == {"bob", "charlie"}


# ── resolve_api_key ───────────────────────────────────────


class TestResolveApiKey:
    def test_direct_key(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.model_config.api_key = "sk-direct"
        assert agent._resolve_api_key() == "sk-direct"

    def test_env_fallback(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.model_config.api_key = None
        agent.model_config.api_key_env = "MY_KEY"
        with patch.dict("os.environ", {"MY_KEY": "sk-env"}):
            assert agent._resolve_api_key() == "sk-env"

    def test_none_when_no_key(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.model_config.api_key = None
        agent.model_config.api_key_env = "NONEXISTENT_KEY"
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("NONEXISTENT_KEY", None)
            assert agent._resolve_api_key() is None


# ── run_cycle ─────────────────────────────────────────────


class TestRunCycle:
    async def test_mode_b_returns_result(self, tmp_path):
        agent = _make_agent(tmp_path, resolved_mode="B")
        mock_result = MagicMock()
        mock_result.text = "Assisted response"
        agent._executor.execute = AsyncMock(return_value=mock_result)

        mock_build_result = BuildResult(system_prompt="sysprompt")
        with patch("core.agent.build_system_prompt", return_value=mock_build_result), \
             patch("core.agent.inject_shortterm", return_value="sysprompt"), \
             patch("core.agent.ShortTermMemory") as MockST:
            MockST.return_value.has_pending.return_value = False
            MockST.return_value.clear = MagicMock()

            result = await agent.run_cycle("Hello", trigger="test")
            assert isinstance(result, CycleResult)
            assert result.summary == "Assisted response"
            assert result.trigger == "test"

    async def test_mode_a2_returns_result(self, tmp_path):
        agent = _make_agent(tmp_path, model="openai/gpt-4o")
        mock_result = MagicMock()
        mock_result.text = "A2 response"
        agent._executor.execute = AsyncMock(return_value=mock_result)

        mock_build_result = BuildResult(system_prompt="sysprompt")
        with patch("core.agent.build_system_prompt", return_value=mock_build_result), \
             patch("core.agent.inject_shortterm", return_value="sysprompt"), \
             patch("core.agent.ShortTermMemory") as MockST:
            MockST.return_value.has_pending.return_value = False
            MockST.return_value.clear = MagicMock()

            result = await agent.run_cycle("Hello", trigger="test")
            assert isinstance(result, CycleResult)
            assert result.summary == "A2 response"

    async def test_mode_a1_returns_result(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-20250514")
        agent._sdk_available = True
        # Force mode a1
        agent._resolve_execution_mode = MagicMock(return_value="a1")

        mock_result = MagicMock()
        mock_result.text = "A1 response"
        mock_result.result_message = MagicMock()
        mock_result.result_message.num_turns = 3
        mock_result.result_message.session_id = "sess-1"
        agent._executor.execute = AsyncMock(return_value=mock_result)

        mock_build_result = BuildResult(system_prompt="sysprompt")
        with patch("core.agent.build_system_prompt", return_value=mock_build_result), \
             patch("core.agent.inject_shortterm", return_value="sysprompt"), \
             patch("core.agent.ShortTermMemory") as MockST, \
             patch("core.agent.ContextTracker") as MockCT:
            MockST.return_value.has_pending.return_value = False
            MockST.return_value.clear = MagicMock()
            MockCT.return_value.threshold_exceeded = False
            MockCT.return_value.usage_ratio = 0.3

            result = await agent.run_cycle("Hello", trigger="manual")
            assert isinstance(result, CycleResult)
            assert result.summary == "A1 response"
            assert result.total_turns == 3


# ── Permission parsing ────────────────────────────────────


class TestPermissionParsing:
    def test_permission_regex(self):
        from core.agent import AgentCore
        pattern = AgentCore._PERMISSION_RE
        assert pattern.match("- web_search: OK")
        assert pattern.match("* slack: yes")
        assert pattern.match("  gmail: enabled")
        assert pattern.match("github: true")
        assert not pattern.match("- web_search: no")
        assert not pattern.match("# heading")
