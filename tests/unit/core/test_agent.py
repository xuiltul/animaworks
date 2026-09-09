"""Unit tests for core/agent.py — AgentCore orchestrator."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.base import ExecutionResult
from core.prompt.builder import BuildResult
from core.schemas import CycleResult, ModelConfig

# ── Helper to construct AgentCore with mocked dependencies ─


def _make_agent(
    anima_dir: Path,
    model: str = "claude-sonnet-4-6",
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

    with (
        patch("core.agent.ToolHandler"),
        patch("core.agent.AgentCore._check_sdk", return_value=False),
        patch("core.agent.AgentCore._init_tool_registry", return_value=[]),
        patch("core.agent.AgentCore._discover_personal_tools", return_value={}),
        patch("core.agent.AgentCore._create_executor") as mock_create,
    ):
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
        assert agent._resolve_execution_mode() == "a"

    def test_auto_claude_without_sdk(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-6")
        agent._sdk_available = False
        # Availability must not silently change the configured auth realm.
        assert agent._resolve_execution_mode() == "s"

    def test_auto_claude_with_sdk(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-6")
        agent._sdk_available = True
        assert agent._resolve_execution_mode() == "s"

    def test_autonomous_explicit_non_claude(self, tmp_path):
        agent = _make_agent(tmp_path, model="openai/gpt-4o", execution_mode="autonomous")
        assert agent._resolve_execution_mode() == "a"


class TestIsClaude:
    def test_claude_prefix(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-6")
        assert agent._is_claude_model() is True

    def test_anthropic_prefix(self, tmp_path):
        agent = _make_agent(tmp_path, model="anthropic/claude-haiku-3.5")
        assert agent._is_claude_model() is True

    def test_non_claude(self, tmp_path):
        agent = _make_agent(tmp_path, model="openai/gpt-4o")
        assert agent._is_claude_model() is False


# ── Mode C fallback ───────────────────────────────────────


class TestModeCFallback:
    def test_mode_c_prefers_configured_fallback_when_sdk_missing(self, tmp_path):
        agent = _make_agent(tmp_path, model="codex/gpt-5.3-codex", resolved_mode="C")
        agent.model_config.fallback_models = ["a:openai/gpt-4o"]
        fallback = agent.model_config.model_copy(
            update={"model": "openai/gpt-4o", "execution_mode": "A", "resolved_mode": "A"}
        )
        sentinel = MagicMock()
        guard = MagicMock()
        guard.config.default_block_seconds = 60

        with (
            patch("core.execution.codex_sdk.is_codex_sdk_available", return_value=False),
            patch("core.execution.rate_guard.get_rate_guard", return_value=guard),
            patch("core.config.model_config.resolve_effective_model_config", return_value=fallback),
            patch("core.execution.fallback_activity.log_model_fallback") as log_fallback,
            patch("core.execution.LiteLLMExecutor", return_value=sentinel) as litellm,
        ):
            created = agent._create_executor()

        assert created is sentinel
        assert litellm.call_args.kwargs["model_config"] == fallback
        log_fallback.assert_called_once()

    def test_mode_c_without_configured_fallback_fails_explicitly(self, tmp_path):
        from core.exceptions import ExecutorUnavailableError

        agent = _make_agent(tmp_path, model="codex/gpt-5.3-codex", resolved_mode="C")
        agent.model_config.api_key_env = "OPENAI_API_KEY"
        with (
            patch("core.execution.codex_sdk.is_codex_sdk_available", return_value=False),
            patch("core.execution.LiteLLMExecutor") as litellm,
            pytest.raises(ExecutorUnavailableError, match="fallback_models"),
        ):
            agent._create_executor()
        litellm.assert_not_called()

    def test_mode_c_never_guesses_model_from_api_key_shape(self, tmp_path):
        from core.exceptions import ExecutorUnavailableError

        agent = _make_agent(tmp_path, model="codex/gpt-5.3-codex", resolved_mode="C")
        agent.model_config.api_key = "sk-ant-test"
        with (
            patch("core.execution.codex_sdk.is_codex_sdk_available", return_value=False),
            patch("core.execution.LiteLLMExecutor") as litellm,
            pytest.raises(ExecutorUnavailableError),
        ):
            agent._create_executor()
        litellm.assert_not_called()
        assert agent.model_config.model == "codex/gpt-5.3-codex"


# ── Mode X executor / fallback ────────────────────────────


def test_missing_claude_sdk_does_not_construct_unusable_adapter(tmp_path):
    from core.exceptions import ExecutorUnavailableError

    agent = _make_agent(tmp_path, model="claude-sonnet-4-6", resolved_mode="S")
    with (
        patch("core.execution.agent_sdk.AgentSDKExecutor") as sdk,
        pytest.raises(ExecutorUnavailableError),
    ):
        agent._create_executor()
    sdk.assert_not_called()


def test_missing_claude_sdk_uses_only_configured_fallback(tmp_path):
    agent = _make_agent(tmp_path, model="claude-sonnet-4-6", resolved_mode="S")
    fallback = agent.model_config.model_copy(update={"model": "openai/backup", "resolved_mode": "A"})
    with (
        patch("core.config.model_config.resolve_unavailable_model_config", return_value=fallback) as resolve,
        patch("core.execution.fallback_activity.log_model_fallback"),
        patch("core.execution.LiteLLMExecutor") as adapter,
    ):
        assert agent._create_executor() is adapter.return_value
    resolve.assert_called_once_with(agent.model_config, unavailable_modes=frozenset({"S"}))


class TestModeXExecutor:
    def test_mode_x_prefers_configured_fallback_when_cli_missing(self, tmp_path):
        agent = _make_agent(tmp_path, model="grok/grok-4.5", resolved_mode="X")
        agent.model_config.fallback_models = ["a:openai/gpt-4o"]
        fallback = agent.model_config.model_copy(
            update={"model": "openai/gpt-4o", "execution_mode": "A", "resolved_mode": "A"}
        )
        sentinel = MagicMock()
        guard = MagicMock()
        guard.config.default_block_seconds = 60

        with (
            patch("core.execution.grok_cli.is_grok_cli_available", return_value=False),
            patch("core.execution.rate_guard.get_rate_guard", return_value=guard),
            patch("core.config.model_config.resolve_effective_model_config", return_value=fallback),
            patch("core.execution.fallback_activity.log_model_fallback") as log_fallback,
            patch("core.execution.LiteLLMExecutor", return_value=sentinel) as litellm,
        ):
            created = agent._create_executor()

        assert created is sentinel
        assert litellm.call_args.kwargs["model_config"] == fallback
        log_fallback.assert_called_once()

    def test_mode_x_creates_grok_executor_with_runtime_dependencies(self, tmp_path):
        agent = _make_agent(
            tmp_path,
            model="grok/grok-4.5",
            resolved_mode="X",
        )
        sentinel_executor = MagicMock(name="grok_executor")

        with (
            patch("core.execution.grok_cli.is_grok_cli_available", return_value=True),
            patch("core.execution.grok_cli.GrokCLIExecutor", return_value=sentinel_executor) as mock_grok,
        ):
            created = agent._create_executor()

        assert created is sentinel_executor
        mock_grok.assert_called_once_with(
            model_config=agent.model_config,
            anima_dir=tmp_path,
            tool_registry=agent._tool_registry,
            personal_tools=agent._personal_tools,
            interrupt_event=agent._interrupt_event,
        )

    def test_mode_x_does_not_implicitly_switch_to_paid_api(self, tmp_path):
        from core.exceptions import ExecutorUnavailableError

        agent = _make_agent(tmp_path, model="grok/grok-4.5", resolved_mode="X")
        with (
            patch("core.execution.grok_cli.is_grok_cli_available", return_value=False),
            patch("core.execution.LiteLLMExecutor") as litellm,
            pytest.raises(ExecutorUnavailableError),
        ):
            agent._create_executor()
        litellm.assert_not_called()
        assert agent.model_config.model == "grok/grok-4.5"


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
    async def test_mode_x_terminal_error_propagates_action_and_reason(self, tmp_path):
        agent = _make_agent(tmp_path, model="grok/grok-4.5", resolved_mode="X")
        agent._executor.execute = AsyncMock(return_value=ExecutionResult(text="", error=True, reason="quota_exhausted"))

        with (
            patch("core._agent_cycle.build_system_prompt", return_value=BuildResult(system_prompt="sysprompt")),
            patch("core._agent_cycle.ShortTermMemory") as shortterm,
        ):
            shortterm.return_value.has_pending.return_value = False
            result = await agent.run_cycle("Hello", trigger="cron:test")

        assert result.action == "error"
        assert result.reason == "quota_exhausted"
        assert result.stop_kind == "stream_error"

    async def test_mode_b_returns_result(self, tmp_path):
        agent = _make_agent(tmp_path, resolved_mode="B")
        mock_result = MagicMock()
        mock_result.text = "Assisted response"
        mock_result.usage = None
        agent._executor.execute = AsyncMock(return_value=mock_result)

        mock_build_result = BuildResult(system_prompt="sysprompt")
        with (
            patch("core._agent_cycle.build_system_prompt", return_value=mock_build_result),
            patch("core._agent_cycle.inject_shortterm", return_value="sysprompt"),
            patch("core._agent_cycle.ShortTermMemory") as MockST,
        ):
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
        mock_result.usage = None
        agent._executor.execute = AsyncMock(return_value=mock_result)

        mock_build_result = BuildResult(system_prompt="sysprompt")
        with (
            patch("core._agent_cycle.build_system_prompt", return_value=mock_build_result),
            patch("core._agent_cycle.inject_shortterm", return_value="sysprompt"),
            patch("core._agent_cycle.ShortTermMemory") as MockST,
        ):
            MockST.return_value.has_pending.return_value = False
            MockST.return_value.clear = MagicMock()

            result = await agent.run_cycle("Hello", trigger="test")
            assert isinstance(result, CycleResult)
            assert result.summary == "A2 response"
            assert agent._executor.execute.call_args.kwargs["trigger"] == "test"

    async def test_mode_s_returns_result(self, tmp_path):
        agent = _make_agent(tmp_path, model="claude-sonnet-4-6")
        agent._sdk_available = True
        # Force mode s (SDK mode)
        agent._resolve_execution_mode = MagicMock(return_value="s")

        mock_result = MagicMock()
        mock_result.text = "S mode response"
        mock_result.usage = None
        mock_result.result_message = MagicMock()
        mock_result.result_message.num_turns = 3
        mock_result.result_message.session_id = "sess-1"
        agent._executor.execute = AsyncMock(return_value=mock_result)

        mock_build_result = BuildResult(system_prompt="sysprompt")
        with (
            patch("core._agent_cycle.build_system_prompt", return_value=mock_build_result),
            patch("core._agent_cycle.inject_shortterm", return_value="sysprompt"),
            patch("core._agent_cycle.ShortTermMemory") as MockST,
            patch("core._agent_cycle.ContextTracker") as MockCT,
        ):
            MockST.return_value.has_pending.return_value = False
            MockST.return_value.clear = MagicMock()
            MockCT.return_value.threshold_exceeded = False
            MockCT.return_value.usage_ratio = 0.3

            result = await agent.run_cycle("Hello", trigger="manual")
            assert isinstance(result, CycleResult)
            assert result.summary == "S mode response"
            assert result.total_turns == 3


# ── Permission parsing ────────────────────────────────────


class TestPermissionParsing:
    def test_permission_regex(self):
        from core.tooling.permissions import (
            _PERMISSION_ALLOW_RE,
            _PERMISSION_DENY_RE,
        )

        assert _PERMISSION_ALLOW_RE.match("- web_search: OK")
        assert _PERMISSION_ALLOW_RE.match("* slack: yes")
        assert _PERMISSION_ALLOW_RE.match("  gmail: enabled")
        assert _PERMISSION_ALLOW_RE.match("github: true")
        assert _PERMISSION_ALLOW_RE.match("- chatwork: 全権限")
        assert not _PERMISSION_ALLOW_RE.match("- web_search: no")
        assert not _PERMISSION_ALLOW_RE.match("# heading")
        assert _PERMISSION_DENY_RE.match("- chatwork: no")
        assert _PERMISSION_DENY_RE.match("* gmail: disabled")
        assert not _PERMISSION_DENY_RE.match("- slack: yes")
