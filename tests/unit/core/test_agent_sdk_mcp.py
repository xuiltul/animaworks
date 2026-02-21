"""Unit tests for AgentSDKExecutor MCP integration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from core.schemas import ModelConfig


# ── Helpers ──────────────────────────────────────────────────

def _make_executor(
    anima_dir: Path,
    model: str = "claude-sonnet-4-20250514",
) -> "AgentSDKExecutor":
    """Create an AgentSDKExecutor with minimal config."""
    from core.execution.agent_sdk import AgentSDKExecutor

    mc = ModelConfig(model=model, api_key="test-key")
    return AgentSDKExecutor(model_config=mc, anima_dir=anima_dir)


# ── TestBuildMcpEnv ──────────────────────────────────────────


class TestBuildMcpEnv:
    """Tests for AgentSDKExecutor._build_mcp_env()."""

    def test_returns_anima_dir(self, tmp_path: Path) -> None:
        """ANIMAWORKS_ANIMA_DIR is set to the anima directory."""
        executor = _make_executor(tmp_path / "animas" / "test-anima")
        env = executor._build_mcp_env()
        assert env["ANIMAWORKS_ANIMA_DIR"] == str(tmp_path / "animas" / "test-anima")

    def test_returns_project_dir(self, tmp_path: Path) -> None:
        """ANIMAWORKS_PROJECT_DIR is set to PROJECT_DIR."""
        from core.paths import PROJECT_DIR

        executor = _make_executor(tmp_path / "animas" / "test-anima")
        env = executor._build_mcp_env()
        assert env["ANIMAWORKS_PROJECT_DIR"] == str(PROJECT_DIR)

    def test_returns_pythonpath(self, tmp_path: Path) -> None:
        """PYTHONPATH is set to PROJECT_DIR."""
        from core.paths import PROJECT_DIR

        executor = _make_executor(tmp_path / "animas" / "test-anima")
        env = executor._build_mcp_env()
        assert env["PYTHONPATH"] == str(PROJECT_DIR)

    def test_returns_path_from_os_environ(self, tmp_path: Path) -> None:
        """PATH is taken from os.environ."""
        executor = _make_executor(tmp_path / "animas" / "test-anima")
        with patch.dict(os.environ, {"PATH": "/custom/bin:/other/bin"}):
            env = executor._build_mcp_env()
        assert env["PATH"] == "/custom/bin:/other/bin"

    def test_path_fallback_when_missing(self, tmp_path: Path) -> None:
        """When PATH is missing from os.environ, falls back to /usr/bin:/bin."""
        executor = _make_executor(tmp_path / "animas" / "test-anima")
        env_copy = os.environ.copy()
        env_copy.pop("PATH", None)
        with patch.dict(os.environ, env_copy, clear=True):
            env = executor._build_mcp_env()
        assert env["PATH"] == "/usr/bin:/bin"


# ── TestMcpServerConfig ─────────────────────────────────────


class TestMcpServerConfig:
    """Tests for MCP server configuration passed to ClaudeAgentOptions in execute()."""

    @pytest.fixture
    def mock_sdk(self):
        """Provide a comprehensive mock of claude_agent_sdk module."""
        # Create mock module with all required classes
        mock_module = MagicMock()

        # Create named mock classes so isinstance checks work
        mock_module.AssistantMessage = type("AssistantMessage", (), {})
        mock_module.ResultMessage = type("ResultMessage", (), {})
        mock_module.SystemMessage = type("SystemMessage", (), {})
        mock_module.TextBlock = type("TextBlock", (), {})
        mock_module.HookMatcher = MagicMock()

        # ClaudeAgentOptions: capture kwargs on construction
        captured_options = {}

        class FakeClaudeAgentOptions:
            def __init__(self, **kwargs):
                captured_options.update(kwargs)

        mock_module.ClaudeAgentOptions = FakeClaudeAgentOptions

        # query: return an async iterator that yields a ResultMessage
        result_msg = MagicMock()
        result_msg.usage = {"input_tokens": 100, "output_tokens": 50}

        async def fake_query(**kwargs):
            # Yield a ResultMessage-like object
            msg = MagicMock(spec=[])
            # Make it an instance of our mock ResultMessage via type tag
            msg.__class__ = mock_module.ResultMessage
            msg.usage = {"input_tokens": 100, "output_tokens": 50}
            yield msg

        mock_module.query = fake_query

        # Types sub-module
        mock_types = MagicMock()
        mock_types.HookContext = MagicMock()
        mock_types.HookInput = MagicMock()
        mock_types.PostToolUseHookSpecificOutput = MagicMock()
        mock_types.SyncHookJSONOutput = MagicMock(return_value=MagicMock())
        mock_types.PreToolUseHookSpecificOutput = MagicMock()

        return mock_module, mock_types, captured_options

    @pytest.mark.asyncio
    async def test_options_include_mcp_servers_with_aw(
        self, tmp_path: Path, mock_sdk,
    ) -> None:
        """ClaudeAgentOptions is called with mcp_servers containing 'aw' server."""
        mock_module, mock_types, captured_options = mock_sdk
        executor = _make_executor(tmp_path / "animas" / "test-anima")

        with patch.dict("sys.modules", {
            "claude_agent_sdk": mock_module,
            "claude_agent_sdk.types": mock_types,
        }):
            await executor.execute(prompt="hello", system_prompt="sys")

        assert "mcp_servers" in captured_options
        assert "aw" in captured_options["mcp_servers"]

    @pytest.mark.asyncio
    async def test_aw_server_uses_sys_executable(
        self, tmp_path: Path, mock_sdk,
    ) -> None:
        """The 'aw' MCP server uses sys.executable as command."""
        mock_module, mock_types, captured_options = mock_sdk
        executor = _make_executor(tmp_path / "animas" / "test-anima")

        with patch.dict("sys.modules", {
            "claude_agent_sdk": mock_module,
            "claude_agent_sdk.types": mock_types,
        }):
            await executor.execute(prompt="hello", system_prompt="sys")

        aw_config = captured_options["mcp_servers"]["aw"]
        assert aw_config["command"] == sys.executable

    @pytest.mark.asyncio
    async def test_aw_server_uses_correct_args(
        self, tmp_path: Path, mock_sdk,
    ) -> None:
        """The 'aw' MCP server uses ['-m', 'core.mcp.server'] as args."""
        mock_module, mock_types, captured_options = mock_sdk
        executor = _make_executor(tmp_path / "animas" / "test-anima")

        with patch.dict("sys.modules", {
            "claude_agent_sdk": mock_module,
            "claude_agent_sdk.types": mock_types,
        }):
            await executor.execute(prompt="hello", system_prompt="sys")

        aw_config = captured_options["mcp_servers"]["aw"]
        assert aw_config["args"] == ["-m", "core.mcp.server"]

    @pytest.mark.asyncio
    async def test_allowed_tools_include_mcp_wildcard(
        self, tmp_path: Path, mock_sdk,
    ) -> None:
        """allowed_tools includes 'mcp__aw__*' wildcard."""
        mock_module, mock_types, captured_options = mock_sdk
        executor = _make_executor(tmp_path / "animas" / "test-anima")

        with patch.dict("sys.modules", {
            "claude_agent_sdk": mock_module,
            "claude_agent_sdk.types": mock_types,
        }):
            await executor.execute(prompt="hello", system_prompt="sys")

        assert "mcp__aw__*" in captured_options["allowed_tools"]


# ── TestBashSendRemoved ─────────────────────────────────────


class TestBashSendRemoved:
    """Verify that bash send tracking has been removed from the module."""

    def test_bash_send_re_not_on_module(self) -> None:
        """_BASH_SEND_RE attribute does NOT exist on the module."""
        import core.execution.agent_sdk as mod

        assert not hasattr(mod, "_BASH_SEND_RE")

    def test_check_unconfirmed_sends_not_on_class(self) -> None:
        """_check_unconfirmed_sends method does NOT exist on AgentSDKExecutor."""
        from core.execution.agent_sdk import AgentSDKExecutor

        assert not hasattr(AgentSDKExecutor, "_check_unconfirmed_sends")

    def test_pending_sends_not_in_pre_tool_hook_signature(self) -> None:
        """pending_sends is NOT in the _build_pre_tool_hook signature."""
        from core.execution.agent_sdk import _build_pre_tool_hook

        sig = inspect.signature(_build_pre_tool_hook)
        assert "pending_sends" not in sig.parameters


# ── TestMcpStatusLogging ────────────────────────────────────


class TestMcpStatusLogging:
    """Tests for SystemMessage MCP server status handling in execute()."""

    @pytest.fixture
    def mock_sdk_with_system_message(self):
        """Provide a mock SDK that yields a SystemMessage with MCP status."""
        mock_module = MagicMock()

        # Real-ish classes for isinstance checks
        mock_module.AssistantMessage = type("AssistantMessage", (), {})
        mock_module.ResultMessage = type("ResultMessage", (), {})
        mock_module.TextBlock = type("TextBlock", (), {})
        mock_module.HookMatcher = MagicMock()

        # SystemMessage with controllable subtype and data
        class FakeSystemMessage:
            def __init__(self, subtype: str, data: dict | None):
                self.subtype = subtype
                self.data = data

        mock_module.SystemMessage = FakeSystemMessage

        class FakeClaudeAgentOptions:
            def __init__(self, **kwargs):
                pass

        mock_module.ClaudeAgentOptions = FakeClaudeAgentOptions

        mock_types = MagicMock()
        mock_types.SyncHookJSONOutput = MagicMock(return_value=MagicMock())
        mock_types.PreToolUseHookSpecificOutput = MagicMock()

        return mock_module, mock_types, FakeSystemMessage

    @pytest.mark.asyncio
    async def test_connected_status_logs_info(
        self, tmp_path: Path, mock_sdk_with_system_message,
    ) -> None:
        """When MCP server status is 'connected', should log info."""
        mock_module, mock_types, FakeSystemMessage = mock_sdk_with_system_message

        sys_msg = FakeSystemMessage(
            subtype="init",
            data={"mcp_servers": [{"name": "aw", "status": "connected"}]},
        )
        result_msg = MagicMock()
        result_msg.__class__ = mock_module.ResultMessage
        result_msg.usage = {"input_tokens": 100, "output_tokens": 50}

        async def fake_query(**kwargs):
            yield sys_msg
            yield result_msg

        mock_module.query = fake_query

        executor = _make_executor(tmp_path / "animas" / "test-anima")

        with patch.dict("sys.modules", {
            "claude_agent_sdk": mock_module,
            "claude_agent_sdk.types": mock_types,
        }), patch(
            "core.execution.agent_sdk.logger"
        ) as mock_logger:
            await executor.execute(prompt="hello", system_prompt="sys")

        mock_logger.info.assert_any_call(
            "MCP server '%s' connected successfully", "aw",
        )
        # Ensure error was NOT called for the MCP server status
        for c in mock_logger.error.call_args_list:
            assert "MCP server" not in str(c) or "failed to connect" not in str(c)

    @pytest.mark.asyncio
    async def test_non_connected_status_logs_error(
        self, tmp_path: Path, mock_sdk_with_system_message,
    ) -> None:
        """When MCP server status is not 'connected', should log error."""
        mock_module, mock_types, FakeSystemMessage = mock_sdk_with_system_message

        sys_msg = FakeSystemMessage(
            subtype="init",
            data={"mcp_servers": [{"name": "aw", "status": "error"}]},
        )
        result_msg = MagicMock()
        result_msg.__class__ = mock_module.ResultMessage
        result_msg.usage = {"input_tokens": 100, "output_tokens": 50}

        async def fake_query(**kwargs):
            yield sys_msg
            yield result_msg

        mock_module.query = fake_query

        executor = _make_executor(tmp_path / "animas" / "test-anima")

        with patch.dict("sys.modules", {
            "claude_agent_sdk": mock_module,
            "claude_agent_sdk.types": mock_types,
        }), patch(
            "core.execution.agent_sdk.logger"
        ) as mock_logger:
            await executor.execute(prompt="hello", system_prompt="sys")

        mock_logger.error.assert_any_call(
            "MCP server '%s' failed to connect: status=%s",
            "aw", "error",
        )
