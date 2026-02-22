# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for uvicorn timeout settings and Agent SDK environment config.

Validates that:
- Server startup passes correct timeout/ping settings to uvicorn
- Agent SDK executor disables skill improvement in child process env
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Server uvicorn config E2E ─────────────────────────────────


class TestUvicornConfigE2E:
    """Verify uvicorn.run() receives correct timeout and WebSocket settings."""

    @patch("cli.commands.server._remove_pid_file")
    @patch("uvicorn.run")
    @patch("server.app.create_app")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.paths.get_animas_dir", return_value=Path("/tmp/animas"))
    @patch("core.init.ensure_runtime_dir")
    @patch("cli.commands.server._write_pid_file")
    @patch("cli.commands.server._kill_orphan_runners", return_value=0)
    @patch("cli.commands.server._find_server_pid_by_process", return_value=None)
    @patch("cli.commands.server._is_process_alive", return_value=False)
    @patch("cli.commands.server._read_pid", return_value=None)
    def test_server_start_passes_all_uvicorn_settings(
        self, mock_pid, mock_alive, mock_find, mock_kill,
        mock_write_pid,
        mock_ensure, mock_animas, mock_shared, mock_create, mock_uvicorn,
        mock_remove,
    ):
        """Full integration: cmd_start configures uvicorn for SSE + WebSocket."""
        from cli.commands.server import cmd_start

        mock_create.return_value = MagicMock()
        args = argparse.Namespace(host="0.0.0.0", port=18500)
        cmd_start(args)

        call_kwargs = mock_uvicorn.call_args.kwargs
        # timeout_keep_alive must be large enough for Agent SDK execution (up to 37s+)
        assert call_kwargs["timeout_keep_alive"] >= 60
        # ws_ping_interval must be shorter than timeout_keep_alive
        assert call_kwargs["ws_ping_interval"] < call_kwargs["timeout_keep_alive"]
        # ws_ping_timeout must be reasonable (not too long)
        assert 1 <= call_kwargs["ws_ping_timeout"] <= 10


# ── Agent SDK env config E2E ──────────────────────────────────


class TestAgentSDKEnvE2E:
    """Verify AgentSDKExecutor disables skill improvement in child env."""

    def test_executor_env_disables_skill_improvement(self, tmp_path):
        """Agent SDK child process env must disable Claude Code skill improvement."""
        from tests.helpers.mocks import patch_agent_sdk

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)

        from core.schemas import ModelConfig
        config = ModelConfig(
            model="claude-sonnet-4-20250514",
            api_key="sk-test",
        )

        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()

        assert env["CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT"] == "true"

    def test_executor_env_preserves_existing_keys(self, tmp_path):
        """Skill improvement flag does not break existing env keys."""
        from tests.helpers.mocks import patch_agent_sdk

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)

        from core.schemas import ModelConfig
        config = ModelConfig(
            model="claude-sonnet-4-20250514",
            api_key="sk-test-key",
            api_base_url="https://custom.api",
        )

        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()

        assert env["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)
        # A1 mode sets ANTHROPIC_API_KEY to empty string to block parent leakage
        assert env["ANTHROPIC_API_KEY"] == ""
        assert env["ANTHROPIC_BASE_URL"] == "https://custom.api"
        assert env["CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT"] == "true"
