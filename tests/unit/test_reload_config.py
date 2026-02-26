"""Unit tests for status.json hot-reload — DigitalAnima.reload_config(),
AgentCore.update_model_config(), IPC handler, and CLI command.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import ModelConfig


# ── DigitalAnima.reload_config ──────────────────────────────


class TestAnimaReloadConfig:
    """Test DigitalAnima.reload_config() detects and applies changes."""

    def _make_anima_mock(
        self, old_model: str = "claude-sonnet-4-6", new_model: str = "claude-opus-4-6"
    ) -> MagicMock:
        from core.anima import DigitalAnima

        anima = MagicMock(spec=DigitalAnima)
        anima.model_config = ModelConfig(model=old_model, max_tokens=4096)
        anima.memory = MagicMock()
        anima.memory.read_model_config.return_value = ModelConfig(
            model=new_model, max_tokens=16384
        )
        anima.agent = MagicMock()
        anima.reload_config = DigitalAnima.reload_config.__get__(anima)
        return anima

    def test_reload_detects_model_change(self):
        anima = self._make_anima_mock()
        result = anima.reload_config()

        assert result["status"] == "ok"
        assert result["model"] == "claude-opus-4-6"
        assert "model" in result["changes"]
        assert "max_tokens" in result["changes"]

    def test_reload_updates_model_config(self):
        anima = self._make_anima_mock()
        anima.reload_config()

        assert anima.model_config.model == "claude-opus-4-6"
        assert anima.model_config.max_tokens == 16384

    def test_reload_propagates_to_agent(self):
        anima = self._make_anima_mock()
        anima.reload_config()

        anima.agent.update_model_config.assert_called_once()
        new_config = anima.agent.update_model_config.call_args[0][0]
        assert new_config.model == "claude-opus-4-6"

    def test_reload_no_changes(self):
        anima = self._make_anima_mock(
            old_model="claude-sonnet-4-6", new_model="claude-sonnet-4-6"
        )
        anima.memory.read_model_config.return_value = ModelConfig(
            model="claude-sonnet-4-6", max_tokens=4096
        )
        result = anima.reload_config()

        assert result["status"] == "ok"
        assert result["changes"] == []


# ── AgentCore.update_model_config ───────────────────────────


class TestAgentUpdateModelConfig:
    """Test AgentCore.update_model_config() updates internal state."""

    @patch("core.agent.AgentCore._create_executor")
    @patch("core.agent.AgentCore._check_sdk", return_value=False)
    @patch("core.agent.AgentCore._build_human_notifier", return_value=None)
    @patch("core.agent.AgentCore._build_background_manager", return_value=None)
    @patch("core.agent.AgentCore._init_tool_registry", return_value=[])
    @patch("core.agent.AgentCore._discover_personal_tools", return_value={})
    @patch("core.agent.AgentCore._is_debug_superuser", return_value=False)
    def test_update_rebuilds_executor(
        self,
        _sup, _tools, _personal, _registry, _bg, _notifier, mock_create_executor,
        tmp_path,
    ):
        from core.agent import AgentCore
        from core.memory import MemoryManager

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# test", encoding="utf-8")

        memory = MagicMock(spec=MemoryManager)
        old_config = ModelConfig(model="claude-sonnet-4-6")
        agent = AgentCore(anima_dir, memory, old_config)

        call_count_before = mock_create_executor.call_count

        new_config = ModelConfig(model="claude-opus-4-6")
        agent.update_model_config(new_config)

        assert agent.model_config.model == "claude-opus-4-6"
        assert mock_create_executor.call_count == call_count_before + 1

    @patch("core.agent.AgentCore._create_executor")
    @patch("core.agent.AgentCore._check_sdk", return_value=False)
    @patch("core.agent.AgentCore._build_human_notifier", return_value=None)
    @patch("core.agent.AgentCore._build_background_manager", return_value=None)
    @patch("core.agent.AgentCore._init_tool_registry", return_value=[])
    @patch("core.agent.AgentCore._discover_personal_tools", return_value={})
    @patch("core.agent.AgentCore._is_debug_superuser", return_value=False)
    def test_update_refreshes_context_window(
        self,
        _sup, _tools, _personal, _registry, _bg, _notifier, _exec,
        tmp_path,
    ):
        from core.agent import AgentCore
        from core.memory import MemoryManager

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# test", encoding="utf-8")

        memory = MagicMock(spec=MemoryManager)
        agent = AgentCore(anima_dir, memory, ModelConfig(model="claude-sonnet-4-6"))
        old_cw = agent._tool_handler._context_window

        with patch("core.config.models.resolve_context_window", return_value=200_000):
            agent.update_model_config(ModelConfig(model="claude-opus-4-6"))

        assert agent._tool_handler._context_window == 200_000


# ── IPC Handler ─────────────────────────────────────────────


class TestIPCReloadHandler:
    """Test AnimaRunner._handle_reload_config IPC handler."""

    @pytest.mark.asyncio
    async def test_handler_calls_anima_reload(self):
        from core.supervisor.runner import AnimaRunner

        runner = MagicMock(spec=AnimaRunner)
        runner.anima = MagicMock()
        runner.anima.reload_config.return_value = {
            "status": "ok", "model": "test", "changes": ["model"]
        }

        handler = AnimaRunner._handle_reload_config.__get__(runner)
        result = await handler({})

        runner.anima.reload_config.assert_called_once()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_handler_raises_when_no_anima(self):
        from core.supervisor.runner import AnimaRunner

        runner = MagicMock(spec=AnimaRunner)
        runner.anima = None

        handler = AnimaRunner._handle_reload_config.__get__(runner)
        with pytest.raises(RuntimeError, match="Anima not initialized"):
            await handler({})


# ── CLI Command ─────────────────────────────────────────────


class TestCLIReloadCommand:
    """Test cmd_anima_reload() HTTP calls."""

    def test_single_reload(self, tmp_path):
        import requests as _req_mod
        from cli.commands.anima_mgmt import cmd_anima_reload

        pid_file = tmp_path / "server.pid"
        pid_file.write_text("12345")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok", "model": "claude-opus-4-6", "changes": ["model"]
        }
        mock_response.raise_for_status = MagicMock()

        args = argparse.Namespace(
            anima="kotoha",
            all=False,
            gateway_url="http://localhost:18500",
        )

        with (
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch.object(_req_mod, "post", return_value=mock_response) as mock_post,
        ):
            cmd_anima_reload(args)

        mock_post.assert_called_once_with(
            "http://localhost:18500/api/animas/kotoha/reload",
            timeout=10.0,
        )

    def test_reload_all(self, tmp_path):
        import requests as _req_mod
        from cli.commands.anima_mgmt import cmd_anima_reload

        pid_file = tmp_path / "server.pid"
        pid_file.write_text("12345")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "results": {"alice": {"status": "ok", "model": "m", "changes": []}},
        }
        mock_response.raise_for_status = MagicMock()

        args = argparse.Namespace(
            anima=None,
            all=True,
            gateway_url="http://localhost:18500",
        )

        with (
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch.object(_req_mod, "post", return_value=mock_response) as mock_post,
        ):
            cmd_anima_reload(args)

        mock_post.assert_called_once_with(
            "http://localhost:18500/api/animas/reload-all",
            timeout=30.0,
        )
