"""Unit tests for A1 mode environment and MCP configuration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from core.execution.agent_sdk import AgentSDKExecutor
from core.schemas import ModelConfig


# ── _build_env() ─────────────────────────────────────────


class TestBuildEnvPathAndProjectDir:
    """Verify _build_env() exposes anima_dir in PATH and sets PROJECT_DIR."""

    def _make_executor(self, anima_dir: Path) -> AgentSDKExecutor:
        mc = ModelConfig(model="claude-sonnet-4-20250514")
        return AgentSDKExecutor(model_config=mc, anima_dir=anima_dir)

    def test_anima_dir_in_path(self, tmp_path: Path) -> None:
        """PATH should start with anima_dir so tools are discoverable."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_env()

        assert "PATH" in env
        path_entries = env["PATH"].split(":")
        assert str(anima_dir) == path_entries[0], (
            "anima_dir must be the first entry in PATH"
        )

    def test_system_path_preserved(self, tmp_path: Path) -> None:
        """System PATH entries should be preserved after anima_dir."""
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)

        original_path = "/usr/local/bin:/usr/bin:/bin"
        with patch.dict(os.environ, {"PATH": original_path}):
            executor = self._make_executor(anima_dir)
            env = executor._build_env()

        assert env["PATH"] == f"{anima_dir}:{original_path}"

    def test_project_dir_set(self, tmp_path: Path) -> None:
        """ANIMAWORKS_PROJECT_DIR should be set to the project root."""
        from core.paths import PROJECT_DIR

        anima_dir = tmp_path / "animas" / "carol"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_env()

        assert "ANIMAWORKS_PROJECT_DIR" in env
        assert env["ANIMAWORKS_PROJECT_DIR"] == str(PROJECT_DIR)

    def test_anima_dir_env_set(self, tmp_path: Path) -> None:
        """ANIMAWORKS_ANIMA_DIR should still be set."""
        anima_dir = tmp_path / "animas" / "dave"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_env()

        assert env["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)

    def test_fallback_path_when_no_env(self, tmp_path: Path) -> None:
        """When PATH is not in os.environ, fall back to /usr/bin:/bin."""
        anima_dir = tmp_path / "animas" / "eve"
        anima_dir.mkdir(parents=True)

        env_without_path = {k: v for k, v in os.environ.items() if k != "PATH"}
        with patch.dict(os.environ, env_without_path, clear=True):
            executor = self._make_executor(anima_dir)
            env = executor._build_env()

        assert env["PATH"] == f"{anima_dir}:/usr/bin:/bin"


# ── _build_mcp_env() ─────────────────────────────────────


class TestBuildMcpEnv:
    """Verify _build_mcp_env() provides correct env for MCP server subprocess."""

    def _make_executor(self, anima_dir: Path) -> AgentSDKExecutor:
        mc = ModelConfig(model="claude-sonnet-4-20250514")
        return AgentSDKExecutor(model_config=mc, anima_dir=anima_dir)

    def test_anima_dir_set(self, tmp_path: Path) -> None:
        """ANIMAWORKS_ANIMA_DIR should point to the anima directory."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_mcp_env()

        assert env["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)

    def test_project_dir_set(self, tmp_path: Path) -> None:
        """ANIMAWORKS_PROJECT_DIR should be set to the project root."""
        from core.paths import PROJECT_DIR

        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_mcp_env()

        assert env["ANIMAWORKS_PROJECT_DIR"] == str(PROJECT_DIR)

    def test_pythonpath_set(self, tmp_path: Path) -> None:
        """PYTHONPATH should be set to PROJECT_DIR so core modules are importable."""
        from core.paths import PROJECT_DIR

        anima_dir = tmp_path / "animas" / "carol"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_mcp_env()

        assert env["PYTHONPATH"] == str(PROJECT_DIR)

    def test_path_includes_system_path(self, tmp_path: Path) -> None:
        """PATH should include the system PATH."""
        anima_dir = tmp_path / "animas" / "dave"
        anima_dir.mkdir(parents=True)

        original_path = "/usr/local/bin:/usr/bin:/bin"
        with patch.dict(os.environ, {"PATH": original_path}):
            executor = self._make_executor(anima_dir)
            env = executor._build_mcp_env()

        assert env["PATH"] == original_path

    def test_path_fallback_when_no_env(self, tmp_path: Path) -> None:
        """When PATH is not in os.environ, fall back to /usr/bin:/bin."""
        anima_dir = tmp_path / "animas" / "eve"
        anima_dir.mkdir(parents=True)

        env_without_path = {k: v for k, v in os.environ.items() if k != "PATH"}
        with patch.dict(os.environ, env_without_path, clear=True):
            executor = self._make_executor(anima_dir)
            env = executor._build_mcp_env()

        assert env["PATH"] == "/usr/bin:/bin"
