"""Unit tests for Mode C/S venv package preflight and fallback credential guard."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import ExecutorUnavailableError
from core.schemas import ModelConfig

# ── Helpers ───────────────────────────────────────────────────


def _write_status(animas_dir: Path, name: str, **fields) -> Path:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    path = anima_dir / "status.json"
    path.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")
    return anima_dir


def _make_mode_c_agent(anima_dir: Path, *, api_key: str | None = "test-key"):
    """AgentCore with Mode C config; _create_executor left real for fallback tests."""
    mc = ModelConfig(
        model="codex/gpt-5.3-codex",
        resolved_mode="C",
        api_key=api_key,
        api_key_env="OPENAI_API_KEY",
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


# ── (a) Preflight inspection ─────────────────────────────────


class TestExecutionSdkPreflight:
    def test_critical_when_mode_c_present_and_codex_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from cli.commands.server import _run_execution_sdk_preflight

        animas = tmp_path / "animas"
        _write_status(animas, "ayame", model="codex/gpt-5.6-sol", execution_mode="C")
        _write_status(animas, "nagi", model="openai/gpt-4o")

        with (
            patch("cli.commands.server._package_importable", return_value=False),
            caplog.at_level(logging.CRITICAL, logger="animaworks"),
        ):
            _run_execution_sdk_preflight(animas)

        assert any(
            "Mode C anima" in r.message and "openai-codex" in r.message and r.levelno >= logging.CRITICAL
            for r in caplog.records
        )
        assert "ayame" in caplog.text

    def test_no_critical_when_no_mode_c(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from cli.commands.server import _run_execution_sdk_preflight

        animas = tmp_path / "animas"
        _write_status(animas, "nagi", model="openai/gpt-4o")
        _write_status(animas, "mio", model="grok/grok-4.5", execution_mode="X")

        with (
            patch("cli.commands.server._package_importable", return_value=False),
            caplog.at_level(logging.CRITICAL, logger="animaworks"),
        ):
            _run_execution_sdk_preflight(animas)

        assert "Mode C anima" not in caplog.text
        assert "Mode S anima" not in caplog.text

    def test_critical_when_mode_s_present_and_claude_sdk_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from cli.commands.server import _run_execution_sdk_preflight

        animas = tmp_path / "animas"
        _write_status(animas, "kotoha", model="claude-sonnet-5", execution_mode="S")

        def _importable(name: str) -> bool:
            return False

        with (
            patch("cli.commands.server._package_importable", side_effect=_importable),
            caplog.at_level(logging.CRITICAL, logger="animaworks"),
        ):
            _run_execution_sdk_preflight(animas)

        assert any(
            "Mode S anima" in r.message and "claude_agent_sdk" in r.message and r.levelno >= logging.CRITICAL
            for r in caplog.records
        )

    def test_mode_c_by_model_prefix_without_execution_mode(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from cli.commands.server import _run_execution_sdk_preflight

        animas = tmp_path / "animas"
        _write_status(animas, "codex-bot", model="codex/o4-mini")

        with (
            patch("cli.commands.server._package_importable", return_value=False),
            caplog.at_level(logging.CRITICAL, logger="animaworks"),
        ):
            _run_execution_sdk_preflight(animas)

        assert "Mode C anima" in caplog.text
        assert "codex-bot" in caplog.text


# ── (b) Fallback credential guard ────────────────────────────


class TestModeCFallbackCredentialGuard:
    def test_raises_when_openai_fallback_has_no_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_mode_c_agent(tmp_path, api_key=None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with (
            patch("core.execution.codex_sdk.is_codex_sdk_available", return_value=False),
            pytest.raises(ExecutorUnavailableError, match="fallback_models"),
        ):
            agent._create_executor()

    def test_returns_litellm_when_api_key_present(self, tmp_path: Path) -> None:
        agent = _make_mode_c_agent(tmp_path, api_key="sk-test")
        agent.model_config.fallback_model = "a:openai/gpt-4.1"
        from core.config.schemas import AnimaWorksConfig, CredentialConfig

        config = AnimaWorksConfig(credentials={"openai": CredentialConfig(api_key="configured-key")})
        sentinel = MagicMock(name="litellm_executor")

        with (
            patch("core.config.io.load_config", return_value=config),
            patch("core.execution.codex_sdk.is_codex_sdk_available", return_value=False),
            patch("core.execution.LiteLLMExecutor", return_value=sentinel) as mock_litellm,
        ):
            created = agent._create_executor()

        assert created is sentinel
        assert mock_litellm.call_args.kwargs["model_config"].model == "openai/gpt-4.1"
        assert mock_litellm.call_args.kwargs["model_config"].api_key == "configured-key"

    def test_openai_env_does_not_authorize_implicit_model_switch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_mode_c_agent(tmp_path, api_key=None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        with (
            patch("core.execution.codex_sdk.is_codex_sdk_available", return_value=False),
            patch("core.execution.LiteLLMExecutor") as litellm,
            pytest.raises(ExecutorUnavailableError),
        ):
            agent._create_executor()
        litellm.assert_not_called()

    def test_executor_unavailable_is_non_retryable(self) -> None:
        assert ExecutorUnavailableError.retryable is False
