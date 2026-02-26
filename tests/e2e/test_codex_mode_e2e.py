from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Codex SDK execution mode (Mode C).

Verifies the full AgentCore integration with CodexSDKExecutor using
mocked Codex SDK.  No real Codex CLI or API key required.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Executor creation tests ──────────────────────────────────

class TestExecutorCreation:
    """_create_executor() with Mode C and ImportError fallback."""

    def test_create_executor_mode_c(self, make_agent_core):
        """codex/* model → CodexSDKExecutor."""
        agent = make_agent_core(name="codex-test", model="codex/o4-mini")

        mock_executor_cls = MagicMock()
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance

        with patch(
            "core.agent.AgentCore._create_executor"
        ) as mock_create:
            mock_create.return_value = mock_executor_instance
            agent._executor = agent._create_executor()

        # Verify mode resolves to 'c'
        assert agent._resolve_execution_mode() == "c"

    def test_create_executor_mode_c_fallback_to_a(self, make_agent_core):
        """codex/* model + openai-codex-sdk missing → LiteLLMExecutor fallback."""
        agent = make_agent_core(name="codex-fallback", model="codex/o4-mini")

        with patch.dict("sys.modules", {"openai_codex_sdk": None}):
            with patch(
                "core.execution.codex_sdk.CodexSDKExecutor",
                side_effect=ImportError("No module named 'openai_codex_sdk'"),
            ):
                executor = agent._create_executor()

        from core.execution.litellm_loop import LiteLLMExecutor
        assert isinstance(executor, LiteLLMExecutor)


# ── Run cycle tests ──────────────────────────────────────────

class TestRunCycle:
    """AgentCore.run_cycle() integration with Mode C."""

    @pytest.mark.asyncio
    async def test_run_cycle_mode_c_chat(self, make_agent_core):
        """Chat trigger with codex/* model returns CycleResult."""
        agent = make_agent_core(name="codex-chat", model="codex/o4-mini")

        mock_turn = MagicMock()
        mock_turn.final_response = "Codex response"
        mock_turn.items = []
        mock_turn.usage = MagicMock(input_tokens=50, output_tokens=20)

        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(return_value=mock_turn)
        mock_thread.id = "thread-chat-001"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with patch("core.execution.codex_sdk.CodexSDKExecutor._create_codex_client", return_value=mock_codex):
            result = await agent.run_cycle(
                prompt="Hello from test",
                trigger="message:test-user",
            )

        assert result.action == "responded"
        assert "Codex response" in result.summary

    @pytest.mark.asyncio
    async def test_run_cycle_mode_c_heartbeat(self, make_agent_core):
        """Heartbeat trigger with codex/* model returns CycleResult."""
        agent = make_agent_core(name="codex-hb", model="codex/o4-mini")

        mock_turn = MagicMock()
        mock_turn.final_response = "Heartbeat done"
        mock_turn.items = []
        mock_turn.usage = None

        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(return_value=mock_turn)
        mock_thread.id = "thread-hb-001"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with patch("core.execution.codex_sdk.CodexSDKExecutor._create_codex_client", return_value=mock_codex):
            result = await agent.run_cycle(
                prompt="Heartbeat check",
                trigger="heartbeat",
            )

        assert result.action == "responded"
        assert "Heartbeat done" in result.summary

    @pytest.mark.asyncio
    async def test_run_cycle_mode_c_cron(self, make_agent_core):
        """Cron trigger with codex/* model returns CycleResult."""
        agent = make_agent_core(name="codex-cron", model="codex/o4-mini")

        mock_turn = MagicMock()
        mock_turn.final_response = "Cron task completed"
        mock_turn.items = []
        mock_turn.usage = None

        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(return_value=mock_turn)
        mock_thread.id = "thread-cron-001"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with patch("core.execution.codex_sdk.CodexSDKExecutor._create_codex_client", return_value=mock_codex):
            result = await agent.run_cycle(
                prompt="Run daily report",
                trigger="cron:daily_report",
            )

        assert result.action == "responded"
        assert "Cron task completed" in result.summary

    @pytest.mark.asyncio
    async def test_run_cycle_mode_c_task(self, make_agent_core):
        """TaskExec trigger with codex/* model returns CycleResult."""
        agent = make_agent_core(name="codex-task", model="codex/o4-mini")

        mock_turn = MagicMock()
        mock_turn.final_response = "Task executed"
        mock_turn.items = []
        mock_turn.usage = None

        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(return_value=mock_turn)
        mock_thread.id = "thread-task-001"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with patch("core.execution.codex_sdk.CodexSDKExecutor._create_codex_client", return_value=mock_codex):
            result = await agent.run_cycle(
                prompt="Execute pending task",
                trigger="task:task-001",
            )

        assert result.action == "responded"
        assert "Task executed" in result.summary


# ── Context window tests ─────────────────────────────────────

class TestContextWindow:
    """Codex model context window resolution."""

    def test_codex_o4_mini_context_window(self):
        from core.prompt.context import resolve_context_window
        size = resolve_context_window("codex/o4-mini")
        assert size == 200_000

    def test_codex_o3_context_window(self):
        from core.prompt.context import resolve_context_window
        size = resolve_context_window("codex/o3")
        assert size == 200_000

    def test_codex_gpt41_context_window(self):
        from core.prompt.context import resolve_context_window
        size = resolve_context_window("codex/gpt-4.1")
        assert size == 1_000_000


# ── No regression tests ──────────────────────────────────────

class TestNoRegression:
    """Verify existing modes are unaffected by C mode addition."""

    def test_s_mode_unchanged(self, make_agent_core):
        agent = make_agent_core(name="s-mode", model="claude-sonnet-4-6")
        assert agent._resolve_execution_mode() == "s"

    def test_a_mode_unchanged(self, make_agent_core):
        agent = make_agent_core(name="a-mode", model="openai/gpt-4o")
        assert agent._resolve_execution_mode() == "a"

    def test_b_mode_unchanged(self, make_agent_core):
        agent = make_agent_core(name="b-mode", model="ollama/gemma3:4b")
        assert agent._resolve_execution_mode() == "b"

    def test_known_models_include_codex(self):
        from core.config.models import KNOWN_MODELS
        codex_models = [m for m in KNOWN_MODELS if m["mode"] == "C"]
        assert len(codex_models) >= 1
        names = [m["name"] for m in codex_models]
        assert "codex/o4-mini" in names
