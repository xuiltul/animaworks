"""Unit tests for dynamic tool overhead and budget-based fit in PrimingMixin."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_agent(anima_dir: Path, model: str = "claude-sonnet-4-20250514"):
    """Create AgentCore with all external dependencies mocked."""
    from core.schemas import ModelConfig

    mc = ModelConfig(model=model, api_key="test-key")
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.anima_dir = anima_dir
    memory.collect_distilled_knowledge.return_value = []

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


class TestEstimateToolOverhead:
    """Test dynamic tool overhead estimation."""

    def test_default_returns_minimum(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        overhead = agent._estimate_tool_overhead()
        assert overhead >= 5000

    def test_many_tools_increases_overhead(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        agent._tool_registry = ["tool1", "tool2", "tool3"] * 30  # 90 tools
        overhead = agent._estimate_tool_overhead()
        assert overhead > 5000

    def test_overhead_capped_at_max(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        agent._tool_registry = ["t"] * 500
        overhead = agent._estimate_tool_overhead()
        assert overhead <= 20000


class TestFitPromptBudgetShrink:
    """Test _fit_prompt_to_context_window uses budget shrinking."""

    def test_prompt_that_fits_is_unchanged(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        original = "Short prompt"
        result = agent._fit_prompt_to_context_window(
            original,
            "user msg",
            200_000,
            priming_section="",
            mode="a",
            trigger="chat",
        )
        assert result == original

    def test_oversized_prompt_gets_shrunk(self, tmp_path: Path):
        """When prompt exceeds budget, it should be rebuilt with smaller budget."""
        agent = _make_agent(tmp_path)
        large_prompt = "x" * 100_000
        with (
            patch.object(agent, "memory") as mock_memory,
            patch("core._agent_priming.build_system_prompt") as mock_build,
        ):
            mock_build.return_value = MagicMock(system_prompt="small result")
            mock_memory.anima_dir = tmp_path
            result = agent._fit_prompt_to_context_window(
                large_prompt,
                "user msg",
                32_000,
                priming_section="",
                mode="a",
                trigger="chat",
            )
        assert mock_build.called or len(result) < len(large_prompt)
