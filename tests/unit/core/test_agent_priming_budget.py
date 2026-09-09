"""Unit tests for dynamic tool overhead and budget-based fit in PrimingMixin."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(anima_dir: Path, model: str = "claude-sonnet-4-20250514"):
    """Create AgentCore with all external dependencies mocked."""
    from core.schemas import ModelConfig

    mc = ModelConfig(model=model, api_key="test-key")
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

    def test_mode_s_uses_higher_per_schema(self, tmp_path: Path):
        """Mode S/C uses 200 tokens/schema; Mode A uses 150."""
        agent = _make_agent(tmp_path)
        agent._tool_registry = ["t"] * 50  # 50 tools
        overhead_s = agent._estimate_tool_overhead(mode="s")
        overhead_a = agent._estimate_tool_overhead(mode="a")
        assert overhead_s > overhead_a
        assert overhead_s == min(50 * 200, 20000)
        assert overhead_a == min(50 * 150, 20000)


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

    def test_all_shrink_passes_preserve_protected_context(self, tmp_path: Path):
        """Even the last shrink must pass typed recall and human decisions through."""
        from core.exceptions import ExecutionError

        agent = _make_agent(tmp_path)
        priming = (
            '<priming source="resident_knowledge" trust="medium" render_mode="guardrail">'
            "REQUIRES_HUMAN_APPROVAL</priming>\n"
            '<priming source="pending_tasks" trust="trusted">DO_NOT_DUPLICATE_TASK</priming>'
        )
        with patch("core._agent_priming.build_system_prompt") as build:
            build.return_value = MagicMock(system_prompt="REQUIRED_AUTHORITY " * 20_000)
            with pytest.raises(ExecutionError):
                agent._fit_prompt_to_context_window(
                    "large " * 20_000,
                    "user instruction",
                    16_000,
                    priming_section=priming,
                    pending_human_notifications="HUMAN_SAYS_STOP",
                    shortterm_text="AUTHORIZED_SCOPE_ONLY",
                    mode="a",
                    trigger="chat",
                )
        assert build.call_count == 3
        for call in build.call_args_list:
            assert call.kwargs["priming_section"] == priming
            assert call.kwargs["pending_human_notifications"] == "HUMAN_SAYS_STOP"
            assert call.kwargs["shortterm_text"] == "AUTHORIZED_SCOPE_ONLY"
        agent._executor.execute.assert_not_called()

    def test_real_builder_overflow_fails_instead_of_truncating_authority(self, data_dir, make_anima):
        """Exercise the final AgentCore fit, not just the section allocator."""
        from core.exceptions import ExecutionError
        from core.memory.manager import MemoryManager
        from core.prompt.builder import build_system_prompt

        anima_dir = make_anima("fit-safety")
        identity = "MANDATORY_IDENTITY_BEGIN\n" + "Always preserve human authority. " * 12_000
        identity += "\nMANDATORY_IDENTITY_END"
        (anima_dir / "identity.md").write_text(identity, encoding="utf-8")
        agent = _make_agent(anima_dir)
        agent.memory = MemoryManager(anima_dir)
        priming = (
            '<priming source="resident_knowledge" trust="medium" render_mode="guardrail">'
            "REQUIRES_HUMAN_APPROVAL</priming>\n"
            '<priming source="pending_tasks" trust="trusted">DO_NOT_DUPLICATE_TASK</priming>'
        )
        built = build_system_prompt(
            agent.memory,
            priming_section=priming,
            pending_human_notifications="HUMAN_SAYS_STOP",
            execution_mode="a",
            trigger="chat",
            context_window=16_000,
        ).system_prompt
        assert "MANDATORY_IDENTITY_END" in built
        assert "REQUIRES_HUMAN_APPROVAL" in built
        assert "DO_NOT_DUPLICATE_TASK" in built
        assert "HUMAN_SAYS_STOP" in built
        with (
            patch.object(agent, "_get_retriever", return_value=None),
            pytest.raises(ExecutionError),
        ):
            agent._fit_prompt_to_context_window(
                built,
                "user instruction",
                16_000,
                priming_section=priming,
                pending_human_notifications="HUMAN_SAYS_STOP",
                mode="a",
                trigger="chat",
            )
        assert (anima_dir / "identity.md").read_text(encoding="utf-8") == identity
        agent._executor.execute.assert_not_called()
