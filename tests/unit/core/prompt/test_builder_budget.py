"""Unit tests for budget-based prompt scaling in builder.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import (
    _MIN_SYSTEM_BUDGET,
    PromptBudget,
    SectionEntry,
    _allocate_sections,
    _compute_system_budget,
    build_system_prompt,
)

# ── _compute_system_budget ─────────────────────────────────


class TestComputeSystemBudget:
    """Test budget computation from context window."""

    def test_128k_budget(self):
        budget = _compute_system_budget(128_000)
        assert budget == PromptBudget(target=6_000, ceiling=44_800)

    def test_32k_budget(self):
        budget = _compute_system_budget(32_000)
        assert budget == PromptBudget(target=6_000, ceiling=11_200)

    def test_8k_budget(self):
        budget = _compute_system_budget(8_000)
        assert budget == PromptBudget(target=2_800, ceiling=2_800)

    def test_minimum_budget(self):
        budget = _compute_system_budget(100)
        assert budget == PromptBudget(target=_MIN_SYSTEM_BUDGET, ceiling=_MIN_SYSTEM_BUDGET)

    def test_zero_context_window(self):
        budget = _compute_system_budget(0)
        assert budget == PromptBudget(target=_MIN_SYSTEM_BUDGET, ceiling=_MIN_SYSTEM_BUDGET)

    def test_explicit_budget_clamped_to_auto(self):
        """system_budget cannot exceed auto-computed budget."""
        budget = _compute_system_budget(32_000, system_budget=999_999)
        assert budget == PromptBudget(target=6_000, ceiling=11_200)

    def test_explicit_budget_below_auto(self):
        """system_budget below auto is used as-is."""
        budget = _compute_system_budget(128_000, system_budget=10_000)
        assert budget == PromptBudget(target=6_000, ceiling=10_000)

    def test_explicit_budget_below_minimum(self):
        """system_budget cannot go below _MIN_SYSTEM_BUDGET."""
        budget = _compute_system_budget(128_000, system_budget=500)
        assert budget == PromptBudget(target=_MIN_SYSTEM_BUDGET, ceiling=_MIN_SYSTEM_BUDGET)

    def test_200k_budget(self):
        budget = _compute_system_budget(200_000)
        assert budget == PromptBudget(target=6_000, ceiling=70_000)

    def test_configured_target_and_ceiling(self):
        config = MagicMock()
        config.prompt.system_prompt_target_tokens = 9_000
        config.prompt.system_prompt_ceiling_pct = 0.20
        with patch("core.config.load_config", return_value=config):
            budget = _compute_system_budget(100_000)
        assert budget == PromptBudget(target=9_000, ceiling=20_000)


# ── _allocate_sections ──────────────────────────────────────


class TestAllocateSections:
    """Test Rigid/Elastic budget allocation."""

    def test_p1_always_included(self):
        """Priority-1 rigid sections are included even when budget=0."""
        sections = [
            SectionEntry(id="identity", priority=1, kind="rigid", content="I am X"),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=0, ceiling=0))
        assert len(allocated) == 1
        assert allocated[0].id == "identity"

    def test_p2_excluded_when_budget_tight(self):
        """P2 sections excluded when remaining budget is insufficient."""
        sections = [
            SectionEntry(id="identity", priority=1, kind="rigid", content="x" * 100),
            SectionEntry(id="behavior", priority=2, kind="rigid", content="y" * 200),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=10, ceiling=65))
        ids = {s.id for s in allocated}
        assert "identity" in ids
        assert "behavior" not in ids

    def test_p4_excluded_before_p2(self):
        """P4 sections are excluded before P2 when budget is tight."""
        sections = [
            SectionEntry(id="identity", priority=1, kind="rigid", content="x" * 50),
            SectionEntry(id="behavior", priority=2, kind="rigid", content="y" * 100),
            SectionEntry(id="emotion", priority=4, kind="rigid", content="z" * 100),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=10, ceiling=50))
        ids = {s.id for s in allocated}
        assert "identity" in ids
        assert "behavior" in ids
        assert "emotion" not in ids

    def test_elastic_items_trim_by_priority(self):
        """Lower-priority elastic items are removed before important ones."""
        sections = [
            SectionEntry(id="identity", priority=1, kind="rigid", content="identity"),
            SectionEntry(id="priming", priority=2, kind="elastic", content="important" * 20),
            SectionEntry(id="dk", priority=3, kind="elastic", content="optional" * 20),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=70, ceiling=1000))
        ids = {section.id for section in allocated}
        assert "priming" in ids
        assert "dk" not in ids

    def test_elastic_all_included_when_budget_sufficient(self):
        """Elastic sections included fully when budget allows."""
        sections = [
            SectionEntry(id="identity", priority=1, kind="rigid", content="x" * 100),
            SectionEntry(id="priming", priority=2, kind="elastic", content="p" * 200),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=10000, ceiling=10000))
        priming = next(s for s in allocated if s.id == "priming")
        assert len(priming.content) == 200

    def test_order_preserved(self):
        """Original section ordering is preserved in output."""
        sections = [
            SectionEntry(id="a", priority=1, kind="rigid", content="AAA"),
            SectionEntry(id="b", priority=4, kind="rigid", content="BBB"),
            SectionEntry(id="c", priority=2, kind="rigid", content="CCC"),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=10000, ceiling=10000))
        ids = [s.id for s in allocated]
        assert ids == ["a", "b", "c"]

    def test_empty_sections(self):
        """Empty sections list returns empty result."""
        assert _allocate_sections([], budget=PromptBudget(target=10000, ceiling=10000)) == []

    def test_no_rigid_sections_cut(self):
        """Rigid sections with sufficient budget are never truncated."""
        content = "x" * 500
        sections = [
            SectionEntry(id="full", priority=2, kind="rigid", content=content),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=1000, ceiling=1000))
        assert allocated[0].content == content

    def test_elastic_item_is_never_partially_retained(self):
        """A too-large elastic item is dropped rather than sliced."""
        sections = [
            SectionEntry(id="identity", priority=1, kind="rigid", content="x" * 900),
            SectionEntry(id="priming", priority=2, kind="elastic", content="p" * 1000),
        ]
        allocated = _allocate_sections(sections, budget=PromptBudget(target=300, ceiling=1000))
        ids = {s.id for s in allocated}
        assert "priming" not in ids


# ── Integration tests ──────────────────────────────────────


def _make_mock_memory(
    tmp_path: Path,
    data_dir: Path,
    *,
    identity: str = "I am TestAnima",
    injection: str = "行動指針",
) -> MagicMock:
    """Create a mock MemoryManager."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text(identity, encoding="utf-8")
    (anima_dir / "heartbeat.md").write_text("- check", encoding="utf-8")

    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_identity.return_value = identity
    memory.read_injection.return_value = injection
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = ""
    memory.read_current_state.return_value = "status: idle"
    memory.read_pending.return_value = ""
    memory.read_resolutions.return_value = []
    memory.read_model_config.return_value = None
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.list_procedure_metas.return_value = []
    memory.list_shared_users.return_value = []
    return memory


class TestBudgetIntegration:
    """Integration tests: context_window → prompt size within budget."""

    @pytest.mark.parametrize("context_window", [8_000, 32_000, 64_000, 128_000, 200_000])
    def test_prompt_fits_within_budget(self, tmp_path: Path, data_dir: Path, context_window: int):
        """Generated prompt must not exceed system budget."""
        memory = _make_mock_memory(tmp_path, data_dir)
        budget = _compute_system_budget(context_window)
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory,
                execution_mode="a",
                context_window=context_window,
            )
        from core.prompt.tokens import estimate_tokens

        # XML boundary tags add a small amount beyond section-content accounting.
        assert estimate_tokens(result.system_prompt) <= budget.ceiling + 500, (
            f"Prompt exceeds ceiling {budget.ceiling} tokens at cw={context_window}"
        )

    def test_identity_always_present(self, tmp_path: Path, data_dir: Path):
        """Identity is present even at 8K context window."""
        memory = _make_mock_memory(tmp_path, data_dir)
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory,
                execution_mode="a",
                context_window=8_000,
            )
        assert "I am TestAnima" in result

    def test_128k_equals_200k_quality(self, tmp_path: Path, data_dir: Path):
        """128K and 200K should produce similar prompts (both at scale=1.0)."""
        mem1 = _make_mock_memory(tmp_path / "a", data_dir)
        mem2 = _make_mock_memory(tmp_path / "b", data_dir)
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            r128 = build_system_prompt(mem1, execution_mode="a", context_window=128_000)
            r200 = build_system_prompt(mem2, execution_mode="a", context_window=200_000)
        # Should be very similar in content (may differ slightly due to budget)
        # 128K budget = 83200, 200K budget = 130000, but with small mock sections
        # both include everything
        assert abs(len(r128.system_prompt) - len(r200.system_prompt)) < 100

    def test_system_budget_parameter(self, tmp_path: Path, data_dir: Path):
        """Explicit system_budget reduces prompt size."""
        memory = _make_mock_memory(tmp_path, data_dir)
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            full = build_system_prompt(memory, execution_mode="a", context_window=200_000)
            constrained = build_system_prompt(
                memory,
                execution_mode="a",
                context_window=200_000,
                system_budget=3000,
            )
        assert len(constrained.system_prompt) <= len(full.system_prompt)

    def test_no_rigid_section_truncated(self, tmp_path: Path, data_dir: Path):
        """No rigid section should appear partially in the output."""
        memory = _make_mock_memory(
            tmp_path,
            data_dir,
            identity="UNIQUE_IDENTITY_MARKER_FULL_TEXT",
        )
        with patch("core.prompt.builder.load_prompt", return_value="section"):
            result = build_system_prompt(
                memory,
                execution_mode="a",
                context_window=8_000,
            )
        # If identity appears at all, it must be complete
        if "UNIQUE_IDENTITY_MARKER" in result.system_prompt:
            assert "UNIQUE_IDENTITY_MARKER_FULL_TEXT" in result.system_prompt
