"""Unit tests for distilled knowledge collection and injection planning."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.manager import MemoryManager


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def memory_manager(tmp_path: Path, data_dir: Path) -> MemoryManager:
    """Create a MemoryManager with isolated temp directories."""
    anima_dir = data_dir / "animas" / "test"
    anima_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
        (anima_dir / sub).mkdir(exist_ok=True)

    return MemoryManager(anima_dir)


# ── collect_distilled_knowledge ──────────────────────────


class TestCollectDistilledKnowledgeEmpty:
    def test_collect_distilled_knowledge_empty(
        self, memory_manager: MemoryManager,
    ) -> None:
        """Both knowledge/ and procedures/ are empty -> returns []."""
        result = memory_manager.collect_distilled_knowledge()
        assert result == []


class TestCollectDistilledKnowledgeWithFiles:
    def test_collect_distilled_knowledge_with_files(
        self, memory_manager: MemoryManager,
    ) -> None:
        """Has knowledge and procedure files -> returns all with correct metadata."""
        (memory_manager.knowledge_dir / "topic1.md").write_text(
            "Knowledge about topic 1", encoding="utf-8",
        )
        (memory_manager.procedures_dir / "deploy.md").write_text(
            "Deployment procedure", encoding="utf-8",
        )

        result = memory_manager.collect_distilled_knowledge()
        assert len(result) == 2

        names = [e["name"] for e in result]
        assert "topic1" in names
        assert "deploy" in names

        for entry in result:
            assert "path" in entry
            assert "name" in entry
            assert "content" in entry
            assert "confidence" in entry


class TestCollectDistilledKnowledgeWithFrontmatter:
    def test_collect_distilled_knowledge_with_frontmatter(
        self, memory_manager: MemoryManager,
    ) -> None:
        """Files with YAML frontmatter containing confidence -> extracts correctly."""
        (memory_manager.knowledge_dir / "rated.md").write_text(
            "---\nconfidence: 0.8\n---\nHigh confidence knowledge",
            encoding="utf-8",
        )

        result = memory_manager.collect_distilled_knowledge()
        assert len(result) == 1
        assert result[0]["confidence"] == 0.8


class TestCollectDistilledKnowledgeWithoutFrontmatter:
    def test_collect_distilled_knowledge_without_frontmatter(
        self, memory_manager: MemoryManager,
    ) -> None:
        """Files without frontmatter -> confidence defaults to 0.5."""
        (memory_manager.knowledge_dir / "plain.md").write_text(
            "Plain knowledge without frontmatter", encoding="utf-8",
        )

        result = memory_manager.collect_distilled_knowledge()
        assert len(result) == 1
        assert result[0]["confidence"] == 0.5


# ── _extract_confidence ──────────────────────────────────


class TestExtractConfidenceWithValidYaml:
    def test_extract_confidence_with_valid_yaml(self) -> None:
        """YAML frontmatter with confidence: 0.8 -> returns 0.8."""
        content = "---\nconfidence: 0.8\ntags: [a, b]\n---\nBody text"
        assert MemoryManager._extract_confidence(content) == 0.8


class TestExtractConfidenceWithoutFrontmatter:
    def test_extract_confidence_without_frontmatter(self) -> None:
        """No frontmatter -> returns 0.5."""
        content = "Just plain text without frontmatter"
        assert MemoryManager._extract_confidence(content) == 0.5


class TestExtractConfidenceMalformedYaml:
    def test_extract_confidence_malformed_yaml(self) -> None:
        """Malformed YAML -> returns 0.5."""
        content = "---\n: invalid: yaml: [broken\n---\nBody"
        assert MemoryManager._extract_confidence(content) == 0.5


class TestExtractConfidenceNoConfidenceField:
    def test_extract_confidence_no_confidence_field(self) -> None:
        """YAML without confidence key -> returns 0.5."""
        content = "---\ntags: [tag1, tag2]\nauthor: test\n---\nBody"
        assert MemoryManager._extract_confidence(content) == 0.5


# ── compute_injection_plan ───────────────────────────────


class TestComputeInjectionPlanAllFit:
    def test_compute_injection_plan_all_fit(self) -> None:
        """Total tokens < budget -> all injected, no overflow."""
        distilled = [
            {"name": "a", "content": "short content", "confidence": 0.9},
            {"name": "b", "content": "another short", "confidence": 0.7},
        ]
        # context_window=100000, budget_ratio=0.10 -> budget=10000 tokens
        injected, overflow = MemoryManager.compute_injection_plan(
            distilled, context_window=100000, budget_ratio=0.10,
        )
        assert len(injected) == 2
        assert overflow == []


class TestComputeInjectionPlanPartialOverflow:
    def test_compute_injection_plan_partial_overflow(self) -> None:
        """Some files exceed budget -> high-confidence injected, low overflow."""
        # Create entries where total tokens exceed budget
        # Budget: context_window * budget_ratio = 300 * 0.10 = 30 tokens
        # Each "x" * 90 = 30 tokens (90 / 3 = 30)
        distilled = [
            {"name": "high", "content": "x" * 90, "confidence": 0.9},
            {"name": "low", "content": "x" * 90, "confidence": 0.3},
        ]
        injected, overflow = MemoryManager.compute_injection_plan(
            distilled, context_window=300, budget_ratio=0.10,
        )
        assert len(injected) == 1
        assert injected[0]["name"] == "high"
        assert "low" in overflow


class TestComputeInjectionPlanConfidenceSorting:
    def test_compute_injection_plan_confidence_sorting(self) -> None:
        """Entries are sorted by confidence desc in injected list."""
        distilled = [
            {"name": "low", "content": "a", "confidence": 0.3},
            {"name": "high", "content": "b", "confidence": 0.9},
            {"name": "mid", "content": "c", "confidence": 0.6},
        ]
        injected, overflow = MemoryManager.compute_injection_plan(
            distilled, context_window=100000,
        )
        assert len(injected) == 3
        assert injected[0]["name"] == "high"
        assert injected[1]["name"] == "mid"
        assert injected[2]["name"] == "low"


class TestComputeInjectionPlanEmptyInput:
    def test_compute_injection_plan_empty_input(self) -> None:
        """Empty list -> returns ([], [])."""
        injected, overflow = MemoryManager.compute_injection_plan(
            [], context_window=100000,
        )
        assert injected == []
        assert overflow == []


class TestComputeInjectionPlanSingleLargeFile:
    def test_compute_injection_plan_single_large_file(self) -> None:
        """One file exceeds entire budget -> in overflow."""
        # Budget: 100 * 0.10 = 10 tokens
        # Content: 300 chars = 100 tokens (300 / 3)
        distilled = [
            {"name": "huge", "content": "x" * 300, "confidence": 0.9},
        ]
        injected, overflow = MemoryManager.compute_injection_plan(
            distilled, context_window=100, budget_ratio=0.10,
        )
        assert injected == []
        assert "huge" in overflow
