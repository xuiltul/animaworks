"""Unit tests for distilled knowledge injection in builder.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import BuildResult, build_system_prompt


# ── Helpers ──────────────────────────────────────────────


def _make_mock_memory(anima_dir: Path, data_dir: Path) -> MagicMock:
    """Create a mock MemoryManager with standard stubs."""
    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_company_vision.return_value = ""
    memory.read_identity.return_value = "I am Test Anima"
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_current_state.return_value = ""
    memory.read_pending.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_skill_summaries.return_value = []
    memory.list_common_skill_summaries.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.list_procedure_metas.return_value = []
    memory.common_skills_dir = data_dir / "common_skills"
    memory.list_shared_users.return_value = []
    memory.collect_distilled_knowledge.return_value = []
    return memory


# ── Distilled entries injection ──────────────────────────


class TestDistilledEntriesInjected:
    def test_distilled_entries_injected(self, tmp_path: Path, data_dir: Path) -> None:
        """Pre-computed entries appear in system prompt as 'Distilled Knowledge'."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = _make_mock_memory(anima_dir, data_dir)

        distilled_entries = [
            {
                "name": "python-basics",
                "content": "Python is a dynamically typed language.",
                "confidence": 0.9,
                "path": "/tmp/knowledge/python-basics.md",
            },
            {
                "name": "deploy-procedure",
                "content": "Deploy using docker compose up.",
                "confidence": 0.7,
                "path": "/tmp/procedures/deploy-procedure.md",
            },
        ]

        result = build_system_prompt(
            memory,
            distilled_entries=distilled_entries,
            overflow_files=[],
        )
        assert isinstance(result, BuildResult)
        assert "## Distilled Knowledge" in result
        assert "python-basics" in result
        assert "Python is a dynamically typed language" in result
        assert "deploy-procedure" in result
        assert "Deploy using docker compose up" in result


class TestDistilledEntriesEmpty:
    def test_distilled_entries_empty(self, tmp_path: Path, data_dir: Path) -> None:
        """Empty list -> no 'Distilled Knowledge' section."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = _make_mock_memory(anima_dir, data_dir)

        result = build_system_prompt(
            memory,
            distilled_entries=[],
            overflow_files=[],
        )
        assert "## Distilled Knowledge" not in result


class TestOverflowFilesInBuildResult:
    def test_overflow_files_in_build_result(self, tmp_path: Path, data_dir: Path) -> None:
        """overflow_files is correctly returned in BuildResult."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = _make_mock_memory(anima_dir, data_dir)

        overflow = ["overflow_file1", "overflow_file2"]
        result = build_system_prompt(
            memory,
            distilled_entries=[],
            overflow_files=overflow,
        )
        assert isinstance(result, BuildResult)
        assert result.overflow_files == overflow


class TestAutoComputeWhenNoEntries:
    def test_auto_compute_when_no_entries(self, tmp_path: Path, data_dir: Path) -> None:
        """When distilled_entries=None, builder computes from memory."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = _make_mock_memory(anima_dir, data_dir)
        memory.collect_distilled_knowledge.return_value = [
            {
                "name": "auto-computed",
                "content": "Auto computed knowledge",
                "confidence": 0.8,
                "path": "/tmp/knowledge/auto-computed.md",
            },
        ]

        with patch(
            "core.prompt.builder.MemoryManager.compute_injection_plan",
            return_value=(
                [
                    {
                        "name": "auto-computed",
                        "content": "Auto computed knowledge",
                        "confidence": 0.8,
                    },
                ],
                [],
            ),
        ):
            result = build_system_prompt(
                memory,
                distilled_entries=None,
            )
            assert isinstance(result, BuildResult)
            # collect_distilled_knowledge should have been called
            memory.collect_distilled_knowledge.assert_called_once()
            assert "auto-computed" in result
            assert "Auto computed knowledge" in result


class TestConfidenceSortingInOutput:
    def test_confidence_sorting_in_output(self, tmp_path: Path, data_dir: Path) -> None:
        """Higher confidence entries appear first in the output."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("I am Alice", encoding="utf-8")

        memory = _make_mock_memory(anima_dir, data_dir)

        # Entries are already sorted by confidence desc from compute_injection_plan
        distilled_entries = [
            {
                "name": "high-conf",
                "content": "HIGH_CONFIDENCE_CONTENT",
                "confidence": 0.9,
                "path": "/tmp/knowledge/high-conf.md",
            },
            {
                "name": "low-conf",
                "content": "LOW_CONFIDENCE_CONTENT",
                "confidence": 0.3,
                "path": "/tmp/knowledge/low-conf.md",
            },
        ]

        result = build_system_prompt(
            memory,
            distilled_entries=distilled_entries,
            overflow_files=[],
        )
        # Both should appear
        assert "HIGH_CONFIDENCE_CONTENT" in result
        assert "LOW_CONFIDENCE_CONTENT" in result
        # High confidence should appear before low confidence
        high_pos = result.system_prompt.index("HIGH_CONFIDENCE_CONTENT")
        low_pos = result.system_prompt.index("LOW_CONFIDENCE_CONTENT")
        assert high_pos < low_pos
