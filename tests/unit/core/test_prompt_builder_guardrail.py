# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the commander hiring guardrail in core/prompt/builder.py.

Verifies that the "雇用ルール" section is appended to the system prompt
only when a "newstaff" skill is present in the anima's skill summaries.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import build_system_prompt
from core.schemas import SkillMeta


def _make_mock_memory(
    tmp_path: Path,
    skill_summaries: list[tuple[str, str]],
) -> MagicMock:
    """Create a minimal MemoryManager mock for build_system_prompt.

    Sets up all required method return values with reasonable defaults,
    injecting the given *skill_summaries* for ``list_skill_summaries()``.
    """
    anima_dir = tmp_path / "animas" / "testanima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text("# Test Anima", encoding="utf-8")

    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = ""
    memory.read_identity.return_value = ""
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_current_state.return_value = ""
    memory.read_pending.return_value = ""
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_skill_summaries.return_value = skill_summaries
    memory.list_common_skill_summaries.return_value = []
    memory.list_skill_metas.return_value = [
        SkillMeta(
            name=name,
            description=desc,
            path=Path(f"/tmp/test/skills/{name}.md"),
            is_common=False,
        )
        for name, desc in skill_summaries
    ]
    memory.list_common_skill_metas.return_value = []
    memory.common_skills_dir = tmp_path / "common_skills"
    memory.common_skills_dir.mkdir(parents=True, exist_ok=True)
    memory.list_shared_users.return_value = []

    return memory


class TestCommanderHiringGuardrail:
    """Test the hiring guardrail that injects 雇用ルール when newstaff skill exists."""

    def test_guardrail_present_when_newstaff_skill_exists(
        self, tmp_path: Path, data_dir: Path,
    ) -> None:
        """When skill_summaries contains 'newstaff', the prompt must include
        the 雇用ルール section with create_anima tool instruction."""
        memory = _make_mock_memory(
            tmp_path,
            skill_summaries=[("newstaff", "新しい社員雇用")],
        )

        with patch("core.prompt.builder.load_prompt", return_value="prompt section"):
            result = build_system_prompt(memory)

        assert "雇用ルール" in result
        assert "create-anima" in result

    def test_guardrail_absent_when_no_newstaff_skill(
        self, tmp_path: Path, data_dir: Path,
    ) -> None:
        """When skill_summaries has skills but no 'newstaff', the prompt
        must NOT include the 雇用ルール section."""
        memory = _make_mock_memory(
            tmp_path,
            skill_summaries=[("other_skill", "something")],
        )

        with patch("core.prompt.builder.load_prompt", return_value="prompt section"):
            result = build_system_prompt(memory)

        assert "雇用ルール" not in result

    def test_guardrail_absent_when_skill_summaries_empty(
        self, tmp_path: Path, data_dir: Path,
    ) -> None:
        """When skill_summaries is an empty list, the prompt must NOT
        include the 雇用ルール section."""
        memory = _make_mock_memory(
            tmp_path,
            skill_summaries=[],
        )

        with patch("core.prompt.builder.load_prompt", return_value="prompt section"):
            result = build_system_prompt(memory)

        assert "雇用ルール" not in result

    def test_guardrail_content_completeness(
        self, tmp_path: Path, data_dir: Path,
    ) -> None:
        """Verify the full guardrail text includes all key directives."""
        memory = _make_mock_memory(
            tmp_path,
            skill_summaries=[("newstaff", "新しい社員雇用")],
        )

        with patch("core.prompt.builder.load_prompt", return_value="prompt section"):
            result = build_system_prompt(memory)

        # All key phrases from the guardrail block
        assert "create-anima" in result
        assert "identity.md" in result
        assert "キャラクターシート" in result

    def test_guardrail_with_newstaff_among_multiple_skills(
        self, tmp_path: Path, data_dir: Path,
    ) -> None:
        """When newstaff is one of several skills, the guardrail must still
        be injected."""
        memory = _make_mock_memory(
            tmp_path,
            skill_summaries=[
                ("coding", "コーディング支援"),
                ("newstaff", "新しい社員雇用"),
                ("reporting", "レポート作成"),
            ],
        )

        with patch("core.prompt.builder.load_prompt", return_value="prompt section"):
            result = build_system_prompt(memory)

        assert "雇用ルール" in result
        assert "create-anima" in result
