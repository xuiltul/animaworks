# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for unified skill/procedure table in build_system_prompt.

After the builder refactor, skill matching and full-text injection were removed.
ALL skills (personal + common) and procedures now appear in a single table with
columns: | 名前 | 種別 | 概要 |.  No skill body text is ever injected.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import build_system_prompt
from core.schemas import SkillMeta


# ── Helpers ──────────────────────────────────────────────


def _make_skill_file(path: Path, *, name: str, description: str, body: str) -> None:
    """Write a Claude Code format skill file with YAML frontmatter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\n"
        f"name: {name}\n"
        f"description: >-\n"
        f"  {description}\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def _make_mock_memory(anima_dir: Path, tmp_path: Path) -> MagicMock:
    """Create a MagicMock MemoryManager with standard empty returns."""
    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.common_skills_dir = tmp_path / "common_skills"
    memory.read_company_vision.return_value = ""
    memory.read_identity.return_value = "I am Alice"
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_current_state.return_value = ""
    memory.read_pending.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_shared_users.return_value = []
    memory.load_recent_heartbeat_summary.return_value = ""
    memory.list_procedure_metas.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    return memory


_SKILLS_GUIDE_TEXT = (
    "## スキルと手順書\n\n"
    "スキルと手順書はあなたが持つ能力・作業手順です。\n"
    "使用する際は該当ファイルをReadで読んでから実行してください。"
)


def _fake_load_prompt(name: str, **kwargs) -> str:
    """Minimal load_prompt mock for builder tests.

    Returns the static skills_guide text for 'skills_guide' calls (no kwargs)
    and empty string for all other templates.
    """
    if name == "skills_guide":
        return _SKILLS_GUIDE_TEXT
    return ""


# Common patches applied to every test to isolate build_system_prompt from
# side-effectful functions that read real filesystem / config.
_BUILDER_PATCHES = [
    "core.prompt.builder.load_prompt",
    "core.prompt.builder._build_org_context",
    "core.prompt.builder._discover_other_animas",
    "core.prompt.builder._build_messaging_section",
]


# ── Test Cases ───────────────────────────────────────────


class TestUnifiedSkillTableE2E:
    """build_system_prompt places all skills and procedures in a unified table."""

    def test_all_skills_appear_in_unified_table(self, tmp_path: Path):
        """Personal and common skills both appear as table rows with correct types."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Personal skill
        personal_path = tmp_path / "skills" / "cron-management.md"
        _make_skill_file(
            personal_path,
            name="cron-management",
            description="cron.mdの読み書き・更新スキル",
            body="## cron.mdの構造\n\nschedule: フィールドにcron式を記載する。",
        )
        personal_meta = SkillMeta(
            name="cron-management",
            description="cron.mdの読み書き・更新スキル",
            path=personal_path,
            is_common=False,
        )

        # Common skill
        common_path = tmp_path / "common_skills" / "animaworks-guide.md"
        _make_skill_file(
            common_path,
            name="animaworks-guide",
            description="AnimaWorksフレームワークの仕組みガイド",
            body="## AnimaWorks とは\n\nAIエージェントの自律フレームワークです。",
        )
        common_meta = SkillMeta(
            name="animaworks-guide",
            description="AnimaWorksフレームワークの仕組みガイド",
            path=common_path,
            is_common=True,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [personal_meta]
        memory.list_common_skill_metas.return_value = [common_meta]

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="cron設定をして")

        prompt = result.system_prompt

        # Table header present
        assert "| 名前 | 種別 | 概要 |" in prompt

        # Personal skill row
        assert "| cron-management | 個人 | cron.mdの読み書き・更新スキル |" in prompt

        # Common skill row
        assert "| animaworks-guide | 共通 | AnimaWorksフレームワークの仕組みガイド |" in prompt

    def test_procedures_appear_in_unified_table(self, tmp_path: Path):
        """Procedure metas appear in the table with type '手順'."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        procedure_meta = SkillMeta(
            name="deploy-procedure",
            description="本番デプロイの手順書",
            path=tmp_path / "procedures" / "deploy-procedure.md",
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_procedure_metas.return_value = [procedure_meta]

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="デプロイ手順を教えて")

        prompt = result.system_prompt

        # Table header present
        assert "| 名前 | 種別 | 概要 |" in prompt

        # Procedure row with type 手順
        assert "| deploy-procedure | 手順 | 本番デプロイの手順書 |" in prompt

    def test_no_full_text_injection_regardless_of_message(self, tmp_path: Path):
        """Even when description keywords match the message, only the table row
        appears -- no full skill body text is injected into the prompt."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Skill whose description closely matches the message
        skill_path = tmp_path / "skills" / "cron-management.md"
        _make_skill_file(
            skill_path,
            name="cron-management",
            description=(
                "cron.mdの読み書き・更新を正しいフォーマットで行うためのスキル。"
                "「cron設定」「cronタスク」「定時タスク」等の場面で使用。"
            ),
            body="## cron.mdの構造\n\nschedule: フィールドにcron式を記載する。",
        )
        skill_meta = SkillMeta(
            name="cron-management",
            description=(
                "cron.mdの読み書き・更新を正しいフォーマットで行うためのスキル。"
                "「cron設定」「cronタスク」「定時タスク」等の場面で使用。"
            ),
            path=skill_path,
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [skill_meta]

        # Send a message that directly matches the skill keywords
        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="cron設定をして")

        prompt = result.system_prompt

        # Skill name appears in table row
        assert "cron-management" in prompt

        # Skill body text must NOT appear anywhere in the prompt
        assert "cron.mdの構造" not in prompt
        assert "schedule: フィールドにcron式を記載する。" not in prompt

        # No full-text injection header pattern
        assert "スキル: cron-management" not in prompt

    def test_empty_skills_no_table(self, tmp_path: Path):
        """When there are no skills and no procedures, the table section is
        omitted entirely from the prompt."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        memory = _make_mock_memory(anima_dir, tmp_path)
        # All meta lists are already empty by default in _make_mock_memory

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="おはよう")

        prompt = result.system_prompt

        # No table header
        assert "| 名前 | 種別 | 概要 |" not in prompt

        # No skills_guide static text either (only appended when rows exist)
        assert _SKILLS_GUIDE_TEXT not in prompt

    def test_injected_procedures_always_empty(self, tmp_path: Path):
        """BuildResult.injected_procedures is always an empty list."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Create skills and procedures that would have been injected before
        skill_meta = SkillMeta(
            name="cron-management",
            description="cron.mdの読み書き・更新スキル。「cron設定」で使用。",
            path=tmp_path / "skills" / "cron-management.md",
            is_common=False,
        )
        procedure_meta = SkillMeta(
            name="deploy-procedure",
            description="本番デプロイの手順書",
            path=tmp_path / "procedures" / "deploy-procedure.md",
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_procedure_metas.return_value = [procedure_meta]

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="cron設定をして")

        assert result.injected_procedures == []

    def test_mixed_personal_common_procedures(self, tmp_path: Path):
        """Personal skills, common skills, and procedures all appear in one
        unified table with correct type labels."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Personal skill
        personal_meta = SkillMeta(
            name="tool-creator",
            description="新しいPythonツールモジュールを作成するためのメタスキル",
            path=tmp_path / "skills" / "tool-creator.md",
            is_common=False,
        )

        # Common skill
        common_meta = SkillMeta(
            name="animaworks-guide",
            description="AnimaWorksフレームワークの仕組みガイド",
            path=tmp_path / "common_skills" / "animaworks-guide.md",
            is_common=True,
        )

        # Procedure
        procedure_meta = SkillMeta(
            name="incident-response",
            description="障害発生時の対応手順書",
            path=tmp_path / "procedures" / "incident-response.md",
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [personal_meta]
        memory.list_common_skill_metas.return_value = [common_meta]
        memory.list_procedure_metas.return_value = [procedure_meta]

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="ツール作成したい")

        prompt = result.system_prompt

        # Table header
        assert "| 名前 | 種別 | 概要 |" in prompt

        # All three items with correct types
        assert "| tool-creator | 個人 | 新しいPythonツールモジュールを作成するためのメタスキル |" in prompt
        assert "| animaworks-guide | 共通 | AnimaWorksフレームワークの仕組みガイド |" in prompt
        assert "| incident-response | 手順 | 障害発生時の対応手順書 |" in prompt

        # Skills guide header text is present
        assert "スキルと手順書" in prompt
