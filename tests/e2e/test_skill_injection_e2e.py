# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for skill tool integration after unified table removal.

The skill catalog lives in system prompt Group 4 (``<available_skills>``).
The ``skill`` tool carries a short usage description only; full content is
returned on-demand when the skill tool is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.i18n import t
from core.prompt.builder import build_system_prompt
from core.schemas import SkillMeta
from core.tooling.schemas import build_tool_list
from core.tooling.skill_tool import build_skill_tool_description, load_and_render_skill


# ── Helpers ──────────────────────────────────────────────


def _make_skill_file(path: Path, *, name: str, description: str, body: str) -> None:
    """Write a Claude Code format skill file with YAML frontmatter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: >-\n  {description}\n---\n\n# {name}\n\n{body}\n"
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
    memory.collect_distilled_knowledge_separated.return_value = ([], [])
    return memory


def _fake_load_prompt(name: str, **kwargs) -> str:
    """Minimal load_prompt mock for builder tests."""
    return ""


# ── Test Cases ───────────────────────────────────────────


class TestUnifiedSkillTableE2E:
    """Skills/procedures are not in a markdown table; catalog is in Group 4 XML block."""

    def test_skills_not_in_system_prompt_table(self, tmp_path: Path):
        """Personal and common skills must NOT appear as table rows; catalog lists names in Group 4."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        personal_meta = SkillMeta(
            name="cron-management",
            description="cron.mdの読み書き・更新スキル",
            path=tmp_path / "skills" / "cron-management.md",
            is_common=False,
        )
        common_meta = SkillMeta(
            name="animaworks-guide",
            description="AnimaWorksフレームワークの仕組みガイド",
            path=tmp_path / "common_skills" / "animaworks-guide.md",
            is_common=True,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [personal_meta]
        memory.list_common_skill_metas.return_value = [common_meta]

        with (
            patch("core.memory.skill_metadata.SkillMetadataService") as mock_svc_class,
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            svc_inst = MagicMock()
            svc_inst.list_common_skill_metas.return_value = [common_meta]
            mock_svc_class.return_value = svc_inst

            result = build_system_prompt(memory, message="cron設定をして")

        prompt = result.system_prompt

        # Table header must NOT be present (table removed)
        assert "| 名前 | 種別 | 概要 |" not in prompt
        # Skill names should NOT appear as table rows in the prompt
        assert "| cron-management |" not in prompt
        assert "| animaworks-guide |" not in prompt
        assert "<available_skills>" in prompt
        assert "cron-management" in prompt
        assert "animaworks-guide" in prompt

    def test_skill_tool_present_in_build_tool_list(self, tmp_path: Path):
        """build_tool_list with include_skill_tools=True adds a 'skill' tool."""
        personal_meta = SkillMeta(
            name="cron-management",
            description="cron.mdの読み書き・更新スキル",
            path=tmp_path / "skills" / "cron-management.md",
            is_common=False,
        )
        common_meta = SkillMeta(
            name="animaworks-guide",
            description="AnimaWorksフレームワークの仕組みガイド",
            path=tmp_path / "common_skills" / "animaworks-guide.md",
            is_common=True,
        )

        tools = build_tool_list(
            include_skill_tools=True,
            skill_metas=[personal_meta],
            common_skill_metas=[common_meta],
            procedure_metas=[],
        )
        names = {x["name"] for x in tools}
        assert "skill" in names

        skill_tool = next(x for x in tools if x["name"] == "skill")
        assert "cron-management" not in skill_tool["description"]
        assert "animaworks-guide" not in skill_tool["description"]
        assert t("skill.desc_line1") in skill_tool["description"]

    def test_procedures_not_in_skill_tool_description(self, tmp_path: Path):
        """Procedure metas are not embedded in the skill tool description (catalog is in prompt)."""
        procedure_meta = SkillMeta(
            name="deploy-procedure",
            description="本番デプロイの手順書",
            path=tmp_path / "procedures" / "deploy-procedure.md",
            is_common=False,
        )

        desc = build_skill_tool_description([], [], [procedure_meta])

        assert "deploy-procedure" not in desc
        assert "<available_skills>" not in desc

    def test_no_full_text_in_system_prompt(self, tmp_path: Path):
        """Even when description keywords match the message, no skill body text
        is injected into the system prompt (skills are on-demand via tool)."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        skill_path = tmp_path / "skills" / "cron-management.md"
        _make_skill_file(
            skill_path,
            name="cron-management",
            description="cron.mdの読み書き・更新スキル",
            body="## cron.mdの構造\n\nschedule: フィールドにcron式を記載する。",
        )
        skill_meta = SkillMeta(
            name="cron-management",
            description="cron.mdの読み書き・更新スキル",
            path=skill_path,
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [skill_meta]

        with (
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="cron設定をして")

        prompt = result.system_prompt

        # Skill body text must NOT appear in the prompt
        assert "cron.mdの構造" not in prompt
        assert "schedule: フィールドにcron式を記載する。" not in prompt
        # No full-text injection header
        assert "スキル: cron-management" not in prompt

    def test_empty_skills_no_table(self, tmp_path: Path):
        """When there are no skills and no procedures, no table or guide is injected."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        memory = _make_mock_memory(anima_dir, tmp_path)

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

    def test_injected_procedures_always_empty(self, tmp_path: Path):
        """BuildResult.injected_procedures is always an empty list."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

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

    def test_mixed_skills_ignored_in_tool_description(self, tmp_path: Path):
        """build_skill_tool_description ignores meta lists (catalog is in system prompt)."""
        personal_meta = SkillMeta(
            name="tool-creator",
            description="新しいPythonツールモジュールを作成するためのメタスキル",
            path=tmp_path / "skills" / "tool-creator.md",
            is_common=False,
        )
        common_meta = SkillMeta(
            name="animaworks-guide",
            description="AnimaWorksフレームワークの仕組みガイド",
            path=tmp_path / "common_skills" / "animaworks-guide.md",
            is_common=True,
        )
        procedure_meta = SkillMeta(
            name="incident-response",
            description="障害発生時の対応手順書",
            path=tmp_path / "procedures" / "incident-response.md",
            is_common=False,
        )

        desc = build_skill_tool_description(
            [personal_meta],
            [common_meta],
            [procedure_meta],
        )

        assert "tool-creator" not in desc
        assert "animaworks-guide" not in desc
        assert "incident-response" not in desc
        assert desc == "\n".join([t("skill.desc_line1"), t("skill.desc_line2")])

    def test_skill_tool_loads_content_on_demand(self, tmp_path: Path):
        """load_and_render_skill returns the full skill content when invoked."""
        skills_dir = tmp_path / "skills"
        common_skills_dir = tmp_path / "common_skills"
        procedures_dir = tmp_path / "procedures"
        anima_dir = tmp_path / "animas" / "alice"
        for d in (skills_dir, common_skills_dir, procedures_dir, anima_dir):
            d.mkdir(parents=True, exist_ok=True)

        _make_skill_file(
            skills_dir / "cron-management" / "SKILL.md",
            name="cron-management",
            description="cron.mdの読み書き・更新スキル",
            body="## cron.mdの構造\n\nschedule: フィールドにcron式を記載する。",
        )

        result = load_and_render_skill(
            skill_name="cron-management",
            anima_dir=anima_dir,
            skills_dir=skills_dir,
            common_skills_dir=common_skills_dir,
            procedures_dir=procedures_dir,
        )

        # Full content is returned
        assert "cron.mdの構造" in result
        assert "schedule: フィールドにcron式を記載する。" in result

    def test_skill_description_survives_db_override(self, tmp_path: Path):
        """Skill tool description must not be overwritten by apply_db_descriptions.

        The skill tool is appended AFTER apply_db_descriptions, so even if
        the DB has an entry for 'skill', the short usage text from ``t()`` wins.
        """
        personal_meta = SkillMeta(
            name="deploy",
            description="デプロイ手順",
            path=tmp_path / "skills" / "deploy.md",
            is_common=False,
        )

        # Mock the prompt DB to return an overriding description for 'skill'
        fake_store = MagicMock()
        fake_store.list_descriptions.return_value = [
            {"name": "skill", "description": "DB OVERRIDE — should not appear"},
        ]

        with patch("core.tooling.prompt_db.get_prompt_store", return_value=fake_store):
            tools = build_tool_list(
                include_skill_tools=True,
                skill_metas=[personal_meta],
                common_skill_metas=[],
                procedure_metas=[],
            )

        skill_tool = next(x for x in tools if x["name"] == "skill")
        assert "deploy" not in skill_tool["description"]
        assert "<available_skills>" not in skill_tool["description"]
        assert "DB OVERRIDE" not in skill_tool["description"]
        assert t("skill.desc_line1") in skill_tool["description"]


class TestHandleSkillIntegration:
    """Integration tests: invoke skill via ToolHandler._handle_skill."""

    @staticmethod
    def _make_handler(anima_dir: Path, common_skills_dir: Path) -> ToolHandler:
        """Create a ToolHandler with minimal mocking."""
        from core.tooling.handler import ToolHandler

        memory = MagicMock()
        memory.anima_dir = anima_dir

        with patch("core.tooling.handler.ActivityLogger"), patch("core.tooling.handler.ExternalToolDispatcher"):
            handler = ToolHandler(anima_dir=anima_dir, memory=memory)
        return handler

    def test_handle_skill_returns_content(self, tmp_path: Path):
        """ToolHandler._handle_skill loads and returns skill content."""
        anima_dir = tmp_path / "animas" / "alice"
        skills_dir = anima_dir / "skills"
        common_skills_dir = tmp_path / "common_skills"
        procedures_dir = anima_dir / "procedures"
        for d in (anima_dir, skills_dir, common_skills_dir, procedures_dir):
            d.mkdir(parents=True, exist_ok=True)

        _make_skill_file(
            skills_dir / "test-skill" / "SKILL.md",
            name="test-skill",
            description="A test skill for integration",
            body="## Steps\n\n1. Do this.\n2. Do that.",
        )

        handler = self._make_handler(anima_dir, common_skills_dir)
        with patch("core.paths.get_common_skills_dir", return_value=common_skills_dir):
            result = handler._handle_skill({"skill_name": "test-skill"})

        assert "Steps" in result
        assert "Do this." in result
        assert "Do that." in result

    def test_handle_skill_nonexistent_returns_error(self, tmp_path: Path):
        """ToolHandler._handle_skill for missing skill returns error."""
        anima_dir = tmp_path / "animas" / "alice"
        skills_dir = anima_dir / "skills"
        common_skills_dir = tmp_path / "common_skills"
        procedures_dir = anima_dir / "procedures"
        for d in (anima_dir, skills_dir, common_skills_dir, procedures_dir):
            d.mkdir(parents=True, exist_ok=True)

        handler = self._make_handler(anima_dir, common_skills_dir)
        with patch("core.paths.get_common_skills_dir", return_value=common_skills_dir):
            result = handler._handle_skill({"skill_name": "nonexistent"})

        assert "見つかりません" in result

    def test_handle_skill_empty_name_returns_error(self, tmp_path: Path):
        """ToolHandler._handle_skill with empty skill_name returns error."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        common_skills_dir = tmp_path / "common_skills"
        common_skills_dir.mkdir(parents=True)

        handler = self._make_handler(anima_dir, common_skills_dir)
        result = handler._handle_skill({"skill_name": ""})

        assert "必須" in result

    def test_handle_skill_path_traversal_blocked(self, tmp_path: Path):
        """ToolHandler._handle_skill rejects path traversal attempts."""
        anima_dir = tmp_path / "animas" / "alice"
        skills_dir = anima_dir / "skills"
        common_skills_dir = tmp_path / "common_skills"
        procedures_dir = anima_dir / "procedures"
        for d in (anima_dir, skills_dir, common_skills_dir, procedures_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Create a decoy file outside skills directory
        decoy = tmp_path / "secret.md"
        decoy.write_text("TOP SECRET DATA", encoding="utf-8")

        handler = self._make_handler(anima_dir, common_skills_dir)
        with patch("core.paths.get_common_skills_dir", return_value=common_skills_dir):
            result = handler._handle_skill({"skill_name": "../../secret"})

        assert "見つかりません" in result
        assert "TOP SECRET DATA" not in result
