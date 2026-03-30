# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for skill catalog (Group 4) and create_skill tool schema."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.prompt.builder import build_system_prompt
from core.schemas import SkillMeta
from core.tooling.schemas import build_tool_list


# ── Helpers ──────────────────────────────────────────────


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


class TestSkillCatalogE2E:
    """Skills catalog lives in system prompt Group 4 (``<available_skills>``)."""

    def test_skills_paths_in_system_prompt(self, tmp_path: Path):
        """Personal and common skills appear as read_memory_file-compatible paths in Group 4."""
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

        assert "| 名前 | 種別 | 概要 |" not in prompt
        assert "<available_skills>" in prompt
        assert "skills/cron-management/SKILL.md" in prompt
        assert "common_skills/animaworks-guide/SKILL.md" in prompt

    def test_create_skill_in_build_tool_list(self, tmp_path: Path) -> None:
        """build_tool_list with include_create_skill=True adds create_skill only."""
        tools = build_tool_list(include_create_skill=True)
        names = {x["name"] for x in tools}
        assert "create_skill" in names
        assert "skill" not in names
        create = next(x for x in tools if x["name"] == "create_skill")
        assert create["name"] == "create_skill"
        assert len(create.get("description", "")) > 0

    def test_no_full_skill_body_in_system_prompt(self, tmp_path: Path):
        """Skill file body text is not injected into the system prompt (use read_memory_file)."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        skill_path = tmp_path / "skills" / "cron-management.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(
            "---\nname: cron-management\ndescription: x\n---\n\n## Body\nSECRET",
            encoding="utf-8",
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
        assert "SECRET" not in prompt

    def test_empty_skills_no_table(self, tmp_path: Path):
        """When there are no skills, no legacy table is injected."""
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
