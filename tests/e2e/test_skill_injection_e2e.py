# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for skill catalog (Group 4) and create_skill tool schema."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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

        # Create actual SKILL.md files so SkillIndex discovers them from disk
        personal_skill_dir = anima_dir / "skills" / "cron-management"
        personal_skill_dir.mkdir(parents=True)
        (personal_skill_dir / "SKILL.md").write_text(
            "---\nname: cron-management\ndescription: cron.mdの読み書き・更新スキル\n---\n\nBody",
            encoding="utf-8",
        )

        common_skills_dir = tmp_path / "common_skills"
        common_skill_dir = common_skills_dir / "animaworks-guide"
        common_skill_dir.mkdir(parents=True)
        (common_skill_dir / "SKILL.md").write_text(
            "---\nname: animaworks-guide\ndescription: AnimaWorksフレームワークの仕組みガイド\n---\n\nBody",
            encoding="utf-8",
        )

        memory = _make_mock_memory(anima_dir, tmp_path)

        with (
            patch("core.paths.get_common_skills_dir", return_value=common_skills_dir),
            patch(
                "core.prompt.builder._load_skill_catalog_router_settings",
                return_value=SimpleNamespace(enabled=False, top_k=5, min_score=1.15, include_body=True),
            ),
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="cron設定をして")

        prompt = result.system_prompt

        assert "| 名前 | 種別 | 概要 |" not in prompt
        assert "<available_skills>" in prompt
        assert "skills/cron-management/SKILL.md" in prompt
        assert "common_skills/animaworks-guide/SKILL.md" in prompt

    def test_background_prompt_excludes_human_approval_skills(self, tmp_path: Path):
        """Cron/background prompts do not expose skills that require separate human approval."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        safe_skill_dir = anima_dir / "skills" / "daily-report"
        safe_skill_dir.mkdir(parents=True)
        (safe_skill_dir / "SKILL.md").write_text(
            "---\nname: daily-report\ndescription: Safe daily report drafting\n---\n\nBody",
            encoding="utf-8",
        )

        risky_skill_dir = anima_dir / "skills" / "send-status"
        risky_skill_dir.mkdir(parents=True)
        (risky_skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: send-status\n"
            "description: Send status to an external channel\n"
            "risk:\n"
            "  external_send: true\n"
            "  requires_human_approval: true\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        common_skills_dir = tmp_path / "common_skills"
        common_skills_dir.mkdir(parents=True)
        memory = _make_mock_memory(anima_dir, tmp_path)
        settings = SimpleNamespace(enabled=False, top_k=5, min_score=1.15, include_body=True)

        def _build(trigger: str = "") -> str:
            with (
                patch("core.paths.get_common_skills_dir", return_value=common_skills_dir),
                patch("core.prompt.builder._load_skill_catalog_router_settings", return_value=settings),
                patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
                patch("core.prompt.builder._build_org_context", return_value=""),
                patch("core.prompt.builder._discover_other_animas", return_value=[]),
                patch("core.prompt.builder._build_messaging_section", return_value=""),
            ):
                return build_system_prompt(memory, message="send status", trigger=trigger).system_prompt

        chat_prompt = _build()
        assert "skills/send-status/SKILL.md" in chat_prompt
        assert "human-approval" in chat_prompt

        cron_prompt = _build("cron:daily")
        assert "skills/daily-report/SKILL.md" in cron_prompt
        assert "skills/send-status/SKILL.md" not in cron_prompt
        assert "human-approval" not in cron_prompt

    def test_skill_catalog_router_flag_limits_prompt_to_matches(self, tmp_path: Path):
        """When enabled, the skill catalog lists routed candidates instead of every skill."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        personal_skill_dir = anima_dir / "skills" / "cron-management"
        personal_skill_dir.mkdir(parents=True)
        (personal_skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: cron-management\n"
            "description: cron.mdの読み書き・定時タスク更新スキル\n"
            "tags: [cron]\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        common_skills_dir = tmp_path / "common_skills"
        image_skill_dir = common_skills_dir / "image-gen-tool"
        image_skill_dir.mkdir(parents=True)
        (image_skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: image-gen-tool\n"
            "description: 画像生成スキル\n"
            "tags: [image]\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        settings = SimpleNamespace(
            enabled=True,
            top_k=3,
            min_score=1.15,
            include_body=False,
        )

        with (
            patch("core.paths.get_common_skills_dir", return_value=common_skills_dir),
            patch("core.prompt.builder._load_skill_catalog_router_settings", return_value=settings),
            patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            result = build_system_prompt(memory, message="cron.mdに毎朝9時の定時タスクを追加して")

        prompt = result.system_prompt
        assert "<available_skills>" in prompt
        assert "skills/cron-management/SKILL.md" in prompt
        assert "match=high" in prompt
        assert "common_skills/image-gen-tool/SKILL.md" not in prompt

    def test_create_skill_in_build_tool_list(self, tmp_path: Path) -> None:
        """build_tool_list with include_create_skill=True adds skill creation tools."""
        tools = build_tool_list(include_create_skill=True)
        names = {x["name"] for x in tools}
        assert "create_skill" in names
        assert "promote_procedure_to_skill" in names
        assert "skill" not in names
        create = next(x for x in tools if x["name"] == "create_skill")
        assert create["name"] == "create_skill"
        assert len(create.get("description", "")) > 0

    def test_no_full_skill_body_in_system_prompt(self, tmp_path: Path):
        """Skill file body text is not injected into the system prompt (use read_memory_file)."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        skill_path = tmp_path / "skills" / "cron-management" / "SKILL.md"
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

    def test_build_result_has_no_injected_procedures_field(self, tmp_path: Path):
        """BuildResult no longer tracks prompt-injected procedures."""
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

        assert not hasattr(result, "injected_procedures")
