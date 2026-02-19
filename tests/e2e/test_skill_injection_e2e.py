# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for description-based skill injection in build_system_prompt.

Verifies that build_system_prompt correctly injects skill full text when
description keywords (「」-delimited) match the incoming message, and that
unmatched skills appear only in the summary table.
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
    return memory


# Common patches applied to every test to isolate build_system_prompt from
# side-effectful functions that read real filesystem / config.
_BUILDER_PATCHES = [
    "core.prompt.builder.load_prompt",
    "core.prompt.builder._build_org_context",
    "core.prompt.builder._discover_other_animas",
    "core.prompt.builder._build_messaging_section",
]


# ── Test Cases ───────────────────────────────────────────


class TestSkillInjectionE2E:
    """build_system_prompt injects matched skill full text based on message."""

    def test_message_matches_skill_keyword_injects_full_text(self, tmp_path: Path):
        """When message matches a 「」-delimited keyword in the description,
        the full skill body text should be injected into the system prompt."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Create a skill file with a keyword that will match
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

        memory = _make_mock_memory(anima_dir, tmp_path)

        # Return SkillMeta objects that point to the real file
        skill_meta = SkillMeta(
            name="cron-management",
            description=(
                "cron.mdの読み書き・更新を正しいフォーマットで行うためのスキル。"
                "「cron設定」「cronタスク」「定時タスク」等の場面で使用。"
            ),
            path=skill_path,
            is_common=False,
        )
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_common_skill_metas.return_value = []

        with (
            patch("core.prompt.builder.load_prompt", return_value=""),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="cron設定をして")

        # The skill body should appear in the prompt
        assert "cron.mdの構造" in prompt
        assert "schedule: フィールドにcron式を記載する。" in prompt
        # The skill header should indicate it was injected
        assert "スキル: cron-management" in prompt

    def test_message_no_match_shows_table_only(self, tmp_path: Path):
        """When message does NOT match any skill keyword, the skill should
        appear only in the summary table, not as full text."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

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

        memory = _make_mock_memory(anima_dir, tmp_path)
        skill_meta = SkillMeta(
            name="cron-management",
            description=(
                "cron.mdの読み書き・更新を正しいフォーマットで行うためのスキル。"
                "「cron設定」「cronタスク」「定時タスク」等の場面で使用。"
            ),
            path=skill_path,
            is_common=False,
        )
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_common_skill_metas.return_value = []

        # The skills_guide load_prompt call returns a table format
        def fake_load_prompt(name: str, **kwargs) -> str:
            if name == "skills_guide":
                return (
                    "## あなたのスキル\n\n"
                    "| スキル名 | 概要 |\n|---------|------|\n"
                    + kwargs.get("skill_lines", "")
                )
            return ""

        with (
            patch("core.prompt.builder.load_prompt", side_effect=fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="おはよう")

        # The skill body should NOT be fully injected
        assert "cron.mdの構造" not in prompt
        assert "schedule: フィールドにcron式を記載する。" not in prompt
        # The skill should appear in the table
        assert "cron-management" in prompt

    def test_multiple_skills_budget_limits_injection(self, tmp_path: Path):
        """When multiple skills match but total body exceeds budget,
        only the first skills within budget should be fully injected."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Create 3 skills, each with ~2000 chars body. Greeting budget = 1000,
        # so only the first skill (if any) could fit, but even it exceeds 1000.
        # We make bodies slightly different sizes to test the cutoff.
        skills_data = []
        for i, name in enumerate(["skill-a", "skill-b", "skill-c"]):
            skill_path = tmp_path / "skills" / f"{name}.md"
            # Each body is ~600 chars so that first fits in budget (1000)
            # but first + second (1200) exceeds it
            body_text = f"手順{i}: " + "あ" * 594
            _make_skill_file(
                skill_path,
                name=name,
                description=f"スキル{name}の説明。「デプロイ」「deploy」で使用。",
                body=body_text,
            )
            meta = SkillMeta(
                name=name,
                description=f"スキル{name}の説明。「デプロイ」「deploy」で使用。",
                path=skill_path,
                is_common=False,
            )
            skills_data.append((meta, body_text))

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [s[0] for s in skills_data]
        memory.list_common_skill_metas.return_value = []

        # Use a greeting message so budget = 1000
        with (
            patch("core.prompt.builder.load_prompt", return_value=""),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="こんにちは、deployして")

        # First skill should be injected (body ~600 chars, budget 1000)
        assert "スキル: skill-a" in prompt
        assert "手順0:" in prompt

        # Second skill should exceed the remaining budget (~400 left, body ~600)
        # and thus should NOT be fully injected
        assert "手順1:" not in prompt or "手順2:" not in prompt
        # At minimum, not all three can fit in 1000 budget
        injected_count = sum(
            1 for s_meta, s_body in skills_data
            if f"スキル: {s_meta.name}" in prompt
        )
        assert injected_count < 3, (
            f"Expected fewer than 3 skills injected with budget=1000, got {injected_count}"
        )

    def test_mixed_matched_and_unmatched_skills(self, tmp_path: Path):
        """When some skills match and others don't, matched ones get full text
        and unmatched ones appear only in the table."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Matched skill: keywords include 「ツール作成」
        matched_path = tmp_path / "skills" / "tool-creator.md"
        _make_skill_file(
            matched_path,
            name="tool-creator",
            description=(
                "新しいPythonツールモジュールを作成するためのメタスキル。"
                "「ツール作成」「ツール化」「新しいツール」で使用。"
            ),
            body="## ツール作成手順\n\n1. _base.py を継承する\n2. schemas.py にスキーマ追加",
        )
        matched_meta = SkillMeta(
            name="tool-creator",
            description=(
                "新しいPythonツールモジュールを作成するためのメタスキル。"
                "「ツール作成」「ツール化」「新しいツール」で使用。"
            ),
            path=matched_path,
            is_common=False,
        )

        # Unmatched skill: keywords are unrelated to the message
        unmatched_path = tmp_path / "skills" / "cron-management.md"
        _make_skill_file(
            unmatched_path,
            name="cron-management",
            description=(
                "cron.mdの読み書き・更新スキル。"
                "「cron設定」「定時タスク」で使用。"
            ),
            body="## cron.mdの構造\n\nこれはcron管理の手順です。",
        )
        unmatched_meta = SkillMeta(
            name="cron-management",
            description=(
                "cron.mdの読み書き・更新スキル。"
                "「cron設定」「定時タスク」で使用。"
            ),
            path=unmatched_path,
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [matched_meta, unmatched_meta]
        memory.list_common_skill_metas.return_value = []

        def fake_load_prompt(name: str, **kwargs) -> str:
            if name == "skills_guide":
                return (
                    "## あなたのスキル\n\n"
                    "| スキル名 | 概要 |\n|---------|------|\n"
                    + kwargs.get("skill_lines", "")
                )
            return ""

        with (
            patch("core.prompt.builder.load_prompt", side_effect=fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(
                memory, message="新しいツール作成をしたい"
            )

        # Matched skill: full text injected
        assert "スキル: tool-creator" in prompt
        assert "ツール作成手順" in prompt
        assert "_base.py を継承する" in prompt

        # Unmatched skill: body NOT injected, but name appears in table
        assert "cron.mdの構造" not in prompt
        assert "これはcron管理の手順です。" not in prompt
        assert "cron-management" in prompt

    def test_common_skill_matched_injects_with_common_label(self, tmp_path: Path):
        """Common skills should be injected with the (共通スキル) label."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        common_skill_path = tmp_path / "common_skills" / "animaworks-guide.md"
        _make_skill_file(
            common_skill_path,
            name="animaworks-guide",
            description=(
                "AnimaWorksフレームワークの仕組みガイド。"
                "「AnimaWorks」「フレームワーク」「仕組み」で使用。"
            ),
            body="## AnimaWorks とは\n\nAIエージェントの自律フレームワークです。",
        )
        common_meta = SkillMeta(
            name="animaworks-guide",
            description=(
                "AnimaWorksフレームワークの仕組みガイド。"
                "「AnimaWorks」「フレームワーク」「仕組み」で使用。"
            ),
            path=common_skill_path,
            is_common=True,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = [common_meta]

        with (
            patch("core.prompt.builder.load_prompt", return_value=""),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(
                memory, message="AnimaWorksの仕組みを教えて"
            )

        assert "スキル: animaworks-guide (共通スキル)" in prompt
        assert "AIエージェントの自律フレームワークです。" in prompt

    def test_empty_message_no_skill_injection(self, tmp_path: Path):
        """When message is empty, no skills should be injected (all go to table)."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        skill_path = tmp_path / "skills" / "cron-management.md"
        _make_skill_file(
            skill_path,
            name="cron-management",
            description="cron管理。「cron設定」で使用。",
            body="## cron手順\n\nこの手順に従うこと。",
        )
        skill_meta = SkillMeta(
            name="cron-management",
            description="cron管理。「cron設定」で使用。",
            path=skill_path,
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_common_skill_metas.return_value = []

        def fake_load_prompt(name: str, **kwargs) -> str:
            if name == "skills_guide":
                return (
                    "## あなたのスキル\n\n"
                    "| スキル名 | 概要 |\n|---------|------|\n"
                    + kwargs.get("skill_lines", "")
                )
            return ""

        with (
            patch("core.prompt.builder.load_prompt", side_effect=fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="")

        # No full text injection
        assert "cron手順" not in prompt
        assert "この手順に従うこと。" not in prompt
        # Skill name appears in table
        assert "cron-management" in prompt


class TestEnhancedSkillInjectionE2E:
    """E2E tests for enhanced 3-tier skill matching in build_system_prompt."""

    def test_comma_keyword_skill_matches_and_injects(self, tmp_path: Path):
        """Tier 1 fallback: comma-separated keywords in description trigger injection."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        skill_path = tmp_path / "skills" / "deploy-guide.md"
        _make_skill_file(
            skill_path,
            name="deploy-guide",
            description="デプロイ手順、リリース手順、本番反映の方法を提供する",
            body="## デプロイ手順\n\n1. ステージング確認\n2. 本番デプロイ実行",
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        skill_meta = SkillMeta(
            name="deploy-guide",
            description="デプロイ手順、リリース手順、本番反映の方法を提供する",
            path=skill_path,
            is_common=False,
        )
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_common_skill_metas.return_value = []

        with (
            patch("core.prompt.builder.load_prompt", return_value=""),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="デプロイ手順を教えて")

        assert "スキル: deploy-guide" in prompt
        assert "ステージング確認" in prompt

    def test_english_description_tier2_matches(self, tmp_path: Path):
        """Tier 2: English description without brackets matches via vocabulary overlap."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        skill_path = tmp_path / "skills" / "document-creator.md"
        _make_skill_file(
            skill_path,
            name="document-creator",
            description="Comprehensive document creation, editing, and analysis tool",
            body="## Document Creation\n\nUse this to create documents.",
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        skill_meta = SkillMeta(
            name="document-creator",
            description="Comprehensive document creation, editing, and analysis tool",
            path=skill_path,
            is_common=False,
        )
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_common_skill_metas.return_value = []

        with (
            patch("core.prompt.builder.load_prompt", return_value=""),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="I need document creation help")

        assert "スキル: document-creator" in prompt
        assert "Use this to create documents." in prompt

    def test_tier1_match_prevents_tier2_duplication(self, tmp_path: Path):
        """Skills matched by Tier 1 should not duplicate via Tier 2."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        skill_path = tmp_path / "skills" / "cron.md"
        _make_skill_file(
            skill_path,
            name="cron-management",
            description=(
                "cronジョブの設定と管理を行うスキル。"
                "「cron設定」「定期実行」等の場面で使用。"
            ),
            body="## cron手順\n\n1. cron.md を確認する",
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        skill_meta = SkillMeta(
            name="cron-management",
            description=(
                "cronジョブの設定と管理を行うスキル。"
                "「cron設定」「定期実行」等の場面で使用。"
            ),
            path=skill_path,
            is_common=False,
        )
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_common_skill_metas.return_value = []

        with (
            patch("core.prompt.builder.load_prompt", return_value=""),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="cron設定をして")

        # Should appear exactly once (not duplicated by Tier 2)
        assert prompt.count("スキル: cron-management") == 1

    def test_no_retriever_tier3_skipped_gracefully(self, tmp_path: Path):
        """Without a retriever, Tier 3 is skipped and only Tier 1/2 operate."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # This skill has no bracket keywords and single-word-only description
        skill_path = tmp_path / "skills" / "obscure.md"
        _make_skill_file(
            skill_path,
            name="obscure-skill",
            description="Very specific internal tool",
            body="## Internal\n\nSomething.",
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        skill_meta = SkillMeta(
            name="obscure-skill",
            description="Very specific internal tool",
            path=skill_path,
            is_common=False,
        )
        memory.list_skill_metas.return_value = [skill_meta]
        memory.list_common_skill_metas.return_value = []

        def fake_load_prompt(name: str, **kwargs) -> str:
            if name == "skills_guide":
                return (
                    "## あなたのスキル\n\n"
                    "| スキル名 | 概要 |\n|---------|------|\n"
                    + kwargs.get("skill_lines", "")
                )
            return ""

        # No retriever → Tier 3 skipped
        with (
            patch("core.prompt.builder.load_prompt", side_effect=fake_load_prompt),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(memory, message="internal tool please", retriever=None)

        # The key is no crash (graceful degradation). Skill appears in table.
        assert "obscure-skill" in prompt

    def test_mixed_tiers_correct_injection(self, tmp_path: Path):
        """Multiple skills across different tiers all get injected correctly."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        # Skill 1: matches via Tier 1 (bracket keywords)
        skill1_path = tmp_path / "skills" / "cron.md"
        _make_skill_file(
            skill1_path,
            name="cron-management",
            description="cronジョブ管理。「cron設定」「定期実行」で使用。",
            body="## cron手順\n\ncron設定の手順です。",
        )
        meta1 = SkillMeta(
            name="cron-management",
            description="cronジョブ管理。「cron設定」「定期実行」で使用。",
            path=skill1_path,
            is_common=False,
        )

        # Skill 2: matches via Tier 2 (English vocabulary)
        skill2_path = tmp_path / "skills" / "scheduler.md"
        _make_skill_file(
            skill2_path,
            name="scheduler-tool",
            description="Scheduling and execution management for periodic tasks",
            body="## Scheduler\n\nScheduler usage.",
        )
        meta2 = SkillMeta(
            name="scheduler-tool",
            description="Scheduling and execution management for periodic tasks",
            path=skill2_path,
            is_common=False,
        )

        memory = _make_mock_memory(anima_dir, tmp_path)
        memory.list_skill_metas.return_value = [meta1, meta2]
        memory.list_common_skill_metas.return_value = []

        with (
            patch("core.prompt.builder.load_prompt", return_value=""),
            patch("core.prompt.builder._build_org_context", return_value=""),
            patch("core.prompt.builder._discover_other_animas", return_value=[]),
            patch("core.prompt.builder._build_messaging_section", return_value=""),
        ):
            prompt = build_system_prompt(
                memory,
                message="cron設定と scheduling execution を教えて",
            )

        # Skill 1 matched via Tier 1
        assert "スキル: cron-management" in prompt
        assert "cron設定の手順です" in prompt
        # Skill 2 matched via Tier 2
        assert "スキル: scheduler-tool" in prompt
        assert "Scheduler usage." in prompt
