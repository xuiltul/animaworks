"""Unit tests for core/memory/skill_metadata.py — SkillMetadataService."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.skill_metadata import SkillMetadataService
from core.schemas import SkillMeta


# ── extract_skill_meta ─────────────────────────────────────


class TestExtractSkillMetaWithFrontmatter:
    """YAML frontmatter with name + description is correctly parsed."""

    def test_parses_name_and_description(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "deploy.md"
        skill_file.write_text(
            "---\n"
            "name: deploy-skill\n"
            "description: デプロイ手順「deploy」「リリース」\n"
            "---\n"
            "\n"
            "# Deploy Skill\n"
            "\nBody content here.\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.name == "deploy-skill"
        assert meta.description == "デプロイ手順「deploy」「リリース」"
        assert meta.path == skill_file
        assert meta.is_common is False

    def test_extra_fields_do_not_interfere(self, tmp_path: Path) -> None:
        """Extra YAML fields (version, tags) are ignored gracefully."""
        skill_file = tmp_path / "advanced.md"
        skill_file.write_text(
            "---\n"
            "name: advanced-tool\n"
            "description: 高度な検索「search」\n"
            "version: 2.1\n"
            "tags: [search, query]\n"
            "---\n"
            "\nBody.\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.name == "advanced-tool"
        assert meta.description == "高度な検索「search」"

    def test_description_is_stripped(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace in description is stripped."""
        skill_file = tmp_path / "ws.md"
        skill_file.write_text(
            "---\n"
            "name: ws-skill\n"
            "description: '  spaced out  '\n"
            "---\n"
            "\nContent.\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.description == "spaced out"


class TestExtractSkillMetaWithoutFrontmatter:
    """File without frontmatter uses filename stem as name."""

    def test_filename_fallback(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "my-skill.md"
        skill_file.write_text(
            "# My Skill\n\nSome instructions.\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.name == "my-skill"
        assert meta.description == ""
        assert meta.path == skill_file

    def test_plain_text_no_heading(self, tmp_path: Path) -> None:
        """File with no heading and no frontmatter returns stem and empty description."""
        skill_file = tmp_path / "simple.md"
        skill_file.write_text("Just some text.\n", encoding="utf-8")

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.name == "simple"
        assert meta.description == ""


class TestExtractSkillMetaLegacyFormat:
    """Legacy format with ## 概要 section extracts description."""

    def test_legacy_overview_section(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "legacy.md"
        skill_file.write_text(
            "# レガシースキル\n"
            "\n"
            "## 概要\n"
            "\n"
            "cronジョブの設定と管理を行うスキル\n"
            "\n"
            "## 手順\n"
            "\n"
            "1. 手順内容\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.name == "legacy"
        assert meta.description == "cronジョブの設定と管理を行うスキル"

    def test_legacy_overview_stops_at_next_heading(self, tmp_path: Path) -> None:
        """Only the first non-empty line after ## 概要 is used."""
        skill_file = tmp_path / "multi.md"
        skill_file.write_text(
            "# Skill\n"
            "## 概要\n"
            "First line is the description\n"
            "Second line is ignored\n"
            "## 手順\n"
            "Steps here\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.description == "First line is the description"

    def test_legacy_empty_overview(self, tmp_path: Path) -> None:
        """Empty ## 概要 section yields empty description."""
        skill_file = tmp_path / "empty-overview.md"
        skill_file.write_text(
            "# Skill\n"
            "## 概要\n"
            "## 手順\n"
            "Steps here\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)

        assert meta.description == ""


class TestExtractSkillMetaIsCommon:
    """is_common flag is correctly propagated."""

    def test_is_common_false_by_default(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "personal.md"
        skill_file.write_text(
            "---\nname: p\ndescription: personal\n---\n\nContent.\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file)
        assert meta.is_common is False

    def test_is_common_true(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "shared.md"
        skill_file.write_text(
            "---\nname: shared\ndescription: shared skill\n---\n\nContent.\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file, is_common=True)
        assert meta.is_common is True

    def test_is_common_false_explicit(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "mine.md"
        skill_file.write_text(
            "---\nname: mine\ndescription: mine\n---\n\nContent.\n",
            encoding="utf-8",
        )

        meta = SkillMetadataService.extract_skill_meta(skill_file, is_common=False)
        assert meta.is_common is False


# ── list_skill_metas ───────────────────────────────────────


class TestListSkillMetas:
    """Tests for listing personal skill metadata files."""

    def test_lists_multiple_skills(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        (skills_dir / "alpha.md").write_text(
            "---\nname: alpha\ndescription: Alpha skill\n---\n\nAlpha body.\n",
            encoding="utf-8",
        )
        (skills_dir / "beta.md").write_text(
            "---\nname: beta\ndescription: Beta skill\n---\n\nBeta body.\n",
            encoding="utf-8",
        )

        service = SkillMetadataService(skills_dir, common_dir)
        metas = service.list_skill_metas()

        assert len(metas) == 2
        names = [m.name for m in metas]
        assert "alpha" in names
        assert "beta" in names
        # All personal skills should have is_common=False
        assert all(not m.is_common for m in metas)

    def test_ignores_non_md_files(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        (skills_dir / "valid.md").write_text("# Valid\n", encoding="utf-8")
        (skills_dir / "ignore.txt").write_text("not a skill\n", encoding="utf-8")

        service = SkillMetadataService(skills_dir, common_dir)
        metas = service.list_skill_metas()

        assert len(metas) == 1
        assert metas[0].name == "valid"

    def test_empty_skills_dir(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        service = SkillMetadataService(skills_dir, common_dir)
        metas = service.list_skill_metas()

        assert metas == []

    def test_returns_sorted_order(self, tmp_path: Path) -> None:
        """Metas are returned in sorted (alphabetical) order."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        (skills_dir / "charlie.md").write_text("# Charlie\n", encoding="utf-8")
        (skills_dir / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
        (skills_dir / "bravo.md").write_text("# Bravo\n", encoding="utf-8")

        service = SkillMetadataService(skills_dir, common_dir)
        metas = service.list_skill_metas()

        names = [m.name for m in metas]
        assert names == ["alpha", "bravo", "charlie"]


# ── list_common_skill_metas ────────────────────────────────


class TestListCommonSkillMetas:
    """Tests for listing common (shared) skill metadata."""

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "nonexistent_common_skills"
        # Do NOT create common_dir — it should not exist

        service = SkillMetadataService(skills_dir, common_dir)
        metas = service.list_common_skill_metas()

        assert metas == []

    def test_lists_common_skills_with_is_common_true(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        (common_dir / "shared-tool.md").write_text(
            "---\nname: shared-tool\ndescription: A shared tool\n---\n\nShared.\n",
            encoding="utf-8",
        )

        service = SkillMetadataService(skills_dir, common_dir)
        metas = service.list_common_skill_metas()

        assert len(metas) == 1
        assert metas[0].name == "shared-tool"
        assert metas[0].is_common is True


# ── list_skill_summaries ───────────────────────────────────


class TestListSkillSummaries:
    """Tests for (name, description) tuple output."""

    def test_returns_name_description_tuples(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        (skills_dir / "coding.md").write_text(
            "---\nname: coding\ndescription: Write code efficiently\n---\n\n# Coding\n",
            encoding="utf-8",
        )
        (skills_dir / "review.md").write_text(
            "---\nname: review\ndescription: Code review process\n---\n\n# Review\n",
            encoding="utf-8",
        )

        service = SkillMetadataService(skills_dir, common_dir)
        summaries = service.list_skill_summaries()

        assert len(summaries) == 2
        assert all(isinstance(s, tuple) and len(s) == 2 for s in summaries)
        names = [s[0] for s in summaries]
        assert "coding" in names
        assert "review" in names
        # Check description values
        descs = {s[0]: s[1] for s in summaries}
        assert descs["coding"] == "Write code efficiently"
        assert descs["review"] == "Code review process"

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        service = SkillMetadataService(skills_dir, common_dir)
        summaries = service.list_skill_summaries()

        assert summaries == []


# ── list_common_skill_summaries ────────────────────────────


class TestListCommonSkillSummaries:
    """Tests for common skill (name, description) tuple output."""

    def test_returns_common_summaries(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()

        (common_dir / "cron-management.md").write_text(
            "---\nname: cron-management\ndescription: Manage cron tasks\n---\n\n# Cron\n",
            encoding="utf-8",
        )

        service = SkillMetadataService(skills_dir, common_dir)
        summaries = service.list_common_skill_summaries()

        assert len(summaries) == 1
        assert summaries[0] == ("cron-management", "Manage cron tasks")

    def test_nonexistent_common_dir_returns_empty(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        common_dir = tmp_path / "nonexistent"

        service = SkillMetadataService(skills_dir, common_dir)
        summaries = service.list_common_skill_summaries()

        assert summaries == []
