from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for skill directory structure and metadata listing."""

import pytest

from core.tooling.skill_creator import create_skill_directory
from core.memory.skill_metadata import SkillMetadataService


def _list_available_names(skills_dir, common_skills_dir, procedures_dir):
    """Mirror resolution listing for tests (skills/common procedures names)."""
    names: list[str] = []
    for d in (skills_dir, common_skills_dir):
        if d.is_dir():
            names.extend(f.parent.name for f in sorted(d.glob("*/SKILL.md")))
    if procedures_dir.is_dir():
        names.extend(f.stem for f in sorted(procedures_dir.glob("*.md")))
    return names


class TestSkillDirectoryE2E:
    """End-to-end tests for skill directory structure."""

    @pytest.fixture
    def skill_dirs(self, tmp_path):
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        for d in (skills, common, procedures):
            d.mkdir()
        return skills, common, procedures

    def test_create_and_read_personal_skill_file(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("my-tool", "A tool skill", "# My Tool\n\nDo stuff.", skills)
        text = (skills / "my-tool" / "SKILL.md").read_text(encoding="utf-8")
        assert "# My Tool" in text
        assert "Do stuff." in text

    def test_create_and_read_common_skill_file(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("shared-skill", "Shared desc", "# Shared\n\nShared body.", common)
        text = (common / "shared-skill" / "SKILL.md").read_text(encoding="utf-8")
        assert "# Shared" in text

    def test_create_skill_with_references_and_read(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        refs = [{"filename": "guide.md", "content": "# Reference Guide\n\nDetails."}]
        create_skill_directory(
            "ref-skill",
            "Has references",
            "# Ref Skill\n\nSee references/guide.md",
            skills,
            references=refs,
        )
        guide = (skills / "ref-skill" / "references" / "guide.md").read_text()
        assert "Reference Guide" in guide

    def test_metadata_listing(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("alpha", "Alpha desc", "body", skills)
        create_skill_directory("beta", "Beta desc", "body", skills)
        svc = SkillMetadataService(skills, common)
        metas = svc.list_skill_metas()
        names = [m.name for m in metas]
        assert "alpha" in names
        assert "beta" in names

    def test_available_names_includes_procedures(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("my-skill", "desc", "body", skills)
        (procedures / "my-proc.md").write_text(
            "---\nname: my-proc\ndescription: proc\n---\n# Proc", encoding="utf-8"
        )
        names = _list_available_names(skills, common, procedures)
        assert "my-skill" in names
        assert "my-proc" in names

    def test_old_flat_file_not_in_skill_glob(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        (skills / "old-skill.md").write_text("---\nname: old-skill\n---\n# Old", encoding="utf-8")
        names = _list_available_names(skills, common, procedures)
        assert "old-skill" not in names

    def test_personal_overrides_common_file_presence(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("same-name", "Common version", "Common body", common)
        create_skill_directory("same-name", "Personal version", "Personal body", skills)
        personal = (skills / "same-name" / "SKILL.md").read_text(encoding="utf-8")
        assert "Personal body" in personal
