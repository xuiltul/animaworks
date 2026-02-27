from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for skill directory structure and resolution flow."""

import pytest
from pathlib import Path

from core.tooling.skill_creator import create_skill_directory
from core.tooling.skill_tool import load_and_render_skill, _list_available_names
from core.memory.skill_metadata import SkillMetadataService


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

    def test_create_and_load_personal_skill(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("my-tool", "A tool skill", "# My Tool\n\nDo stuff.", skills)
        result = load_and_render_skill("my-tool", tmp_path, skills, common, procedures)
        assert "# My Tool" in result
        assert "Do stuff." in result

    def test_create_and_load_common_skill(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("shared-skill", "Shared desc", "# Shared\n\nShared body.", common)
        result = load_and_render_skill("shared-skill", tmp_path, skills, common, procedures)
        assert "# Shared" in result

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

    def test_old_flat_file_not_recognized(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        (skills / "old-skill.md").write_text("---\nname: old-skill\n---\n# Old", encoding="utf-8")
        result = load_and_render_skill("old-skill", tmp_path, skills, common, procedures)
        assert "見つかりません" in result

    def test_personal_overrides_common(self, tmp_path, skill_dirs):
        skills, common, procedures = skill_dirs
        create_skill_directory("same-name", "Common version", "Common body", common)
        create_skill_directory("same-name", "Personal version", "Personal body", skills)
        result = load_and_render_skill("same-name", tmp_path, skills, common, procedures)
        assert "Personal body" in result
