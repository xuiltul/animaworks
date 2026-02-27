from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.tooling.skill_creator."""

import yaml
import pytest
from pathlib import Path

from core.tooling.skill_creator import create_skill_directory, _validate_filename


class TestValidateFilename:
    """Test _validate_filename path traversal prevention."""

    def test_valid_filename(self, tmp_path):
        assert _validate_filename("guide.md", tmp_path) is True

    def test_empty_rejected(self, tmp_path):
        assert _validate_filename("", tmp_path) is False

    def test_slash_rejected(self, tmp_path):
        assert _validate_filename("foo/bar.md", tmp_path) is False

    def test_backslash_rejected(self, tmp_path):
        assert _validate_filename("foo\\bar.md", tmp_path) is False

    def test_dotdot_rejected(self, tmp_path):
        assert _validate_filename("../secret", tmp_path) is False


class TestCreateSkillDirectory:
    """Test create_skill_directory directory creation and security."""

    def test_basic_creation(self, tmp_path):
        result = create_skill_directory("my-skill", "A test skill", "# My Skill\n\nBody", tmp_path)
        assert "my-skill" in result
        skill_md = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "---" in skill_md
        fm_text = skill_md.split("---")[1]
        fm = yaml.safe_load(fm_text)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "A test skill"
        assert "# My Skill" in skill_md

    def test_with_references(self, tmp_path):
        refs = [{"filename": "guide.md", "content": "# Guide\nContent"}]
        result = create_skill_directory("my-skill", "desc", "body", tmp_path, references=refs)
        assert "references/guide.md" in result
        assert (tmp_path / "my-skill" / "references" / "guide.md").is_file()

    def test_with_templates(self, tmp_path):
        tpls = [{"filename": "tmpl.md", "content": "# Template"}]
        result = create_skill_directory("my-skill", "desc", "body", tmp_path, templates=tpls)
        assert "templates/tmpl.md" in result
        assert (tmp_path / "my-skill" / "templates" / "tmpl.md").is_file()

    def test_path_traversal_skill_name(self, tmp_path):
        result = create_skill_directory("../evil", "desc", "body", tmp_path)
        assert "無効" in result

    def test_path_traversal_reference_filename(self, tmp_path):
        refs = [{"filename": "../../etc/passwd", "content": "evil"}]
        result = create_skill_directory("safe-skill", "desc", "body", tmp_path, references=refs)
        assert "safe-skill" in result
        assert not (tmp_path / "etc").exists()
        assert not (tmp_path / "safe-skill" / "references" / "../../etc/passwd").exists()

    def test_path_traversal_template_filename(self, tmp_path):
        tpls = [{"filename": "../../../secret", "content": "evil"}]
        result = create_skill_directory("safe-skill", "desc", "body", tmp_path, templates=tpls)
        assert "safe-skill" in result
        assert not (tmp_path / "secret").exists()

    def test_overwrite_existing(self, tmp_path):
        create_skill_directory("my-skill", "v1", "body1", tmp_path)
        create_skill_directory("my-skill", "v2", "body2", tmp_path)
        skill_md = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "v2" in skill_md

    def test_allowed_tools(self, tmp_path):
        result = create_skill_directory(
            "my-skill", "desc", "body", tmp_path, allowed_tools=["web_search", "read_file"]
        )
        skill_md = (tmp_path / "my-skill" / "SKILL.md").read_text()
        fm_text = skill_md.split("---")[1]
        fm = yaml.safe_load(fm_text)
        assert fm["allowed_tools"] == ["web_search", "read_file"]

    def test_empty_refs_and_templates(self, tmp_path):
        result = create_skill_directory(
            "my-skill", "desc", "body", tmp_path, references=[], templates=[]
        )
        assert "SKILL.md" in result
        assert not (tmp_path / "my-skill" / "references").exists()
        assert not (tmp_path / "my-skill" / "templates").exists()

    def test_multiple_references(self, tmp_path):
        refs = [
            {"filename": "a.md", "content": "A"},
            {"filename": "b.md", "content": "B"},
        ]
        result = create_skill_directory("my-skill", "desc", "body", tmp_path, references=refs)
        assert (tmp_path / "my-skill" / "references" / "a.md").read_text() == "A"
        assert (tmp_path / "my-skill" / "references" / "b.md").read_text() == "B"
