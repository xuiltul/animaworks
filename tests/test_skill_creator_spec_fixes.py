"""Unit tests for skill-creator spec alignment and path fixes.

Tests cover:
- common_skills/ path redirect in write_memory_file
- common_skills/ path traversal prevention
- _validate_skill_format with optional frontmatter fields
- English template directory structure (no flat files)
- Japanese skill-creator template content
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import core.execution._sanitize  # noqa: F401
from core.tooling.handler_base import _validate_skill_format


# ── Helpers ──────────────────────────────────────────────────


def _build_handler(tmp_path: Path):
    """Build a minimal MemoryToolsMixin-like object for write testing."""
    from core.memory import MemoryManager
    from core.messenger import Messenger
    from core.tooling.handler import ToolHandler

    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "inbox").mkdir(exist_ok=True)
    (shared_dir / "channels").mkdir(exist_ok=True)

    anima_dir = tmp_path / "animas" / "test_anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text("# test_anima\n", encoding="utf-8")
    (anima_dir / "activity_log").mkdir(exist_ok=True)
    (anima_dir / "skills").mkdir(exist_ok=True)

    memory = MagicMock(spec=MemoryManager)
    memory.read_permissions.return_value = ""
    messenger = Messenger(shared_dir, "test_anima")

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
    )
    return handler, anima_dir


# ── common_skills/ write redirect ────────────────────────────


class TestCommonSkillsWriteRedirect:
    """Test that write_memory_file("common_skills/...") redirects properly."""

    def test_write_to_common_skills_creates_file(self, tmp_path: Path) -> None:
        cs_dir = tmp_path / "common_skills_shared"
        cs_dir.mkdir()

        handler, _ = _build_handler(tmp_path)
        content = "---\nname: test-skill\ndescription: >-\n  test 「test」\n---\n\n# test\n\n## Procedure\n\n1. step\n"

        with patch("core.paths.get_common_skills_dir", return_value=cs_dir):
            result = handler.handle("write_memory_file", {
                "path": "common_skills/test-skill/SKILL.md",
                "content": content,
            })

        target = cs_dir / "test-skill" / "SKILL.md"
        assert target.exists(), f"Expected {target} to be created"
        assert target.read_text(encoding="utf-8") == content

    def test_write_path_traversal_blocked(self, tmp_path: Path) -> None:
        cs_dir = tmp_path / "common_skills_shared"
        cs_dir.mkdir()

        handler, _ = _build_handler(tmp_path)

        with patch("core.paths.get_common_skills_dir", return_value=cs_dir):
            result = handler.handle("write_memory_file", {
                "path": "common_skills/../../etc/passwd",
                "content": "malicious",
            })

        assert "PermissionDenied" in result or "denied" in result.lower()
        assert not (cs_dir / ".." / ".." / "etc" / "passwd").exists()

    def test_write_common_skills_flat_file(self, tmp_path: Path) -> None:
        """Flat format write still works (goes to correct shared dir)."""
        cs_dir = tmp_path / "common_skills_shared"
        cs_dir.mkdir()

        handler, _ = _build_handler(tmp_path)
        content = "---\nname: flat\ndescription: >-\n  flat 「flat」\n---\n\n# flat\n"

        with patch("core.paths.get_common_skills_dir", return_value=cs_dir):
            result = handler.handle("write_memory_file", {
                "path": "common_skills/flat.md",
                "content": content,
            })

        assert (cs_dir / "flat.md").exists()


# ── _validate_skill_format ───────────────────────────────────


class TestValidateSkillFormat:
    """Test skill format validation accepts optional fields."""

    def test_name_and_description_only(self) -> None:
        content = '---\nname: test\ndescription: "test 「test」"\n---\n\n# test\n'
        result = _validate_skill_format(content)
        assert result == ""

    def test_allowed_tools_field_accepted(self) -> None:
        content = (
            "---\n"
            "name: test\n"
            'description: "test 「test」"\n'
            "allowed_tools:\n"
            "  - web_search\n"
            "  - read_file\n"
            "---\n\n# test\n"
        )
        result = _validate_skill_format(content)
        assert result == ""

    def test_tags_field_accepted(self) -> None:
        content = (
            "---\n"
            "name: test\n"
            'description: "test 「test」"\n'
            "tags: [search, web]\n"
            "---\n\n# test\n"
        )
        result = _validate_skill_format(content)
        assert result == ""

    def test_all_optional_fields_accepted(self) -> None:
        content = (
            "---\n"
            "name: test\n"
            'description: "test 「test」"\n'
            "allowed_tools:\n"
            "  - web_search\n"
            "tags: [search]\n"
            "---\n\n# test\n"
        )
        result = _validate_skill_format(content)
        assert result == ""

    def test_missing_name_still_errors(self) -> None:
        content = '---\ndescription: "test 「test」"\n---\n\n# test\n'
        result = _validate_skill_format(content)
        assert result != ""

    def test_missing_description_still_errors(self) -> None:
        content = "---\nname: test\n---\n\n# test\n"
        result = _validate_skill_format(content)
        assert result != ""

    def test_legacy_section_warns(self) -> None:
        content = '---\nname: test\ndescription: "test 「test」"\n---\n\n# test\n\n## 概要\n\nfoo\n'
        result = _validate_skill_format(content)
        assert result != ""


# ── Template structure (E2E) ─────────────────────────────────


_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestEnglishTemplateDirectoryStructure:
    """Verify all English common_skills use {name}/SKILL.md format."""

    def test_no_flat_md_files(self) -> None:
        en_skills = _REPO_ROOT / "templates" / "en" / "common_skills"
        if not en_skills.exists():
            pytest.skip("English templates not present")

        flat_files = [
            f.name for f in en_skills.iterdir()
            if f.is_file() and f.suffix == ".md"
        ]
        assert flat_files == [], (
            f"Flat .md files found in templates/en/common_skills/: {flat_files}. "
            "All skills should be in {name}/SKILL.md format."
        )

    def test_all_directories_have_skill_md(self) -> None:
        en_skills = _REPO_ROOT / "templates" / "en" / "common_skills"
        if not en_skills.exists():
            pytest.skip("English templates not present")

        missing = []
        for d in sorted(en_skills.iterdir()):
            if d.is_dir():
                if not (d / "SKILL.md").exists():
                    missing.append(d.name)
        assert missing == [], (
            f"Directories missing SKILL.md: {missing}"
        )

    def test_converted_skills_have_name_field(self) -> None:
        """Verify converted skills have matching name in frontmatter."""
        import yaml

        en_skills = _REPO_ROOT / "templates" / "en" / "common_skills"
        if not en_skills.exists():
            pytest.skip("English templates not present")

        converted = [
            "skill-creator", "image-posting", "animaworks-guide",
            "subordinate-management", "tool-creator", "subagent-cli",
            "cron-management",
        ]
        for name in converted:
            skill_path = en_skills / name / "SKILL.md"
            assert skill_path.exists(), f"{name}/SKILL.md not found"
            content = skill_path.read_text(encoding="utf-8")
            assert content.startswith("---"), f"{name}/SKILL.md missing frontmatter"
            end = content.find("---", 3)
            fm = yaml.safe_load(content[3:end])
            assert fm.get("name") == name, (
                f"{name}/SKILL.md frontmatter name is '{fm.get('name')}', expected '{name}'"
            )


class TestJapaneseSkillCreatorContent:
    """Verify Japanese skill-creator template content matches spec."""

    def test_create_skill_recommended(self) -> None:
        path = _REPO_ROOT / "templates" / "ja" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "create_skill" in content, "create_skill should be recommended"
        assert 'create_skill(skill_name="{name}"' in content

    def test_write_memory_file_deprecated(self) -> None:
        path = _REPO_ROOT / "templates" / "ja" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "write_memory_file" in content, "write_memory_file should still be mentioned"
        assert "非推奨" in content or "参照できない" in content

    def test_current_optional_metadata_mentioned(self) -> None:
        path = _REPO_ROOT / "templates" / "ja" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "allowed_tools" in content
        assert "skill_policy" in content
        assert "trigger_phrases" in content
        assert "source_origin" in content

    def test_no_exclusive_field_restriction(self) -> None:
        path = _REPO_ROOT / "templates" / "ja" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "以外のフィールドを入れない" not in content

    def test_skill_template_no_legacy_section(self) -> None:
        path = _REPO_ROOT / "templates" / "ja" / "common_skills" / "skill-creator" / "templates" / "skill_template.md"
        content = path.read_text(encoding="utf-8")
        assert "## 概要" not in content

    def test_create_skill_in_checklist(self) -> None:
        path = _REPO_ROOT / "templates" / "ja" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "create_skill" in content
        lines = content.splitlines()
        checklist_lines = [l for l in lines if l.strip().startswith("- [ ]")]
        create_skill_items = [l for l in checklist_lines if "create_skill" in l]
        assert len(create_skill_items) >= 1


class TestEnglishSkillCreatorContent:
    """Verify English skill-creator has create_skill references."""

    def test_create_skill_recommended(self) -> None:
        path = _REPO_ROOT / "templates" / "en" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "create_skill" in content
        assert 'create_skill(skill_name="{name}"' in content

    def test_no_exclusive_field_restriction(self) -> None:
        path = _REPO_ROOT / "templates" / "en" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "Do not add fields other than" not in content

    def test_allowed_tools_mentioned(self) -> None:
        path = _REPO_ROOT / "templates" / "en" / "common_skills" / "skill-creator" / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert "allowed_tools" in content
