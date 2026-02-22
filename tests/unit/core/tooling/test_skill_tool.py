from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for skill tool — Progressive Disclosure skill loading.

Covers:
- apply_builtins: placeholder replacement
- build_skill_tool_description: dynamic description generation with budget
- _resolve_skill_path: skill name resolution (personal > common > procedure)
- _strip_frontmatter: YAML frontmatter removal
- _resolve_builtins: builtin variable resolution
- load_and_render_skill: full skill loading and rendering pipeline
"""

from pathlib import Path

import pytest

from core.schemas import SkillMeta
from core.tooling.skill_tool import (
    _resolve_builtins,
    _resolve_skill_path,
    _strip_frontmatter,
    apply_builtins,
    build_skill_tool_description,
    load_and_render_skill,
)


# ── Helpers ──────────────────────────────────────────────


def _make_skill_file(
    directory: Path,
    name: str,
    *,
    content: str = "# Skill\nDo something.",
    frontmatter: str = "",
) -> Path:
    """Create a .md skill file in the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    text = f"---\n{frontmatter}---\n\n{content}" if frontmatter else content
    path = directory / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def _make_skill_meta(
    name: str,
    description: str = "",
    *,
    is_common: bool = False,
    path: Path | None = None,
) -> SkillMeta:
    """Create a SkillMeta instance for testing."""
    return SkillMeta(
        name=name,
        description=description or f"{name} description",
        path=path or Path(f"/dummy/{name}.md"),
        is_common=is_common,
    )


# ── apply_builtins ───────────────────────────────────────


class TestApplyBuiltins:
    """Test apply_builtins() placeholder replacement."""

    def test_single_placeholder_replaced(self):
        content = "Hello, {{anima_name}}!"
        result = apply_builtins(content, {"anima_name": "alice"})
        assert result == "Hello, alice!"

    def test_no_placeholder_returns_unchanged(self):
        content = "No placeholders here."
        result = apply_builtins(content, {"anima_name": "alice"})
        assert result == "No placeholders here."

    def test_multiple_placeholders_replaced(self):
        content = "Name: {{anima_name}}, Dir: {{anima_dir}}, Time: {{now_jst}}"
        builtins = {
            "anima_name": "bob",
            "anima_dir": "/data/animas/bob",
            "now_jst": "2026-02-22T10:00:00+09:00",
        }
        result = apply_builtins(content, builtins)
        assert result == (
            "Name: bob, Dir: /data/animas/bob, "
            "Time: 2026-02-22T10:00:00+09:00"
        )

    def test_repeated_placeholder_replaced_all_occurrences(self):
        content = "{{anima_name}} says hello, {{anima_name}}."
        result = apply_builtins(content, {"anima_name": "carol"})
        assert result == "carol says hello, carol."

    def test_empty_builtins_returns_unchanged(self):
        content = "Nothing to replace {{unknown}}."
        result = apply_builtins(content, {})
        assert result == "Nothing to replace {{unknown}}."


# ── build_skill_tool_description ─────────────────────────


class TestBuildSkillToolDescription:
    """Test build_skill_tool_description() dynamic description generation."""

    def test_skill_metas_included_in_available_skills_block(self):
        metas = [_make_skill_meta("deploy", "デプロイ手順")]
        result = build_skill_tool_description(metas, [], [])
        assert "<available_skills>" in result
        assert "</available_skills>" in result
        assert "- deploy: デプロイ手順" in result

    def test_all_types_included(self):
        personal = [_make_skill_meta("coding", "コーディング規約")]
        common = [_make_skill_meta("review", "レビュー手順", is_common=True)]
        procedures = [_make_skill_meta("onboarding", "入社手続き")]
        result = build_skill_tool_description(personal, common, procedures)
        assert "- coding: コーディング規約" in result
        assert "- review (共通): レビュー手順" in result
        assert "- onboarding (手順): 入社手続き" in result

    def test_budget_exceeded_truncation(self):
        """When entries exceed 8000 chars, truncation marker appears."""
        # Create enough metas to exceed the 8000-char budget.
        # Each entry is roughly "- skill_NNN: description_for_skill_NNN" ~40 chars.
        # Header lines consume ~150 chars, so we need ~200 entries to exceed.
        metas = [
            _make_skill_meta(
                f"skill_{i:03d}",
                f"A reasonably long description for skill number {i:03d} to fill budget",
            )
            for i in range(300)
        ]
        result = build_skill_tool_description(metas, [], [])
        assert "(以降省略)" in result
        # The closing tag should still be present
        assert "</available_skills>" in result

    def test_budget_exceeded_in_common_skills(self):
        """Budget overflow in common skills section also triggers truncation."""
        personal = [
            _make_skill_meta(f"p{i}", f"description {'x' * 80}")
            for i in range(100)
        ]
        common = [
            _make_skill_meta(f"c{i}", f"common description {'y' * 80}", is_common=True)
            for i in range(100)
        ]
        result = build_skill_tool_description(personal, common, [])
        assert "(以降省略)" in result

    def test_empty_meta_lists_generate_minimal_description(self):
        result = build_skill_tool_description([], [], [])
        assert "<available_skills>" in result
        assert "</available_skills>" in result
        # Should still have the base instructional text
        assert "スキル" in result


# ── _resolve_skill_path ──────────────────────────────────


class TestResolveSkillPath:
    """Test _resolve_skill_path() name resolution priority."""

    def test_personal_skill_highest_priority(self, tmp_path: Path):
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        _make_skill_file(skills, "deploy")
        _make_skill_file(common, "deploy")
        _make_skill_file(procedures, "deploy")

        path, skill_type = _resolve_skill_path("deploy", skills, common, procedures)
        assert path == skills / "deploy.md"
        assert skill_type == "個人"

    def test_fallback_to_common_skill(self, tmp_path: Path):
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        skills.mkdir(parents=True)
        _make_skill_file(common, "review")

        path, skill_type = _resolve_skill_path("review", skills, common, procedures)
        assert path == common / "review.md"
        assert skill_type == "共通"

    def test_fallback_to_procedure(self, tmp_path: Path):
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        skills.mkdir(parents=True)
        common.mkdir(parents=True)
        _make_skill_file(procedures, "onboarding")

        path, skill_type = _resolve_skill_path("onboarding", skills, common, procedures)
        assert path == procedures / "onboarding.md"
        assert skill_type == "手順"

    def test_not_found_returns_none(self, tmp_path: Path):
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        skills.mkdir(parents=True)
        common.mkdir(parents=True)
        procedures.mkdir(parents=True)

        path, skill_type = _resolve_skill_path("nonexistent", skills, common, procedures)
        assert path is None
        assert skill_type == ""

    def test_personal_overrides_common_same_name(self, tmp_path: Path):
        """When both personal and common have the same file, personal wins."""
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        _make_skill_file(skills, "shared_skill", content="Personal version")
        _make_skill_file(common, "shared_skill", content="Common version")
        procedures.mkdir(parents=True)

        path, skill_type = _resolve_skill_path(
            "shared_skill", skills, common, procedures
        )
        assert path == skills / "shared_skill.md"
        assert skill_type == "個人"
        assert "Personal version" in path.read_text(encoding="utf-8")

    def test_nonexistent_directories_return_none(self, tmp_path: Path):
        """Directories that don't exist should not cause errors."""
        skills = tmp_path / "nonexistent_skills"
        common = tmp_path / "nonexistent_common"
        procedures = tmp_path / "nonexistent_procedures"
        # Directories intentionally NOT created

        path, skill_type = _resolve_skill_path("anything", skills, common, procedures)
        assert path is None
        assert skill_type == ""

    # ── Path traversal guard ────────────────────────────

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "../../etc/passwd",
            "../secret",
            "foo/bar",
            "foo\\bar",
            "..\\windows\\system32",
            "skills/../../../etc/shadow",
        ],
    )
    def test_path_traversal_rejected(self, tmp_path: Path, malicious_name: str):
        """Names containing /, \\, or .. must be rejected."""
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        skills.mkdir(parents=True)
        common.mkdir(parents=True)
        procedures.mkdir(parents=True)

        path, skill_type = _resolve_skill_path(
            malicious_name, skills, common, procedures
        )
        assert path is None
        assert skill_type == ""

    def test_valid_name_with_hyphen_and_underscore(self, tmp_path: Path):
        """Names with hyphens and underscores are valid."""
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        procedures = tmp_path / "procedures"
        _make_skill_file(skills, "my-skill_v2")
        common.mkdir(parents=True)
        procedures.mkdir(parents=True)

        path, skill_type = _resolve_skill_path(
            "my-skill_v2", skills, common, procedures
        )
        assert path == skills / "my-skill_v2.md"
        assert skill_type == "個人"


# ── _strip_frontmatter ───────────────────────────────────


class TestStripFrontmatter:
    """Test _strip_frontmatter() YAML frontmatter removal."""

    def test_frontmatter_removed(self):
        text = "---\ndescription: Deploy procedure\ntags: [deploy]\n---\n\n# Deploy\nStep 1."
        content, fm = _strip_frontmatter(text)
        assert content == "# Deploy\nStep 1."
        assert fm["description"] == "Deploy procedure"
        assert fm["tags"] == ["deploy"]

    def test_no_frontmatter_returns_original(self):
        text = "# Simple Skill\nJust content."
        content, fm = _strip_frontmatter(text)
        assert content == "# Simple Skill\nJust content."
        assert fm == {}

    def test_invalid_yaml_returns_empty_dict(self):
        text = "---\n: invalid: yaml: [broken\n---\n\n# Content"
        content, fm = _strip_frontmatter(text)
        assert fm == {}
        assert "# Content" in content

    def test_frontmatter_with_allowed_tools(self):
        text = (
            "---\n"
            "description: Test skill\n"
            "allowed_tools:\n"
            "  - web_search\n"
            "  - read_memory_file\n"
            "---\n\n"
            "# Test\nContent here."
        )
        content, fm = _strip_frontmatter(text)
        assert content == "# Test\nContent here."
        assert fm["allowed_tools"] == ["web_search", "read_memory_file"]

    def test_single_separator_returns_original(self):
        """Text with only one '---' (no closing) returns as-is."""
        text = "---\nsome text without closing separator"
        content, fm = _strip_frontmatter(text)
        assert content == text
        assert fm == {}

    def test_empty_frontmatter(self):
        text = "---\n---\n\n# Content"
        content, fm = _strip_frontmatter(text)
        assert "# Content" in content
        assert fm == {}

    def test_frontmatter_with_non_dict_yaml(self):
        """If YAML parses to non-dict (e.g., a list), treat as empty."""
        text = "---\n- item1\n- item2\n---\n\n# Content"
        content, fm = _strip_frontmatter(text)
        assert fm == {}
        assert "# Content" in content


# ── _resolve_builtins ────────────────────────────────────


class TestResolveBuiltins:
    """Test _resolve_builtins() builtin variable resolution."""

    def test_contains_now_jst(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        builtins = _resolve_builtins(anima_dir)
        assert "now_jst" in builtins
        # Should be a valid ISO8601 string with timezone info
        assert "+" in builtins["now_jst"] or "T" in builtins["now_jst"]

    def test_contains_anima_name(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        builtins = _resolve_builtins(anima_dir)
        assert builtins["anima_name"] == "alice"

    def test_contains_anima_dir(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)
        builtins = _resolve_builtins(anima_dir)
        assert builtins["anima_dir"] == str(anima_dir)

    def test_all_expected_keys_present(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        builtins = _resolve_builtins(anima_dir)
        assert set(builtins.keys()) == {"now_jst", "anima_name", "anima_dir"}


# ── load_and_render_skill ────────────────────────────────


class TestLoadAndRenderSkill:
    """Test load_and_render_skill() full pipeline."""

    def _setup_dirs(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        """Create anima_dir + skill directories and return paths."""
        anima_dir = tmp_path / "animas" / "test-anima"
        skills_dir = anima_dir / "skills"
        common_skills_dir = tmp_path / "common_skills"
        procedures_dir = anima_dir / "procedures"
        for d in (anima_dir, skills_dir, common_skills_dir, procedures_dir):
            d.mkdir(parents=True, exist_ok=True)
        return anima_dir, skills_dir, common_skills_dir, procedures_dir

    def test_skill_content_returned_without_frontmatter(self, tmp_path: Path):
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(
            skills_dir,
            "deploy",
            frontmatter="description: Deploy to production\n",
            content="# Deploy\n\nStep 1: Build.\nStep 2: Push.",
        )
        result = load_and_render_skill(
            "deploy", anima_dir, skills_dir, common, procs
        )
        assert "# Deploy" in result
        assert "Step 1: Build." in result
        assert "Step 2: Push." in result
        # Frontmatter should not appear
        assert "description: Deploy to production" not in result

    def test_builtin_placeholders_replaced(self, tmp_path: Path):
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(
            skills_dir,
            "greet",
            content="Hello from {{anima_name}} at {{anima_dir}}.",
        )
        result = load_and_render_skill(
            "greet", anima_dir, skills_dir, common, procs
        )
        assert "test-anima" in result
        assert str(anima_dir) in result
        # Placeholders should be gone
        assert "{{anima_name}}" not in result
        assert "{{anima_dir}}" not in result

    def test_allowed_tools_soft_constraint_included(self, tmp_path: Path):
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(
            skills_dir,
            "search",
            frontmatter=(
                "description: Web search skill\n"
                "allowed_tools:\n"
                "  - web_search\n"
                "  - read_memory_file\n"
            ),
            content="# Search\nSearch the web.",
        )
        result = load_and_render_skill(
            "search", anima_dir, skills_dir, common, procs
        )
        assert "ツール制約" in result
        assert "web_search" in result
        assert "read_memory_file" in result

    def test_no_allowed_tools_no_constraint_section(self, tmp_path: Path):
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(
            skills_dir,
            "simple",
            frontmatter="description: Simple skill\n",
            content="# Simple\nJust do it.",
        )
        result = load_and_render_skill(
            "simple", anima_dir, skills_dir, common, procs
        )
        assert "ツール制約" not in result

    def test_nonexistent_skill_error_with_available_list(self, tmp_path: Path):
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        # Create some skills so the available list is populated
        _make_skill_file(skills_dir, "deploy")
        _make_skill_file(common, "review")
        _make_skill_file(procs, "onboarding")

        result = load_and_render_skill(
            "nonexistent", anima_dir, skills_dir, common, procs
        )
        assert "見つかりません" in result
        assert "nonexistent" in result
        assert "deploy" in result
        assert "review" in result
        assert "onboarding" in result

    def test_nonexistent_skill_empty_dirs(self, tmp_path: Path):
        """Error for nonexistent skill when no skills exist at all."""
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        result = load_and_render_skill(
            "missing", anima_dir, skills_dir, common, procs
        )
        assert "見つかりません" in result
        assert "missing" in result

    def test_context_parameter_appended(self, tmp_path: Path):
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(skills_dir, "deploy", content="# Deploy\nDeploy steps.")
        result = load_and_render_skill(
            "deploy",
            anima_dir,
            skills_dir,
            common,
            procs,
            context="Deploy to staging environment only.",
        )
        assert "コンテキスト" in result
        assert "Deploy to staging environment only." in result

    def test_empty_context_omits_section(self, tmp_path: Path):
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(skills_dir, "deploy", content="# Deploy\nDeploy steps.")
        result = load_and_render_skill(
            "deploy", anima_dir, skills_dir, common, procs, context=""
        )
        assert "コンテキスト" not in result

    def test_context_default_omits_section(self, tmp_path: Path):
        """When context is not provided at all, section is omitted."""
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(skills_dir, "deploy", content="# Deploy\nDeploy steps.")
        result = load_and_render_skill(
            "deploy", anima_dir, skills_dir, common, procs
        )
        assert "コンテキスト" not in result

    def test_now_jst_placeholder_replaced(self, tmp_path: Path):
        """The {{now_jst}} builtin is replaced with a timestamp."""
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(
            skills_dir, "timed", content="Current time: {{now_jst}}"
        )
        result = load_and_render_skill(
            "timed", anima_dir, skills_dir, common, procs
        )
        assert "{{now_jst}}" not in result
        # Should contain an ISO8601-like timestamp with T separator
        assert "T" in result

    def test_common_skill_loaded(self, tmp_path: Path):
        """Skills from common_skills_dir are loaded correctly."""
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(common, "shared_review", content="# Review\nReview steps.")
        result = load_and_render_skill(
            "shared_review", anima_dir, skills_dir, common, procs
        )
        assert "# Review" in result
        assert "Review steps." in result

    def test_procedure_loaded(self, tmp_path: Path):
        """Skills from procedures_dir are loaded correctly."""
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        _make_skill_file(procs, "hiring", content="# Hiring\nHiring process.")
        result = load_and_render_skill(
            "hiring", anima_dir, skills_dir, common, procs
        )
        assert "# Hiring" in result
        assert "Hiring process." in result

    def test_path_traversal_returns_error(self, tmp_path: Path):
        """Path traversal attempts return error, not file contents."""
        anima_dir, skills_dir, common, procs = self._setup_dirs(tmp_path)
        # Create a file that traversal would try to reach
        secret = tmp_path / "secret.md"
        secret.write_text("TOP SECRET", encoding="utf-8")
        _make_skill_file(skills_dir, "legit", content="# Legit skill")

        result = load_and_render_skill(
            "../../secret", anima_dir, skills_dir, common, procs
        )
        assert "見つかりません" in result
        assert "TOP SECRET" not in result
        # Available list should show legit skills
        assert "legit" in result
