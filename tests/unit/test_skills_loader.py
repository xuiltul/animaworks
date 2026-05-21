from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.skills.loader — SKILL.md parsing."""

from pathlib import Path

import pytest

from core.skills.loader import load_skill_body, load_skill_document, load_skill_metadata
from core.skills.models import SkillMetadata, SkillTrustLevel


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    """Create a skill directory with a fully-featured SKILL.md."""
    d = tmp_path / "github-pr-review"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\n"
        "name: github-pr-review\n"
        "description: Review GitHub pull requests\n"
        "category: software-development\n"
        "trust_level: official\n"
        "source:\n"
        "  type: anima\n"
        "  owner_anima: engineer\n"
        "  origin: manual\n"
        "version: 2\n"
        "allowed_tools:\n"
        "  - github\n"
        "  - read_file\n"
        "---\n\n"
        "# PR Review\n\nReview pull requests automatically.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def legacy_skill(tmp_path: Path) -> Path:
    """Create a legacy skill file with minimal frontmatter."""
    d = tmp_path / "old-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: old-skill\ndescription: Legacy skill\n---\n\n# Old Skill\n\nDoes things.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def no_frontmatter_skill(tmp_path: Path) -> Path:
    """Create a skill file with no YAML frontmatter."""
    d = tmp_path / "bare-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "# Bare Skill\n\n## 概要\n\nA skill with no frontmatter.\n\n## Usage\n\nDo stuff.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def procedure_file(tmp_path: Path) -> Path:
    """Create a procedure .md file (flat, not in a SKILL.md directory)."""
    f = tmp_path / "deploy-checklist.md"
    f.write_text(
        "---\n"
        "description: Pre-deploy checklist\n"
        "confidence: 0.8\n"
        "---\n\n"
        "# Deploy Checklist\n\n1. Run tests\n2. Build\n3. Deploy\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture()
def hermes_skill(tmp_path: Path) -> Path:
    """Create a Hermes-style skill with many extra fields."""
    d = tmp_path / "hermes-format"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\n"
        "name: hermes-format\n"
        "description: Hermes-compatible skill\n"
        "trust_level: community\n"
        "source:\n"
        "  type: hub\n"
        "  identifier: hermes-hub/skill-123\n"
        "version: 5\n"
        "usage_count: 100\n"
        "success_count: 95\n"
        "failure_count: 5\n"
        "patch_count: 3\n"
        "pinned: true\n"
        "protected: true\n"
        "security:\n"
        "  verdict: safe\n"
        "  scan_status: completed\n"
        "  findings: []\n"
        "hermes_custom_field: should_be_ignored\n"
        "another_custom: true\n"
        "---\n\n"
        "# Hermes Format Skill\n\nBody content.\n",
        encoding="utf-8",
    )
    return d


class TestLoadSkillMetadata:
    def test_full_frontmatter(self, skill_dir: Path):
        meta = load_skill_metadata(skill_dir / "SKILL.md")
        assert meta.name == "github-pr-review"
        assert meta.description == "Review GitHub pull requests"
        assert meta.category == "software-development"
        assert meta.trust_level == SkillTrustLevel.official
        assert meta.source.type == "anima"
        assert meta.source.owner_anima == "engineer"
        assert meta.version == 2
        assert meta.allowed_tools == ["github", "read_file"]
        assert meta.path == skill_dir / "SKILL.md"

    def test_legacy_frontmatter(self, legacy_skill: Path):
        meta = load_skill_metadata(legacy_skill / "SKILL.md")
        assert meta.name == "old-skill"
        assert meta.description == "Legacy skill"
        assert meta.trust_level == SkillTrustLevel.trusted
        assert meta.version == 1

    def test_legacy_string_source_is_normalized(self, tmp_path: Path):
        d = tmp_path / "legacy-source-skill"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\n"
            "name: legacy-source-skill\n"
            "description: Legacy scalar source field\n"
            "source: activity_log\n"
            "---\n\n"
            "# Legacy Source Skill\n",
            encoding="utf-8",
        )
        meta = load_skill_metadata(d / "SKILL.md")
        assert meta.source.type == "activity_log"
        assert meta.source.identifier is None

    def test_no_frontmatter_infers_name_from_directory(self, no_frontmatter_skill: Path):
        meta = load_skill_metadata(no_frontmatter_skill / "SKILL.md")
        assert meta.name == "bare-skill"
        assert meta.description == "A skill with no frontmatter."
        assert meta.trust_level == SkillTrustLevel.trusted

    def test_procedure_file(self, procedure_file: Path):
        meta = load_skill_metadata(procedure_file)
        assert meta.name == "deploy-checklist"
        assert meta.description == "Pre-deploy checklist"

    def test_hermes_extra_fields_ignored(self, hermes_skill: Path):
        meta = load_skill_metadata(hermes_skill / "SKILL.md")
        assert meta.name == "hermes-format"
        assert meta.trust_level == SkillTrustLevel.community
        assert meta.source.type == "hub"
        assert meta.source.identifier == "hermes-hub/skill-123"
        assert meta.version == 5
        assert meta.usage_count == 100
        assert meta.pinned is True
        assert meta.protected is True
        assert meta.security.verdict.value == "safe"

    def test_invalid_yaml_uses_defaults(self, tmp_path: Path):
        d = tmp_path / "bad-yaml"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: [invalid yaml structure\n---\n\n# Bad YAML\n\n## 概要\n\nFallback description.\n",
            encoding="utf-8",
        )
        meta = load_skill_metadata(d / "SKILL.md")
        assert meta.name == "bad-yaml"
        assert meta.description == "Fallback description."


class TestLoadSkillBody:
    def test_returns_body_only(self, skill_dir: Path):
        body = load_skill_body(skill_dir / "SKILL.md")
        assert "# PR Review" in body
        assert "---" not in body
        assert "trust_level" not in body

    def test_no_frontmatter(self, no_frontmatter_skill: Path):
        body = load_skill_body(no_frontmatter_skill / "SKILL.md")
        assert "# Bare Skill" in body


class TestLoadSkillDocument:
    def test_returns_both(self, skill_dir: Path):
        meta, body = load_skill_document(skill_dir / "SKILL.md")
        assert isinstance(meta, SkillMetadata)
        assert meta.name == "github-pr-review"
        assert "# PR Review" in body
