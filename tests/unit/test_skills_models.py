from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.skills.models — Pydantic models and enums."""

from datetime import UTC, datetime
from pathlib import Path

from core.skills.models import (
    SkillMetadata,
    SkillScanVerdict,
    SkillSecurityScan,
    SkillSource,
    SkillTrustLevel,
)


class TestSkillTrustLevel:
    def test_values(self):
        assert SkillTrustLevel.builtin == "builtin"
        assert SkillTrustLevel.blocked == "blocked"
        assert SkillTrustLevel.quarantine == "quarantine"

    def test_string_behavior(self):
        assert SkillTrustLevel.trusted.value == "trusted"
        assert str(SkillTrustLevel.community) == "SkillTrustLevel.community"


class TestSkillScanVerdict:
    def test_values(self):
        assert SkillScanVerdict.safe == "safe"
        assert SkillScanVerdict.dangerous == "dangerous"


class TestSkillSource:
    def test_defaults(self):
        src = SkillSource()
        assert src.type == "local"
        assert src.identifier is None
        assert src.owner_anima is None
        assert src.origin is None

    def test_custom(self):
        src = SkillSource(type="anima", owner_anima="sakura", origin="manual")
        assert src.type == "anima"
        assert src.owner_anima == "sakura"


class TestSkillSecurityScan:
    def test_defaults(self):
        scan = SkillSecurityScan()
        assert scan.verdict == SkillScanVerdict.safe
        assert scan.scan_status == "not_scanned"
        assert scan.findings == []
        assert scan.scanned_at is None

    def test_with_findings(self):
        scan = SkillSecurityScan(
            verdict=SkillScanVerdict.warn,
            findings=[{"type": "shell_exec", "line": 42}],
            scanned_at=datetime(2026, 5, 1, tzinfo=UTC),
        )
        assert len(scan.findings) == 1
        assert scan.scanned_at is not None


class TestSkillMetadata:
    def test_minimal(self):
        meta = SkillMetadata(name="test-skill")
        assert meta.name == "test-skill"
        assert meta.description == ""
        assert meta.trust_level == SkillTrustLevel.trusted
        assert meta.skill_policy.use_mode == "primary_guidance"
        assert meta.skill_policy.injection == "body_allowed"
        assert meta.version == 1
        assert meta.is_common is False
        assert meta.is_procedure is False

    def test_probation_policy_construction(self):
        meta = SkillMetadata(
            name="auto-skill",
            trust_level=SkillTrustLevel.community,
            promotion_status="probation",
            skill_policy={"use_mode": "candidate_hint", "injection": "pointer_preferred"},
        )
        assert meta.promotion_status == "probation"
        assert meta.skill_policy.use_mode == "candidate_hint"
        assert meta.skill_policy.injection == "pointer_preferred"

    def test_full_construction(self):
        meta = SkillMetadata(
            name="github-pr-review",
            description="Review GitHub pull requests",
            category="software-development",
            trust_level=SkillTrustLevel.official,
            source=SkillSource(type="anima", owner_anima="engineer"),
            version=3,
            usage_count=42,
            allowed_tools=["github", "read_file"],
            path=Path("/tmp/skills/github-pr-review/SKILL.md"),
            is_common=True,
        )
        assert meta.category == "software-development"
        assert meta.trust_level == SkillTrustLevel.official
        assert meta.version == 3
        assert meta.usage_count == 42
        assert meta.allowed_tools == ["github", "read_file"]

    def test_extra_fields_ignored(self):
        meta = SkillMetadata(
            name="test",
            hermes_custom_field="should_be_ignored",  # type: ignore[call-arg]
            another_unknown=123,  # type: ignore[call-arg]
        )
        assert meta.name == "test"
        assert not hasattr(meta, "hermes_custom_field")

    def test_to_legacy(self):
        meta = SkillMetadata(
            name="web-search",
            description="Search the web",
            path=Path("/tmp/skills/web-search/SKILL.md"),
            is_common=True,
            allowed_tools=["web_search"],
        )
        legacy = meta.to_legacy()
        assert legacy.name == "web-search"
        assert legacy.description == "Search the web"
        assert legacy.path == Path("/tmp/skills/web-search/SKILL.md")
        assert legacy.is_common is True
        assert legacy.allowed_tools == ["web_search"]

    def test_to_legacy_no_path(self):
        meta = SkillMetadata(name="no-path")
        legacy = meta.to_legacy()
        assert legacy.path == Path(".")

    def test_model_copy(self):
        meta = SkillMetadata(name="test", is_common=False)
        updated = meta.model_copy(update={"is_common": True, "is_procedure": True})
        assert updated.is_common is True
        assert updated.is_procedure is True
        assert meta.is_common is False
