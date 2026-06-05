from __future__ import annotations

"""Tests for skill-index integration gaps (Issue 20260508_01).

Covers:
- Gap 1: builder.py catalog uses SkillIndex, shows trust_level tags
- Gap 3: read_memory_file blocks loading of blocked skills
- Gap 4: common_skills nested directory support
- Gap 2: usage_count policy (view_count + use_count)
"""

from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import _format_trust_tag, build_system_prompt
from core.skills.index import SkillIndex
from core.skills.models import (
    SkillMetadata,
    SkillScanVerdict,
    SkillSecurityScan,
    SkillTrustLevel,
)

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture()
def skill_dirs(tmp_path):
    """Create skills, common_skills, and procedures directories with test files."""
    skills_dir = tmp_path / "skills"
    common_skills_dir = tmp_path / "common_skills"
    procedures_dir = tmp_path / "procedures"
    skills_dir.mkdir()
    common_skills_dir.mkdir()
    procedures_dir.mkdir()

    # Personal skill: trusted (default)
    (skills_dir / "coding").mkdir()
    (skills_dir / "coding" / "SKILL.md").write_text(
        "---\nname: coding\ndescription: Write code\n---\n# Coding\n",
        encoding="utf-8",
    )

    # Personal skill: builtin trust level
    (skills_dir / "search").mkdir()
    (skills_dir / "search" / "SKILL.md").write_text(
        "---\nname: search\ndescription: Search the web\ntrust_level: builtin\n---\n# Search\n",
        encoding="utf-8",
    )

    # Personal skill: blocked
    (skills_dir / "danger").mkdir()
    (skills_dir / "danger" / "SKILL.md").write_text(
        "---\nname: danger\ndescription: Dangerous skill\ntrust_level: blocked\n---\n# Danger\n",
        encoding="utf-8",
    )

    # Common skill: flat
    (common_skills_dir / "deploy").mkdir()
    (common_skills_dir / "deploy" / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: Deploy apps\n---\n# Deploy\n",
        encoding="utf-8",
    )

    # Common skill: nested (builtin/analytics)
    (common_skills_dir / "builtin").mkdir()
    (common_skills_dir / "builtin" / "analytics").mkdir()
    (common_skills_dir / "builtin" / "analytics" / "SKILL.md").write_text(
        "---\nname: analytics\ndescription: Data analytics\ntrust_level: builtin\n---\n# Analytics\n",
        encoding="utf-8",
    )

    # Common skill: quarantine (should be excluded)
    (common_skills_dir / "suspect").mkdir()
    (common_skills_dir / "suspect" / "SKILL.md").write_text(
        "---\nname: suspect\ndescription: Suspect skill\ntrust_level: quarantine\n---\n# Suspect\n",
        encoding="utf-8",
    )

    # Procedure
    (procedures_dir / "deploy-steps.md").write_text(
        "---\nname: deploy-steps\ndescription: Step-by-step deploy\n---\n# Deploy Steps\n",
        encoding="utf-8",
    )

    return skills_dir, common_skills_dir, procedures_dir


# ── Gap 1: SkillIndex in builder catalog ──────────────────


class TestCatalogUsesSkillIndex:
    """Verify builder catalog uses SkillIndex with trust_level filtering."""

    def test_trust_tag_trusted_is_empty(self):
        meta = SkillMetadata(name="x", trust_level=SkillTrustLevel.trusted)
        assert _format_trust_tag(meta) == ""

    def test_trust_tag_builtin_shown(self):
        meta = SkillMetadata(name="x", trust_level=SkillTrustLevel.builtin)
        assert _format_trust_tag(meta) == " [builtin]"

    def test_trust_tag_community_shown(self):
        meta = SkillMetadata(name="x", trust_level=SkillTrustLevel.community)
        assert _format_trust_tag(meta) == " [community]"

    def test_trust_tag_none_returns_empty(self):
        """Legacy objects without trust_level field return empty tag."""
        obj = MagicMock(spec=[])
        assert _format_trust_tag(obj) == ""

    def test_blocked_skills_excluded_from_catalog(self, skill_dirs):
        skills_dir, common_skills_dir, procedures_dir = skill_dirs
        idx = SkillIndex(skills_dir, common_skills_dir, procedures_dir)
        names = [m.name for m in idx.all_skills]
        assert "danger" not in names
        assert "suspect" not in names
        assert "coding" in names
        assert "search" in names

    def test_catalog_contains_trust_tags(self, skill_dirs, tmp_path):
        """Full builder integration: catalog shows [builtin] tag."""
        skills_dir, common_skills_dir, procedures_dir = skill_dirs
        anima_dir = tmp_path / "anima_dir"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("I am Test", encoding="utf-8")
        (anima_dir / "skills").symlink_to(skills_dir)
        (anima_dir / "procedures").symlink_to(procedures_dir)

        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = ""
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.list_procedure_metas.return_value = []
        memory.list_shared_users.return_value = []

        with patch("core.paths.get_common_skills_dir", return_value=common_skills_dir):
            result = build_system_prompt(memory)
            prompt = result.system_prompt

        assert "skills/search/SKILL.md [builtin]" in prompt
        assert "skills/coding/SKILL.md:" in prompt
        # blocked/quarantine excluded
        assert "danger" not in prompt
        assert "suspect" not in prompt


# ── Gap 3: read_memory_file blocks loading of blocked skills ──


class TestReadMemoryFileBlocksBlockedSkills:
    """Verify _handle_read_memory_file rejects blocked skills."""

    def test_blocked_skill_returns_error(self, tmp_path):
        from core.tooling.handler_memory import MemoryToolsMixin

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        skill_dir = anima_dir / "skills" / "evil"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: evil\ndescription: Evil skill\ntrust_level: blocked\n---\n# Evil\n",
            encoding="utf-8",
        )

        mixin = MagicMock(spec=MemoryToolsMixin)
        mixin._anima_dir = anima_dir
        mixin._superuser = False
        mixin._subordinate_activity_dirs = []
        mixin._subordinate_management_files = []
        mixin._descendant_activity_dirs = []
        mixin._descendant_state_files = []
        mixin._descendant_state_dirs = []
        mixin._read_paths = set()
        mixin._is_skill_path = MemoryToolsMixin._is_skill_path
        mixin._record_skill_view_if_applicable = MagicMock()

        result = MemoryToolsMixin._handle_read_memory_file(mixin, {"path": "skills/evil/SKILL.md"})
        assert "SkillBlocked" in result
        assert "evil" in result

    def test_trusted_skill_loads_normally(self, tmp_path):
        from core.tooling.handler_memory import MemoryToolsMixin

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        skill_dir = anima_dir / "skills" / "good"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: good\ndescription: Good skill\ntrust_level: trusted\n---\n# Good Skill Body\n",
            encoding="utf-8",
        )

        mixin = MagicMock(spec=MemoryToolsMixin)
        mixin._anima_dir = anima_dir
        mixin._superuser = False
        mixin._subordinate_activity_dirs = []
        mixin._subordinate_management_files = []
        mixin._descendant_activity_dirs = []
        mixin._descendant_state_files = []
        mixin._descendant_state_dirs = []
        mixin._read_paths = set()
        mixin._is_skill_path = MemoryToolsMixin._is_skill_path
        mixin._record_skill_view_if_applicable = MagicMock()

        result = MemoryToolsMixin._handle_read_memory_file(mixin, {"path": "skills/good/SKILL.md"})
        assert "# Good Skill Body" in result
        assert "SkillBlocked" not in result

    def test_dangerous_scan_verdict_blocks_loading(self, tmp_path):
        from core.tooling.handler_memory import MemoryToolsMixin

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        skill_dir = anima_dir / "skills" / "risky"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: risky\ndescription: Risky\ntrust_level: trusted\n---\n# Risky\n",
            encoding="utf-8",
        )

        mixin = MagicMock(spec=MemoryToolsMixin)
        mixin._anima_dir = anima_dir
        mixin._superuser = False
        mixin._subordinate_activity_dirs = []
        mixin._subordinate_management_files = []
        mixin._descendant_activity_dirs = []
        mixin._descendant_state_files = []
        mixin._descendant_state_dirs = []
        mixin._read_paths = set()
        mixin._is_skill_path = MemoryToolsMixin._is_skill_path
        mixin._record_skill_view_if_applicable = MagicMock()

        # Override the metadata to have dangerous verdict
        dangerous_meta = SkillMetadata(
            name="risky",
            trust_level=SkillTrustLevel.trusted,
            security=SkillSecurityScan(verdict=SkillScanVerdict.dangerous),
        )
        with patch("core.skills.loader.load_skill_metadata", return_value=dangerous_meta):
            result = MemoryToolsMixin._handle_read_memory_file(mixin, {"path": "skills/risky/SKILL.md"})
        assert "SkillBlocked" in result


# ── Gap 4: Nested common_skills directory support ─────────


class TestNestedCommonSkills:
    """Verify nested common_skills subdirectories are discovered."""

    def test_skill_index_finds_nested(self, skill_dirs):
        skills_dir, common_skills_dir, procedures_dir = skill_dirs
        idx = SkillIndex(skills_dir, common_skills_dir, procedures_dir)
        names = [m.name for m in idx.all_skills if m.is_common]
        assert "analytics" in names
        assert "deploy" in names

    def test_skill_metadata_service_finds_nested(self, tmp_path):
        from core.memory.skill_metadata import SkillMetadataService

        common_dir = tmp_path / "common_skills"
        common_dir.mkdir()
        (common_dir / "flat").mkdir()
        (common_dir / "flat" / "SKILL.md").write_text(
            "---\nname: flat\ndescription: Flat skill\n---\n",
            encoding="utf-8",
        )
        (common_dir / "cat" / "nested").mkdir(parents=True)
        (common_dir / "cat" / "nested" / "SKILL.md").write_text(
            "---\nname: nested\ndescription: Nested skill\n---\n",
            encoding="utf-8",
        )

        svc = SkillMetadataService(tmp_path / "skills", common_dir)
        metas = svc.list_common_skill_metas()
        names = [m.name for m in metas]
        assert "flat" in names
        assert "nested" in names


# ── Gap 2: Usage count policy ─────────────────────────────


class TestUsageCountPolicy:
    """Verify usage_count = view_count + use_count in SkillIndex."""

    def test_usage_count_merges_view_and_use(self, tmp_path):
        from core.skills.models import SkillUsageEventType
        from core.skills.usage import SkillUsageTracker

        skills_dir = tmp_path / "skills" / "myskill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: myskill\ndescription: My skill\n---\n",
            encoding="utf-8",
        )

        # Record usage events (view is debounced per-session, so use multiple instances)
        tracker1 = SkillUsageTracker(tmp_path)
        tracker1.record("myskill", SkillUsageEventType.view)
        tracker2 = SkillUsageTracker(tmp_path)
        tracker2.record("myskill", SkillUsageEventType.view)
        tracker3 = SkillUsageTracker(tmp_path)
        tracker3.record("myskill", SkillUsageEventType.view)
        tracker3.record("myskill", SkillUsageEventType.success)

        idx = SkillIndex(
            skills_dir=tmp_path / "skills",
            common_skills_dir=tmp_path / "common_skills_empty",
            anima_dir=tmp_path,
        )
        # common_skills_empty doesn't exist but that's OK
        skills = idx.build_index()
        assert len(skills) == 1
        meta = skills[0]
        # usage_count = view_count(3) + use_count(0) = 3
        assert meta.usage_count == 3
        assert meta.success_count == 1


# ── _is_skill_path helper ─────────────────────────────────


class TestIsSkillPath:
    """Verify _is_skill_path detects skill/procedure paths correctly."""

    def test_personal_skill(self):
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._is_skill_path("skills/coding/SKILL.md") is True

    def test_common_skill(self):
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._is_skill_path("common_skills/deploy/SKILL.md") is True

    def test_procedure(self):
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._is_skill_path("procedures/backup.md") is True

    def test_non_skill(self):
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._is_skill_path("knowledge/topic.md") is False
        assert MemoryToolsMixin._is_skill_path("state/current_state.md") is False
