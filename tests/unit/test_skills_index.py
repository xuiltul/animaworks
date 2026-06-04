from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.skills.index — SkillIndex scanning and search."""

from pathlib import Path

import pytest

from core.skills.index import SkillIndex
from core.skills.models import SkillUsageEventType
from core.skills.usage import SkillUsageTracker


def _write_skill(base: Path, name: str, *, trust_level: str = "trusted", desc: str = "") -> Path:
    """Helper: create a skill directory with SKILL.md."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\nname: {name}\ndescription: {desc or name + ' skill'}\n"
        f"trust_level: {trust_level}\n---\n\n# {name}\n\nBody.\n"
    )
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def _write_flat_skill(base: Path, name: str, *, trust_level: str = "trusted", desc: str = "") -> Path:
    """Helper: create a legacy direct-child skills/*.md skill."""
    f = base / f"{name}.md"
    content = (
        f"---\nname: {name}\ndescription: {desc or name + ' skill'}\n"
        f"trust_level: {trust_level}\n---\n\n# {name}\n\nBody.\n"
    )
    f.write_text(content, encoding="utf-8")
    return f


def _write_procedure(base: Path, name: str, *, desc: str = "") -> Path:
    """Helper: create a procedure .md file."""
    f = base / f"{name}.md"
    content = f"---\ndescription: {desc or name + ' procedure'}\n---\n\n# {name}\n\nSteps.\n"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture()
def skill_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create personal, common, and procedures directories with sample files."""
    skills = tmp_path / "skills"
    common = tmp_path / "common_skills"
    procs = tmp_path / "procedures"
    skills.mkdir()
    common.mkdir()
    procs.mkdir()

    _write_skill(skills, "web-search", desc="Search the web")
    _write_skill(skills, "code-review", desc="Code review tool")
    _write_skill(common, "machine-tool", trust_level="builtin", desc="Machine tool CLI")
    _write_skill(common, "blocked-skill", trust_level="blocked", desc="This is blocked")
    _write_skill(common, "quarantined", trust_level="quarantine", desc="Quarantined skill")

    # Nested subcategory
    nested = common / "official"
    nested.mkdir()
    _write_skill(nested, "official-tool", trust_level="official", desc="Official tool")

    _write_procedure(procs, "deploy-checklist", desc="Pre-deploy checklist")
    _write_procedure(procs, "incident-response", desc="Incident response steps")

    return skills, common, procs


class TestSkillIndexBuild:
    def test_builds_full_index(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.build_index()

        names = [m.name for m in results]
        assert "web-search" in names
        assert "code-review" in names
        assert "machine-tool" in names
        assert "official-tool" in names
        assert "deploy-checklist" in names
        assert "incident-response" in names

    def test_blocked_and_quarantine_excluded(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        _write_flat_skill(skills, "flat-blocked", trust_level="blocked", desc="Flat blocked")
        _write_flat_skill(skills, "flat-quarantined", trust_level="quarantine", desc="Flat quarantined")
        idx = SkillIndex(skills, common, procs)
        results = idx.build_index()

        names = [m.name for m in results]
        assert "blocked-skill" not in names
        assert "quarantined" not in names
        assert "flat-blocked" not in names
        assert "flat-quarantined" not in names

    def test_legacy_flat_personal_skill_is_indexed(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        flat = _write_flat_skill(skills, "flat-review", desc="Legacy flat review")
        nested_note = skills / "web-search" / "references" / "note.md"
        nested_note.parent.mkdir(parents=True)
        nested_note.write_text("---\nname: nested-note\n---\n\n# Not a skill\n", encoding="utf-8")

        idx = SkillIndex(skills, common, procs)
        results = idx.build_index()

        by_name = {m.name: m for m in results}
        assert by_name["flat-review"].path == flat
        assert by_name["flat-review"].is_common is False
        assert by_name["flat-review"].is_procedure is False
        assert "nested-note" not in by_name

    def test_is_common_flag(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.build_index()

        personal = [m for m in results if not m.is_common and not m.is_procedure]
        common_metas = [m for m in results if m.is_common]
        assert len(personal) == 2
        assert len(common_metas) == 2  # machine-tool + official-tool

    def test_is_procedure_flag(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.build_index()

        procedures = [m for m in results if m.is_procedure]
        assert len(procedures) == 2
        assert all(not p.is_common for p in procedures)

    def test_sort_order(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.build_index()

        tiers = []
        for m in results:
            if m.is_procedure:
                tiers.append(2)
            elif m.is_common:
                tiers.append(1)
            else:
                tiers.append(0)
        assert tiers == sorted(tiers)

    def test_usage_stats_merge_by_canonical_ref(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "mei"
        skills = anima_dir / "skills"
        common = tmp_path / "common_skills"
        procs = anima_dir / "procedures"
        skills.mkdir(parents=True)
        common.mkdir()
        procs.mkdir()
        _write_skill(skills, "same", desc="Personal same")
        nested = common / "community"
        nested.mkdir()
        _write_skill(nested, "same", desc="Common same")
        _write_procedure(procs, "same", desc="Procedure same")

        tracker = SkillUsageTracker(anima_dir)
        tracker.record("same", SkillUsageEventType.use, ref="skills/same/SKILL.md")
        tracker.record("same", SkillUsageEventType.use, is_common=True, ref="common_skills/community/same/SKILL.md")
        tracker.record("same", SkillUsageEventType.use, is_procedure=True, ref="procedures/same.md")

        idx = SkillIndex(skills, common, procs, anima_dir=anima_dir)
        results = idx.build_index()

        by_scope = {(m.name, m.is_common, m.is_procedure): m for m in results}
        assert by_scope[("same", False, False)].usage_count == 1
        assert by_scope[("same", True, False)].usage_count == 1
        assert by_scope[("same", False, True)].usage_count == 1

    def test_caching(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        r1 = idx.all_skills
        r2 = idx.all_skills
        assert r1 is r2

    def test_invalidate(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        r1 = idx.all_skills
        idx.invalidate()
        r2 = idx.all_skills
        assert r1 is not r2
        assert len(r1) == len(r2)


class TestSkillIndexSearch:
    def test_search_by_name(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.search("web-search")
        assert len(results) == 1
        assert results[0].name == "web-search"

    def test_search_by_description(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.search("deploy")
        assert len(results) == 1
        assert results[0].name == "deploy-checklist"

    def test_search_case_insensitive(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.search("MACHINE")
        assert len(results) == 1
        assert results[0].name == "machine-tool"

    def test_search_include_blocked(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.search("blocked", include_blocked=True)
        assert any(m.name == "blocked-skill" for m in results)

    def test_empty_query_returns_all(self, skill_dirs: tuple[Path, Path, Path]):
        skills, common, procs = skill_dirs
        idx = SkillIndex(skills, common, procs)
        results = idx.search("")
        assert len(results) == len(idx.all_skills)

    def test_no_procedures_dir(self, tmp_path: Path):
        skills = tmp_path / "skills"
        common = tmp_path / "common_skills"
        skills.mkdir()
        common.mkdir()
        _write_skill(skills, "only-skill", desc="Only skill")
        idx = SkillIndex(skills, common, procedures_dir=None)
        results = idx.build_index()
        assert len(results) == 1
        assert results[0].name == "only-skill"
