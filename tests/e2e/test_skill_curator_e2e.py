from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for Skill Curator tool + catalog access flow."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.skills.curator import SkillCurator
from core.skills.index import SkillIndex
from core.tooling.handler import ToolHandler


def _write_skill(anima_dir: Path, name: str) -> None:
    skill_dir = anima_dir / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {name} release workflow\n"
        "trigger_phrases: [release workflow]\n"
        "domains: [software-delivery]\n"
        "---\n\n"
        f"# {name}\n",
        encoding="utf-8",
    )


def test_curator_tool_archives_and_restores_skill_access(data_dir: Path) -> None:
    tmp_path = data_dir
    anima_dir = tmp_path / "animas" / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True, exist_ok=True)
    _write_skill(anima_dir, "old-skill")

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    handler = ToolHandler(anima_dir=anima_dir, memory=memory, messenger=None, tool_registry=[])

    archived = json.loads(
        handler.handle(
            "archive_skill",
            {"skill_name": "old-skill", "reason": "unused", "absorbed_into": None, "actor": "mallory"},
        )
    )
    assert archived["to_state"] == "archived"
    assert archived["actor"] == "alice"
    assert archived["event_type"] == "state_change_proposed"

    index = SkillIndex(anima_dir / "skills", common_dir, anima_dir=anima_dir)
    assert "old-skill" in {meta.name for meta in index.all_skills}
    assert "SkillBlocked" not in handler.handle("read_memory_file", {"path": "skills/old-skill/SKILL.md"})

    # Routine model requests are proposals; an explicit host acceptance applies
    # the transition. Both catalog invalidation and read access follow it.
    curator = SkillCurator(anima_dir, common_skills_dir=common_dir)
    curator.change_state("old-skill", "archived", reason="accepted", actor="human")
    assert "old-skill" not in {meta.name for meta in index.all_skills}
    blocked_read = handler.handle("read_memory_file", {"path": "skills/old-skill/SKILL.md"})
    assert "SkillBlocked" in blocked_read

    restored = json.loads(
        handler.handle(
            "restore_skill",
            {"skill_name": "old-skill", "reason": "still useful"},
        )
    )
    assert restored["to_state"] == "active"
    assert restored["event_type"] == "state_change_proposed"
    assert "old-skill" not in {meta.name for meta in index.all_skills}
    curator.change_state("old-skill", "active", reason="accepted", actor="human")
    assert "old-skill" in {meta.name for meta in index.all_skills}
    assert "SkillBlocked" not in handler.handle("read_memory_file", {"path": "skills/old-skill/SKILL.md"})


def test_curate_skills_reports_metadata_gaps_and_duplicates(data_dir: Path) -> None:
    tmp_path = data_dir
    anima_dir = tmp_path / "animas" / "alice"
    _write_skill(anima_dir, "gmail-draft")
    _write_skill(anima_dir, "gmail-drafts")
    sparse = anima_dir / "skills" / "sparse"
    sparse.mkdir(parents=True)
    (sparse / "SKILL.md").write_text(
        "---\nname: sparse\ndescription: sparse skill\n---\n\n# Sparse\n",
        encoding="utf-8",
    )

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    handler = ToolHandler(anima_dir=anima_dir, memory=memory, messenger=None, tool_registry=[])

    report = json.loads(handler.handle("curate_skills", {}))

    assert "sparse" in report["metadata_gaps"]
    assert any({"gmail-draft", "gmail-drafts"} == {d["skill_name"], d["related_skill"]} for d in report["duplicates"])
