from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for skill reference rewrite proposals."""

import json
from pathlib import Path

from core.skills.reference_rewriter import collect_reference_rewrite_changes, rewrite_skill_references_in_text


def test_absorbed_into_replaces_cron_skill_pointer_without_duplicates() -> None:
    text = (
        "## Daily\n"
        "schedule: 0 9 * * *\n"
        "skills:\n"
        "  - old-skill\n"
        "  - umbrella-skill\n"
        "  - other-skill\n"
        "Run daily.\n"
    )

    rewritten = rewrite_skill_references_in_text(text, "old-skill", absorbed_into="umbrella-skill")

    assert "old-skill" not in rewritten
    assert rewritten.count("umbrella-skill") == 1
    assert "other-skill" in rewritten


def test_archive_without_absorbed_into_removes_empty_skills_field() -> None:
    text = "## Daily\nskills: [old-skill]\nDo work.\n"

    rewritten = rewrite_skill_references_in_text(text, "old-skill", absorbed_into=None)

    assert "skills:" not in rewritten
    assert "old-skill" not in rewritten
    assert "Do work." in rewritten


def test_jsonl_task_skill_pointer_rewrite(tmp_path: Path) -> None:
    task = {
        "task_id": "t1",
        "meta": {
            "skills": ["old-skill", "umbrella-skill"],
            "skill_name": "old-skill",
        },
    }
    text = json.dumps(task, ensure_ascii=False) + "\n"

    rewritten = rewrite_skill_references_in_text(text, "old-skill", absorbed_into="umbrella-skill")
    parsed = json.loads(rewritten)

    assert parsed["meta"]["skills"] == ["umbrella-skill"]
    assert parsed["meta"]["skill_name"] == "umbrella-skill"


def test_json_array_and_scalar_skill_rewrites() -> None:
    text = json.dumps([{"skill": "old-skill"}, {"skills": ["old-skill", "new-skill"]}])

    rewritten = rewrite_skill_references_in_text(text, "old-skill", absorbed_into="new-skill")
    parsed = json.loads(rewritten)

    assert parsed == [{"skill": "new-skill"}, {"skills": ["new-skill"]}]

    scalar = "skill_name: old-skill\nskill_pointer: other-skill\n"
    rewritten_scalar = rewrite_skill_references_in_text(scalar, "old-skill", absorbed_into="new-skill")
    assert "skill_name: new-skill" in rewritten_scalar
    assert "skill_pointer: other-skill" in rewritten_scalar

    scalar_skills = "skills: old-skill\n"
    rewritten_skills = rewrite_skill_references_in_text(scalar_skills, "old-skill", absorbed_into=None)
    assert "skills:" not in rewritten_skills


def test_scalar_comma_and_pointer_skill_refs_rewrite() -> None:
    text = (
        "## Daily\n"
        "skills: old-skill, skills/old-skill/SKILL.md, common_skills/community/kept/SKILL.md\n"
        "Do work.\n"
    )

    rewritten = rewrite_skill_references_in_text(text, "old-skill", absorbed_into="new-skill")

    assert "old-skill" not in rewritten
    assert "skills: new-skill, common_skills/community/kept/SKILL.md" in rewritten


def test_block_pointer_skill_ref_removed() -> None:
    text = (
        "## Daily\n"
        "skills:\n"
        "  - common_skills/community/old-skill/SKILL.md\n"
        "  - kept\n"
        "Do work.\n"
    )

    rewritten = rewrite_skill_references_in_text(text, "old-skill", absorbed_into=None)

    assert "old-skill" not in rewritten
    assert "  - kept" in rewritten


def test_malformed_json_falls_back_to_yamlish_rewrite() -> None:
    text = "{not json}\nskill: old-skill\n"

    rewritten = rewrite_skill_references_in_text(text, "old-skill", absorbed_into=None)

    assert "{not json}" in rewritten
    assert "skill: old-skill" not in rewritten


def test_collect_reference_rewrite_changes_covers_allowed_metadata_paths(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "goals").mkdir()
    (anima_dir / "cron.md").write_text("## Daily\nskills: [old-skill, kept]\n", encoding="utf-8")
    (anima_dir / "state" / "task_queue.jsonl").write_text(
        json.dumps({"task_id": "t1", "meta": {"skills": ["old-skill"]}}) + "\n",
        encoding="utf-8",
    )
    (anima_dir / "state" / "taskboard.json").write_text(
        json.dumps({"skill_pointer": "old-skill"}),
        encoding="utf-8",
    )
    (anima_dir / "state" / "goal_state.jsonl").write_text(
        json.dumps({"event_type": "set", "payload": {"skills": ["old-skill"]}}) + "\n",
        encoding="utf-8",
    )
    (anima_dir / "state" / "activity_log.jsonl").write_text(
        json.dumps({"skill_pointer": "old-skill"}) + "\n",
        encoding="utf-8",
    )
    (anima_dir / "goals" / "rollout.yaml").write_text("skills:\n\n  - old-skill\n", encoding="utf-8")

    changes = collect_reference_rewrite_changes(anima_dir, "old-skill", absorbed_into=None)

    assert {change.path for change in changes} == {
        "cron.md",
        "goals/rollout.yaml",
        "state/goal_state.jsonl",
        "state/task_queue.jsonl",
        "state/taskboard.json",
    }
    assert all("old-skill" in change.before for change in changes)
    assert all("old-skill" not in change.after for change in changes)
