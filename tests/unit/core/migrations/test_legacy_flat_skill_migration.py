from __future__ import annotations

import json
from pathlib import Path

from core.migrations.steps import step_legacy_flat_skill_migration
from core.skills.loader import load_skill_metadata
from core.skills.models import SkillScanVerdict, SkillTrustLevel
from core.skills.policy import policy_for_skill
from core.skills.reference_rewriter import rewrite_skill_pointers_in_text, rewrite_skill_references_in_text


def _make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "runtime"
    data_dir.mkdir()
    (data_dir / "config.json").write_text("{}", encoding="utf-8")
    (data_dir / "animas").mkdir()
    return data_dir


def _make_anima(data_dir: Path, name: str = "mei") -> Path:
    anima_dir = data_dir / "animas" / name
    (anima_dir / "skills").mkdir(parents=True)
    (anima_dir / "state").mkdir()
    (anima_dir / "goals").mkdir()
    (anima_dir / "identity.md").write_text(f"# {name}\n", encoding="utf-8")
    return anima_dir


def _backup_files(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "shared" / "migration_backups" / "legacy_flat_skills").glob("*/**/*.md"))


def test_migrates_personal_flat_skill_to_trusted_bundle_and_rewrites_refs(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path)
    anima_dir = _make_anima(data_dir)
    flat = anima_dir / "skills" / "legacy.md"
    flat.write_text(
        "---\n"
        "name: old-name\n"
        "description: Legacy workflow\n"
        "trust_level: untrusted\n"
        "promotion_status: probation\n"
        "skill_policy:\n"
        "  use_mode: candidate_hint\n"
        "  injection: pointer_preferred\n"
        "security:\n"
        "  verdict: dangerous\n"
        "  scan_status: scanned\n"
        "---\n\n"
        "# Legacy\n\nDo the trusted old workflow.\n",
        encoding="utf-8",
    )
    (anima_dir / "state" / "active_skills.json").write_text(
        json.dumps({"version": 1, "threads": {"default": {"refs": ["skills/legacy.md", "legacy"]}}}),
        encoding="utf-8",
    )
    (anima_dir / "cron.md").write_text(
        'skill: skills/legacy.md\nskill_name: legacy\nskills: ["skills/legacy.md", "legacy"]\n',
        encoding="utf-8",
    )
    (anima_dir / "state" / "taskboard.json").write_text(
        json.dumps({"cards": [{"skills": ["skills/legacy.md"]}]}),
        encoding="utf-8",
    )
    (anima_dir / "goals" / "launch.md").write_text("skills:\n  - skills/legacy.md\n", encoding="utf-8")

    result = step_legacy_flat_skill_migration(data_dir, dry_run=False, verbose=True)

    assert result.error is None
    assert result.changed >= 4
    destination = anima_dir / "skills" / "legacy" / "SKILL.md"
    assert destination.is_file()
    assert not flat.exists()
    assert any(path.match("*/animas/mei/skills/legacy.md") for path in _backup_files(data_dir))

    meta = load_skill_metadata(destination)
    assert meta.name == "legacy"
    assert meta.description == "Legacy workflow"
    assert meta.trust_level == SkillTrustLevel.trusted
    assert meta.promotion_status == "trusted"
    assert meta.skill_policy.use_mode == "primary_guidance"
    assert meta.skill_policy.injection == "body_allowed"
    assert meta.security.verdict == SkillScanVerdict.safe
    assert meta.source.type == "local"
    assert meta.source.identifier == "skills/legacy.md"
    assert meta.source.origin == "legacy_flat_skill_migration"
    assert meta.source.owner_anima == "mei"

    policy = policy_for_skill(meta)
    assert policy.use_mode == "primary_guidance"
    assert policy.injection == "body_allowed"
    assert policy.render_section == "trusted"

    active_state = json.loads((anima_dir / "state" / "active_skills.json").read_text(encoding="utf-8"))
    assert active_state["threads"]["default"]["refs"] == ["skills/legacy/SKILL.md", "legacy"]
    cron = (anima_dir / "cron.md").read_text(encoding="utf-8")
    assert "skill: skills/legacy/SKILL.md" in cron
    assert "skill_name: legacy" in cron
    assert "skills/legacy.md" not in cron
    taskboard = json.loads((anima_dir / "state" / "taskboard.json").read_text(encoding="utf-8"))
    assert taskboard["cards"][0]["skills"] == ["skills/legacy/SKILL.md"]
    goal = (anima_dir / "goals" / "launch.md").read_text(encoding="utf-8")
    assert "skills/legacy/SKILL.md" in goal


def test_migrates_common_flat_skill_and_rewrites_common_refs_for_animas(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path)
    anima_dir = _make_anima(data_dir, "alice")
    common_dir = data_dir / "common_skills"
    common_dir.mkdir()
    flat = common_dir / "shared.md"
    flat.write_text("# Shared\n\nA shared legacy skill.\n", encoding="utf-8")
    from core.memory.task_queue import TaskQueueManager

    queue = TaskQueueManager(anima_dir)
    queue.submit(
        {
            "task_id": "shared-skill",
            "title": "Use shared",
            "description": "Use the shared skill",
            "skills": ["common_skills/shared.md"],
        }
    )

    result = step_legacy_flat_skill_migration(data_dir, dry_run=False, verbose=True)

    assert result.error is None
    destination = common_dir / "shared" / "SKILL.md"
    assert destination.is_file()
    assert not flat.exists()
    meta = load_skill_metadata(destination)
    assert meta.name == "shared"
    assert meta.trust_level == SkillTrustLevel.trusted
    assert meta.source.identifier == "common_skills/shared.md"
    assert meta.source.owner_anima is None
    task = queue.store.get_input(anima_dir.name, "shared-skill")
    assert task["skills"] == ["common_skills/shared/SKILL.md"]
    assert task["description"] == "Use the shared skill"
    assert not queue.queue_path.exists()


def test_destination_collision_keeps_existing_bundle_and_removes_flat_source(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path)
    anima_dir = _make_anima(data_dir)
    flat = anima_dir / "skills" / "collision.md"
    flat.write_text("# Flat\n\nDo not overwrite.\n", encoding="utf-8")
    destination = anima_dir / "skills" / "collision" / "SKILL.md"
    destination.parent.mkdir()
    destination.write_text(
        "---\nname: collision\ndescription: Existing bundle\n---\n\n# Existing\n",
        encoding="utf-8",
    )
    (anima_dir / "state" / "active_skills.json").write_text(
        json.dumps({"version": 1, "threads": {"default": {"refs": ["skills/collision.md"]}}}),
        encoding="utf-8",
    )

    result = step_legacy_flat_skill_migration(data_dir, dry_run=False, verbose=True)

    assert result.error is None
    assert not flat.exists()
    assert "Existing bundle" in destination.read_text(encoding="utf-8")
    assert any(path.match("*/animas/mei/skills/collision.md") for path in _backup_files(data_dir))
    active_state = json.loads((anima_dir / "state" / "active_skills.json").read_text(encoding="utf-8"))
    assert active_state["threads"]["default"]["refs"] == ["skills/collision/SKILL.md"]


def test_no_frontmatter_flat_skill_gets_inferred_metadata(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path)
    anima_dir = _make_anima(data_dir)
    (anima_dir / "skills" / "bare.md").write_text(
        "# Bare\n\n## 概要\n\nA bare legacy skill description.\n\nMore details.\n",
        encoding="utf-8",
    )

    result = step_legacy_flat_skill_migration(data_dir, dry_run=False, verbose=True)

    assert result.error is None
    meta = load_skill_metadata(anima_dir / "skills" / "bare" / "SKILL.md")
    assert meta.name == "bare"
    assert meta.description == "A bare legacy skill description."
    assert meta.trust_level == SkillTrustLevel.trusted


def test_pointer_rewriter_updates_flat_paths_without_rewriting_names() -> None:
    text = (
        "skill: skills/foo.md\nskill_name: foo\nskills: [skills/foo.md, foo]\nother: keep skills/foo.md inside prose\n"
    )

    rewritten = rewrite_skill_pointers_in_text(text, {"skills/foo.md": "skills/foo/SKILL.md"})

    assert "skill: skills/foo/SKILL.md" in rewritten
    assert "skill_name: foo" in rewritten
    assert 'skills: ["skills/foo/SKILL.md", "foo"]' in rewritten
    assert "other: keep skills/foo.md inside prose" in rewritten


def test_pointer_rewriter_preserves_unmatched_skill_line_formatting() -> None:
    text = "skills: [unrelated]\nskills: 'quoted-unrelated'\nskills:\n  - 'block-unrelated'\n"

    rewritten = rewrite_skill_pointers_in_text(text, {"skills/foo.md": "skills/foo/SKILL.md"})

    assert rewritten == text


def test_absorbed_reference_rewriter_matches_flat_skill_pointers() -> None:
    text = "skill: skills/foo.md\nskills:\n  - common_skills/foo.md\n  - bar\n"

    rewritten = rewrite_skill_references_in_text(text, "foo", absorbed_into="bar")

    assert rewritten == "skill: bar\nskills:\n  - bar\n"
