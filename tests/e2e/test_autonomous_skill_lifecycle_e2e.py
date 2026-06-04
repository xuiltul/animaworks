from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from core.skills.autolearn import AutonomousSkillLearner
from core.skills.usage import SkillUsageTracker
from core.time_utils import now_iso


def _write_procedure(anima_dir: Path, name: str, body: str = "Follow the stable procedure.", extra: str = "") -> Path:
    proc_dir = anima_dir / "procedures"
    proc_dir.mkdir(parents=True, exist_ok=True)
    path = proc_dir / f"{name}.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"title: {name.title()}\n"
        f"description: Procedure for {name}\n"
        "success_count: 3\n"
        "failure_count: 0\n"
        "confidence: 1.0\n"
        f"last_used_at: {now_iso()}\n"
        f"{extra}"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _write_skill(anima_dir: Path, name: str) -> Path:
    skill_dir = anima_dir / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: Existing skill\n---\n\n# Existing\n", encoding="utf-8")
    return path


def test_autonomous_learner_creates_probation_skill_without_approval(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_procedure(anima_dir, "mail-helper")

    result = AutonomousSkillLearner(anima_dir, common_skills_dir=common_dir).run()

    assert len(result.created) == 1
    created = result.created[0]
    assert created.requires_human_approval is False
    assert created.message == "スキル化しました: mail-helper (skills/mail-helper/SKILL.md)"
    skill_text = (anima_dir / "skills" / "mail-helper" / "SKILL.md").read_text(encoding="utf-8")
    assert "trust_level: community" in skill_text
    assert "promotion_status: probation" in skill_text
    assert "origin: auto_created" in skill_text
    assert "injection: pointer_preferred" in skill_text
    stats = SkillUsageTracker(anima_dir).get_stats("mail-helper")
    assert stats.create_origins == {"auto_created": 1}


def test_autonomous_learner_skips_risky_procedure(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_procedure(
        anima_dir,
        "send-helper",
        extra="risk:\n  external_send: true\n",
    )

    result = AutonomousSkillLearner(anima_dir, common_skills_dir=common_dir).run()

    assert result.created == []
    assert result.skipped[0].reason == "risk_external_send"
    assert not (anima_dir / "skills" / "send-helper").exists()


def test_autonomous_learner_skips_private_or_billing_risk(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_procedure(
        anima_dir,
        "billing-helper",
        extra="risk:\n  billing: true\n  private-data: true\n",
    )

    result = AutonomousSkillLearner(anima_dir, common_skills_dir=common_dir).run()

    assert result.created == []
    assert result.skipped[0].reason == "risk_billing"
    assert not (anima_dir / "skills" / "billing-helper").exists()


def test_autonomous_learner_skips_nested_routing_risk(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_procedure(
        anima_dir,
        "routing-send-helper",
        extra="routing:\n  risk:\n    external_send: true\n",
    )

    result = AutonomousSkillLearner(anima_dir, common_skills_dir=common_dir).run()

    assert result.created == []
    assert result.skipped[0].reason == "risk_external_send"
    assert not (anima_dir / "skills" / "routing-send-helper").exists()


def test_autonomous_learner_records_ineligible_candidate_reason(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_procedure(
        anima_dir,
        "too-new",
        extra="success_count: 1\nconfidence: 0.5\n",
    )

    result = AutonomousSkillLearner(anima_dir, common_skills_dir=common_dir).run()

    assert result.created == []
    assert "success_count_below_threshold" in result.skipped[0].reason
    assert "confidence_below_threshold" in result.skipped[0].reason


def test_autonomous_learner_blocks_dangerous_candidate(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_procedure(anima_dir, "credential-helper", body="Run: cat .env")

    result = AutonomousSkillLearner(anima_dir, common_skills_dir=common_dir).run()

    assert result.created == []
    assert result.blocked[0].status == "blocked"
    assert not (anima_dir / "skills" / "credential-helper").exists()


def test_autonomous_learner_skips_duplicate_skill(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir()
    _write_skill(anima_dir, "mail-helper")
    _write_procedure(anima_dir, "mail-helper")

    result = AutonomousSkillLearner(anima_dir, common_skills_dir=common_dir).run()

    assert result.created == []
    assert result.skipped[0].reason == "duplicate_skill"
