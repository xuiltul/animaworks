from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for procedure-to-skill promotion."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.memory.frontmatter import parse_frontmatter
from core.skills.models import SkillUsageEventType
from core.skills.promotion import (
    PROMOTION_DRAFT_CREATED,
    ProcedureToSkillConverter,
)
from core.skills.usage import SkillUsageTracker
from core.time_utils import now_iso


def _write_procedure(anima_dir: Path, name: str, *, body: str = "Step 1: Do the safe thing.") -> Path:
    procedures = anima_dir / "procedures"
    procedures.mkdir(parents=True, exist_ok=True)
    path = procedures / f"{name}.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Procedure for {name}\n"
        "success_count: 3\n"
        "failure_count: 0\n"
        "confidence: 0.95\n"
        f"last_used: {now_iso()}\n"
        "domains: [operations]\n"
        "trigger_phrases: [run deploy]\n"
        "---\n\n"
        f"# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return path


@contextmanager
def _resolved_promotion_approval(tmp_path: Path, anima_name: str, skill_name: str, actor: str):
    mock_auth = MagicMock()
    mock_auth.secret_key = "skill-promotion-unit-test"
    with (
        patch("core.notification.interactive.get_data_dir", return_value=tmp_path),
        patch("core.notification.interactive.get_shared_dir", return_value=tmp_path / "shared"),
        patch("core.notification.interactive.load_auth", return_value=mock_auth),
    ):
        import core.notification.interactive as interactive_mod
        from core.notification.interactive import get_interaction_router

        interactive_mod._router = None
        router = get_interaction_router()
        req = asyncio.run(
            router.create(
                anima_name,
                "skill_promotion",
                ["approve", "reject", "comment"],
                metadata={"skill_name": skill_name},
            )
        )
        asyncio.run(router.resolve(req.callback_id, "approve", actor, "test"))
        try:
            yield req.callback_id
        finally:
            interactive_mod._router = None


def test_find_candidates_uses_policy_thresholds(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    procedure = _write_procedure(anima_dir, "deploy-flow")

    converter = ProcedureToSkillConverter(anima_dir)
    candidate = converter.candidate_from_path(procedure)

    assert candidate is not None
    assert candidate.eligible is True
    assert converter.find_candidates() == [candidate]


def test_candidate_stats_jsonl_override_frontmatter(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    procedure = _write_procedure(anima_dir, "jsonl-flow")
    text = procedure.read_text(encoding="utf-8")
    procedure.write_text(
        text.replace("success_count: 3", "success_count: 0").replace("confidence: 0.95", "confidence: 0.0"),
        encoding="utf-8",
    )
    tracker = SkillUsageTracker(anima_dir)
    for _ in range(3):
        tracker.record("jsonl-flow", SkillUsageEventType.success)

    candidate = ProcedureToSkillConverter(anima_dir).candidate_from_path(procedure)

    assert candidate is not None
    assert candidate.eligible is True
    assert candidate.success_count == 3
    assert candidate.confidence == 1.0


def test_draft_rejects_low_confidence_candidate(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    procedure = _write_procedure(anima_dir, "immature-flow")
    text = procedure.read_text(encoding="utf-8")
    procedure.write_text(text.replace("success_count: 3", "success_count: 1"), encoding="utf-8")

    try:
        ProcedureToSkillConverter(anima_dir).create_quarantine_skill("procedures/immature-flow.md")
    except ValueError as exc:
        assert "success_count_below_threshold" in str(exc)
    else:
        raise AssertionError("Expected low-confidence procedure to be rejected")


def test_draft_creates_quarantine_skill_with_required_metadata(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_procedure(anima_dir, "deploy-flow")

    result = ProcedureToSkillConverter(anima_dir).create_quarantine_skill("procedures/deploy-flow.md")

    assert result.status == "review"
    assert result.requires_human_approval is True
    assert not (anima_dir / "skills" / "deploy-flow" / "SKILL.md").exists()

    skill_md = anima_dir / result.quarantine_path
    meta, body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    assert meta["trust_level"] == "quarantine"
    assert meta["promotion_status"] == "review"
    assert meta["source"]["type"] == "anima"
    assert meta["source"]["origin"] == "procedure_promotion"
    assert meta["version"] == 1
    assert meta["use_when"]
    assert meta["trigger_phrases"]
    assert meta["negative_phrases"] == []
    assert meta["domains"] == ["operations"]
    assert meta["security"]["scan_status"] == "scanned"
    assert "## Pitfalls" in body
    assert "## Verification" in body

    audit_line = (anima_dir / "state" / "skill_promotion.jsonl").read_text(encoding="utf-8").splitlines()[0]
    audit = json.loads(audit_line)
    assert audit["event_type"] == PROMOTION_DRAFT_CREATED
    assert audit["skill_name"] == "deploy-flow"


def test_draft_rejects_active_name_collision(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_procedure(anima_dir, "deploy-flow")
    active = anima_dir / "skills" / "deploy-flow"
    active.mkdir(parents=True)
    (active / "SKILL.md").write_text("---\nname: deploy-flow\n---\n\n# Existing\n", encoding="utf-8")

    try:
        ProcedureToSkillConverter(anima_dir).create_quarantine_skill("procedures/deploy-flow.md")
    except FileExistsError as exc:
        assert "Active skill already exists" in str(exc)
    else:
        raise AssertionError("Expected active skill name collision to be rejected")


def test_dangerous_draft_aborts_before_quarantine(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_procedure(anima_dir, "dangerous-flow", body="Run rm -rf / before continuing.")

    result = ProcedureToSkillConverter(anima_dir).create_quarantine_skill("procedures/dangerous-flow.md")

    assert result.status == "blocked"
    assert result.scan_verdict == "dangerous"
    assert not (anima_dir / "skills" / "quarantine" / "dangerous-flow").exists()
    assert not (anima_dir / "skills" / "dangerous-flow").exists()


def test_approve_moves_skill_to_active_and_records_create_event(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_procedure(anima_dir, "deploy-flow")
    converter = ProcedureToSkillConverter(anima_dir)
    converter.create_quarantine_skill("procedures/deploy-flow.md")

    with _resolved_promotion_approval(tmp_path, "alice", "deploy-flow", "mei") as callback_id:
        converter.register_approval_request("deploy-flow", callback_id)
        result = converter.approve_skill("deploy-flow", approval_callback_id=callback_id, approved_by="mei")

    assert result.status == "active"
    assert not (anima_dir / "skills" / "quarantine" / "deploy-flow").exists()
    active_skill = anima_dir / "skills" / "deploy-flow" / "SKILL.md"
    assert active_skill.exists()
    meta, _body = parse_frontmatter(active_skill.read_text(encoding="utf-8"))
    assert meta["trust_level"] == "trusted"
    assert meta["promotion_status"] == "active"
    assert meta["approved_by"] == "mei"
    assert meta["approved_at"]

    usage_lines = (anima_dir / "state" / "skill_usage.jsonl").read_text(encoding="utf-8").splitlines()
    usage = json.loads(usage_lines[-1])
    assert usage["skill_name"] == "deploy-flow"
    assert usage["event_type"] == "create"


def test_approve_requires_verified_approval_callback(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_procedure(anima_dir, "deploy-flow")
    converter = ProcedureToSkillConverter(anima_dir)
    converter.create_quarantine_skill("procedures/deploy-flow.md")

    try:
        converter.approve_skill("deploy-flow", approval_callback_id="", approved_by="mei")
    except ValueError as exc:
        assert "approval_callback_id is required" in str(exc)
    else:
        raise AssertionError("Expected approval without callback_id to be rejected")


def test_external_send_risk_keeps_runtime_human_approval(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_procedure(anima_dir, "send-status")
    converter = ProcedureToSkillConverter(anima_dir)
    converter.create_quarantine_skill(
        "procedures/send-status.md",
        metadata_overrides={"risk": {"external_send": True}},
    )

    with _resolved_promotion_approval(tmp_path, "alice", "send-status", "mei") as callback_id:
        converter.register_approval_request("send-status", callback_id)
        converter.approve_skill("send-status", approval_callback_id=callback_id, approved_by="mei")

    meta, _body = parse_frontmatter((anima_dir / "skills" / "send-status" / "SKILL.md").read_text(encoding="utf-8"))
    assert meta["risk"]["external_send"] is True
    assert meta["risk"]["requires_human_approval"] is True
