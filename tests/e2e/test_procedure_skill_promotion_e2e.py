from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for procedure-to-skill promotion."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.memory.frontmatter import parse_frontmatter
from core.skills.index import SkillIndex
from core.skills.promotion import ProcedureToSkillConverter
from core.time_utils import now_iso
from core.tooling.handler import ToolHandler
from core.tooling.schemas import build_tool_list


def _write_procedure(anima_dir: Path) -> None:
    procedures = anima_dir / "procedures"
    procedures.mkdir(parents=True, exist_ok=True)
    (procedures / "incident-summary.md").write_text(
        "---\n"
        "name: incident-summary\n"
        "description: Summarize an incident safely\n"
        "success_count: 4\n"
        "failure_count: 0\n"
        "confidence: 0.9\n"
        f"last_used: {now_iso()}\n"
        "domains: [operations]\n"
        "trigger_phrases: [incident summary]\n"
        "---\n\n"
        "# Incident Summary\n\n"
        "1. Read the incident notes.\n"
        "2. Summarize impact and next actions.\n",
        encoding="utf-8",
    )


@contextmanager
def _resolved_promotion_approval(tmp_path: Path, anima_name: str, skill_name: str, actor: str):
    mock_auth = MagicMock()
    mock_auth.secret_key = "skill-promotion-e2e-test"
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


def test_promotion_flow_activates_only_after_approval(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    common_skills = tmp_path / "common_skills"
    common_skills.mkdir(parents=True)
    (anima_dir / "skills").mkdir(parents=True)
    _write_procedure(anima_dir)

    converter = ProcedureToSkillConverter(anima_dir)
    draft = converter.create_quarantine_skill("procedures/incident-summary.md")
    assert draft.status == "review"

    index = SkillIndex(anima_dir / "skills", common_skills, anima_dir / "procedures", anima_dir=anima_dir)
    assert "incident-summary" not in {skill.name for skill in index.all_skills if not skill.is_procedure}

    with _resolved_promotion_approval(tmp_path, "alice", "incident-summary", "ops-lead") as callback_id:
        converter.register_approval_request("incident-summary", callback_id)
        approved = converter.approve_skill(
            "incident-summary",
            approval_callback_id=callback_id,
            approved_by="ops-lead",
        )
    assert approved.status == "active"

    index.invalidate()
    names = {skill.name for skill in index.all_skills if not skill.is_procedure}
    assert "incident-summary" in names
    from core.skills.router import SkillRouter

    routed = SkillRouter(min_score=0.1).route("incident summary", index.all_skills, top_k=3)
    assert any(candidate.name == "incident-summary" for candidate in routed)

    meta, _body = parse_frontmatter((anima_dir / "skills" / "incident-summary" / "SKILL.md").read_text("utf-8"))
    assert meta["trust_level"] == "trusted"
    assert meta["promotion_status"] == "active"
    assert meta["approved_by"] == "ops-lead"


def test_tool_handler_promotes_and_approves_procedure(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    (anima_dir / "skills").mkdir(parents=True)
    _write_procedure(anima_dir)

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    handler = ToolHandler(anima_dir=anima_dir, memory=memory, messenger=None, tool_registry=[])

    mock_auth = MagicMock()
    mock_auth.secret_key = "skill-promotion-test"
    with (
        patch("core.notification.interactive.get_data_dir", return_value=tmp_path),
        patch("core.notification.interactive.get_shared_dir", return_value=tmp_path / "shared"),
        patch("core.notification.interactive.load_auth", return_value=mock_auth),
    ):
        import core.notification.interactive as interactive_mod
        from core.notification.interactive import get_interaction_router

        interactive_mod._router = None
        draft_text = handler.handle(
            "promote_procedure_to_skill",
            {"path": "procedures/incident-summary.md", "skill_name": "incident-summary"},
        )
        draft = json.loads(draft_text)
        assert draft["status"] == "review"
        assert draft["approval_callback_id"]
        assert (anima_dir / "skills" / "quarantine" / "incident-summary" / "SKILL.md").exists()

        rejected_text = handler.handle(
            "promote_procedure_to_skill",
            {"action": "approve", "skill_name": "incident-summary", "approval_callback_id": draft["approval_callback_id"]},
        )
        assert json.loads(rejected_text)["error_type"] == "ApprovalRequired"

        router = get_interaction_router()
        asyncio.run(router.resolve(draft["approval_callback_id"], "approve", "ops-lead", "test"))

        approved_text = handler.handle(
            "promote_procedure_to_skill",
            {
                "action": "approve",
                "skill_name": "incident-summary",
                "approval_callback_id": draft["approval_callback_id"],
            },
        )
        interactive_mod._router = None

    approved = json.loads(approved_text)
    assert approved["status"] == "active"
    assert (anima_dir / "skills" / "incident-summary" / "SKILL.md").exists()


def test_tool_schema_exposes_promotion_tool() -> None:
    tools = build_tool_list(include_create_skill=True)
    names = {tool["name"] for tool in tools}
    assert "create_skill" in names
    assert "promote_procedure_to_skill" in names
    promote = next(tool for tool in tools if tool["name"] == "promote_procedure_to_skill")
    assert "approval_callback_id" in promote["parameters"]["properties"]
