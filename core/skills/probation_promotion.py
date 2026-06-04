from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Approval-less low-risk procedure-to-probation skill creation."""

import shutil
from pathlib import Path
from typing import Any

from core.i18n import t
from core.skills.models import SkillScanVerdict, SkillUsageEventType
from core.skills.promotion_utils import runtime_approval_required, scan_security_metadata
from core.skills.usage import SkillUsageTracker


def create_probation_skill_for_converter(
    converter: Any,
    procedure_path: str | Path,
    *,
    skill_name: str | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    enforce_policy: bool = True,
) -> Any:
    """Generate an active low-risk probation skill without human approval."""
    from core.skills.promotion import AUTO_SKILL_BLOCKED, AUTO_SKILL_CREATED, SkillPromotionResult

    source_path = converter._resolve_procedure_path(procedure_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Procedure not found: {source_path}")

    from core.memory.frontmatter import parse_frontmatter
    from core.skills.promotion_utils import safe_skill_name

    text = source_path.read_text(encoding="utf-8")
    proc_meta, proc_body = parse_frontmatter(text)
    final_skill_name = safe_skill_name(skill_name or proc_meta.get("name") or source_path.stem)
    active_skill_dir = converter._skills_dir / final_skill_name
    if active_skill_dir.exists():
        raise FileExistsError(f"Active skill already exists: {active_skill_dir}")
    quarantine_skill_dir = converter._quarantine_dir / final_skill_name
    if quarantine_skill_dir.exists():
        raise FileExistsError(f"Quarantine skill already exists: {quarantine_skill_dir}")

    candidate = converter.candidate_from_path(source_path)
    if enforce_policy and candidate is not None and not candidate.eligible:
        raise ValueError("Procedure is not eligible for skill promotion: " + ", ".join(candidate.reasons))

    meta = converter._build_skill_metadata(
        skill_name=final_skill_name,
        procedure_path=source_path,
        procedure_metadata=proc_meta,
        overrides=metadata_overrides or {},
    )
    meta["trust_level"] = "community"
    meta["promotion_status"] = "probation"
    meta["source"]["origin"] = "auto_created"
    meta["skill_policy"] = {
        "use_mode": "candidate_hint",
        "injection": "pointer_preferred",
    }
    meta["trusted_by"] = None
    meta["trusted_at"] = None
    meta["trust_reason"] = None

    body = converter._build_skill_body(final_skill_name, proc_meta, proc_body)
    active_skill_dir.mkdir(parents=True, exist_ok=False)
    skill_md = active_skill_dir / "SKILL.md"
    converter._write_skill_file(skill_md, meta, body)

    scan_result = converter._scanner.scan_skill(active_skill_dir, source="anima")
    meta["security"] = scan_security_metadata(scan_result)
    meta["risk"]["requires_human_approval"] = runtime_approval_required(meta["risk"], scan_result)
    converter._write_skill_file(skill_md, meta, body)

    if (
        scan_result.verdict != SkillScanVerdict.safe
        or scan_result.size_violations
        or meta["risk"]["requires_human_approval"]
    ):
        shutil.rmtree(active_skill_dir, ignore_errors=True)
        converter._append_audit(
            {
                "event_type": AUTO_SKILL_BLOCKED,
                "procedure_path": str(source_path.relative_to(converter._anima_dir)),
                "skill_name": final_skill_name,
                "scan_verdict": scan_result.verdict.value,
                "size_violations": scan_result.size_violations,
                "requires_human_approval": meta["risk"]["requires_human_approval"],
            }
        )
        return SkillPromotionResult(
            status="blocked" if scan_result.verdict == SkillScanVerdict.dangerous else "review",
            skill_name=final_skill_name,
            procedure_path=str(source_path.relative_to(converter._anima_dir)),
            scan_verdict=scan_result.verdict.value,
            requires_human_approval=True,
            message="Auto-created skill candidate requires review before activation.",
            findings=[f.model_dump(mode="json") for f in scan_result.findings],
            size_violations=scan_result.size_violations,
        )

    SkillUsageTracker(converter._anima_dir).record(
        final_skill_name,
        SkillUsageEventType.create,
        is_common=False,
        notes="auto_probation_skill",
        source_origin="auto_created",
    )
    converter._append_audit(
        {
            "event_type": AUTO_SKILL_CREATED,
            "procedure_path": str(source_path.relative_to(converter._anima_dir)),
            "skill_name": final_skill_name,
            "target_path": str(skill_md.relative_to(converter._anima_dir)),
            "scan_verdict": scan_result.verdict.value,
        }
    )
    return SkillPromotionResult(
        status="probation",
        skill_name=final_skill_name,
        procedure_path=str(source_path.relative_to(converter._anima_dir)),
        active_path=str(skill_md.relative_to(converter._anima_dir)),
        scan_verdict=scan_result.verdict.value,
        requires_human_approval=False,
        message=t(
            "skill_auto.created",
            skill_name=final_skill_name,
            path=skill_md.relative_to(converter._anima_dir),
        ),
        findings=[f.model_dump(mode="json") for f in scan_result.findings],
        size_violations=scan_result.size_violations,
    )
