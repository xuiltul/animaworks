from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Procedure-to-skill promotion pipeline.

Generated skills are written to ``skills/quarantine`` first. Approval is a
separate explicit step that moves the skill into the active personal catalog.
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.memory._io import atomic_write_text
from core.memory.frontmatter import parse_frontmatter
from core.skills.guard import SkillScanner
from core.skills.models import SkillScanVerdict, SkillUsageEventType
from core.skills.promotion_approval import (
    SkillPromotionApprovalMismatchError,
    SkillPromotionApprovalRequiredError,
    verify_skill_promotion_approval,
)
from core.skills.promotion_utils import (
    as_float,
    as_int,
    as_list,
    as_optional_str,
    first_non_empty,
    normalise_risk,
    runtime_approval_required,
    safe_skill_name,
    scan_security_metadata,
    within_days,
)
from core.skills.usage import SkillUsageTracker
from core.time_utils import now_iso

PROMOTION_DRAFT_CREATED = "promotion_draft_created"
PROMOTION_APPROVED = "promotion_approved"


@dataclass(slots=True)
class PromotionPolicy:
    """Thresholds for detecting procedures that are worth promotion."""

    success_count_threshold: int = 3
    confidence_threshold: float = 0.8
    failure_count_max: int = 1
    last_used_within_days: int = 180
    auto_activate: bool = False
    require_approval_on_warn: bool = True


@dataclass(slots=True)
class ProcedurePromotionCandidate:
    """A procedure and the decision inputs for promotion eligibility."""

    path: Path
    name: str
    metadata: dict[str, Any]
    success_count: int
    failure_count: int
    confidence: float
    last_used_at: str | None
    eligible: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SkillPromotionResult:
    """Result from draft or approval actions."""

    status: str
    skill_name: str
    procedure_path: str | None = None
    quarantine_path: str | None = None
    active_path: str | None = None
    approval_callback_id: str | None = None
    scan_verdict: str | None = None
    requires_human_approval: bool = True
    message: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    size_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "skill_name": self.skill_name,
            "procedure_path": self.procedure_path,
            "quarantine_path": self.quarantine_path,
            "active_path": self.active_path,
            "approval_callback_id": self.approval_callback_id,
            "scan_verdict": self.scan_verdict,
            "requires_human_approval": self.requires_human_approval,
            "message": self.message,
            "findings": self.findings,
            "size_violations": self.size_violations,
        }


class ProcedureToSkillConverter:
    """Convert proven procedures into reviewed personal skills."""

    def __init__(
        self,
        anima_dir: Path,
        *,
        scanner: SkillScanner | None = None,
        policy: PromotionPolicy | None = None,
        owner_anima: str | None = None,
    ) -> None:
        self._anima_dir = anima_dir
        self._procedures_dir = anima_dir / "procedures"
        self._skills_dir = anima_dir / "skills"
        self._quarantine_dir = self._skills_dir / "quarantine"
        self._audit_path = anima_dir / "state" / "skill_promotion.jsonl"
        self._scanner = scanner or SkillScanner()
        self._policy = policy or PromotionPolicy()
        self._owner_anima = owner_anima or anima_dir.name

    @property
    def policy(self) -> PromotionPolicy:
        return self._policy

    def find_candidates(self, *, eligible_only: bool = True) -> list[ProcedurePromotionCandidate]:
        """Return procedure files that satisfy, or nearly satisfy, promotion policy."""
        if not self._procedures_dir.exists():
            return []

        candidates: list[ProcedurePromotionCandidate] = []
        for path in sorted(self._procedures_dir.glob("*.md")):
            candidate = self.candidate_from_path(path)
            if candidate is None:
                continue
            if eligible_only and not candidate.eligible:
                continue
            candidates.append(candidate)
        return candidates

    def candidate_from_path(self, path: str | Path) -> ProcedurePromotionCandidate | None:
        procedure_path = self._resolve_procedure_path(path)
        if not procedure_path.exists():
            return None

        meta, _body = parse_frontmatter(procedure_path.read_text(encoding="utf-8"))
        name = safe_skill_name(meta.get("name") or procedure_path.stem)
        stats = self._usage_stats_for_procedure(meta.get("name") or procedure_path.stem, name)
        if stats.success_count or stats.failure_count:
            success_count = stats.success_count
            failure_count = stats.failure_count
            confidence = success_count / max(1, success_count + failure_count)
            last_used_at = stats.last_used_at
        else:
            success_count = as_int(meta.get("success_count"), default=0)
            failure_count = as_int(meta.get("failure_count"), default=0)
            confidence = as_float(meta.get("confidence"), default=0.0)
            last_used_at = as_optional_str(meta.get("last_used") or meta.get("last_used_at"))

        reasons: list[str] = []
        if success_count < self._policy.success_count_threshold:
            reasons.append("success_count_below_threshold")
        if confidence < self._policy.confidence_threshold:
            reasons.append("confidence_below_threshold")
        if failure_count > self._policy.failure_count_max:
            reasons.append("failure_count_above_threshold")
        if not within_days(last_used_at, self._policy.last_used_within_days):
            reasons.append("last_used_outside_window")

        return ProcedurePromotionCandidate(
            path=procedure_path,
            name=name,
            metadata=meta,
            success_count=success_count,
            failure_count=failure_count,
            confidence=confidence,
            last_used_at=last_used_at,
            eligible=not reasons,
            reasons=reasons,
        )

    def create_quarantine_skill(
        self,
        procedure_path: str | Path,
        *,
        skill_name: str | None = None,
        metadata_overrides: dict[str, Any] | None = None,
        enforce_policy: bool = True,
    ) -> SkillPromotionResult:
        """Generate a quarantine SKILL.md and scan it before review."""
        source_path = self._resolve_procedure_path(procedure_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Procedure not found: {source_path}")

        text = source_path.read_text(encoding="utf-8")
        proc_meta, proc_body = parse_frontmatter(text)
        final_skill_name = safe_skill_name(skill_name or proc_meta.get("name") or source_path.stem)
        active_skill_dir = self._skills_dir / final_skill_name
        if active_skill_dir.exists():
            raise FileExistsError(f"Active skill already exists: {active_skill_dir}")
        quarantine_skill_dir = self._quarantine_dir / final_skill_name
        if quarantine_skill_dir.exists():
            raise FileExistsError(f"Quarantine skill already exists: {quarantine_skill_dir}")
        candidate = self.candidate_from_path(source_path)
        if enforce_policy and candidate is not None and not candidate.eligible:
            raise ValueError("Procedure is not eligible for skill promotion: " + ", ".join(candidate.reasons))

        meta = self._build_skill_metadata(
            skill_name=final_skill_name,
            procedure_path=source_path,
            procedure_metadata=proc_meta,
            overrides=metadata_overrides or {},
        )
        body = self._build_skill_body(final_skill_name, proc_meta, proc_body)

        quarantine_skill_dir.mkdir(parents=True, exist_ok=False)
        skill_md = quarantine_skill_dir / "SKILL.md"
        self._write_skill_file(skill_md, meta, body)

        scan_result = self._scanner.scan_skill(quarantine_skill_dir, source="anima")
        meta["security"] = scan_security_metadata(scan_result)
        meta["risk"]["requires_human_approval"] = runtime_approval_required(meta["risk"], scan_result)
        self._write_skill_file(skill_md, meta, body)

        if scan_result.verdict == SkillScanVerdict.dangerous or scan_result.size_violations:
            shutil.rmtree(quarantine_skill_dir, ignore_errors=True)
            self._append_audit(
                {
                    "event_type": "promotion_draft_blocked",
                    "procedure_path": str(source_path.relative_to(self._anima_dir)),
                    "skill_name": final_skill_name,
                    "scan_verdict": scan_result.verdict.value,
                    "size_violations": scan_result.size_violations,
                }
            )
            return SkillPromotionResult(
                status="blocked",
                skill_name=final_skill_name,
                procedure_path=str(source_path.relative_to(self._anima_dir)),
                scan_verdict=scan_result.verdict.value,
                requires_human_approval=True,
                message="Dangerous or oversized promoted skill draft was blocked before quarantine.",
                findings=[f.model_dump(mode="json") for f in scan_result.findings],
                size_violations=scan_result.size_violations,
            )

        self._append_audit(
            {
                "event_type": PROMOTION_DRAFT_CREATED,
                "procedure_path": str(source_path.relative_to(self._anima_dir)),
                "skill_name": final_skill_name,
                "target_path": str(skill_md.relative_to(self._anima_dir)),
                "scan_verdict": scan_result.verdict.value,
                "requires_human_approval": True,
            }
        )

        return SkillPromotionResult(
            status="review",
            skill_name=final_skill_name,
            procedure_path=str(source_path.relative_to(self._anima_dir)),
            quarantine_path=str(skill_md.relative_to(self._anima_dir)),
            scan_verdict=scan_result.verdict.value,
            requires_human_approval=True,
            message="Quarantine skill draft created. Human approval is required before activation.",
            findings=[f.model_dump(mode="json") for f in scan_result.findings],
            size_violations=scan_result.size_violations,
        )

    def register_approval_request(self, skill_name: str, callback_id: str) -> None:
        """Attach an interactive approval callback to a quarantine skill draft."""
        final_skill_name = safe_skill_name(skill_name)
        skill_md = self._quarantine_dir / final_skill_name / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"Quarantine skill not found: {skill_md}")
        meta, body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        meta["approval_callback_id"] = callback_id
        self._write_skill_file(skill_md, meta, body)

    def approve_skill(
        self,
        skill_name: str,
        *,
        approval_callback_id: str,
        approved_by: str | None = None,
    ) -> SkillPromotionResult:
        """Move a reviewed quarantine skill into the active personal catalog."""
        final_skill_name = safe_skill_name(skill_name)
        quarantine_skill_dir = self._quarantine_dir / final_skill_name
        quarantine_skill_md = quarantine_skill_dir / "SKILL.md"
        if not quarantine_skill_md.exists():
            raise FileNotFoundError(f"Quarantine skill not found: {quarantine_skill_md}")

        active_skill_dir = self._skills_dir / final_skill_name
        if active_skill_dir.exists():
            raise FileExistsError(f"Active skill already exists: {active_skill_dir}")

        meta, body = parse_frontmatter(quarantine_skill_md.read_text(encoding="utf-8"))
        stored_callback_id = as_optional_str(meta.get("approval_callback_id"))
        if not approval_callback_id:
            raise SkillPromotionApprovalRequiredError("approval_callback_id is required")
        if not stored_callback_id:
            raise SkillPromotionApprovalRequiredError("Quarantine skill has no registered approval_callback_id")
        if stored_callback_id != approval_callback_id:
            raise SkillPromotionApprovalMismatchError("approval_callback_id does not match quarantine skill metadata")
        approval = verify_skill_promotion_approval(
            approval_callback_id,
            owner_anima=self._owner_anima,
            skill_name=final_skill_name,
        )
        if approved_by is not None and approved_by != approval.actor:
            raise SkillPromotionApprovalMismatchError("approved_by does not match approval actor")
        approved_by = approval.actor

        scan = self._scanner.scan_skill(quarantine_skill_dir, source="anima")
        if scan.verdict == SkillScanVerdict.dangerous or scan.size_violations:
            raise ValueError("Dangerous or oversized quarantine skill cannot be approved")

        meta["security"] = scan_security_metadata(scan)
        meta["trust_level"] = "trusted"
        meta["promotion_status"] = "active"
        meta["approved_by"] = approved_by
        meta["approved_at"] = now_iso()
        meta.setdefault("risk", {})
        meta["risk"]["requires_human_approval"] = runtime_approval_required(meta["risk"], scan)

        self._write_skill_file(quarantine_skill_md, meta, body)
        active_skill_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(quarantine_skill_dir), str(active_skill_dir))

        SkillUsageTracker(self._anima_dir).record(
            final_skill_name,
            SkillUsageEventType.create,
            is_common=False,
            notes="procedure_promotion_approved",
        )
        self._append_audit(
            {
                "event_type": PROMOTION_APPROVED,
                "skill_name": final_skill_name,
                "target_path": str((active_skill_dir / "SKILL.md").relative_to(self._anima_dir)),
                "approved_by": approved_by,
                "approval_callback_id": meta.get("approval_callback_id"),
                "scan_verdict": scan.verdict.value,
            }
        )

        return SkillPromotionResult(
            status="active",
            skill_name=final_skill_name,
            active_path=str((active_skill_dir / "SKILL.md").relative_to(self._anima_dir)),
            approval_callback_id=meta.get("approval_callback_id"),
            scan_verdict=scan.verdict.value,
            requires_human_approval=meta["risk"]["requires_human_approval"],
            message="Skill approved and activated.",
            findings=[f.model_dump(mode="json") for f in scan.findings],
            size_violations=scan.size_violations,
        )

    def _usage_stats_for_procedure(self, raw_name: Any, safe_name: str):
        tracker = SkillUsageTracker(self._anima_dir)
        names = [str(raw_name or "").strip(), safe_name]
        seen: set[str] = set()
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            stats = tracker.get_stats(name)
            if stats.success_count or stats.failure_count:
                return stats
        return tracker.get_stats(safe_name)

    def _resolve_procedure_path(self, path: str | Path) -> Path:
        raw = Path(path)
        target = raw if raw.is_absolute() else self._anima_dir / raw
        resolved = target.resolve()
        procedures_root = self._procedures_dir.resolve()
        if not resolved.is_relative_to(procedures_root) or resolved.suffix.lower() != ".md":
            raise ValueError("Procedure path must point to a .md file under procedures/")
        return resolved

    def _build_skill_metadata(
        self,
        *,
        skill_name: str,
        procedure_path: Path,
        procedure_metadata: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        description = first_non_empty(
            overrides.get("description"),
            procedure_metadata.get("description"),
            procedure_metadata.get("title"),
            f"Promoted procedure skill for {skill_name}",
        )
        use_when = as_list(overrides.get("use_when") or procedure_metadata.get("use_when") or description)
        trigger_phrases = as_list(
            overrides.get("trigger_phrases")
            or procedure_metadata.get("trigger_phrases")
            or skill_name.replace("-", " ")
        )
        negative_phrases = as_list(
            overrides.get("negative_phrases")
            or procedure_metadata.get("negative_phrases")
            or procedure_metadata.get("do_not_use_when")
        )
        domains = as_list(
            overrides.get("domains") or procedure_metadata.get("domains") or procedure_metadata.get("tags")
        )
        if not domains:
            domains = ["general"]
        tags = as_list(overrides.get("tags") or procedure_metadata.get("tags"))
        risk = normalise_risk(overrides.get("risk") or procedure_metadata.get("risk") or {})

        return {
            "name": skill_name,
            "description": description,
            "version": 1,
            "trust_level": "quarantine",
            "promotion_status": "review",
            "source": {
                "type": "anima",
                "owner_anima": self._owner_anima,
                "origin": "procedure_promotion",
                "identifier": str(procedure_path.relative_to(self._anima_dir)),
            },
            "use_when": use_when,
            "trigger_phrases": trigger_phrases,
            "negative_phrases": negative_phrases,
            "domains": domains,
            "tags": tags,
            "risk": risk,
        }

    def _build_skill_body(
        self,
        skill_name: str,
        procedure_metadata: dict[str, Any],
        procedure_body: str,
    ) -> str:
        title = first_non_empty(procedure_metadata.get("title"), skill_name.replace("-", " ").title())
        body = procedure_body.strip()
        if not body:
            body = f"# {title}\n\nFollow the promoted procedure for {skill_name}."
        if "## Pitfalls" not in body:
            body += "\n\n## Pitfalls\n\n- Review the original procedure assumptions before applying this skill."
        if "## Verification" not in body:
            body += "\n\n## Verification\n\n- Confirm the task result and report the skill outcome."
        return body.rstrip() + "\n"

    def _write_skill_file(self, path: Path, metadata: dict[str, Any], body: str) -> None:
        frontmatter = yaml.dump(
            metadata,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).strip()
        atomic_write_text(path, f"---\n{frontmatter}\n---\n\n{body.rstrip()}\n")

    def _append_audit(self, event: dict[str, Any]) -> None:
        event = {"ts": now_iso(), **event}
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
