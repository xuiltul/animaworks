from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Autonomous procedure-memory to probation-skill learning."""

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.memory.frontmatter import parse_frontmatter
from core.paths import get_common_skills_dir
from core.skills.index import SkillIndex
from core.skills.models import SkillMetadata
from core.skills.promotion import ProcedureToSkillConverter, SkillPromotionResult
from core.skills.promotion_utils import normalise_risk
from core.skills.router import fill_routing_metadata_gaps
from core.time_utils import now_iso

AUTO_SKILL_SKIPPED = "auto_skill_skipped"
AUTO_CREATED_REPORT_KEY = "skill_auto.created"
_DUPLICATE_SIMILARITY_THRESHOLD = 0.88


@dataclass(slots=True)
class AutoSkillSkip:
    skill_name: str
    procedure_path: str
    reason: str
    related_skill: str | None = None


@dataclass(slots=True)
class AutoSkillLearnResult:
    created: list[SkillPromotionResult] = field(default_factory=list)
    skipped: list[AutoSkillSkip] = field(default_factory=list)
    blocked: list[SkillPromotionResult] = field(default_factory=list)

    @property
    def report_lines(self) -> list[str]:
        return [result.message for result in self.created if result.message]


class AutonomousSkillLearner:
    """Create low-risk probation skills from proven procedure memories."""

    def __init__(
        self,
        anima_dir: Path,
        *,
        converter: ProcedureToSkillConverter | None = None,
        common_skills_dir: Path | None = None,
    ) -> None:
        self.anima_dir = anima_dir
        self.converter = converter or ProcedureToSkillConverter(anima_dir)
        self.common_skills_dir = common_skills_dir or get_common_skills_dir()
        self.audit_path = anima_dir / "state" / "skill_promotion.jsonl"

    def run(self) -> AutoSkillLearnResult:
        """Create all currently eligible low-risk probation skills."""
        result = AutoSkillLearnResult()
        existing = self._existing_skills()
        for candidate in self.converter.find_candidates(eligible_only=False):
            procedure_rel = str(candidate.path.relative_to(self.anima_dir))
            if not candidate.eligible:
                skip = AutoSkillSkip(candidate.name, procedure_rel, ",".join(candidate.reasons) or "ineligible")
                self._record_skip(skip)
                result.skipped.append(skip)
                continue
            duplicate = self._duplicate_for(candidate.name, candidate.metadata, existing)
            if duplicate is not None:
                skip = AutoSkillSkip(candidate.name, procedure_rel, "duplicate_skill", duplicate.name)
                self._record_skip(skip)
                result.skipped.append(skip)
                continue
            risk_reason = self._risk_skip_reason(candidate.metadata)
            if risk_reason is not None:
                skip = AutoSkillSkip(candidate.name, procedure_rel, risk_reason)
                self._record_skip(skip)
                result.skipped.append(skip)
                continue

            promotion = self.converter.create_probation_skill(
                candidate.path,
                skill_name=candidate.name,
                metadata_overrides=self._metadata_overrides(candidate.path, candidate.name, candidate.metadata),
            )
            if promotion.status == "probation":
                result.created.append(promotion)
                existing = self._existing_skills()
            else:
                result.blocked.append(promotion)
        return result

    def _existing_skills(self) -> list[SkillMetadata]:
        index = SkillIndex(
            self.anima_dir / "skills",
            self.common_skills_dir,
            self.anima_dir / "procedures",
            anima_dir=self.anima_dir,
        )
        return [meta for meta in index.search("", include_blocked=True) if not meta.is_procedure]

    def _duplicate_for(self, skill_name: str, metadata: dict[str, Any], existing: list[SkillMetadata]) -> SkillMetadata | None:
        description = str(metadata.get("description") or metadata.get("title") or "").strip()
        for meta in existing:
            if meta.name == skill_name:
                return meta
            score = max(
                difflib.SequenceMatcher(None, skill_name.casefold(), meta.name.casefold()).ratio(),
                difflib.SequenceMatcher(None, description.casefold(), meta.description.casefold()).ratio()
                if description and meta.description
                else 0.0,
            )
            if score >= _DUPLICATE_SIMILARITY_THRESHOLD:
                return meta
        return None

    def _risk_skip_reason(self, metadata: dict[str, Any]) -> str | None:
        risk = normalise_risk(metadata.get("risk") or {})
        for risk_field in (
            "destructive",
            "external_send",
            "handles_untrusted_data",
            "open_world",
            "credential",
            "production",
            "billing",
            "private_data",
            "requires_human_approval",
        ):
            if risk.get(risk_field):
                return f"risk_{risk_field}"
        return None

    def _metadata_overrides(self, path: Path, skill_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        proc_meta, body = parse_frontmatter(text)
        description = str(
            proc_meta.get("description")
            or proc_meta.get("title")
            or metadata.get("description")
            or metadata.get("title")
            or skill_name
        ).strip()
        return fill_routing_metadata_gaps(dict(proc_meta), skill_name=skill_name, description=description, body=body)

    def _record_skip(self, skip: AutoSkillSkip) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": now_iso(),
            "event_type": AUTO_SKILL_SKIPPED,
            "procedure_path": skip.procedure_path,
            "skill_name": skip.skill_name,
            "reason": skip.reason,
            "related_skill": skip.related_skill,
        }
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


__all__ = [
    "AUTO_CREATED_REPORT_KEY",
    "AUTO_SKILL_SKIPPED",
    "AutoSkillLearnResult",
    "AutoSkillSkip",
    "AutonomousSkillLearner",
]
