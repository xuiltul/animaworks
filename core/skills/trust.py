from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Human-driven promotion from probation skills to trusted skills."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from core.memory._io import atomic_write_text
from core.memory.frontmatter import parse_frontmatter
from core.paths import get_common_skills_dir
from core.skills.guard import SkillScanner
from core.skills.index import SkillIndex
from core.skills.models import SkillLifecycleState, SkillScanVerdict, SkillTrustLevel, SkillUsageEventType
from core.skills.promotion_utils import scan_security_metadata
from core.skills.usage import SkillUsageTracker
from core.time_utils import now_iso


@dataclass(slots=True)
class TrustedPromotionResult:
    skill_name: str
    path: str
    trusted_by: str
    trusted_at: str
    trust_reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "skill_name": self.skill_name,
            "path": self.path,
            "trusted_by": self.trusted_by,
            "trusted_at": self.trusted_at,
            "trust_reason": self.trust_reason,
        }


def promote_skill_to_trusted(
    anima_dir: Path,
    ref: str,
    *,
    trusted_by: str = "user",
    trust_reason: str = "human_instruction",
    scanner: SkillScanner | None = None,
) -> TrustedPromotionResult:
    """Promote a visible, safe skill to trusted operating guidance."""
    index = SkillIndex(
        anima_dir / "skills",
        get_common_skills_dir(),
        anima_dir / "procedures",
        anima_dir=anima_dir,
    )
    meta = index.resolve_skill_reference(ref)
    if meta is None or meta.path is None:
        raise ValueError(f"Skill not found: {ref}")
    if meta.is_procedure:
        raise ValueError("Procedure memories cannot be promoted directly; create a skill first")
    if meta.lifecycle_state in {
        SkillLifecycleState.archived,
        SkillLifecycleState.blocked,
        SkillLifecycleState.deleted,
    }:
        raise ValueError(f"Cannot promote lifecycle state: {meta.lifecycle_state.value}")
    if meta.trust_level in {SkillTrustLevel.blocked, SkillTrustLevel.quarantine}:
        raise ValueError(f"Cannot promote trust level: {meta.trust_level.value}")
    if str(meta.promotion_status or "").lower() != "probation":
        raise ValueError("Only probation skills can be promoted to trusted")
    if meta.security.verdict in {SkillScanVerdict.dangerous, SkillScanVerdict.warn, SkillScanVerdict.caution}:
        raise ValueError(f"Cannot promote unsafe skill: {meta.security.verdict.value}")

    skill_path = meta.path
    text = skill_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    scan = (scanner or SkillScanner()).scan_skill(skill_path.parent if skill_path.name == "SKILL.md" else skill_path, source="anima")
    if scan.verdict in {SkillScanVerdict.dangerous, SkillScanVerdict.warn, SkillScanVerdict.caution} or scan.size_violations:
        raise ValueError(f"Cannot promote unsafe skill: {scan.verdict.value}")

    trusted_at = now_iso()
    source = frontmatter.get("source")
    if isinstance(source, str):
        source = {"type": source}
    if not isinstance(source, dict):
        source = {"type": "anima"}
    frontmatter.update(
        {
            "trust_level": "trusted",
            "promotion_status": "trusted",
            "source": source,
            "skill_policy": {
                "use_mode": "primary_guidance",
                "injection": "body_allowed",
            },
            "trusted_by": trusted_by,
            "trusted_at": trusted_at,
            "trust_reason": trust_reason,
            "security": scan_security_metadata(scan),
        }
    )
    rendered = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
    atomic_write_text(skill_path, f"---\n{rendered}\n---\n\n{body.rstrip()}\n")
    SkillUsageTracker(anima_dir).record(meta.name, SkillUsageEventType.patch, is_common=meta.is_common, notes="trusted_promotion")
    return TrustedPromotionResult(
        skill_name=meta.name,
        path=_pointer_for_path(anima_dir, skill_path),
        trusted_by=trusted_by,
        trusted_at=trusted_at,
        trust_reason=trust_reason,
    )


def _pointer_for_path(anima_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(anima_dir))
    except ValueError:
        return str(path)


__all__ = ["TrustedPromotionResult", "promote_skill_to_trusted"]
