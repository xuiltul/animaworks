from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Trust-aware policy for skill activation and prompt rendering."""

from typing import Literal

from pydantic import BaseModel

from core.skills.curator import is_unloadable_lifecycle_state
from core.skills.models import SkillMetadata, SkillScanVerdict, SkillTrustLevel


class SkillActivationPolicy(BaseModel):
    """How a routed or active skill may be presented to an anima."""

    use_mode: Literal["primary_guidance", "candidate_hint"] = "candidate_hint"
    injection: Literal["body_allowed", "pointer_preferred", "blocked"] = "pointer_preferred"
    trust_weight: float = 0.0
    render_section: Literal["trusted", "candidate", "blocked"] = "candidate"
    reason: str = "candidate_hint"
    blocked: bool = False


def policy_for_skill(meta: SkillMetadata) -> SkillActivationPolicy:
    """Return the trust policy for a skill without changing relevance scoring."""
    if meta.security.verdict == SkillScanVerdict.dangerous:
        return _blocked("security_dangerous")
    if is_unloadable_lifecycle_state(meta.lifecycle_state):
        return _blocked(f"curator_{meta.lifecycle_state.value}")
    if meta.trust_level in (SkillTrustLevel.blocked, SkillTrustLevel.quarantine):
        return _blocked(f"trust_level_{meta.trust_level.value}")
    if meta.skill_policy.injection == "blocked":
        return _blocked("skill_policy_blocked")

    promotion_status = str(meta.promotion_status or "").strip().lower()
    if promotion_status == "probation" or meta.skill_policy.use_mode == "candidate_hint":
        return _candidate("probation_candidate", trust_weight=0.55)

    if promotion_status == "trusted" or meta.trust_level in (
        SkillTrustLevel.builtin,
        SkillTrustLevel.official,
        SkillTrustLevel.trusted,
    ):
        return SkillActivationPolicy(
            use_mode="primary_guidance",
            injection="body_allowed",
            trust_weight=1.0,
            render_section="trusted",
            reason="trusted_guidance",
        )

    if meta.trust_level == SkillTrustLevel.community:
        return _candidate("community_candidate", trust_weight=0.35)
    if meta.trust_level == SkillTrustLevel.untrusted:
        return _candidate("untrusted_candidate", trust_weight=0.2)
    return _candidate("candidate_hint", trust_weight=0.5)


def _candidate(reason: str, *, trust_weight: float) -> SkillActivationPolicy:
    return SkillActivationPolicy(
        use_mode="candidate_hint",
        injection="pointer_preferred",
        trust_weight=trust_weight,
        render_section="candidate",
        reason=reason,
    )


def _blocked(reason: str) -> SkillActivationPolicy:
    return SkillActivationPolicy(
        use_mode="candidate_hint",
        injection="blocked",
        trust_weight=0.0,
        render_section="blocked",
        reason=reason,
        blocked=True,
    )


__all__ = ["SkillActivationPolicy", "policy_for_skill"]
