from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from core.skills.models import SkillLifecycleState, SkillMetadata, SkillScanVerdict, SkillTrustLevel
from core.skills.policy import policy_for_skill


def test_trusted_skill_policy_allows_body_guidance() -> None:
    policy = policy_for_skill(SkillMetadata(name="trusted", trust_level=SkillTrustLevel.trusted))
    assert policy.use_mode == "primary_guidance"
    assert policy.injection == "body_allowed"
    assert policy.render_section == "trusted"
    assert policy.trust_weight == 1.0


def test_probation_skill_policy_is_candidate_hint() -> None:
    policy = policy_for_skill(
        SkillMetadata(
            name="auto",
            trust_level=SkillTrustLevel.community,
            promotion_status="probation",
            skill_policy={"use_mode": "candidate_hint", "injection": "pointer_preferred"},
        )
    )
    assert policy.use_mode == "candidate_hint"
    assert policy.injection == "pointer_preferred"
    assert policy.render_section == "candidate"
    assert policy.trust_weight == 0.55


def test_archived_or_dangerous_skill_policy_is_blocked() -> None:
    archived = policy_for_skill(SkillMetadata(name="old", lifecycle_state=SkillLifecycleState.archived))
    dangerous = policy_for_skill(SkillMetadata(name="bad", security={"verdict": SkillScanVerdict.dangerous}))
    assert archived.blocked is True
    assert dangerous.blocked is True
