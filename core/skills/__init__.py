from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Skill metadata package — Hermes-compatible loader and index.

Re-exports for convenience::

    from core.skills import SkillMetadata, SkillTrustLevel, SkillIndex
    from core.skills import load_skill_metadata, load_skill_document
"""

from core.skills.activation import (
    ActiveSkillAttachment,
    ActiveSkillContextResult,
    ActiveSkillItem,
    ActiveSkillRejection,
    ActiveSkillResolution,
    ActiveSkillWarning,
    build_active_skill_context,
    get_active_skill_refs,
    get_active_skill_state,
    list_skill_catalog,
    set_active_skill_refs,
    validate_thread_id,
)
from core.skills.curator import (
    CuratorReplay,
    DuplicateCandidate,
    LifecycleSuggestion,
    SkillCurator,
)
from core.skills.guard import SkillScanner
from core.skills.hub import SkillHub, SkillHubResult
from core.skills.index import SkillIndex
from core.skills.loader import (
    is_skill_blocked,
    is_skill_loadable,
    load_skill_body,
    load_skill_document,
    load_skill_metadata,
    skill_access_decision,
)
from core.skills.models import (
    ScanFinding,
    ScanResult,
    SkillBundle,
    SkillCuratorEvent,
    SkillCuratorEventType,
    SkillHubLockEntry,
    SkillLifecycleState,
    SkillMetadata,
    SkillPolicyMetadata,
    SkillRiskMetadata,
    SkillRoutingMetadata,
    SkillScanVerdict,
    SkillSecurityScan,
    SkillSource,
    SkillTrustLevel,
    SkillUsageEvent,
    SkillUsageEventType,
    SkillUsageStats,
    ThreatPattern,
)
from core.skills.policy import SkillActivationPolicy, policy_for_skill
from core.skills.promotion import (
    ProcedurePromotionCandidate,
    ProcedureToSkillConverter,
    PromotionPolicy,
    SkillPromotionResult,
)
from core.skills.router import SkillRouteCandidate, SkillRouter
from core.skills.trust import TrustedPromotionResult, promote_skill_to_trusted
from core.skills.usage import SkillUsageTracker

__all__ = [
    "ScanFinding",
    "ScanResult",
    "SkillBundle",
    "CuratorReplay",
    "DuplicateCandidate",
    "LifecycleSuggestion",
    "SkillCurator",
    "SkillIndex",
    "SkillCuratorEvent",
    "SkillCuratorEventType",
    "SkillHubLockEntry",
    "SkillLifecycleState",
    "SkillMetadata",
    "SkillPolicyMetadata",
    "SkillActivationPolicy",
    "SkillPromotionResult",
    "SkillRouteCandidate",
    "SkillRouter",
    "SkillRiskMetadata",
    "SkillRoutingMetadata",
    "SkillScanVerdict",
    "SkillScanner",
    "SkillHub",
    "SkillHubResult",
    "SkillSecurityScan",
    "SkillSource",
    "SkillTrustLevel",
    "SkillUsageEvent",
    "SkillUsageEventType",
    "SkillUsageStats",
    "SkillUsageTracker",
    "TrustedPromotionResult",
    "ActiveSkillAttachment",
    "ActiveSkillContextResult",
    "ActiveSkillItem",
    "ActiveSkillRejection",
    "ActiveSkillResolution",
    "ActiveSkillWarning",
    "ThreatPattern",
    "ProcedurePromotionCandidate",
    "ProcedureToSkillConverter",
    "PromotionPolicy",
    "build_active_skill_context",
    "get_active_skill_refs",
    "get_active_skill_state",
    "is_skill_blocked",
    "is_skill_loadable",
    "list_skill_catalog",
    "load_skill_body",
    "load_skill_document",
    "load_skill_metadata",
    "set_active_skill_refs",
    "skill_access_decision",
    "policy_for_skill",
    "promote_skill_to_trusted",
    "validate_thread_id",
]
