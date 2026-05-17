from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified approval lookup for procedure-to-skill promotion."""

from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VerifiedSkillPromotionApproval:
    """Resolved approval details trusted by the promotion pipeline."""

    actor: str
    source: str
    decision: str


class SkillPromotionApprovalError(ValueError):
    """Base class for approval-gate failures."""

    error_type = "ApprovalError"
    suggestion: str | None = None


class SkillPromotionApprovalRequiredError(SkillPromotionApprovalError):
    error_type = "ApprovalRequired"
    suggestion = "Wait for the interactive approval to be resolved, then retry with approval_callback_id"


class SkillPromotionApprovalMismatchError(SkillPromotionApprovalError):
    error_type = "ApprovalMismatch"


class SkillPromotionApprovalRejectedError(SkillPromotionApprovalError):
    error_type = "ApprovalRejected"


def run_coroutine_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result(timeout=60)


def verify_skill_promotion_approval(
    callback_id: str,
    *,
    owner_anima: str,
    skill_name: str,
) -> VerifiedSkillPromotionApproval:
    """Return a resolved human approval or raise an approval-gate error."""
    if not callback_id:
        raise SkillPromotionApprovalRequiredError("approval_callback_id is required")

    from core.notification.interactive import get_interaction_router

    resolved = run_coroutine_sync(get_interaction_router().lookup_resolved(callback_id))
    if resolved is None:
        raise SkillPromotionApprovalRequiredError("Human approval has not been resolved for this skill promotion")

    req, approval_result = resolved
    expected_name = str(req.metadata.get("skill_name") or "")
    if req.category != "skill_promotion" or req.anima_name != owner_anima or expected_name != skill_name:
        raise SkillPromotionApprovalMismatchError("approval_callback_id does not match this skill promotion")
    if approval_result.decision.lower() not in {"approve", "approved", "yes"}:
        raise SkillPromotionApprovalRejectedError(f"Skill promotion was not approved: {approval_result.decision}")
    return VerifiedSkillPromotionApproval(
        actor=approval_result.actor,
        source=approval_result.source,
        decision=approval_result.decision,
    )
