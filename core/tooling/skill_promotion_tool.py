from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool handler support for procedure-to-skill promotion."""

import json
import logging
from typing import Any

from core.skills.promotion_approval import (
    SkillPromotionApprovalError,
    run_coroutine_sync,
    verify_skill_promotion_approval,
)
from core.tooling.handler_base import _error_result

logger = logging.getLogger("animaworks.tool_handler")


def handle_promote_procedure_to_skill(handler: Any, args: dict[str, Any]) -> str:
    """Create a quarantine draft or activate it after verified human approval."""
    from core.config.models import load_config
    from core.notification.interactive import get_interaction_router
    from core.skills.promotion import ProcedureToSkillConverter, PromotionPolicy

    action = args.get("action", "draft")
    pcfg = load_config().skills.promotion
    policy = PromotionPolicy(
        success_count_threshold=pcfg.success_count_threshold,
        confidence_threshold=pcfg.confidence_threshold,
        failure_count_max=pcfg.failure_count_max,
        last_used_within_days=pcfg.last_used_within_days,
        auto_activate=pcfg.auto_activate,
        require_approval_on_warn=pcfg.require_approval_on_warn,
    )
    converter = ProcedureToSkillConverter(
        handler._anima_dir,
        policy=policy,
        owner_anima=getattr(handler, "_anima_name", None) or handler._anima_dir.name,
    )

    try:
        if action == "draft":
            result = _draft_skill(handler, args, converter, get_interaction_router())
        elif action == "approve":
            result = _approve_skill(handler, args, converter, get_interaction_router())
            if isinstance(result, str):
                return result
        else:
            return _error_result(
                "InvalidArguments",
                f"Unsupported action: {action}",
                suggestion="Use action='draft' or action='approve'",
            )
    except FileExistsError as exc:
        return _error_result("FileExists", str(exc))
    except FileNotFoundError as exc:
        return _error_result("FileNotFound", str(exc))
    except SkillPromotionApprovalError as exc:
        return _error_result(exc.error_type, str(exc), suggestion=exc.suggestion)
    except ValueError as exc:
        return _error_result("InvalidArguments", str(exc))
    except Exception as exc:
        logger.exception("promote_procedure_to_skill failed")
        return _error_result("PromotionFailed", str(exc))

    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def _draft_skill(handler: Any, args: dict[str, Any], converter: Any, router: Any) -> Any:
    rel_path = args.get("path", "")
    if not rel_path:
        raise ValueError("path is required for action=draft")
    overrides = {
        key: args[key]
        for key in (
            "description",
            "use_when",
            "trigger_phrases",
            "negative_phrases",
            "domains",
            "tags",
            "risk",
        )
        if key in args
    }
    result = converter.create_quarantine_skill(
        rel_path,
        skill_name=args.get("skill_name"),
        metadata_overrides=overrides,
    )
    if result.status != "review":
        return result

    req = run_coroutine_sync(
        router.create(
            handler._anima_name,
            "skill_promotion",
            ["approve", "reject", "comment"],
            metadata={
                "kind": "skill_promotion",
                "skill_name": result.skill_name,
                "procedure_path": result.procedure_path,
                "quarantine_path": result.quarantine_path,
                "scan_verdict": result.scan_verdict,
            },
        )
    )
    converter.register_approval_request(result.skill_name, req.callback_id)
    result.approval_callback_id = req.callback_id
    _notify_skill_promotion_approval(handler, result.to_dict(), req)
    return result


def _approve_skill(handler: Any, args: dict[str, Any], converter: Any, router: Any) -> Any:
    skill_name = args.get("skill_name", "")
    if not skill_name:
        return _error_result("InvalidArguments", "skill_name is required for action=approve")
    callback_id = args.get("approval_callback_id", "")
    if not callback_id:
        return _error_result("InvalidArguments", "approval_callback_id is required for action=approve")
    approval = _verify_skill_promotion_approval(
        handler,
        router,
        callback_id=callback_id,
        skill_name=skill_name,
    )
    if isinstance(approval, str):
        return approval
    return converter.approve_skill(
        skill_name,
        approved_by=approval["actor"],
        approval_callback_id=callback_id,
    )


def _verify_skill_promotion_approval(
    handler: Any,
    router: Any,
    *,
    callback_id: str,
    skill_name: str,
) -> dict[str, str] | str:
    del router
    try:
        approval = verify_skill_promotion_approval(
            callback_id,
            owner_anima=handler._anima_name,
            skill_name=skill_name,
        )
    except SkillPromotionApprovalError as exc:
        return _error_result(
            exc.error_type,
            str(exc),
            suggestion=exc.suggestion,
        )
    return {"actor": approval.actor, "source": approval.source}


def _notify_skill_promotion_approval(handler: Any, result: dict[str, Any], interaction_req: Any) -> None:
    if handler._human_notifier is None:
        return
    subject = f"Approve skill promotion: {result['skill_name']}"
    body = (
        f"Procedure: {result.get('procedure_path')}\n"
        f"Draft: {result.get('quarantine_path')}\n"
        f"Scan verdict: {result.get('scan_verdict')}\n"
        f"Callback ID: {result.get('approval_callback_id')}"
    )
    try:
        run_coroutine_sync(
            handler._human_notifier.notify(
                subject,
                body,
                "normal",
                anima_name=handler._anima_name,
                interaction=interaction_req,
            )
        )
    except Exception:
        logger.debug("Failed to send skill promotion approval notification", exc_info=True)
