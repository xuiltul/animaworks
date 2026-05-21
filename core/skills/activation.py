from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Thread-scoped explicit skill activation for chat sessions."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.skills.activation_render import render_active_skill_context
from core.skills.activation_state import dedupe_refs, get_active_skill_refs, validate_thread_id, write_thread_refs
from core.skills.index import SkillIndex
from core.skills.loader import load_skill_body, skill_access_decision
from core.skills.models import SkillMetadata, SkillScanVerdict, SkillUsageEventType
from core.skills.policy import SkillActivationPolicy, policy_for_skill
from core.skills.usage import SkillUsageTracker

logger = logging.getLogger(__name__)

DEFAULT_MAX_SKILL_CHARS = 6000
DEFAULT_MAX_TOTAL_CHARS = 12000


@dataclass(slots=True)
class ActiveSkillItem:
    ref: str
    name: str
    path: str
    description: str = ""
    is_common: bool = False
    is_procedure: bool = False
    trust_level: str = ""
    security_verdict: str = ""
    active: bool = False
    activation_policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, ref: str, path: str, meta: SkillMetadata, *, active: bool = False) -> ActiveSkillItem:
        policy = policy_for_skill(meta)
        return cls(
            ref=ref,
            name=meta.name,
            path=path,
            description=meta.description,
            is_common=meta.is_common,
            is_procedure=meta.is_procedure,
            trust_level=str(meta.trust_level.value),
            security_verdict=str(meta.security.verdict.value),
            active=active,
            activation_policy=policy.model_dump(mode="json"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "is_common": self.is_common,
            "is_procedure": self.is_procedure,
            "trust_level": self.trust_level,
            "security_verdict": self.security_verdict,
            "active": self.active,
            "activation_policy": self.activation_policy,
        }


@dataclass(slots=True)
class ActiveSkillAttachment:
    name: str
    path: str
    body: str
    description: str = ""
    policy: SkillActivationPolicy = field(default_factory=SkillActivationPolicy)
    is_common: bool = False
    truncated: bool = False
    truncation_reason: str | None = None


@dataclass(slots=True)
class ActiveSkillRejection:
    ref: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"ref": self.ref, "reason": self.reason}


@dataclass(slots=True)
class ActiveSkillWarning:
    ref: str
    reason: str
    name: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"ref": self.ref, "reason": self.reason, "name": self.name, "path": self.path}


@dataclass(slots=True)
class ActiveSkillResolution:
    accepted: list[ActiveSkillItem] = field(default_factory=list)
    rejections: list[ActiveSkillRejection] = field(default_factory=list)
    warnings: list[ActiveSkillWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [item.to_dict() for item in self.accepted],
            "rejections": [item.to_dict() for item in self.rejections],
            "warnings": [item.to_dict() for item in self.warnings],
        }


@dataclass(slots=True)
class ActiveSkillContextResult:
    attachments: list[ActiveSkillAttachment] = field(default_factory=list)
    rejections: list[ActiveSkillRejection] = field(default_factory=list)
    warnings: list[ActiveSkillWarning] = field(default_factory=list)

    def render(self) -> str:
        return render_active_skill_context(self.attachments)


def set_active_skill_refs(
    anima_dir: Path,
    refs: list[str] | None,
    *,
    thread_id: str = "default",
    confirm_risk: bool = False,
) -> ActiveSkillResolution:
    """Validate and persist active skill refs for a chat thread."""
    thread_id = validate_thread_id(thread_id)
    requested = dedupe_refs(refs or [])
    if not requested:
        write_thread_refs(anima_dir, thread_id, [])
        return ActiveSkillResolution()

    common_skills_dir = _get_common_skills_dir()
    index = _build_index(anima_dir, common_skills_dir)
    resolution = _resolve_refs(
        anima_dir,
        common_skills_dir,
        index,
        requested,
        confirm_risk=confirm_risk,
    )
    write_thread_refs(anima_dir, thread_id, [item.path for item in resolution.accepted])
    return resolution


def get_active_skill_state(anima_dir: Path, *, thread_id: str = "default") -> ActiveSkillResolution:
    """Resolve currently stored active skill refs for API reads."""
    refs = get_active_skill_refs(anima_dir, thread_id)
    if not refs:
        return ActiveSkillResolution()
    common_skills_dir = _get_common_skills_dir()
    return _resolve_refs(
        anima_dir,
        common_skills_dir,
        _build_index(anima_dir, common_skills_dir),
        refs,
        confirm_risk=True,
    )


def list_skill_catalog(anima_dir: Path, *, thread_id: str = "default") -> list[dict[str, Any]]:
    """List visible skills with thread-local active status."""
    validate_thread_id(thread_id)
    common_skills_dir = _get_common_skills_dir()
    index = _build_index(anima_dir, common_skills_dir)
    active_paths = {item.path for item in get_active_skill_state(anima_dir, thread_id=thread_id).accepted}

    result: list[dict[str, Any]] = []
    for meta in index.all_skills:
        if meta.path is None:
            continue
        pointer = _pointer_for_path(anima_dir, common_skills_dir, meta.path)
        item = ActiveSkillItem.from_metadata(pointer, pointer, meta, active=pointer in active_paths)
        result.append(item.to_dict())
    return result


def build_active_skill_context(
    anima_dir: Path,
    *,
    thread_id: str = "default",
    record_usage: bool = True,
    max_skill_chars: int = DEFAULT_MAX_SKILL_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> ActiveSkillContextResult:
    """Render thread-scoped active skill bodies for chat system prompts."""
    refs = get_active_skill_refs(anima_dir, thread_id)
    if not refs:
        return ActiveSkillContextResult()

    common_skills_dir = _get_common_skills_dir()
    index = _build_index(anima_dir, common_skills_dir)
    resolution = _resolve_refs(
        anima_dir,
        common_skills_dir,
        index,
        refs,
        confirm_risk=True,
    )
    attachments: list[ActiveSkillAttachment] = []

    for item in resolution.accepted:
        meta = index.resolve_skill_reference(item.path)
        if meta is None or meta.path is None:
            continue
        policy = policy_for_skill(meta)
        if policy.blocked:
            resolution.rejections.append(ActiveSkillRejection(item.path, policy.reason))
            continue
        if policy.injection == "pointer_preferred":
            attachments.append(
                ActiveSkillAttachment(
                    name=meta.name,
                    path=item.path,
                    body="",
                    description=meta.description,
                    policy=policy,
                    is_common=meta.is_common,
                )
            )
            if record_usage:
                _record_active_usage(anima_dir, meta)
            continue
        try:
            body = load_skill_body(meta.path)
        except Exception:
            resolution.rejections.append(ActiveSkillRejection(item.path, "read_failed"))
            continue

        body_chars = len(body.strip())
        full_attachment = ActiveSkillAttachment(
            name=meta.name,
            path=item.path,
            body=body,
            description=meta.description,
            policy=policy,
            is_common=meta.is_common,
        )
        truncated_reason = _truncation_reason(
            body_chars,
            attachments,
            full_attachment,
            max_skill_chars,
            max_total_chars,
        )
        if truncated_reason is not None:
            attachments.append(
                ActiveSkillAttachment(
                    name=meta.name,
                    path=item.path,
                    body="",
                    description=meta.description,
                    policy=policy,
                    is_common=meta.is_common,
                    truncated=True,
                    truncation_reason=truncated_reason,
                )
            )
        else:
            attachments.append(full_attachment)

        if record_usage:
            _record_active_usage(anima_dir, meta)

    return ActiveSkillContextResult(
        attachments=attachments,
        rejections=resolution.rejections,
        warnings=resolution.warnings,
    )


def _resolve_refs(
    anima_dir: Path,
    common_skills_dir: Path,
    index: SkillIndex,
    refs: list[str],
    *,
    confirm_risk: bool,
) -> ActiveSkillResolution:
    accepted: list[ActiveSkillItem] = []
    rejections: list[ActiveSkillRejection] = []
    warnings: list[ActiveSkillWarning] = []
    seen_paths: set[str] = set()

    for ref in dedupe_refs(refs):
        meta = index.resolve_skill_reference(ref)
        if meta is None or meta.path is None:
            rejections.append(ActiveSkillRejection(ref, "not_found"))
            continue

        allowed, reason = _active_access_decision(meta, anima_dir, confirm_risk=confirm_risk)
        pointer = _pointer_for_path(anima_dir, common_skills_dir, meta.path)
        if not allowed:
            rejections.append(ActiveSkillRejection(ref, reason))
            continue
        if pointer in seen_paths:
            continue
        seen_paths.add(pointer)
        item = ActiveSkillItem.from_metadata(ref, pointer, meta, active=True)
        accepted.append(item)
        warnings.extend(_allowed_warnings(ref, meta, pointer, confirm_risk=confirm_risk))

    return ActiveSkillResolution(accepted=accepted, rejections=rejections, warnings=warnings)


def _active_access_decision(meta: SkillMetadata, anima_dir: Path, *, confirm_risk: bool) -> tuple[bool, str]:
    policy = policy_for_skill(meta)
    if policy.blocked:
        return False, policy.reason
    allowed, reason = skill_access_decision(meta, anima_dir=anima_dir)
    if not allowed:
        return False, reason

    verdict = meta.security.verdict
    if verdict in (SkillScanVerdict.warn, SkillScanVerdict.caution) and not confirm_risk:
        return False, f"security_{verdict.value}"

    risk_reason = _first_risk_reason(meta)
    if risk_reason is not None and not confirm_risk:
        return False, risk_reason

    return True, "allowed"


def _allowed_warnings(ref: str, meta: SkillMetadata, pointer: str, *, confirm_risk: bool) -> list[ActiveSkillWarning]:
    if not confirm_risk:
        return []
    result: list[ActiveSkillWarning] = []
    verdict = meta.security.verdict
    if verdict in (SkillScanVerdict.warn, SkillScanVerdict.caution):
        result.append(ActiveSkillWarning(ref, f"security_{verdict.value}_allowed", meta.name, pointer))
    for reason in _risk_reasons(meta):
        result.append(ActiveSkillWarning(ref, f"{reason}_allowed", meta.name, pointer))
    return result


def _first_risk_reason(meta: SkillMetadata) -> str | None:
    reasons = _risk_reasons(meta)
    return reasons[0] if reasons else None


def _risk_reasons(meta: SkillMetadata) -> list[str]:
    reasons: list[str] = []
    for risk_field, reason in (
        ("destructive", "risk_destructive"),
        ("external_send", "risk_external_send"),
        ("handles_untrusted_data", "risk_handles_untrusted_data"),
        ("credential", "risk_credential"),
        ("production", "risk_production"),
        ("billing", "risk_billing"),
        ("private_data", "risk_private_data"),
        ("open_world", "risk_open_world"),
        ("requires_human_approval", "risk_requires_human_approval"),
    ):
        if _risk_flag(meta, risk_field):
            reasons.append(reason)
    return reasons


def _risk_flag(meta: SkillMetadata, field: str) -> bool:
    return bool(getattr(meta.risk, field, False) or getattr(meta.routing.risk, field, False))


def _truncation_reason(
    body_chars: int,
    attachments: list[ActiveSkillAttachment],
    full_attachment: ActiveSkillAttachment,
    max_skill_chars: int,
    max_total_chars: int,
) -> str | None:
    if body_chars > max_skill_chars:
        return "max_skill_chars_exceeded"
    if _rendered_attachment_chars([*attachments, full_attachment]) > max_total_chars:
        return "max_total_chars_exceeded"
    return None


def _rendered_attachment_chars(attachments: list[ActiveSkillAttachment]) -> int:
    return len(ActiveSkillContextResult(attachments=attachments).render())


def _record_active_usage(anima_dir: Path, meta: SkillMetadata) -> None:
    try:
        SkillUsageTracker(anima_dir).record(
            meta.name,
            SkillUsageEventType.use,
            is_common=meta.is_common,
            notes="active_skill",
        )
    except Exception:
        logger.debug("Failed to record active skill usage for %s", meta.name, exc_info=True)


def _build_index(anima_dir: Path, common_skills_dir: Path) -> SkillIndex:
    return SkillIndex(
        anima_dir / "skills",
        common_skills_dir,
        anima_dir / "procedures",
        anima_dir=anima_dir,
    )


def _pointer_for_path(anima_dir: Path, common_skills_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(anima_dir))
    except ValueError:
        pass
    try:
        return str(path.relative_to(common_skills_dir.parent))
    except ValueError:
        return str(path)


def _get_common_skills_dir() -> Path:
    from core.paths import get_common_skills_dir

    return get_common_skills_dir()


__all__ = [
    "ActiveSkillAttachment",
    "ActiveSkillContextResult",
    "ActiveSkillItem",
    "ActiveSkillRejection",
    "ActiveSkillResolution",
    "ActiveSkillWarning",
    "build_active_skill_context",
    "get_active_skill_refs",
    "get_active_skill_state",
    "list_skill_catalog",
    "set_active_skill_refs",
    "validate_thread_id",
]
