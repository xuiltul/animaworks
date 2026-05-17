from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Thread-scoped explicit skill activation for chat sessions."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.skills.index import SkillIndex
from core.skills.loader import load_skill_body, skill_access_decision
from core.skills.models import SkillMetadata, SkillScanVerdict, SkillUsageEventType
from core.skills.usage import SkillUsageTracker
from core.time_utils import now_iso

logger = logging.getLogger(__name__)

ACTIVE_SKILLS_STATE_FILE = "active_skills.json"
DEFAULT_MAX_SKILL_CHARS = 6000
DEFAULT_MAX_TOTAL_CHARS = 12000
_THREAD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,36}$")


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

    @classmethod
    def from_metadata(cls, ref: str, path: str, meta: SkillMetadata, *, active: bool = False) -> ActiveSkillItem:
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
        }


@dataclass(slots=True)
class ActiveSkillAttachment:
    name: str
    path: str
    body: str
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
        if not self.attachments:
            return ""
        sections = ["## Active Skills"]
        for item in self.attachments:
            sections.extend(_render_attachment_lines(item))
        return "\n".join(sections).strip()


def validate_thread_id(thread_id: str) -> str:
    value = str(thread_id or "default").strip() or "default"
    if not _THREAD_ID_PATTERN.match(value):
        raise ValueError(
            f"Invalid thread_id: {value!r}. Must be 1-36 alphanumeric, underscore, or hyphen characters."
        )
    return value


def get_active_skill_refs(anima_dir: Path, thread_id: str = "default") -> list[str]:
    """Return stored active skill refs for an anima thread."""
    thread_id = validate_thread_id(thread_id)
    state = _read_state(anima_dir)
    thread_entry = state.get("threads", {}).get(thread_id, {})
    if isinstance(thread_entry, list):
        refs = thread_entry
    elif isinstance(thread_entry, dict):
        refs = thread_entry.get("refs", [])
    else:
        refs = []
    return _dedupe_refs([str(ref) for ref in refs])


def set_active_skill_refs(
    anima_dir: Path,
    refs: list[str] | None,
    *,
    thread_id: str = "default",
    confirm_risk: bool = False,
) -> ActiveSkillResolution:
    """Validate and persist active skill refs for a chat thread."""
    thread_id = validate_thread_id(thread_id)
    requested = _dedupe_refs(refs or [])
    if not requested:
        _write_thread_refs(anima_dir, thread_id, [])
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
    _write_thread_refs(anima_dir, thread_id, [item.path for item in resolution.accepted])
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

    for ref in _dedupe_refs(refs):
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
    for field, reason in (
        ("destructive", "risk_destructive"),
        ("external_send", "risk_external_send"),
        ("requires_human_approval", "risk_requires_human_approval"),
    ):
        if _risk_flag(meta, field):
            reasons.append(reason)
    return reasons


def _risk_flag(meta: SkillMetadata, field: str) -> bool:
    return bool(getattr(meta.risk, field, False) or getattr(meta.routing.risk, field, False))


def _render_attachment_lines(item: ActiveSkillAttachment) -> list[str]:
    lines = [
        f"### {item.name}",
        f"path: {item.path}",
    ]
    if item.truncated:
        lines.extend(
            [
                "body: omitted",
                f"reason: {item.truncation_reason or 'truncated'}",
                "",
            ]
        )
    else:
        lines.extend(["", item.body.strip(), ""])
    return lines


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


def _dedupe_refs(skill_refs: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for ref in skill_refs:
        value = str(ref).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _state_path(anima_dir: Path) -> Path:
    return anima_dir / "state" / ACTIVE_SKILLS_STATE_FILE


def _read_state(anima_dir: Path) -> dict[str, Any]:
    path = _state_path(anima_dir)
    if not path.is_file():
        return {"version": 1, "threads": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read active skill state from %s; using empty state", path)
        return {"version": 1, "threads": {}}
    if not isinstance(data, dict):
        return {"version": 1, "threads": {}}
    threads = data.get("threads")
    if not isinstance(threads, dict):
        data["threads"] = {}
    data["version"] = 1
    return data


def _write_thread_refs(anima_dir: Path, thread_id: str, refs: list[str]) -> None:
    state = _read_state(anima_dir)
    threads = state.setdefault("threads", {})
    if refs:
        threads[thread_id] = {
            "refs": _dedupe_refs(refs),
            "updated_at": now_iso(),
        }
    else:
        threads.pop(thread_id, None)
    path = _state_path(anima_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


__all__ = [
    "ActiveSkillAttachment", "ActiveSkillContextResult", "ActiveSkillItem",
    "ActiveSkillRejection", "ActiveSkillResolution", "ActiveSkillWarning",
    "build_active_skill_context", "get_active_skill_refs", "get_active_skill_state",
    "list_skill_catalog", "set_active_skill_refs", "validate_thread_id",
]
