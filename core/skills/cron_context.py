from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Skill context resolution for cron execution."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.config.models import SkillCronConfig, load_config
from core.skills.index import SkillIndex
from core.skills.loader import load_skill_body, skill_access_decision
from core.skills.models import SkillMetadata, SkillScanVerdict, SkillUsageEventType
from core.skills.usage import SkillUsageTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SkillContextAttachment:
    name: str
    path: str
    body: str
    truncated: bool = False
    truncation_reason: str | None = None


@dataclass(slots=True)
class SkillContextRejection:
    ref: str
    reason: str


@dataclass(slots=True)
class SkillContextWarning:
    ref: str
    reason: str
    name: str
    path: str


@dataclass(slots=True)
class SkillContextResult:
    attachments: list[SkillContextAttachment]
    rejections: list[SkillContextRejection]
    warnings: list[SkillContextWarning] = field(default_factory=list)

    def render(self) -> str:
        if not self.attachments and not self.rejections:
            return ""
        sections: list[str] = []
        if self.attachments:
            sections.append("## Cron Skills")
            for item in self.attachments:
                sections.extend(_render_attachment_lines(item))
        if self.rejections:
            sections.append("## Rejected Cron Skills")
            sections.extend(f"- {item.ref}: {item.reason}" for item in self.rejections)
        return "\n".join(sections).strip()


def build_cron_skill_context(anima_dir: Path, skill_refs: list[str] | None) -> SkillContextResult:
    """Resolve cron ``skills:`` references into safe prompt attachments.

    Resolution and safety are intentionally centralized here so scheduler entry
    points only pass skill refs through to ``DigitalAnima.run_cron_task``.
    """
    refs = _dedupe_refs(skill_refs or [])
    if not refs:
        return SkillContextResult([], [])

    from core.paths import get_common_skills_dir

    common_skills_dir = get_common_skills_dir()
    index = SkillIndex(
        anima_dir / "skills",
        common_skills_dir,
        anima_dir / "procedures",
        anima_dir=anima_dir,
    )
    cron_config = _load_cron_config()
    attachments: list[SkillContextAttachment] = []
    rejections: list[SkillContextRejection] = []
    warnings: list[SkillContextWarning] = []

    for ref in refs:
        meta = index.resolve_skill_reference(ref)
        if meta is None or meta.path is None:
            rejections.append(SkillContextRejection(ref, "not_found"))
            continue

        allowed, reason = _cron_access_decision(meta, anima_dir, cron_config)
        if not allowed:
            rejections.append(SkillContextRejection(ref, reason))
            continue

        try:
            body = load_skill_body(meta.path)
        except Exception:
            rejections.append(SkillContextRejection(ref, "read_failed"))
            continue

        pointer = _pointer_for_path(anima_dir, common_skills_dir, meta.path)
        body_chars = len(body.strip())
        full_attachment = SkillContextAttachment(meta.name, pointer, body)
        truncated_reason = _truncation_reason(body_chars, attachments, full_attachment, cron_config)
        if truncated_reason is not None:
            attachments.append(
                SkillContextAttachment(
                    meta.name,
                    pointer,
                    "",
                    truncated=True,
                    truncation_reason=truncated_reason,
                )
            )
        else:
            attachments.append(full_attachment)
        warning_reason = _allowed_warning_reason(meta, cron_config)
        if warning_reason is not None:
            warnings.append(SkillContextWarning(ref, warning_reason, meta.name, pointer))
        _record_cron_usage(anima_dir, meta)

    return SkillContextResult(attachments, rejections, warnings)


def _load_cron_config() -> SkillCronConfig:
    try:
        return load_config().skills.cron
    except Exception:
        logger.debug("Failed to load skills.cron config; using defaults", exc_info=True)
        return SkillCronConfig()


def _cron_access_decision(meta: SkillMetadata, anima_dir: Path, cron_config: SkillCronConfig) -> tuple[bool, str]:
    allowed, reason = skill_access_decision(meta, anima_dir=anima_dir)
    if not allowed:
        return False, reason

    verdict = meta.security.verdict
    if verdict in (SkillScanVerdict.warn, SkillScanVerdict.caution) and not cron_config.allow_warn_caution:
        return False, f"security_{verdict.value}"

    if _risk_flag(meta, "destructive") and not cron_config.allow_destructive:
        return False, "risk_destructive"
    if _risk_flag(meta, "external_send") and not cron_config.allow_external_send:
        return False, "risk_external_send"

    return True, "allowed"


def _risk_flag(meta: SkillMetadata, field: str) -> bool:
    return bool(getattr(meta.risk, field, False) or getattr(meta.routing.risk, field, False))


def _allowed_warning_reason(meta: SkillMetadata, cron_config: SkillCronConfig) -> str | None:
    verdict = meta.security.verdict
    if verdict in (SkillScanVerdict.warn, SkillScanVerdict.caution) and cron_config.allow_warn_caution:
        return f"security_{verdict.value}_allowed"
    if _risk_flag(meta, "destructive") and cron_config.allow_destructive:
        return "risk_destructive_allowed"
    if _risk_flag(meta, "external_send") and cron_config.allow_external_send:
        return "risk_external_send_allowed"
    return None


def _truncation_reason(
    body_chars: int,
    attachments: list[SkillContextAttachment],
    full_attachment: SkillContextAttachment,
    cron_config: SkillCronConfig,
) -> str | None:
    if body_chars > cron_config.max_skill_chars:
        return "max_skill_chars_exceeded"
    if _rendered_attachment_chars([*attachments, full_attachment]) > cron_config.max_total_chars:
        return "max_total_chars_exceeded"
    return None


def _render_attachment_lines(item: SkillContextAttachment) -> list[str]:
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


def _rendered_attachment_chars(attachments: list[SkillContextAttachment]) -> int:
    return len(SkillContextResult(attachments, []).render())


def _record_cron_usage(anima_dir: Path, meta: SkillMetadata) -> None:
    try:
        from core.skills.usage import usage_ref_from_path

        SkillUsageTracker(anima_dir).record(
            meta.name,
            SkillUsageEventType.use,
            is_common=meta.is_common,
            ref=usage_ref_from_path(meta.path, name=meta.name, is_common=meta.is_common),
            notes="cron",
        )
    except Exception:
        logger.debug("Failed to record cron skill usage for %s", meta.name, exc_info=True)


def _pointer_for_path(anima_dir: Path, common_skills_dir: Path, path: Path) -> str:
    from core.company_resources import company_resource_pointer

    company_pointer = company_resource_pointer(path)
    if company_pointer is not None:
        return company_pointer
    try:
        return str(path.relative_to(anima_dir))
    except ValueError:
        pass
    try:
        return str(path.relative_to(common_skills_dir.parent))
    except ValueError:
        return str(path)


def _dedupe_refs(skill_refs: list[str]) -> list[str]:
    result: list[str] = []
    for ref in skill_refs:
        value = str(ref).strip()
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "SkillContextAttachment",
    "SkillContextRejection",
    "SkillContextResult",
    "SkillContextWarning",
    "build_cron_skill_context",
]
