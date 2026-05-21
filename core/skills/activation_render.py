from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt rendering helpers for explicitly activated skills."""

from typing import Any


def render_active_skill_context(attachments: list[Any]) -> str:
    if not attachments:
        return ""
    sections: list[str] = []
    trusted = [item for item in attachments if item.policy.render_section == "trusted"]
    candidates = [item for item in attachments if item.policy.render_section == "candidate"]
    if trusted:
        sections.append("## Trusted Skills")
        for item in trusted:
            sections.extend(_render_attachment_lines(item))
    if candidates:
        if sections:
            sections.append("")
        sections.append("## Candidate Skill Hints")
        for item in candidates:
            sections.extend(_render_candidate_hint_lines(item))
    return "\n".join(sections).strip()


def _render_attachment_lines(item: Any) -> list[str]:
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


def _render_candidate_hint_lines(item: Any) -> list[str]:
    summary = item.description.strip() or "(no description)"
    return [
        f"- {item.name}",
        f"  path: {item.path}",
        f"  summary: {summary}",
        "  policy: Candidate hint only. Verify against the current task before relying on it.",
    ]
