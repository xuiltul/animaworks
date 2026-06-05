from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Section assembly and budget allocation for prompt building."""

import re
from dataclasses import dataclass

# ── Budget-based prompt scaling ───────────────────────────────
_REFERENCE_WINDOW = 128_000
_TOOL_RESERVATION_PCT = 0.15
_OUTPUT_RESERVATION_PCT = 0.10
_CONVERSATION_RESERVATION_PCT = 0.10
_MIN_SYSTEM_BUDGET = 2000

_GROUP_HEADER_RE = re.compile(r"^group(\d+)_header$")


@dataclass
class SectionEntry:
    """A prompt section with budget allocation metadata."""

    id: str
    priority: int  # 1=mandatory, 2=important, 3=nice-to-have, 4=optional
    kind: str  # "rigid" or "elastic"
    content: str


def _normalize_headings(content: str) -> str:
    """Shift H1 headings (``# text``) to H2 (``## text``).

    Preserves headings inside fenced code blocks (````` ``).
    Only H1 is shifted; H2+ remain unchanged.
    """
    lines = content.split("\n")
    result: list[str] = []
    in_code_block = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
        if not in_code_block and stripped.startswith("# ") and not stripped.startswith("## "):
            leading = line[: len(line) - len(stripped)]
            line = leading + "#" + stripped
        result.append(line)
    return "\n".join(result)


def _assemble_with_tags(allocated: list[SectionEntry]) -> str:
    """Join allocated sections using XML group/section boundary tags.

    Group-header sections (id matching ``groupN_header``) open/close
    ``<group_N>`` tags.  All other sections are wrapped in
    ``<section name="...">`` tags with heading normalization applied.
    """
    parts: list[str] = []
    current_group: str | None = None

    for s in allocated:
        m = _GROUP_HEADER_RE.match(s.id)
        if m:
            if current_group is not None:
                parts.append(f"</group_{current_group}>")
            current_group = m.group(1)
            title = s.content.strip()
            parts.append(f'<group_{current_group} title="{title}">')
        else:
            body = _normalize_headings(s.content)
            parts.append(f'<section name="{s.id}">\n{body}\n</section>')

    if current_group is not None:
        parts.append(f"</group_{current_group}>")

    return "\n\n".join(parts)


def _compute_system_budget(context_window: int, system_budget: int | None = None) -> int:
    """Compute the character budget for the system prompt.

    Reserves portions of the context window for tools, output, and conversation,
    then returns the remaining space as the system prompt budget.
    """
    usable = int(
        context_window * (1.0 - _TOOL_RESERVATION_PCT - _OUTPUT_RESERVATION_PCT - _CONVERSATION_RESERVATION_PCT)
    )
    auto = max(usable, _MIN_SYSTEM_BUDGET)
    if system_budget is not None:
        return max(min(system_budget, auto), _MIN_SYSTEM_BUDGET)
    return auto


def _allocate_sections(sections: list[SectionEntry], budget: int) -> list[SectionEntry]:
    """Apply Rigid/Elastic budget allocation preserving original order.

    Rigid sections are included entirely or excluded entirely (all-or-nothing).
    Priority-1 rigid sections are always included regardless of budget.
    Elastic sections share the remaining budget proportionally.
    """
    included_rigid: set[int] = set()
    remaining = budget

    for priority in range(1, 5):
        for i, s in enumerate(sections):
            if s.kind != "rigid" or s.priority != priority:
                continue
            cost = len(s.content)
            if s.priority == 1 or cost <= remaining:
                included_rigid.add(i)
                remaining -= cost

    elastic_indices = [i for i, s in enumerate(sections) if s.kind == "elastic"]
    included_elastic: dict[int, str] = {}

    if remaining > 0 and elastic_indices:
        total_elastic = sum(len(sections[i].content) for i in elastic_indices)
        if total_elastic <= remaining:
            for i in elastic_indices:
                included_elastic[i] = sections[i].content
        elif total_elastic > 0:
            ratio = remaining / total_elastic
            for i in elastic_indices:
                allowed = int(len(sections[i].content) * ratio)
                if allowed > 100:
                    included_elastic[i] = sections[i].content[:allowed]

    result: list[SectionEntry] = []
    for i, s in enumerate(sections):
        if i in included_rigid:
            result.append(s)
        elif i in included_elastic:
            result.append(SectionEntry(id=s.id, priority=s.priority, kind=s.kind, content=included_elastic[i]))

    return result
