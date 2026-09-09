from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Section assembly and token-budget allocation for prompt building."""

import logging
import re
from dataclasses import dataclass, replace
from typing import Literal

from core.prompt.tokens import estimate_tokens

logger = logging.getLogger("animaworks.prompt_builder")

# ── Budget-based prompt scaling ──────────────────────
_REFERENCE_WINDOW = 128_000
_MIN_SYSTEM_BUDGET = 2000

_GROUP_HEADER_RE = re.compile(r"^group(\d+)_header$")
_PRIMING_BLOCK_RE = re.compile(r"<priming\b[^>]*>.*?</priming>", re.DOTALL | re.IGNORECASE)
_PRIMING_OPEN_RE = re.compile(r"<priming\b[^>]*>", re.IGNORECASE)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class PromptBudget:
    """Normal operating target and hard ceiling, both measured in tokens."""

    target: int
    ceiling: int


@dataclass
class SectionEntry:
    """A prompt section with budget allocation metadata."""

    id: str
    priority: int  # 1=mandatory, 2=important, 3=nice-to-have, 4=optional
    kind: str  # "rigid" or "elastic"
    content: str
    items: tuple[str, ...] | None = None
    trim_from: Literal["head", "tail"] = "tail"
    budget_group: Literal["framework", "recall"] = "framework"


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
    ``<group_N>`` tags. All other sections are wrapped in
    ``<section name="...">`` tags with heading normalization applied.
    """
    parts: list[str] = []
    current_group: str | None = None

    for section in allocated:
        match = _GROUP_HEADER_RE.match(section.id)
        if match:
            if current_group is not None:
                parts.append(f"</group_{current_group}>")
            current_group = match.group(1)
            title = section.content.strip()
            parts.append(f'<group_{current_group} title="{title}">')
        else:
            body = _normalize_headings(section.content)
            parts.append(f'<section name="{section.id}">\n{body}\n</section>')

    if current_group is not None:
        parts.append(f"</group_{current_group}>")

    return "\n\n".join(parts)


def _compute_system_budget(context_window: int, system_budget: int | None = None) -> PromptBudget:
    """Compute the normal target and hard ceiling in tokens."""
    target_tokens = 6_000
    ceiling_pct = 0.35
    try:
        from core.config import load_config

        prompt_config = load_config().prompt
        target_tokens = int(getattr(prompt_config, "system_prompt_target_tokens", target_tokens))
        ceiling_pct = float(getattr(prompt_config, "system_prompt_ceiling_pct", ceiling_pct))
    except Exception:
        logger.debug("Failed to load prompt budget settings; using defaults", exc_info=True)

    ceiling = max(int(max(context_window, 0) * ceiling_pct), _MIN_SYSTEM_BUDGET)
    target = max(min(target_tokens, ceiling), _MIN_SYSTEM_BUDGET)
    if system_budget is not None:
        explicit_cap = max(system_budget, _MIN_SYSTEM_BUDGET)
        target = min(target, explicit_cap)
        ceiling = min(ceiling, explicit_cap)
    return PromptBudget(target=target, ceiling=ceiling)


def _split_content_items(content: str) -> tuple[str, ...]:
    """Split elastic content into droppable paragraphs without splitting priming blocks."""
    items: list[str] = []
    cursor = 0
    for match in _PRIMING_BLOCK_RE.finditer(content):
        prefix = content[cursor : match.start()]
        items.extend(part.strip() for part in _PARAGRAPH_SPLIT_RE.split(prefix) if part.strip())
        block = match.group(0).strip()
        if block:
            items.append(block)
        cursor = match.end()
    suffix = content[cursor:]
    items.extend(part.strip() for part in _PARAGRAPH_SPLIT_RE.split(suffix) if part.strip())
    return tuple(items)


def _entry_items(section: SectionEntry) -> tuple[str, ...]:
    if section.items is not None:
        return tuple(item.strip() for item in section.items if item.strip())
    return _split_content_items(section.content)


def _normalize_duplicate_key(paragraph: str) -> str:
    without_priming_attributes = _PRIMING_OPEN_RE.sub("<priming>", paragraph)
    return re.sub(r"\s+", " ", without_priming_attributes).strip()


def _deduplicate_sections(sections: list[SectionEntry]) -> list[SectionEntry]:
    """Remove repeated long paragraphs from later elastic sections."""
    seen: set[str] = set()
    deduplicated: list[SectionEntry] = []
    removed_by_section: dict[str, int] = {}

    for section in sections:
        removed_before = removed_by_section.get(section.id, 0)
        source_items = _entry_items(section)
        kept_items: list[str] = []
        for item in source_items:
            key = _normalize_duplicate_key(item)
            duplicate = len(key) >= 80 and key in seen
            if duplicate and section.kind == "elastic":
                removed_by_section[section.id] = removed_by_section.get(section.id, 0) + 1
                continue
            kept_items.append(item)
            if len(key) >= 80:
                seen.add(key)

        if section.kind == "elastic":
            removed_here = removed_by_section.get(section.id, 0) > removed_before
            content = "\n\n".join(kept_items) if removed_here else section.content
            deduplicated.append(replace(section, content=content, items=tuple(kept_items)))
        else:
            deduplicated.append(section)

    if removed_by_section:
        details = ",".join(f"{name}:{count}" for name, count in removed_by_section.items())
        logger.info("Prompt paragraph deduplication: removed=%s", details)
    return deduplicated


def _allocate_sections(
    sections: list[SectionEntry],
    budget: PromptBudget | int,
) -> list[SectionEntry]:
    """Allocate whole elastic items while preserving section order.

    Elastic content is deliberately never sliced: partial Markdown/XML items
    are less useful than dropping a complete low-priority item.
    """
    if not sections:
        return []
    if isinstance(budget, int):
        budget = PromptBudget(target=budget, ceiling=budget)

    prepared = _deduplicate_sections(sections)
    before_tokens = sum(estimate_tokens(section.content) for section in prepared)
    included_rigid = {i for i, section in enumerate(prepared) if section.kind == "rigid"}
    elastic_items = {i: list(_entry_items(section)) for i, section in enumerate(prepared) if section.kind == "elastic"}
    dropped_items: dict[str, int] = {}
    trimmed_elastic_indices: set[int] = set()

    def total_tokens(group: str | None = None) -> int:
        rigid_cost = sum(
            estimate_tokens(prepared[i].content)
            for i in included_rigid
            if group is None or prepared[i].budget_group == group
        )
        elastic_cost = sum(
            estimate_tokens("\n\n".join(items))
            for i, items in elastic_items.items()
            if group is None or prepared[i].budget_group == group
        )
        return rigid_cost + elastic_cost

    # Low-priority elastic sections give way first. For ties, trim whichever
    # section is currently largest so one oversized source cannot dominate.
    def trim_elastic(target: int, group: str | None = None) -> None:
        for priority in range(4, 0, -1):
            while total_tokens(group) > target:
                candidates = [
                    i
                    for i, items in elastic_items.items()
                    if items
                    and prepared[i].priority == priority
                    and (group is None or prepared[i].budget_group == group)
                ]
                if not candidates:
                    break
                index = max(candidates, key=lambda i: (estimate_tokens("\n\n".join(elastic_items[i])), -i))
                section = prepared[index]
                if section.trim_from == "head":
                    elastic_items[index].pop(0)
                else:
                    elastic_items[index].pop()
                trimmed_elastic_indices.add(index)
                dropped_items[section.id] = dropped_items.get(section.id, 0) + 1

    # The framework target does not consume the separately configured recall
    # allowance. Both still share the model's hard context ceiling.
    recall_target = 2000
    if any(section.budget_group == "recall" for section in prepared):
        try:
            from core.config import load_config

            configured_target = load_config().priming.max_tokens
            if (
                isinstance(configured_target, int)
                and not isinstance(configured_target, bool)
                and configured_target >= 200
            ):
                recall_target = configured_target
        except Exception:
            logger.debug("Using default recall prompt budget", exc_info=True)
    trim_elastic(budget.target, "framework")
    trim_elastic(max(0, min(recall_target, budget.ceiling)), "recall")

    dropped_rigid: list[str] = []
    # Rigid sections may exceed the target, but only the hard ceiling is
    # allowed to evict them. Priority-1 content and group boundaries survive.
    for priority in range(4, 1, -1):
        if total_tokens() <= budget.ceiling:
            break
        candidates = [
            i for i in included_rigid if prepared[i].priority == priority and not _GROUP_HEADER_RE.match(prepared[i].id)
        ]
        for index in sorted(candidates, key=lambda i: estimate_tokens(prepared[i].content), reverse=True):
            if total_tokens() <= budget.ceiling:
                break
            included_rigid.remove(index)
            dropped_rigid.append(prepared[index].id)

    trim_elastic(budget.ceiling)

    result: list[SectionEntry] = []
    for i, section in enumerate(prepared):
        if i in included_rigid:
            result.append(section)
        elif i in elastic_items and elastic_items[i]:
            items = tuple(elastic_items[i])
            content = "\n\n".join(items) if i in trimmed_elastic_indices else section.content
            result.append(replace(section, content=content, items=items))

    after_tokens = sum(estimate_tokens(section.content) for section in result)
    if dropped_items or dropped_rigid:
        item_details = ",".join(f"{name}:{count}" for name, count in dropped_items.items()) or "none"
        rigid_details = ",".join(dropped_rigid) or "none"
        logger.info(
            "Prompt allocation: dropped_items=%s dropped_rigid=%s before_tokens=%d "
            "after_tokens=%d target=%d ceiling=%d",
            item_details,
            rigid_details,
            before_tokens,
            after_tokens,
            budget.target,
            budget.ceiling,
        )
    return result
