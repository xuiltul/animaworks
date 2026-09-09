from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-channel consolidation for itemized priming memories."""

import logging
import re
from dataclasses import replace
from difflib import SequenceMatcher

from core.i18n import t
from core.memory.priming.items import MemoryItem, render_items
from core.memory.priming.result import PrimingResult

logger = logging.getLogger("animaworks.priming")

_TIME_RE = re.compile(r"\[\d{2}:\d{2}\]")
_WHITESPACE_RE = re.compile(r"\s+")
_HEADERS = {
    "important_knowledge": "### [IMPORTANT] Knowledge (summary pointers)",
    "related_knowledge": "",
    "related_knowledge_untrusted": "",
    "recent_activity": "",
    "episodes": "",
    "pending_tasks": "",
    "recent_outbound": "",
    "pending_human_notifications": "## Pending Human Notifications (last 24h)",
}
_DIRECT_FIELDS = frozenset(
    {
        "recent_activity",
        "episodes",
        "pending_tasks",
        "recent_outbound",
        "pending_human_notifications",
    }
)


def _normalized_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _TIME_RE.sub("", text)).strip()


def _near_identical(left: str, right: str) -> bool:
    """Return whether *right* adds little beyond a shared 60-char body."""
    if len(left) < 60 or len(right) < 60:
        return False
    common_size = SequenceMatcher(None, left, right, autojunk=False).find_longest_match().size
    # A long shared preface alone is not duplication: substantial text unique
    # to the later item may be a correction or an important addendum.
    return common_size >= 60 and len(right) - common_size < 40


_PATH_SOURCE_PRIORITY = {
    "important_knowledge": 2,
    "related_knowledge": 1,
    "related_knowledge_untrusted": 1,
    "episodes": 0,
}


def _key_preference(item: MemoryItem) -> tuple[int, str]:
    """Prefer intentional resident/search pointers over episode copies."""
    return (_PATH_SOURCE_PRIORITY.get(item.source, 0), item.updated)


def consolidate_items(result: PrimingResult) -> PrimingResult:
    """Drop duplicate/obsolete items and rebuild their channel strings."""
    flattened = [item for channel_items in result.items.values() for item in channel_items]
    normalized_texts = [_normalized_text(item.text) for item in flattened]
    preferred_by_key: dict[str, int] = {}
    latest_important_by_ref: dict[str, int] = {}
    for index, item in enumerate(flattened):
        effective_key = item.key or normalized_texts[index]
        current_index = preferred_by_key.get(effective_key)
        if current_index is None or _key_preference(item) > _key_preference(flattened[current_index]):
            preferred_by_key[effective_key] = index
        if item.source == "important_knowledge" and item.ref:
            current_ref_index = latest_important_by_ref.get(item.ref)
            if current_ref_index is None or item.updated > flattened[current_ref_index].updated:
                latest_important_by_ref[item.ref] = index

    kept: list[MemoryItem] = []
    normalized_kept: list[str] = []
    dropped: list[tuple[MemoryItem, str, MemoryItem | None]] = []

    for index, item in enumerate(flattened):
        effective_key = item.key or normalized_texts[index]
        preferred_index = preferred_by_key.get(effective_key)
        if preferred_index != index:
            dropped.append((item, "same_key", flattened[preferred_index] if preferred_index is not None else None))
            continue
        if item.source == "important_knowledge" and item.ref and latest_important_by_ref.get(item.ref) != index:
            kept_index = latest_important_by_ref[item.ref]
            dropped.append((item, "older_important_ref", flattened[kept_index]))
            continue
        normalized = normalized_texts[index]
        duplicate_index = next(
            (
                kept_index
                for kept_index, previous in enumerate(normalized_kept)
                if _near_identical(previous, normalized)
            ),
            None,
        )
        if duplicate_index is not None:
            dropped.append((item, "near_identical", kept[duplicate_index]))
            continue
        kept.append(item)
        normalized_kept.append(normalized)

    consolidated: dict[str, tuple[MemoryItem, ...]] = {}
    for source in result.items:
        consolidated[source] = tuple(item for item in kept if item.source == source)

    updates: dict[str, object] = {"items": consolidated}
    for source in _DIRECT_FIELDS:
        if source in result.items:
            header = t("priming.outbound_header") if source == "recent_outbound" else _HEADERS[source]
            updates[source] = render_items(consolidated.get(source, ()), header)

    if "important_knowledge" in result.items or "related_knowledge" in result.items:
        important = render_items(
            consolidated.get("important_knowledge", ()),
            _HEADERS["important_knowledge"],
        )
        related = (
            render_items(consolidated.get("related_knowledge", ()), "")
            if "related_knowledge" in result.items
            else result.related_knowledge
        )
        updates["related_knowledge"] = f"{important}\n\n{related}" if important and related else important or related

    if "related_knowledge_untrusted" in result.items:
        updates["related_knowledge_untrusted"] = render_items(
            consolidated.get("related_knowledge_untrusted", ()),
            "",
        )

    for item, reason, retained in dropped:
        logger.debug(
            "Priming consolidation dropped item: source=%s key=%s ref=%s reason=%s retained_source=%s retained_key=%s",
            item.source,
            item.key,
            item.ref,
            reason,
            retained.source if retained else "",
            retained.key if retained else "",
        )
    logger.info("Priming consolidation: kept=%d dropped=%d", len(kept), len(dropped))
    return replace(result, **updates)
