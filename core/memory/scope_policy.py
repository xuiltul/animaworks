from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Scope routing policy for hybrid Neo4j and legacy memory search."""

from dataclasses import dataclass
from typing import Any

NEO4J_SCOPE_MAP: dict[str, str] = {
    "knowledge": "fact",
    "episodes": "episode",
    "procedures": "fact",
    "all": "all",
}

NEO4J_BACKED_SCOPES: frozenset[str] = frozenset(NEO4J_SCOPE_MAP)

LEGACY_ONLY_SCOPES: frozenset[str] = frozenset(
    {
        "common_knowledge",
        "skills",
        "facts",
        "activity_log",
    }
)

LEGACY_ONLY_SCOPES_FOR_ALL: tuple[str, ...] = (
    "common_knowledge",
    "skills",
    "facts",
    "activity_log",
)

LEGACY_SCOPE_TITLES: dict[str, str] = {
    "common_knowledge": "Common Knowledge",
    "skills": "Skills",
    "facts": "Facts",
    "activity_log": "Activity Log",
}


def is_neo4j_backed_scope(scope: str) -> bool:
    """Return True when *scope* has Neo4j graph coverage."""
    return scope in NEO4J_BACKED_SCOPES


def is_legacy_only_scope(scope: str) -> bool:
    """Return True when *scope* must still be searched through legacy RAG."""
    return scope in LEGACY_ONLY_SCOPES


def neo4j_scope_for(scope: str) -> str:
    """Map a public search scope to the Neo4j backend scope."""
    return NEO4J_SCOPE_MAP.get(scope, "all")


def title_for_legacy_scope(scope: str) -> str:
    """Return the output section title for a legacy-only scope."""
    return LEGACY_SCOPE_TITLES.get(scope, scope.replace("_", " ").title())


@dataclass(frozen=True, slots=True)
class SearchResultItem:
    """A sectioned search result item assembled before pagination."""

    section: str
    item_type: str
    value: Any


def format_graph_memory_entry(mem: Any, index: int) -> str:
    """Format a Neo4j RetrievedMemory entry for search_memory output."""
    return f"\n[{index}] score={mem.score:.2f} | {mem.source}\n{mem.content}\n"


def format_legacy_memory_entry(result: dict[str, Any], index: int) -> str:
    """Format a legacy RAG result entry for search_memory output."""
    source = result.get("source_file", "unknown")
    score = result.get("score", 0.0)
    chunk_idx = result.get("chunk_index", 0)
    total_chunks = result.get("total_chunks", 1)
    content = result.get("content", "")
    entry_header = f"[{index}] score={score:.2f} | {source} | chunk {chunk_idx + 1}/{total_chunks}"
    return f"\n{entry_header}\n{content}\n"


def format_hybrid_search_results(
    *,
    query: str,
    items: list[SearchResultItem],
    offset: int,
    context_window: int,
    search_max_tokens: int,
    search_context_base: int,
    search_min_results: int,
) -> str:
    """Format scope='all' hybrid search results after combined-list pagination."""
    visible_items = items[offset : offset + 10]
    if not visible_items:
        return ""

    end_index = offset + len(visible_items)
    header = f'Search results for "{query}" (hybrid, all, {offset + 1}-{end_index}):\n'
    scale = min(1.0, context_window / search_context_base)
    max_tokens = int(search_max_tokens * scale)
    parts: list[str] = [header]
    total_tokens = len(header) // 4
    current_section: str | None = None

    for shown_count, item in enumerate(visible_items):
        index = offset + shown_count + 1
        if item.section != current_section:
            section_header = f"\n## {item.section}\n"
            parts.append(section_header)
            total_tokens += len(section_header) // 4
            current_section = item.section

        if item.item_type == "graph":
            entry = format_graph_memory_entry(item.value, index)
        else:
            entry = format_legacy_memory_entry(item.value, index)

        entry_tokens = len(entry) // 4
        if total_tokens + entry_tokens > max_tokens and shown_count >= search_min_results:
            parts.append("\n(truncated - output limit reached)")
            break
        parts.append(entry)
        total_tokens += entry_tokens

    if len(items) > end_index:
        parts.append(f"\nUse offset={end_index} to see next page.")

    return "".join(parts)
