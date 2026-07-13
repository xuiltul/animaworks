from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Channel G: Graph context — community summaries + recent facts.

Fetches community context and recent facts from the MemoryBackend
abstraction layer.  For Neo4j backends this returns graph data; for the
legacy ChromaDB backend, communities are empty and recent_facts now prefers
atomic facts before falling back to BM25 activity-log search.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.file_access_policy import load_denied_roots, memory_source_is_allowed
from core.memory.priming.utils import truncate_tail

if TYPE_CHECKING:
    from core.memory.backend.base import MemoryBackend

logger = logging.getLogger("animaworks.priming")


async def collect_graph_context(
    backend: MemoryBackend,
    query: str,
    *,
    budget_tokens: int = 500,
    anima_dir: Path | None = None,
) -> str:
    """Fetch community context and recent facts from *backend*.

    Args:
        backend: Active :class:`MemoryBackend` instance.
        query: Natural-language query for relevance filtering.
        budget_tokens: Maximum token budget for the combined output.

    Returns:
        Formatted markdown string, or ``""`` when both sources are empty.
    """
    try:
        communities, facts = await asyncio.gather(
            backend.get_community_context(query, limit=3),
            backend.get_recent_facts(query, hours=24, limit=10),
        )
    except Exception:
        logger.debug("Channel G: backend call failed", exc_info=True)
        return ""

    denied_roots = load_denied_roots(anima_dir) if anima_dir is not None else ()

    def source_is_allowed(memory: object) -> bool:
        if anima_dir is None:
            return True
        metadata = getattr(memory, "metadata", {})
        meta = metadata if isinstance(metadata, dict) else {}
        source = str(
            meta.get("source_file", "")
            or meta.get("source_path", "")
            or meta.get("source", "")
            or getattr(memory, "source", "")
        )
        return memory_source_is_allowed(anima_dir, source, denied_roots)

    parts: list[str] = []

    if communities:
        visible_communities = [mem for mem in communities if source_is_allowed(mem)]
        if visible_communities:
            parts.append("## Communities")
        for mem in visible_communities:
            parts.append(f"- {mem.content}")

    if facts:
        visible_facts = [mem for mem in facts if source_is_allowed(mem)]
        if visible_facts:
            parts.append("## Recent Facts")
        for mem in visible_facts:
            meta = mem.metadata if isinstance(mem.metadata, dict) else {}
            source_entity = str(meta.get("source_entity", "") or "")
            edge_type = str(meta.get("edge_type", "") or "")
            target_entity = str(meta.get("target_entity", "") or "")
            relation = " ".join(part for part in (source_entity, edge_type, target_entity) if part)
            prefix = f"[{relation}] " if relation else ""
            parts.append(f"- {prefix}{mem.content}")

    if not parts:
        return ""

    text = "\n".join(parts)
    return truncate_tail(text, budget_tokens)
