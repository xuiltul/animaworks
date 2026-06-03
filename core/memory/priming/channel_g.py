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
from typing import TYPE_CHECKING

from core.memory.priming.utils import truncate_tail

if TYPE_CHECKING:
    from core.memory.backend.base import MemoryBackend

logger = logging.getLogger("animaworks.priming")


async def collect_graph_context(
    backend: MemoryBackend,
    query: str,
    *,
    budget_tokens: int = 500,
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

    parts: list[str] = []

    if communities:
        parts.append("## Communities")
        for mem in communities:
            parts.append(f"- {mem.content}")

    if facts:
        parts.append("## Recent Facts")
        for mem in facts:
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
