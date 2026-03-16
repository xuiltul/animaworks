from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel F: Episode memory search (vector search)."""

import logging
from pathlib import Path

from core.memory.priming.utils import build_dual_queries, search_and_merge

logger = logging.getLogger("animaworks.priming")


async def channel_f_episodes(
    anima_dir: Path,
    episodes_dir: Path,
    get_retriever: callable,
    keywords: list[str],
    *,
    message: str = "",
) -> str:
    """Channel F: Episode memory search (vector search).

    Searches episodes/ via dense vector retrieval to surface
    semantically relevant past experiences.  Complements Channel B
    (recent activity timeline) by looking further back in time and
    ranking by semantic similarity rather than recency alone.
    """
    if not episodes_dir.is_dir():
        return ""

    try:
        retriever = get_retriever()
        if retriever is None:
            return ""

        queries = build_dual_queries(message, keywords)
        if not queries:
            return ""
        anima_name = anima_dir.name

        _min_score: float | None = None
        try:
            from core.config.models import load_config as _load_cfg

            _min_score = _load_cfg().rag.min_retrieval_score
        except Exception:
            logger.debug("Failed to load rag.min_retrieval_score from config, using default")

        results = search_and_merge(
            retriever,
            queries,
            anima_name,
            memory_type="episodes",
            top_k=3,
            min_score=_min_score,
        )

        if not results:
            return ""

        retriever.record_access(results, anima_name)

        parts: list[str] = []
        for i, result in enumerate(results):
            source = result.metadata.get("source_file", result.doc_id)
            parts.append(f"--- Episode {i + 1} (score: {result.score:.3f}, source: {source}) ---\n{result.content}\n")

        logger.debug(
            "Channel F: Episode search returned %d results",
            len(results),
        )
        return "\n".join(parts)

    except Exception as e:
        logger.warning("Channel F: Episode search failed: %s", e)
        return ""
