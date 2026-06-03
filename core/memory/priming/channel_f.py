from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel F: Episode memory search (vector search)."""

import json
import logging
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from core.memory.priming.utils import build_queries
from core.memory.retrieval.unified_search import UnifiedMemorySearch
from core.time_utils import now_local

logger = logging.getLogger(__name__)


def _single_line(text: str, limit: int = 160) -> str:
    """Collapse prompt cue text to one bounded line."""
    collapsed = " ".join(str(text or "").split())
    return collapsed[:limit]


def _quote_path(path: str) -> str:
    """Return a JSON string literal for read_memory_file path examples."""
    return json.dumps(path, ensure_ascii=False)


def to_episode_memory_path(source: str) -> str:
    """Normalize retriever/backend source to a read_memory_file episode path."""
    if not source:
        return ""
    source_path = str(source).split("#", 1)[0]
    marker = "episodes/"
    if marker in source_path:
        return marker + source_path.split(marker, 1)[1]
    if source_path.endswith(".md"):
        return f"episodes/{Path(source_path).name}"
    return ""


def extract_episode_summary(content: str, source: str) -> str:
    """Return a short cue for an episode without injecting the full episode body."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return _single_line(stripped.lstrip("#").strip(), 120)
    path = to_episode_memory_path(source)
    if path:
        return _single_line(Path(path).stem.replace("-", " ").replace("_", " "))
    return "Related episode memory"


def format_episode_pointer(
    *,
    index: int,
    score: float,
    source: str,
    content: str,
    path: str,
) -> str:
    """Format an episode result as a pointer cue instead of raw payload."""
    summary = extract_episode_summary(content, source)
    return (
        f"--- Episode {index} (score: {score:.3f}, source: {_single_line(source, 120)}) ---\n"
        f"{summary}\n"
        f"  -> read_memory_file(path={_quote_path(path)})\n"
    )


async def channel_f_episodes(
    anima_dir: Path,
    episodes_dir: Path,
    get_retriever: Callable[..., Any],
    keywords: list[str],
    *,
    message: str = "",
    recent_human_messages: list[str] | None = None,
    get_memory_backend: Callable[[], Any] | None = None,
) -> str:
    """Channel F: Episode memory search (vector search).

    Searches episodes/ via dense vector retrieval to surface
    semantically relevant past experiences.  Complements Channel B
    (recent activity timeline) by looking further back in time and
    ranking by semantic similarity rather than recency alone.

    When the active memory backend is Neo4j, uses ``retrieve()`` with a
    7-day ``time_start`` window so recent episodes are preferred.
    """
    try:
        queries = build_queries(message, keywords, recent_human_messages)
        if not queries:
            return ""

        _min_score: float | None = None
        try:
            from core.config.models import load_config as _load_cfg

            _min_score = _load_cfg().rag.min_retrieval_score
        except Exception:
            logger.debug("Failed to load rag.min_retrieval_score from config, using default")

        if get_memory_backend is not None:
            backend = get_memory_backend()
            if backend is not None:
                from core.memory.backend.neo4j_graph import Neo4jGraphBackend

                if isinstance(backend, Neo4jGraphBackend):
                    time_start = (now_local() - timedelta(days=7)).isoformat()
                    best: dict[str, Any] = {}
                    min_score_val = float(_min_score) if _min_score is not None else 0.0
                    for query in queries:
                        merged_batch = await backend.retrieve(
                            query,
                            scope="episode",
                            limit=5,
                            min_score=min_score_val,
                            time_start=time_start,
                        )
                        for m in merged_batch:
                            existing = best.get(m.source)
                            if existing is None or m.score > existing.score:
                                best[m.source] = m
                    merged = sorted(best.values(), key=lambda m: m.score, reverse=True)[:5]
                    if not merged:
                        return ""
                    parts: list[str] = []
                    accessed_memories = []
                    for mem in merged:
                        meta = mem.metadata if isinstance(mem.metadata, dict) else {}
                        source = meta.get("source_file") or meta.get("source") or mem.source
                        path = to_episode_memory_path(source)
                        if not path:
                            logger.debug("Channel F: skipping Neo4j episode without readable path: %s", mem.source)
                            continue
                        accessed_memories.append(mem)
                        parts.append(
                            format_episode_pointer(
                                index=len(parts) + 1,
                                score=mem.score,
                                source=source,
                                content=mem.content,
                                path=path,
                            ),
                        )
                    if accessed_memories:
                        try:
                            await backend.record_access(accessed_memories)
                        except Exception:
                            logger.debug("Channel F: record_access skipped", exc_info=True)

                    logger.debug(
                        "Channel F: Neo4j episode search returned %d results",
                        len(merged),
                    )
                    return "\n".join(parts) if parts else ""

        if not episodes_dir.is_dir():
            return ""

        searcher = UnifiedMemorySearch(anima_dir)
        results = searcher.search_many(
            queries,
            scope="episodes",
            limit=5,
            trigger="chat",
            min_score=float(_min_score) if _min_score is not None else 0.0,
        )
        if bool(searcher.last_search_meta.get("abstain", False)):
            logger.debug("Channel F: unified search abstained")
            return ""

        if not results:
            return ""

        parts = []
        for result in results:
            source = str(result.get("source_file", "") or result.get("doc_id", "") or "")
            path = to_episode_memory_path(source)
            if not path:
                logger.debug("Channel F: skipping episode without readable path: %s", result.get("doc_id", ""))
                continue
            parts.append(
                format_episode_pointer(
                    index=len(parts) + 1,
                    score=float(result.get("score", 0.0) or 0.0),
                    source=source,
                    content=str(result.get("content", "") or ""),
                    path=path,
                )
            )

        logger.debug(
            "Channel F: Episode search returned %d results",
            len(results),
        )
        return "\n".join(parts) if parts else ""

    except Exception as e:
        logger.warning("Channel F: Episode search failed: %s", e)
        return ""
