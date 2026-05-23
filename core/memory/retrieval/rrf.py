from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Reciprocal Rank Fusion — unified Legacy + Neo4j result merging."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScoredItem:
    """An item with a unique key and combined RRF score."""

    key: str
    score: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


def legacy_result_key(item: dict[str, Any]) -> str:
    """Stable dedup key for Chroma/BM25 legacy result dicts."""
    source_file = str(item.get("source_file", "") or "")
    chunk_index = item.get("chunk_index", 0)
    ts = item.get("ts", "")
    memory_type = item.get("memory_type", "")
    return f"{source_file}\x00{chunk_index}\x00{ts}\x00{memory_type}"


def rrf_merge(
    result_lists: list[list[dict[str, Any]]],
    *,
    key_field: str = "uuid",
    key_fn: Callable[[dict[str, Any]], str] | None = None,
    k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """Merge ranked lists using RRF (Neo4j graph rows or generic dicts).

    Args:
        result_lists: Each inner list is ranked best-first.
        key_field: Field for dedup when *key_fn* is None (Neo4j rows).
        key_fn: Optional custom key extractor (Legacy dicts).
        k: RRF smoothing constant.
        top_k: Maximum merged results.

    Returns:
        Merged dicts with ``rrf_score`` set, sorted descending.
    """
    if key_fn is None:

        def key_fn(item: dict[str, Any]) -> str:
            return str(item.get(key_field, "") or legacy_result_key(item))

    scores: dict[str, ScoredItem] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            item_key = key_fn(item)
            if not item_key:
                item_key = f"unknown_{rank}"
            rrf_score = 1.0 / (k + rank + 1)

            if item_key in scores:
                scores[item_key].score += rrf_score
            else:
                scores[item_key] = ScoredItem(key=item_key, score=rrf_score, data=dict(item))

    merged = sorted(scores.values(), key=lambda s: s.score, reverse=True)

    result: list[dict[str, Any]] = []
    for item in merged[:top_k]:
        row = dict(item.data)
        row["rrf_score"] = item.score
        row["score"] = item.score
        if "search_method" not in row:
            row["search_method"] = "rrf"
        result.append(row)

    return result


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Merge Legacy-style result lists (activity_log + vector, etc.)."""
    if not ranked_lists:
        return []
    limit = top_k if top_k is not None else 10
    return rrf_merge(
        list(ranked_lists),
        key_fn=legacy_result_key,
        k=k,
        top_k=limit,
    )
