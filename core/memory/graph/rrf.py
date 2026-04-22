# Copyright 2026 AnimaWorks contributors — Apache-2.0
"""Reciprocal Rank Fusion — merge multiple ranked result lists."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── ScoredItem ──────────────────────────────────────────────


@dataclass
class ScoredItem:
    """An item with a unique key and combined RRF score."""

    key: str
    score: float = 0.0
    data: dict = field(default_factory=dict)


# ── rrf_merge ───────────────────────────────────────────────


def rrf_merge(
    result_lists: list[list[dict]],
    *,
    key_field: str = "uuid",
    k: int = 60,
    top_k: int = 30,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of result lists, each sorted by relevance.
        key_field: Field name used as unique key for deduplication.
        k: RRF smoothing constant (default 60).
        top_k: Maximum results to return.

    Returns:
        Merged results sorted by descending RRF score.
    """
    scores: dict[str, ScoredItem] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            item_key = str(item.get(key_field, f"unknown_{rank}"))
            rrf_score = 1.0 / (k + rank + 1)

            if item_key in scores:
                scores[item_key].score += rrf_score
            else:
                scores[item_key] = ScoredItem(
                    key=item_key,
                    score=rrf_score,
                    data=dict(item),
                )

    merged = sorted(scores.values(), key=lambda s: s.score, reverse=True)

    result = []
    for item in merged[:top_k]:
        d = dict(item.data)
        d["rrf_score"] = item.score
        result.append(d)

    return result
