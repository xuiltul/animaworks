from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Opt-in temporal ranking helpers for memory retrieval."""

import re
from dataclasses import dataclass
from typing import Any

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class TemporalBoostConfig:
    """Configuration for additive temporal candidate scoring."""

    enabled: bool = False
    boost: float = 0.05
    max_boost: float = 0.10
    category: int | None = None


def apply_temporal_boost(
    query: str,
    candidates: list[dict[str, Any]],
    config: TemporalBoostConfig | None,
) -> list[dict[str, Any]]:
    """Apply a conservative additive temporal boost to category-2 candidates."""
    if config is None or not config.enabled or config.category != 2:
        return candidates

    query_years = set(_YEAR_RE.findall(query))
    boosted: list[dict[str, Any]] = []
    for candidate in candidates:
        if not _has_event_time(candidate):
            boosted.append(candidate)
            continue

        row = dict(candidate)
        base_score = float(row.get("score", 0.0) or 0.0)
        row["base_score"] = base_score

        temporal_boost = max(0.0, float(config.boost))
        if query_years and query_years & _candidate_years(row):
            temporal_boost += float(config.boost)
        temporal_boost = min(temporal_boost, float(config.max_boost))

        row["temporal_boost"] = temporal_boost
        row["score"] = base_score + temporal_boost
        boosted.append(row)

    boosted.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return boosted


def _has_event_time(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("valid_at") or candidate.get("event_time_iso"))


def _candidate_years(candidate: dict[str, Any]) -> set[str]:
    text = " ".join(str(candidate.get(key, "") or "") for key in ("content", "event_time_text", "event_time_iso"))
    return set(_YEAR_RE.findall(text))
