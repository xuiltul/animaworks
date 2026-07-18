from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Opt-in temporal ranking helpers for memory retrieval."""

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import PurePath
from typing import Any

from core.memory.retrieval.time_expr import TimeRange

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SOURCE_DATE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class TemporalBoostConfig:
    """Configuration for additive temporal candidate scoring."""

    enabled: bool = False
    boost: float = 0.05
    max_boost: float = 0.10
    category: int | None = None
    time_range: TimeRange | None = None
    recency: bool = False
    half_life_days: float = 7.0
    now: datetime | None = None


def apply_temporal_boost(
    query: str,
    candidates: list[dict[str, Any]],
    config: TemporalBoostConfig | None,
) -> list[dict[str, Any]]:
    """Apply conservative range or recency boosts to temporal candidates.

    An explicitly non-temporal legacy category remains disabled for backwards
    compatibility.  Missing or malformed candidate timestamps never remove or
    penalize a result.
    """
    if config is None or not config.enabled or config.category not in (None, 2):
        return candidates

    time_range = config.time_range
    recency_intent = config.recency or bool(time_range and time_range.recency)
    has_range = bool(time_range and (time_range.start is not None or time_range.end is not None))
    query_years = set(_YEAR_RE.findall(query))
    legacy_year_intent = config.category == 2 and not has_range and not recency_intent

    # Merely enabling the new optional layer is not itself a ranking intent.
    if not has_range and not recency_intent and not legacy_year_intent:
        return candidates

    boost_weight = max(0.0, float(config.boost))
    boost_cap = max(0.0, float(config.max_boost))
    now = _as_naive(config.now) if config.now is not None else datetime.now()
    boosted: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_time = resolve_candidate_time(candidate)
        if candidate_time is None:
            boosted.append(candidate)
            continue

        temporal_boost = 0.0
        if has_range and time_range is not None and _in_range(candidate_time, time_range):
            temporal_boost += boost_weight

        if recency_intent:
            temporal_boost += _recency_boost(
                candidate_time,
                now=now,
                weight=boost_weight,
                half_life_days=config.half_life_days,
            )

        if legacy_year_intent:
            # Preserve the opt-in category-2 behavior used by older callers:
            # temporal candidates receive one boost, plus one for year match.
            temporal_boost += boost_weight
            if query_years and query_years & _candidate_years(candidate):
                temporal_boost += boost_weight

        temporal_boost = min(temporal_boost, boost_cap)
        if temporal_boost <= 0.0:
            boosted.append(candidate)
            continue

        row = dict(candidate)
        base_score = float(row.get("score", 0.0) or 0.0)
        row["base_score"] = base_score
        row["temporal_boost"] = temporal_boost
        row["score"] = base_score + temporal_boost
        boosted.append(row)

    boosted.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return boosted


def resolve_candidate_time(candidate: dict[str, Any]) -> datetime | None:
    """Resolve a candidate timestamp in the documented precedence order."""
    sources = [candidate]
    metadata = candidate.get("metadata")
    if isinstance(metadata, dict):
        sources.append(metadata)

    for key in ("valid_at", "event_time_iso", "ts"):
        for source in sources:
            if key not in source:
                continue
            parsed = _parse_datetime(source.get(key))
            if parsed is not None:
                return parsed

    for source in sources:
        parsed = _parse_source_file_date(source.get("source_file"))
        if parsed is not None:
            return parsed
    return None


def _in_range(candidate_time: datetime, time_range: TimeRange) -> bool:
    start = _as_naive(time_range.start) if time_range.start is not None else None
    end = _as_naive(time_range.end) if time_range.end is not None else None
    return (start is None or candidate_time >= start) and (end is None or candidate_time <= end)


def _recency_boost(
    candidate_time: datetime,
    *,
    now: datetime,
    weight: float,
    half_life_days: float,
) -> float:
    if weight <= 0.0 or half_life_days <= 0.0:
        return 0.0
    age_days = max(0.0, (now - candidate_time).total_seconds() / 86400.0)
    return weight * math.exp(-age_days / float(half_life_days))


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_naive(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value))
        except (OSError, OverflowError, ValueError):
            return None

    text = str(value or "").strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None and math.isfinite(numeric):
        try:
            return datetime.fromtimestamp(numeric)
        except (OSError, OverflowError, ValueError):
            return None

    try:
        return _as_naive(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        try:
            return datetime.combine(date.fromisoformat(text), time.min)
        except ValueError:
            return None


def _parse_source_file_date(value: Any) -> datetime | None:
    name = PurePath(str(value or "")).name
    match = _SOURCE_DATE_RE.match(name)
    if match is None:
        return None
    try:
        return datetime.combine(date.fromisoformat(match.group("date")), time.min)
    except ValueError:
        return None


def _as_naive(value: datetime) -> datetime:
    # Time expressions use the project's naive-local convention.  Preserve the
    # timestamp's displayed local wall time while making comparisons safe.
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _candidate_years(candidate: dict[str, Any]) -> set[str]:
    text = " ".join(str(candidate.get(key, "") or "") for key in ("content", "event_time_text", "event_time_iso"))
    return set(_YEAR_RE.findall(text))
