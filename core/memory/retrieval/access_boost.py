from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Access-count LTP boost for final retrieval ranking."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class AccessBoostConfig:
    """Configuration for multiplicative access-count ranking boost."""

    enabled: bool = True
    weight: float = 0.05
    cap: float = 0.25
    half_life_days: float = 30.0


def apply_access_boost(
    candidates: list[dict[str, Any]],
    config: AccessBoostConfig | None,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Apply a capped multiplicative access boost and return sorted candidates."""
    if config is None or not config.enabled:
        return candidates

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        base_score = float(row.get("score", 0.0) or 0.0)
        boost = compute_access_boost(
            access_count=_candidate_access_count(row),
            last_accessed_at=_candidate_last_accessed_at(row),
            config=config,
            now=now,
        )
        if boost > 0.0:
            row.setdefault("base_score", base_score)
            row["access_boost"] = boost
            row["score"] = base_score * (1.0 + boost)
        scored.append(row)

    scored.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return scored


def compute_access_boost(
    *,
    access_count: Any,
    last_accessed_at: Any,
    config: AccessBoostConfig,
    now: datetime | None = None,
) -> float:
    """Compute ``min(cap, log1p(access_count) * weight) * recency_factor``."""
    count = _coerce_access_count(access_count)
    if not config.enabled or count <= 0:
        return 0.0
    raw = min(max(0.0, float(config.cap)), math.log1p(count) * max(0.0, float(config.weight)))
    if raw <= 0.0:
        return 0.0
    return raw * _recency_factor(last_accessed_at, config=config, now=now)


def _candidate_access_count(candidate: dict[str, Any]) -> Any:
    for source in (candidate, candidate.get("metadata", {})):
        if isinstance(source, dict) and "access_count" in source:
            return source.get("access_count")
    return 0


def _candidate_last_accessed_at(candidate: dict[str, Any]) -> Any:
    for source in (candidate, candidate.get("metadata", {})):
        if isinstance(source, dict) and "last_accessed_at" in source:
            return source.get("last_accessed_at")
    return ""


def _coerce_access_count(value: Any) -> int:
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _recency_factor(
    value: Any,
    *,
    config: AccessBoostConfig,
    now: datetime | None,
) -> float:
    accessed_at = _parse_datetime(value)
    if accessed_at is None:
        return 1.0
    reference = _ensure_aware(now or datetime.now(UTC))
    age_days = max(0.0, (reference - accessed_at).total_seconds() / 86400.0)
    half_life = max(0.001, float(config.half_life_days))
    return math.exp(-age_days / half_life)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _ensure_aware(datetime.fromisoformat(text))
    except ValueError:
        return None


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
