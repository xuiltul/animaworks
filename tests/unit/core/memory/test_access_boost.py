from __future__ import annotations

import math
from datetime import UTC, datetime

from core.memory.retrieval.access_boost import AccessBoostConfig, apply_access_boost, compute_access_boost


def test_compute_access_boost_applies_formula() -> None:
    config = AccessBoostConfig(weight=0.05, cap=0.25, half_life_days=30.0)

    boost = compute_access_boost(
        access_count=3,
        last_accessed_at="",
        config=config,
        now=datetime(2026, 6, 5, tzinfo=UTC),
    )

    assert boost == math.log1p(3) * 0.05


def test_compute_access_boost_caps_high_counts() -> None:
    config = AccessBoostConfig(weight=0.05, cap=0.25, half_life_days=30.0)

    boost = compute_access_boost(
        access_count=10_000,
        last_accessed_at="",
        config=config,
        now=datetime(2026, 6, 5, tzinfo=UTC),
    )

    assert boost == 0.25


def test_compute_access_boost_applies_exponential_recency_decay() -> None:
    config = AccessBoostConfig(weight=0.05, cap=0.25, half_life_days=30.0)

    boost = compute_access_boost(
        access_count=10_000,
        last_accessed_at="2026-05-06T00:00:00+00:00",
        config=config,
        now=datetime(2026, 6, 5, tzinfo=UTC),
    )

    assert boost == 0.25 * math.exp(-1.0)


def test_apply_access_boost_reorders_same_relevance_candidates() -> None:
    config = AccessBoostConfig(weight=0.05, cap=0.25, half_life_days=30.0)
    candidates = [
        {"content": "low access", "score": 0.8, "source_file": "a.md", "access_count": 0},
        {"content": "high access", "score": 0.8, "source_file": "b.md", "access_count": 20},
    ]

    boosted = apply_access_boost(candidates, config, now=datetime(2026, 6, 5, tzinfo=UTC))

    assert boosted[0]["content"] == "high access"
    assert boosted[0]["access_boost"] > 0.0
    assert boosted[0]["score"] > boosted[1]["score"]


def test_apply_access_boost_treats_missing_access_count_as_zero() -> None:
    config = AccessBoostConfig(weight=0.05, cap=0.25, half_life_days=30.0)

    boosted = apply_access_boost([{"content": "new", "score": 0.8}], config)

    assert boosted[0]["score"] == 0.8
    assert "access_boost" not in boosted[0]


def test_apply_access_boost_can_be_disabled() -> None:
    candidates = [{"content": "hit", "score": 0.8, "access_count": 100}]

    assert apply_access_boost(candidates, None) is candidates
    assert apply_access_boost(candidates, AccessBoostConfig(enabled=False)) is candidates


def test_apply_access_boost_reads_nested_metadata() -> None:
    config = AccessBoostConfig(weight=0.05, cap=0.25, half_life_days=30.0)
    candidates = [
        {
            "content": "nested",
            "score": 0.8,
            "metadata": {
                "access_count": "10",
                "last_accessed_at": "2026-06-05T00:00:00Z",
            },
        }
    ]

    boosted = apply_access_boost(candidates, config, now=datetime(2026, 6, 5, tzinfo=UTC))

    assert boosted[0]["access_boost"] > 0.0


def test_compute_access_boost_handles_invalid_and_zero_weight_inputs() -> None:
    config = AccessBoostConfig(weight=0.0, cap=0.25, half_life_days=30.0)

    assert compute_access_boost(access_count="not-int", last_accessed_at="", config=config) == 0.0
    assert compute_access_boost(access_count=5, last_accessed_at="bad-date", config=config) == 0.0
