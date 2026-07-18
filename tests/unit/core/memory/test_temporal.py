from __future__ import annotations

import math
from datetime import datetime

import pytest

from core.memory.retrieval.temporal import (
    TemporalBoostConfig,
    apply_temporal_boost,
    resolve_candidate_time,
)
from core.memory.retrieval.time_expr import TimeRange


def test_range_boost_ranks_inside_candidate_above_outside_candidate() -> None:
    candidates = [
        {"content": "outside", "score": 0.50, "event_time_iso": "2026-07-16T12:00:00"},
        {"content": "inside", "score": 0.49, "event_time_iso": "2026-07-17T12:00:00"},
    ]
    config = TemporalBoostConfig(
        enabled=True,
        time_range=TimeRange(
            start=datetime(2026, 7, 17),
            end=datetime(2026, 7, 17, 23, 59, 59, 999999),
        ),
    )

    result = apply_temporal_boost("昨日のミーティング", candidates, config)

    assert [item["content"] for item in result] == ["inside", "outside"]
    assert result[0]["temporal_boost"] == pytest.approx(0.05)
    assert "temporal_boost" not in result[1]


def test_yesterday_episode_source_filename_is_boosted() -> None:
    candidates = [
        {"content": "unrelated", "score": 0.50, "source_file": "2026-07-10_session.md"},
        {"content": "meeting", "score": 0.49, "source_file": "2026-07-17_session.md"},
    ]
    config = TemporalBoostConfig(
        enabled=True,
        time_range=TimeRange(
            start=datetime(2026, 7, 17),
            end=datetime(2026, 7, 17, 23, 59, 59, 999999),
        ),
    )

    result = apply_temporal_boost("昨日のミーティング", candidates, config)

    assert result[0]["content"] == "meeting"
    assert result[0]["temporal_boost"] == pytest.approx(0.05)


def test_recency_intent_applies_exponential_decay() -> None:
    now = datetime(2026, 7, 18, 12)
    candidates = [
        {"content": "old", "score": 0.50, "valid_at": datetime(2026, 6, 18, 12).timestamp()},
        {"content": "new", "score": 0.48, "valid_at": datetime(2026, 7, 17, 12).timestamp()},
    ]
    config = TemporalBoostConfig(
        enabled=True,
        time_range=TimeRange(start=None, end=None, recency=True),
        half_life_days=7.0,
        now=now,
    )

    result = apply_temporal_boost("最近の話", candidates, config)

    assert result[0]["content"] == "new"
    assert result[0]["temporal_boost"] == pytest.approx(0.05 * math.exp(-1 / 7))
    assert result[1]["temporal_boost"] == pytest.approx(0.05 * math.exp(-30 / 7))


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (
            {
                "valid_at": "2026-07-14T09:00:00",
                "event_time_iso": "2026-07-15T09:00:00",
                "ts": "2026-07-16T09:00:00",
                "source_file": "2026-07-17_session.md",
            },
            datetime(2026, 7, 14, 9),
        ),
        (
            {
                "event_time_iso": "2026-07-15T09:00:00",
                "ts": "2026-07-16T09:00:00",
                "source_file": "2026-07-17_session.md",
            },
            datetime(2026, 7, 15, 9),
        ),
        (
            {"ts": "2026-07-16T09:00:00", "source_file": "2026-07-17_session.md"},
            datetime(2026, 7, 16, 9),
        ),
        ({"source_file": "/episodes/2026-07-17_session.md"}, datetime(2026, 7, 17)),
    ],
)
def test_candidate_timestamp_resolution_order(candidate: dict[str, object], expected: datetime) -> None:
    assert resolve_candidate_time(candidate) == expected


def test_invalid_higher_priority_timestamp_falls_back_to_next() -> None:
    candidate = {
        "valid_at": "not-a-time",
        "event_time_iso": "2026-07-15T09:00:00",
        "source_file": "2026-07-17_session.md",
    }

    assert resolve_candidate_time(candidate) == datetime(2026, 7, 15, 9)


def test_missing_temporal_metadata_is_unchanged() -> None:
    candidate = {"content": "undated", "score": 0.42}
    config = TemporalBoostConfig(
        enabled=True,
        time_range=TimeRange(start=None, end=None, recency=True),
        now=datetime(2026, 7, 18),
    )

    result = apply_temporal_boost("recently", [candidate], config)

    assert result == [candidate]
    assert result[0] is candidate


def test_category_none_supports_range_but_explicit_non_temporal_category_does_not() -> None:
    candidate = {"content": "event", "score": 0.4, "event_time_iso": "2026-07-17T12:00:00"}
    time_range = TimeRange(start=datetime(2026, 7, 17), end=datetime(2026, 7, 18))

    enabled = apply_temporal_boost(
        "yesterday",
        [candidate],
        TemporalBoostConfig(enabled=True, category=None, time_range=time_range),
    )
    disabled = apply_temporal_boost(
        "yesterday",
        [candidate],
        TemporalBoostConfig(enabled=True, category=4, time_range=time_range),
    )

    assert enabled[0]["temporal_boost"] == pytest.approx(0.05)
    assert disabled == [candidate]


@pytest.mark.parametrize(
    "time_range",
    [
        TimeRange(start=datetime(2026, 7, 17), end=None),
        TimeRange(start=None, end=datetime(2026, 7, 17, 23, 59, 59)),
    ],
)
def test_open_ended_range_boosts_matching_candidate(time_range: TimeRange) -> None:
    candidate = {"content": "event", "score": 0.4, "event_time_iso": "2026-07-17T12:00:00"}

    result = apply_temporal_boost(
        "explicit range",
        [candidate],
        TemporalBoostConfig(enabled=True, time_range=time_range),
    )

    assert result[0]["temporal_boost"] == pytest.approx(0.05)
