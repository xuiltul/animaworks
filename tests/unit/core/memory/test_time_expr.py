from __future__ import annotations

from datetime import datetime

import pytest

from core.memory.retrieval.time_expr import TimeRange, extract_time_range

NOW = datetime(2026, 7, 18, 15, 42, 30)
DAY_END = (23, 59, 59, 999999)


def _expected(start: tuple[int, int, int], end: tuple[int, int, int] | None = None) -> TimeRange:
    end = end or start
    return TimeRange(
        start=datetime(*start),
        end=datetime(*end, *DAY_END),
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("2026-07-18 の議事録", _expected((2026, 7, 18))),
        ("2026/7/18 の議事録", _expected((2026, 7, 18))),
        ("7月18日の議事録", _expected((2026, 7, 18))),
        ("notes from July 18", _expected((2026, 7, 18))),
        ("2026年7月の議事録", _expected((2026, 7, 1), (2026, 7, 31))),
    ],
)
def test_extract_absolute_expressions(query: str, expected: TimeRange) -> None:
    assert extract_time_range(query, now=NOW) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("今日の予定", _expected((2026, 7, 18))),
        ("昨日の話", _expected((2026, 7, 17))),
        ("一昨日の話", _expected((2026, 7, 16))),
        ("今週の予定", _expected((2026, 7, 13), (2026, 7, 19))),
        ("先週のタスク", _expected((2026, 7, 6), (2026, 7, 12))),
        ("今月の予定", _expected((2026, 7, 1), (2026, 7, 31))),
        ("先月の予定", _expected((2026, 6, 1), (2026, 6, 30))),
        ("3日前の話", _expected((2026, 7, 15))),
        ("2週間前の話", _expected((2026, 6, 29), (2026, 7, 5))),
        ("what happened yesterday?", _expected((2026, 7, 17))),
        ("tasks from last week", _expected((2026, 7, 6), (2026, 7, 12))),
        ("notes from 3 days ago", _expected((2026, 7, 15))),
    ],
)
def test_extract_relative_expressions(query: str, expected: TimeRange) -> None:
    assert extract_time_range(query, now=NOW) == expected


@pytest.mark.parametrize("query", ["最近の話", "さっきの話", "直近のタスク", "mentioned recently", "sent just now"])
def test_extract_recency_expressions(query: str) -> None:
    assert extract_time_range(query, now=NOW) == TimeRange(start=None, end=None, recency=True)


def test_extract_returns_none_for_unmatched_query() -> None:
    assert extract_time_range("プロジェクトの設計方針", now=NOW) is None


def test_now_is_mandatory() -> None:
    with pytest.raises(TypeError):
        extract_time_range("昨日")  # type: ignore[call-arg]


def test_aware_now_is_normalized_to_naive_local_time() -> None:
    aware_now = datetime.fromisoformat("2026-07-18T15:42:30+09:00")
    result = extract_time_range("今日", now=aware_now)

    assert result == _expected((2026, 7, 18))
    assert result is not None
    assert result.start is not None
    assert result.start.tzinfo is None
