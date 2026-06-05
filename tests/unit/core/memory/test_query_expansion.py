from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from core.memory.retrieval.query_expansion import coerce_reference_time, expand_query, filter_ranked_lists_by_time_hint


REFERENCE = datetime(2023, 5, 8, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("query", "expected_start", "expected_end"),
    [
        ("what happened today?", "2023-05-08", "2023-05-08"),
        ("what happened yesterday?", "2023-05-07", "2023-05-07"),
        ("what happened last night?", "2023-05-07", "2023-05-07"),
        ("what happens tomorrow?", "2023-05-09", "2023-05-09"),
        ("what happened last week?", "2023-05-01", "2023-05-08"),
        ("what happened this week?", "2023-05-08", "2023-05-08"),
        ("what happens next week?", "2023-05-09", "2023-05-15"),
        ("what happened last month?", "2023-04-08", "2023-05-08"),
        ("what happened this month?", "2023-05-01", "2023-05-08"),
        ("what happens next month?", "2023-06-01", "2023-06-30"),
        ("what happened last year?", "2022-01-01", "2022-12-31"),
        ("what happened this year?", "2023-01-01", "2023-05-08"),
        ("what happens next year?", "2024-01-01", "2024-12-31"),
        ("what happened 3 days ago?", "2023-05-05", "2023-05-05"),
        ("what happened two weeks ago?", "2023-04-24", "2023-05-01"),
        ("what happened past 5 days?", "2023-05-03", "2023-05-08"),
    ],
)
def test_expand_query_relative_date_patterns(query: str, expected_start: str, expected_end: str) -> None:
    expanded = expand_query(query, reference_time=REFERENCE)

    assert expanded.time_hint_start == expected_start
    assert expanded.time_hint_end == expected_end
    assert expected_start in expanded.search_text
    assert expected_end in expanded.search_text


def test_expand_query_yesterday_acceptance_case() -> None:
    expanded = expand_query("What did Caroline do yesterday?", reference_time=REFERENCE)

    assert expanded.time_hint_start == "2023-05-07"
    assert expanded.time_hint_end == "2023-05-07"


def test_expand_query_extracts_boost_phrases_and_content_tokens() -> None:
    expanded = expand_query('What book did Caroline call "Becoming Nicole" yesterday?', reference_time=REFERENCE)

    assert expanded.boost_phrases == ("Becoming Nicole",)
    assert "caroline" in expanded.bm25_extra
    assert "becoming" in expanded.bm25_extra
    assert "what" not in expanded.bm25_extra
    assert "Becoming Nicole" in expanded.search_text


def test_expand_query_skips_temporal_expansion_when_iso_date_present() -> None:
    expanded = expand_query("What happened on 2023-05-07 yesterday?", reference_time=REFERENCE)

    assert expanded.time_hint_start is None
    assert expanded.time_hint_end is None
    assert expanded.search_text.count("2023-05-07") == 1


def test_coerce_reference_time_accepts_api_shapes() -> None:
    assert coerce_reference_time(None) is None
    assert coerce_reference_time(date(2023, 5, 8)).isoformat() == "2023-05-08T00:00:00+00:00"
    assert coerce_reference_time("2023-05-08").isoformat() == "2023-05-08T00:00:00+00:00"
    assert coerce_reference_time("2023-05-08T12:00:00Z").isoformat() == "2023-05-08T12:00:00+00:00"
    assert coerce_reference_time("not a date") is None


def test_filter_ranked_lists_by_time_hint_keeps_matching_and_untimed_candidates() -> None:
    ranked = [
        [
            {
                "doc_id": "numeric",
                "valid_at": datetime(2023, 5, 7, 12, 0, tzinfo=UTC).timestamp(),
            },
            {"doc_id": "iso-z", "event_time_iso": "2023-05-07T09:00:00Z"},
            {"doc_id": "metadata", "metadata": {"valid_at_iso": "2023-05-07T09:00:00+00:00"}},
            {"doc_id": "untimed"},
            {"doc_id": "outside", "event_time_iso": "2023-04-01T09:00:00+00:00"},
            {"doc_id": "invalid", "event_time_iso": "not a date"},
        ]
    ]

    filtered = filter_ranked_lists_by_time_hint(
        ranked,
        time_hint_start="2023-05-07",
        time_hint_end="2023-05-07",
        widen_days=0,
    )

    assert [item["doc_id"] for item in filtered[0]] == ["numeric", "iso-z", "metadata", "untimed", "invalid"]


def test_filter_ranked_lists_by_time_hint_handles_invalid_and_reversed_hints() -> None:
    ranked = [[{"doc_id": "inside", "event_time_iso": "2023-05-07T09:00:00+00:00"}]]

    assert filter_ranked_lists_by_time_hint(ranked, time_hint_start="bad", time_hint_end=None) is ranked
    filtered = filter_ranked_lists_by_time_hint(
        ranked,
        time_hint_start="2023-05-08",
        time_hint_end="2023-05-07",
        widen_days=0,
    )
    assert filtered[0][0]["doc_id"] == "inside"


def test_expand_query_covers_month_year_boundary_patterns() -> None:
    next_month = expand_query("what happens next month?", reference_time=datetime(2023, 12, 15, tzinfo=UTC))
    two_years = expand_query("what happened 2 years ago?", reference_time=datetime(2024, 2, 29, tzinfo=UTC))
    three_months = expand_query("what happened 3 months ago?", reference_time=REFERENCE)

    assert (next_month.time_hint_start, next_month.time_hint_end) == ("2024-01-01", "2024-01-31")
    assert (two_years.time_hint_start, two_years.time_hint_end) == ("2022-02-28", "2023-02-28")
    assert (three_months.time_hint_start, three_months.time_hint_end) == ("2023-02-07", "2023-03-09")
