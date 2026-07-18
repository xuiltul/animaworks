from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Deterministic Japanese and English time-expression extraction."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from re import Match


@dataclass(frozen=True)
class TimeRange:
    """An inclusive local-time range or a range-less recency intent."""

    start: datetime | None
    end: datetime | None
    recency: bool = False


_MONTH_NAMES: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_ENGLISH_COUNTS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_ENGLISH_MONTH_PATTERN = "|".join(_MONTH_NAMES)
_ENGLISH_COUNT_PATTERN = "|".join((r"\d{1,3}", *_ENGLISH_COUNTS))
_JAPANESE_COUNT_PATTERN = r"\d{1,3}|[一二三四五六七八九十]{1,3}"

_ISO_DATE_RE = re.compile(r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?!\d)")
_SLASH_DATE_RE = re.compile(r"(?<!\d)(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})(?!\d)")
_JP_FULL_DATE_RE = re.compile(r"(?<!\d)(?P<year>\d{4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日")
_JP_MONTH_DAY_RE = re.compile(r"(?<!年)(?<!\d)(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日")
_EN_MONTH_DAY_RE = re.compile(
    rf"\b(?P<month_name>{_ENGLISH_MONTH_PATTERN})\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
    r"(?:\s*,?\s*(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_JP_YEAR_MONTH_RE = re.compile(r"(?<!\d)(?P<year>\d{4})年\s*(?P<month>\d{1,2})月(?!\s*\d{1,2}日)")

_RECENCY_RE = re.compile(r"最近|さっき|直近|\brecently\b|\bjust\s+now\b", re.IGNORECASE)


def extract_time_range(query: str, *, now: datetime) -> TimeRange | None:
    """Resolve the first supported time expression relative to ``now``.

    ``now`` is deliberately mandatory so callers and tests control the clock.
    Returned datetimes are naive local values, even if the supplied value has
    timezone information.
    """

    text = str(query or "")
    reference = now.replace(tzinfo=None)

    absolute = _extract_absolute(text, reference)
    if absolute is not None:
        return absolute

    relative = _extract_relative(text, reference.date())
    if relative is not None:
        return relative

    if _RECENCY_RE.search(text):
        return TimeRange(start=None, end=None, recency=True)
    return None


def _extract_absolute(query: str, reference: datetime) -> TimeRange | None:
    for pattern, factory in (
        (_ISO_DATE_RE, _numeric_date),
        (_SLASH_DATE_RE, _numeric_date),
        (_JP_FULL_DATE_RE, _numeric_date),
        (_JP_MONTH_DAY_RE, lambda match: _numeric_date(match, default_year=reference.year)),
        (_EN_MONTH_DAY_RE, lambda match: _english_date(match, default_year=reference.year)),
        (_JP_YEAR_MONTH_RE, _numeric_month),
    ):
        match = pattern.search(query)
        if match is None:
            continue
        resolved = factory(match)
        if resolved is not None:
            return resolved
    return None


def _numeric_date(match: Match[str], *, default_year: int | None = None) -> TimeRange | None:
    year_text = match.groupdict().get("year")
    if not year_text and default_year is None:
        return None
    return _day_range(
        int(year_text) if year_text else default_year,
        int(match.group("month")),
        int(match.group("day")),
    )


def _english_date(match: Match[str], *, default_year: int) -> TimeRange | None:
    year_text = match.groupdict().get("year")
    return _day_range(
        int(year_text) if year_text else default_year,
        _MONTH_NAMES[match.group("month_name").lower()],
        int(match.group("day")),
    )


def _numeric_month(match: Match[str]) -> TimeRange | None:
    try:
        start_date = date(int(match.group("year")), int(match.group("month")), 1)
    except ValueError:
        return None
    return _month_range(start_date.year, start_date.month)


def _extract_relative(query: str, reference: date) -> TimeRange | None:
    fixed_rules: tuple[tuple[re.Pattern[str], Callable[[date], TimeRange]], ...] = (
        (re.compile(r"一昨日|おととい"), lambda ref: _relative_day(ref, -2)),
        (re.compile(r"昨日"), lambda ref: _relative_day(ref, -1)),
        (re.compile(r"今日"), lambda ref: _relative_day(ref, 0)),
        (re.compile(r"先週"), lambda ref: _week_range(ref, -1)),
        (re.compile(r"今週"), lambda ref: _week_range(ref, 0)),
        (re.compile(r"先月"), lambda ref: _relative_month(ref, -1)),
        (re.compile(r"今月"), lambda ref: _relative_month(ref, 0)),
        (re.compile(r"\byesterday\b", re.IGNORECASE), lambda ref: _relative_day(ref, -1)),
        (re.compile(r"\blast\s+week\b", re.IGNORECASE), lambda ref: _week_range(ref, -1)),
    )
    for pattern, factory in fixed_rules:
        if pattern.search(query):
            return factory(reference)

    jp_days = re.search(rf"(?P<count>{_JAPANESE_COUNT_PATTERN})\s*日前", query)
    if jp_days:
        return _relative_day(reference, -_parse_japanese_count(jp_days.group("count")))

    jp_weeks = re.search(rf"(?P<count>{_JAPANESE_COUNT_PATTERN})\s*週間前", query)
    if jp_weeks:
        return _week_range(reference, -_parse_japanese_count(jp_weeks.group("count")))

    en_days = re.search(
        rf"\b(?P<count>{_ENGLISH_COUNT_PATTERN})\s+days?\s+ago\b",
        query,
        re.IGNORECASE,
    )
    if en_days:
        return _relative_day(reference, -_parse_english_count(en_days.group("count")))

    return None


def _relative_day(reference: date, offset: int) -> TimeRange:
    target = reference + timedelta(days=offset)
    resolved = _day_range(target.year, target.month, target.day)
    assert resolved is not None
    return resolved


def _week_range(reference: date, week_offset: int) -> TimeRange:
    monday = reference - timedelta(days=reference.weekday()) + timedelta(weeks=week_offset)
    sunday = monday + timedelta(days=6)
    return TimeRange(start=datetime.combine(monday, time.min), end=datetime.combine(sunday, time.max))


def _relative_month(reference: date, month_offset: int) -> TimeRange:
    month_index = reference.year * 12 + reference.month - 1 + month_offset
    year, zero_based_month = divmod(month_index, 12)
    return _month_range(year, zero_based_month + 1)


def _month_range(year: int, month: int) -> TimeRange:
    start_date = date(year, month, 1)
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    end_date = next_month - timedelta(days=1)
    return TimeRange(start=datetime.combine(start_date, time.min), end=datetime.combine(end_date, time.max))


def _day_range(year: int | None, month: int, day: int) -> TimeRange | None:
    if year is None:
        return None
    try:
        value = date(year, month, day)
    except ValueError:
        return None
    return TimeRange(start=datetime.combine(value, time.min), end=datetime.combine(value, time.max))


def _parse_english_count(raw: str) -> int:
    normalized = raw.lower()
    return int(normalized) if normalized.isdigit() else _ENGLISH_COUNTS[normalized]


def _parse_japanese_count(raw: str) -> int:
    if raw.isdigit():
        return int(raw)
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if raw == "十":
        return 10
    if "十" in raw:
        tens, ones = raw.split("十", 1)
        return (digits.get(tens, 1) * 10) + digits.get(ones, 0)
    value = 0
    for character in raw:
        value = value * 10 + digits[character]
    return value
