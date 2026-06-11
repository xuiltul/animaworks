from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Rule-based query expansion for Legacy memory retrieval."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_QUOTE_RE = re.compile(r"[\"']([^\"']{2,120})[\"']")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+'-]*|\d{4}|[\u3040-\u30ff\u3400-\u9fff]{2,}")
_NUMBER_WORDS: dict[str, int] = {
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
_JP_NUMBER_WORDS: dict[str, int] = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十" + "一": 11,
    "十" + "二": 12,
}
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "in",
        "is",
        "it",
        "last",
        "many",
        "me",
        "of",
        "on",
        "or",
        "please",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    },
)


@dataclass(frozen=True)
class ExpandedQuery:
    """Deterministic retrieval query expansion output."""

    original: str
    search_text: str
    time_hint_start: str | None
    time_hint_end: str | None
    boost_phrases: tuple[str, ...] = ()
    bm25_extra: tuple[str, ...] = ()


DateRangeFactory = Callable[[date, re.Match[str]], tuple[date, date]]


@dataclass(frozen=True)
class _DateRule:
    pattern: re.Pattern[str]
    factory: DateRangeFactory


def expand_query(query: str, *, reference_time: datetime | None = None) -> ExpandedQuery:
    """Expand a user query with deterministic temporal and keyword hints.

    English and Japanese relative-date patterns are applied together so mixed
    language queries still receive deterministic time hints. Already expanded
    ISO dates are left untouched.
    """
    original = str(query or "")
    reference = _ensure_reference_time(reference_time)
    boost_phrases = _quoted_phrases(original)
    bm25_extra = _content_tokens(original)

    time_start: str | None = None
    time_end: str | None = None
    expanded_dates: tuple[str, ...] = ()
    if not _ISO_DATE_RE.search(original):
        date_range = _match_relative_date(original, reference.date())
        if date_range is not None:
            start, end = date_range
            time_start = start.isoformat()
            time_end = end.isoformat()
            expanded_dates = (time_start,) if time_start == time_end else (time_start, time_end)

    pieces = [original]
    pieces.extend(expanded_dates)
    pieces.extend(boost_phrases)
    pieces.extend(bm25_extra)
    search_text = " ".join(part.strip() for part in pieces if part and part.strip())
    return ExpandedQuery(
        original=original,
        search_text=search_text,
        time_hint_start=time_start,
        time_hint_end=time_end,
        boost_phrases=boost_phrases,
        bm25_extra=bm25_extra,
    )


def coerce_reference_time(value: Any | None) -> datetime | None:
    """Coerce API-provided reference time values into datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _ensure_reference_time(datetime.fromisoformat(text))
    except ValueError:
        try:
            return datetime.combine(date.fromisoformat(text), time.min, tzinfo=UTC)
        except ValueError:
            return None


def filter_ranked_lists_by_time_hint(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    time_hint_start: str | None,
    time_hint_end: str | None,
    widen_days: int = 7,
) -> list[list[dict[str, Any]]]:
    """Filter candidates with event-time metadata to a widened hint window."""
    if not time_hint_start and not time_hint_end:
        return ranked_lists
    start = _parse_date_hint(time_hint_start)
    end = _parse_date_hint(time_hint_end)
    if start is None and end is None:
        return ranked_lists
    if start is None:
        start = end
    if end is None:
        end = start
    assert start is not None and end is not None
    if start > end:
        start, end = end, start
    start = start - timedelta(days=max(0, int(widen_days)))
    end = end + timedelta(days=max(0, int(widen_days)))

    filtered: list[list[dict[str, Any]]] = []
    for ranked in ranked_lists:
        kept = [item for item in ranked if _candidate_in_time_window(item, start=start, end=end)]
        if kept:
            filtered.append(kept)
    return filtered


def _ensure_reference_time(reference_time: datetime | None) -> datetime:
    if reference_time is None:
        return datetime.now(UTC)
    if reference_time.tzinfo is None:
        return reference_time.replace(tzinfo=UTC)
    return reference_time


def _quoted_phrases(query: str) -> tuple[str, ...]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in _QUOTE_RE.finditer(query):
        phrase = re.sub(r"\s+", " ", match.group(1).strip())
        key = phrase.lower()
        if phrase and key not in seen:
            seen.add(key)
            phrases.append(phrase)
    return tuple(phrases)


def _content_tokens(query: str) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0).strip("'").lower()
        token = token.removesuffix("'s")
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tuple(tokens)


def _match_relative_date(query: str, ref: date) -> tuple[date, date] | None:
    for rule in _DATE_RULES:
        match = rule.pattern.search(query)
        if match:
            return rule.factory(ref, match)
    return None


def _day_offset(days: int) -> DateRangeFactory:
    def factory(ref: date, _match: re.Match[str]) -> tuple[date, date]:
        target = ref + timedelta(days=days)
        return target, target

    return factory


def _last_n_days(default_days: int) -> DateRangeFactory:
    def factory(ref: date, match: re.Match[str]) -> tuple[date, date]:
        count = _match_count(match, default_days)
        return ref - timedelta(days=count), ref

    return factory


def _ago(unit: str) -> DateRangeFactory:
    def factory(ref: date, match: re.Match[str]) -> tuple[date, date]:
        count = _match_count(match, 1)
        if unit == "day":
            target = ref - timedelta(days=count)
            return target, target
        if unit == "week":
            end = ref - timedelta(days=max(1, count - 1) * 7)
            start = ref - timedelta(days=count * 7)
            return start, end
        if unit == "month":
            end = ref - timedelta(days=max(1, count - 1) * 30)
            start = ref - timedelta(days=count * 30)
            return start, end
        end = _shift_year(ref, -(count - 1))
        start = _shift_year(ref, -count)
        return start, end

    return factory


def _this_week(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    return ref - timedelta(days=ref.weekday()), ref


def _next_week(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    start = ref + timedelta(days=1)
    return start, start + timedelta(days=6)


def _previous_week(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    end = ref - timedelta(days=ref.weekday() + 1)
    return end - timedelta(days=6), end


def _this_month(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    return ref.replace(day=1), ref


def _previous_month(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    first_this_month = ref.replace(day=1)
    end = first_this_month - timedelta(days=1)
    return end.replace(day=1), end


def _next_month(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    if ref.month == 12:
        start = date(ref.year + 1, 1, 1)
    else:
        start = date(ref.year, ref.month + 1, 1)
    if start.month == 12:
        next_start = date(start.year + 1, 1, 1)
    else:
        next_start = date(start.year, start.month + 1, 1)
    return start, next_start - timedelta(days=1)


def _this_year(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    return date(ref.year, 1, 1), ref


def _last_year(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    return date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31)


def _next_year(ref: date, _match: re.Match[str]) -> tuple[date, date]:
    return date(ref.year + 1, 1, 1), date(ref.year + 1, 12, 31)


def _jp_weeks_ago(ref: date, match: re.Match[str]) -> tuple[date, date]:
    count = _match_count(match, 1)
    start = ref - timedelta(days=ref.weekday() + count * 7)
    return start, start + timedelta(days=6)


def _jp_months_ago(ref: date, match: re.Match[str]) -> tuple[date, date]:
    count = _match_count(match, 1)
    year = ref.year
    month = ref.month - count
    while month <= 0:
        year -= 1
        month += 12
    start = date(year, month, 1)
    if month == 12:
        next_start = date(year + 1, 1, 1)
    else:
        next_start = date(year, month + 1, 1)
    return start, next_start - timedelta(days=1)


def _shift_year(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _match_count(match: re.Match[str], default: int) -> int:
    raw = str(match.groupdict().get("count", "") or "").lower()
    if not raw:
        return default
    if raw.isdigit():
        return max(1, int(raw))
    if raw in _JP_NUMBER_WORDS:
        return _JP_NUMBER_WORDS[raw]
    return _NUMBER_WORDS.get(raw, default)


def _parse_date_hint(value: str | None) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _candidate_in_time_window(item: dict[str, Any], *, start: date, end: date) -> bool:
    candidate_date = _candidate_event_date(item)
    if candidate_date is None:
        return True
    return start <= candidate_date <= end


def _candidate_event_date(item: dict[str, Any]) -> date | None:
    for source in (item, item.get("metadata", {})):
        if not isinstance(source, dict):
            continue
        for key in ("valid_at", "valid_at_iso", "event_time_iso", "valid_from", "recorded_at"):
            if key not in source:
                continue
            parsed = _parse_candidate_time(source.get(key), numeric_timestamp=(key == "valid_at"))
            if parsed is not None:
                return parsed
    return None


def _parse_candidate_time(value: Any, *, numeric_timestamp: bool) -> date | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC).date()
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    if numeric_timestamp and _is_numeric_timestamp(text):
        try:
            return datetime.fromtimestamp(float(text), tz=UTC).date()
        except (OSError, OverflowError, ValueError):
            return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _is_numeric_timestamp(value: str) -> bool:
    return bool(value and value.replace(".", "", 1).replace("-", "", 1).isdigit())


_COUNT = r"(?P<count>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_JP_COUNT = r"(?P<count>\d{1,3}|[一二三四五六七八九十]{1,3})"
_DATE_RULES: tuple[_DateRule, ...] = (
    _DateRule(re.compile(r"(?:一昨日|おととい)"), _day_offset(-2)),
    _DateRule(re.compile(r"昨日"), _day_offset(-1)),
    _DateRule(re.compile(r"今日"), _day_offset(0)),
    _DateRule(re.compile(r"明日"), _day_offset(1)),
    _DateRule(re.compile(r"先週"), _previous_week),
    _DateRule(re.compile(r"今週"), _this_week),
    _DateRule(re.compile(r"来週"), _next_week),
    _DateRule(re.compile(r"先月"), _previous_month),
    _DateRule(re.compile(r"今月"), _this_month),
    _DateRule(re.compile(r"来月"), _next_month),
    _DateRule(re.compile(r"去年"), _last_year),
    _DateRule(re.compile(r"今年"), _this_year),
    _DateRule(re.compile(r"来年"), _next_year),
    _DateRule(re.compile(rf"{_JP_COUNT}\s*日前"), _ago("day")),
    _DateRule(re.compile(rf"{_JP_COUNT}\s*週間前"), _jp_weeks_ago),
    _DateRule(re.compile(rf"{_JP_COUNT}\s*(?:か月|ヶ月|カ月|月)前"), _jp_months_ago),
    _DateRule(re.compile(rf"{_JP_COUNT}\s*年前"), _ago("year")),
    _DateRule(re.compile(r"\bearlier\s+today\b", re.IGNORECASE), _day_offset(0)),
    _DateRule(re.compile(r"\btoday\b", re.IGNORECASE), _day_offset(0)),
    _DateRule(re.compile(r"\byesterday\b", re.IGNORECASE), _day_offset(-1)),
    _DateRule(re.compile(r"\blast\s+night\b", re.IGNORECASE), _day_offset(-1)),
    _DateRule(re.compile(r"\btomorrow\b", re.IGNORECASE), _day_offset(1)),
    _DateRule(re.compile(r"\blast\s+week\b", re.IGNORECASE), _last_n_days(7)),
    _DateRule(re.compile(r"\bthis\s+week\b", re.IGNORECASE), _this_week),
    _DateRule(re.compile(r"\bnext\s+week\b", re.IGNORECASE), _next_week),
    _DateRule(re.compile(r"\blast\s+month\b", re.IGNORECASE), _last_n_days(30)),
    _DateRule(re.compile(r"\bthis\s+month\b", re.IGNORECASE), _this_month),
    _DateRule(re.compile(r"\bnext\s+month\b", re.IGNORECASE), _next_month),
    _DateRule(re.compile(r"\blast\s+year\b", re.IGNORECASE), _last_year),
    _DateRule(re.compile(r"\bthis\s+year\b", re.IGNORECASE), _this_year),
    _DateRule(re.compile(r"\bnext\s+year\b", re.IGNORECASE), _next_year),
    _DateRule(re.compile(rf"\b{_COUNT}\s+days?\s+ago\b", re.IGNORECASE), _ago("day")),
    _DateRule(re.compile(rf"\b{_COUNT}\s+weeks?\s+ago\b", re.IGNORECASE), _ago("week")),
    _DateRule(re.compile(rf"\b{_COUNT}\s+months?\s+ago\b", re.IGNORECASE), _ago("month")),
    _DateRule(re.compile(rf"\b{_COUNT}\s+years?\s+ago\b", re.IGNORECASE), _ago("year")),
    _DateRule(re.compile(rf"\b(?:past|last)\s+{_COUNT}\s+days?\b", re.IGNORECASE), _last_n_days(7)),
)
