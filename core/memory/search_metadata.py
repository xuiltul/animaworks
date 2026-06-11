from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Search-result freshness and provenance display helpers."""

from datetime import UTC, date, datetime
from typing import Any

_EXTERNAL_ORIGINS = frozenset(
    {
        "external_platform",
        "external_web",
        "mixed",
        "consolidation_external",
    }
)


def format_result_metadata_line(result: dict[str, Any]) -> str:
    """Return a compact metadata line for a search/priming result."""
    memory_type = _memory_type(result)
    source = str(result.get("source_file", "") or result.get("source", "") or "")
    if memory_type == "facts" or source.startswith(("facts/", "fact:")):
        return _format_fact_metadata(result)
    if memory_type in {"knowledge", "common_knowledge"} or source.startswith(("knowledge/", "common_knowledge/")):
        return _format_knowledge_metadata(result)
    return ""


def _format_fact_metadata(result: dict[str, Any]) -> str:
    valid_at = _format_date(_first_present(result, "valid_at_iso", "event_time_iso", "valid_from", "valid_at"))
    valid_until = _format_date(result.get("valid_until"))
    recorded_at = _format_date(result.get("recorded_at"))
    parts: list[str] = []
    if valid_at:
        parts.append(f"valid: {valid_at}〜{valid_until or 'present'}")
    if recorded_at:
        parts.append(f"recorded: {recorded_at}")
    return " | ".join(parts)


def _format_knowledge_metadata(result: dict[str, Any]) -> str:
    updated_at = _format_date(_first_present(result, "updated_at", "updated", "created_at", "valid_at"))
    origin = str(result.get("origin", "") or "").strip()
    parts: list[str] = []
    if updated_at:
        parts.append(f"updated: {updated_at}")
    if origin in _EXTERNAL_ORIGINS:
        parts.append(f"origin: {origin}")
    return " | ".join(parts)


def _first_present(result: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = result.get(key)
        if value not in (None, ""):
            return value
    return None


def _format_date(value: Any) -> str:
    parsed = _parse_date(value)
    return parsed.isoformat() if parsed is not None else ""


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC).date()
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    if _is_numeric_timestamp(text):
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


def _memory_type(result: dict[str, Any]) -> str:
    memory_type = str(result.get("memory_type", "") or "").strip()
    if memory_type:
        return memory_type
    source = str(result.get("source_file", "") or result.get("source", "") or "")
    if not source:
        return ""
    if source.startswith("fact:"):
        return "facts"
    return source.replace("\\", "/").split("/", 1)[0]
