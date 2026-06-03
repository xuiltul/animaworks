from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Episode heading event-time metadata helpers."""

import re
from typing import Any

from core.time_utils import ensure_aware

try:
    from dateutil import parser as dateutil_parser
except Exception:  # pragma: no cover - optional dependency fallback
    dateutil_parser = None  # type: ignore[assignment]

_LOCOMO_SESSION_HEADING_RE = re.compile(
    r"^##\s+Session\s+(?P<session>\d+)\s+(?:—|-)\s+(?P<when>.+?)\s*$",
    re.IGNORECASE,
)


def apply_episode_heading_event_time(metadata: dict[str, Any], heading: str) -> None:
    """Attach LoCoMo-style session event-time metadata to an episode chunk."""
    match = _LOCOMO_SESSION_HEADING_RE.match(heading.strip())
    if not match:
        return

    metadata["session_index"] = int(match.group("session"))
    event_time_text = match.group("when").strip()
    metadata["event_time_text"] = event_time_text

    if dateutil_parser is None:
        metadata["event_time_iso"] = ""
        metadata["event_time_parse_error"] = True
        return

    try:
        parsed = ensure_aware(dateutil_parser.parse(event_time_text))
    except (ValueError, TypeError, OverflowError):
        metadata["event_time_iso"] = ""
        metadata["event_time_parse_error"] = True
        return

    metadata["event_time_iso"] = parsed.isoformat()
    metadata["valid_at"] = parsed.timestamp()
    metadata["event_time_parse_error"] = False
