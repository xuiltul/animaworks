from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Activity log data models, constants, and shared helpers.

Internal module — import from :mod:`core.memory.activity` instead.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from core.time_utils import ensure_aware

# Rough characters-per-token for Japanese/English mixed text.
CHARS_PER_TOKEN = 4

# Alias mapping: old event type → new canonical name.
EVENT_TYPE_ALIASES: dict[str, str] = {
    "dm_sent": "message_sent",
    "dm_received": "message_received",
}


def resolve_type_filter(types: list[str] | None) -> set[str] | None:
    """Expand a type filter list to include aliases.

    When the caller requests ``["message_sent"]``, the resolved set
    also includes ``"dm_sent"`` so that older log entries still match.
    Conversely, requesting ``["dm_sent"]`` also matches ``"message_sent"``.
    """
    if types is None:
        return None
    resolved: set[str] = set()
    for t in types:
        resolved.add(t)
        for old, new in EVENT_TYPE_ALIASES.items():
            if t == new:
                resolved.add(old)
            elif t == old:
                resolved.add(new)
    return resolved


# ── Data model ────────────────────────────────────────────────


@dataclass
class ActivityEntry:
    """A single entry in the unified activity log."""

    ts: str
    type: str
    content: str = ""
    summary: str = ""
    from_person: str = ""
    to_person: str = ""
    channel: str = ""
    tool: str = ""
    via: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    _line_number: int = field(default=0, init=False, repr=False)
    _anima_name: str = field(default="", init=False, repr=False)
    _tool_result_data: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict, omitting empty fields."""
        d = asdict(self)
        d.pop("_line_number", None)
        d.pop("_anima_name", None)
        d = {k: v for k, v in d.items() if v}
        if "from_person" in d:
            d["from"] = d.pop("from_person")
        if "to_person" in d:
            d["to"] = d.pop("to_person")
        return d

    def to_api_dict(self, anima_name: str = "") -> dict[str, Any]:
        """Serialise for API response, including all fields."""
        name = anima_name or self._anima_name
        d: dict[str, Any] = {
            "id": f"{name}:{self.ts}:{self.type}:{self._line_number}",
            "ts": self.ts,
            "type": self.type,
            "anima": name,
            "summary": self.summary,
            "content": self.content,
            "from_person": self.from_person,
            "to_person": self.to_person,
            "channel": self.channel,
            "tool": self.tool,
            "via": self.via,
            "meta": self.meta,
        }
        return d


@dataclass
class EntryGroup:
    """A group of related activity entries for compact priming display."""

    type: str                     # "dm", "hb", "cron", "single"
    start_ts: str                 # Group start timestamp
    end_ts: str                   # Group end timestamp
    entries: list[ActivityEntry]  # Entries in this group
    label: str                    # Group label (e.g. "yuki: boto3問題")
    source_lines: str             # JSONL line number range (e.g. "L2-6")


@dataclass
class ActivityPage:
    """Paginated result for API responses."""

    entries: list[ActivityEntry]
    total: int
    offset: int
    limit: int
    has_more: bool


# ── Grouping helpers ─────────────────────────────────────────


def get_peer(group: EntryGroup) -> str:
    """Get the peer name from the first entry of a DM group."""
    e = group.entries[0]
    if e.type in ("dm_sent", "message_sent"):
        return e.to_person
    return e.from_person


def dm_label(peer: str, first_entry: ActivityEntry) -> str:
    """Generate a DM group label."""
    text = first_entry.summary or first_entry.content[:30]
    return f"{peer}: {text}"


def get_task_name(group: EntryGroup) -> str:
    """Get the task_name from the first entry of a cron group."""
    return group.entries[0].meta.get("task_name", "")


def time_diff(ts1: str, ts2: str) -> float:
    """Return the difference between two ISO timestamps in seconds."""
    try:
        t1 = ensure_aware(datetime.fromisoformat(ts1))
        t2 = ensure_aware(datetime.fromisoformat(ts2))
        return abs((t2 - t1).total_seconds())
    except (ValueError, TypeError):
        return float("inf")


def set_source_lines(group: EntryGroup) -> None:
    """Set source_lines from entry _line_number values."""
    by_date: dict[str, list[int]] = {}
    for e in group.entries:
        if e._line_number > 0:
            date_str = e.ts[:10] if len(e.ts) >= 10 else "unknown"
            by_date.setdefault(date_str, []).append(e._line_number)

    parts: list[str] = []
    for date_str, lines in sorted(by_date.items()):
        lines.sort()
        if len(lines) == 1:
            ref = f"L{lines[0]}"
        elif lines[-1] - lines[0] == len(lines) - 1:
            ref = f"L{lines[0]}-{lines[-1]}"
        else:
            ref = "L" + ",".join(str(n) for n in lines)
        parts.append(f"activity_log/{date_str}.jsonl#{ref}")

    group.source_lines = " + ".join(parts)


def find_tool_result_fallback(
    entries: list[ActivityEntry],
    tool_use_entry: ActivityEntry,
) -> ActivityEntry | None:
    """Fallback: find tool_result by timestamp proximity + tool name."""
    found = False
    for e in entries:
        if e is tool_use_entry:
            found = True
            continue
        if not found:
            continue
        if e.type == "tool_result" and e.tool == tool_use_entry.tool:
            if time_diff(tool_use_entry.ts, e.ts) < 300:
                return e
        if e.type in ("message_received", "response_sent"):
            break
    return None
