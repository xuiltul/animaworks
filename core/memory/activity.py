from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Unified activity log — single timeline for all Anima interactions.

Records every interaction (user conversations, channel posts/reads, DM
send/receive, human notifications, tool usage, heartbeat/cron) as
append-only JSONL entries in ``{anima_dir}/activity_log/{date}.jsonl``.

This module serves as the single data source for the Priming layer's
"Recent Activity" channel, replacing the previously scattered transcript,
dm_log, and heartbeat_history files.
"""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.time_utils import ensure_aware, now_iso, now_jst

logger = logging.getLogger("animaworks.activity")

# Rough characters-per-token for Japanese/English mixed text.
_CHARS_PER_TOKEN = 4


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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict, omitting empty fields."""
        d = asdict(self)
        d.pop("_line_number", None)
        d.pop("_anima_name", None)
        d = {k: v for k, v in d.items() if v}
        # Rename Python field names to JSONL keys (avoid Python reserved word 'from')
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


def _get_peer(group: EntryGroup) -> str:
    """Get the peer name from the first entry of a DM group."""
    e = group.entries[0]
    if e.type == "dm_sent":
        return e.to_person
    return e.from_person


def _dm_label(peer: str, first_entry: ActivityEntry) -> str:
    """Generate a DM group label."""
    text = first_entry.summary or first_entry.content[:30]
    return f"{peer}: {text}"


def _get_task_name(group: EntryGroup) -> str:
    """Get the task_name from the first entry of a cron group."""
    return group.entries[0].meta.get("task_name", "")


def _time_diff(ts1: str, ts2: str) -> float:
    """Return the difference between two ISO timestamps in seconds."""
    try:
        t1 = ensure_aware(datetime.fromisoformat(ts1))
        t2 = ensure_aware(datetime.fromisoformat(ts2))
        return abs((t2 - t1).total_seconds())
    except (ValueError, TypeError):
        return float("inf")


def _set_source_lines(group: EntryGroup) -> None:
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
            # Consecutive lines: L2-6
            ref = f"L{lines[0]}-{lines[-1]}"
        else:
            # Non-consecutive: L1,3,5
            ref = "L" + ",".join(str(n) for n in lines)
        parts.append(f"activity_log/{date_str}.jsonl#{ref}")

    group.source_lines = " + ".join(parts)


# ── ActivityLogger ────────────────────────────────────────────


class ActivityLogger:
    """Unified activity recorder for a single Anima.

    All interactions are written as append-only JSONL to
    ``{anima_dir}/activity_log/{date}.jsonl``.
    """

    def __init__(self, anima_dir: Path) -> None:
        self.anima_dir = anima_dir
        self._log_dir = anima_dir / "activity_log"

    # ── Recording ─────────────────────────────────────────────

    def log(
        self,
        event_type: str,
        *,
        content: str = "",
        summary: str = "",
        from_person: str = "",
        to_person: str = "",
        channel: str = "",
        tool: str = "",
        via: str = "",
        meta: dict[str, Any] | None = None,
    ) -> ActivityEntry:
        """Record an activity entry.

        Args:
            event_type: Event kind (e.g. ``message_received``,
                ``dm_sent``, ``channel_post``, ``tool_use``).
            content: Full content text (may be long).
            summary: Short description (preferred for large content).
            from_person: Sender name.
            to_person: Recipient name.
            channel: Channel name (``chat``, ``general``, etc.).
            tool: Tool name (for ``tool_use`` events).
            via: Delivery channel (for ``human_notify`` events).
            meta: Arbitrary metadata dict.

        Returns:
            The recorded :class:`ActivityEntry`.
        """
        entry = ActivityEntry(
            ts=now_iso(),
            type=event_type,
            content=content,
            summary=summary,
            from_person=from_person,
            to_person=to_person,
            channel=channel,
            tool=tool,
            via=via,
            meta=meta or {},
        )
        self._append(entry)
        return entry

    def _append(self, entry: ActivityEntry) -> None:
        """Append *entry* to today's JSONL file with fsync."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            date_str = entry.ts[:10]
            path = self._log_dir / f"{date_str}.jsonl"
            line = json.dumps(entry.to_dict(), ensure_ascii=False)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.exception("Failed to append activity log")

    # ── Retrieval ─────────────────────────────────────────────

    def _load_entries(
        self,
        days: int = 2,
        hours: int | None = None,
        types: list[str] | None = None,
        involving: str | None = None,
    ) -> list[ActivityEntry]:
        """Load all matching entries (no limit/offset).

        Args:
            days: Number of past days to scan (including today).
            hours: If given, overrides *days* with ``ceil(hours/24)``
                and filters entries to the last *hours* hours.
            types: If given, only include these event types.
            involving: If given, only entries where *from_person*,
                *to_person*, or *channel* matches this value.

        Returns:
            Chronologically sorted list of all matching entries.
        """
        entries: list[ActivityEntry] = []
        now = now_jst()
        today = now.date()
        type_set = set(types) if types else None

        # Determine scan range
        scan_days = days
        cutoff: datetime | None = None
        if hours is not None:
            # +1 to handle midnight boundary (e.g. 00:15 - 1h = yesterday)
            scan_days = math.ceil(hours / 24) + 1
            cutoff = now - timedelta(hours=hours)

        for day_offset in range(scan_days):
            target = today - timedelta(days=day_offset)
            path = self._log_dir / f"{target.isoformat()}.jsonl"
            if not path.exists():
                continue
            try:
                for line_num, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if type_set and raw.get("type") not in type_set:
                        continue
                    if involving and not self._involves(raw, involving):
                        continue
                    if cutoff:
                        try:
                            ts = datetime.fromisoformat(raw.get("ts", ""))
                            # Normalise both sides to aware for comparison
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=cutoff.tzinfo)
                            if ts < cutoff:
                                continue
                        except (ValueError, TypeError):
                            logger.debug("Failed to parse timestamp for cutoff filtering", exc_info=True)
                    # Map JSONL keys to Python field names
                    if "from" in raw:
                        raw["from_person"] = raw.pop("from")
                    if "to" in raw:
                        raw["to_person"] = raw.pop("to")
                    entry = ActivityEntry(**{
                        k: v for k, v in raw.items()
                        if k in ActivityEntry.__dataclass_fields__
                    })
                    entry._line_number = line_num
                    entries.append(entry)
            except Exception:
                logger.exception("Failed to read activity log %s", path)

        # Chronological sort, newest last
        entries.sort(key=lambda e: e.ts)
        return entries

    def recent(
        self,
        days: int = 2,
        limit: int = 100,
        types: list[str] | None = None,
        involving: str | None = None,
    ) -> list[ActivityEntry]:
        """Load recent entries from the activity log.

        Args:
            days: Number of past days to scan (including today).
            limit: Maximum entries to return.
            types: If given, only include these event types.
            involving: If given, only entries where *from_person*,
                *to_person*, or *channel* matches this value.

        Returns:
            List of :class:`ActivityEntry` in chronological order.
        """
        entries = self._load_entries(
            days=days, types=types, involving=involving,
        )

        # Return the most recent *limit* entries
        if len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def recent_page(
        self,
        *,
        days: int = 2,
        hours: int | None = None,
        limit: int = 200,
        offset: int = 0,
        types: list[str] | None = None,
        involving: str | None = None,
    ) -> ActivityPage:
        """Load recent entries with pagination for API responses.

        Args:
            days: Number of past days to scan (including today).
            hours: If given, overrides *days* and filters to last N hours.
            limit: Page size (default 200, max 500).
            offset: Number of entries to skip from newest.
            types: If given, only include these event types.
            involving: If given, only entries where *from_person*,
                *to_person*, or *channel* matches this value.

        Returns:
            :class:`ActivityPage` with paginated entries (newest first).
        """
        offset = max(0, offset)

        all_entries = self._load_entries(
            days=days, hours=hours, types=types, involving=involving,
        )
        total = len(all_entries)

        # Reverse to newest-first for pagination
        all_entries.reverse()

        # limit <= 0 means "return all" (used for cross-anima merging)
        if limit <= 0:
            page = all_entries[offset:]
            return ActivityPage(
                entries=page,
                total=total,
                offset=offset,
                limit=total,
                has_more=False,
            )

        limit = min(limit, 500)
        page = all_entries[offset:offset + limit]

        return ActivityPage(
            entries=page,
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        )

    @staticmethod
    def _involves(raw: dict[str, Any], name: str) -> bool:
        """Return True if *name* appears in from/to/channel fields."""
        return (
            raw.get("from", raw.get("from_person")) == name
            or raw.get("to", raw.get("to_person")) == name
            or raw.get("channel") == name
        )

    # ── Priming formatter ─────────────────────────────────────

    def format_for_priming(
        self,
        entries: list[ActivityEntry],
        budget_tokens: int = 1300,
        content_trim: int = 200,
    ) -> str:
        """Format activity entries for system prompt injection.

        Groups related entries (DM conversations, heartbeats, cron tasks)
        and renders a compact timeline, truncated to *budget_tokens*.

        Args:
            entries: Entries to format (should be chronological).
            budget_tokens: Target token budget.
            content_trim: Maximum characters for content display in
                each entry.  Set to ``0`` for no trim.

        Returns:
            Formatted string.
        """
        if not entries:
            return ""

        groups = self._group_entries(entries)
        max_chars = budget_tokens * _CHARS_PER_TOKEN
        lines: list[str] = []
        total_chars = 0

        # Process groups newest-first (budget cuts oldest groups)
        for group in reversed(groups):
            formatted = self._format_group(group, content_trim=content_trim)
            if total_chars + len(formatted) + 1 > max_chars:
                break
            lines.append(formatted)
            total_chars += len(formatted) + 1  # +1 for newline

        if not lines:
            return ""

        lines.reverse()
        return "\n".join(lines)

    @staticmethod
    def _format_entry(entry: ActivityEntry, content_trim: int = 200) -> str:
        """Format a single entry as a concise timeline line."""
        ts = entry.ts[11:16] if len(entry.ts) >= 16 else entry.ts
        text = entry.summary or entry.content

        # Truncate long content with pointer (0 = no trim)
        if content_trim > 0 and len(text) > content_trim:
            date_str = entry.ts[:10] if len(entry.ts) >= 10 else "unknown"
            pointer = f"\n  -> activity_log/{date_str}.jsonl"
            trim_window = text[:content_trim]
            last_boundary = max(
                trim_window.rfind("\u3002"),
                trim_window.rfind("\n"),
                trim_window.rfind(". "),
            )
            if last_boundary > content_trim * 0.5:
                text = trim_window[:last_boundary + 1] + pointer
            else:
                text = trim_window[:max(1, content_trim - 20)] + "..." + pointer

        type_map: dict[str, str] = {
            "message_received": "MSG<",
            "response_sent": "MSG>",
            "channel_read": "CH.R",
            "channel_post": "CH.W",
            "dm_received": "DM<",
            "dm_sent": "DM>",
            "human_notify": "NTFY",
            "tool_use": "TOOL",
            "heartbeat_start": "HB",
            "heartbeat_end": "HB",
            "cron_executed": "CRON",
            "memory_write": "MEM",
            "error": "ERR",
            "issue_resolved": "RSLV",
            "task_created": "TSK+",
            "task_updated": "TSK~",
        }
        icon = type_map.get(entry.type, "•")

        # Build context suffix
        context_parts: list[str] = []
        if entry.from_person:
            context_parts.append(f"from:{entry.from_person}")
        if entry.to_person:
            context_parts.append(f"to:{entry.to_person}")
        if entry.channel:
            context_parts.append(f"#{entry.channel}")
        if entry.tool:
            context_parts.append(f"tool:{entry.tool}")
        if entry.via:
            context_parts.append(f"via:{entry.via}")

        ctx = f"({', '.join(context_parts)})" if context_parts else ""
        return f"[{ts}] {icon} {entry.type}{ctx}: {text}"

    # ── Grouping ─────────────────────────────────────────────

    @staticmethod
    def _group_entries(
        entries: list[ActivityEntry],
        time_gap_minutes: int = 30,
    ) -> list[EntryGroup]:
        """Group related activity entries for compact display.

        Grouping rules:
        1. DM: Same peer, within time_gap_minutes → 1 group
        2. HB: Consecutive heartbeat_start/heartbeat_end only
        3. CRON: Same task_name → 1 group
        4. Others: Single-entry group
        """
        groups: list[EntryGroup] = []
        current_group: EntryGroup | None = None
        gap_seconds = time_gap_minutes * 60

        for entry in entries:
            entry_type = entry.type

            # DM grouping
            if entry_type in ("dm_sent", "dm_received"):
                peer = entry.to_person if entry_type == "dm_sent" else entry.from_person
                if (
                    current_group
                    and current_group.type == "dm"
                    and _get_peer(current_group) == peer
                    and _time_diff(current_group.end_ts, entry.ts) <= gap_seconds
                ):
                    current_group.entries.append(entry)
                    current_group.end_ts = entry.ts
                    continue
                # New DM group
                if current_group:
                    groups.append(current_group)
                current_group = EntryGroup(
                    type="dm",
                    start_ts=entry.ts,
                    end_ts=entry.ts,
                    entries=[entry],
                    label=_dm_label(peer, entry),
                    source_lines="",
                )
                continue

            # HB grouping
            if entry_type in ("heartbeat_start", "heartbeat_end"):
                if current_group and current_group.type == "hb":
                    current_group.entries.append(entry)
                    current_group.end_ts = entry.ts
                    continue
                if current_group:
                    groups.append(current_group)
                current_group = EntryGroup(
                    type="hb",
                    start_ts=entry.ts,
                    end_ts=entry.ts,
                    entries=[entry],
                    label="",
                    source_lines="",
                )
                continue

            # CRON grouping
            if entry_type == "cron_executed":
                task_name = entry.meta.get("task_name", "")
                if (
                    current_group
                    and current_group.type == "cron"
                    and _get_task_name(current_group) == task_name
                ):
                    current_group.entries.append(entry)
                    current_group.end_ts = entry.ts
                    continue
                if current_group:
                    groups.append(current_group)
                current_group = EntryGroup(
                    type="cron",
                    start_ts=entry.ts,
                    end_ts=entry.ts,
                    entries=[entry],
                    label=task_name,
                    source_lines="",
                )
                continue

            # Other: close current group, create single-entry group
            if current_group:
                groups.append(current_group)
                current_group = None
            groups.append(EntryGroup(
                type="single",
                start_ts=entry.ts,
                end_ts=entry.ts,
                entries=[entry],
                label="",
                source_lines="",
            ))

        if current_group:
            groups.append(current_group)

        # Generate source_lines for all groups
        for group in groups:
            _set_source_lines(group)

        return groups

    @staticmethod
    def _format_group(group: EntryGroup, content_trim: int = 200) -> str:
        """Format a group of entries for priming display."""
        start_time = group.start_ts[11:16] if len(group.start_ts) >= 16 else group.start_ts
        end_time = group.end_ts[11:16] if len(group.end_ts) >= 16 else group.end_ts

        time_range = (
            f"[{start_time}-{end_time}]"
            if start_time != end_time
            else f"[{start_time}]"
        )

        if group.type == "dm":
            # Header: [HH:MM-HH:MM] DM {peer}:
            peer = _get_peer(group)
            lines = [f"{time_range} DM {peer}:"]
            # Child lines: indented DM direction + summary
            for e in group.entries:
                direction = "DM<" if e.type == "dm_received" else "DM>"
                text = e.summary or e.content[:100]
                if len(text) > 100:
                    text = text[:100]
                lines.append(f"  {direction} {text}")
            # Pointer line
            if group.source_lines:
                lines.append(f"  -> {group.source_lines}")
            return "\n".join(lines)

        if group.type == "hb":
            # Use heartbeat_end summary if available
            hb_summary = ""
            for e in group.entries:
                if e.type == "heartbeat_end":
                    hb_summary = (e.summary or e.content)[:50]
                    break
            header = f"{time_range} HB: {hb_summary}" if hb_summary else f"{time_range} HB"
            lines = [header]
            if group.source_lines:
                lines.append(f"  -> {group.source_lines}")
            return "\n".join(lines)

        if group.type == "cron":
            # Get exit code from first entry if available
            exit_code = group.entries[0].meta.get("exit_code", "")
            exit_info = f": exit={exit_code}" if exit_code != "" else ""
            header = f"{time_range} CRON {group.label}{exit_info}"
            lines = [header]
            if group.source_lines:
                lines.append(f"  -> {group.source_lines}")
            return "\n".join(lines)

        # Single entry: use _format_entry
        return ActivityLogger._format_entry(group.entries[0], content_trim=content_trim)
