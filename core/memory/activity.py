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

# Alias mapping: old event type → new canonical name.
# Used for backward compatibility with existing JSONL logs.
_EVENT_TYPE_ALIASES: dict[str, str] = {
    "dm_sent": "message_sent",
    "dm_received": "message_received",
}


def _resolve_type_filter(types: list[str] | None) -> set[str] | None:
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
        for old, new in _EVENT_TYPE_ALIASES.items():
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
    if e.type in ("dm_sent", "message_sent"):
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
        except OSError as exc:
            from core.exceptions import MemoryWriteError
            logger.exception("Failed to append activity log")
            raise MemoryWriteError(f"Activity log write failed: {exc}") from exc
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
        type_set = _resolve_type_filter(types)

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

        # tool_result: compact meta-only format for consolidation
        if entry.type == "tool_result":
            return ActivityLogger._format_tool_result_entry(entry, ts)

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
            "message_sent": "MSG>",
            "response_sent": "RESP>",
            "channel_read": "CH.R",
            "channel_post": "CH.W",
            "dm_received": "MSG<",
            "dm_sent": "MSG>",
            "human_notify": "NTFY",
            "tool_use": "TOOL",
            "tool_result": "TRES",
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

    @staticmethod
    def _format_tool_result_entry(entry: ActivityEntry, ts: str) -> str:
        """Format tool_result as compact meta-only line for consolidation.

        Output: ``[HH:MM] TRES tool_name → ok (12件, 3.2KB)``
        Avoids injecting raw result content (which can be huge).
        """
        tool = entry.tool or "unknown"
        meta = entry.meta or {}
        status = meta.get("result_status", "ok")
        result_bytes = meta.get("result_bytes", 0)
        result_count = meta.get("result_count")

        if result_bytes >= 1024:
            size_str = f"{result_bytes / 1024:.1f}KB"
        else:
            size_str = f"{result_bytes}B"

        parts = []
        if result_count is not None:
            parts.append(f"{result_count}件")
        parts.append(size_str)

        detail = f" ({', '.join(parts)})" if parts else ""

        if status == "fail":
            err_hint = (entry.content or "")[:60]
            return f"[{ts}] TRES {tool} → fail: {err_hint}"
        return f"[{ts}] TRES {tool} → ok{detail}"

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

            # DM / inter-Anima message grouping (includes legacy aliases).
            # Human-chat message_received (from_type="human") is excluded.
            _is_dm_event = entry_type in ("dm_sent", "dm_received", "message_sent")
            if entry_type == "message_received":
                _is_dm_event = entry.meta.get("from_type") == "anima"
            if _is_dm_event:
                peer = entry.to_person if entry_type in ("dm_sent", "message_sent") else entry.from_person
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
            for e in group.entries:
                direction = "MSG<" if e.type in ("dm_received", "message_received") else "MSG>"
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

    # ── Trigger-based grouping (for timeline UI) ──────────────────

    _TRIGGER_TYPES = frozenset({
        "heartbeat_start", "message_received", "cron_executed",
        "task_created", "task_updated",
        "inbox_processing_start", "task_exec_start",
    })

    _CLOSE_MAP: dict[str, frozenset[str]] = {
        "heartbeat": frozenset({"heartbeat_end"}),
        "chat": frozenset({"response_sent"}),
        "dm": frozenset({"response_sent", "message_sent", "dm_sent"}),
        "inbox": frozenset({"inbox_processing_end"}),
        # task_exec: no explicit close — stays open until the next trigger
        # so that the follow-up message_sent (completion report) is absorbed.
    }

    @staticmethod
    def group_by_trigger(
        entries: list[ActivityEntry],
    ) -> list[dict[str, Any]]:
        """Group entries by trigger events for timeline display.

        Trigger events (heartbeat_start, message_received, cron_executed,
        task_created, inbox_processing_start, task_exec_start) open a new
        group.  Subsequent events are absorbed into the open group until a
        closing event or the next trigger *for the same Anima*.

        Each Anima's open group is tracked independently so that
        cross-Anima interleaving does not break grouping.

        tool_use / tool_result pairs are merged into a single entry
        with ``tool_result`` field attached.

        This is a static method — no ``ActivityLogger`` instance is needed.
        """
        _AL = ActivityLogger
        paired = _AL._pair_tool_results(entries)
        groups: list[dict[str, Any]] = []
        current_by_anima: dict[str, dict[str, Any]] = {}

        for entry in paired:
            etype = entry.type
            anima = entry._anima_name
            evt_dict = _AL._entry_to_event_dict(entry)
            is_trigger = etype in _AL._TRIGGER_TYPES
            cur = current_by_anima.get(anima)

            if is_trigger:
                if cur is not None:
                    _AL._finalize_group(cur)
                    groups.append(cur)
                current_by_anima[anima] = _AL._open_group(entry, evt_dict)
                continue

            if etype == "heartbeat_end":
                if cur and cur["type"] == "heartbeat":
                    cur["events"].append(evt_dict)
                    cur["end_ts"] = entry.ts
                    cur["is_open"] = False
                    if entry.summary:
                        cur["summary"] = entry.summary
                    _AL._finalize_group(cur)
                    groups.append(cur)
                    del current_by_anima[anima]
                    continue
                # HB was interrupted (e.g. by a user chat). Retroactively
                # append to the most recent finalized heartbeat group for
                # this anima so the end event is not orphaned.
                retrogrp = _AL._find_recent_group(
                    groups, anima, "heartbeat",
                )
                if retrogrp is not None:
                    retrogrp["events"].append(evt_dict)
                    retrogrp["end_ts"] = entry.ts
                    retrogrp["is_open"] = False
                    if entry.summary:
                        retrogrp["summary"] = entry.summary
                    retrogrp["event_count"] = len(retrogrp["events"])
                    continue
                # No matching heartbeat at all — absorb into current if any
                if cur is not None:
                    cur["events"].append(evt_dict)
                    cur["end_ts"] = entry.ts
                    continue

            if cur is not None:
                close_types = _AL._CLOSE_MAP.get(cur["type"], frozenset())
                cur["events"].append(evt_dict)
                cur["end_ts"] = entry.ts
                if etype in close_types:
                    cur["is_open"] = False
                    _AL._finalize_group(cur)
                    groups.append(cur)
                    del current_by_anima[anima]
                continue

            # Last resort: try to append to the most recent finalized
            # group for this anima (catches post-close follow-up events
            # like message_sent after task_exec_end).
            retrogrp = _AL._find_recent_group(groups, anima)
            if retrogrp is not None:
                retrogrp["events"].append(evt_dict)
                retrogrp["end_ts"] = entry.ts
                retrogrp["event_count"] = len(retrogrp["events"])
                continue

            groups.append(_AL._make_single_group(entry, evt_dict))

        for cur in current_by_anima.values():
            _AL._finalize_group(cur)
            groups.append(cur)

        return groups

    @staticmethod
    def _pair_tool_results(
        entries: list[ActivityEntry],
    ) -> list[ActivityEntry]:
        """Attach tool_result data to tool_use entries, remove paired results.

        Sets ``_tool_result_data`` on tool_use entries and returns a
        filtered list excluding consumed tool_result entries.
        """
        result_by_id: dict[str, ActivityEntry] = {}
        for e in entries:
            if e.type == "tool_result":
                tid = e.meta.get("tool_use_id", "")
                if tid:
                    result_by_id[tid] = e

        paired_ids: set[int] = set()
        for e in entries:
            if e.type == "tool_use":
                tid = e.meta.get("tool_use_id", "")
                result_entry = result_by_id.get(tid) if tid else None
                if not result_entry:
                    result_entry = ActivityLogger._find_tool_result_fallback(
                        entries, e,
                    )
                if result_entry:
                    e._tool_result_data = {
                        "content": result_entry.content or result_entry.summary,
                        "is_error": result_entry.meta.get("is_error", False),
                    }
                    paired_ids.add(id(result_entry))

        return [e for e in entries if id(e) not in paired_ids]

    @staticmethod
    def _entry_to_event_dict(entry: ActivityEntry) -> dict[str, Any]:
        """Convert an entry to API dict, attaching tool_result if paired."""
        d = entry.to_api_dict()
        if entry.type == "tool_use" and entry._tool_result_data is not None:
            d["tool_result"] = entry._tool_result_data
        return d

    @staticmethod
    def _open_group(entry: ActivityEntry, evt_dict: dict[str, Any]) -> dict[str, Any]:
        """Create a new group dict from a trigger entry."""
        etype = entry.type
        anima = entry._anima_name

        if etype == "heartbeat_start":
            gtype = "heartbeat"
        elif etype == "message_received":
            from_type = entry.meta.get("from_type", "human")
            gtype = "dm" if from_type == "anima" else "chat"
        elif etype == "cron_executed":
            gtype = "cron"
        elif etype in ("task_created", "task_updated"):
            gtype = "task"
        elif etype == "inbox_processing_start":
            gtype = "inbox"
        elif etype == "task_exec_start":
            gtype = "task_exec"
        else:
            gtype = "single"

        summary = entry.summary or entry.meta.get("task_name", "")
        if gtype == "dm":
            summary = entry.from_person or entry.to_person or summary

        return {
            "id": f"grp-{anima}:{entry.ts}:{gtype}",
            "type": gtype,
            "anima": anima,
            "start_ts": entry.ts,
            "end_ts": entry.ts,
            "summary": summary,
            "event_count": 1,
            "is_open": True,
            "events": [evt_dict],
        }

    @staticmethod
    def _make_single_group(entry: ActivityEntry, evt_dict: dict[str, Any]) -> dict[str, Any]:
        anima = entry._anima_name
        return {
            "id": f"grp-{anima}:{entry.ts}:single",
            "type": "single",
            "anima": anima,
            "start_ts": entry.ts,
            "end_ts": entry.ts,
            "summary": entry.summary,
            "event_count": 1,
            "is_open": False,
            "events": [evt_dict],
        }

    @staticmethod
    def _find_recent_group(
        groups: list[dict[str, Any]],
        anima: str,
        gtype: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the most recently finalized group for *anima*.

        Searches *groups* in reverse.  If *gtype* is given, only groups of
        that type are considered.
        """
        for grp in reversed(groups):
            if grp["anima"] != anima:
                continue
            if gtype is not None and grp["type"] != gtype:
                continue
            return grp
        return None

    @staticmethod
    def _finalize_group(group: dict[str, Any]) -> None:
        """Update event_count before returning the group."""
        group["event_count"] = len(group["events"])

    # ── Conversation view ───────────────────────────────────────

    # Event types relevant to conversation timeline display.
    _CONVERSATION_TYPES = {
        "message_received", "response_sent",
        "tool_use", "tool_result",
        "heartbeat_start", "heartbeat_end",
        "cron_executed",
        "error",
    }

    def get_conversation_view(
        self,
        *,
        before: str | None = None,
        limit: int = 50,
        session_gap_minutes: int = 10,
    ) -> dict[str, Any]:
        """Build a conversation view from activity log entries.

        Reads activity log events, pairs tool_use/tool_result, groups into
        sessions (separated by gaps >= *session_gap_minutes*), and returns a
        structure ready for UI rendering.

        Args:
            before: If given, only include entries with ``ts < before``
                (cursor-based pagination, ISO 8601 timestamp).
            limit: Maximum number of *messages* to return (tool_calls are
                nested within assistant messages and do not count).
            session_gap_minutes: Minimum gap in minutes to start a new session.

        Returns:
            ``{"sessions": [...], "has_more": bool, "next_before": str | None}``
        """
        entries = self._load_conversation_entries(before=before, limit=limit)
        messages = self._entries_to_messages(entries)

        # Apply limit (keep oldest N for chronological order display)
        has_more = len(messages) > limit
        if has_more:
            messages = messages[-limit:]

        # Determine next_before cursor
        next_before: str | None = None
        if has_more and messages:
            next_before = messages[0]["ts"]

        # Group into sessions
        sessions = self._group_into_sessions(messages, session_gap_minutes)

        return {
            "sessions": sessions,
            "has_more": has_more,
            "next_before": next_before,
        }

    def _load_conversation_entries(
        self,
        *,
        before: str | None = None,
        limit: int = 50,
    ) -> list[ActivityEntry]:
        """Load conversation-relevant entries, scanning backwards.

        Returns entries in chronological order.  Scans enough days to
        collect at least ``limit * 3`` raw entries (to account for
        tool_use/tool_result pairs being folded into messages).
        """
        target_raw = limit * 3 + 50  # overshoot to ensure enough messages
        entries: list[ActivityEntry] = []
        today = now_jst().date()
        max_scan_days = 365  # safety cap

        for day_offset in range(max_scan_days):
            target = today - timedelta(days=day_offset)
            path = self._log_dir / f"{target.isoformat()}.jsonl"
            if not path.exists():
                # Keep scanning further back; gaps in dates are normal
                if day_offset > 30 and not entries:
                    break  # give up after 30 empty days at start
                continue

            day_entries: list[ActivityEntry] = []
            try:
                for line_num, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1,
                ):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if raw.get("type") not in self._CONVERSATION_TYPES:
                        continue
                    ts = raw.get("ts", "")
                    if before and ts >= before:
                        continue
                    # Map JSONL keys
                    if "from" in raw:
                        raw["from_person"] = raw.pop("from")
                    if "to" in raw:
                        raw["to_person"] = raw.pop("to")
                    entry = ActivityEntry(**{
                        k: v for k, v in raw.items()
                        if k in ActivityEntry.__dataclass_fields__
                    })
                    entry._line_number = line_num
                    day_entries.append(entry)
            except Exception:
                logger.exception("Failed to read activity log %s", path)

            entries = day_entries + entries  # prepend (older first)
            if len(entries) >= target_raw:
                break

        return entries

    def _entries_to_messages(
        self,
        entries: list[ActivityEntry],
    ) -> list[dict[str, Any]]:
        """Convert raw activity entries to conversation messages.

        Pairs ``tool_use``/``tool_result`` by ``meta.tool_use_id`` and
        nests them into the preceding ``response_sent`` message.
        """
        # Build tool_result lookup by tool_use_id
        tool_results: dict[str, ActivityEntry] = {}
        for e in entries:
            if e.type == "tool_result":
                tid = e.meta.get("tool_use_id", "")
                if tid:
                    tool_results[tid] = e

        messages: list[dict[str, Any]] = []
        pending_tool_calls: list[dict[str, Any]] = []

        for e in entries:
            if e.type == "message_received":
                # Flush pending tools to last assistant message
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "human",
                    "content": e.content,
                    "from_person": e.from_person,
                    "tool_calls": [],
                })

            elif e.type == "response_sent":
                # Create assistant message first, then flush pending tools into it
                msg = {
                    "ts": e.ts,
                    "role": "assistant",
                    "content": e.content,
                    "from_person": "",
                    "tool_calls": [],
                }
                if e.meta.get("thinking_text"):
                    msg["thinking_text"] = e.meta["thinking_text"]
                messages.append(msg)
                # Attach any pending tool calls to this assistant message
                if pending_tool_calls:
                    msg["tool_calls"].extend(pending_tool_calls)
                    pending_tool_calls.clear()

            elif e.type == "tool_use":
                tid = e.meta.get("tool_use_id", "")
                result_entry = tool_results.get(tid) if tid else None
                # Fallback: try matching by timestamp proximity + tool name
                if not result_entry:
                    result_entry = self._find_tool_result_fallback(
                        entries, e,
                    )
                tc: dict[str, Any] = {
                    "tool_use_id": tid,
                    "tool_name": e.tool,
                    "input": e.meta.get("args", e.content),
                    "result": result_entry.content if result_entry else "",
                    "is_error": (
                        result_entry.meta.get("is_error", False)
                        if result_entry else False
                    ),
                }
                if e.meta.get("blocked"):
                    tc["is_error"] = True
                    tc["result"] = f"ブロック: {e.meta.get('reason', '')}"
                pending_tool_calls.append(tc)

            elif e.type == "tool_result":
                # Already handled via lookup; skip standalone
                pass

            elif e.type == "heartbeat_start":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": e.summary or "定期巡回開始",
                    "from_person": "",
                    "tool_calls": [],
                    "_trigger": "heartbeat",
                })

            elif e.type == "heartbeat_end":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": e.summary or e.content or "定期巡回完了",
                    "from_person": "",
                    "tool_calls": [],
                    "_trigger": "heartbeat",
                })

            elif e.type == "cron_executed":
                self._flush_tool_calls(messages, pending_tool_calls)
                task_name = e.meta.get("task_name", "")
                content = e.summary or e.content or task_name or "cronタスク実行"
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": content,
                    "from_person": "",
                    "tool_calls": [],
                    "_trigger": "cron",
                })

            elif e.type == "error":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": f"[エラー] {e.summary or e.content}",
                    "from_person": "",
                    "tool_calls": [],
                })

        # Flush any remaining tool calls
        self._flush_tool_calls(messages, pending_tool_calls)

        return messages

    @staticmethod
    def _flush_tool_calls(
        messages: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> None:
        """Attach pending tool calls to the last assistant message."""
        if not pending:
            return
        # Find last assistant message to attach to
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                msg["tool_calls"].extend(pending)
                pending.clear()
                return
        # No assistant message found; discard (shouldn't normally happen)
        pending.clear()

    @staticmethod
    def _find_tool_result_fallback(
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
                # Within 5 minutes
                if _time_diff(tool_use_entry.ts, e.ts) < 300:
                    return e
            # Stop searching after too many entries
            if e.type in ("message_received", "response_sent"):
                break
        return None

    @staticmethod
    def _group_into_sessions(
        messages: list[dict[str, Any]],
        gap_minutes: int,
    ) -> list[dict[str, Any]]:
        """Group messages into sessions based on time gaps."""
        if not messages:
            return []

        gap_seconds = gap_minutes * 60
        sessions: list[dict[str, Any]] = []
        current_msgs: list[dict[str, Any]] = []
        current_trigger = "chat"

        for msg in messages:
            # Detect trigger from system messages
            msg_trigger = msg.pop("_trigger", None)
            if msg_trigger:
                current_trigger = msg_trigger

            if current_msgs:
                prev_ts = current_msgs[-1]["ts"]
                if _time_diff(prev_ts, msg["ts"]) >= gap_seconds:
                    # Close current session
                    sessions.append({
                        "session_start": current_msgs[0]["ts"],
                        "session_end": current_msgs[-1]["ts"],
                        "trigger": current_trigger,
                        "messages": current_msgs,
                    })
                    current_msgs = []
                    current_trigger = msg_trigger or "chat"

            current_msgs.append(msg)

        # Close final session
        if current_msgs:
            sessions.append({
                "session_start": current_msgs[0]["ts"],
                "session_end": current_msgs[-1]["ts"],
                "trigger": current_trigger,
                "messages": current_msgs,
            })

        return sessions

    # ── Rotation ──────────────────────────────────────────────

    def rotate(
        self,
        *,
        mode: str = "size",
        max_size_mb: int = 1024,
        max_age_days: int = 7,
    ) -> dict[str, Any]:
        """Rotate activity log files by deleting old entries.

        Args:
            mode: ``"size"`` (total size cap), ``"time"`` (age cap),
                or ``"both"`` (time then size).
            max_size_mb: Maximum total size in MB (per-anima).
            max_age_days: Maximum age in days for ``"time"``/``"both"``.

        Returns:
            Dict with ``deleted_files`` count and ``freed_bytes``.
        """
        if not self._log_dir.exists():
            return {"deleted_files": 0, "freed_bytes": 0}

        today_str = date.today().isoformat()
        files = sorted(self._log_dir.glob("*.jsonl"))

        deleted_count = 0
        freed_bytes = 0

        # Phase 1: time-based deletion
        if mode in ("time", "both"):
            cutoff = date.today() - timedelta(days=max_age_days)
            remaining: list[Path] = []
            for f in files:
                file_date_str = f.stem
                if file_date_str == today_str:
                    remaining.append(f)
                    continue
                try:
                    file_date = date.fromisoformat(file_date_str)
                except ValueError:
                    remaining.append(f)
                    continue
                if file_date < cutoff:
                    size = f.stat().st_size
                    f.unlink()
                    deleted_count += 1
                    freed_bytes += size
                    logger.debug("Rotation (time): deleted %s (%d bytes)", f.name, size)
                else:
                    remaining.append(f)
            files = remaining

        # Phase 2: size-based deletion
        if mode in ("size", "both"):
            max_bytes = max_size_mb * 1024 * 1024
            # Cache file sizes to avoid double stat() calls
            file_sizes = {f: f.stat().st_size for f in files}
            total_size = sum(file_sizes.values())
            # Delete oldest first (files already sorted by name = date)
            for f in files:
                if total_size <= max_bytes:
                    break
                if f.stem == today_str:
                    continue
                size = file_sizes[f]
                f.unlink()
                total_size -= size
                deleted_count += 1
                freed_bytes += size
                logger.debug("Rotation (size): deleted %s (%d bytes)", f.name, size)

        if deleted_count:
            logger.info(
                "Rotation completed for %s: deleted=%d freed=%d bytes",
                self.anima_dir.name, deleted_count, freed_bytes,
            )

        return {"deleted_files": deleted_count, "freed_bytes": freed_bytes}

    @staticmethod
    def rotate_all(
        animas_dir: Path,
        *,
        mode: str = "size",
        max_size_mb: int = 1024,
        max_age_days: int = 7,
    ) -> dict[str, dict[str, Any]]:
        """Run rotation for all Animas under *animas_dir*.

        Returns:
            Dict mapping anima name to rotation result.
        """
        results: dict[str, dict[str, Any]] = {}
        if not animas_dir.exists():
            return results
        for anima_dir in sorted(animas_dir.iterdir()):
            if not anima_dir.is_dir():
                continue
            log_dir = anima_dir / "activity_log"
            if not log_dir.exists():
                continue
            try:
                al = ActivityLogger(anima_dir)
                result = al.rotate(
                    mode=mode,
                    max_size_mb=max_size_mb,
                    max_age_days=max_age_days,
                )
                if result["deleted_files"] > 0:
                    results[anima_dir.name] = result
            except Exception:
                logger.exception("Rotation failed for %s", anima_dir.name)
        return results
