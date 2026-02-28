from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Priming formatter mixin for ActivityLogger.

Internal module — import from :mod:`core.memory.activity` instead.
"""

from typing import Any

from core.i18n import t
from core.memory._activity_models import (
    CHARS_PER_TOKEN,
    ActivityEntry,
    EntryGroup,
    dm_label,
    get_peer,
    get_task_name,
    set_source_lines,
    time_diff,
)


class PrimingMixin:
    """Mixin providing priming format methods for ActivityLogger."""

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
        max_chars = budget_tokens * CHARS_PER_TOKEN
        lines: list[str] = []
        total_chars = 0

        for group in reversed(groups):
            formatted = self._format_group(group, content_trim=content_trim)
            if total_chars + len(formatted) + 1 > max_chars:
                break
            lines.append(formatted)
            total_chars += len(formatted) + 1

        if not lines:
            return ""

        lines.reverse()
        return "\n".join(lines)

    @staticmethod
    def _format_entry(entry: ActivityEntry, content_trim: int = 200) -> str:
        """Format a single entry as a concise timeline line."""
        ts = entry.ts[11:16] if len(entry.ts) >= 16 else entry.ts

        if entry.type == "tool_result":
            return PrimingMixin._format_tool_result_entry(entry, ts)

        text = entry.summary or entry.content

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
            parts.append(t("activity.items_count", count=result_count))
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

            _is_dm_event = entry_type in ("dm_sent", "dm_received", "message_sent")
            if entry_type == "message_received":
                _is_dm_event = entry.meta.get("from_type") == "anima"
            if _is_dm_event:
                peer = entry.to_person if entry_type in ("dm_sent", "message_sent") else entry.from_person
                if (
                    current_group
                    and current_group.type == "dm"
                    and get_peer(current_group) == peer
                    and time_diff(current_group.end_ts, entry.ts) <= gap_seconds
                ):
                    current_group.entries.append(entry)
                    current_group.end_ts = entry.ts
                    continue
                if current_group:
                    groups.append(current_group)
                current_group = EntryGroup(
                    type="dm",
                    start_ts=entry.ts,
                    end_ts=entry.ts,
                    entries=[entry],
                    label=dm_label(peer, entry),
                    source_lines="",
                )
                continue

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

            if entry_type == "cron_executed":
                task_name = entry.meta.get("task_name", "")
                if (
                    current_group
                    and current_group.type == "cron"
                    and get_task_name(current_group) == task_name
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

        for group in groups:
            set_source_lines(group)

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
            peer = get_peer(group)
            lines = [f"{time_range} DM {peer}:"]
            for e in group.entries:
                direction = "MSG<" if e.type in ("dm_received", "message_received") else "MSG>"
                text = e.summary or e.content[:100]
                if len(text) > 100:
                    text = text[:100]
                lines.append(f"  {direction} {text}")
            if group.source_lines:
                lines.append(f"  -> {group.source_lines}")
            return "\n".join(lines)

        if group.type == "hb":
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
            exit_code = group.entries[0].meta.get("exit_code", "")
            exit_info = f": exit={exit_code}" if exit_code != "" else ""
            header = f"{time_range} CRON {group.label}{exit_info}"
            lines = [header]
            if group.source_lines:
                lines.append(f"  -> {group.source_lines}")
            return "\n".join(lines)

        return PrimingMixin._format_entry(group.entries[0], content_trim=content_trim)
