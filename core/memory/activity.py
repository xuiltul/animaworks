from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.

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
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict, omitting empty fields."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v}


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
            ts=datetime.now().isoformat(),
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
        """Append *entry* to today's JSONL file."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            date_str = entry.ts[:10]
            path = self._log_dir / f"{date_str}.jsonl"
            line = json.dumps(entry.to_dict(), ensure_ascii=False)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            logger.exception("Failed to append activity log")

    # ── Retrieval ─────────────────────────────────────────────

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
        entries: list[ActivityEntry] = []
        today = date.today()
        type_set = set(types) if types else None

        for offset in range(days):
            target = today - timedelta(days=offset)
            path = self._log_dir / f"{target.isoformat()}.jsonl"
            if not path.exists():
                continue
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
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
                    entries.append(ActivityEntry(**{
                        k: v for k, v in raw.items()
                        if k in ActivityEntry.__dataclass_fields__
                    }))
            except Exception:
                logger.exception("Failed to read activity log %s", path)

        # Chronological sort, newest last
        entries.sort(key=lambda e: e.ts)

        # Return the most recent *limit* entries
        if len(entries) > limit:
            entries = entries[-limit:]

        return entries

    @staticmethod
    def _involves(raw: dict[str, Any], name: str) -> bool:
        """Return True if *name* appears in from/to/channel fields."""
        return (
            raw.get("from_person") == name
            or raw.get("to_person") == name
            or raw.get("channel") == name
        )

    # ── Priming formatter ─────────────────────────────────────

    def format_for_priming(
        self,
        entries: list[ActivityEntry],
        budget_tokens: int = 1300,
    ) -> str:
        """Format activity entries for system prompt injection.

        Renders a human-readable timeline, truncated to *budget_tokens*.

        Args:
            entries: Entries to format (should be chronological).
            budget_tokens: Target token budget.

        Returns:
            Formatted Markdown string.
        """
        if not entries:
            return ""

        max_chars = budget_tokens * _CHARS_PER_TOKEN
        lines: list[str] = []
        total_chars = 0

        # Build lines in reverse-chronological order (newest first)
        # so that if we truncate, we keep the most recent events.
        for entry in reversed(entries):
            line = self._format_entry(entry)
            line_len = len(line) + 1  # +1 for newline
            if total_chars + line_len > max_chars:
                break
            lines.append(line)
            total_chars += line_len

        if not lines:
            return ""

        # Reverse back to chronological order
        lines.reverse()
        return "\n".join(lines)

    @staticmethod
    def _format_entry(entry: ActivityEntry) -> str:
        """Format a single entry as a concise timeline line."""
        ts = entry.ts[11:16] if len(entry.ts) >= 16 else entry.ts
        text = entry.summary or entry.content

        # Truncate long content
        if len(text) > 200:
            text = text[:200] + "..."

        type_map: dict[str, str] = {
            "message_received": "📩",
            "response_sent": "💬",
            "channel_read": "👁",
            "channel_post": "📢",
            "dm_received": "📨",
            "dm_sent": "✉️",
            "human_notify": "🔔",
            "tool_use": "🔧",
            "heartbeat_start": "💓",
            "heartbeat_end": "💓",
            "cron_executed": "⏰",
            "memory_write": "📝",
            "error": "❌",
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

        ctx = f" ({', '.join(context_parts)})" if context_parts else ""
        return f"[{ts}] {icon} {entry.type}{ctx}: {text}"
