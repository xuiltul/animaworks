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

Implementation is split across submodules; this file is the public
façade that re-exports every symbol and defines the core
``ActivityLogger`` class.
"""

import json  # noqa: F401  — kept at module level for mock.patch compat
import logging
import math
import os  # noqa: F401  — kept at module level for mock.patch compat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.time_utils import ensure_aware, now_iso, now_jst  # noqa: F401

# ── Re-export data models & helpers (public API) ─────────────
from core.memory._activity_models import (  # noqa: F401
    ActivityEntry,
    ActivityPage,
    EntryGroup,
)

# ── Re-export with legacy private names for test compat ──────
from core.memory._activity_models import (
    CHARS_PER_TOKEN as _CHARS_PER_TOKEN,  # noqa: F401
    EVENT_TYPE_ALIASES as _EVENT_TYPE_ALIASES,  # noqa: F401
    dm_label as _dm_label,  # noqa: F401
    find_tool_result_fallback as _find_tool_result_fallback,  # noqa: F401
    get_peer as _get_peer,  # noqa: F401
    get_task_name as _get_task_name,  # noqa: F401
    resolve_type_filter as _resolve_type_filter,  # noqa: F401
    set_source_lines as _set_source_lines,  # noqa: F401
    time_diff as _time_diff,  # noqa: F401
)

# ── Mixin imports ────────────────────────────────────────────
from core.memory._activity_conversation import ConversationMixin
from core.memory._activity_priming import PrimingMixin
from core.memory._activity_rotation import RotationMixin
from core.memory._activity_timeline import TimelineMixin

logger = logging.getLogger("animaworks.activity")


# ── ActivityLogger ────────────────────────────────────────────


class ActivityLogger(
    PrimingMixin, TimelineMixin, ConversationMixin, RotationMixin,
):
    """Unified activity recorder for a single Anima.

    All interactions are written as append-only JSONL to
    ``{anima_dir}/activity_log/{date}.jsonl``.

    Core recording / retrieval lives here; formatting, timeline
    grouping, conversation view, and rotation are provided by mixins.
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

        scan_days = days
        cutoff: datetime | None = None
        if hours is not None:
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
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=cutoff.tzinfo)
                            if ts < cutoff:
                                continue
                        except (ValueError, TypeError):
                            logger.debug("Failed to parse timestamp for cutoff filtering", exc_info=True)
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

        all_entries.reverse()

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
