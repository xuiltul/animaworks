from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Recent outbound and pending human notifications collection."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from core.i18n import t
from core.memory.priming.items import ItemizedMemory, MemoryItem, render_items, select_within_budget
from core.prompt.tokens import estimate_tokens
from core.time_utils import ensure_aware, now_local

logger = logging.getLogger("animaworks.priming")

_HUMAN_NOTIFY_BUDGET_TOKENS = 500


def _timestamp_rank(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


async def collect_recent_outbound(anima_dir: Path, max_entries: int = 3) -> str:
    """Collect recent outbound actions (channel_post, message_sent).

    Reads activity_log for the last 2 hours and formats a short summary.
    This replaces the former ``_build_recent_outbound_section`` in builder.py,
    ensuring builder.py never reads ActivityLogger directly (hippocampus model).
    """
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        entries = activity.recent(
            days=1,
            limit=20,
            types=["channel_post", "message_sent"],
        )
    except Exception:
        return ""

    if not entries:
        return ""

    cutoff = now_local() - timedelta(hours=2)

    recent: list = []
    for e in reversed(entries):
        try:
            ts = ensure_aware(datetime.fromisoformat(e.ts))
            if ts >= cutoff:
                recent.append(e)
        except (ValueError, TypeError):
            continue
        if len(recent) >= max_entries:
            break

    if not recent:
        return ""

    items: list[MemoryItem] = []
    for e in reversed(recent):
        time_str = e.ts[11:16] if len(e.ts) >= 16 else e.ts
        text_preview = (e.summary or e.content or "")[:200]
        if e.type == "channel_post":
            ch = e.channel or "?"
            text = t("priming.outbound_posted", time_str=time_str, ch=ch, text_preview=text_preview)
        elif e.type in ("dm_sent", "message_sent"):
            to = e.to_person or "?"
            text = t("priming.outbound_sent", time_str=time_str, to=to, text_preview=text_preview)
        else:
            continue
        items.append(
            MemoryItem(
                source="recent_outbound",
                key=f"{e.ts}|{e.channel}|{e.from_person}",
                text=text,
                updated=e.ts,
                rank=_timestamp_rank(e.ts),
            )
        )
    return ItemizedMemory(render_items(items, t("priming.outbound_header")), items) if items else ""


async def collect_pending_human_notifications(anima_dir: Path, *, channel: str = "") -> str:
    """Collect recent call_human notifications for context injection.

    Returns formatted string of human_notify entries from last 24 hours.
    Only active for chat, heartbeat, and message: sessions.
    """
    if channel not in ("chat", "heartbeat") and not channel.startswith("message:"):
        return ""

    from core.memory.activity import ActivityLogger
    from core.taskboard.attention_resolver import notification_key_for, resolver_for_anima_dir

    activity = ActivityLogger(anima_dir)
    entries = activity.recent(days=1, limit=10, types=["human_notify"])
    if not entries:
        return ""

    items: list[MemoryItem] = []
    header = "## Pending Human Notifications (last 24h)"
    try:
        resolver = resolver_for_anima_dir(anima_dir)
    except Exception:
        logger.debug("TaskBoard human_notify gate unavailable; using activity entries as-is", exc_info=True)
        resolver = None

    for entry in reversed(entries):
        ts = entry.ts[:16]
        body = entry.content or entry.summary or ""
        via = entry.via or ""
        subject = str(entry.meta.get("subject") or "")
        notification_key = str(entry.meta.get("notification_key") or notification_key_for(subject, body))
        if resolver is not None and not resolver.should_show_human_notify(anima_dir.name, notification_key, entry.ts):
            continue
        line = f"[{ts}] call_human (via {via}):\n{body}"
        items.append(
            MemoryItem(
                source="pending_human_notifications",
                key=notification_key,
                text=line,
                updated=entry.ts,
                rank=_timestamp_rank(entry.ts),
            )
        )

    available = _HUMAN_NOTIFY_BUDGET_TOKENS - estimate_tokens(header)
    selected = select_within_budget(items, available)
    while selected and estimate_tokens(render_items(selected, header)) > _HUMAN_NOTIFY_BUDGET_TOKENS:
        selected.pop()
    if not selected:
        return ""
    selected.sort(key=lambda item: item.updated)
    return ItemizedMemory(render_items(selected, header), selected)
