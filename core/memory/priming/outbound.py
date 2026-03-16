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
from core.memory.priming.constants import _CHARS_PER_TOKEN
from core.time_utils import ensure_aware, now_local

logger = logging.getLogger("animaworks.priming")

_HUMAN_NOTIFY_BUDGET_TOKENS = 500


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

    lines = [t("priming.outbound_header"), ""]
    for e in reversed(recent):
        time_str = e.ts[11:16] if len(e.ts) >= 16 else e.ts
        text_preview = (e.summary or e.content or "")[:200]
        if e.type == "channel_post":
            ch = e.channel or "?"
            lines.append(t("priming.outbound_posted", time_str=time_str, ch=ch, text_preview=text_preview))
        elif e.type in ("dm_sent", "message_sent"):
            to = e.to_person or "?"
            lines.append(t("priming.outbound_sent", time_str=time_str, to=to, text_preview=text_preview))
    lines.append("")
    return "\n".join(lines)


async def collect_pending_human_notifications(anima_dir: Path, *, channel: str = "") -> str:
    """Collect recent call_human notifications for context injection.

    Returns formatted string of human_notify entries from last 24 hours.
    Only active for chat, heartbeat, and message: sessions.
    """
    if channel not in ("chat", "heartbeat") and not channel.startswith("message:"):
        return ""

    from core.memory.activity import ActivityLogger

    activity = ActivityLogger(anima_dir)
    entries = activity.recent(days=1, limit=10, types=["human_notify"])
    if not entries:
        return ""

    lines: list[str] = []
    budget_chars = _HUMAN_NOTIFY_BUDGET_TOKENS * _CHARS_PER_TOKEN
    total = 0
    for entry in reversed(entries):
        ts = entry.ts[:16]
        body = entry.content or entry.summary or ""
        via = entry.via or ""
        line = f"[{ts}] call_human (via {via}):\n{body}"
        if total + len(line) > budget_chars:
            break
        lines.append(line)
        total += len(line)

    if not lines:
        return ""

    lines.reverse()
    header = "## Pending Human Notifications (last 24h)"
    return header + "\n\n" + "\n\n".join(lines)
