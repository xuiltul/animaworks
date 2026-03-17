from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel B: Recent activity from unified activity log."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from core.time_utils import ensure_aware, now_local, today_local
from core.tools._async_compat import run_sync

logger = logging.getLogger("animaworks.priming")

# Event types that are noise for heartbeat/cron priming — tool invocations
# and heartbeat lifecycle events crowd out actionable messages.
_HEARTBEAT_NOISE_TYPES = frozenset(
    {
        "tool_use",
        "tool_result",
        "heartbeat_start",
        "heartbeat_reflection",
        "inbox_processing_start",
        "inbox_processing_end",
    }
)

# Event types to exclude from chat priming — tool invocations, memory writes,
# and background lifecycle events crowd out meaningful communication events.
# Only message exchanges, errors, and task lifecycle events survive.
_CHAT_NOISE_TYPES = frozenset(
    {
        "tool_use",
        "tool_result",
        "memory_write",
        "cron_executed",
        "heartbeat_start",
        "heartbeat_end",
        "heartbeat_reflection",
        "inbox_processing_start",
        "inbox_processing_end",
    }
)

_OWN_ACTION_TYPES = frozenset(
    {
        "message_sent",
        "response_sent",
        "message_received",
    }
)


async def channel_b_recent_activity(
    anima_dir: Path,
    shared_dir: Path | None,
    sender_name: str,
    keywords: list[str],
    *,
    channel: str = "",
) -> str:
    """Channel B: Recent activity from unified activity log.

    Replaces old Channel B (episodes) and Channel E (shared channels).
    Reads from activity_log/{date}.jsonl for a unified timeline,
    plus shared/channels/*.jsonl for cross-Anima visibility.
    Falls back to episodes/ if activity_log is empty (migration period).

    When *channel* is ``"heartbeat"`` or starts with ``"cron:"``,
    tool_use / tool_result / heartbeat lifecycle events are filtered
    out so that the limited priming budget contains only actionable
    communication events (messages, channel posts, errors, etc.).
    """
    from core.memory.activity import ActivityLogger

    activity = ActivityLogger(anima_dir)
    entries = activity.recent(days=2, limit=100)

    is_background = channel in ("heartbeat",) or channel.startswith("cron:")

    if is_background and entries:
        entries = [e for e in entries if e.type not in _HEARTBEAT_NOISE_TYPES]
    elif entries:
        entries = [e for e in entries if e.type not in _CHAT_NOISE_TYPES]

    # Always read shared channels for cross-Anima visibility
    channel_entries = read_shared_channels(anima_dir, shared_dir, limit_per_channel=5)
    entries.extend(channel_entries)

    if entries:
        prioritized = prioritize_entries(entries, sender_name, keywords)
        prioritized = prioritized[:50]
        return activity.format_for_priming(prioritized, budget_tokens=1300)

    # Fallback: read old episodes if no activity log exists yet
    return await fallback_episodes_and_channels(anima_dir, shared_dir)


def read_shared_channels(
    anima_dir: Path,
    shared_dir: Path | None,
    limit_per_channel: int = 5,
) -> list:
    """Read recent entries from shared channels for cross-Anima visibility.

    Reads shared/channels/*.jsonl and converts entries to ActivityEntry
    format for unified priming display.  Prioritises 24h human posts
    and @mentions.

    Args:
        limit_per_channel: Max entries per channel (latest N).

    Returns:
        List of ActivityEntry from shared channels.
    """
    from core.memory.activity import ActivityEntry

    if not shared_dir:
        return []

    channels_dir = shared_dir / "channels"
    if not channels_dir.is_dir():
        return []

    anima_name = anima_dir.name
    mention_tag = f"@{anima_name}"
    now = now_local()
    cutoff_24h = now - timedelta(hours=24)

    result: list[ActivityEntry] = []

    try:
        from core.messenger import is_channel_member

        for channel_file in sorted(channels_dir.glob("*.jsonl")):
            channel_name = channel_file.stem

            if not is_channel_member(shared_dir, channel_name, anima_name):
                continue

            try:
                content = channel_file.read_text(encoding="utf-8")
            except OSError:
                continue

            lines = content.strip().splitlines()
            if not lines:
                continue

            all_entries: list[dict] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    all_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            if not all_entries:
                continue

            selected: list[dict] = []
            seen_indices: set[int] = set()

            for i in range(max(0, len(all_entries) - limit_per_channel), len(all_entries)):
                if i not in seen_indices:
                    selected.append(all_entries[i])
                    seen_indices.add(i)

            for i, entry in enumerate(all_entries):
                if i in seen_indices:
                    continue
                ts_str = entry.get("ts", "")
                try:
                    ts = ensure_aware(datetime.fromisoformat(ts_str))
                except (ValueError, TypeError):
                    continue
                is_human = entry.get("source") == "human"
                is_mention = mention_tag in entry.get("text", "")
                if (is_human and ts >= cutoff_24h) or is_mention:
                    selected.append(entry)
                    seen_indices.add(i)

            for entry in selected:
                result.append(
                    ActivityEntry(
                        ts=entry.get("ts", ""),
                        type="channel_post",
                        content=entry.get("text", ""),
                        summary=entry.get("text", "")[:100],
                        from_person=entry.get("from", ""),
                        channel=channel_name,
                    )
                )

    except Exception:
        logger.warning("Failed to read shared channels", exc_info=True)

    _MAX_CHANNEL_ENTRIES = 15
    if len(result) > _MAX_CHANNEL_ENTRIES:
        result.sort(key=lambda e: e.ts, reverse=True)
        result = result[:_MAX_CHANNEL_ENTRIES]

    return result


def prioritize_entries(
    entries: list,
    sender_name: str,
    keywords: list[str],
) -> list:
    """Prioritize activity entries for priming.

    Priority order:
    1. Own actions (message_sent, response_sent, message_received)
    2. Entries involving the current sender (most relevant)
    3. Entries matching keywords (topically relevant)
    4. Most recent entries (temporal relevance, timestamp-based)
    """
    from core.memory.activity import ActivityEntry

    keywords_lower = {kw.lower() for kw in keywords} if keywords else set()

    base_ts: datetime | None = None
    if entries:
        try:
            base_ts = datetime.fromisoformat(entries[0].ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    scored: list[tuple[float, int, ActivityEntry]] = []
    for i, entry in enumerate(entries):
        score = 0.0

        if entry.type in _OWN_ACTION_TYPES:
            if entry.type == "message_received":
                from_type = (entry.meta or {}).get("from_type", "")
                if from_type != "anima":
                    score += 15.0
                else:
                    origin_chain = (entry.meta or {}).get("origin_chain") or []
                    if "human" in origin_chain:
                        score += 15.0
            else:
                score += 15.0

        if entry.from_person == sender_name or entry.to_person == sender_name:
            score += 10.0

        text = (entry.content + " " + entry.summary).lower()
        matching_kw = sum(1 for kw in keywords_lower if kw in text)
        score += matching_kw * 3.0

        if base_ts is not None:
            try:
                entry_ts = datetime.fromisoformat(entry.ts.replace("Z", "+00:00"))
                elapsed_seconds = (entry_ts - base_ts).total_seconds()
                score += elapsed_seconds / 600
            except (ValueError, AttributeError):
                score += i * 0.1
        else:
            score += i * 0.1

        scored.append((score, i, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_entries = [e for _, _, e in scored[:50]]
    top_entries.sort(key=lambda e: e.ts)
    return top_entries


async def fallback_episodes_and_channels(anima_dir: Path, shared_dir: Path | None) -> str:
    """Fallback: read old episodes + channels when activity_log is empty."""
    parts: list[str] = []

    episodes = await read_old_episodes(anima_dir)
    if episodes:
        parts.append(episodes)

    channels = await read_old_channels(anima_dir, shared_dir)
    if channels:
        parts.append(channels)

    return "\n\n---\n\n".join(parts) if parts else ""


async def read_old_episodes(anima_dir: Path) -> str:
    """Read old episode files (migration fallback for Channel B)."""
    episodes_dir = anima_dir / "episodes"
    if not episodes_dir.is_dir():
        return ""

    parts: list[str] = []
    today = today_local()

    for offset in range(2):
        target_date = today - timedelta(days=offset)
        path = episodes_dir / f"{target_date.isoformat()}.md"

        if not path.exists():
            continue

        try:
            content = await run_sync(path.read_text, encoding="utf-8")
            lines = content.strip().splitlines()
            if len(lines) > 30:
                lines = lines[-30:]
            parts.append("\n".join(lines))
        except Exception as e:
            logger.warning("Channel B fallback: Failed to read episode %s: %s", path, e)

    if not parts:
        return ""

    return "\n\n---\n\n".join(parts)


async def read_old_channels(anima_dir: Path, shared_dir: Path | None) -> str:
    """Read old shared channel files (migration fallback for Channel E)."""
    if not shared_dir:
        return ""

    channels_dir = shared_dir / "channels"
    if not channels_dir.is_dir():
        return ""

    anima_name = anima_dir.name
    mention_tag = f"@{anima_name}"
    now = now_local()
    cutoff_24h = now - timedelta(hours=24)

    parts: list[str] = []

    for channel_name in ("general", "ops"):
        channel_file = channels_dir / f"{channel_name}.jsonl"
        if not channel_file.exists():
            continue

        try:
            content = await run_sync(channel_file.read_text, encoding="utf-8")
        except OSError:
            continue

        lines = content.strip().splitlines()
        if not lines:
            continue

        entries: list[dict] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not entries:
            continue

        selected: list[dict] = []
        seen_indices: set[int] = set()

        for i in range(max(0, len(entries) - 5), len(entries)):
            if i not in seen_indices:
                selected.append(entries[i])
                seen_indices.add(i)

        for i, entry in enumerate(entries):
            if i in seen_indices:
                continue
            ts_str = entry.get("ts", "")
            try:
                ts = ensure_aware(datetime.fromisoformat(ts_str))
            except (ValueError, TypeError):
                continue
            is_human = entry.get("source") == "human"
            is_mention = mention_tag in entry.get("text", "")
            if (is_human and ts >= cutoff_24h) or is_mention:
                selected.append(entry)
                seen_indices.add(i)

        selected.sort(key=lambda e: e.get("ts", ""))

        channel_parts: list[str] = []
        for entry in selected:
            src = entry.get("source", "anima")
            marker = " [human]" if src == "human" else ""
            channel_parts.append(f"[{entry.get('ts', '?')}] {entry.get('from', '?')}{marker}: {entry.get('text', '')}")

        if channel_parts:
            parts.append(f"### #{channel_name}")
            parts.extend(channel_parts)
            parts.append("")

    if not parts:
        return ""

    return "\n".join(parts)
