from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reply routing for call_human notifications.

Maps Slack message timestamps (ts) to originating Anima names so that
threaded replies to call_human notifications can be routed back to the
Anima that sent the original notification.

Storage: ``{data_dir}/run/notification_map.json``

Note: Only notifications sent via the Bot Token API (``chat.postMessage``)
support reply routing, because the API returns a message ``ts`` that can
be persisted in the mapping.  Slack Incoming Webhooks do **not** return a
``ts`` in their response, so notifications sent that way cannot be mapped
and thread replies to them will not be routed back.
"""

import fcntl
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.paths import get_data_dir

logger = logging.getLogger("animaworks.notification.reply_routing")

_MAX_AGE_DAYS = 7
_MAX_REPLY_LENGTH = 4000
_THREAD_CTX_SUMMARY_LIMIT = 150

# Slack mrkdwn patterns
_RE_USER_MENTION = re.compile(r"<@[A-Za-z0-9]+>")
_RE_LINK = re.compile(r"<(https?://[^|>]+)\|([^>]+)>")
_RE_LINK_BARE = re.compile(r"<(https?://[^>]+)>")
_RE_CHANNEL = re.compile(r"<#[A-Za-z0-9]+\|([^>]+)>")


def _map_path() -> Path:
    return get_data_dir() / "run" / "notification_map.json"


# ── Public API ──────────────────────────────────────────


def save_notification_mapping(
    ts: str,
    channel: str,
    anima_name: str,
    *,
    notification_text: str = "",
) -> None:
    """Persist a Slack message ts → Anima mapping for reply routing.

    Args:
        ts: Slack message timestamp of the notification.
        channel: Slack channel ID where the notification was posted.
        anima_name: Name of the Anima that sent the notification.
        notification_text: Original notification content (subject + body)
            so that thread replies can include context about what was notified.
    """
    path = _map_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("a+") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                fd.seek(0)
                raw = fd.read()
                data: dict[str, Any] = json.loads(raw) if raw.strip() else {}

                entry: dict[str, Any] = {
                    "anima": anima_name,
                    "channel": channel,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                if notification_text:
                    entry["notification_text"] = notification_text[:2000]
                data[ts] = entry
                _prune_old_entries_inplace(data)

                fd.seek(0)
                fd.truncate()
                fd.write(json.dumps(data, ensure_ascii=False, indent=2))
                fd.flush()
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        logger.exception("Failed to save notification mapping for ts=%s", ts)


def lookup_notification_mapping(thread_ts: str) -> dict[str, str] | None:
    """Look up which Anima sent the notification with the given ts.

    Returns ``{"anima": "...", "channel": "...", "notification_text": "..."}``
    or ``None``.  ``notification_text`` may be empty for older mappings.
    """
    path = _map_path()
    if not path.exists():
        return None
    try:
        with path.open("r") as fd:
            fcntl.flock(fd, fcntl.LOCK_SH)
            try:
                data = json.load(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    except (json.JSONDecodeError, OSError):
        return None
    entry = data.get(thread_ts)
    if entry is None:
        return None
    return {
        "anima": entry["anima"],
        "channel": entry["channel"],
        "notification_text": entry.get("notification_text", ""),
    }


def prune_old_entries(max_age_days: int = _MAX_AGE_DAYS) -> None:
    """Remove entries older than *max_age_days* from the mapping file."""
    path = _map_path()
    if not path.exists():
        return
    try:
        with path.open("a+") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                fd.seek(0)
                raw = fd.read()
                data: dict[str, Any] = json.loads(raw) if raw.strip() else {}
                before = len(data)
                _prune_old_entries_inplace(data, max_age_days)
                if len(data) < before:
                    fd.seek(0)
                    fd.truncate()
                    fd.write(json.dumps(data, ensure_ascii=False, indent=2))
                    fd.flush()
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        logger.exception("Failed to prune notification mapping")


def _prune_old_entries_inplace(
    data: dict[str, Any],
    max_age_days: int = _MAX_AGE_DAYS,
) -> None:
    """Remove stale entries from *data* dict in-place."""
    now = datetime.now(UTC)
    stale = [ts for ts, entry in data.items() if _age_days(entry.get("created_at", ""), now) > max_age_days]
    for ts in stale:
        del data[ts]


def _age_days(iso_str: str, now: datetime) -> float:
    try:
        created = datetime.fromisoformat(iso_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (now - created).total_seconds() / 86400
    except (ValueError, TypeError):
        return float("inf")


def route_thread_reply(
    event: dict,
    shared_dir: Path,
    *,
    slack_token: str = "",
) -> bool:
    """Route a Slack thread reply to the originating Anima if applicable.

    Checks event.thread_ts against the notification mapping. If a match
    is found, sanitizes the reply text, fetches thread context (so the
    Anima knows what the reply is about), and delivers it to the
    originating Anima's inbox via Messenger.receive_external().

    Args:
        event: Slack message event dict containing at minimum 'thread_ts',
               'text', 'ts', 'user', 'channel' keys.
        shared_dir: Path to the AnimaWorks shared directory.
        slack_token: Bot token for fetching thread context via Slack API.

    Returns:
        True if the reply was routed (caller should stop processing),
        False if no matching notification was found (caller should fall through).
    """
    thread_ts = event.get("thread_ts")
    if not thread_ts:
        return False

    mapping = lookup_notification_mapping(thread_ts)
    if mapping is None:
        return False

    target = mapping["anima"]
    text = sanitize_slack_reply(event.get("text", ""))
    if not text:
        return False

    # Fetch full thread context via Slack API so the Anima knows
    # what the reply is about (e.g. which call_human notification).
    thread_ctx = ""
    ctx_source = "none"
    channel_id = event.get("channel", "")
    if slack_token and channel_id:
        thread_ctx = _fetch_thread_context_for_reply(slack_token, channel_id, thread_ts)
        if thread_ctx:
            ctx_source = "api"

    # Fallback: use stored notification text if API fetch failed or unavailable
    if not thread_ctx:
        notification_text = mapping.get("notification_text", "")
        if notification_text:
            summary = notification_text.replace("\n", " ")[:_THREAD_CTX_SUMMARY_LIMIT]
            thread_ctx = (
                f"[Thread context — this is a reply to a call_human notification]\n"
                f"  {target}: {summary}\n"
                f"[/Thread context]\n\n"
            )
            ctx_source = "stored"

    content = thread_ctx + text if thread_ctx else text

    from core.messenger import Messenger

    messenger = Messenger(shared_dir, target)
    messenger.receive_external(
        content=content,
        source="slack",
        source_message_id=event.get("ts", ""),
        external_user_id=event.get("user", ""),
        external_channel_id=event.get("channel", ""),
        external_thread_ts=event.get("thread_ts", ""),
    )
    logger.info(
        "Thread reply routed: thread_ts=%s -> anima=%s (ctx=%s)",
        thread_ts,
        target,
        ctx_source,
    )
    return True


def _fetch_thread_context_for_reply(
    token: str,
    channel_id: str,
    thread_ts: str,
    *,
    limit: int = 10,
) -> str:
    """Fetch Slack thread context for call_human reply routing.

    Returns a concise ``[Thread context]`` block with the parent message's
    first line and reply count, or an empty string on failure.
    """
    if not token or not thread_ts:
        return ""
    try:
        from core.tools.slack import SlackClient

        client = SlackClient(token=token)
        replies = client.thread_replies(channel_id, thread_ts)
        if len(replies) <= 1:
            return ""
        parent = replies[0]
        parent_user = parent.get("user", "unknown")
        parent_text = parent.get("text", "").replace("\n", " ")[:_THREAD_CTX_SUMMARY_LIMIT]
        reply_count = len(replies) - 1
        lines = [
            "[Thread context — this message is a reply in a Slack thread]",
            f"  <@{parent_user}>: {parent_text}",
            f"  ({reply_count} replies in thread)",
            "[/Thread context]",
            "",
        ]
        return "\n".join(lines)
    except Exception:
        logger.warning("Failed to fetch thread context for reply routing", exc_info=True)
        return ""


def sanitize_slack_reply(text: str, max_length: int = _MAX_REPLY_LENGTH) -> str:
    """Strip Slack mrkdwn formatting and truncate for safe inbox delivery."""
    # Remove bot @mentions  <@U0123BOT>
    text = _RE_USER_MENTION.sub("", text)
    # <url|label> → label
    text = _RE_LINK.sub(r"\2", text)
    # <url> → url
    text = _RE_LINK_BARE.sub(r"\1", text)
    # <#C123|channel-name> → #channel-name
    text = _RE_CHANNEL.sub(r"#\1", text)
    # Bold *text* → text, italic _text_ → text, strike ~text~ → text
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)~([^~]+)~(?!\w)", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text
