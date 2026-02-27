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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import get_data_dir

logger = logging.getLogger("animaworks.notification.reply_routing")

_MAX_AGE_DAYS = 7
_MAX_REPLY_LENGTH = 4000

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
) -> None:
    """Persist a Slack message ts → Anima mapping for reply routing."""
    path = _map_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("a+") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                fd.seek(0)
                raw = fd.read()
                data: dict[str, Any] = json.loads(raw) if raw.strip() else {}

                data[ts] = {
                    "anima": anima_name,
                    "channel": channel,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
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

    Returns ``{"anima": "...", "channel": "..."}`` or ``None``.
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
    return {"anima": entry["anima"], "channel": entry["channel"]}


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
    now = datetime.now(timezone.utc)
    stale = [
        ts
        for ts, entry in data.items()
        if _age_days(entry.get("created_at", ""), now) > max_age_days
    ]
    for ts in stale:
        del data[ts]


def _age_days(iso_str: str, now: datetime) -> float:
    try:
        created = datetime.fromisoformat(iso_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created).total_seconds() / 86400
    except (ValueError, TypeError):
        return float("inf")


def route_thread_reply(event: dict, shared_dir: Path) -> bool:
    """Route a Slack thread reply to the originating Anima if applicable.

    Checks event.thread_ts against the notification mapping. If a match
    is found, sanitizes the reply text and delivers it to the originating
    Anima's inbox via Messenger.receive_external().

    Args:
        event: Slack message event dict containing at minimum 'thread_ts',
               'text', 'ts', 'user', 'channel' keys.
        shared_dir: Path to the AnimaWorks shared directory.

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

    from core.messenger import Messenger

    messenger = Messenger(shared_dir, target)
    messenger.receive_external(
        content=text,
        source="slack",
        source_message_id=event.get("ts", ""),
        external_user_id=event.get("user", ""),
        external_channel_id=event.get("channel", ""),
    )
    logger.info(
        "Thread reply routed: thread_ts=%s -> anima=%s",
        thread_ts,
        target,
    )
    return True


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
