# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Discord markup helpers: plain-text cleanup and length limits."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────

DISCORD_MESSAGE_LIMIT = 2000

JST = timezone(timedelta(hours=9))

# Discord epoch for snowflake timestamp extraction (milliseconds).
_DISCORD_EPOCH_MS = 1420070400000

# ── Patterns ───────────────────────────────────────────────

_RE_USER_MENTION = re.compile(r"<@!?(\d+)>")
_RE_CHANNEL_MENTION = re.compile(r"<#(\d+)>")
_RE_EMOJI = re.compile(r"<a?:(\w+):\d+>")
_RE_TIMESTAMP = re.compile(r"<t:(\d+)(?::\w)?>")


# ── Public API ─────────────────────────────────────────────


def clean_discord_markup(text: str, cache: dict[str, str] | None = None) -> str:
    """Convert Discord-specific markup to readable plain text.

    Standard Markdown (bold, italic, code fences) is left unchanged.

    Args:
        text: Raw message content from Discord.
        cache: Optional user-ID → display-name mapping for mention resolution.

    Returns:
        Text with mentions, custom emoji, and timestamp tags simplified.
    """
    if not text:
        return ""

    def _replace_user_mention(m: re.Match[str]) -> str:
        uid = m.group(1)
        if cache and uid in cache:
            return f"@{cache[uid]}"
        return f"@{uid}"

    out = text
    out = _RE_USER_MENTION.sub(_replace_user_mention, out)
    out = _RE_CHANNEL_MENTION.sub(r"#\1", out)
    out = _RE_EMOJI.sub(r":\1:", out)

    def _ts_replace(m: re.Match[str]) -> str:
        try:
            ts = int(m.group(1))
            dt = datetime.fromtimestamp(float(ts), tz=UTC).astimezone(JST)
            return dt.strftime("%Y-%m-%d %H:%M:%S JST")
        except (ValueError, OSError):
            return m.group(0)

    out = _RE_TIMESTAMP.sub(_ts_replace, out)
    return out


def md_to_discord(text: str) -> str:
    """Prepare text for Discord ``content`` field with length cap.

    Discord supports CommonMark-style formatting natively, so this is
    mostly a pass-through that enforces :data:`DISCORD_MESSAGE_LIMIT`.
    Newlines are preserved (Discord renders them as line breaks).

    Args:
        text: Markdown source.

    Returns:
        Truncated text safe for ``content`` fields.
    """
    if not text:
        return ""
    if len(text) <= DISCORD_MESSAGE_LIMIT:
        return text
    return text[: DISCORD_MESSAGE_LIMIT - 3] + "..."


def truncate(text: str, limit: int = 2000) -> str:
    """Collapse newlines to spaces and truncate, appending ``...`` when trimmed.

    Args:
        text: Input string.
        limit: Maximum length before truncation.

    Returns:
        Single-line truncated string.
    """
    if not text:
        return ""
    single = " ".join(text.split())
    if len(single) <= limit:
        return single
    if limit <= 3:
        return single[:limit]
    return single[: limit - 3] + "..."


def format_discord_timestamp(snowflake_id: str) -> str:
    """Format a Discord snowflake ID as a JST datetime string.

    Timestamp ms = ``(int(snowflake_id) >> 22) + 1420070400000``.

    Args:
        snowflake_id: Numeric snowflake string.

    Returns:
        ISO-like local JST string, or the original value on parse errors.
    """
    try:
        snow = int(snowflake_id)
    except (ValueError, TypeError):
        logger.debug("Invalid snowflake for timestamp: %s", snowflake_id)
        return str(snowflake_id)

    ms = (snow >> 22) + _DISCORD_EPOCH_MS
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=UTC).astimezone(JST)
        return dt.strftime("%Y-%m-%d %H:%M:%S JST")
    except (ValueError, OSError):
        return str(snowflake_id)
