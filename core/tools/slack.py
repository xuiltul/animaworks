# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Slack integration for AnimaWorks.

Provides:
- SlackClient: Slack Web API wrapper with rate-limit retry and pagination
- MessageCache: SQLite cache for offline search and unreplied detection
- get_tool_schemas(): Anthropic tool_use schemas
- cli_main(): standalone CLI entry point
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.tools._async_compat import run_sync
from core.tools._base import ToolConfigError, get_credential, logger
from core.tools._cache import BaseMessageCache
from core.tools._retry import retry_on_rate_limit

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "channels": {"expected_seconds": 10, "background_eligible": False},
    "messages": {"expected_seconds": 30, "background_eligible": False},
    "send":     {"expected_seconds": 10, "background_eligible": False},
    "search":   {"expected_seconds": 30, "background_eligible": False},
    "unreplied": {"expected_seconds": 30, "background_eligible": False},
}

WebClient = None
SlackApiError = None

def _require_slack_sdk():
    global WebClient, SlackApiError
    if WebClient is None:
        try:
            from slack_sdk import WebClient as _WC
            from slack_sdk.errors import SlackApiError as _SAE
            WebClient = _WC
            SlackApiError = _SAE
        except ImportError:
            raise ImportError(
                "slack tool requires 'slack-sdk'. Install with: pip install animaworks[communication]"
            )
    return WebClient

# ============================================================
# Constants
# ============================================================

JST = timezone(timedelta(hours=9))
RATE_LIMIT_RETRY_MAX = 5
RATE_LIMIT_WAIT_DEFAULT = 30

DEFAULT_CACHE_DIR = Path.home() / ".animaworks" / "cache" / "slack"


# ============================================================
# Helper functions
# ============================================================

def format_slack_ts(ts: str) -> str:
    """Convert a Slack timestamp (e.g. '1707123456.789012') to JST datetime string."""
    try:
        epoch = float(ts)
        dt = datetime.fromtimestamp(epoch, tz=JST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return ts


def md_to_slack_mrkdwn(text: str) -> str:
    """Convert standard Markdown to Slack mrkdwn format.

    Handles:
    - **bold** → *bold*
    - *italic* → _italic_
    - ***bold italic*** → *_bold italic_*
    - ~~strikethrough~~ → ~strikethrough~
    - [text](url) → <url|text>
    - ![alt](url) → <url>
    - # Heading → *Heading*
    - Bullet lists (- / *) → • item
    - Horizontal rules (---) → ───────────────
    - Code blocks and inline code are preserved as-is.
    """
    if not text:
        return ""

    # ── Protect code blocks / inline code from conversion ──
    _placeholders: list[str] = []

    def _save(matched_text: str) -> str:
        _placeholders.append(matched_text)
        return f"\x00PH{len(_placeholders) - 1}\x00"

    # Fenced code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", lambda m: _save(m.group(0)), text)
    # Inline code (` ... `)
    text = re.sub(r"`[^`]+`", lambda m: _save(m.group(0)), text)

    # ── Images: ![alt](url) → <url> ──
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"<\2>", text)

    # ── Links: [text](url) → <url|text> ──
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    # ── Headers: # Heading → *Heading* (protect from italic pass) ──
    text = re.sub(
        r"^#{1,6}\s+(.+)$",
        lambda m: _save(f"*{m.group(1)}*"),
        text,
        flags=re.MULTILINE,
    )

    # ── Bold+Italic: ***text*** → *_text_* ──
    text = re.sub(
        r"\*{3}(.+?)\*{3}",
        lambda m: _save(f"*_{m.group(1)}_*"),
        text,
    )

    # ── Bold: **text** → *text* (protect from italic pass) ──
    text = re.sub(
        r"\*{2}(.+?)\*{2}",
        lambda m: _save(f"*{m.group(1)}*"),
        text,
    )

    # ── Italic: *text* → _text_ ──
    # Avoid matching ** (already handled) and "* " (bullet list).
    text = re.sub(r"(?<![*])\*(?![* ])(.+?)(?<![* ])\*(?![*])", r"_\1_", text)

    # ── Strikethrough: ~~text~~ → ~text~ ──
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)

    # ── Bullet lists: - item / * item → • item ──
    text = re.sub(r"^(\s*)[-*]\s+", r"\1• ", text, flags=re.MULTILINE)

    # ── Horizontal rules: --- / *** / ___ → ─────────────── ──
    text = re.sub(
        r"^[-*_]{3,}\s*$", "───────────────", text, flags=re.MULTILINE
    )

    # ── Restore placeholders ──
    for i, ph in enumerate(_placeholders):
        text = text.replace(f"\x00PH{i}\x00", ph)

    return text


def clean_slack_markup(text: str, cache: dict | None = None) -> str:
    """Convert Slack markup to readable plain text.

    - <@U123ABC> -> @display_name (resolved via cache if available)
    - <#C123ABC|channel-name> -> #channel-name
    - <#C123ABC> -> #C123ABC
    - <https://example.com|Example> -> Example (https://example.com)
    - <https://example.com> -> https://example.com
    - &amp; -> &, &lt; -> <, &gt; -> >
    """
    if not text:
        return ""

    # User mentions: <@U06MJKLV0TG>
    def replace_user_mention(m):
        user_id = m.group(1)
        if cache and user_id in cache:
            return f"@{cache[user_id]}"
        return f"@{user_id}"

    text = re.sub(r"<@(U[A-Z0-9]+)>", replace_user_mention, text)

    # Channel references: <#C123|name> or <#C123>
    def replace_channel_ref(m):
        channel_id = m.group(1)
        name = m.group(2)
        if name:
            return f"#{name}"
        return f"#{channel_id}"

    text = re.sub(r"<#(C[A-Z0-9]+)(?:\|([^>]*))?>", replace_channel_ref, text)

    # URL links: <URL|label> or <URL>
    def replace_url(m):
        url = m.group(1)
        label = m.group(2)
        if label:
            return f"{label} ({url})"
        return url

    text = re.sub(r"<(https?://[^|>]+)(?:\|([^>]*))?>", replace_url, text)

    # Remaining <...> tags (mailto, etc.)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    return text


def truncate(s: str, length: int = 80) -> str:
    """Truncate a string, replacing newlines with spaces."""
    s = s.replace("\n", " ").strip()
    return s[:length] + "..." if len(s) > length else s


# ============================================================
# Slack API Client
# ============================================================

class SlackClient:
    """Slack Web API wrapper with rate-limit retry and cursor pagination."""

    def __init__(self, token: str | None = None):
        _require_slack_sdk()
        if token is None:
            token = get_credential("slack", "slack", env_var="SLACK_BOT_TOKEN")
        self.client = WebClient(token=token)
        self.my_user_id: str | None = None
        self.my_name: str | None = None
        self._channel_cache: dict[str, dict] = {}  # channel_id -> {id, name, ...}
        self._user_cache: dict[str, str] = {}  # user_id -> display_name

    def _call(self, method_name: str, **kwargs):
        """Call a WebClient method with 429 rate-limit retry."""
        method = getattr(self.client, method_name)

        class _SlackRateLimitError(Exception):
            """Wrapper to distinguish rate-limit errors for retry."""

            def __init__(self, original: Exception, retry_after: int) -> None:
                self.original = original
                self.retry_after = retry_after
                super().__init__(str(original))

        def _do_call():
            try:
                return method(**kwargs)
            except SlackApiError as e:
                if e.response.status_code == 429:
                    retry_after = int(
                        e.response.headers.get(
                            "Retry-After", RATE_LIMIT_WAIT_DEFAULT
                        )
                    )
                    raise _SlackRateLimitError(e, retry_after) from e
                raise

        def _get_retry_after(exc: Exception) -> float | None:
            if isinstance(exc, _SlackRateLimitError):
                return float(exc.retry_after)
            return None

        try:
            return retry_on_rate_limit(
                _do_call,
                max_retries=RATE_LIMIT_RETRY_MAX,
                default_wait=RATE_LIMIT_WAIT_DEFAULT,
                get_retry_after=_get_retry_after,
                retry_on=(_SlackRateLimitError,),
            )
        except _SlackRateLimitError as exc:
            raise exc.original from None

    async def _acall(self, method_name: str, **kwargs):
        """Async wrapper around :meth:`_call` using a thread-pool executor."""
        return await run_sync(self._call, method_name, **kwargs)

    def _paginate(self, method_name: str, response_key: str, **kwargs) -> list:
        """Cursor-based pagination. Returns all items."""
        all_items = []
        cursor = None
        while True:
            call_kwargs = dict(kwargs)
            if cursor:
                call_kwargs["cursor"] = cursor
            response = self._call(method_name, **call_kwargs)
            items = response.get(response_key, [])
            all_items.extend(items)
            # Check for next page
            metadata = response.get("response_metadata", {})
            next_cursor = metadata.get("next_cursor", "")
            if not next_cursor:
                break
            cursor = next_cursor
        return all_items

    def auth_test(self):
        """Get the bot's own info, setting my_user_id / my_name."""
        response = self._call("auth_test")
        self.my_user_id = response.get("user_id", "")
        self.my_name = response.get("user", "")
        return response

    def channels(self, exclude_archived: bool = True) -> list[dict]:
        """List joined channels (tries all types, skips on missing_scope)."""
        all_channels = []
        type_groups = [
            "public_channel",
            "private_channel",
            "im",
            "mpim",
        ]
        for ch_type in type_groups:
            try:
                chs = self._paginate(
                    "conversations_list",
                    "channels",
                    types=ch_type,
                    exclude_archived=exclude_archived,
                    limit=200,
                )
                all_channels.extend(chs)
            except SlackApiError as e:
                if "missing_scope" in str(e):
                    pass  # Scope not available, skip this type
                else:
                    raise
        # Cache channel info
        for ch in all_channels:
            self._channel_cache[ch["id"]] = ch
        return all_channels

    def channel_history(self, channel_id: str, limit: int = 100) -> list[dict]:
        """Get messages via conversations.history with pagination."""
        all_messages = []
        cursor = None
        remaining = limit
        while remaining > 0:
            fetch_count = min(remaining, 200)
            call_kwargs = {"channel": channel_id, "limit": fetch_count}
            if cursor:
                call_kwargs["cursor"] = cursor
            response = self._call("conversations_history", **call_kwargs)
            messages = response.get("messages", [])
            all_messages.extend(messages)
            remaining -= len(messages)
            if not messages:
                break
            metadata = response.get("response_metadata", {})
            next_cursor = metadata.get("next_cursor", "")
            if not next_cursor:
                break
            cursor = next_cursor
        return all_messages

    def thread_replies(self, channel_id: str, ts: str) -> list[dict]:
        """Get thread replies via conversations.replies."""
        all_replies = self._paginate(
            "conversations_replies",
            "messages",
            channel=channel_id,
            ts=ts,
            limit=200,
        )
        return all_replies

    def post_message(
        self, channel_id: str, text: str, thread_ts: str | None = None
    ) -> dict:
        """Send a message via chat.postMessage."""
        kwargs = {"channel": channel_id, "text": text}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        response = self._call("chat_postMessage", **kwargs)
        return response

    def users_list(self) -> list[dict]:
        """Get all workspace users."""
        all_users = self._paginate("users_list", "members", limit=200)
        # Cache display names
        for u in all_users:
            display = (
                u.get("profile", {}).get("display_name", "")
                or u.get("real_name", "")
                or u.get("name", "")
            )
            self._user_cache[u["id"]] = display
        return all_users

    def resolve_channel(self, name_or_id: str) -> str:
        """Resolve a channel name or ID to a channel_id.

        - C/D/G + alphanumeric is treated as an ID
        - Otherwise, search by name (exact match first, then partial)
        """
        # Looks like an ID
        if re.match(r"^[CDG][A-Z0-9]{8,}$", name_or_id):
            return name_or_id

        # Strip leading #
        name_or_id = name_or_id.lstrip("#")

        # Populate cache if empty
        if not self._channel_cache:
            self.channels()

        # Exact match first
        for ch_id, ch in self._channel_cache.items():
            ch_name = ch.get("name", "")
            if ch_name == name_or_id:
                return ch_id

        # Partial match
        matches = []
        for ch_id, ch in self._channel_cache.items():
            ch_name = ch.get("name", "")
            if name_or_id.lower() in ch_name.lower():
                matches.append((ch_id, ch))

        if len(matches) == 1:
            return matches[0][0]

        if len(matches) > 1:
            names = ", ".join(
                f"{cid} #{c.get('name', '?')}" for cid, c in matches
            )
            raise ToolConfigError(
                f"Multiple channels matched '{name_or_id}': {names}. "
                f"Specify the channel ID directly."
            )

        raise ToolConfigError(f"Channel '{name_or_id}' not found")

    def resolve_user_name(self, user_id: str) -> str:
        """Resolve user_id to a display name. Fetches from API if not cached."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        # Fetch user info from API
        try:
            response = self._call("users_info", user=user_id)
            user = response.get("user", {})
            display = (
                user.get("profile", {}).get("display_name", "")
                or user.get("real_name", "")
                or user.get("name", "")
            )
            self._user_cache[user_id] = display
            return display
        except SlackApiError:
            self._user_cache[user_id] = user_id
            return user_id

    def get_channel_name(self, channel_id: str) -> str:
        """Get channel name from channel_id."""
        if channel_id in self._channel_cache:
            ch = self._channel_cache[channel_id]
            name = ch.get("name", "")
            if name:
                return name
            # DM: return other user's name
            if ch.get("is_im"):
                other_user = ch.get("user", "")
                if other_user:
                    return f"DM:{self.resolve_user_name(other_user)}"
            return channel_id

        # Not cached: fetch via conversations.info
        try:
            response = self._call("conversations_info", channel=channel_id)
            ch = response.get("channel", {})
            self._channel_cache[channel_id] = ch
            name = ch.get("name", "")
            if name:
                return name
            if ch.get("is_im"):
                other_user = ch.get("user", "")
                if other_user:
                    return f"DM:{self.resolve_user_name(other_user)}"
            return channel_id
        except SlackApiError:
            return channel_id


# ============================================================
# Local Message Cache (SQLite)
# ============================================================

_SLACK_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    name TEXT,
    type TEXT,
    is_member INTEGER DEFAULT 0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    real_name TEXT,
    is_bot INTEGER DEFAULT 0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    channel_id TEXT,
    ts TEXT,
    user_id TEXT,
    user_name TEXT,
    text TEXT,
    thread_ts TEXT,
    reply_count INTEGER DEFAULT 0,
    ts_epoch REAL,
    send_time_jst TEXT,
    PRIMARY KEY (channel_id, ts)
);

CREATE TABLE IF NOT EXISTS sync_state (
    channel_id TEXT PRIMARY KEY,
    last_synced TEXT,
    oldest_ts TEXT,
    newest_ts TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_text ON messages(text);
CREATE INDEX IF NOT EXISTS idx_messages_ts_epoch ON messages(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(channel_id, thread_ts);
"""


class MessageCache(BaseMessageCache):
    """SQLite-backed cache for Slack messages, enabling offline search and
    unreplied-mention detection."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = DEFAULT_CACHE_DIR / "messages.db"
        super().__init__(db_path, _SLACK_SCHEMA_SQL)

    def upsert_channel(self, channel: dict):
        """Save/update channel info."""
        channel_id = channel.get("id", "")
        name = channel.get("name", "")
        # DM: name may be empty, use user field
        if not name and channel.get("is_im"):
            name = f"DM:{channel.get('user', '')}"
        if not name and channel.get("is_mpim"):
            name = channel.get("name_normalized", channel_id)

        # Determine channel type
        if channel.get("is_im"):
            ch_type = "im"
        elif channel.get("is_mpim"):
            ch_type = "mpim"
        elif channel.get("is_private"):
            ch_type = "private_channel"
        else:
            ch_type = "public_channel"

        is_member = 1 if channel.get("is_member", False) else 0
        # DM/mpim are always "joined"
        if ch_type in ("im", "mpim"):
            is_member = 1

        self.conn.execute(
            """INSERT OR REPLACE INTO channels
               (channel_id, name, type, is_member, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (channel_id, name, ch_type, is_member,
             datetime.now(JST).isoformat()),
        )
        self.conn.commit()

    def upsert_user(self, user: dict):
        """Save/update user info."""
        user_id = user.get("id", "")
        display_name = (
            user.get("profile", {}).get("display_name", "")
            or user.get("real_name", "")
            or user.get("name", "")
        )
        real_name = user.get("real_name", "")
        is_bot = 1 if user.get("is_bot", False) else 0

        self.conn.execute(
            """INSERT OR REPLACE INTO users
               (user_id, name, real_name, is_bot, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, display_name, real_name, is_bot,
             datetime.now(JST).isoformat()),
        )
        self.conn.commit()

    def upsert_messages(self, channel_id: str, messages: list[dict]):
        """Save/update messages to DB."""
        for m in messages:
            ts = m.get("ts", "")
            user_id = m.get("user", m.get("bot_id", ""))
            user_name = m.get("user_name", "")
            # Resolve user_name from users table if empty
            if not user_name and user_id:
                row = self.conn.execute(
                    "SELECT name FROM users WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if row:
                    user_name = row["name"]
            text = m.get("text", "")
            thread_ts = m.get("thread_ts", "")
            reply_count = m.get("reply_count", 0)
            try:
                ts_epoch = float(ts)
            except (ValueError, TypeError):
                ts_epoch = 0.0
            send_time_jst = format_slack_ts(ts)

            self.conn.execute(
                """INSERT OR REPLACE INTO messages
                   (channel_id, ts, user_id, user_name, text, thread_ts,
                    reply_count, ts_epoch, send_time_jst)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (channel_id, ts, user_id, user_name, text, thread_ts,
                 reply_count, ts_epoch, send_time_jst),
            )
        self.conn.commit()

    def update_sync_state(self, channel_id: str):
        """Update sync state with oldest/newest timestamps."""
        row = self.conn.execute(
            """SELECT MIN(ts) as oldest, MAX(ts) as newest
               FROM messages WHERE channel_id = ?""",
            (channel_id,),
        ).fetchone()
        oldest_ts = row["oldest"] if row else None
        newest_ts = row["newest"] if row else None

        self.conn.execute(
            """INSERT OR REPLACE INTO sync_state
               (channel_id, last_synced, oldest_ts, newest_ts)
               VALUES (?, ?, ?, ?)""",
            (channel_id, datetime.now(JST).isoformat(), oldest_ts, newest_ts),
        )
        self.conn.commit()

    def search(
        self,
        keyword: str,
        channel_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search cached messages by keyword."""
        query = """
            SELECT m.*, c.name as channel_name
            FROM messages m
            LEFT JOIN channels c ON m.channel_id = c.channel_id
            WHERE m.text LIKE ?
        """
        params: list = [f"%{keyword}%"]
        if channel_id:
            query += " AND m.channel_id = ?"
            params.append(channel_id)
        query += " ORDER BY m.ts_epoch DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, channel_id: str, limit: int = 20) -> list[dict]:
        """Get the most recent messages from a channel."""
        rows = self.conn.execute(
            """SELECT m.*, c.name as channel_name
               FROM messages m
               LEFT JOIN channels c ON m.channel_id = c.channel_id
               WHERE m.channel_id = ?
               ORDER BY m.ts_epoch DESC LIMIT ?""",
            (channel_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_mentions(
        self,
        my_user_id: str,
        limit: int = 200,
        config: dict | None = None,
    ) -> list[dict]:
        """Find messages addressed to me.

        Detection patterns:
        1. <@USER_ID> in message text
        2. All messages in DM channels from the other anima
        3. Messages in watch_channels from other users
        """
        if config is None:
            config = {}
        unreplied_cfg = config.get("unreplied", {})
        results: list[dict] = []
        seen: set[tuple[str, str]] = set()

        # 1) Text mentions: <@my_user_id> (all channels)
        mention_pattern = f"%<@{my_user_id}>%"
        query1 = """
            SELECT m.*, c.name as channel_name
            FROM messages m
            LEFT JOIN channels c ON m.channel_id = c.channel_id
            WHERE m.text LIKE ?
              AND m.user_id != ?
            ORDER BY m.ts_epoch DESC LIMIT ?
        """
        rows = self.conn.execute(
            query1, (mention_pattern, my_user_id, limit)
        ).fetchall()
        for r in rows:
            d = dict(r)
            key = (d["channel_id"], d["ts"])
            if key not in seen:
                seen.add(key)
                results.append(d)

        # 2) DM channels: all messages from the other anima
        if unreplied_cfg.get("include_direct_messages", True):
            dm_rows = self.conn.execute(
                "SELECT channel_id FROM channels WHERE type = 'im'"
            ).fetchall()
            dm_channel_ids = [r["channel_id"] for r in dm_rows]
            if dm_channel_ids:
                placeholders = ",".join("?" for _ in dm_channel_ids)
                query2 = f"""
                    SELECT m.*, c.name as channel_name
                    FROM messages m
                    LEFT JOIN channels c ON m.channel_id = c.channel_id
                    WHERE m.channel_id IN ({placeholders})
                      AND m.user_id != ?
                    ORDER BY m.ts_epoch DESC LIMIT ?
                """
                params2 = dm_channel_ids + [my_user_id, limit]
                rows2 = self.conn.execute(query2, params2).fetchall()
                for r in rows2:
                    d = dict(r)
                    key = (d["channel_id"], d["ts"])
                    if key not in seen:
                        seen.add(key)
                        results.append(d)

        # 3) watch_channels: messages from other users
        watch_channels = unreplied_cfg.get("watch_channels", [])
        if watch_channels:
            watch_ids = [
                wc.get("channel_id", "")
                for wc in watch_channels
                if wc.get("channel_id")
            ]
            if watch_ids:
                placeholders = ",".join("?" for _ in watch_ids)
                query3 = f"""
                    SELECT m.*, c.name as channel_name
                    FROM messages m
                    LEFT JOIN channels c ON m.channel_id = c.channel_id
                    WHERE m.channel_id IN ({placeholders})
                      AND m.user_id != ?
                    ORDER BY m.ts_epoch DESC LIMIT ?
                """
                params3 = watch_ids + [my_user_id, limit]
                rows3 = self.conn.execute(query3, params3).fetchall()
                for r in rows3:
                    d = dict(r)
                    key = (d["channel_id"], d["ts"])
                    if key not in seen:
                        seen.add(key)
                        results.append(d)

        # Sort by time descending
        results.sort(key=lambda x: x.get("ts_epoch", 0), reverse=True)
        return results[:limit]

    def find_unreplied(
        self,
        my_user_id: str,
        limit: int = 200,
        config: dict | None = None,
    ) -> list[dict]:
        """Find messages addressed to me that I haven't replied to.

        - Threaded messages: check if I replied in the same thread
        - Top-level messages: check if I posted in the channel after it
        """
        mentions = self.find_mentions(my_user_id, limit, config=config)
        unreplied = []

        for m in mentions:
            channel_id = m["channel_id"]
            ts_epoch = m.get("ts_epoch", 0)
            thread_ts = m.get("thread_ts", "")

            if thread_ts:
                # Threaded: check same thread for my reply
                row = self.conn.execute(
                    """SELECT COUNT(*) as c FROM messages
                       WHERE channel_id = ?
                         AND thread_ts = ?
                         AND user_id = ?
                         AND ts_epoch > ?""",
                    (channel_id, thread_ts, my_user_id, ts_epoch),
                ).fetchone()
                if row["c"] == 0:
                    unreplied.append(m)
            else:
                # Top-level: check if I posted in the channel after this
                row = self.conn.execute(
                    """SELECT COUNT(*) as c FROM messages
                       WHERE channel_id = ?
                         AND user_id = ?
                         AND ts_epoch > ?""",
                    (channel_id, my_user_id, ts_epoch),
                ).fetchone()
                if row["c"] == 0:
                    unreplied.append(m)

        return unreplied

    def get_channel_name(self, channel_id: str) -> str:
        """Get channel name by ID from the cache DB."""
        row = self.conn.execute(
            "SELECT name FROM channels WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        return row["name"] if row else channel_id

    def get_user_name(self, user_id: str) -> str:
        """Get user name by ID from the cache DB."""
        row = self.conn.execute(
            "SELECT name FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["name"] if row else user_id

    def get_user_name_cache(self) -> dict:
        """Return a user_id -> name mapping dict."""
        rows = self.conn.execute("SELECT user_id, name FROM users").fetchall()
        return {r["user_id"]: r["name"] for r in rows}

    def get_stats(self) -> dict:
        """Return cache statistics."""
        channels = self.conn.execute(
            "SELECT COUNT(*) as c FROM channels"
        ).fetchone()["c"]
        users = self.conn.execute(
            "SELECT COUNT(*) as c FROM users"
        ).fetchone()["c"]
        msgs = self.conn.execute(
            "SELECT COUNT(*) as c FROM messages"
        ).fetchone()["c"]

        # Oldest/newest message timestamps
        oldest = self.conn.execute(
            "SELECT MIN(send_time_jst) as t FROM messages"
        ).fetchone()["t"]
        newest = self.conn.execute(
            "SELECT MAX(send_time_jst) as t FROM messages"
        ).fetchone()["t"]

        # Synced channel count
        synced = self.conn.execute(
            "SELECT COUNT(*) as c FROM sync_state"
        ).fetchone()["c"]

        return {
            "channels": channels,
            "users": users,
            "messages": msgs,
            "synced_channels": synced,
            "oldest_message": oldest or "N/A",
            "newest_message": newest or "N/A",
        }


# ============================================================
# Tool schemas (Anthropic tool_use format)
# ============================================================

def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for Slack tools."""
    return [
        {
            "name": "slack_send",
            "description": (
                "Send a message to a Slack channel or DM. "
                "The channel can be specified by ID or name. "
                "Optionally reply in a thread by providing thread_ts."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name or ID (e.g. '#general' or 'C01234ABCDE').",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message text to send.",
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Optional thread timestamp to reply in a thread.",
                    },
                },
                "required": ["channel", "message"],
            },
        },
        {
            "name": "slack_messages",
            "description": (
                "Get recent messages from a Slack channel. "
                "Fetches from API and caches locally, then returns from cache."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name or ID.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return (default 20).",
                        "default": 20,
                    },
                },
                "required": ["channel"],
            },
        },
        {
            "name": "slack_search",
            "description": (
                "Search cached Slack messages by keyword. "
                "Searches the local SQLite cache. Run sync first for fresh data."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Search keyword.",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Optional channel name or ID to filter results.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 50).",
                        "default": 50,
                    },
                },
                "required": ["keyword"],
            },
        },
        {
            "name": "slack_unreplied",
            "description": (
                "Find Slack messages addressed to me that I haven't replied to. "
                "Uses the local cache. Includes DMs and watched channels."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "slack_channels",
            "description": "List Slack channels that the bot has joined.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    ]


# ============================================================
# CLI entry point
# ============================================================


def get_cli_guide() -> str:
    """Return CLI usage guide for Slack tools."""
    return """\
### Slack
```bash
animaworks-tool slack channels -j
animaworks-tool slack messages <チャンネル名またはID> -j
animaworks-tool slack send <チャンネル名またはID> "メッセージ本文"
animaworks-tool slack search "キーワード" -j
animaworks-tool slack unreplied -j
```"""


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI entry point for the Slack tool."""
    parser = argparse.ArgumentParser(
        prog="animaworks-slack",
        description="Slack CLI (AnimaWorks integration)",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # send
    p = sub.add_parser("send", help="Send a message")
    p.add_argument("channel", help="Channel name or ID")
    p.add_argument("message", nargs="+", help="Message body")
    p.add_argument("--thread", help="Thread timestamp for threaded reply")

    # messages
    p = sub.add_parser("messages", help="Get recent messages")
    p.add_argument("channel", help="Channel name or ID")
    p.add_argument(
        "-n", "--num", type=int, default=20, help="Number of messages (default 20)"
    )

    # search
    p = sub.add_parser("search", help="Search cached messages")
    p.add_argument("keyword", nargs="+", help="Search keyword")
    p.add_argument("-c", "--channel", help="Filter by channel name or ID")
    p.add_argument(
        "-n", "--num", type=int, default=50, help="Max results (default 50)"
    )

    # unreplied
    p = sub.add_parser("unreplied", help="Show unreplied messages addressed to me")
    p.add_argument("--json", action="store_true", help="Output as JSON")

    # channels
    sub.add_parser("channels", help="List joined channels")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        client = SlackClient()
    except ToolConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        _run_cli_command(client, args)
    except SlackApiError as e:
        error = e.response.get("error", str(e))
        print(f"Slack API error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


def _run_cli_command(client: SlackClient, args) -> None:
    """Dispatch CLI subcommands."""
    if args.command == "send":
        channel_id = client.resolve_channel(args.channel)
        message = md_to_slack_mrkdwn(" ".join(args.message))
        thread_ts = getattr(args, "thread", None)
        response = client.post_message(channel_id, message, thread_ts=thread_ts)
        ts = response.get("ts", "")
        channel = response.get("channel", channel_id)
        print(f"Sent (channel: {channel}, ts: {ts})")

    elif args.command == "messages":
        channel_id = client.resolve_channel(args.channel)
        cache = MessageCache()
        try:
            print("Fetching messages...", file=sys.stderr, end=" ", flush=True)
            msgs = client.channel_history(channel_id, limit=args.num)
            print(f"{len(msgs)} fetched", file=sys.stderr)

            if msgs:
                # Resolve user names and cache
                for m in msgs:
                    uid = m.get("user", m.get("bot_id", ""))
                    if uid:
                        m["user_name"] = client.resolve_user_name(uid)
                cache.upsert_messages(channel_id, msgs)
                cache.update_sync_state(channel_id)

            # Display from cache
            cached = cache.get_recent(channel_id, limit=args.num)
            user_name_map = cache.get_user_name_cache()

            if cached:
                for m in reversed(cached):
                    ts = m.get("send_time_jst", "")
                    user_name = m.get("user_name", "")
                    if not user_name and m.get("user_id"):
                        user_name = cache.get_user_name(m["user_id"])
                    channel_name = m.get("channel_name", "")
                    channel_tag = f"[#{channel_name}] " if channel_name else ""
                    text = clean_slack_markup(
                        m.get("text", ""), cache=user_name_map
                    )
                    print(f"{ts} {channel_tag}{user_name}")
                    for line in text.strip().split("\n"):
                        print(f"  {line}")
                    print()
            else:
                print("No messages found.")
        finally:
            cache.close()

    elif args.command == "search":
        cache = MessageCache()
        try:
            keyword = " ".join(args.keyword)
            channel_id = None
            if args.channel:
                channel_id = client.resolve_channel(args.channel)
            results = cache.search(keyword, channel_id=channel_id, limit=args.num)
            user_name_map = cache.get_user_name_cache()

            if not results:
                print(f"No messages matching '{keyword}'.")
            else:
                for m in results:
                    m["text"] = clean_slack_markup(
                        m.get("text", ""), cache=user_name_map
                    )
                    if not m.get("user_name") and m.get("user_id"):
                        m["user_name"] = cache.get_user_name(m["user_id"])
                print(f"Results: {len(results)} (keyword: '{keyword}')\n")
                for m in reversed(results):
                    ts = m.get("send_time_jst", "")
                    user_name = m.get("user_name", "?")
                    channel_name = m.get("channel_name", "")
                    channel_tag = f"[#{channel_name}] " if channel_name else ""
                    text = m.get("text", "").strip()
                    print(f"{ts} {channel_tag}{user_name}")
                    for line in text.split("\n"):
                        print(f"  {line}")
                    print()
        finally:
            cache.close()

    elif args.command == "unreplied":
        cache = MessageCache()
        try:
            client.auth_test()
            my_user_id = client.my_user_id
            my_name = client.my_name

            unreplied = cache.find_unreplied(my_user_id)
            user_name_map = cache.get_user_name_cache()

            if getattr(args, "json", False):
                output = []
                for m in unreplied:
                    text_clean = clean_slack_markup(
                        m.get("text", ""), cache=user_name_map
                    )
                    output.append({
                        "channel_id": m.get("channel_id", ""),
                        "channel_name": m.get(
                            "channel_name", m.get("channel_id", "")
                        ),
                        "ts": m.get("ts", ""),
                        "user_id": m.get("user_id", ""),
                        "user_name": m.get("user_name", ""),
                        "text": text_clean.strip(),
                        "ts_epoch": m.get("ts_epoch", 0),
                        "send_time_jst": m.get("send_time_jst", ""),
                        "thread_ts": m.get("thread_ts", ""),
                    })
                print(json.dumps(output, ensure_ascii=False, indent=2))
            elif not unreplied:
                print(f"No unreplied messages ({my_name} / ID: {my_user_id})")
            else:
                print(f"=== Unreplied: {len(unreplied)} ({my_name}) ===\n")
                for m in unreplied:
                    ts = m.get("send_time_jst", "")
                    user_name = m.get("user_name", "")
                    if not user_name and m.get("user_id"):
                        user_name = cache.get_user_name(m["user_id"])
                    channel_name = m.get(
                        "channel_name", m.get("channel_id", "")
                    )
                    text = clean_slack_markup(
                        m.get("text", ""), cache=user_name_map
                    )
                    text_clean = text.strip()
                    text_preview = text_clean.replace("\n", " ")[:120]
                    if len(text_clean) > 120:
                        text_preview += "..."
                    thread_ts = m.get("thread_ts", "")
                    thread_info = (
                        f"  (thread: {thread_ts})" if thread_ts else ""
                    )
                    print(f"{ts} [#{channel_name}]{thread_info}")
                    print(f"  From: {user_name}")
                    print(f"  {text_preview}")
                    print()
        finally:
            cache.close()

    elif args.command == "channels":
        all_channels = client.channels()
        cache = MessageCache()
        try:
            for ch in all_channels:
                cache.upsert_channel(ch)

            member_channels = [
                ch
                for ch in all_channels
                if ch.get("is_member", False)
                or ch.get("is_im", False)
                or ch.get("is_mpim", False)
            ]
            member_channels.sort(
                key=lambda c: c.get("updated", 0), reverse=True
            )

            print(f"{'ID':>12}  {'Type':10}  {'Members':>7}  {'Name'}")
            print("-" * 70)
            for ch in member_channels:
                ch_id = ch.get("id", "")
                name = ch.get("name", "")
                num_members = ch.get("num_members", 0)

                if ch.get("is_im"):
                    ch_type = "DM"
                    other_user = ch.get("user", "")
                    if other_user:
                        name = f"DM:{client.resolve_user_name(other_user)}"
                elif ch.get("is_mpim"):
                    ch_type = "GroupDM"
                elif ch.get("is_private"):
                    ch_type = "private"
                else:
                    ch_type = "public"

                print(f"{ch_id:>12}  {ch_type:10}  {num_members:>7}  {name}")

            print(f"\nTotal: {len(member_channels)} channels")
        finally:
            cache.close()


# ── Dispatch ──────────────────────────────────────────

def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    if name == "slack_send":
        client = SlackClient()
        channel_id = client.resolve_channel(args["channel"])
        return client.post_message(
            channel_id,
            md_to_slack_mrkdwn(args["message"]),
            thread_ts=args.get("thread_ts"),
        )
    if name == "slack_messages":
        client = SlackClient()
        channel_id = client.resolve_channel(args["channel"])
        cache = MessageCache()
        try:
            limit = args.get("limit", 20)
            msgs = client.channel_history(channel_id, limit=limit)
            if msgs:
                for m in msgs:
                    uid = m.get("user", m.get("bot_id", ""))
                    if uid:
                        m["user_name"] = client.resolve_user_name(uid)
                cache.upsert_messages(channel_id, msgs)
                cache.update_sync_state(channel_id)
            return cache.get_recent(channel_id, limit=limit)
        finally:
            cache.close()
    if name == "slack_search":
        client = SlackClient()
        cache = MessageCache()
        try:
            channel_id = None
            if args.get("channel"):
                channel_id = client.resolve_channel(args["channel"])
            return cache.search(
                args["keyword"], channel_id=channel_id, limit=args.get("limit", 50),
            )
        finally:
            cache.close()
    if name == "slack_unreplied":
        client = SlackClient()
        cache = MessageCache()
        try:
            client.auth_test()
            return cache.find_unreplied(client.my_user_id)
        finally:
            cache.close()
    if name == "slack_channels":
        client = SlackClient()
        return client.channels()
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    cli_main()