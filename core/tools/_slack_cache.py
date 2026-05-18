# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""SQLite message cache for Slack (offline search, unreplied detection)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from core.tools._cache import BaseMessageCache
from core.tools._slack_client import JST
from core.tools._slack_markdown import format_slack_ts

# Cache directory for Slack message cache.
# Can be overridden via ANIMAWORKS_SLACK_CACHE_DIR environment variable.
# This allows TaskExec/Codex sandbox environments to redirect cache writes
# to a writable location (e.g. /tmp/animaworks-cache/slack).
_DEFAULT_CACHE_DIR = Path.home() / ".animaworks" / "cache" / "slack"
DEFAULT_CACHE_DIR = Path(os.environ.get("ANIMAWORKS_SLACK_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))

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
            (channel_id, name, ch_type, is_member, datetime.now(JST).isoformat()),
        )
        self.conn.commit()

    def upsert_user(self, user: dict):
        """Save/update user info."""
        user_id = user.get("id", "")
        display_name = (
            user.get("profile", {}).get("display_name", "") or user.get("real_name", "") or user.get("name", "")
        )
        real_name = user.get("real_name", "")
        is_bot = 1 if user.get("is_bot", False) else 0

        self.conn.execute(
            """INSERT OR REPLACE INTO users
               (user_id, name, real_name, is_bot, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, display_name, real_name, is_bot, datetime.now(JST).isoformat()),
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
                (channel_id, ts, user_id, user_name, text, thread_ts, reply_count, ts_epoch, send_time_jst),
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
        rows = self.conn.execute(query1, (mention_pattern, my_user_id, limit)).fetchall()
        for r in rows:
            d = dict(r)
            key = (d["channel_id"], d["ts"])
            if key not in seen:
                seen.add(key)
                results.append(d)

        # 2) DM channels: all messages from the other anima
        if unreplied_cfg.get("include_direct_messages", True):
            dm_rows = self.conn.execute("SELECT channel_id FROM channels WHERE type = 'im'").fetchall()
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
            watch_ids = [wc.get("channel_id", "") for wc in watch_channels if wc.get("channel_id")]
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
        channels = self.conn.execute("SELECT COUNT(*) as c FROM channels").fetchone()["c"]
        users = self.conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
        msgs = self.conn.execute("SELECT COUNT(*) as c FROM messages").fetchone()["c"]

        # Oldest/newest message timestamps
        oldest = self.conn.execute("SELECT MIN(send_time_jst) as t FROM messages").fetchone()["t"]
        newest = self.conn.execute("SELECT MAX(send_time_jst) as t FROM messages").fetchone()["t"]

        # Synced channel count
        synced = self.conn.execute("SELECT COUNT(*) as c FROM sync_state").fetchone()["c"]

        return {
            "channels": channels,
            "users": users,
            "messages": msgs,
            "synced_channels": synced,
            "oldest_message": oldest or "N/A",
            "newest_message": newest or "N/A",
        }
