# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""SQLite message cache for Discord (offline search, sync state)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from core.paths import get_data_dir
from core.tools._cache import BaseMessageCache

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────

JST = timezone(timedelta(hours=9))

_DISCORD_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS guilds (
    id TEXT PRIMARY KEY,
    name TEXT,
    icon TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    guild_id TEXT,
    name TEXT,
    type INTEGER,
    position INTEGER,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT,
    display_name TEXT,
    avatar TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    channel_id TEXT,
    author_id TEXT,
    content TEXT,
    timestamp TEXT,
    edited_timestamp TEXT,
    reference_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_channel_timestamp
    ON messages(channel_id, timestamp);

CREATE TABLE IF NOT EXISTS sync_state (
    channel_id TEXT PRIMARY KEY,
    last_message_id TEXT,
    synced_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_content ON messages(content);
"""


# ── MessageCache ───────────────────────────────────────────


class MessageCache(BaseMessageCache):
    """SQLite-backed cache for Discord messages."""

    def __init__(self) -> None:
        db_path = get_data_dir() / "cache" / "discord" / "messages.db"
        super().__init__(db_path, _DISCORD_SCHEMA_SQL)

    def upsert_guild(self, guild: dict) -> None:
        """Insert or replace a guild row."""
        gid = str(guild.get("id", ""))
        if not gid:
            return
        name = str(guild.get("name", ""))
        icon = guild.get("icon")
        icon_s = str(icon) if icon is not None else ""
        now = datetime.now(JST).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO guilds (id, name, icon, updated_at)
               VALUES (?, ?, ?, ?)""",
            (gid, name, icon_s, now),
        )
        self.conn.commit()

    def upsert_channel(self, channel: dict) -> None:
        """Insert or replace a channel row."""
        cid = str(channel.get("id", ""))
        if not cid:
            return
        guild_id = str(channel.get("guild_id") or "")
        name = str(channel.get("name", ""))
        ctype = int(channel.get("type", 0))
        pos = int(channel.get("position", 0))
        now = datetime.now(JST).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO channels
               (id, guild_id, name, type, position, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cid, guild_id, name, ctype, pos, now),
        )
        self.conn.commit()

    def upsert_messages(self, channel_id: str, messages: list[dict]) -> None:
        """Insert or replace message rows for a channel."""
        for m in messages:
            mid = str(m.get("id", ""))
            if not mid:
                continue
            author = m.get("author") or {}
            author_id = str(author.get("id", "")) if isinstance(author, dict) else ""
            content = str(m.get("content", ""))
            ts = str(m.get("timestamp", ""))
            edited = m.get("edited_timestamp")
            edited_s = str(edited) if edited is not None else ""
            ref = m.get("message_reference") or {}
            ref_id = ""
            if isinstance(ref, dict):
                ref_id = str(ref.get("message_id", "") or "")

            self.conn.execute(
                """INSERT OR REPLACE INTO messages
                   (id, channel_id, author_id, content, timestamp,
                    edited_timestamp, reference_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (mid, channel_id, author_id, content, ts, edited_s, ref_id or None),
            )

            if isinstance(author, dict) and author_id:
                uname = str(author.get("username", ""))
                global_name = author.get("global_name")
                display = str(global_name) if global_name else uname
                avatar = author.get("avatar")
                avatar_s = str(avatar) if avatar is not None else ""
                now_u = datetime.now(JST).isoformat()
                self.conn.execute(
                    """INSERT OR REPLACE INTO users
                       (id, username, display_name, avatar, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (author_id, uname, display, avatar_s, now_u),
                )
        self.conn.commit()

    def get_recent(self, channel_id: str, limit: int = 50) -> list[dict]:
        """Return the most recent cached messages for a channel (newest first)."""
        rows = self.conn.execute(
            """SELECT * FROM messages
               WHERE channel_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (channel_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search(
        self,
        keyword: str,
        *,
        channel_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search cached messages by substring match on ``content``."""
        q = """SELECT * FROM messages WHERE content LIKE ?"""
        params: list = [f"%{keyword}%"]
        if channel_id:
            q += " AND channel_id = ?"
            params.append(channel_id)
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def update_sync_state(self, channel_id: str, last_message_id: str = "") -> None:
        """Record last synced message id and sync time for a channel."""
        now = datetime.now(JST).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO sync_state
               (channel_id, last_message_id, synced_at)
               VALUES (?, ?, ?)""",
            (channel_id, last_message_id, now),
        )
        self.conn.commit()

    # ── User name helpers (for CLI display) ────────────────

    def get_user_name_cache(self) -> dict[str, str]:
        """Return a mapping of user_id → display_name from the users table."""
        rows = self.conn.execute("SELECT id, display_name FROM users").fetchall()
        return {r["id"]: r["display_name"] for r in rows if r["display_name"]}

    def get_user_name(self, user_id: str) -> str:
        """Return display name for a single user_id, or empty string."""
        row = self.conn.execute("SELECT display_name FROM users WHERE id = ?", (user_id,)).fetchone()
        return row["display_name"] if row and row["display_name"] else ""
