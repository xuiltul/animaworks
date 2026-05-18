# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""SQLite message cache for Chatwork offline search and unreplied detection."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from core.tools._cache import BaseMessageCache
from core.tools._chatwork_client import JST

# ── Constants ──────────────────────────────────────────────

# Cache directory for Chatwork message cache.
# Can be overridden via ANIMAWORKS_CHATWORK_CACHE_DIR environment variable.
# This allows TaskExec/Codex sandbox environments to redirect cache writes
# to a writable location (e.g. /tmp/animaworks-cache/chatwork).
_DEFAULT_CACHE_DIR = Path.home() / ".animaworks" / "cache" / "chatwork"
DEFAULT_CACHE_DIR = Path(os.environ.get("ANIMAWORKS_CHATWORK_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))

_CHATWORK_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS rooms (
    room_id TEXT PRIMARY KEY,
    name TEXT,
    type TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT,
    room_id TEXT,
    account_id TEXT,
    account_name TEXT,
    body TEXT,
    send_time INTEGER,
    send_time_jst TEXT,
    PRIMARY KEY (room_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_body ON messages(body);
CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(send_time);
CREATE TABLE IF NOT EXISTS sync_state (
    room_id TEXT PRIMARY KEY,
    last_synced TEXT
);
"""


# ── Helpers ──────────────────────────────────────────────────


def _format_timestamp(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=JST).strftime("%Y-%m-%d %H:%M")


# ── MessageCache ─────────────────────────────────────────────


class MessageCache(BaseMessageCache):
    """SQLite-backed cache for Chatwork messages, enabling offline search and
    unreplied-mention detection."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = DEFAULT_CACHE_DIR / "messages.db"
        super().__init__(db_path, _CHATWORK_SCHEMA_SQL)

    def upsert_room(self, room: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO rooms (room_id, name, type, updated_at) VALUES (?,?,?,?)",
            (str(room["room_id"]), room["name"], room.get("type", ""), datetime.now(JST).isoformat()),
        )
        self.conn.commit()

    def upsert_messages(self, room_id: str, messages: list[dict]):
        for m in messages:
            send_time = m.get("send_time", 0)
            dt = datetime.fromtimestamp(send_time, tz=JST)
            account = m.get("account", {})
            self.conn.execute(
                """INSERT OR REPLACE INTO messages
                   (message_id, room_id, account_id, account_name, body, send_time, send_time_jst)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    str(m["message_id"]),
                    str(room_id),
                    str(account.get("account_id", "")),
                    account.get("name", ""),
                    m.get("body", ""),
                    send_time,
                    dt.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        self.conn.commit()

    def update_sync_state(self, room_id: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO sync_state (room_id, last_synced) VALUES (?,?)",
            (room_id, datetime.now(JST).isoformat()),
        )
        self.conn.commit()

    def search(
        self,
        keyword: str,
        room_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        query = """
            SELECT m.*, r.name as room_name
            FROM messages m
            LEFT JOIN rooms r ON m.room_id = r.room_id
            WHERE m.body LIKE ?
        """
        params: list = [f"%{keyword}%"]
        if room_id:
            query += " AND m.room_id = ?"
            params.append(room_id)
        query += " ORDER BY m.send_time DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, room_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT m.*, r.name as room_name
               FROM messages m
               LEFT JOIN rooms r ON m.room_id = r.room_id
               WHERE m.room_id = ?
               ORDER BY m.send_time DESC LIMIT ?""",
            (room_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_room_name(self, room_id: str) -> str:
        row = self.conn.execute("SELECT name FROM rooms WHERE room_id = ?", (room_id,)).fetchone()
        return row["name"] if row else room_id

    def _get_personal_room_ids(self, config: dict) -> set[str]:
        """Return room IDs for DMs + watch_rooms from config."""
        personal: set[str] = set()
        unreplied_cfg = config.get("unreplied", {})

        # type=direct rooms from DB
        if unreplied_cfg.get("include_direct_messages", True):
            rows = self.conn.execute("SELECT room_id FROM rooms WHERE type = 'direct'").fetchall()
            personal.update(str(r["room_id"]) for r in rows)

        # Explicitly watched rooms
        for wr in unreplied_cfg.get("watch_rooms", []):
            personal.add(str(wr["room_id"]))

        return personal

    def find_mentions(
        self,
        my_account_id: str,
        exclude_toall: bool = True,
        limit: int = 200,
        config: dict | None = None,
    ) -> list[dict]:
        """Find messages addressed to me.

        - Group rooms: messages containing [To:my_id]
        - DM / watch_rooms: all messages from other people (no [To:] needed)
        """
        if config is None:
            config = {}
        personal_rooms = self._get_personal_room_ids(config)

        results: list[dict] = []
        seen: set[tuple[str, str]] = set()

        # 1) Normal [To:my_id] mentions (all rooms)
        to_tag = f"%[To:{my_account_id}]%"
        query = """
            SELECT m.*, r.name as room_name
            FROM messages m
            LEFT JOIN rooms r ON m.room_id = r.room_id
            WHERE m.body LIKE ?
              AND m.account_id != ?
        """
        params: list = [to_tag, my_account_id]
        if exclude_toall:
            query += " AND m.body NOT LIKE '%[toall]%'"
        query += " ORDER BY m.send_time DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        for r in rows:
            d = dict(r)
            key = (d["room_id"], d["message_id"])
            if key not in seen:
                seen.add(key)
                results.append(d)

        # 2) DM / watch_rooms: all messages from others
        if personal_rooms:
            placeholders = ",".join("?" for _ in personal_rooms)
            query2 = f"""
                SELECT m.*, r.name as room_name
                FROM messages m
                LEFT JOIN rooms r ON m.room_id = r.room_id
                WHERE m.room_id IN ({placeholders})
                  AND m.account_id != ?
                ORDER BY m.send_time DESC LIMIT ?
            """
            params2 = list(personal_rooms) + [my_account_id, limit]
            rows2 = self.conn.execute(query2, params2).fetchall()
            for r in rows2:
                d = dict(r)
                key = (d["room_id"], d["message_id"])
                if key not in seen:
                    seen.add(key)
                    results.append(d)

        # Sort by time descending
        results.sort(key=lambda x: x.get("send_time", 0), reverse=True)
        return results[:limit]

    def find_unreplied(
        self,
        my_account_id: str,
        exclude_toall: bool = True,
        limit: int = 200,
        config: dict | None = None,
    ) -> list[dict]:
        """Find messages addressed to me that I haven't replied to."""
        mentions = self.find_mentions(my_account_id, exclude_toall, limit, config=config)
        unreplied = []
        for m in mentions:
            # Check if I have sent any message in this room after this one
            row = self.conn.execute(
                """SELECT COUNT(*) as c FROM messages
                   WHERE room_id = ? AND account_id = ? AND send_time > ?""",
                (m["room_id"], my_account_id, m["send_time"]),
            ).fetchone()
            if row["c"] == 0:
                unreplied.append(m)
        return unreplied

    def get_stats(self) -> dict:
        """Return cache statistics."""
        rooms = self.conn.execute("SELECT COUNT(*) as c FROM rooms").fetchone()["c"]
        msgs = self.conn.execute("SELECT COUNT(*) as c FROM messages").fetchone()["c"]
        return {"rooms": rooms, "messages": msgs}
