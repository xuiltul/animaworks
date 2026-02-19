# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Chatwork integration for AnimaWorks.

Provides:
- ChatworkClient: HTTP API client for Chatwork
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
    "rooms":    {"expected_seconds": 10, "background_eligible": False},
    "messages": {"expected_seconds": 30, "background_eligible": False},
    "send":     {"expected_seconds": 10, "background_eligible": False},
    "search":   {"expected_seconds": 30, "background_eligible": False},
    "unreplied": {"expected_seconds": 60, "background_eligible": False},
}

requests = None

def _require_requests():
    global requests
    if requests is None:
        try:
            import requests as _req
            requests = _req
        except ImportError:
            raise ImportError(
                "chatwork tool requires 'requests'. Install with: pip install animaworks[communication]"
            )
    return requests

# ============================================================
# Constants
# ============================================================

JST = timezone(timedelta(hours=9))
BASE_URL = "https://api.chatwork.com/v2"
RATE_LIMIT_RETRY_MAX = 5
RATE_LIMIT_WAIT_DEFAULT = 60

DEFAULT_CACHE_DIR = Path.home() / ".animaworks" / "cache" / "chatwork"


# ============================================================
# Text utility
# ============================================================

def clean_chatwork_tags(text: str) -> str:
    """Remove Chatwork special tags to make text more readable."""
    text = re.sub(r"\[To:\d+\][^\n]*\n?", "", text)
    text = re.sub(r"\[toall\]", "", text)
    text = re.sub(r"\[info\].*?\[/info\]", "[info block]", text, flags=re.DOTALL)
    text = re.sub(r"\[/?[a-z_]+\]", "", text)
    return text.strip()


def _format_timestamp(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=JST).strftime("%Y-%m-%d %H:%M")


# ============================================================
# Chatwork API Client
# ============================================================

class ChatworkClient:
    """HTTP client for the Chatwork v2 API with rate-limit retry."""

    def __init__(self, api_token: str | None = None):
        _require_requests()
        if api_token is None:
            api_token = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            "X-ChatWorkToken": api_token,
            "Accept": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs) -> dict | list | None:
        """Send an HTTP request with rate-limit retry."""
        url = f"{BASE_URL}{path}"

        class _RateLimitError(Exception):
            """Raised when Chatwork returns HTTP 429."""

            def __init__(self, retry_after: int) -> None:
                self.retry_after = retry_after
                super().__init__(f"Rate limited, retry after {retry_after}s")

        def _do_request() -> dict | list | None:
            resp = self.session.request(method, url, **kwargs)
            if resp.status_code == 429:
                retry_after = int(
                    resp.headers.get("Retry-After", RATE_LIMIT_WAIT_DEFAULT)
                )
                raise _RateLimitError(retry_after)
            if resp.status_code == 204:
                return None
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return None
            return resp.json()

        def _get_retry_after(exc: Exception) -> float | None:
            if isinstance(exc, _RateLimitError):
                return float(exc.retry_after)
            return None

        return retry_on_rate_limit(
            _do_request,
            max_retries=RATE_LIMIT_RETRY_MAX,
            default_wait=RATE_LIMIT_WAIT_DEFAULT,
            get_retry_after=_get_retry_after,
            retry_on=(_RateLimitError,),
        )

    async def _arequest(
        self, method: str, path: str, **kwargs
    ) -> dict | list | None:
        """Async wrapper around :meth:`_request` using a thread-pool executor."""
        return await run_sync(self._request, method, path, **kwargs)

    def get(self, path: str, params: dict | None = None):
        return self._request("GET", path, params=params)

    def post(self, path: str, data: dict | None = None):
        return self._request("POST", path, data=data)

    def put(self, path: str, data: dict | None = None):
        return self._request("PUT", path, data=data)

    def delete(self, path: str):
        return self._request("DELETE", path)

    # --- High-level API methods ---

    def me(self) -> dict:
        return self.get("/me")

    def rooms(self) -> list[dict]:
        return self.get("/rooms")

    def room_members(self, room_id: str) -> list[dict]:
        return self.get(f"/rooms/{room_id}/members")

    def contacts(self) -> list[dict]:
        return self.get("/contacts")

    def get_messages(self, room_id: str, force: bool = False) -> list[dict] | None:
        """Get messages. force=True to include already-read messages."""
        return self.get(
            f"/rooms/{room_id}/messages",
            params={"force": 1 if force else 0},
        )

    def post_message(self, room_id: str, body: str) -> dict:
        if len(body) > 10000:
            raise ValueError(
                f"Message exceeds 10,000 characters ({len(body)} chars)"
            )
        return self.post(f"/rooms/{room_id}/messages", data={"body": body})

    def my_tasks(self, status: str = "open") -> list[dict]:
        """List my tasks. status: open / done"""
        return self.get("/my/tasks", params={"status": status}) or []

    def room_tasks(self, room_id: str, status: str = "open") -> list[dict]:
        """List tasks in a room."""
        return self.get(f"/rooms/{room_id}/tasks", params={"status": status}) or []

    def add_task(
        self,
        room_id: str,
        body: str,
        to_ids: str,
        limit: int = 0,
        limit_type: str = "time",
    ) -> dict:
        return self.post(f"/rooms/{room_id}/tasks", data={
            "body": body,
            "to_ids": to_ids,
            "limit": limit,
            "limit_type": limit_type,
        })

    def get_room_by_name(self, name: str) -> dict | None:
        """Search for a room by name (exact match preferred, then partial)."""
        rooms = self.rooms()
        # Exact match first
        for r in rooms:
            if r["name"] == name:
                return r
        # Partial match
        matches = [r for r in rooms if name.lower() in r["name"].lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                "Multiple rooms matched '%s': %s",
                name,
                ", ".join(f"[{r['room_id']}] {r['name']}" for r in matches),
            )
            return None
        return None

    def resolve_room_id(self, room: str) -> str:
        """Resolve a room name or ID to a numeric room_id string."""
        if room.isdigit():
            return room
        r = self.get_room_by_name(room)
        if r is None:
            raise ToolConfigError(f"Room '{room}' not found")
        return str(r["room_id"])


# ============================================================
# Local Message Cache (SQLite)
# ============================================================

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
            (str(room["room_id"]), room["name"], room.get("type", ""),
             datetime.now(JST).isoformat()),
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
                (str(m["message_id"]), str(room_id),
                 str(account.get("account_id", "")),
                 account.get("name", ""),
                 m.get("body", ""),
                 send_time,
                 dt.strftime("%Y-%m-%d %H:%M:%S")),
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
        row = self.conn.execute(
            "SELECT name FROM rooms WHERE room_id = ?", (room_id,)
        ).fetchone()
        return row["name"] if row else room_id

    def _get_personal_room_ids(self, config: dict) -> set[str]:
        """Return room IDs for DMs + watch_rooms from config."""
        personal: set[str] = set()
        unreplied_cfg = config.get("unreplied", {})

        # type=direct rooms from DB
        if unreplied_cfg.get("include_direct_messages", True):
            rows = self.conn.execute(
                "SELECT room_id FROM rooms WHERE type = 'direct'"
            ).fetchall()
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
        mentions = self.find_mentions(
            my_account_id, exclude_toall, limit, config=config
        )
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


# ============================================================
# Tool schemas (Anthropic tool_use format)
# ============================================================

def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for Chatwork tools."""
    return [
        {
            "name": "chatwork_send",
            "description": (
                "Send a message to a Chatwork room. "
                "The room can be specified by numeric ID or by name (partial match)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": "Room name or numeric room ID.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message body text to send.",
                    },
                },
                "required": ["room", "message"],
            },
        },
        {
            "name": "chatwork_messages",
            "description": (
                "Get recent messages from a Chatwork room. "
                "Fetches from API and caches locally, then returns from cache."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": "Room name or numeric room ID.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return (default 20).",
                        "default": 20,
                    },
                },
                "required": ["room"],
            },
        },
        {
            "name": "chatwork_search",
            "description": (
                "Search cached Chatwork messages by keyword. "
                "Searches the local SQLite cache. Run sync first for fresh results."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Search keyword.",
                    },
                    "room": {
                        "type": "string",
                        "description": "Optional room name or ID to filter results.",
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
            "name": "chatwork_unreplied",
            "description": (
                "Find Chatwork messages addressed to me that I haven't replied to. "
                "Uses the local cache. Includes DMs and explicitly watched rooms."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "include_toall": {
                        "type": "boolean",
                        "description": "Include @all mentions (default false).",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "chatwork_rooms",
            "description": "List Chatwork rooms accessible to the authenticated user.",
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
    """Return CLI usage guide for Chatwork tools."""
    return """\
### Chatwork
```bash
animaworks-tool chatwork rooms -j
animaworks-tool chatwork messages <ルーム名またはID> -j
animaworks-tool chatwork send <ルーム名またはID> "メッセージ本文"
animaworks-tool chatwork search "キーワード" -j
animaworks-tool chatwork unreplied -j
```"""


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI entry point for the Chatwork tool."""
    parser = argparse.ArgumentParser(
        prog="animaworks-chatwork",
        description="Chatwork CLI (AnimaWorks integration)",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # send
    p = sub.add_parser("send", help="Send a message")
    p.add_argument("room", help="Room name or ID")
    p.add_argument("message", nargs="+", help="Message body")

    # messages
    p = sub.add_parser("messages", help="Get recent messages")
    p.add_argument("room", help="Room name or ID")
    p.add_argument(
        "-n", "--num", type=int, default=20, help="Number of messages (default 20)"
    )

    # search
    p = sub.add_parser("search", help="Search cached messages")
    p.add_argument("keyword", nargs="+", help="Search keyword")
    p.add_argument("-r", "--room", help="Filter by room name or ID")
    p.add_argument(
        "-n", "--num", type=int, default=50, help="Max results (default 50)"
    )

    # unreplied
    p = sub.add_parser("unreplied", help="Show unreplied messages addressed to me")
    p.add_argument(
        "--include-toall", action="store_true", help="Include @all mentions"
    )
    p.add_argument("--json", action="store_true", help="Output as JSON")

    # rooms
    sub.add_parser("rooms", help="List accessible rooms")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        client = ChatworkClient()
    except ToolConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "send":
        room_id = client.resolve_room_id(args.room)
        message = " ".join(args.message)
        result = client.post_message(room_id, message)
        if result and "message_id" in result:
            print(f"Sent (message_id: {result['message_id']})")
        else:
            print(f"Result: {result}")

    elif args.command == "messages":
        room_id = client.resolve_room_id(args.room)
        cache = MessageCache()
        try:
            msgs = client.get_messages(room_id, force=True)
            if msgs:
                cache.upsert_messages(room_id, msgs)
                cache.update_sync_state(room_id)
                print(f"({len(msgs)} fetched & cached)\n", file=sys.stderr)
            cached = cache.get_recent(room_id, limit=args.num)
            if cached:
                for m in reversed(cached):
                    ts = m.get("send_time_jst", "")
                    name = m.get("account_name", "?")
                    room_name = m.get("room_name", "")
                    room_tag = f"[{room_name}] " if room_name else ""
                    body = m.get("body", "").strip()
                    print(f"{ts} {room_tag}{name}")
                    for line in body.split("\n"):
                        print(f"  {line}")
                    print()
            else:
                print("No messages found. Run 'sync' first.")
        finally:
            cache.close()

    elif args.command == "search":
        cache = MessageCache()
        try:
            keyword = " ".join(args.keyword)
            room_id = None
            if args.room:
                room_id = client.resolve_room_id(args.room)
            results = cache.search(keyword, room_id=room_id, limit=args.num)
            if not results:
                print(f"No messages matching '{keyword}'.")
            else:
                print(f"Results: {len(results)} (keyword: '{keyword}')\n")
                for m in reversed(results):
                    ts = m.get("send_time_jst", "")
                    name = m.get("account_name", "?")
                    room_name = m.get("room_name", "")
                    room_tag = f"[{room_name}] " if room_name else ""
                    body = m.get("body", "").strip()
                    print(f"{ts} {room_tag}{name}")
                    for line in body.split("\n"):
                        print(f"  {line}")
                    print()
        finally:
            cache.close()

    elif args.command == "unreplied":
        cache = MessageCache()
        try:
            my_info = client.me()
            my_id = str(my_info["account_id"])
            my_name = my_info["name"]
            unreplied = cache.find_unreplied(
                my_id, exclude_toall=(not args.include_toall)
            )
            if getattr(args, "json", False):
                output = []
                for m in unreplied:
                    output.append({
                        "message_id": m.get("message_id", ""),
                        "room_id": m.get("room_id", ""),
                        "room_name": m.get("room_name", m.get("room_id", "")),
                        "account_id": m.get("account_id", ""),
                        "account_name": m.get("account_name", ""),
                        "body": m.get("body", "").strip(),
                        "send_time": m.get("send_time", 0),
                        "send_time_jst": m.get("send_time_jst", ""),
                    })
                print(json.dumps(output, ensure_ascii=False, indent=2))
            elif not unreplied:
                print(f"No unreplied messages ({my_name} / ID:{my_id})")
            else:
                print(f"=== Unreplied: {len(unreplied)} ({my_name}) ===\n")
                for m in unreplied:
                    ts = m.get("send_time_jst", "")
                    name = m.get("account_name", "?")
                    room_name = m.get("room_name", m.get("room_id", ""))
                    body = m.get("body", "").strip()
                    body_clean = re.sub(
                        r"\[To:\d+\][^\n]*\n?", "", body
                    ).strip()
                    body_preview = body_clean.replace("\n", " ")[:120]
                    if len(body_clean) > 120:
                        body_preview += "..."
                    print(f"{ts} [{room_name}]")
                    print(f"  From: {name}")
                    print(f"  {body_preview}")
                    print()
        finally:
            cache.close()

    elif args.command == "rooms":
        rooms = client.rooms()
        rooms.sort(key=lambda r: r.get("last_update_time", 0), reverse=True)
        print(f"{'ID':>12}  {'Updated':19}  {'Name'}")
        print("-" * 70)
        for r in rooms:
            ts = _format_timestamp(r.get("last_update_time", 0))
            print(f"{r['room_id']:>12}  {ts}  {r['name']}")


# ── Dispatch ──────────────────────────────────────────

def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    if name == "chatwork_send":
        client = ChatworkClient()
        room_id = client.resolve_room_id(args["room"])
        return client.post_message(room_id, args["message"])
    if name == "chatwork_messages":
        client = ChatworkClient()
        room_id = client.resolve_room_id(args["room"])
        cache = MessageCache()
        try:
            msgs = client.get_messages(room_id, force=True)
            if msgs:
                cache.upsert_messages(room_id, msgs)
                cache.update_sync_state(room_id)
            return cache.get_recent(room_id, limit=args.get("limit", 20))
        finally:
            cache.close()
    if name == "chatwork_search":
        client = ChatworkClient()
        cache = MessageCache()
        try:
            room_id = None
            if args.get("room"):
                room_id = client.resolve_room_id(args["room"])
            return cache.search(
                args["keyword"], room_id=room_id, limit=args.get("limit", 50),
            )
        finally:
            cache.close()
    if name == "chatwork_unreplied":
        client = ChatworkClient()
        cache = MessageCache()
        try:
            my_info = client.me()
            my_id = str(my_info["account_id"])
            return cache.find_unreplied(
                my_id, exclude_toall=not args.get("include_toall", False),
            )
        finally:
            cache.close()
    if name == "chatwork_rooms":
        client = ChatworkClient()
        return client.rooms()
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    cli_main()