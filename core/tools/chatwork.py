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

from pathlib import Path
from typing import Any

from core.tools._base import ToolConfigError
from core.tools._chatwork_cache import (  # noqa: F401
    MessageCache,
    _format_timestamp,
    resolve_cache_db_path,
)
from core.tools._chatwork_cli import cli_main, get_cli_guide  # noqa: F401
from core.tools._chatwork_client import ChatworkClient  # noqa: F401
from core.tools._chatwork_identity import check_write_allowed, resolve_identity

# ── Re-exports (backward compatibility) ──────────────────────
from core.tools._chatwork_markdown import clean_chatwork_tags, md_to_chatwork  # noqa: F401

# ── Execution Profile ────────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "rooms": {"expected_seconds": 10, "background_eligible": False},
    "messages": {"expected_seconds": 30, "background_eligible": False},
    # gated: requires explicit "chatwork_send" in permissions allow list.
    "send": {"expected_seconds": 10, "background_eligible": False, "gated": True},
    # gated under chatwork_send (upload = message with attachment)
    "upload": {"expected_seconds": 30, "background_eligible": False, "gated": True, "gated_as": "send"},
    "search": {"expected_seconds": 30, "background_eligible": False},
    "unreplied": {"expected_seconds": 60, "background_eligible": False},
    "sync": {"expected_seconds": 60, "background_eligible": True},
    "me": {"expected_seconds": 5, "background_eligible": False},
    "members": {"expected_seconds": 10, "background_eligible": False},
    "contacts": {"expected_seconds": 10, "background_eligible": False},
    "task": {"expected_seconds": 10, "background_eligible": False},
    "mytasks": {"expected_seconds": 10, "background_eligible": False},
    "tasks": {"expected_seconds": 10, "background_eligible": False},
    "mentions": {"expected_seconds": 60, "background_eligible": False},
    "stats": {"expected_seconds": 5, "background_eligible": False},
    "files": {"expected_seconds": 10, "background_eligible": False},
    "download": {"expected_seconds": 60, "background_eligible": True},
    "delete": {"expected_seconds": 10, "background_eligible": False},
}


# ── Tool schemas ─────────────────────────────────────────────


def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for Chatwork tools."""
    return []


# ── Dispatch helpers ────────────────────────────────────────


def _load_chatwork_tool_config() -> dict:
    """Load tool-local config from cache dir / config.json."""
    import json

    from core.tools._chatwork_cache import DEFAULT_CACHE_DIR

    config_path = DEFAULT_CACHE_DIR / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _sync_rooms(client: ChatworkClient, cache: MessageCache, sync_limit: int = 30) -> dict:
    """Fetch room metadata + sync messages. Used by dispatch."""
    from core.tools._chatwork_cli import _sync_rooms as _do_sync

    return _do_sync(client, cache, sync_limit=sync_limit)


# ── Dispatch ─────────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    as_identity = args.get("as")
    anima_dir = args.get("anima_dir")
    identity = resolve_identity(as_identity, anima_dir=anima_dir)
    client = ChatworkClient(api_token=identity.token)

    if name == "chatwork_send":
        check_write_allowed(as_identity, anima_dir=anima_dir)
        room_id = client.resolve_room_id(args["room"])
        message = md_to_chatwork(args["message"])
        return client.post_message(room_id, message)
    if name == "chatwork_upload":
        check_write_allowed(as_identity, anima_dir=anima_dir)
        room_id = client.resolve_room_id(args["room"])
        message = md_to_chatwork(args["message"]) if args.get("message") else None
        return client.upload_file(room_id, Path(args["file"]).expanduser(), message)
    if name == "chatwork_messages":
        room_id = client.resolve_room_id(args["room"])
        cache = MessageCache(db_path=resolve_cache_db_path(client))
        try:
            msgs = client.get_messages(room_id, force=True)
            if msgs:
                cache.upsert_messages(room_id, msgs)
                cache.update_sync_state(room_id)
            return cache.get_recent(room_id, limit=args.get("limit", 20))
        finally:
            cache.close()
    if name == "chatwork_search":
        cache = MessageCache(db_path=resolve_cache_db_path(client))
        try:
            room_id = None
            if args.get("room"):
                room_id = client.resolve_room_id(args["room"])
            return cache.search(
                args["keyword"],
                room_id=room_id,
                limit=args.get("limit", 50),
            )
        finally:
            cache.close()
    if name == "chatwork_unreplied":
        cache = MessageCache(db_path=resolve_cache_db_path(client))
        try:
            my_info = client.me()
            my_id = str(my_info["account_id"])
            cli_config = _load_chatwork_tool_config()
            return cache.find_unreplied(
                my_id,
                exclude_toall=not args.get("include_toall", False),
                config=cli_config,
            )
        finally:
            cache.close()
    if name == "chatwork_delete":
        check_write_allowed(as_identity, anima_dir=anima_dir)
        room_id = client.resolve_room_id(args["room"])
        message_id = args["message_id"]
        # Ownership check: only allow deleting own messages
        my_info = client.me()
        my_account_id = str(my_info["account_id"])
        msg = client.get_message(room_id, message_id)
        msg_account_id = str(msg["account"]["account_id"])
        if msg_account_id != my_account_id:
            raise ToolConfigError(
                f"Cannot delete message {message_id}: it was posted by "
                f"'{msg['account']['name']}' (account_id={msg_account_id}), "
                f"not by you (account_id={my_account_id}). "
                f"You can only delete your own messages."
            )
        client.delete_message(room_id, message_id)
        return {"deleted": True, "message_id": message_id, "room_id": room_id}
    if name == "chatwork_rooms":
        return client.rooms()
    if name == "chatwork_sync":
        cache = MessageCache(db_path=resolve_cache_db_path(client))
        try:
            return _sync_rooms(client, cache, sync_limit=args.get("limit", 30))
        finally:
            cache.close()
    if name == "chatwork_mentions":
        cache = MessageCache(db_path=resolve_cache_db_path(client))
        try:
            my_info = client.me()
            my_id = str(my_info["account_id"])
            cli_config = _load_chatwork_tool_config()
            return cache.find_mentions(
                my_id,
                exclude_toall=not args.get("include_toall", False),
                limit=args.get("limit", 200),
                config=cli_config,
            )
        finally:
            cache.close()
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    cli_main()
