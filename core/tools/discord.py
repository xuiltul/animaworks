# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Discord integration for AnimaWorks.

Provides:
- DiscordClient: Discord REST API v10 wrapper with rate-limit retry
- MessageCache: SQLite cache for offline search
- get_tool_schemas(): Anthropic tool_use schemas
- cli_main(): standalone CLI entry point
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.tools._base import logger

# Re-exports (also used by :func:`dispatch`)
from core.tools._discord_cache import MessageCache
from core.tools._discord_cli import cli_main, get_cli_guide  # noqa: F401
from core.tools._discord_client import DiscordClient
from core.tools._discord_markdown import (  # noqa: F401
    clean_discord_markup,
    md_to_discord,
    truncate,
)

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "guilds": {"expected_seconds": 10, "background_eligible": False},
    "channels": {"expected_seconds": 10, "background_eligible": False},
    "messages": {"expected_seconds": 30, "background_eligible": False},
    "send": {"expected_seconds": 10, "background_eligible": False},
    "search": {"expected_seconds": 30, "background_eligible": False},
    # gated: requires explicit "discord_channel_post: yes" in permissions.md.
    "channel_post": {"expected_seconds": 10, "background_eligible": False, "gated": True},
}


# ── Token Resolution ───────────────────────────────────────


def _resolve_per_anima_token(anima_dir: str | Path | None) -> str | None:
    """Resolve per-Anima Discord bot token from anima_dir path.

    Uses ``DISCORD_BOT_TOKEN__<anima_name>`` from vault.json / shared/credentials.json.
    Returns None to fall back to the shared token.
    """
    if not anima_dir:
        return None
    from core.tools._base import _lookup_shared_credentials, _lookup_vault_credential

    anima_name = Path(anima_dir).name
    per_anima_key = f"DISCORD_BOT_TOKEN__{anima_name}"
    token = _lookup_vault_credential(per_anima_key)
    if token:
        logger.debug("Using per-Anima Discord token for '%s'", anima_name)
        return token
    token = _lookup_shared_credentials(per_anima_key)
    if token:
        logger.debug("Using per-Anima Discord token for '%s'", anima_name)
        return token
    return None


def _resolve_discord_token(args: dict[str, Any]) -> str | None:
    """Resolve per-Anima Discord bot token from tool dispatch args."""
    return _resolve_per_anima_token(args.get("anima_dir"))


def _resolve_discord_identity(args: dict[str, Any]) -> tuple[str, str]:
    """Resolve Anima display name and icon URL for Discord-related context.

    See :func:`core.tools._anima_icon_url.resolve_anima_icon_identity`.
    """
    from core.tools._anima_icon_url import resolve_anima_icon_identity

    anima_dir = args.get("anima_dir")
    if not anima_dir:
        return ("", "")
    return resolve_anima_icon_identity(Path(anima_dir).name, channel_config=None)


# ── Tool Schemas ───────────────────────────────────────────


def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for Discord tools.

    ``discord_channel_post`` is a gated action: it requires
    ``discord_channel_post: yes`` in permissions.md.
    """
    return [
        {
            "name": "discord_channel_post",
            "description": (
                "Post a message to a Discord text channel via Bot Token API. "
                "Returns the message ID for future reference."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text (Markdown supported, max 2000 chars)",
                    },
                },
                "required": ["channel_id", "text"],
            },
        },
    ]


# ── Dispatch ───────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    if name == "discord_send":
        client = DiscordClient(token=_resolve_discord_token(args))
        try:
            return client.send_message(
                args["channel_id"],
                args["message"],
                reply_to=args.get("reply_to"),
            )
        finally:
            client.close()
    if name == "discord_messages":
        client = DiscordClient(token=_resolve_discord_token(args))
        cache = MessageCache()
        try:
            channel_id = args["channel_id"]
            limit = int(args.get("limit", 20))
            msgs = client.channel_history(channel_id, limit=limit)
            if msgs:
                for m in msgs:
                    author = m.get("author")
                    if isinstance(author, dict):
                        uid = author.get("id", "")
                        if uid and not m.get("user_id"):
                            m["user_id"] = uid
                        uname = author.get("global_name") or author.get("username", "")
                        if uname and not m.get("user_name"):
                            m["user_name"] = uname
                cache.upsert_messages(channel_id, msgs)
                cache.update_sync_state(channel_id)
            return cache.get_recent(channel_id, limit=limit)
        finally:
            client.close()
            cache.close()
    if name == "discord_search":
        cache = MessageCache()
        try:
            channel_id = args.get("channel_id")
            return cache.search(
                args["keyword"],
                channel_id=channel_id,
                limit=int(args.get("limit", 50)),
            )
        finally:
            cache.close()
    if name == "discord_guilds":
        client = DiscordClient(token=_resolve_discord_token(args))
        try:
            return client.guilds()
        finally:
            client.close()
    if name == "discord_channels":
        client = DiscordClient(token=_resolve_discord_token(args))
        try:
            return client.channels(args["guild_id"])
        finally:
            client.close()
    if name == "discord_react":
        client = DiscordClient(token=_resolve_discord_token(args))
        try:
            return client.add_reaction(
                args["channel_id"],
                args["message_id"],
                args["emoji"],
            )
        finally:
            client.close()
    if name == "discord_channel_post":
        client = DiscordClient(token=_resolve_discord_token(args))
        try:
            discord_text = md_to_discord(args["text"])
            resp = client.send_message(
                args["channel_id"],
                discord_text,
            )
            mid = resp.get("id", "") if isinstance(resp, dict) else ""
            return {"status": "ok", "channel_id": args["channel_id"], "message_id": mid}
        finally:
            client.close()
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    cli_main()
