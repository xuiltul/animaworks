# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Standalone CLI entry point for Discord tools."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core.tools._base import ToolConfigError
from core.tools._comm_cli import print_json, run_cli_safely
from core.tools._discord_cache import MessageCache
from core.tools._discord_client import DiscordAPIError, DiscordClient
from core.tools._discord_markdown import clean_discord_markup


def get_cli_guide() -> str:
    """Return CLI usage guide for Discord tools."""
    return """\
### Discord
```bash
animaworks-tool discord guilds -j
animaworks-tool discord channels GUILD_ID -j
animaworks-tool discord messages CHANNEL_ID [-n 20] -j
animaworks-tool discord send CHANNEL_ID "メッセージ本文" [--reply-to MSG_ID]
animaworks-tool discord search "キーワード" [-c CHANNEL_ID] [-n 50] -j
```"""


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
        return token
    token = _lookup_shared_credentials(per_anima_key)
    if token:
        return token
    return None


def _resolve_cli_token() -> str | None:
    """Resolve per-Anima Discord bot token for CLI invocations.

    Reads ``ANIMAWORKS_ANIMA_DIR`` env var set by the framework when
    spawning Anima subprocesses (Mode S / Mode A).
    """
    return _resolve_per_anima_token(os.environ.get("ANIMAWORKS_ANIMA_DIR"))


def _enrich_message_authors(msgs: list[dict]) -> None:
    """Normalize Discord API message dicts for caching and display."""
    for m in msgs:
        author = m.get("author")
        if isinstance(author, dict):
            uid = author.get("id", "")
            if uid and not m.get("user_id"):
                m["user_id"] = uid
            uname = author.get("global_name") or author.get("username", "")
            if uname and not m.get("user_name"):
                m["user_name"] = uname


def _channel_type_label(type_id: int) -> str:
    """Map Discord channel type integer to a short label."""
    labels: dict[int, str] = {
        0: "text",
        1: "dm",
        2: "voice",
        3: "group_dm",
        4: "category",
        5: "news",
        10: "news_thread",
        11: "public_thread",
        12: "private_thread",
        13: "stage",
        15: "forum",
        16: "media",
    }
    return labels.get(int(type_id), str(type_id))


def _run_cli_command(client: DiscordClient, args: argparse.Namespace) -> None:
    """Dispatch CLI subcommands."""
    # ── guilds ───────────────────────────────────────────────
    if args.command == "guilds":
        guilds = client.guilds()
        if getattr(args, "json", False):
            print_json(guilds)
            return
        print(f"{'ID':>20}  {'Members':>8}  {'Name'}")
        print("-" * 70)
        for g in guilds:
            gid = g.get("id", "")
            name = g.get("name", "")
            members = g.get("approximate_member_count", g.get("member_count", ""))
            print(f"{gid:>20}  {str(members):>8}  {name}")
        print(f"\nTotal: {len(guilds)} guild(s)")
        return

    # ── channels ─────────────────────────────────────────────
    if args.command == "channels":
        raw = client.channels(args.guild_id)
        text_channels = [c for c in raw if int(c.get("type", -1)) in (0, 5, 15, 16)]
        if getattr(args, "json", False):
            print_json(text_channels)
            return
        print(f"{'ID':>20}  {'Type':12}  {'Name'}")
        print("-" * 70)
        for ch in text_channels:
            cid = ch.get("id", "")
            name = ch.get("name", "")
            ctype = _channel_type_label(int(ch.get("type", 0)))
            print(f"{cid:>20}  {ctype:12}  {name}")
        print(f"\nTotal: {len(text_channels)} text-compatible channel(s)")
        return

    # ── send ─────────────────────────────────────────────────
    if args.command == "send":
        body = " ".join(args.message)
        reply_to = getattr(args, "reply_to", None)
        response = client.send_message(
            args.channel_id,
            body,
            reply_to=reply_to,
        )
        mid = response.get("id", "") if isinstance(response, dict) else ""
        ch = response.get("channel_id", args.channel_id) if isinstance(response, dict) else args.channel_id
        print(f"Sent (channel: {ch}, id: {mid})")
        return

    # ── messages ─────────────────────────────────────────────
    if args.command == "messages":
        channel_id = args.channel_id
        cache = MessageCache()
        try:
            print("Fetching messages...", file=sys.stderr, end=" ", flush=True)
            msgs = client.channel_history(channel_id, limit=args.num)
            print(f"{len(msgs)} fetched", file=sys.stderr)

            _enrich_message_authors(msgs)
            if msgs:
                cache.upsert_messages(channel_id, msgs)
                cache.update_sync_state(channel_id)

            if getattr(args, "json", False):
                cached = cache.get_recent(channel_id, limit=args.num)
                out: list[dict] = []
                for m in cached:
                    text_raw = m.get("text", "")
                    out.append(
                        {
                            "channel_id": m.get("channel_id", channel_id),
                            "id": m.get("ts", m.get("id", "")),
                            "user_id": m.get("user_id", ""),
                            "user_name": m.get("user_name", ""),
                            "text": clean_discord_markup(text_raw),
                            "send_time_jst": m.get("send_time_jst", ""),
                        }
                    )
                print_json(out)
                return

            cached = cache.get_recent(channel_id, limit=args.num)
            user_name_map = cache.get_user_name_cache()

            if cached:
                for m in reversed(cached):
                    ts = m.get("send_time_jst", "")
                    user_name = m.get("user_name", "")
                    if not user_name and m.get("user_id"):
                        user_name = cache.get_user_name(m["user_id"])
                    text = clean_discord_markup(m.get("text", ""), cache=user_name_map)
                    print(f"{ts} {user_name}")
                    for line in text.strip().split("\n"):
                        print(f"  {line}")
                    print()
            else:
                print("No messages found.")
        finally:
            cache.close()
        return

    # ── search ────────────────────────────────────────────────
    if args.command == "search":
        cache = MessageCache()
        try:
            keyword = " ".join(args.keyword)
            channel_id = getattr(args, "channel", None)
            results = cache.search(keyword, channel_id=channel_id, limit=args.num)
            user_name_map = cache.get_user_name_cache()

            if getattr(args, "json", False):
                payload = []
                for m in results:
                    text_clean = clean_discord_markup(m.get("text", ""), cache=user_name_map)
                    payload.append(
                        {
                            **m,
                            "text": text_clean,
                        }
                    )
                print_json(payload)
                return

            if not results:
                print(f"No messages matching '{keyword}'.")
            else:
                for m in results:
                    m["text"] = clean_discord_markup(m.get("text", ""), cache=user_name_map)
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
        return


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI entry point for the Discord tool."""

    def _run() -> None:
        parser = argparse.ArgumentParser(
            prog="animaworks-discord",
            description="Discord CLI (AnimaWorks integration)",
        )
        sub = parser.add_subparsers(dest="command", help="Command")

        p = sub.add_parser("guilds", help="List guilds the bot can see")
        p.add_argument("-j", "--json", action="store_true", help="Output as JSON")

        p = sub.add_parser("channels", help="List text channels in a guild")
        p.add_argument("guild_id", help="Discord guild (server) ID")
        p.add_argument("-j", "--json", action="store_true", help="Output as JSON")

        p = sub.add_parser("send", help="Send a message")
        p.add_argument("channel_id", help="Discord channel ID")
        p.add_argument("message", nargs="+", help="Message body")
        p.add_argument("--reply-to", dest="reply_to", help="Message ID to reply to")

        p = sub.add_parser("messages", help="Get recent messages")
        p.add_argument("channel_id", help="Discord channel ID")
        p.add_argument("-n", "--num", type=int, default=20, help="Number of messages (default 20)")
        p.add_argument("-j", "--json", action="store_true", help="Output as JSON")

        p = sub.add_parser("search", help="Search cached messages")
        p.add_argument("keyword", nargs="+", help="Search keyword")
        p.add_argument("-c", "--channel", help="Filter by channel ID")
        p.add_argument("-n", "--num", type=int, default=50, help="Max results (default 50)")
        p.add_argument("-j", "--json", action="store_true", help="Output as JSON")

        args = parser.parse_args(argv)

        if not args.command:
            parser.print_help()
            sys.exit(0)

        token = _resolve_cli_token()
        try:
            client = DiscordClient(token=token)
        except ToolConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        _run_cli_command(client, args)

    run_cli_safely(_run, api_error_type=DiscordAPIError)
