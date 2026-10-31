# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Standalone CLI entry point for Slack tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.integrations._base import ToolConfigError
from core.integrations._comm_cli import run_cli_safely
from core.integrations._slack_cache import MessageCache
from core.integrations._slack_client import SlackClient, _require_slack_sdk
from core.integrations._slack_markdown import clean_slack_markup, md_to_slack_mrkdwn
from core.platform.env import anima_dir_env


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


def _resolve_cli_token() -> str | None:
    """Resolve per-Anima Slack bot token for CLI invocations.

    Reads ``ANIMAWORKS_ANIMA_DIR`` env var set by the framework when
    spawning Anima subprocesses (Mode S / Mode A).
    """
    from core.channels.tokens import resolve_per_anima_token
    from core.credentials import resolve_env_style_credential

    return resolve_per_anima_token(
        "slack",
        anima_dir_env(),
        credential_lookup=resolve_env_style_credential,
    )


def _resolve_cli_identity() -> tuple[str, str]:
    """Resolve Anima display name and icon URL for CLI invocations.

    Uses ``ANIMAWORKS_ANIMA_DIR`` env var (set by framework for subprocesses).
    Returns (username, icon_url) — either may be empty string.
    """
    anima_dir = anima_dir_env()
    if not anima_dir:
        return ("", "")
    return _resolve_slack_identity({"anima_dir": anima_dir})


def _resolve_slack_identity(args: dict) -> tuple[str, str]:
    """Resolve Anima display name and icon URL for Slack messages."""
    anima_dir = args.get("anima_dir")
    if not anima_dir:
        return ("", "")

    from core.integrations._anima_icon_url import resolve_anima_icon_identity

    return resolve_anima_icon_identity(Path(anima_dir).name, channel_config=None)


def _run_cli_command(client: SlackClient, args) -> None:
    """Dispatch CLI subcommands."""
    if args.command == "send":
        channel_id = client.resolve_channel(args.channel)
        message = md_to_slack_mrkdwn(" ".join(args.message))
        thread_ts = getattr(args, "thread", None)
        username, icon_url = _resolve_cli_identity()
        response = client.post_message(
            channel_id,
            message,
            thread_ts=thread_ts,
            username=username,
            icon_url=icon_url,
        )
        ts = response.get("ts", "")
        channel = response.get("channel", channel_id)
        anima_dir = anima_dir_env()
        if anima_dir:
            from core.notification.reply_routing import remember_anima_slack_post

            remember_anima_slack_post(
                Path(anima_dir).name, channel_id, response, thread_ts=thread_ts, text=" ".join(args.message)
            )
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
                    text = clean_slack_markup(m.get("text", ""), cache=user_name_map)
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
                    m["text"] = clean_slack_markup(m.get("text", ""), cache=user_name_map)
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

            unreplied = cache.find_unreplied(my_user_id or "")
            user_name_map = cache.get_user_name_cache()

            if getattr(args, "json", False):
                output = []
                for m in unreplied:
                    text_clean = clean_slack_markup(m.get("text", ""), cache=user_name_map)
                    output.append(
                        {
                            "channel_id": m.get("channel_id", ""),
                            "channel_name": m.get("channel_name", m.get("channel_id", "")),
                            "ts": m.get("ts", ""),
                            "user_id": m.get("user_id", ""),
                            "user_name": m.get("user_name", ""),
                            "text": text_clean.strip(),
                            "ts_epoch": m.get("ts_epoch", 0),
                            "send_time_jst": m.get("send_time_jst", ""),
                            "thread_ts": m.get("thread_ts", ""),
                        }
                    )
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
                    channel_name = m.get("channel_name", m.get("channel_id", ""))
                    text = clean_slack_markup(m.get("text", ""), cache=user_name_map)
                    text_clean = text.strip()
                    text_preview = text_clean.replace("\n", " ")[:120]
                    if len(text_clean) > 120:
                        text_preview += "..."
                    thread_ts = m.get("thread_ts", "")
                    thread_info = f"  (thread: {thread_ts})" if thread_ts else ""
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
                if ch.get("is_member", False) or ch.get("is_im", False) or ch.get("is_mpim", False)
            ]
            member_channels.sort(key=lambda c: c.get("updated", 0), reverse=True)

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


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI entry point for the Slack tool."""
    _require_slack_sdk()  # Ensure SlackApiError is available for run_cli_safely
    from core.integrations._slack_client import SlackApiError

    def _run() -> None:
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
        p.add_argument("-n", "--num", type=int, default=20, help="Number of messages (default 20)")

        # search
        p = sub.add_parser("search", help="Search cached messages")
        p.add_argument("keyword", nargs="+", help="Search keyword")
        p.add_argument("-c", "--channel", help="Filter by channel name or ID")
        p.add_argument("-n", "--num", type=int, default=50, help="Max results (default 50)")

        # unreplied
        p = sub.add_parser("unreplied", help="Show unreplied messages addressed to me")
        p.add_argument("--json", action="store_true", help="Output as JSON")

        # channels
        sub.add_parser("channels", help="List joined channels")

        args = parser.parse_args(argv)

        if not args.command:
            parser.print_help()
            sys.exit(0)

        token = _resolve_cli_token()
        try:
            client = SlackClient(token=token)
        except ToolConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        _run_cli_command(client, args)

    run_cli_safely(_run, api_error_type=SlackApiError)  # type: ignore[arg-type]
