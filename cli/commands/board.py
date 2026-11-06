# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys

from cli._anima_tool import current_anima_dir, run_and_print
from core.messaging.board_fanout import fanout_board_mentions
from core.tooling.standalone import notify_server_message_sent


def _refuse_impersonation(anima_name: str, requested_name: str) -> None:
    print(
        f"Error: anima '{anima_name}' cannot act as '{requested_name}'",
        file=sys.stderr,
    )
    raise SystemExit(1)


# ── Board Read ────────────────────────────────────────────


def cmd_board_read(args: argparse.Namespace) -> None:
    """Read recent messages from a shared channel."""
    if current_anima_dir() is not None:
        run_and_print(
            "read_channel",
            {
                "channel": args.channel,
                "limit": getattr(args, "limit", 20),
                "human_only": getattr(args, "human_only", False),
            },
        )
        return

    from core.infra.runtime_init import ensure_runtime_dir
    from core.messaging.messenger import Messenger
    from core.paths import get_shared_dir

    ensure_runtime_dir()
    messenger = Messenger(get_shared_dir(), "cli")
    messages = messenger.read_channel(
        args.channel,
        limit=getattr(args, "limit", 20),
        human_only=getattr(args, "human_only", False),
    )
    if not messages:
        print(f"No messages in #{args.channel}")
        return
    print(json.dumps(messages, ensure_ascii=False, indent=2))


# ── Board Post ────────────────────────────────────────────


def cmd_board_post(args: argparse.Namespace) -> None:
    """Post a message to a shared channel."""
    anima_dir = current_anima_dir()
    if anima_dir is not None:
        if args.from_anima != anima_dir.name:
            _refuse_impersonation(anima_dir.name, args.from_anima)
        run_and_print(
            "post_channel",
            {
                "channel": args.channel,
                "text": args.text,
            },
        )
        return

    from core.exceptions import ChannelAccessDeniedError, ChannelNotFoundError
    from core.infra.runtime_init import ensure_runtime_dir
    from core.messaging.messenger import Messenger
    from core.paths import get_shared_dir

    ensure_runtime_dir()
    messenger = Messenger(get_shared_dir(), args.from_anima)
    try:
        messenger.post_channel(args.channel, args.text)
    except (ChannelNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except ChannelAccessDeniedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Posted to #{args.channel}")

    # Operator posts use the same ACL/company-filtered mention fan-out as the
    # ToolHandler path; there is only one implementation of this behavior.
    fanout_board_mentions(messenger, args.from_anima, args.channel, args.text)
    _notify_server_board_posted(args.from_anima, args.channel, args.text)


# ── Board DM History ──────────────────────────────────────


def cmd_board_dm_history(args: argparse.Namespace) -> None:
    """Read DM history with a specific peer."""
    anima_dir = current_anima_dir()
    if anima_dir is not None:
        if args.from_anima != anima_dir.name:
            _refuse_impersonation(anima_dir.name, args.from_anima)
        run_and_print(
            "read_dm_history",
            {
                "peer": args.peer,
                "limit": getattr(args, "limit", 20),
                "direction": getattr(args, "direction", "both"),
                "hours": getattr(args, "hours", None),
                "keyword": getattr(args, "keyword", None),
            },
        )
        return

    from core.infra.runtime_init import ensure_runtime_dir
    from core.messaging.messenger import Messenger
    from core.paths import get_shared_dir

    ensure_runtime_dir()
    messenger = Messenger(get_shared_dir(), args.from_anima)
    messages = messenger.read_dm_history(
        args.peer,
        limit=getattr(args, "limit", 20),
        direction=getattr(args, "direction", "both"),
        hours=getattr(args, "hours", None),
        keyword=getattr(args, "keyword", None),
    )
    if not messages:
        print(f"No DM history with {args.peer}")
        return
    print(json.dumps(messages, ensure_ascii=False, indent=2))


def _notify_server_board_posted(
    from_anima: str,
    channel: str,
    text: str,
) -> None:
    """Keep operator Board posts on the shared server-notification path."""
    notify_server_message_sent(from_anima, f"#channel:{channel}", text)
