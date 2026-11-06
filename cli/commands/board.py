# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys

from cli._anima_tool import current_anima_dir, run_and_print


def _print_untrusted_result(tool_name: str, result: str) -> None:
    from core.trust import wrap_tool_result

    print(wrap_tool_result(tool_name, result))


def _run_anima_untrusted_tool(tool_name: str, display_name: str, tool_args: dict[str, object]) -> None:
    from cli._anima_tool import run_anima_tool, tool_result_is_error

    result = run_anima_tool(tool_name, tool_args)
    _print_untrusted_result(display_name, result)
    if tool_result_is_error(result):
        raise SystemExit(1)


def _refuse_impersonation(anima_name: str, requested_name: str) -> None:
    from core.i18n import t

    print(t("cli.board_impersonation_refused", anima=anima_name, requested=requested_name), file=sys.stderr)
    raise SystemExit(1)


# ── Board Read ────────────────────────────────────────────


def cmd_board_read(args: argparse.Namespace) -> None:
    """Read recent messages from a shared channel."""
    from core.i18n import t

    if current_anima_dir() is not None:
        _run_anima_untrusted_tool(
            "read_channel",
            "animaworks-tool board read",
            {
                "channel": args.channel,
                "limit": getattr(args, "limit", 20),
                "human_only": getattr(args, "human_only", False),
            },
        )
        return

    from core.exceptions import RecipientNotFoundError
    from core.infra.runtime_init import ensure_runtime_dir
    from core.messaging.board import board_access, operator_principal
    from core.messaging.messenger import Messenger, _validate_name
    from core.paths import get_shared_dir

    ensure_runtime_dir()
    shared_dir = get_shared_dir()
    try:
        _validate_name(args.channel, "channel name")
    except RecipientNotFoundError as exc:
        _print_untrusted_result("animaworks-tool board read", t("cli.board_read_invalid_channel", detail=str(exc)))
        raise SystemExit(1) from exc

    channels_dir = shared_dir / "channels"
    if (
        not (channels_dir / f"{args.channel}.jsonl").is_file()
        and not (channels_dir / f"{args.channel}.meta.json").is_file()
    ):
        _print_untrusted_result("animaworks-tool board read", t("handler.channel_not_found", channel=args.channel))
        raise SystemExit(1)

    access = board_access(shared_dir, args.channel, operator_principal(), "read")
    if not access.allowed:
        _print_untrusted_result("animaworks-tool board read", t("handler.channel_acl_denied", channel=args.channel))
        raise SystemExit(1)

    messenger = Messenger(shared_dir, "cli")
    messages = messenger.read_channel(
        args.channel,
        limit=getattr(args, "limit", 20),
        human_only=getattr(args, "human_only", False),
        source="human",
    )
    if not messages:
        _print_untrusted_result("animaworks-tool board read", t("handler.channel_empty", channel=args.channel))
        return
    _print_untrusted_result("animaworks-tool board read", json.dumps(messages, ensure_ascii=False, indent=2))


# ── Board Post ────────────────────────────────────────────


def cmd_board_post(args: argparse.Namespace) -> None:
    """Post a message to a shared channel."""
    from core.i18n import t

    post_args = getattr(args, "post_args", None)
    if post_args is None:
        # Keep direct Python callers of the former parser namespace compatible.
        post_args = [args.from_anima, args.channel, args.text]
    if len(post_args) > 3:
        print(t("cli.board_post_invalid_args"), file=sys.stderr)
        raise SystemExit(1)
    if len(post_args) == 1:
        legacy_from = None
        channel = post_args[0]
        raw_text = None
    elif len(post_args) == 2:
        legacy_from = None
        channel, raw_text = post_args
    else:
        legacy_from, channel, raw_text = post_args

    if raw_text is None or raw_text == "-":
        text = sys.stdin.read()
        if text.endswith("\n"):
            text = text[:-1]
            if text.endswith("\r"):
                text = text[:-1]
    else:
        text = raw_text
    if not text:
        print(t("cli.board_post_empty"), file=sys.stderr)
        raise SystemExit(1)

    anima_dir = current_anima_dir()
    if anima_dir is not None:
        for requested_name in (legacy_from, getattr(args, "as_name", None)):
            if requested_name and requested_name != anima_dir.name:
                _refuse_impersonation(anima_dir.name, requested_name)
        run_and_print(
            "post_channel",
            {
                "channel": channel,
                "text": text,
            },
        )
        return

    poster_name = getattr(args, "as_name", None) or legacy_from or "operator"
    from core.infra.runtime_init import ensure_runtime_dir

    ensure_runtime_dir()
    try:
        from core.messaging.board_client import BoardPostClientError, post_via_server

        response = post_via_server(
            channel,
            text,
            from_anima=poster_name,
            source="operator",
            from_name=poster_name,
        )
    except BoardPostClientError as exc:
        print(t("cli.board_post_failed", detail=str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(t("cli.board_post_failed", detail=str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc
    post = response.get("post", response)
    if hasattr(post, "to_dict"):
        post = post.to_dict()
    if not isinstance(post, dict) or not (post.get("committed") or post.get("duplicate")):
        error = post.get("error", "Board post failed") if isinstance(post, dict) else "Invalid Board post response"
        print(t("cli.board_post_failed", detail=error), file=sys.stderr)
        raise SystemExit(1)
    print(t("cli.board_post_success", channel=channel))


# ── Board DM History ──────────────────────────────────────


def cmd_board_dm_history(args: argparse.Namespace) -> None:
    """Read DM history with a specific peer."""
    from core.i18n import t

    anima_dir = current_anima_dir()
    if anima_dir is not None:
        if args.from_anima != anima_dir.name:
            _refuse_impersonation(anima_dir.name, args.from_anima)
        _run_anima_untrusted_tool(
            "read_dm_history",
            "animaworks-tool board dm-history",
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
        _print_untrusted_result("animaworks-tool board dm-history", t("handler.dm_history_empty", peer=args.peer))
        return
    _print_untrusted_result("animaworks-tool board dm-history", json.dumps(messages, ensure_ascii=False, indent=2))
