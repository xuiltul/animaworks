# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Slack integration for AnimaWorks.

Provides:
- SlackClient: Slack Web API wrapper with rate-limit retry and pagination
- MessageCache: SQLite cache for offline search and unreplied detection
- get_tool_schemas(): Anthropic tool_use schemas
- cli_main(): standalone CLI entry point
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.integrations._base import dispatch_by_table

# Re-exports for backward compatibility
from core.integrations._slack_cache import MessageCache  # noqa: F401
from core.integrations._slack_cli import cli_main, get_cli_guide  # noqa: F401
from core.integrations._slack_client import SlackClient  # noqa: F401
from core.integrations._slack_markdown import (  # noqa: F401
    clean_slack_markup,
    format_slack_ts,
    md_to_slack_mrkdwn,
    taskboard_md_to_slack,
    truncate,
)

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "channels": {"expected_seconds": 10, "background_eligible": False},
    "messages": {"expected_seconds": 30, "background_eligible": False},
    # gated by explicit permission, except replies to a valid Slack reply grant.
    "send": {"expected_seconds": 10, "background_eligible": False, "gated": True},
    "search": {"expected_seconds": 30, "background_eligible": False},
    "unreplied": {"expected_seconds": 30, "background_eligible": False},
    "channel_post": {"expected_seconds": 10, "background_eligible": False, "gated": True},
    "channel_update": {"expected_seconds": 10, "background_eligible": False, "gated": True},
}


# ── Token Resolution ───────────────────────────────────────


def _resolve_slack_token(args: dict[str, Any]) -> str | None:
    """Resolve per-Anima Slack bot token from tool dispatch args."""
    from core.channels.tokens import resolve_per_anima_token
    from core.credentials import resolve_env_style_credential

    return resolve_per_anima_token(
        "slack",
        args.get("anima_dir"),
        credential_lookup=resolve_env_style_credential,
        log=True,
    )


def _resolve_slack_identity(args: dict[str, Any]) -> tuple[str, str]:
    """Resolve Anima display name and icon URL for Slack messages.

    See :func:`core.integrations._anima_icon_url.resolve_anima_icon_identity`.
    """
    from core.integrations._anima_icon_url import resolve_anima_icon_identity

    anima_dir = args.get("anima_dir")
    if not anima_dir:
        return ("", "")
    return resolve_anima_icon_identity(Path(anima_dir).name, channel_config=None)


# ── Tool Schemas ───────────────────────────────────────────


def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for Slack tools.

    ``slack_channel_post`` / ``slack_channel_update`` are gated actions.
    A matching, unexpired Slack reply grant permits only a thread reply;
    other posts require ``slack_channel_post`` and updates require
    ``slack_channel_update`` in permissions.md.
    """
    return [
        {
            "name": "slack_channel_post",
            "description": (
                "Post a message to an actual Slack channel via Bot Token API. "
                "Returns the message ts for future updates via slack_channel_update. "
                "Use this for external Slack channels (not internal Board)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Slack channel ID (e.g. C0AJ4J5KK46)",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text (Markdown will be converted to Slack mrkdwn)",
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Optional parent message ts to reply in-thread",
                    },
                },
                "required": ["channel_id", "text"],
            },
        },
        {
            "name": "slack_channel_update",
            "description": (
                "Update an existing Slack message by ts. "
                "The message is silently replaced (no notification). "
                "Use this for live dashboards like task-board."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Slack channel ID"},
                    "ts": {
                        "type": "string",
                        "description": "Message timestamp to update (from slack_channel_post result)",
                    },
                    "text": {"type": "string", "description": "New message text"},
                },
                "required": ["channel_id", "ts", "text"],
            },
        },
    ]


# ── Dispatch ───────────────────────────────────────────────

logger = logging.getLogger("animaworks.integrations.slack")


def _record_external_reply(
    args: dict[str, Any],
    *,
    body: str,
    channel: str,
    thread_ts: str | None,
    response: Any,
    recipient_channel: str | None = None,
) -> None:
    """Record a user-addressed Slack post for the chat mirror (best effort)."""
    anima_dir = args.get("anima_dir")
    if not anima_dir:
        return
    try:
        from core.activity.logger import ActivityLogger
        from core.activity.slack_mirror import display_text_for_slack, slack_reply_recipient
        from core.execution.session.session_context import RuntimeSessionContext

        context = RuntimeSessionContext.from_env()
        trigger = context.trigger if context is not None else ""
        from core.notification.reply_routing import lookup_notification_mapping
        from core.notification.slack_names import user_name_snapshot

        anima_name = Path(str(anima_dir)).name
        mapping = lookup_notification_mapping(str(thread_ts)) if thread_ts else None
        recipient = slack_reply_recipient(
            text=body,
            channel=recipient_channel or channel,
            thread_ts=str(thread_ts or ""),
            trigger=trigger,
            anima_name=anima_name,
            notification_thread_owner=str(mapping.get("anima") or "") if mapping else None,
        )
        if recipient is None:
            return
        response_ts = ""
        response_channel = ""
        if isinstance(response, dict):
            response_ts = str(response.get("ts", "") or "")
            response_channel = str(response.get("channel", "") or "")
        ActivityLogger(Path(str(anima_dir))).log(
            "external_reply_sent",
            content=display_text_for_slack(body, user_name_snapshot()),
            to_person=recipient,
            channel="slack",
            via="slack",
            ctx="chat",
            meta={
                "thread_id": "default",
                "external_channel_id": response_channel or channel,
                "external_thread_ts": str(thread_ts or ""),
                "external_ts": response_ts,
            },
            safe=True,
        )
    except Exception:
        logger.warning("Failed to record Slack external reply", exc_info=True)


def _remember_post(args: dict[str, Any], channel: str, response: Any, thread_ts: str | None, text: str) -> None:
    """Let human replies in this thread reach the posting Anima (best effort)."""
    anima_dir = args.get("anima_dir")
    if not anima_dir:
        return
    from core.notification.reply_routing import remember_anima_slack_post

    remember_anima_slack_post(Path(str(anima_dir)).name, channel, response, thread_ts=thread_ts, text=text)


def _dispatch_slack_send(args: dict[str, Any]) -> Any:
    client = SlackClient(token=_resolve_slack_token(args))
    channel_id = client.resolve_channel(args["channel"])
    username, icon_url = _resolve_slack_identity(args)
    response = client.post_message(
        channel_id,
        md_to_slack_mrkdwn(args["message"]),
        thread_ts=args.get("thread_ts"),
        username=username,
        icon_url=icon_url,
    )
    _record_external_reply(
        args,
        body=args["message"],
        channel=channel_id,
        thread_ts=args.get("thread_ts"),
        response=response,
        recipient_channel=str(args.get("channel") or ""),
    )
    _remember_post(args, channel_id, response, args.get("thread_ts"), args["message"])
    return response


def _dispatch_slack_messages(args: dict[str, Any]) -> Any:
    from core.integrations._slack_cache import MessageCache  # noqa: F811

    client = SlackClient(token=_resolve_slack_token(args))
    channel_id = client.resolve_channel(args["channel"])
    cache = MessageCache()
    try:
        limit = args.get("limit", 20)
        messages = client.channel_history(channel_id, limit=limit)
        if messages:
            for message in messages:
                user_id = message.get("user", message.get("bot_id", ""))
                if user_id:
                    message["user_name"] = client.resolve_user_name(user_id)
            cache.upsert_messages(channel_id, messages)
            cache.update_sync_state(channel_id)
        return cache.get_recent(channel_id, limit=limit)
    finally:
        cache.close()


def _dispatch_slack_search(args: dict[str, Any]) -> Any:
    from core.integrations._slack_cache import MessageCache  # noqa: F811

    client = SlackClient(token=_resolve_slack_token(args))
    cache = MessageCache()
    try:
        channel_id = None
        if args.get("channel"):
            channel_id = client.resolve_channel(args["channel"])
        return cache.search(
            args["keyword"],
            channel_id=channel_id,
            limit=args.get("limit", 50),
        )
    finally:
        cache.close()


def _dispatch_slack_unreplied(args: dict[str, Any]) -> Any:
    from core.integrations._slack_cache import MessageCache  # noqa: F811

    client = SlackClient(token=_resolve_slack_token(args))
    cache = MessageCache()
    try:
        client.auth_test()
        return cache.find_unreplied(client.my_user_id or "")
    finally:
        cache.close()


def _dispatch_slack_channels(args: dict[str, Any]) -> Any:
    client = SlackClient(token=_resolve_slack_token(args))
    return client.channels()


def _dispatch_slack_react(args: dict[str, Any]) -> Any:
    client = SlackClient(token=_resolve_slack_token(args))
    channel_id = client.resolve_channel(args["channel"])
    return client.add_reaction(
        channel_id,
        args["emoji"],
        args["message_ts"],
    )


def _dispatch_slack_channel_post(args: dict[str, Any]) -> Any:
    client = SlackClient(token=_resolve_slack_token(args))
    slack_text = md_to_slack_mrkdwn(args["text"])
    username, icon_url = _resolve_slack_identity(args)
    response = client.post_message(
        args["channel_id"],
        slack_text,
        thread_ts=args.get("thread_ts"),
        username=username,
        icon_url=icon_url,
    )
    _record_external_reply(
        args,
        body=args["text"],
        channel=args["channel_id"],
        thread_ts=args.get("thread_ts"),
        response=response,
    )
    _remember_post(args, args["channel_id"], response, args.get("thread_ts"), args["text"])
    timestamp = response.get("ts", "") if response is not None else ""
    return {"status": "ok", "channel": args["channel_id"], "ts": timestamp}


def _dispatch_slack_channel_update(args: dict[str, Any]) -> Any:
    client = SlackClient(token=_resolve_slack_token(args))
    slack_text = md_to_slack_mrkdwn(args["text"])
    client.update_message(args["channel_id"], args["ts"], slack_text)
    return {"status": "ok", "channel": args["channel_id"], "ts": args["ts"]}


_DISPATCH_HANDLERS = {
    "slack_send": _dispatch_slack_send,
    "slack_messages": _dispatch_slack_messages,
    "slack_search": _dispatch_slack_search,
    "slack_unreplied": _dispatch_slack_unreplied,
    "slack_channels": _dispatch_slack_channels,
    "slack_react": _dispatch_slack_react,
    "slack_channel_post": _dispatch_slack_channel_post,
    "slack_channel_update": _dispatch_slack_channel_update,
}


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    return dispatch_by_table(_DISPATCH_HANDLERS, name, args)


if __name__ == "__main__":
    cli_main()
