from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Automatic outbound routing for external platform messages.

When an Anima processes an inbox message that originated from Slack,
this module posts the LLM response back to the originating
channel/thread automatically — without relying on the LLM to
explicitly post a Slack reply via tool or CLI.
"""

import logging
import os
from typing import Any, Literal

from core.channels.slack import SlackHTTPClient
from core.credentials import _lookup_shared_credentials, _lookup_vault_credential
from core.messaging.messenger import InboxItem

logger = logging.getLogger("animaworks.outbound_auto")

_SLACK_TIMEOUT = 30.0
_MAX_SLACK_TEXT = 40000


def _resolve_avatar_url(anima_name: str) -> str:
    """Resolve avatar URL for Slack icon_url.

    Delegates to :func:`core.integrations._anima_icon_url.resolve_anima_icon_url`
    which applies the 3-tier resolution (per-Anima / env / config / channel / asset).
    """
    try:
        from core.integrations._anima_icon_url import resolve_anima_icon_url

        return resolve_anima_icon_url(anima_name, channel_config=None)
    except Exception:
        return ""


def _resolve_bot_token(anima_name: str) -> str | None:
    """Resolve the per-Anima or shared Slack bot token."""
    from core.channels.tokens import resolve_per_anima_token

    token = resolve_per_anima_token("slack", anima_name)
    if token:
        return token
    # Fallback to shared token
    token = _lookup_vault_credential("SLACK_BOT_TOKEN")
    if token:
        return token
    token = _lookup_shared_credentials("SLACK_BOT_TOKEN")
    if token:
        return token
    return os.environ.get("SLACK_BOT_TOKEN") or None


class SlackAutoResponder:
    """Post-cycle hook: auto-send LLM responses back to originating Slack channels."""

    async def on_inbox_response(
        self,
        anima_name: str,
        response_text: str,
        inbox_items: list[InboxItem],
        *,
        already_posted: set[str] | None = None,
    ) -> list[str]:
        """Post *response_text* to each Slack-sourced message's channel/thread.

        Args:
            anima_name: Name of the Anima that generated the response.
            response_text: The LLM's accumulated response text.
            inbox_items: The inbox items that were processed in this cycle.
            already_posted: Set of ``"channel_id:thread_ts"`` keys where the
                LLM already posted via tool calls (double-post prevention).

        Returns:
            List of Slack message ``ts`` values for successfully posted messages.
        """
        if not response_text or not response_text.strip():
            return []

        posted_keys = already_posted or set()
        slack_targets = self._collect_slack_targets(inbox_items, posted_keys)
        if not slack_targets:
            return []

        token = _resolve_bot_token(anima_name)
        if not token:
            logger.warning(
                "SlackAutoResponder: no bot token for anima '%s'; skipping",
                anima_name,
            )
            return []

        from core.integrations._slack_markdown import md_to_slack_mrkdwn

        slack_text = md_to_slack_mrkdwn(response_text)[:_MAX_SLACK_TEXT]

        # Resolve icon URL from AVATAR_URL__{name} env/credentials
        icon_url = _resolve_avatar_url(anima_name)

        posted_ts: list[str] = []
        async with SlackHTTPClient(timeout=_SLACK_TIMEOUT) as client:
            for target in slack_targets:
                ts = await self._post_one(
                    client,
                    token=token,
                    channel_id=target["channel_id"],
                    thread_ts=target["thread_ts"],
                    text=slack_text,
                    mention_prefix=target.get("mention_prefix", ""),
                    username=anima_name,
                    icon_url=icon_url,
                )
                if ts:
                    posted_ts.append(ts)

        if posted_ts:
            logger.info(
                "SlackAutoResponder: posted %d auto-response(s) for '%s'",
                len(posted_ts),
                anima_name,
            )
        return posted_ts

    @staticmethod
    def _collect_slack_targets(
        inbox_items: list[InboxItem],
        already_posted: set[str],
    ) -> list[dict[str, str]]:
        """Extract unique Slack targets, skipping already-posted ones."""
        seen: set[str] = set()
        targets: list[dict[str, str]] = []
        for item in inbox_items:
            msg = item.msg
            if getattr(msg, "source", "") != "slack":
                continue
            channel_id = getattr(msg, "external_channel_id", "")
            if not channel_id:
                continue
            thread_ts = getattr(msg, "external_thread_ts", "") or getattr(msg, "source_message_id", "")
            dedup_key = f"{channel_id}:{thread_ts}"
            if dedup_key in seen or dedup_key in already_posted:
                continue
            seen.add(dedup_key)

            mention = ""
            ext_uid = getattr(msg, "external_user_id", "")
            if ext_uid:
                mention = f"<@{ext_uid}> "

            targets.append(
                {
                    "channel_id": channel_id,
                    "thread_ts": thread_ts,
                    "mention_prefix": mention,
                }
            )
        return targets

    @staticmethod
    async def _post_one(
        client: SlackHTTPClient,
        *,
        token: str,
        channel_id: str,
        thread_ts: str,
        text: str,
        mention_prefix: str = "",
        username: str = "",
        icon_url: str = "",
    ) -> str:
        """Post a single message. Return the ts or empty string on failure."""
        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": f"{mention_prefix}{text}" if mention_prefix else text,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
        if username:
            payload["username"] = username
        if icon_url:
            payload["icon_url"] = icon_url
        try:
            data = await client.post_message(token, payload)
            if not data.get("ok"):
                logger.error(
                    "SlackAutoResponder: chat.postMessage failed for %s: %s",
                    channel_id,
                    data.get("error", "unknown"),
                )
                return ""
            return data.get("ts", "")
        except Exception:
            logger.exception("SlackAutoResponder: failed to post to %s", channel_id)
            return ""


class DiscordAutoResponder:
    """Post-cycle hook: auto-send LLM responses back to originating Discord channels."""

    async def on_inbox_response(
        self,
        anima_name: str,
        response_text: str,
        inbox_items: list[InboxItem],
        *,
        already_posted: set[str] | None = None,
    ) -> list[str]:
        """Post *response_text* to each Discord-sourced message's channel via webhook.

        Returns list of Discord message IDs for successfully posted messages.
        """
        if not response_text or not response_text.strip():
            return []

        posted_keys = already_posted or set()
        targets = self._collect_discord_targets(inbox_items, posted_keys)
        if not targets:
            return []

        try:
            from core.messaging.discord_webhooks import get_webhook_manager

            wm = get_webhook_manager()
        except Exception:
            logger.warning("DiscordAutoResponder: webhook manager not available")
            return []

        from core.integrations._discord_markdown import md_to_discord

        discord_text = md_to_discord(response_text)

        import asyncio

        posted_ids: list[str] = []
        for target in targets:
            try:
                mention = target.get("mention_prefix", "")
                text = f"{mention}{discord_text}" if mention else discord_text
                msg_id = await asyncio.to_thread(
                    wm.send_as_anima,
                    target["channel_id"],
                    anima_name,
                    text,
                )
                if msg_id:
                    posted_ids.append(msg_id)
            except Exception:
                logger.exception(
                    "DiscordAutoResponder: failed to post to %s",
                    target["channel_id"],
                )

        if posted_ids:
            logger.info(
                "DiscordAutoResponder: posted %d auto-response(s) for '%s'",
                len(posted_ids),
                anima_name,
            )
        return posted_ids

    @staticmethod
    def _collect_discord_targets(
        inbox_items: list[InboxItem],
        already_posted: set[str],
    ) -> list[dict[str, str]]:
        """Extract unique Discord targets from inbox items."""
        seen: set[str] = set()
        targets: list[dict[str, str]] = []
        for item in inbox_items:
            msg = item.msg
            if getattr(msg, "source", "") != "discord":
                continue
            channel_id = getattr(msg, "external_channel_id", "")
            if not channel_id:
                continue
            reference_id = getattr(msg, "external_thread_ts", "") or getattr(msg, "source_message_id", "")
            dedup_key = f"{channel_id}:{reference_id}"
            if dedup_key in seen or dedup_key in already_posted:
                continue
            seen.add(dedup_key)

            mention = ""
            ext_uid = getattr(msg, "external_user_id", "")
            if ext_uid:
                mention = f"<@{ext_uid}> "

            targets.append(
                {
                    "channel_id": channel_id,
                    "reference_id": reference_id,
                    "mention_prefix": mention,
                }
            )
        return targets


class BoardSlackSync:
    """Sync AnimaWorks board posts back to mapped Slack channels."""

    async def sync_board_post(
        self,
        board_name: str,
        text: str,
        from_person: str,
        *,
        source: str = "",
    ) -> Literal["sent", "skipped", "failed"]:
        """Post a board message to the mapped Slack channel.

        Return a status so the centralized Board writer can record the result.
        """
        if source in {"slack", "discord"}:
            return "skipped"

        from core.config.models import load_config

        try:
            cfg = load_config()
            slack_cfg = cfg.external_messaging.slack
        except Exception:
            logger.exception("BoardSlackSync: failed to load configuration for #%s", board_name)
            return "failed"
        if not slack_cfg.enabled:
            return "skipped"

        outbound_whitelist = getattr(slack_cfg, "board_outbound_sync", [])
        if outbound_whitelist:
            if board_name not in outbound_whitelist:
                return "skipped"
        elif not getattr(slack_cfg, "board_outbound_sync_all", False):
            return "skipped"

        # Reverse lookup: board_name -> slack_channel_id
        channel_id = None
        for ch_id, bname in slack_cfg.board_mapping.items():
            if bname == board_name:
                channel_id = ch_id
                break
        if not channel_id:
            return "skipped"

        token = _resolve_bot_token(from_person)
        if not token:
            logger.warning("BoardSlackSync: no token for board '%s' sync", board_name)
            return "failed"

        from core.integrations._slack_markdown import md_to_slack_mrkdwn

        slack_text = md_to_slack_mrkdwn(text)[:_MAX_SLACK_TEXT]
        icon_url = _resolve_avatar_url(from_person)
        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": slack_text,
            "username": from_person,
        }
        if icon_url:
            payload["icon_url"] = icon_url

        try:
            async with SlackHTTPClient(timeout=_SLACK_TIMEOUT) as client:
                data = await client.post_message(token, payload)
                if not data.get("ok"):
                    logger.error(
                        "BoardSlackSync: failed for '%s' -> %s: %s",
                        board_name,
                        channel_id,
                        data.get("error", "unknown"),
                    )
                    return "failed"
                logger.info("BoardSlackSync: synced '%s' -> %s", board_name, channel_id)
                return "sent"
        except Exception:
            logger.exception("BoardSlackSync: failed to sync '%s'", board_name)
            return "failed"
