from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Automatic outbound routing for external platform messages.

When an Anima processes an inbox message that originated from Slack,
this module posts the LLM response back to the originating
channel/thread automatically — without relying on the LLM to
explicitly call ``slack_channel_post``.
"""

import logging
import os
from typing import Any

import httpx

from core.messenger import InboxItem
from core.tools._base import _lookup_shared_credentials, _lookup_vault_credential

logger = logging.getLogger("animaworks.outbound_auto")

_SLACK_POST_URL = "https://slack.com/api/chat.postMessage"
_SLACK_TIMEOUT = 30.0
_MAX_SLACK_TEXT = 40000


def _resolve_bot_token(anima_name: str) -> str | None:
    """Resolve the per-Anima or shared Slack bot token."""
    per_anima_key = f"SLACK_BOT_TOKEN__{anima_name}"
    token = _lookup_vault_credential(per_anima_key)
    if token:
        return token
    token = _lookup_shared_credentials(per_anima_key)
    if token:
        return token
    token = os.environ.get(per_anima_key)
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

        from core.tools._slack_markdown import md_to_slack_mrkdwn

        slack_text = md_to_slack_mrkdwn(response_text)[:_MAX_SLACK_TEXT]

        posted_ts: list[str] = []
        async with httpx.AsyncClient(timeout=_SLACK_TIMEOUT) as client:
            for target in slack_targets:
                ts = await self._post_one(
                    client,
                    token=token,
                    channel_id=target["channel_id"],
                    thread_ts=target["thread_ts"],
                    text=slack_text,
                    mention_prefix=target.get("mention_prefix", ""),
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

            targets.append({
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "mention_prefix": mention,
            })
        return targets

    @staticmethod
    async def _post_one(
        client: httpx.AsyncClient,
        *,
        token: str,
        channel_id: str,
        thread_ts: str,
        text: str,
        mention_prefix: str = "",
    ) -> str:
        """Post a single message. Return the ts or empty string on failure."""
        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": f"{mention_prefix}{text}" if mention_prefix else text,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
        try:
            resp = await client.post(
                _SLACK_POST_URL,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
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


class BoardSlackSync:
    """Sync AnimaWorks board posts back to mapped Slack channels."""

    async def sync_board_post(
        self,
        board_name: str,
        text: str,
        from_person: str,
        *,
        source: str = "",
    ) -> str | None:
        """Post a board message to the mapped Slack channel.

        Returns the Slack message ts on success, or None.
        Skips messages that originated from Slack (echo prevention).
        """
        if source == "slack":
            return None

        from core.config.models import load_config

        cfg = load_config()
        slack_cfg = cfg.external_messaging.slack
        if not slack_cfg.enabled:
            return None

        # Reverse lookup: board_name -> slack_channel_id
        channel_id = None
        for ch_id, bname in slack_cfg.board_mapping.items():
            if bname == board_name:
                channel_id = ch_id
                break
        if not channel_id:
            return None

        token = _resolve_bot_token(from_person)
        if not token:
            logger.warning("BoardSlackSync: no token for board '%s' sync", board_name)
            return None

        from core.tools._slack_markdown import md_to_slack_mrkdwn

        slack_text = md_to_slack_mrkdwn(text)[:_MAX_SLACK_TEXT]
        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": slack_text,
            "username": from_person,
        }

        try:
            async with httpx.AsyncClient(timeout=_SLACK_TIMEOUT) as client:
                resp = await client.post(
                    _SLACK_POST_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok"):
                    logger.error(
                        "BoardSlackSync: failed for '%s' -> %s: %s",
                        board_name,
                        channel_id,
                        data.get("error", "unknown"),
                    )
                    return None
                logger.info("BoardSlackSync: synced '%s' -> %s", board_name, channel_id)
                return data.get("ts")
        except Exception:
            logger.exception("BoardSlackSync: failed to sync '%s'", board_name)
            return None
