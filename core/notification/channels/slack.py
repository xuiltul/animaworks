from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Slack notification channel via Incoming Webhook or Bot Token API."""

import logging
from typing import TYPE_CHECKING, Any

import httpx

from core.notification.notifier import NotificationChannel, register_channel

if TYPE_CHECKING:
    from core.notification.interactive import InteractionRequest
from core.tools.slack import md_to_slack_mrkdwn

logger = logging.getLogger("animaworks.notification.slack")


def _build_interactive_blocks(text: str, interaction: InteractionRequest) -> list[dict[str, Any]]:
    """Build Slack Block Kit blocks with action buttons for interactive call_human."""
    style_map = {"approve": "primary", "reject": "danger"}
    emoji_map = {"approve": "✅", "reject": "❌", "comment": "💬"}
    elements: list[dict[str, Any]] = []
    for opt in interaction.options:
        btn: dict[str, Any] = {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": f"{emoji_map.get(opt, '▶️')} {opt.capitalize()}",
            },
            "action_id": f"aw_interact_{opt}",
            "value": interaction.callback_id,
        }
        style = style_map.get(opt)
        if style:
            btn["style"] = style
        elements.append(btn)
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "block_id": f"aw_interact:{interaction.callback_id}",
            "elements": elements,
        },
    ]


@register_channel("slack")
class SlackChannel(NotificationChannel):
    """Send notifications to Slack via Incoming Webhook or Bot Token.

    Config options (one of the two modes is required):

    Webhook mode:
        webhook_url: Incoming Webhook URL
        webhook_url_env: Environment variable containing the URL

    Bot Token mode:
        bot_token: Slack Bot Token (xoxb-...)
        bot_token_env: Environment variable containing the token
        channel: Channel ID or User ID to send to
    """

    @property
    def channel_type(self) -> str:
        return "slack"

    async def send(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
        interaction: InteractionRequest | None = None,
    ) -> str:
        bot_token = self._config.get("bot_token", "")
        if not bot_token:
            bot_token = self._resolve_env("bot_token_env")
        if not bot_token and anima_name:
            from core.tools._base import resolve_env_style_credential

            per_key = f"SLACK_BOT_TOKEN__{anima_name}"
            bot_token = resolve_env_style_credential(per_key) or ""
        if not bot_token and self._config.get("channel"):
            try:
                from core.tools._base import get_credential

                bot_token = get_credential("slack", "notification", env_var="SLACK_BOT_TOKEN")
            except Exception:
                bot_token = ""

        if bot_token:
            return await self._send_via_bot(
                bot_token,
                subject,
                body,
                priority,
                anima_name,
                interaction=interaction,
            )

        # Fall back to webhook mode
        webhook_url = self._config.get("webhook_url", "")
        if not webhook_url:
            webhook_url = self._resolve_env("webhook_url_env")
        if not webhook_url:
            return "slack: ERROR - neither bot_token nor webhook_url configured"

        return await self._send_via_webhook(webhook_url, subject, body, priority, anima_name)

    @staticmethod
    def _build_text(subject: str, body: str, priority: str, anima_name: str) -> str:
        prefix = f"[{priority.upper()}] " if priority in ("high", "urgent") else ""
        sender = f" (from {anima_name})" if anima_name else ""
        body = md_to_slack_mrkdwn(body)
        return f"{prefix}*{subject}*{sender}\n{body}"[:40000]

    async def _send_via_bot(
        self,
        bot_token: str,
        subject: str,
        body: str,
        priority: str,
        anima_name: str,
        *,
        interaction: InteractionRequest | None = None,
    ) -> str:
        channel = self._config.get("channel", "")
        if not channel:
            return "slack: ERROR - channel not configured for bot token mode"

        # When username override is active, omit "(from X)" — username is visible as sender
        text = self._build_text(subject, body, priority, "")

        payload: dict[str, Any] = {"channel": channel, "text": text}
        if anima_name:
            payload["username"] = anima_name
            icon_url = ""
            try:
                from core.tools._anima_icon_url import resolve_anima_icon_url

                icon_url = resolve_anima_icon_url(
                    anima_name,
                    channel_config=self._config,
                )
            except Exception:
                logger.debug("resolve_anima_icon_url failed for notification", exc_info=True)
            if icon_url:
                payload["icon_url"] = icon_url

        if interaction is not None:
            payload["blocks"] = _build_interactive_blocks(text, interaction)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {bot_token}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok"):
                    msg = f"slack: ERROR - {data.get('error', 'unknown')}"
                    logger.error(msg)
                    return msg

                if interaction is not None and data.get("ts"):
                    try:
                        import asyncio as _asyncio

                        from core.notification.interactive import (
                            update_interaction_message_ts_resilient,
                        )

                        # Server-API-first: direct run/ writes fail inside
                        # sandboxed MCP servers (read-only filesystem).
                        await _asyncio.to_thread(
                            update_interaction_message_ts_resilient,
                            interaction.callback_id,
                            "slack",
                            str(data["ts"]),
                        )
                    except Exception:
                        logger.debug(
                            "Failed to persist interactive Slack ts",
                            exc_info=True,
                        )

                if anima_name and data.get("ts"):
                    try:
                        import asyncio

                        from core.notification.reply_routing import (
                            save_notification_mapping_resilient,
                        )

                        # Falls back to the server internal API when the
                        # direct run/ write fails (sandboxed MCP server).
                        saved = await asyncio.to_thread(
                            save_notification_mapping_resilient,
                            data["ts"],
                            data.get("channel", channel),
                            anima_name,
                            notification_text=f"{subject}\n{body}"[:2000],
                            callback_id=interaction.callback_id if interaction is not None else "",
                        )
                        if not saved:
                            logger.warning(
                                "Notification mapping not saved for ts=%s; thread replies will not route back",
                                data["ts"],
                            )
                    except Exception:
                        logger.debug(
                            "Failed to save notification mapping",
                            exc_info=True,
                        )

            logger.info("Slack notification sent via bot: %s", subject[:50])
            return "slack: OK"
        except httpx.HTTPStatusError as e:
            msg = f"slack: ERROR - HTTP {e.response.status_code}"
            logger.error(msg)
            return msg
        except Exception:
            logger.exception("Slack bot API request failed for: %s", subject[:50])
            return "slack: ERROR - request failed"

    async def _send_via_webhook(
        self,
        webhook_url: str,
        subject: str,
        body: str,
        priority: str,
        anima_name: str,
    ) -> str:
        # Webhook API does not return a message ts, so we cannot save a
        # notification mapping here.  Reply routing only works for messages
        # sent via the Bot Token API (chat.postMessage).
        text = self._build_text(subject, body, priority, anima_name)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(webhook_url, json={"text": text})
                resp.raise_for_status()
            logger.info("Slack notification sent via webhook: %s", subject[:50])
            return "slack: OK"
        except httpx.HTTPStatusError as e:
            msg = f"slack: ERROR - HTTP {e.response.status_code}"
            logger.error(msg)
            return msg
        except Exception:
            logger.exception("Slack webhook request failed for: %s", subject[:50])
            return "slack: ERROR - request failed"
