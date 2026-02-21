from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Slack notification channel via Incoming Webhook or Bot Token API."""

import logging
from typing import Any

import httpx

from core.notification.notifier import NotificationChannel, register_channel
from core.tools.slack import md_to_slack_mrkdwn

logger = logging.getLogger("animaworks.notification.slack")


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
    ) -> str:
        # Try bot token mode first
        bot_token = self._config.get("bot_token", "")
        if not bot_token:
            bot_token = self._resolve_env("bot_token_env")
        if not bot_token:
            # Fall back to credentials.json
            try:
                from core.tools._base import get_credential
                bot_token = get_credential("slack", "notification", env_var="SLACK_BOT_TOKEN")
            except Exception:
                bot_token = ""

        if bot_token:
            return await self._send_via_bot(bot_token, subject, body, priority, anima_name)

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
    ) -> str:
        channel = self._config.get("channel", "")
        if not channel:
            return "slack: ERROR - channel not configured for bot token mode"

        text = self._build_text(subject, body, priority, anima_name)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {bot_token}"},
                    json={"channel": channel, "text": text},
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok"):
                    msg = f"slack: ERROR - {data.get('error', 'unknown')}"
                    logger.error(msg)
                    return msg
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
