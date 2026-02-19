from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Telegram Bot API notification channel."""

import html
import logging
from typing import Any

import httpx

from core.notification.notifier import NotificationChannel, register_channel

logger = logging.getLogger("animaworks.notification.telegram")

_TELEGRAM_API_BASE = "https://api.telegram.org"


@register_channel("telegram")
class TelegramChannel(NotificationChannel):
    """Send notifications via Telegram Bot API."""

    @property
    def channel_type(self) -> str:
        return "telegram"

    async def send(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
    ) -> str:
        token = self._resolve_env("bot_token_env")
        if not token:
            return "telegram: ERROR - bot_token_env not configured or env var not set"

        chat_id = self._config.get("chat_id", "")
        if not chat_id:
            return "telegram: ERROR - chat_id not configured"

        prefix = f"[{priority.upper()}] " if priority in ("high", "urgent") else ""
        sender = f" (from {anima_name})" if anima_name else ""
        safe_subject = html.escape(subject)
        safe_body = html.escape(body)
        text = f"{prefix}<b>{safe_subject}</b>{sender}\n\n{safe_body}"

        url = f"{_TELEGRAM_API_BASE}/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": "HTML",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            logger.info("Telegram notification sent: %s", subject[:50])
            return "telegram: OK"
        except httpx.HTTPStatusError as e:
            msg = f"telegram: ERROR - HTTP {e.response.status_code}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"telegram: ERROR - {e}"
            logger.error(msg)
            return msg
