from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""LINE Messaging API notification channel."""

import logging
from typing import Any

import httpx

from core.notification.notifier import NotificationChannel, register_channel

logger = logging.getLogger("animaworks.notification.line")

_LINE_API_URL = "https://api.line.me/v2/bot/message/push"


@register_channel("line")
class LineChannel(NotificationChannel):
    """Send notifications via LINE Messaging API (Push Message)."""

    @property
    def channel_type(self) -> str:
        return "line"

    async def send(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
    ) -> str:
        token = self._resolve_env("channel_access_token_env")
        if not token:
            return "line: ERROR - channel_access_token_env not configured or env var not set"

        user_id = self._config.get("user_id", "")
        if not user_id:
            return "line: ERROR - user_id not configured"

        prefix = f"[{priority.upper()}] " if priority in ("high", "urgent") else ""
        sender = f" (from {anima_name})" if anima_name else ""
        text = f"{prefix}{subject}{sender}\n\n{body}"

        payload = {
            "to": user_id,
            "messages": [{"type": "text", "text": text[:5000]}],
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    _LINE_API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
            logger.info("LINE notification sent: %s", subject[:50])
            return "line: OK"
        except httpx.HTTPStatusError as e:
            msg = f"line: ERROR - HTTP {e.response.status_code}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"line: ERROR - {e}"
            logger.error(msg)
            return msg
