from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Chatwork API notification channel."""

import logging
from typing import Any

import httpx

from core.notification.notifier import NotificationChannel, register_channel

logger = logging.getLogger("animaworks.notification.chatwork")

_CHATWORK_API_URL = "https://api.chatwork.com/v2/rooms/{room_id}/messages"


@register_channel("chatwork")
class ChatworkChannel(NotificationChannel):
    """Send notifications via Chatwork REST API."""

    @property
    def channel_type(self) -> str:
        return "chatwork"

    async def send(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
    ) -> str:
        token = self._resolve_env("api_token_env")
        if not token:
            return "chatwork: ERROR - api_token_env not configured or env var not set"

        room_id = self._config.get("room_id", "")
        if not room_id:
            return "chatwork: ERROR - room_id not configured"
        if not room_id.isdigit():
            return "chatwork: ERROR - room_id must be numeric"

        prefix = f"[{priority.upper()}] " if priority in ("high", "urgent") else ""
        sender = f" (from {anima_name})" if anima_name else ""
        message = f"[info][title]{prefix}{subject}{sender}[/title]{body}[/info]"[:28000]

        url = _CHATWORK_API_URL.format(room_id=room_id)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"X-ChatWorkToken": token},
                    data={"body": message},
                )
                resp.raise_for_status()
            logger.info("Chatwork notification sent: %s", subject[:50])
            return "chatwork: OK"
        except httpx.HTTPStatusError as e:
            msg = f"chatwork: ERROR - HTTP {e.response.status_code}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"chatwork: ERROR - {e}"
            logger.error(msg)
            return msg
