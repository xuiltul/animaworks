from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""ntfy push notification channel."""

import logging
from typing import Any

import httpx

from core.notification.notifier import NotificationChannel, register_channel

logger = logging.getLogger("animaworks.notification.ntfy")

# Map AnimaWorks priority to ntfy priority values (1-5)
_PRIORITY_MAP = {
    "low": "2",
    "normal": "3",
    "high": "4",
    "urgent": "5",
}


@register_channel("ntfy")
class NtfyChannel(NotificationChannel):
    """Send notifications via ntfy (self-hosted or ntfy.sh)."""

    @property
    def channel_type(self) -> str:
        return "ntfy"

    async def send(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
    ) -> str:
        server_url = self._config.get("server_url", "https://ntfy.sh")
        topic = self._config.get("topic", "")
        if not topic:
            return "ntfy: ERROR - topic not configured"

        url = f"{server_url.rstrip('/')}/{topic}"

        sender = f" (from {anima_name})" if anima_name else ""
        title = f"{subject}{sender}"[:256]
        headers: dict[str, str] = {
            "Title": title,
            "Priority": _PRIORITY_MAP.get(priority, "3"),
        }

        # Optional auth token
        token = self._resolve_env("token_env")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, content=body[:4096], headers=headers)
                resp.raise_for_status()
            logger.info("ntfy notification sent: %s", subject[:50])
            return "ntfy: OK"
        except httpx.HTTPStatusError as e:
            msg = f"ntfy: ERROR - HTTP {e.response.status_code}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"ntfy: ERROR - {e}"
            logger.error(msg)
            return msg
