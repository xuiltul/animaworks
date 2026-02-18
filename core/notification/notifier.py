from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""HumanNotifier — fan-out notifications to multiple channels.

``HumanNotifier`` owns a list of ``NotificationChannel`` instances and
sends a notification to all of them in parallel via ``asyncio.gather``.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from core.config.models import HumanNotificationConfig, NotificationChannelConfig

logger = logging.getLogger("animaworks.notification")

# ── Priority mapping ────────────────────────────────────────

PRIORITY_LEVELS = ("low", "normal", "high", "urgent")


# ── Abstract base ───────────────────────────────────────────


class NotificationChannel(ABC):
    """Abstract base for a human notification channel."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    @property
    @abstractmethod
    def channel_type(self) -> str:
        """Return the channel type identifier (e.g. 'slack')."""

    @abstractmethod
    async def send(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
    ) -> str:
        """Send a notification. Returns a status message."""

    def _resolve_env(self, key: str) -> str:
        """Resolve a config value that may reference an env var via ``*_env`` suffix."""
        env_key = self._config.get(key)
        if not env_key:
            return ""
        return os.environ.get(env_key, "")


# ── Factory ─────────────────────────────────────────────────

_CHANNEL_REGISTRY: dict[str, type[NotificationChannel]] = {}


def register_channel(channel_type: str):
    """Decorator to register a channel implementation."""
    def decorator(cls: type[NotificationChannel]):
        _CHANNEL_REGISTRY[channel_type] = cls
        return cls
    return decorator


def create_channel(channel_config: NotificationChannelConfig) -> NotificationChannel:
    """Create a channel instance from config."""
    cls = _CHANNEL_REGISTRY.get(channel_config.type)
    if cls is None:
        raise ValueError(f"Unknown notification channel type: {channel_config.type}")
    return cls(channel_config.config)


# ── HumanNotifier ───────────────────────────────────────────


class HumanNotifier:
    """Fan-out notifier that sends to all configured channels in parallel."""

    def __init__(self, channels: list[NotificationChannel]) -> None:
        self._channels = channels

    @classmethod
    def from_config(cls, config: HumanNotificationConfig) -> HumanNotifier:
        """Build a notifier from the global HumanNotificationConfig."""
        # Import channel modules to trigger registration
        _ensure_channels_registered()

        channels: list[NotificationChannel] = []
        for ch_config in config.channels:
            if not ch_config.enabled:
                continue
            try:
                channels.append(create_channel(ch_config))
            except ValueError:
                logger.warning(
                    "Skipping unknown notification channel: %s",
                    ch_config.type,
                )
        return cls(channels)

    @property
    def channel_count(self) -> int:
        return len(self._channels)

    async def notify(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        *,
        anima_name: str = "",
    ) -> list[str]:
        """Send notification to all channels in parallel.

        Returns a list of status messages (one per channel).
        Failed channels return error strings instead of raising.
        """
        if not self._channels:
            return ["No notification channels configured"]

        if priority not in PRIORITY_LEVELS:
            priority = "normal"

        results = await asyncio.gather(
            *[
                ch.send(subject, body, priority, anima_name=anima_name)
                for ch in self._channels
            ],
            return_exceptions=True,
        )

        status: list[str] = []
        for ch, result in zip(self._channels, results):
            if isinstance(result, BaseException):
                msg = f"{ch.channel_type}: ERROR - {result}"
                logger.error("Notification failed for %s: %s", ch.channel_type, result)
                status.append(msg)
            else:
                status.append(str(result))

        logger.info(
            "Human notification sent: subject=%s priority=%s channels=%d",
            subject[:50],
            priority,
            len(self._channels),
        )
        return status


_builtins_registered = False


def _ensure_channels_registered() -> None:
    """Import all built-in channel modules so they register themselves."""
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True
    # Import triggers @register_channel decorators
    import core.notification.channels.slack  # noqa: F401
    import core.notification.channels.line  # noqa: F401
    import core.notification.channels.telegram  # noqa: F401
    import core.notification.channels.chatwork  # noqa: F401
    import core.notification.channels.ntfy  # noqa: F401
