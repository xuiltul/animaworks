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
from core.notification.interactive import InteractionRequest

logger = logging.getLogger("animaworks.notification")

# ── Priority mapping ────────────────────────────────────────

PRIORITY_LEVELS = ("low", "normal", "high", "urgent")


def _record_timeline_notification(
    subject: str,
    body: str,
    priority: str,
    *,
    anima_name: str,
    delivered: bool,
) -> None:
    """Append a timeline ``notified`` item without affecting notification delivery."""
    try:
        from core.config.models import load_config, read_anima_company_checked

        hub = load_config().hub
        if not hub.timeline_enabled:
            return

        company = hub.company
        from_name = "system"
        source = "system"
        if anima_name:
            from core.paths import get_animas_dir

            readable, anima_company = read_anima_company_checked(get_animas_dir() / anima_name)
            if not readable or anima_company != hub.company:
                return
            company = anima_company
            from_name = anima_name
            source = "anima"

        subject_text = str(subject or "").strip()
        body_text = str(body or "").strip()
        if subject_text and body_text:
            text = f"{subject_text}: {body_text}"
        else:
            text = subject_text or body_text

        normalized_priority = priority if priority in PRIORITY_LEVELS else "normal"
        from core.hub.timeline import append_item

        append_item(
            _shared_dir(),
            from_name=from_name,
            company=company,
            kind="notified",
            text=text,
            source=source,
            urgent=normalized_priority in ("high", "urgent"),
            priority=normalized_priority,
            delivered=delivered,
        )
    except Exception:
        logger.warning("Failed to record timeline notification", exc_info=True)


def _shared_dir():
    from core.paths import get_shared_dir

    return get_shared_dir()


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
        interaction: InteractionRequest | None = None,
    ) -> str:
        """Send a notification. Returns a status message."""

    def _resolve_env(self, key: str) -> str:
        """Resolve a config value that may reference an env var via ``*_env`` suffix."""
        env_key = self._config.get(key)
        if not env_key:
            return ""
        return os.environ.get(env_key, "")

    def _resolve_credential_with_vault(
        self,
        config_key: str,
        anima_name: str = "",
        fallback_env: str = "",
    ) -> str:
        """Resolve credential from env → per-anima vault/shared → generic vault/shared.

        Follows the same resolution cascade as
        ``core.notification.channels.slack``.
        """
        token = self._resolve_env(config_key)
        if token:
            return token

        env_key = self._config.get(config_key, fallback_env)
        if not env_key:
            return ""

        from core.credentials import _lookup_shared_credentials, _lookup_vault_credential

        if anima_name:
            per_key = f"{env_key}__{anima_name}"
            val = _lookup_vault_credential(per_key) or _lookup_shared_credentials(per_key) or ""
            if val:
                return val

        return _lookup_vault_credential(env_key) or _lookup_shared_credentials(env_key) or ""


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
    def from_config(cls, config: HumanNotificationConfig, *, include_external: bool = True) -> HumanNotifier:
        """Build a notifier from the global HumanNotificationConfig.

        The built-in Web UI channel is always added (unless ``web_ui`` is
        off) so a fresh install can reach the user without external setup.
        """
        # Import channel modules to trigger registration
        _ensure_channels_registered()

        channels: list[NotificationChannel] = []
        for ch_config in config.channels if include_external else []:
            if not ch_config.enabled:
                continue
            try:
                channels.append(create_channel(ch_config))
            except ValueError:
                logger.warning(
                    "Skipping unknown notification channel: %s",
                    ch_config.type,
                )
        if config.web_ui and not any(ch.channel_type == "web" for ch in channels):
            from core.notification.channels.web import WebChannel

            channels.append(WebChannel({}))
        return cls(channels)

    @property
    def has_external_channels(self) -> bool:
        """Whether any channel besides the built-in Web UI is configured."""
        return any(ch.channel_type != "web" for ch in self._channels)

    @property
    def delivers_to_web(self) -> bool:
        """Whether the Web UI already receives notifications from this notifier."""
        return any(ch.channel_type == "web" for ch in self._channels)

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
        interaction: InteractionRequest | None = None,
    ) -> list[str]:
        """Send notification to all channels in parallel.

        Returns a list of status messages (one per channel).
        Failed channels return error strings instead of raising.
        """
        if not self._channels:
            _record_timeline_notification(
                subject,
                body,
                priority,
                anima_name=anima_name,
                delivered=False,
            )
            return ["No notification channels configured"]

        if priority not in PRIORITY_LEVELS:
            priority = "normal"

        results = await asyncio.gather(
            *[
                ch.send(subject, body, priority, anima_name=anima_name, interaction=interaction)
                for ch in self._channels
            ],
            return_exceptions=True,
        )

        status: list[str] = []
        for ch, result in zip(self._channels, results, strict=False):
            if isinstance(result, BaseException):
                msg = f"{ch.channel_type}: ERROR - {result}"
                logger.error("Notification failed for %s: %s", ch.channel_type, result)
                status.append(msg)
            else:
                status.append(str(result))

        failed_count = sum(1 for s in status if "ERROR" in s)
        if failed_count > 0:
            logger.warning(
                "Human notification partial failure: subject=%s priority=%s channels=%d failed=%d success=%d",
                subject[:50],
                priority,
                len(self._channels),
                failed_count,
                len(self._channels) - failed_count,
            )
        else:
            logger.info(
                "Human notification sent: subject=%s priority=%s channels=%d",
                subject[:50],
                priority,
                len(self._channels),
            )
        _record_timeline_notification(
            subject,
            body,
            priority,
            anima_name=anima_name,
            delivered=len(status) > failed_count,
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
    import core.notification.channels.chatwork  # noqa: F401
    import core.notification.channels.discord  # noqa: F401
    import core.notification.channels.line  # noqa: F401
    import core.notification.channels.ntfy  # noqa: F401
    import core.notification.channels.slack  # noqa: F401
    import core.notification.channels.telegram  # noqa: F401
    import core.notification.channels.web  # noqa: F401
