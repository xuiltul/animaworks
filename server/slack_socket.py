from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Slack Socket Mode integration for real-time message reception."""

import logging

from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from core.config.models import load_config
from core.messenger import Messenger
from core.paths import get_data_dir
from core.tools._base import get_credential

logger = logging.getLogger("animaworks.slack_socket")


class SlackSocketModeManager:
    """Manages Slack Socket Mode WebSocket connection.

    Integrates with the existing FastAPI event loop via ``connect_async()``
    (non-blocking).  Messages are routed to anima inboxes using the same
    ``Messenger.receive_external()`` pipeline as the HTTP webhook endpoint.
    """

    def __init__(self) -> None:
        self._handler: AsyncSocketModeHandler | None = None
        self._app: AsyncApp | None = None

    async def start(self) -> None:
        """Start the Socket Mode connection if enabled in config."""
        config = load_config()
        slack_config = config.external_messaging.slack
        if not slack_config.enabled or slack_config.mode != "socket":
            logger.info("Slack Socket Mode is disabled")
            return

        bot_token = get_credential("slack", "slack_socket", env_var="SLACK_BOT_TOKEN")
        app_token = get_credential("slack_app", "slack_socket", env_var="SLACK_APP_TOKEN")

        self._app = AsyncApp(token=bot_token)
        self._register_handlers(slack_config.anima_mapping)
        self._handler = AsyncSocketModeHandler(self._app, app_token)
        await self._handler.connect_async()
        logger.info("Slack Socket Mode connected")

    def _register_handlers(self, anima_mapping: dict[str, str]) -> None:
        """Register Slack event handlers for message routing."""

        @self._app.event("message")
        async def handle_message(event: dict, say) -> None:  # noqa: ARG001
            # Skip message subtypes (edits, deletes, bot messages, etc.)
            if "subtype" in event:
                return

            channel_id = event.get("channel", "")
            anima_name = anima_mapping.get(channel_id)
            if not anima_name:
                logger.debug(
                    "No anima mapping for channel %s; ignoring message",
                    channel_id,
                )
                return

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name)
            messenger.receive_external(
                content=event.get("text", ""),
                source="slack",
                source_message_id=event.get("ts", ""),
                external_user_id=event.get("user", ""),
                external_channel_id=channel_id,
            )
            logger.info(
                "Socket Mode message routed: channel=%s -> anima=%s",
                channel_id,
                anima_name,
            )

    async def stop(self) -> None:
        """Disconnect the Socket Mode handler gracefully."""
        if self._handler:
            await self._handler.close_async()
            logger.info("Slack Socket Mode disconnected")

    @property
    def is_connected(self) -> bool:
        """Return whether the Socket Mode handler is active."""
        return self._handler is not None
