from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Discord notification channel via Bot Token or Webhook."""

import logging
from typing import Any

from core.notification.interactive import InteractionRequest
from core.notification.notifier import NotificationChannel, register_channel
from core.tools._discord_markdown import md_to_discord

logger = logging.getLogger("animaworks.notification.discord")


def _build_discord_components(interaction: InteractionRequest) -> list[dict[str, Any]]:
    """Build Discord Message Components (Action Row with buttons)."""
    style_map = {"approve": 3, "reject": 4, "comment": 2}  # 3=Success/green, 4=Danger/red, 2=Secondary/gray
    emoji_map = {"approve": "✅", "reject": "❌", "comment": "💬"}
    buttons: list[dict[str, Any]] = []
    for opt in interaction.options:
        buttons.append(
            {
                "type": 2,  # Button
                "style": style_map.get(opt, 2),
                "label": f"{emoji_map.get(opt, '▶️')} {opt.capitalize()}",
                "custom_id": f"aw_interact:{interaction.callback_id}:{opt}",
            }
        )
    return [{"type": 1, "components": buttons}]  # Action Row


async def _persist_discord_msg_id(callback_id: str, msg_id: str) -> None:
    """Persist the Discord message id for button invalidation.

    Server-API-first: direct run/ writes fail inside sandboxed MCP servers
    (read-only filesystem). Best-effort — failures are logged, not raised.
    """
    import asyncio

    from core.notification.interactive import update_interaction_message_ts_resilient

    try:
        await asyncio.to_thread(update_interaction_message_ts_resilient, callback_id, "discord", msg_id)
    except Exception:
        logger.debug("Failed to persist interactive Discord msg_id", exc_info=True)


@register_channel("discord")
class DiscordChannel(NotificationChannel):
    """Send notifications to Discord via Bot Token DM or channel webhook.

    Config options:

    Bot Token mode (preferred for DM notifications):
        bot_token: Discord Bot Token
        bot_token_env: Environment variable containing the token
        user_id: Discord user snowflake ID for DM delivery

    Channel mode (for channel notifications):
        channel_id: Discord channel ID to post to
        webhook_url: Webhook URL for channel posting with Anima identity

    Webhook-only mode:
        webhook_url: Webhook URL
        webhook_url_env: Environment variable containing the URL
    """

    @property
    def channel_type(self) -> str:
        return "discord"

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
        if not bot_token:
            try:
                from core.tools._base import get_credential

                bot_token = get_credential("discord", "discord", env_var="DISCORD_BOT_TOKEN")
            except Exception:
                bot_token = ""

        text = self._build_text(subject, body, priority, anima_name)

        # Try webhook mode first (supports Anima identity)
        webhook_url = self._config.get("webhook_url", "")
        if not webhook_url:
            webhook_url = self._resolve_env("webhook_url_env")

        channel_id = self._config.get("channel_id", "")

        components = _build_discord_components(interaction) if interaction is not None else None

        if channel_id and bot_token:
            # Use webhook manager for channel posting with Anima identity
            return await self._send_via_channel(
                bot_token,
                channel_id,
                text,
                anima_name,
                interaction=interaction,
                components=components,
            )

        user_id = self._config.get("user_id", "")
        if user_id and bot_token:
            return await self._send_via_dm(bot_token, user_id, text, interaction=interaction, components=components)

        if webhook_url:
            return await self._send_via_webhook(
                webhook_url, text, anima_name, interaction=interaction, components=components
            )

        if bot_token and not user_id and not channel_id:
            return "discord: ERROR - bot_token set but no user_id or channel_id configured"

        return "discord: ERROR - no bot_token or webhook_url configured"

    @staticmethod
    def _build_text(subject: str, body: str, priority: str, anima_name: str) -> str:
        prefix = ""
        if priority in ("high", "urgent"):
            prefix = f"**[{priority.upper()}]** "
        sender = f" (from {anima_name})" if anima_name else ""
        body = md_to_discord(body)
        return f"{prefix}**{subject}**{sender}\n{body}"

    @staticmethod
    async def _send_via_dm(
        token: str,
        user_id: str,
        text: str,
        *,
        interaction: InteractionRequest | None = None,
        components: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send a DM to a Discord user."""
        try:
            from core.tools._discord_client import DiscordClient

            client = DiscordClient(token=token)
            dm = client.create_dm(user_id)
            dm_channel_id = str(dm.get("id", ""))
            if not dm_channel_id:
                return f"discord: ERROR - failed to open DM with user {user_id}"
            result = client.send_message(
                dm_channel_id,
                text[:2000],
                components=components,
            )
            msg_id = str(result.get("id", ""))
            logger.info("Discord notification sent via DM: user=%s msg=%s", user_id, msg_id)
            if interaction is not None and msg_id:
                await _persist_discord_msg_id(interaction.callback_id, msg_id)
            return f"discord: DM sent to {user_id} (msg_id={msg_id})"
        except Exception as exc:
            logger.exception("Discord DM notification failed")
            return f"discord: ERROR - {exc}"

    @staticmethod
    async def _send_via_channel(
        token: str,
        channel_id: str,
        text: str,
        anima_name: str,
        *,
        interaction: InteractionRequest | None = None,
        components: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send to a Discord channel via webhook manager (Anima identity)."""
        try:
            from core.discord_webhooks import get_webhook_manager

            wm = get_webhook_manager()
            msg_id = wm.send_as_anima(
                channel_id,
                anima_name or "AnimaWorks",
                text,
                components=components,
            )
            logger.info("Discord notification sent via channel: ch=%s msg=%s", channel_id, msg_id)
            if interaction is not None and msg_id:
                await _persist_discord_msg_id(interaction.callback_id, msg_id)
            return f"discord: channel {channel_id} (msg_id={msg_id})"
        except Exception as exc:
            logger.exception("Discord channel notification failed")
            return f"discord: ERROR - {exc}"

    @staticmethod
    async def _send_via_webhook(
        webhook_url: str,
        text: str,
        anima_name: str,
        *,
        interaction: InteractionRequest | None = None,
        components: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send via raw webhook URL."""
        try:
            import httpx

            payload: dict[str, Any] = {"content": text[:2000]}
            if anima_name:
                payload["username"] = anima_name
                try:
                    from core.tools._anima_icon_url import resolve_anima_icon_url

                    avatar = resolve_anima_icon_url(anima_name)
                    if avatar:
                        payload["avatar_url"] = avatar
                except Exception:
                    logger.debug("Failed to resolve avatar for %s", anima_name, exc_info=True)
            if components:
                payload["components"] = components

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(webhook_url, json=payload, params={"wait": "true"})
                resp.raise_for_status()
                data = resp.json()
                msg_id = str(data.get("id", "")) if isinstance(data, dict) else ""
                logger.info("Discord notification sent via webhook")
                if interaction is not None and msg_id:
                    await _persist_discord_msg_id(interaction.callback_id, msg_id)
                return "discord: webhook sent"
        except Exception as exc:
            logger.exception("Discord webhook notification failed")
            return f"discord: ERROR - {exc}"
