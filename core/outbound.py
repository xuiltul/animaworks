# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Outbound message router for unified send_message delivery.

Resolves recipient names to internal Anima inboxes or external platforms
(Slack, Chatwork) and dispatches accordingly. Designed to be robust against
sloppy recipient specifications from Animas.
"""

import json
import logging
import re
from dataclasses import dataclass

from core.config.models import ExternalMessagingConfig, UserAliasConfig
from core.exceptions import DeliveryError, MessagingError, RecipientNotFoundError  # noqa: F401

logger = logging.getLogger("animaworks.outbound")

# Slack user ID pattern: U followed by alphanumeric (typically 10-11 chars)
_SLACK_USER_ID_RE = re.compile(r"^U[A-Z0-9]{8,}$", re.IGNORECASE)


# ── Resolution result ────────────────────────────────────


@dataclass
class ResolvedRecipient:
    """Result of recipient resolution."""

    is_internal: bool
    name: str  # original or normalized name
    channel: str = ""  # "slack" | "chatwork" | "discord" | "" (internal)
    slack_user_id: str = ""
    chatwork_room_id: str = ""
    discord_user_id: str = ""
    alias_used: str = ""  # the alias that matched (if any)


# ── Recipient resolver ───────────────────────────────────


def resolve_recipient(
    to: str,
    known_animas: set[str],
    config: ExternalMessagingConfig,
) -> ResolvedRecipient:
    """Resolve a recipient name to an internal Anima or external target.

    Resolution priority:
      1. Exact match against known Anima names (case-sensitive) → internal
      2. Case-insensitive match against user_aliases → external via preferred_channel
      3. ``slack:USERID`` prefix → Slack direct
      4. ``chatwork:ROOMID`` prefix → Chatwork direct
      5. Bare Slack user ID pattern (U + alphanumeric) → Slack direct
      6. Case-insensitive match against known Anima names → internal
      7. Unknown → raises ValueError
    """
    raw = to.strip()
    if not raw:
        raise RecipientNotFoundError("Recipient name is empty")

    # 1. Exact Anima name match (case-sensitive, fastest path)
    if raw in known_animas:
        return ResolvedRecipient(is_internal=True, name=raw)

    # 2. User alias match (case-insensitive)
    lower = raw.lower()
    for alias, alias_cfg in config.user_aliases.items():
        if alias.lower() == lower:
            return _resolve_from_alias(alias, alias_cfg, config.preferred_channel)

    # 3. slack: prefix
    if lower.startswith("slack:"):
        uid = raw[6:].strip().upper()
        if uid:
            return ResolvedRecipient(
                is_internal=False,
                name=raw,
                channel="slack",
                slack_user_id=uid,
            )

    # 4. chatwork: prefix
    if lower.startswith("chatwork:"):
        room = raw[9:].strip()
        if room:
            return ResolvedRecipient(
                is_internal=False,
                name=raw,
                channel="chatwork",
                chatwork_room_id=room,
            )

    # 4b. discord: prefix (snowflake user ID)
    if lower.startswith("discord:"):
        uid = raw[8:].strip()
        if uid:
            return ResolvedRecipient(
                is_internal=False,
                name=raw,
                channel="discord",
                discord_user_id=uid,
            )

    # 5. Bare Slack user ID pattern
    if _SLACK_USER_ID_RE.match(raw):
        return ResolvedRecipient(
            is_internal=False,
            name=raw,
            channel="slack",
            slack_user_id=raw.upper(),
        )

    # 6. Case-insensitive Anima name fallback
    for anima in known_animas:
        if anima.lower() == lower:
            return ResolvedRecipient(is_internal=True, name=anima)

    alias_names = list(config.user_aliases.keys())
    raise RecipientNotFoundError(
        f"Unknown recipient '{raw}'. "
        f"Known animas: {sorted(known_animas)}. "
        f"User aliases: {alias_names}. "
        f"Use send_message(to='user') for human contact."
    )


def _resolve_from_alias(
    alias: str,
    alias_cfg: UserAliasConfig,
    preferred_channel: str,
) -> ResolvedRecipient:
    """Build a ResolvedRecipient from a user alias config."""
    channel = preferred_channel

    # Validate that the preferred channel has contact info
    if channel == "slack" and alias_cfg.slack_user_id:
        return ResolvedRecipient(
            is_internal=False,
            name=alias,
            channel="slack",
            slack_user_id=alias_cfg.slack_user_id,
            alias_used=alias,
        )
    if channel == "chatwork" and alias_cfg.chatwork_room_id:
        return ResolvedRecipient(
            is_internal=False,
            name=alias,
            channel="chatwork",
            chatwork_room_id=alias_cfg.chatwork_room_id,
            alias_used=alias,
        )
    if channel == "discord" and alias_cfg.discord_user_id:
        return ResolvedRecipient(
            is_internal=False,
            name=alias,
            channel="discord",
            discord_user_id=alias_cfg.discord_user_id,
            alias_used=alias,
        )

    # Preferred channel not available — fallback to any available channel
    if alias_cfg.slack_user_id:
        return ResolvedRecipient(
            is_internal=False,
            name=alias,
            channel="slack",
            slack_user_id=alias_cfg.slack_user_id,
            alias_used=alias,
        )
    if alias_cfg.chatwork_room_id:
        return ResolvedRecipient(
            is_internal=False,
            name=alias,
            channel="chatwork",
            chatwork_room_id=alias_cfg.chatwork_room_id,
            alias_used=alias,
        )
    if alias_cfg.discord_user_id:
        return ResolvedRecipient(
            is_internal=False,
            name=alias,
            channel="discord",
            discord_user_id=alias_cfg.discord_user_id,
            alias_used=alias,
        )

    raise RecipientNotFoundError(
        f"User alias '{alias}' has no contact info for any channel. "
        f"Configure discord_user_id, slack_user_id, or chatwork_room_id in external_messaging.user_aliases."
    )


# ── External dispatch ────────────────────────────────────


def send_external(
    resolved: ResolvedRecipient,
    content: str,
    sender_name: str = "",
    anima_name: str = "",
) -> str:
    """Send a message to an external platform.

    Returns a status string for tool result display.
    Tries the resolved channel first, then falls back to alternatives.
    """
    channels_to_try = _build_channel_order(resolved)
    if not channels_to_try:
        return json.dumps(
            {
                "status": "error",
                "error_type": "NoChannelConfigured",
                "message": f"No external messaging channel configured for '{resolved.name}'. "
                f"Set up slack, chatwork, or discord in config.json external_messaging.",
            },
            ensure_ascii=False,
        )

    last_error = ""
    for channel in channels_to_try:
        try:
            if channel == "slack":
                return _send_via_slack(resolved.slack_user_id, content, sender_name, anima_name)
            elif channel == "chatwork":
                return _send_via_chatwork(resolved.chatwork_room_id, content, sender_name)
            elif channel == "discord":
                return _send_via_discord(resolved.discord_user_id, content, sender_name, anima_name)
            else:
                last_error = f"Unknown channel: {channel}"
        except Exception as e:
            last_error = f"{channel}: {e}"
            logger.warning(
                "External send failed via %s: %s",
                channel,
                e,
            )

    return json.dumps(
        {
            "status": "error",
            "error_type": "DeliveryFailed",
            "message": f"Message delivery failed to '{resolved.name}'. Last error: {last_error}",
        },
        ensure_ascii=False,
    )


def _build_channel_order(resolved: ResolvedRecipient) -> list[str]:
    """Build ordered list of channels to try."""
    channels = [resolved.channel]
    # Add fallback channels (Slack > Chatwork > Discord to preserve existing behavior)
    if resolved.slack_user_id and "slack" not in channels:
        channels.append("slack")
    if resolved.chatwork_room_id and "chatwork" not in channels:
        channels.append("chatwork")
    if resolved.discord_user_id and "discord" not in channels:
        channels.append("discord")
    return channels


def _resolve_outbound_icon(anima_name: str) -> str:
    """Resolve icon_url for outbound Slack messages (see :mod:`core.tools._anima_icon_url`)."""
    if not anima_name:
        return ""
    try:
        from core.tools._anima_icon_url import resolve_anima_icon_url

        return resolve_anima_icon_url(anima_name, channel_config=None)
    except Exception:
        logger.debug("resolve_anima_icon_url failed for outbound", exc_info=True)
    return ""


def _send_via_slack(user_id: str, content: str, sender_name: str, anima_name: str = "") -> str:
    """Send a DM via Slack API."""
    from core.tools._base import resolve_env_style_credential
    from core.tools.slack import SlackClient, md_to_slack_mrkdwn

    token = None
    if anima_name:
        per_anima_key = f"SLACK_BOT_TOKEN__{anima_name}"
        token = resolve_env_style_credential(per_anima_key)

    prefix = f"[{sender_name}] " if sender_name and not token else ""
    text = f"{prefix}{content}"
    text = md_to_slack_mrkdwn(text)

    username = anima_name or sender_name or ""
    icon_url = _resolve_outbound_icon(anima_name)

    client = SlackClient(token=token)
    response = client.post_message(user_id, text, username=username, icon_url=icon_url)
    ts = response.get("ts", "")
    _channel = response.get("channel", user_id)

    logger.info(
        "External message sent via slack: user=%s ts=%s sender=%s per_anima=%s",
        user_id,
        ts,
        sender_name,
        bool(token),
    )
    return json.dumps(
        {
            "status": "sent",
            "channel": "slack",
            "recipient": user_id,
            "message": f"Message sent via Slack DM to {user_id}",
        },
        ensure_ascii=False,
    )


def _send_via_discord(user_id: str, content: str, sender_name: str, anima_name: str = "") -> str:
    """Send a DM via Discord API using the bot token."""
    from core.tools._base import get_credential
    from core.tools._discord_client import DiscordClient
    from core.tools._discord_markdown import md_to_discord

    token = get_credential("discord", "discord", env_var="DISCORD_BOT_TOKEN")
    client = DiscordClient(token=token)

    # Open DM channel
    dm = client.create_dm(user_id)
    dm_channel_id = str(dm.get("id", ""))
    if not dm_channel_id:
        raise DeliveryError(f"Failed to open DM channel with Discord user {user_id}")

    prefix = f"[{sender_name}] " if sender_name and not anima_name else ""
    text = md_to_discord(f"{prefix}{content}")

    # Send via bot (DMs don't support webhooks with custom identity)
    result = client.send_message(dm_channel_id, text)
    msg_id = result.get("id", "")

    logger.info(
        "External message sent via discord: user=%s msg_id=%s sender=%s",
        user_id,
        msg_id,
        sender_name,
    )
    return json.dumps(
        {
            "status": "sent",
            "channel": "discord",
            "recipient": user_id,
            "message": f"Message sent via Discord DM to {user_id}",
        },
        ensure_ascii=False,
    )


def _send_via_chatwork(room_id: str, content: str, sender_name: str) -> str:
    """Send a message via Chatwork API."""
    from core.tools._base import get_credential
    from core.tools.chatwork import ChatworkClient, md_to_chatwork

    prefix = f"[{sender_name}] " if sender_name else ""
    body = f"{prefix}{content}"
    body = md_to_chatwork(body)

    write_token = get_credential("chatwork_write", "chatwork", env_var="CHATWORK_API_TOKEN_WRITE")
    client = ChatworkClient(api_token=write_token)
    response = client.post_message(room_id, body)
    message_id = response.get("message_id", "") if response else ""

    logger.info(
        "External message sent via chatwork: room=%s msg_id=%s sender=%s",
        room_id,
        message_id,
        sender_name,
    )
    return json.dumps(
        {
            "status": "sent",
            "channel": "chatwork",
            "recipient": room_id,
            "message": f"Message sent via Chatwork to room {room_id}",
        },
        ensure_ascii=False,
    )
