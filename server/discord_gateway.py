from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Discord Gateway integration for real-time message reception.

Uses a single Gateway Bot + per-channel webhooks.  All Animas share one
bot connection; routing is based on Anima name detection in message text
and channel-member configuration.
"""

import asyncio
import collections
import logging
import re
import threading
import time
from typing import Any

import discord

from core.config.models import load_config
from core.messenger import Messenger
from core.paths import get_data_dir
from core.tools._base import get_credential
from core.tools._discord_markdown import clean_discord_markup

logger = logging.getLogger("animaworks.discord_gateway")

# ── Dedup ────────────────────────────────────────────────────

_DEDUP_TTL_SEC = 10
_dedup_lock = threading.Lock()
_recent_ids: collections.OrderedDict[str, float] = collections.OrderedDict()


def _is_duplicate_id(message_id: str) -> bool:
    """Return True if *message_id* was already processed."""
    now = time.monotonic()
    with _dedup_lock:
        while _recent_ids and next(iter(_recent_ids.values())) < now - _DEDUP_TTL_SEC:
            _recent_ids.popitem(last=False)
        if message_id in _recent_ids:
            return True
        _recent_ids[message_id] = now
        return False


# ── User name cache ──────────────────────────────────────────

_USER_NAME_CACHE_MAX = 500
_name_cache_lock = threading.Lock()
_user_name_cache: dict[str, str] = {}


def _cache_user_name(uid: str, name: str) -> None:
    with _name_cache_lock:
        if len(_user_name_cache) >= _USER_NAME_CACHE_MAX and uid not in _user_name_cache:
            try:
                _user_name_cache.pop(next(iter(_user_name_cache)))
            except StopIteration:
                pass
        _user_name_cache[uid] = name


def _get_cached_user_name(uid: str) -> str | None:
    with _name_cache_lock:
        return _user_name_cache.get(uid)


# ── Board routing dedup ──────────────────────────────────────

_board_dedup_lock = threading.Lock()
_board_dedup_ids: collections.OrderedDict[str, float] = collections.OrderedDict()
_BOARD_DEDUP_TTL_SEC = 10


def _route_to_board(channel_id: str, text: str, user_name: str, *, message_id: str = "") -> None:
    """Post a Discord message to the mapped AnimaWorks board (if any)."""
    if message_id:
        now = time.monotonic()
        with _board_dedup_lock:
            while _board_dedup_ids and next(iter(_board_dedup_ids.values())) < now - _BOARD_DEDUP_TTL_SEC:
                _board_dedup_ids.popitem(last=False)
            if message_id in _board_dedup_ids:
                return
            _board_dedup_ids[message_id] = now

    try:
        cfg = load_config()
        board_name = cfg.external_messaging.discord.board_mapping.get(channel_id)
        if not board_name:
            return
        shared_dir = get_data_dir() / "shared"
        messenger = Messenger(shared_dir, user_name or "discord")
        messenger.post_channel(board_name, text, source="discord", from_name=user_name or "discord")
    except Exception:
        logger.debug("Board routing failed for channel %s", channel_id, exc_info=True)


# ── Annotation builder ───────────────────────────────────────


def _build_discord_annotation(is_dm: bool, has_mention: bool) -> str:
    if is_dm:
        return "[discord:DM]\n"
    if has_mention:
        return "[discord:channel — あなたがメンションされています]\n"
    return "[discord:channel — あなたへの直接メンションはありません]\n"


# ── Thread context ───────────────────────────────────────────

_THREAD_CTX_SUMMARY_LIMIT = 150


async def _fetch_thread_context(
    channel: discord.TextChannel | discord.DMChannel,
    reference: discord.MessageReference,
) -> str:
    """Fetch Discord thread context for a reply message."""
    if reference.message_id is None:
        return ""
    try:
        parent = await channel.fetch_message(reference.message_id)
        parent_user = parent.author.display_name or str(parent.author)
        parent_text = (parent.content or "").replace("\n", " ")[:_THREAD_CTX_SUMMARY_LIMIT]
        parent_text = clean_discord_markup(parent_text)
        lines = [
            "[Thread context — this message is a reply in a Discord thread]",
            f"  @{parent_user}: {parent_text}",
            "[/Thread context]",
            "",
        ]
        return "\n".join(lines)
    except Exception:
        logger.warning("Failed to fetch Discord thread context", exc_info=True)
        return ""


# ── Gateway Manager ──────────────────────────────────────────


class DiscordGatewayManager:
    """Manages a single Discord Gateway connection for all Animas.

    Inbound messages are routed to Animas based on:
    1. Thread reply mapping (previous conversation context)
    2. Anima name detection in message text
    3. Channel-member configuration
    4. Default anima fallback
    """

    def __init__(self) -> None:
        self._client: discord.Client | None = None
        self._bot_user_id: int = 0
        self._anima_name_re: re.Pattern[str] | None = None
        self._known_anima_names: set[str] = set()
        self._webhook_names: set[str] = set()
        self._started = False

    @property
    def client(self) -> discord.Client | None:
        return self._client

    async def start(self) -> None:
        """Start the Discord Gateway connection if enabled."""
        config = load_config()
        discord_config = config.external_messaging.discord
        if not discord_config.enabled:
            logger.info("Discord Gateway is disabled")
            return

        try:
            token = get_credential("discord", "discord", env_var="DISCORD_BOT_TOKEN")
        except Exception:
            logger.error("DISCORD_BOT_TOKEN not configured — Discord Gateway cannot start")
            return

        self._build_anima_patterns()

        intents = discord.Intents(
            guilds=True,
            guild_messages=True,
            dm_messages=True,
            message_content=True,
            members=False,
        )

        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready() -> None:
            if client.user:
                self._bot_user_id = client.user.id
                logger.info(
                    "Discord Gateway connected: %s (id=%s)",
                    client.user.name,
                    client.user.id,
                )

        @client.event
        async def on_message(message: discord.Message) -> None:
            await self._handle_message(message)

        # Start in background task (client.start is blocking)
        asyncio.create_task(self._run_client(client, token))

        # Wait for ready with timeout
        for _ in range(60):
            if self._bot_user_id:
                break
            await asyncio.sleep(0.5)

        if not self._bot_user_id:
            logger.error("Discord Gateway did not become ready within 30s")
            return

        self._started = True
        logger.info("Discord Gateway started")

    async def _run_client(self, client: discord.Client, token: str) -> None:
        """Run client.start in a way that doesn't block the event loop."""
        try:
            await client.start(token)
        except discord.LoginFailure:
            logger.error("Discord login failed — check DISCORD_BOT_TOKEN")
        except Exception:
            logger.exception("Discord Gateway connection error")

    async def stop(self) -> None:
        """Gracefully close the Discord Gateway connection."""
        if self._client and not self._client.is_closed():
            await self._client.close()
            logger.info("Discord Gateway stopped")
        self._started = False

    def reload(self) -> None:
        """Rebuild Anima name patterns from current config."""
        self._build_anima_patterns()
        logger.info("Discord Gateway patterns reloaded")

    async def health_check(self) -> dict[str, Any]:
        """Return gateway health status."""
        if not self._client or self._client.is_closed():
            return {"status": "disconnected"}
        latency = self._client.latency
        return {
            "status": "connected",
            "latency_ms": round(latency * 1000, 1) if latency != float("inf") else None,
            "bot_user_id": str(self._bot_user_id),
            "guilds": len(self._client.guilds),
        }

    # ── Internal ─────────────────────────────────────────────

    def _build_anima_patterns(self) -> None:
        """Build regex pattern for Anima name detection from config."""
        try:
            cfg = load_config()
            anima_names: set[str] = set()
            for name in cfg.animas:
                anima_names.add(name)
                anima_cfg = cfg.animas[name]
                for alias in anima_cfg.aliases:
                    anima_names.add(alias)
            self._known_anima_names = anima_names
            if anima_names:
                escaped = [re.escape(n) for n in sorted(anima_names, key=len, reverse=True)]
                self._anima_name_re = re.compile(
                    r"(?:^|\b)(" + "|".join(escaped) + r")(?:\b|[,、。.!?！？\s])",
                    re.IGNORECASE,
                )
            else:
                self._anima_name_re = None
        except Exception:
            logger.debug("Failed to build Anima name patterns", exc_info=True)

    def _resolve_canonical_name(self, matched: str) -> str | None:
        """Resolve a matched name (possibly alias) to canonical Anima name."""
        try:
            cfg = load_config()
            low = matched.lower()
            # Direct match
            for name in cfg.animas:
                if name.lower() == low:
                    return name
            # Alias match
            for name, acfg in cfg.animas.items():
                for alias in acfg.aliases:
                    if alias.lower() == low:
                        return name
        except Exception:
            pass
        return None

    def _detect_target_anima(
        self,
        text: str,
        channel_id: str,
    ) -> str | None:
        """Detect which Anima a message is targeting.

        Returns canonical anima name or None.
        """
        # 1. Anima name in message text
        if self._anima_name_re and text:
            m = self._anima_name_re.search(text)
            if m:
                canonical = self._resolve_canonical_name(m.group(1))
                if canonical:
                    return canonical

        # 2. Channel-member config (if only one member, route to them)
        try:
            cfg = load_config()
            members = cfg.external_messaging.discord.channel_members.get(channel_id, [])
            if len(members) == 1:
                return members[0]
        except Exception:
            pass

        # 3. Default anima
        try:
            cfg = load_config()
            default = cfg.external_messaging.discord.default_anima
            if default:
                return default
        except Exception:
            pass

        return None

    def _is_anima_in_channel(self, anima_name: str, channel_id: str) -> bool:
        """Check if an Anima is configured as a member of a channel."""
        try:
            cfg = load_config()
            members = cfg.external_messaging.discord.channel_members.get(channel_id, [])
            if not members:
                # No membership config → allow all (backward-compatible)
                return True
            return anima_name in members
        except Exception:
            return True

    async def _handle_message(self, message: discord.Message) -> None:
        """Core message handler for all Discord events."""
        # Ignore own messages
        if message.author.id == self._bot_user_id:
            return

        # Ignore webhook messages from AnimaWorks (echo prevention)
        if message.webhook_id is not None:
            author_name = message.author.display_name or message.author.name
            if author_name.lower() in {n.lower() for n in self._known_anima_names}:
                return

        # Dedup
        msg_id = str(message.id)
        if _is_duplicate_id(msg_id):
            return

        # Cache author name
        author_display = message.author.display_name or message.author.name
        _cache_user_name(str(message.author.id), author_display)

        # Build user name cache from mentions
        mention_cache: dict[str, str] = {}
        for user in message.mentions:
            mention_cache[str(user.id)] = user.display_name or user.name
            _cache_user_name(str(user.id), user.display_name or user.name)

        # Clean content
        with _name_cache_lock:
            name_snapshot = dict(_user_name_cache)
        cleaned_text = clean_discord_markup(message.content or "", cache=name_snapshot)

        channel_id = str(message.channel.id)
        ch_name = getattr(message.channel, "name", "") or ""
        is_dm = message.guild is None or ch_name.startswith("dm-")

        # Bot mentioned?
        bot_mentioned = any(u.id == self._bot_user_id for u in message.mentions)

        # Thread context
        thread_ctx = ""
        reference_id: str | None = None
        if message.reference and message.reference.message_id:
            reference_id = str(message.reference.message_id)
            thread_ctx = await _fetch_thread_context(message.channel, message.reference)

        # Determine target Anima
        target_anima: str | None = None

        if is_dm:
            # DM: detect name in text or use default
            target_anima = self._detect_target_anima(cleaned_text, channel_id)
        else:
            # Guild channel
            target_anima = self._detect_target_anima(cleaned_text, channel_id)

            # If bot mentioned but no specific anima found, use default
            if target_anima is None and bot_mentioned:
                try:
                    cfg = load_config()
                    target_anima = cfg.external_messaging.discord.default_anima or None
                except Exception:
                    pass

            # No mention, no name detected: route to channel lead
            # (first member in channel_members is the responsible Anima)
            if target_anima is None:
                try:
                    cfg = load_config()
                    members = cfg.external_messaging.discord.channel_members.get(channel_id, [])
                    if members:
                        target_anima = members[0]
                    else:
                        # Fallback to default_anima
                        default = cfg.external_messaging.discord.default_anima
                        if default and self._is_anima_in_channel(default, channel_id):
                            target_anima = default
                except Exception:
                    pass

        # Enforce channel membership
        if target_anima and not is_dm and not self._is_anima_in_channel(target_anima, channel_id):
            logger.debug(
                "Anima '%s' not a member of channel %s — ignoring",
                target_anima,
                channel_id,
            )
            target_anima = None

        # Board routing (always, regardless of target)
        if not is_dm:
            _route_to_board(channel_id, cleaned_text, author_display, message_id=msg_id)

        # Deliver to Anima inbox if we have a target
        if target_anima:
            has_mention = bot_mentioned or (
                self._anima_name_re is not None and self._anima_name_re.search(cleaned_text) is not None
            )
            annotation = _build_discord_annotation(is_dm, has_mention)
            intent = "question"

            full_content = annotation + thread_ctx + cleaned_text

            try:
                data_dir = get_data_dir()
                anima_dir = data_dir / "animas" / target_anima
                if not anima_dir.is_dir():
                    logger.warning("Anima directory not found: %s", target_anima)
                    return

                shared_dir = data_dir / "shared"
                messenger = Messenger(shared_dir, target_anima)
                messenger.receive_external(
                    content=full_content,
                    source="discord",
                    source_message_id=msg_id,
                    external_user_id=str(message.author.id),
                    external_channel_id=channel_id,
                    external_thread_ts=reference_id or "",
                    intent=intent,
                )

                logger.info(
                    "Discord message routed: %s -> %s (channel=%s, intent=%s)",
                    author_display,
                    target_anima,
                    channel_id,
                    intent or "none",
                )
            except Exception:
                logger.exception(
                    "Failed to deliver Discord message to %s",
                    target_anima,
                )
