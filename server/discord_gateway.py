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
from typing import TYPE_CHECKING, Any

from core.config.models import load_config
from core.i18n import t
from core.messenger import Messenger
from core.paths import get_data_dir, get_shared_dir
from core.tools._base import get_credential
from core.tools._discord_markdown import clean_discord_markup

if TYPE_CHECKING:
    import discord

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


def _route_to_board(
    channel_id: str,
    text: str,
    user_name: str,
    *,
    message_id: str = "",
    board_mapping: dict[str, str] | None = None,
) -> None:
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
        if board_mapping is None:
            board_mapping = load_config().external_messaging.discord.board_mapping
        board_name = board_mapping.get(channel_id)
        if not board_name:
            return
        messenger = Messenger(get_shared_dir(), user_name or "discord")
        messenger.post_channel(board_name, text, source="discord", from_name=user_name or "discord")
    except Exception:
        logger.debug("Board routing failed for channel %s", channel_id, exc_info=True)


# ── Annotation builder ───────────────────────────────────────


def _build_discord_annotation(is_dm: bool, has_mention: bool) -> str:
    if is_dm:
        return "[discord:DM]\n"
    if has_mention:
        return f"[discord:channel — {t('discord.annotation_mentioned')}]\n"
    return f"[discord:channel — {t('discord.annotation_no_mention')}]\n"


# ── Thread context ───────────────────────────────────────────

_THREAD_CTX_SUMMARY_LIMIT = 150


async def _fetch_thread_context(
    channel: Any,
    reference: Any,
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
        self._client: Any = None  # discord.Client (lazy import)
        self._bot_user_id: int = 0
        self._anima_name_re: re.Pattern[str] | None = None
        self._known_anima_names: set[str] = set()
        self._alias_to_canonical: dict[str, str] = {}  # lowercase alias/name → canonical
        self._webhook_names: set[str] = set()
        self._started = False

    @property
    def client(self) -> Any:
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

        try:
            import discord as _discord
        except ImportError:
            logger.error("discord.py is not installed — run: pip install 'animaworks[discord]'")
            return

        self._build_anima_patterns()

        intents = _discord.Intents(
            guilds=True,
            guild_messages=True,
            dm_messages=True,
            message_content=True,
            members=False,
        )

        client = _discord.Client(intents=intents)
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
        async def on_message(message: Any) -> None:
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

    async def _run_client(self, client: Any, token: str) -> None:
        """Run client.start in a way that doesn't block the event loop."""
        try:
            await client.start(token)
        except Exception as exc:
            if type(exc).__name__ == "LoginFailure":
                logger.error("Discord login failed — check DISCORD_BOT_TOKEN")
            else:
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
        """Build regex pattern and alias→canonical mapping from config."""
        try:
            cfg = load_config()
            anima_names: set[str] = set()
            alias_map: dict[str, str] = {}
            for name in cfg.animas:
                anima_names.add(name)
                alias_map[name.lower()] = name
                anima_cfg = cfg.animas[name]
                for alias in anima_cfg.aliases:
                    anima_names.add(alias)
                    alias_map[alias.lower()] = name
            self._known_anima_names = anima_names
            self._alias_to_canonical = alias_map
            if anima_names:
                escaped = [re.escape(n) for n in sorted(anima_names, key=len, reverse=True)]
                self._anima_name_re = re.compile(
                    r"(?:^|\b|(?<=[^a-zA-Z0-9]))(" + "|".join(escaped) + r")(?:\b|(?=[^a-zA-Z0-9])|$)",
                    re.IGNORECASE,
                )
            else:
                self._anima_name_re = None
        except Exception:
            logger.debug("Failed to build Anima name patterns", exc_info=True)

    def _resolve_canonical_name(self, matched: str) -> str | None:
        """Resolve a matched name (possibly alias) to canonical Anima name.

        Uses the cached ``_alias_to_canonical`` mapping built by
        ``_build_anima_patterns()`` — no config reload needed.
        """
        return self._alias_to_canonical.get(matched.lower())

    def _detect_target_anima(
        self,
        text: str,
        channel_id: str,
        discord_cfg: Any,
    ) -> str | None:
        """Detect which Anima a message is targeting.

        Returns canonical anima name or None.
        *discord_cfg* is the pre-loaded ``ExternalMessagingChannelConfig``
        (required — caller must supply it to avoid repeated config loads).
        """
        # 1. Anima name in message text
        if self._anima_name_re and text:
            m = self._anima_name_re.search(text)
            if m:
                canonical = self._resolve_canonical_name(m.group(1))
                if canonical:
                    return canonical

        # 2. Channel-member config (if only one member, route to them)
        members = discord_cfg.channel_members.get(channel_id, [])
        if len(members) == 1:
            return members[0]

        return None

    @staticmethod
    def _is_anima_in_channel(anima_name: str, channel_id: str, discord_cfg: Any) -> bool:
        """Check if an Anima is configured as a member of a channel."""
        members = discord_cfg.channel_members.get(channel_id, [])
        if not members:
            return True
        return anima_name in members

    async def _handle_message(self, message: Any) -> None:
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

        # Load config once per message for routing decisions
        try:
            cfg = load_config()
            discord_cfg = cfg.external_messaging.discord
        except Exception:
            logger.debug("Failed to load config for Discord routing", exc_info=True)
            return

        # Determine target Anima
        target_anima: str | None = None

        # 1. Thread reply mapping — route replies to the Anima that sent the parent
        if reference_id:
            try:
                from core.discord_webhooks import get_webhook_manager

                thread_anima = get_webhook_manager().lookup_thread_anima(reference_id)
                if thread_anima:
                    target_anima = thread_anima
            except Exception:
                logger.debug("Thread map lookup failed", exc_info=True)

        # 2. Text/name detection and channel member routing
        if target_anima is None:
            target_anima = self._detect_target_anima(cleaned_text, channel_id, discord_cfg)

        if target_anima is None and not is_dm:
            if bot_mentioned:
                target_anima = discord_cfg.default_anima or None

            # No mention, no name detected: route to channel lead
            if target_anima is None:
                members = discord_cfg.channel_members.get(channel_id, [])
                if members:
                    target_anima = members[0]
                elif discord_cfg.default_anima and self._is_anima_in_channel(
                    discord_cfg.default_anima, channel_id, discord_cfg
                ):
                    target_anima = discord_cfg.default_anima

        # Enforce channel membership
        if target_anima and not is_dm and not self._is_anima_in_channel(target_anima, channel_id, discord_cfg):
            logger.info(
                "Discord routing: '%s' not a member of channel %s (#%s) — dropping",
                target_anima,
                channel_id,
                ch_name,
            )
            target_anima = None

        logger.info(
            "Discord routing: channel=#%s (%s) is_dm=%s bot_mentioned=%s -> target=%s",
            ch_name,
            channel_id,
            is_dm,
            bot_mentioned,
            target_anima,
        )

        # Board routing (always, regardless of target)
        if not is_dm:
            _route_to_board(
                channel_id, cleaned_text, author_display,
                message_id=msg_id, board_mapping=discord_cfg.board_mapping,
            )

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

                messenger = Messenger(get_shared_dir(), target_anima)
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
