from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Discord channel <-> AnimaWorks board bidirectional auto-sync.

**Forward sync (Discord -> AnimaWorks):**
Lists all text channels in the configured guild.  For every channel,
a corresponding AnimaWorks board is created if it doesn't already exist.

**Reverse sync (AnimaWorks -> Discord):**
For every AnimaWorks board that has no corresponding Discord channel,
a new text channel is created via the Discord REST API.

**Per-Anima DM channels:**
For each enabled Anima, a ``#dm-{name}`` channel is created under a
"DM" category.  ``channel_members`` is auto-configured with only that
Anima, creating a dedicated 1-on-1 channel.

The ``board_mapping`` (channel_id -> board_name) and ``channel_members``
are persisted in config.json.
"""

import asyncio
import json
import logging
import re
from typing import Any

from core.config.models import load_config, save_config
from core.messenger import ChannelMeta, load_channel_meta, save_channel_meta
from core.paths import get_shared_dir
from core.tools._base import get_credential
from core.tools._discord_client import DiscordAPIError, DiscordClient

logger = logging.getLogger("animaworks.discord_channel_sync")

# Discord channel types
_CHANNEL_TYPE_TEXT = 0
_CHANNEL_TYPE_CATEGORY = 4

_DM_CATEGORY_NAME = "DM"
_DM_CHANNEL_PREFIX = "dm-"


# ── Name sanitization ────────────────────────────────────────

_RE_INVALID_BOARD = re.compile(r"[^a-z0-9_-]")


def _channel_to_board_name(name: str) -> str:
    """Sanitize a Discord channel name to an AnimaWorks board name."""
    return _RE_INVALID_BOARD.sub("-", name.lower().strip()).strip("-") or "unnamed"


def _board_to_channel_name(board: str) -> str:
    """Convert an AnimaWorks board name to a Discord channel name."""
    return _RE_INVALID_BOARD.sub("-", board.lower().strip()).strip("-") or "unnamed"


class DiscordChannelSync:
    """Bidirectional sync between Discord channels and AnimaWorks boards."""

    def __init__(self) -> None:
        self._client: DiscordClient | None = None

    def _ensure_client(self) -> DiscordClient:
        if self._client is None:
            token = get_credential("discord", "discord", env_var="DISCORD_BOT_TOKEN")
            self._client = DiscordClient(token=token)
        return self._client

    async def sync(self, gateway_manager: Any = None) -> dict[str, Any]:
        """Run full sync cycle: forward + reverse + DM channels.

        All Discord REST calls are synchronous (httpx), so the heavy lifting
        is offloaded to a thread to avoid blocking the asyncio event loop.

        Returns a summary dict with counts.
        """
        cfg = load_config()
        discord_cfg = cfg.external_messaging.discord
        if not discord_cfg.enabled:
            return {"status": "disabled"}

        guild_id = discord_cfg.guild_id
        if not guild_id:
            logger.warning("Discord channel sync: guild_id not configured")
            return {"status": "no_guild_id"}

        return await asyncio.to_thread(self._sync_blocking, cfg, discord_cfg, guild_id)

    def _sync_blocking(
        self,
        cfg: Any,
        discord_cfg: Any,
        guild_id: str,
    ) -> dict[str, Any]:
        """Synchronous sync body — runs in a worker thread."""
        client = self._ensure_client()
        shared_dir = get_shared_dir()
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)

        try:
            all_channels = client.get_guild_channels(guild_id)
        except DiscordAPIError:
            logger.exception("Failed to list Discord guild channels")
            return {"status": "error"}

        text_channels = [ch for ch in all_channels if ch.get("type") == _CHANNEL_TYPE_TEXT]

        board_mapping = dict(discord_cfg.board_mapping)
        channel_members = dict(discord_cfg.channel_members)

        # ── Phase 1: Forward sync (Discord -> AnimaWorks boards) ──
        forward_created = 0
        for ch in text_channels:
            ch_id = str(ch["id"])
            ch_name = ch.get("name", "")

            if ch_name.startswith(_DM_CHANNEL_PREFIX):
                continue

            if ch_id not in board_mapping:
                board_name = _channel_to_board_name(ch_name)
                board_mapping[ch_id] = board_name

            board_name = board_mapping[ch_id]
            board_file = channels_dir / f"{board_name}.jsonl"

            if not board_file.exists():
                board_file.touch()
                forward_created += 1
                logger.info("Created board '%s' from Discord channel #%s", board_name, ch_name)

            meta = load_channel_meta(shared_dir, board_name)
            if meta is None:
                meta = ChannelMeta(members=[])
            if not meta.description and ch.get("topic"):
                meta.description = ch["topic"]
            save_channel_meta(shared_dir, board_name, meta)

        # ── Phase 2: Reverse sync (AnimaWorks boards -> Discord) ──
        reverse_created = 0
        existing_names = {ch.get("name", "").lower() for ch in text_channels}
        mapped_boards = set(board_mapping.values())

        for board_file in channels_dir.glob("*.jsonl"):
            board_name = board_file.stem
            if board_name.startswith(".") or board_name in mapped_boards:
                continue

            channel_name = _board_to_channel_name(board_name)
            if channel_name in existing_names:
                for ch in text_channels:
                    if ch.get("name", "").lower() == channel_name:
                        board_mapping[str(ch["id"])] = board_name
                        break
                continue

            try:
                result = client.create_channel(guild_id, channel_name)
                ch_id = str(result["id"])
                board_mapping[ch_id] = board_name
                reverse_created += 1
                logger.info("Created Discord channel #%s for board '%s'", channel_name, board_name)
            except DiscordAPIError:
                logger.warning("Failed to create Discord channel for board '%s'", board_name, exc_info=True)

        # ── Phase 3: Per-Anima DM channels ──
        dm_created = 0
        dm_category_id = self._ensure_dm_category(client, guild_id, all_channels)

        enabled_animas = []
        for name, _anima_cfg in cfg.animas.items():
            try:
                from core.paths import get_animas_dir

                status_file = get_animas_dir() / name / "status.json"
                if status_file.is_file():
                    status = json.loads(status_file.read_text(encoding="utf-8"))
                    if status.get("enabled", True):
                        enabled_animas.append(name)
            except Exception:
                enabled_animas.append(name)

        try:
            all_channels = client.get_guild_channels(guild_id)
            text_channels = [ch for ch in all_channels if ch.get("type") == _CHANNEL_TYPE_TEXT]
        except DiscordAPIError:
            pass

        existing_dm_names = {ch.get("name", "").lower() for ch in text_channels}

        for anima_name in enabled_animas:
            dm_channel_name = f"{_DM_CHANNEL_PREFIX}{anima_name}"
            if dm_channel_name.lower() in existing_dm_names:
                for ch in text_channels:
                    if ch.get("name", "").lower() == dm_channel_name.lower():
                        ch_id = str(ch["id"])
                        if ch_id not in channel_members or channel_members[ch_id] != [anima_name]:
                            channel_members[ch_id] = [anima_name]
                        if ch_id not in board_mapping:
                            board_mapping[ch_id] = dm_channel_name
                        break
                continue

            try:
                result = client.create_channel(
                    guild_id,
                    dm_channel_name,
                    parent_id=dm_category_id,
                )
                ch_id = str(result["id"])
                channel_members[ch_id] = [anima_name]
                board_mapping[ch_id] = dm_channel_name
                dm_created += 1
                logger.info("Created DM channel #%s for Anima '%s'", dm_channel_name, anima_name)

                board_file = channels_dir / f"{dm_channel_name}.jsonl"
                if not board_file.exists():
                    board_file.touch()
            except DiscordAPIError:
                logger.warning(
                    "Failed to create DM channel for '%s'",
                    anima_name,
                    exc_info=True,
                )

        # ── Persist config ──
        cfg.external_messaging.discord.board_mapping = board_mapping
        cfg.external_messaging.discord.channel_members = channel_members
        save_config(cfg)

        summary = {
            "status": "ok",
            "forward_created": forward_created,
            "reverse_created": reverse_created,
            "dm_created": dm_created,
            "total_mappings": len(board_mapping),
            "total_members_config": len(channel_members),
        }
        logger.info("Discord channel sync complete: %s", summary)
        return summary

    @staticmethod
    def _ensure_dm_category(
        client: DiscordClient,
        guild_id: str,
        all_channels: list[dict],
    ) -> str | None:
        """Ensure the 'DM' category exists, creating if needed. Returns category ID."""
        for ch in all_channels:
            if ch.get("type") == _CHANNEL_TYPE_CATEGORY and ch.get("name", "").upper() == _DM_CATEGORY_NAME:
                return str(ch["id"])

        try:
            result = client.create_channel(
                guild_id,
                _DM_CATEGORY_NAME,
                channel_type=_CHANNEL_TYPE_CATEGORY,
            )
            cat_id = str(result["id"])
            logger.info("Created Discord category '%s': %s", _DM_CATEGORY_NAME, cat_id)
            return cat_id
        except DiscordAPIError:
            logger.warning("Failed to create DM category", exc_info=True)
            return None
