from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Slack channel <-> AnimaWorks board bidirectional auto-sync.

**Forward sync (Slack -> AnimaWorks):**
On startup and periodically, the default Anima's bot lists all visible
public Slack channels.  For every channel found, a corresponding
AnimaWorks board is created if it doesn't already exist.

**Reverse sync (AnimaWorks -> Slack):**
For every AnimaWorks board (`shared/channels/*.jsonl`) that has no
corresponding Slack channel, a new public Slack channel is created
via ``conversations.create``.

After discovery and creation, **all** per-Anima bots are auto-joined
to every channel so that every Anima can both receive and post messages.

The ``board_mapping`` (channel_id -> board_name) is persisted in
config.json for message routing.
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from core.messenger import ChannelMeta, load_channel_meta, save_channel_meta
from core.paths import get_shared_dir

logger = logging.getLogger("animaworks.slack_channel_sync")

_SLACK_TIMEOUT = 15.0
_CONVERSATIONS_URL = "https://slack.com/api/conversations.list"
_CONVERSATIONS_JOIN_URL = "https://slack.com/api/conversations.join"
_CONVERSATIONS_CREATE_URL = "https://slack.com/api/conversations.create"


# ---------------------------------------------------------------------------
# Slack API helpers
# ---------------------------------------------------------------------------


async def _list_public_channels(token: str) -> list[dict[str, Any]]:
    """Call ``conversations.list`` to get all visible channels.

    Returns all public channels the bot can see plus private channels
    the bot has been invited to.  Each dict includes ``is_member`` so
    callers can decide whether to join.
    """
    channels: list[dict[str, Any]] = []
    cursor = ""
    async with httpx.AsyncClient(timeout=_SLACK_TIMEOUT) as client:
        for _ in range(20):  # pagination safety limit
            params: dict[str, Any] = {
                "types": "public_channel,private_channel",
                "exclude_archived": "true",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            resp = await client.get(
                _CONVERSATIONS_URL,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                logger.warning("conversations.list failed: %s", data.get("error", "unknown"))
                break
            for ch in data.get("channels", []):
                channels.append(ch)
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
    return channels


async def _join_channel_if_needed(
    token: str,
    channel_id: str,
    channel_name: str,
    *,
    bot_label: str = "",
) -> bool:
    """Join a Slack channel if the bot is not already a member.

    Returns True if the bot successfully joined (new join).
    """
    async with httpx.AsyncClient(timeout=_SLACK_TIMEOUT) as client:
        resp = await client.post(
            _CONVERSATIONS_JOIN_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel_id},
        )
        data = resp.json()
        if data.get("ok"):
            logger.info(
                "Bot %s joined Slack channel #%s (%s)",
                bot_label,
                channel_name,
                channel_id,
            )
            return True
        error = data.get("error", "unknown")
        if error == "already_in_channel":
            return False
        logger.warning(
            "Bot %s failed to join #%s: %s",
            bot_label,
            channel_name,
            error,
        )
        return False


async def _create_channel(token: str, name: str) -> dict[str, Any] | None:
    """Create a public Slack channel via ``conversations.create``.

    Returns the channel dict on success, or None if creation failed or
    the channel name is already taken.
    """
    async with httpx.AsyncClient(timeout=_SLACK_TIMEOUT) as client:
        resp = await client.post(
            _CONVERSATIONS_CREATE_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"name": name, "is_private": False},
        )
        data = resp.json()
        if data.get("ok"):
            ch = data.get("channel", {})
            logger.info("Created Slack channel #%s (%s)", name, ch.get("id"))
            return ch
        error = data.get("error", "unknown")
        if error == "name_taken":
            # Channel exists but wasn't visible in conversations.list
            # (e.g. private channel, or bot lacks visibility).
            logger.debug("Slack channel #%s already exists (name_taken)", name)
            return None
        logger.warning("Failed to create Slack channel #%s: %s", name, error)
        return None


# ---------------------------------------------------------------------------
# Name conversion
# ---------------------------------------------------------------------------


def _sanitize_channel_name(board_name: str) -> str:
    """Convert an AnimaWorks board name to a valid Slack channel name.

    Slack channel names must be lowercase, alphanumeric with hyphens and
    underscores only, max 80 characters.

    Returns empty string if the name cannot be converted (e.g. all
    non-ASCII characters like Japanese-only board names).
    """
    name = board_name.lower().strip()
    # Replace spaces/dots with hyphens
    name = re.sub(r"[\s.]+", "-", name)
    # Remove invalid characters (keep alphanumeric, hyphens, underscores)
    name = re.sub(r"[^a-z0-9\-_]", "", name)
    # Collapse multiple hyphens
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name[:80] if name else ""


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------


def _ensure_board(shared_dir: Path, board_name: str, slack_channel_name: str) -> bool:
    """Create an AnimaWorks board if it doesn't exist.

    Returns True if a new board was created.
    """
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)

    board_file = channels_dir / f"{board_name}.jsonl"
    if board_file.exists():
        return False

    # Create empty board file
    board_file.touch()

    # Create metadata
    meta = ChannelMeta(
        members=[],
        created_by="slack_channel_sync",
        created_at=datetime.now(UTC).isoformat(),
        description=f"Auto-synced from Slack channel #{slack_channel_name}",
    )
    save_channel_meta(shared_dir, board_name, meta)

    logger.info("Created board '%s' from Slack channel #%s", board_name, slack_channel_name)
    return True


def _mark_board_slack_deleted(shared_dir: Path, board_name: str) -> None:
    """Prevent a Board from being recreated in Slack after channel deletion."""
    meta = load_channel_meta(shared_dir, board_name)
    if meta is None:
        meta = ChannelMeta(members=[])
    if meta.slack_sync_disabled:
        return
    meta.slack_sync_disabled = True
    meta.slack_deleted_at = datetime.now(UTC).isoformat()
    save_channel_meta(shared_dir, board_name, meta)
    logger.info(
        "Marked board '%s' as Slack-deleted; reverse sync disabled",
        board_name,
    )


def _clear_board_slack_deleted(shared_dir: Path, board_name: str) -> None:
    """Re-enable Slack sync when a matching Slack channel becomes visible again."""
    meta = load_channel_meta(shared_dir, board_name)
    if meta is None or not meta.slack_sync_disabled:
        return
    meta.slack_sync_disabled = False
    meta.slack_deleted_at = ""
    save_channel_meta(shared_dir, board_name, meta)
    logger.info(
        "Cleared Slack-deleted tombstone for board '%s'",
        board_name,
    )


def _list_local_boards(shared_dir: Path) -> list[str]:
    """Return names of all AnimaWorks boards (from .jsonl files)."""
    channels_dir = shared_dir / "channels"
    if not channels_dir.is_dir():
        return []
    return [f.stem for f in channels_dir.glob("*.jsonl")]


# ---------------------------------------------------------------------------
# Main sync class
# ---------------------------------------------------------------------------


class SlackChannelSync:
    """Bidirectional Slack channel <-> AnimaWorks board sync."""

    def __init__(self) -> None:
        # channel_id -> board_name (runtime mapping)
        self.board_mapping: dict[str, str] = {}

    async def sync(self, manager: Any) -> dict[str, str]:
        """Run full bidirectional sync.

        Phase 1: Discover Slack channels -> create AnimaWorks boards
        Phase 2: Discover AnimaWorks boards -> create Slack channels
        Phase 3: Auto-join all per-Anima bots to all channels

        Args:
            manager: ``SlackSocketModeManager`` instance.

        Returns:
            The updated board_mapping (channel_id -> board_name).
        """
        from server.slack_socket import SlackSocketModeManager

        if not isinstance(manager, SlackSocketModeManager):
            return self.board_mapping

        from core.config.models import load_config

        cfg = load_config()
        default_anima = cfg.external_messaging.slack.default_anima or "sakura"

        shared_dir = get_shared_dir()
        previous_mapping = dict(self.board_mapping)
        self.board_mapping = {}
        new_boards = 0
        new_slack_channels = 0

        # Resolve discovery bot (default anima)
        discovery_app = manager._app_map.get(default_anima)
        if discovery_app is None:
            # Default anima not in app_map (likely missing Slack bot credentials).
            # Fall back to any available anima bot for channel discovery.
            available = [k for k in manager._app_map if k != "__shared__"]
            if available:
                fallback = available[0]
                logger.info(
                    "SlackChannelSync: default anima '%s' not in app_map, "
                    "falling back to '%s' for channel discovery. "
                    "Configure SLACK_BOT_TOKEN/SLACK_APP_TOKEN for '%s' to silence this.",
                    default_anima,
                    fallback,
                    default_anima,
                )
                discovery_app = manager._app_map[fallback]
            else:
                logger.warning(
                    "SlackChannelSync: no Slack bots available for channel discovery "
                    "(default_anima='%s' not configured, no fallback)",
                    default_anima,
                )
                return self.board_mapping

        discovery_token = discovery_app.client.token
        if not discovery_token:
            logger.warning("SlackChannelSync: no token for default anima '%s'", default_anima)
            return self.board_mapping

        logger.info("SlackChannelSync: starting sync via %s bot", default_anima)

        # ── Phase 1: Forward sync (Slack -> AnimaWorks boards) ──
        try:
            channels = await _list_public_channels(discovery_token)
        except Exception:
            logger.warning("Failed to list channels for %s", default_anima, exc_info=True)
            return self.board_mapping

        n_public = sum(1 for c in channels if not c.get("is_private"))
        n_private = len(channels) - n_public
        logger.info(
            "SlackChannelSync: %s found %d channels (%d public, %d private)",
            default_anima,
            len(channels),
            n_public,
            n_private,
        )

        channel_ids_to_join: list[tuple[str, str]] = []  # (ch_id, ch_name) — public only
        private_channel_names: list[str] = []  # for logging only
        slack_channel_names: set[str] = set()

        for ch in channels:
            ch_id = ch.get("id", "")
            ch_name = ch.get("name", "")
            if not ch_id or not ch_name:
                continue
            if ch_id.startswith(("D", "G")):
                continue

            is_private = ch.get("is_private", False)
            slack_channel_names.add(ch_name)

            board_name = ch_name
            if _ensure_board(shared_dir, board_name, ch_name):
                new_boards += 1

            self.board_mapping[ch_id] = board_name
            _clear_board_slack_deleted(shared_dir, board_name)
            if is_private:
                private_channel_names.append(ch_name)
            else:
                channel_ids_to_join.append((ch_id, ch_name))

        current_channel_ids = set(self.board_mapping)
        for previous_channel_id, board_name in previous_mapping.items():
            if previous_channel_id in current_channel_ids:
                continue
            if board_name in slack_channel_names:
                continue
            board_file = shared_dir / "channels" / f"{board_name}.jsonl"
            if board_file.exists():
                _mark_board_slack_deleted(shared_dir, board_name)

        # ── Phase 2: Reverse sync (AnimaWorks boards -> Slack channels) ──
        local_boards = _list_local_boards(shared_dir)
        for board_name in local_boards:
            if board_name in slack_channel_names:
                continue  # Already exists in Slack

            meta = load_channel_meta(shared_dir, board_name)
            if meta is not None and meta.slack_sync_disabled:
                logger.info(
                    "Skipping reverse sync for board '%s' (Slack-deleted tombstone)",
                    board_name,
                )
                continue

            slack_name = _sanitize_channel_name(board_name)
            if not slack_name:
                logger.debug(
                    "Skipping reverse sync for board '%s' (invalid Slack name)",
                    board_name,
                )
                continue
            if slack_name in slack_channel_names:
                continue  # Sanitized name matches an existing Slack channel

            try:
                new_ch = await _create_channel(discovery_token, slack_name)
                if new_ch:
                    ch_id = new_ch["id"]
                    self.board_mapping[ch_id] = board_name
                    channel_ids_to_join.append((ch_id, slack_name))
                    slack_channel_names.add(slack_name)
                    new_slack_channels += 1
            except Exception:
                logger.warning(
                    "Failed to create Slack channel for board '%s'",
                    board_name,
                    exc_info=True,
                )

        # ── Phase 3: Auto-join per-Anima bots to public channels ──
        # Private channels are intentionally skipped — membership is
        # managed manually via Slack ``/invite`` to respect access control.
        total_joins = 0
        for anima_name, app in manager._app_map.items():
            if anima_name == "__shared__":
                continue
            token = app.client.token
            if not token:
                continue
            for ch_id, ch_name in channel_ids_to_join:
                try:
                    joined = await _join_channel_if_needed(
                        token,
                        ch_id,
                        ch_name,
                        bot_label=anima_name,
                    )
                    if joined:
                        total_joins += 1
                except Exception:
                    logger.debug(
                        "Failed to auto-join #%s for %s",
                        ch_name,
                        anima_name,
                        exc_info=True,
                    )

        # Persist mapping to config
        self._update_config_mapping()

        if private_channel_names:
            logger.info(
                "SlackChannelSync: %d private channel(s) mapped but not auto-joined "
                "(use /invite in Slack to add bots): %s",
                len(private_channel_names),
                ", ".join(f"#{n}" for n in private_channel_names),
            )

        logger.info(
            "SlackChannelSync: %d mapping(s), %d new board(s), %d new Slack channel(s), %d bot join(s)",
            len(self.board_mapping),
            new_boards,
            new_slack_channels,
            total_joins,
        )

        return self.board_mapping

    def _update_config_mapping(self) -> None:
        """Write board_mapping back to the in-memory config."""
        try:
            from core.config.models import load_config, save_config

            cfg = load_config()
            cfg.external_messaging.slack.board_mapping = dict(self.board_mapping)
            save_config(cfg)
        except Exception:
            logger.warning("Failed to persist board_mapping to config", exc_info=True)

    def get_board_for_channel(self, channel_id: str) -> str | None:
        """Look up the AnimaWorks board name for a Slack channel ID."""
        return self.board_mapping.get(channel_id)
