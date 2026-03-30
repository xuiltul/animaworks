from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Slack channel → AnimaWorks board auto-sync.

On startup and periodically, each per-Anima bot lists Slack channels
it has joined.  For every public channel found, a corresponding
AnimaWorks board is created if it doesn't already exist.  The
``board_mapping`` (channel_id → board_name) is kept in memory and
written to the Slack config section so that message handlers can
route incoming Slack messages to the correct board.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from core.messenger import ChannelMeta, load_channel_meta, save_channel_meta
from core.paths import get_shared_dir

logger = logging.getLogger("animaworks.slack_channel_sync")

_SLACK_TIMEOUT = 15.0
_CONVERSATIONS_URL = "https://slack.com/api/conversations.list"


async def _list_bot_channels(token: str) -> list[dict[str, Any]]:
    """Call ``conversations.list`` to get public channels the bot is in."""
    channels: list[dict[str, Any]] = []
    cursor = ""
    async with httpx.AsyncClient(timeout=_SLACK_TIMEOUT) as client:
        for _ in range(20):  # pagination safety limit
            params: dict[str, Any] = {
                "types": "public_channel",
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
                logger.warning(
                    "conversations.list failed: %s", data.get("error", "unknown")
                )
                break
            for ch in data.get("channels", []):
                if ch.get("is_member"):
                    channels.append(ch)
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
    return channels


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


class SlackChannelSync:
    """Detect Slack channels and auto-create corresponding AnimaWorks boards."""

    def __init__(self) -> None:
        # channel_id → board_name (runtime mapping)
        self.board_mapping: dict[str, str] = {}

    async def sync(self, manager: Any) -> dict[str, str]:
        """Scan all per-Anima bot channels and sync boards.

        Args:
            manager: ``SlackSocketModeManager`` instance.

        Returns:
            The updated board_mapping (channel_id → board_name).
        """
        from server.slack_socket import SlackSocketModeManager

        if not isinstance(manager, SlackSocketModeManager):
            return self.board_mapping

        shared_dir = get_shared_dir()
        new_boards = 0

        for anima_name, app in manager._app_map.items():
            if anima_name == "__shared__":
                continue
            token = app.client.token
            if not token:
                continue
            try:
                channels = await _list_bot_channels(token)
            except Exception:
                logger.debug(
                    "Failed to list channels for %s", anima_name, exc_info=True
                )
                continue

            for ch in channels:
                ch_id = ch.get("id", "")
                ch_name = ch.get("name", "")
                if not ch_id or not ch_name:
                    continue
                # Skip DMs, MPIMs (starts with D/G)
                if ch_id.startswith(("D", "G")):
                    continue

                board_name = ch_name  # use Slack channel name as board name
                if _ensure_board(shared_dir, board_name, ch_name):
                    new_boards += 1

                self.board_mapping[ch_id] = board_name

        # Persist mapping to config (in-memory config object)
        self._update_config_mapping()

        if new_boards:
            logger.info(
                "SlackChannelSync: created %d new board(s), total mappings: %d",
                new_boards,
                len(self.board_mapping),
            )
        else:
            logger.debug(
                "SlackChannelSync: %d channel mapping(s), no new boards",
                len(self.board_mapping),
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
            logger.debug("Failed to persist board_mapping to config", exc_info=True)

    def get_board_for_channel(self, channel_id: str) -> str | None:
        """Look up the AnimaWorks board name for a Slack channel ID."""
        return self.board_mapping.get(channel_id)
