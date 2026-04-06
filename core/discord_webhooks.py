from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Discord webhook manager for per-Anima outbound messages.

Each channel gets a single webhook; per-message ``username`` and ``avatar_url``
parameters make each Anima appear as a distinct identity.
"""

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from core.paths import get_data_dir
from core.tools._anima_icon_url import resolve_anima_icon_url
from core.tools._base import get_credential
from core.tools._discord_client import DiscordAPIError, DiscordClient

from core.tools._discord_markdown import DISCORD_MESSAGE_LIMIT

logger = logging.getLogger("animaworks.discord_webhooks")
_WEBHOOK_NAME = "AnimaWorks"

# Thread-to-Anima mapping TTL
_THREAD_MAP_TTL_DAYS = 7


# ── Singleton ────────────────────────────────────────────────

_instance: DiscordWebhookManager | None = None
_instance_lock = threading.Lock()


def get_webhook_manager() -> DiscordWebhookManager:
    """Return the singleton DiscordWebhookManager instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DiscordWebhookManager()
    return _instance


# ── Manager ──────────────────────────────────────────────────


class DiscordWebhookManager:
    """Manages per-channel webhooks and thread-to-Anima mappings."""

    def __init__(self) -> None:
        self._webhooks: dict[str, dict[str, str]] = {}  # channel_id → {id, token}
        self._thread_map: dict[str, dict[str, Any]] = {}  # message_id → {anima, ts}
        self._lock = threading.Lock()
        self._client: DiscordClient | None = None
        self._load_persisted()

    def _ensure_client(self) -> DiscordClient:
        if self._client is None:
            token = get_credential("discord", "discord", env_var="DISCORD_BOT_TOKEN")
            self._client = DiscordClient(token=token)
        return self._client

    # ── Webhook CRUD ─────────────────────────────────────────

    def _get_or_create_webhook(self, channel_id: str) -> tuple[str, str]:
        """Return (webhook_id, webhook_token) for a channel, creating if needed."""
        with self._lock:
            cached = self._webhooks.get(channel_id)
            if cached:
                return cached["id"], cached["token"]

        client = self._ensure_client()

        # Check existing webhooks first
        try:
            existing = client.list_webhooks(channel_id)
            for wh in existing:
                if wh.get("name") == _WEBHOOK_NAME and wh.get("token"):
                    wh_id = str(wh["id"])
                    wh_token = wh["token"]
                    with self._lock:
                        self._webhooks[channel_id] = {"id": wh_id, "token": wh_token}
                    self._persist()
                    return wh_id, wh_token
        except DiscordAPIError:
            logger.debug("Failed to list webhooks for channel %s", channel_id, exc_info=True)

        # Create new
        try:
            result = client.create_webhook(channel_id, _WEBHOOK_NAME)
            wh_id = str(result["id"])
            wh_token = result["token"]
            with self._lock:
                self._webhooks[channel_id] = {"id": wh_id, "token": wh_token}
            self._persist()
            logger.info("Created webhook for channel %s: %s", channel_id, wh_id)
            return wh_id, wh_token
        except DiscordAPIError:
            logger.exception("Failed to create webhook for channel %s", channel_id)
            raise

    def send_as_anima(
        self,
        channel_id: str,
        anima_name: str,
        content: str,
    ) -> str:
        """Send a message to a Discord channel appearing as a specific Anima.

        Returns the sent message ID (snowflake string).

        Note: Discord webhooks do not support ``message_reference`` (reply
        threading). Use the standard bot ``send_message`` API for replies.
        """
        wh_id, wh_token = self._get_or_create_webhook(channel_id)
        client = self._ensure_client()

        avatar_url = resolve_anima_icon_url(anima_name)

        # Split long messages
        chunks = _split_message(content)
        last_msg_id = ""

        for chunk in chunks:
            try:
                result = client.execute_webhook(
                    wh_id,
                    wh_token,
                    chunk,
                    username=anima_name,
                    avatar_url=avatar_url or None,
                )
                last_msg_id = str(result.get("id", ""))
            except DiscordAPIError as exc:
                if exc.status == 404:
                    # Webhook deleted — recreate and retry
                    with self._lock:
                        self._webhooks.pop(channel_id, None)
                    wh_id, wh_token = self._get_or_create_webhook(channel_id)
                    result = client.execute_webhook(
                        wh_id,
                        wh_token,
                        chunk,
                        username=anima_name,
                        avatar_url=avatar_url or None,
                    )
                    last_msg_id = str(result.get("id", ""))
                else:
                    raise

        # Record thread mapping for reply routing
        if last_msg_id:
            self.record_thread_mapping(last_msg_id, anima_name)

        return last_msg_id

    # ── Thread-to-Anima mapping ──────────────────────────────

    def record_thread_mapping(self, message_id: str, anima_name: str) -> None:
        """Record that a message was sent by an Anima for reply routing."""
        with self._lock:
            self._thread_map[message_id] = {
                "anima": anima_name,
                "ts": time.time(),
            }
        self._persist_thread_map()

    def lookup_thread_anima(self, message_id: str) -> str | None:
        """Look up which Anima sent a message (for reply routing)."""
        with self._lock:
            entry = self._thread_map.get(message_id)
            if not entry:
                return None
            # Check TTL
            age_days = (time.time() - entry["ts"]) / 86400
            if age_days > _THREAD_MAP_TTL_DAYS:
                del self._thread_map[message_id]
                return None
            return entry["anima"]

    # ── Persistence ──────────────────────────────────────────

    def _webhooks_path(self) -> Path:
        return get_data_dir() / "run" / "discord_webhooks.json"

    def _thread_map_path(self) -> Path:
        return get_data_dir() / "run" / "discord_thread_map.json"

    def _load_persisted(self) -> None:
        """Load cached webhook and thread map data from disk."""
        try:
            p = self._webhooks_path()
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._webhooks = data
        except Exception:
            logger.debug("Failed to load webhook cache", exc_info=True)

        try:
            p = self._thread_map_path()
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Prune expired entries
                    now = time.time()
                    cutoff = now - (_THREAD_MAP_TTL_DAYS * 86400)
                    self._thread_map = {
                        k: v for k, v in data.items() if isinstance(v, dict) and v.get("ts", 0) > cutoff
                    }
        except Exception:
            logger.debug("Failed to load thread map", exc_info=True)

    def _persist(self) -> None:
        """Save webhook cache to disk (mode 0o600 — contains tokens)."""
        try:
            p = self._webhooks_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                snapshot = dict(self._webhooks)
            _atomic_write_json(p, snapshot)
        except Exception:
            logger.debug("Failed to persist webhook cache", exc_info=True)

    def _persist_thread_map(self) -> None:
        """Save thread map to disk (mode 0o600 — contains routing data)."""
        try:
            p = self._thread_map_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                snapshot = dict(self._thread_map)
            _atomic_write_json(p, snapshot)
        except Exception:
            logger.debug("Failed to persist thread map", exc_info=True)


# ── Helpers ──────────────────────────────────────────────────


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to *path* atomically via temp file + rename (mode 0o600)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _split_message(content: str) -> list[str]:
    """Split content into chunks that fit Discord's message limit."""
    if len(content) <= DISCORD_MESSAGE_LIMIT:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= DISCORD_MESSAGE_LIMIT:
            chunks.append(content)
            break
        # Try to split at a newline
        cut = content.rfind("\n", 0, DISCORD_MESSAGE_LIMIT)
        if cut <= 0:
            cut = DISCORD_MESSAGE_LIMIT
        chunks.append(content[:cut])
        content = content[cut:].lstrip("\n")
    return chunks
