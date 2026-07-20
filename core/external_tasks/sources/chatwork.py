# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Chatwork external tasks collector (open my-tasks + unreplied To)."""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from core.exceptions import ToolConfigError
from core.external_tasks.models import ExternalTask

logger = logging.getLogger("animaworks.external_tasks.sources.chatwork")

_TASK_PRIORITY = 85
_MENTION_PRIORITY = 80
_LOOKBACK_DAYS = 7
_TITLE_BODY_MAX = 80
_MENTION_LIMIT = 200
_TO_TAG_RE = re.compile(r"\[To:\d+\][^\n]*\n?")
_BRACKET_TAG_RE = re.compile(r"\[.*?\]")


def collect_chatwork() -> list[ExternalTask]:
    """Collect open Chatwork tasks assigned to me.

    Also includes unreplied To/replies from the local message cache when the
    cache and ``/me`` identity are available (last 7 days).

    Task ids:
      - ``chatwork-task-{task_id}`` for open tasks
      - ``chatwork-msg-{room_id}-{message_id}`` for unreplied mentions
    """
    # Local import avoids circular import with collector → sources.
    from core.external_tasks.collector import CredentialNotFoundError

    try:
        from core.tools._chatwork_client import ChatworkClient
    except ImportError as exc:
        raise CredentialNotFoundError(f"Chatwork dependencies unavailable: {exc}") from exc

    try:
        client = ChatworkClient()
    except ToolConfigError as exc:
        raise CredentialNotFoundError(str(exc)) from exc
    except ImportError as exc:
        raise CredentialNotFoundError(str(exc)) from exc

    tasks: list[ExternalTask] = []
    tasks.extend(_collect_open_tasks(client))
    tasks.extend(_collect_unreplied_mentions(client))
    return tasks


def _collect_open_tasks(client: Any) -> list[ExternalTask]:
    raw_tasks = client.my_tasks(status="open") or []
    result: list[ExternalTask] = []
    for item in raw_tasks:
        task_id = item.get("task_id")
        if task_id is None:
            continue
        room = item.get("room") or {}
        room_id = room.get("room_id", "")
        room_name = room.get("name") or str(room_id) or "?"
        body = _preview_body(item.get("body") or "", strip_bracket_tags=True)
        message_id = item.get("message_id") or ""
        source_url = _chatwork_url(room_id, message_id)
        # Prefer deadline; otherwise a registration-style field if present;
        # fall back to a fixed epoch so missing deadlines stay deterministic
        # across collection cycles (no "always newest" via datetime.now).
        iso_ts = _task_timestamp(item)
        result.append(
            ExternalTask(
                id=f"chatwork-task-{task_id}",
                title=f"{room_name}: {body}",
                status="open",
                source_type="chatwork",
                source_icon="chatwork",
                source_url=source_url,
                created_at=iso_ts,
                last_updated_at=iso_ts,
                priority=_TASK_PRIORITY,
            )
        )
    return result


def _collect_unreplied_mentions(client: Any) -> list[ExternalTask]:
    """Collect unreplied To-mentions from the SQLite message cache.

    Uses the same cache-based approach as ``chatwork_unreplied``. If the cache
    module or identity lookup fails, logs and returns an empty list so open
    tasks still surface.
    """
    try:
        from core.tools._chatwork_cache import MessageCache
    except ImportError:
        logger.debug("Chatwork MessageCache unavailable; skipping unreplied mentions")
        return []

    try:
        me = client.me()
        my_account_id = str(me.get("account_id", ""))
        if not my_account_id:
            return []
    except Exception as exc:
        logger.warning("Chatwork me() failed; skipping unreplied mentions: %s", exc)
        return []

    cache = MessageCache()
    try:
        mentions = cache.find_unreplied(my_account_id, limit=_MENTION_LIMIT)
        cutoff = int(time.time() - timedelta(days=_LOOKBACK_DAYS).total_seconds())
        result: list[ExternalTask] = []
        for msg in mentions:
            send_time = int(msg.get("send_time") or 0)
            if send_time and send_time < cutoff:
                continue
            room_id = str(msg.get("room_id") or "")
            message_id = str(msg.get("message_id") or "")
            if not room_id or not message_id:
                continue
            room_name = msg.get("room_name") or room_id
            body = _preview_body(msg.get("body") or "", strip_to_tags=True)
            iso_ts = _unix_to_iso(send_time) if send_time else datetime.now(UTC).isoformat()
            result.append(
                ExternalTask(
                    id=f"chatwork-msg-{room_id}-{message_id}",
                    title=f"{room_name}: {body}",
                    status="open",
                    source_type="chatwork",
                    source_icon="chatwork",
                    source_url=_chatwork_url(room_id, message_id),
                    created_at=iso_ts,
                    last_updated_at=iso_ts,
                    priority=_MENTION_PRIORITY,
                )
            )
        return result
    finally:
        cache.close()


def _chatwork_url(room_id: Any, message_id: Any) -> str | None:
    if not room_id:
        return None
    if message_id:
        return f"https://www.chatwork.com/#!rid{room_id}-{message_id}"
    return f"https://www.chatwork.com/#!rid{room_id}"


_EPOCH_ISO = "1970-01-01T00:00:00+00:00"
# Prefer deadline; Chatwork open-task payloads rarely expose creation time,
# but accept common registration-style keys if present.
_TASK_TS_KEYS = ("limit_time", "created", "created_at", "date", "send_time")


def _task_timestamp(item: dict[str, Any]) -> str:
    for key in _TASK_TS_KEYS:
        raw = item.get(key)
        if raw in (None, "", 0, "0"):
            continue
        if isinstance(raw, (int, float)):
            return _unix_to_iso(raw)
        if isinstance(raw, str):
            # Numeric string → unix; otherwise treat as ISO-ish passthrough.
            try:
                return _unix_to_iso(float(raw))
            except (TypeError, ValueError):
                text = raw.strip()
                if text:
                    return text
    return _EPOCH_ISO


def _unix_to_iso(ts: int | float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return _EPOCH_ISO


def _preview_body(
    body: str,
    *,
    strip_to_tags: bool = False,
    strip_bracket_tags: bool = False,
    max_len: int = _TITLE_BODY_MAX,
) -> str:
    text = body or ""
    if strip_to_tags:
        text = _TO_TAG_RE.sub("", text)
    if strip_bracket_tags:
        text = _BRACKET_TAG_RE.sub("", text)
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "(no text)"
    if len(cleaned) > max_len:
        return cleaned[:max_len]
    return cleaned
