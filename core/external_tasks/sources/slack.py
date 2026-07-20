# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Slack external tasks collector (unreplied mentions via message cache)."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

from core.exceptions import ToolConfigError
from core.external_tasks.models import ExternalTask

_PRIORITY = 80
_LOOKBACK_DAYS = 7
_TITLE_BODY_MAX = 80
_MENTION_LIMIT = 200


def collect_slack() -> list[ExternalTask]:
    """Collect unreplied Slack mentions from the local message cache.

    Uses :class:`~core.tools._slack_client.SlackClient` for auth / identity
    and :class:`~core.tools._slack_cache.MessageCache` for mention detection
    (same approach as the ``slack_unreplied`` tool). Does not hit the network
    for message search; only ``auth_test``. Permalinks are built deterministically.

    Task ids: ``slack-{channel_id}-{ts}``.
    """
    # Local import avoids circular import with collector → sources.
    from core.external_tasks.collector import CredentialNotFoundError

    try:
        from core.tools._slack_cache import MessageCache
        from core.tools._slack_client import SlackClient
    except ImportError as exc:
        raise CredentialNotFoundError(f"Slack dependencies unavailable: {exc}") from exc

    try:
        client = SlackClient()
    except ToolConfigError as exc:
        raise CredentialNotFoundError(str(exc)) from exc
    except ImportError as exc:
        raise CredentialNotFoundError(str(exc)) from exc

    try:
        client.auth_test()
    except Exception:
        # Invalid/revoked token and other API failures propagate for isolation.
        raise

    my_user_id = client.my_user_id or ""
    if not my_user_id:
        raise RuntimeError("Slack auth_test did not return user_id")

    cache = MessageCache()
    try:
        mentions = cache.find_unreplied(my_user_id, limit=_MENTION_LIMIT)
        cutoff = time.time() - timedelta(days=_LOOKBACK_DAYS).total_seconds()
        tasks: list[ExternalTask] = []
        for msg in mentions:
            if not is_actionable_mention(msg, my_user_id, cutoff_epoch=cutoff):
                continue
            task = _message_to_task(msg, client)
            if task is not None:
                tasks.append(task)
        return tasks
    finally:
        cache.close()


def is_actionable_mention(
    message: dict[str, Any],
    my_user_id: str,
    *,
    cutoff_epoch: float | None = None,
) -> bool:
    """Return True if *message* is an unreplied mention worth collecting.

    Limitations (intentionally simple):
    - Relies on cache rows already filtered by ``MessageCache.find_unreplied``
      (no own reply after the mention in thread/channel). Reactions are **not**
      checked: the cache schema has no reaction data, so "no reaction from me"
      cannot be verified.
    - Age filter uses ``ts_epoch`` / ``ts`` only; messages without a parseable
      timestamp are dropped when *cutoff_epoch* is set.
    - Does not re-query Slack for live thread state.
    """
    if not message:
        return False
    # Defensive: skip own messages if they slipped through.
    if message.get("user_id") == my_user_id:
        return False
    if cutoff_epoch is not None:
        ts_epoch = _message_epoch(message)
        if ts_epoch is None or ts_epoch < cutoff_epoch:
            return False
    return True


def _message_epoch(message: dict[str, Any]) -> float | None:
    raw = message.get("ts_epoch")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    ts = message.get("ts")
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def _message_to_task(
    message: dict[str, Any],
    client: Any,
) -> ExternalTask | None:
    channel_id = message.get("channel_id") or ""
    ts = message.get("ts") or ""
    if not channel_id or not ts:
        return None

    channel_name = message.get("channel_name") or _safe_channel_name(client, channel_id) or channel_id
    body = _preview_text(message.get("text") or "")
    title = f"#{channel_name}: {body}"

    iso_ts = _ts_to_iso(ts)
    source_url = _build_permalink(channel_id, ts)

    return ExternalTask(
        id=f"slack-{channel_id}-{ts}",
        title=title,
        status="open",
        source_type="slack",
        source_icon="slack",
        source_url=source_url,
        created_at=iso_ts,
        last_updated_at=iso_ts,
        priority=_PRIORITY,
    )


def _safe_channel_name(client: Any, channel_id: str) -> str:
    try:
        return client.get_channel_name(channel_id) or channel_id
    except Exception:
        return channel_id


def _build_permalink(channel_id: str, ts: str) -> str | None:
    """Build a deterministic Slack archive URL (no API call).

    Format: ``https://slack.com/archives/{channel_id}/p{ts_without_dot}``
    """
    if not channel_id or not ts:
        return None
    ts_digits = str(ts).replace(".", "")
    if not ts_digits.isdigit():
        return None
    return f"https://slack.com/archives/{channel_id}/p{ts_digits}"


def _ts_to_iso(ts: str) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return datetime.now(UTC).isoformat()


def _preview_text(text: str, max_len: int = _TITLE_BODY_MAX) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "(no text)"
    if len(cleaned) > max_len:
        return cleaned[:max_len]
    return cleaned
