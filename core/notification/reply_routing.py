from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reply routing for call_human notifications.

Maps Slack message timestamps (ts) to originating Anima names so that
threaded replies to call_human notifications can be routed back to the
Anima that sent the original notification.

Storage: ``{data_dir}/run/notification_map.json``

Note: Only notifications sent via the Bot Token API (``chat.postMessage``)
support reply routing, because the API returns a message ``ts`` that can
be persisted in the mapping.  Slack Incoming Webhooks do **not** return a
``ts`` in their response, so notifications sent that way cannot be mapped
and thread replies to them will not be routed back.
"""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.paths import get_data_dir
from core.platform.locks import file_lock

logger = logging.getLogger("animaworks.notification.reply_routing")

_MAX_AGE_DAYS = 7
_MAX_REPLY_LENGTH = 4000
_THREAD_CTX_SUMMARY_LIMIT = 150

# Slack mrkdwn patterns
_RE_LINK = re.compile(r"<(https?://[^|>]+)\|([^>]+)>")
_RE_LINK_BARE = re.compile(r"<(https?://[^>]+)>")
_RE_CHANNEL = re.compile(r"<#[A-Za-z0-9]+\|([^>]+)>")


def _map_path() -> Path:
    return get_data_dir() / "run" / "notification_map.json"


# ── Public API ──────────────────────────────────────────


def save_notification_mapping(
    ts: str,
    channel: str,
    anima_name: str,
    *,
    notification_text: str = "",
    callback_id: str = "",
) -> bool:
    """Persist a Slack message ts → Anima mapping for reply routing.

    Args:
        ts: Slack message timestamp of the notification.
        channel: Slack channel ID where the notification was posted.
        anima_name: Name of the Anima that sent the notification.
        notification_text: Original notification content (subject + body)
            so that thread replies can include context about what was notified.
        callback_id: When set, thread replies with a number resolve interactive approval.

    Returns:
        True when the mapping was written, False when the write failed
        (e.g. read-only filesystem inside an execution sandbox).
    """
    path = _map_path()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as fd, file_lock(fd, exclusive=True):
            fd.seek(0)
            raw = fd.read()
            data: dict[str, Any] = json.loads(raw) if raw.strip() else {}

            entry: dict[str, Any] = {
                "anima": anima_name,
                "channel": channel,
                "created_at": datetime.now(UTC).isoformat(),
            }
            if notification_text:
                entry["notification_text"] = notification_text[:2000]
            if callback_id:
                entry["callback_id"] = callback_id
            data[ts] = entry
            _prune_old_entries_inplace(data)

            fd.seek(0)
            fd.truncate()
            fd.write(json.dumps(data, ensure_ascii=False, indent=2))
            fd.flush()
    except OSError:
        logger.exception("Failed to save notification mapping for ts=%s", ts)
        return False
    return True


def post_notification_mapping_via_api(
    ts: str,
    channel: str,
    anima_name: str,
    *,
    notification_text: str = "",
    callback_id: str = "",
    timeout: float = 10.0,
) -> bool:
    """Save a notification mapping through the server's internal API.

    Sandboxed subprocesses (Codex workspace-write / deny profiles) cannot
    write ``{data_dir}/run/`` directly, so they delegate the write to the
    server process via ``POST /api/internal/notification-mapping``.
    """
    from core.internal_api import host_api

    try:
        resp = host_api.post(
            "/api/internal/notification-mapping",
            json={
                "ts": ts,
                "channel": channel,
                "anima_name": anima_name,
                "notification_text": notification_text[:2000],
                "callback_id": callback_id,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return bool(resp.json().get("ok"))
    except Exception:
        logger.warning("Failed to save notification mapping via API for ts=%s", ts, exc_info=True)
        return False


def save_notification_mapping_resilient(
    ts: str,
    channel: str,
    anima_name: str,
    *,
    notification_text: str = "",
    callback_id: str = "",
) -> bool:
    """Save a mapping directly, falling back to the server internal API.

    The direct write fails with EROFS when called from inside an execution
    sandbox; without the fallback, Slack thread replies to the notification
    are never routed back to the Anima (delivered as intent=observe and
    ignored).
    """
    if save_notification_mapping(
        ts,
        channel,
        anima_name,
        notification_text=notification_text,
        callback_id=callback_id,
    ):
        return True
    return post_notification_mapping_via_api(
        ts,
        channel,
        anima_name,
        notification_text=notification_text,
        callback_id=callback_id,
    )


def lookup_notification_mapping(thread_ts: str) -> dict[str, str] | None:
    """Look up which Anima sent the notification with the given ts.

    Returns ``{"anima": "...", "channel": "...", "notification_text": "...", "callback_id": "..."}``
    or ``None``.  ``notification_text`` and ``callback_id`` may be empty for older mappings.
    """
    path = _map_path()
    if not path.exists():
        return None
    try:
        with path.open("r") as fd, file_lock(fd, exclusive=False):
            data = json.load(fd)
    except (json.JSONDecodeError, OSError):
        return None
    entry = data.get(thread_ts)
    if entry is None:
        return None
    return {
        "anima": entry["anima"],
        "channel": entry["channel"],
        "notification_text": entry.get("notification_text", ""),
        "callback_id": entry.get("callback_id", ""),
    }


def remember_anima_slack_post(
    anima_name: str,
    channel: str,
    response: Any,
    *,
    thread_ts: str | None = None,
    text: str = "",
) -> None:
    """Route later thread replies to the Anima that posted this Slack message.

    A top-level post is keyed by its own ts; a reply inside an existing
    thread is keyed by that thread's ts.  An existing mapping is never
    overwritten, so a call_human thread stays with its original owner.
    Best effort: failures are logged and ignored.
    """
    # slack_sdk returns a SlackResponse, which is not a dict but has .get().
    get = getattr(response, "get", None)
    if not anima_name or not callable(get):
        return
    key = str(thread_ts or get("ts") or "")
    if not key:
        return
    try:
        if lookup_notification_mapping(key) is not None:
            return
        save_notification_mapping_resilient(
            key,
            str(get("channel") or channel),
            anima_name,
            notification_text=text[:2000],
        )
    except Exception:
        logger.warning("Failed to remember Slack post for reply routing", exc_info=True)


def prune_old_entries(max_age_days: int = _MAX_AGE_DAYS) -> None:
    """Remove entries older than *max_age_days* from the mapping file."""
    path = _map_path()
    if not path.exists():
        return
    try:
        with path.open("a+") as fd, file_lock(fd, exclusive=True):
            fd.seek(0)
            raw = fd.read()
            data: dict[str, Any] = json.loads(raw) if raw.strip() else {}
            before = len(data)
            _prune_old_entries_inplace(data, max_age_days)
            if len(data) < before:
                fd.seek(0)
                fd.truncate()
                fd.write(json.dumps(data, ensure_ascii=False, indent=2))
                fd.flush()
    except OSError:
        logger.exception("Failed to prune notification mapping")


def _prune_old_entries_inplace(
    data: dict[str, Any],
    max_age_days: int = _MAX_AGE_DAYS,
) -> None:
    """Remove stale entries from *data* dict in-place."""
    now = datetime.now(UTC)
    stale = [ts for ts, entry in data.items() if _age_days(entry.get("created_at", ""), now) > max_age_days]
    for ts in stale:
        del data[ts]


def _age_days(iso_str: str, now: datetime) -> float:
    try:
        created = datetime.fromisoformat(iso_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (now - created).total_seconds() / 86400
    except (ValueError, TypeError):
        return float("inf")


async def _resolve_callback_text_reply(
    callback_id: str,
    text: str,
    actor: str,
    *,
    thread_ts: str = "",
) -> bool:
    """Resolve a numbered callback reply when it matches an approval option."""
    if not callback_id or not text.isdigit():
        return False
    from core.notification.interactive import get_interaction_router

    router = get_interaction_router()
    request = await router.lookup(callback_id)
    if request is None:
        return False
    idx = int(text) - 1
    if not 0 <= idx < len(request.options):
        return False
    decision = request.options[idx]
    result = await router.resolve(callback_id, decision=decision, actor=actor, source="text_reply")
    if result is None:
        return False
    logger.info(
        "Interactive text reply resolved: thread_ts=%s callback_id=%s decision=%s",
        thread_ts,
        callback_id,
        decision,
    )
    return True


def _notification_context(target: str, notification_text: str) -> str:
    summary = notification_text.replace("\n", " ")[:_THREAD_CTX_SUMMARY_LIMIT]
    if not summary:
        return ""
    return (
        f"[Thread context — this is a reply to a call_human notification]\n  {target}: {summary}\n[/Thread context]\n\n"
    )


async def deliver_notification_reply(
    target: str,
    text: str,
    shared_dir: Path,
    *,
    actor: str = "human",
    source: str = "web",
    source_message_id: str = "",
    external_user_id: str = "",
    external_channel_id: str = "",
    external_thread_ts: str = "",
    thread_context: str = "",
    notification_id: str = "",
    callback_id: str = "",
) -> bool:
    """Send a notification reply to the Anima and record it for chat history."""
    reply = (text or "").strip()[:_MAX_REPLY_LENGTH]
    if not target or not reply:
        return False
    from core.messaging.messenger import Messenger

    Messenger(shared_dir, target).receive_external(
        content=(thread_context or "") + reply,
        source=source,
        source_message_id=source_message_id,
        external_user_id=external_user_id,
        external_channel_id=external_channel_id,
        external_thread_ts=external_thread_ts or notification_id,
        intent="question",
    )

    try:
        from core.activity.logger import ActivityLogger
        from core.paths import get_animas_dir

        meta: dict[str, str] = {}
        if external_thread_ts:
            meta["thread_ts"] = external_thread_ts
        if notification_id:
            meta["notification_id"] = notification_id
        if callback_id:
            meta["callback_id"] = callback_id
        await ActivityLogger(get_animas_dir() / target).alog(
            "human_reply",
            content=reply,
            from_person=actor or external_user_id or "human",
            via=source,
            ctx="chat",
            meta=meta,
            safe=True,
        )
    except Exception:
        logger.warning("Failed to log notification reply for anima=%s", target, exc_info=True)
    return True


async def route_thread_reply(
    event: dict,
    shared_dir: Path,
    *,
    slack_token: str = "",
) -> bool:
    """Route a Slack thread reply to the originating Anima if applicable.

    Checks event.thread_ts against the notification mapping. If a match
    is found, sanitizes the reply text, fetches thread context (so the
    Anima knows what the reply is about), and delivers it to the
    originating Anima's inbox via Messenger.receive_external().

    When the mapping includes ``callback_id`` (interactive call_human) and the
    reply is a plain number (1, 2, 3, ...), resolves the interaction instead
    of delivering a free-form thread reply.

    Args:
        event: Slack message event dict containing at minimum 'thread_ts',
               'text', 'ts', 'user', 'channel' keys.
        shared_dir: Path to the AnimaWorks shared directory.
        slack_token: Bot token for fetching thread context via Slack API.

    Returns:
        True if the reply was routed (caller should stop processing),
        False if no matching notification was found (caller should fall through).
    """
    thread_ts = event.get("thread_ts")
    if not thread_ts:
        return False

    mapping = lookup_notification_mapping(thread_ts)
    if mapping is None:
        return False

    callback_id = (mapping.get("callback_id") or "").strip()
    raw_reply = (event.get("text") or "").strip()
    user_id = str(event.get("user") or "")
    actor = user_id
    try:
        from core.notification.slack_names import get_cached_user_name

        actor = get_cached_user_name(user_id) or user_id
    except Exception:
        logger.debug("Failed to resolve Slack display name for text_reply actor", exc_info=True)
    if await _resolve_callback_text_reply(callback_id, raw_reply, actor, thread_ts=thread_ts):
        return True

    target = mapping["anima"]
    text = sanitize_slack_reply(event.get("text", ""))
    if not text:
        return False

    # Fetch full thread context via Slack API so the Anima knows
    # what the reply is about (e.g. which call_human notification).
    thread_ctx = ""
    ctx_source = "none"
    channel_id = event.get("channel", "")
    if slack_token and channel_id:
        thread_ctx = _fetch_thread_context_for_reply(slack_token, channel_id, thread_ts)
        if thread_ctx:
            ctx_source = "api"

    # Fallback: use stored notification text if API fetch failed or unavailable
    if not thread_ctx:
        notification_text = mapping.get("notification_text", "")
        if notification_text:
            thread_ctx = _notification_context(target, notification_text)
            ctx_source = "stored"

    await deliver_notification_reply(
        target,
        text,
        shared_dir,
        actor=actor,
        source="slack",
        source_message_id=str(event.get("ts") or ""),
        external_user_id=user_id,
        external_channel_id=str(event.get("channel") or ""),
        external_thread_ts=thread_ts,
        thread_context=thread_ctx,
        callback_id=callback_id,
    )

    logger.info(
        "Thread reply routed: thread_ts=%s -> anima=%s (ctx=%s)",
        thread_ts,
        target,
        ctx_source,
    )
    return True


def _fetch_thread_context_for_reply(
    token: str,
    channel_id: str,
    thread_ts: str,
    *,
    limit: int = 10,
) -> str:
    """Fetch Slack thread context for call_human reply routing.

    Returns a concise ``[Thread context]`` block with the parent message's
    first line and reply count, or an empty string on failure.
    """
    if not token or not thread_ts:
        return ""
    try:
        from core.integrations.slack import SlackClient

        client = SlackClient(token=token)
        replies = client.thread_replies(channel_id, thread_ts)
        if len(replies) <= 1:
            return ""
        parent = replies[0]
        parent_user = parent.get("user", "unknown")
        parent_text = parent.get("text", "").replace("\n", " ")[:_THREAD_CTX_SUMMARY_LIMIT]
        from core.notification.slack_names import cache_user_name, get_cached_user_name, resolve_slack_mentions

        if get_cached_user_name(parent_user) is None:
            try:
                cache_user_name(parent_user, client.resolve_user_name(parent_user))
            except Exception:
                logger.debug("Failed to resolve display name for user %s, using raw ID", parent_user)
        parent_display = get_cached_user_name(parent_user) or parent_user
        parent_text = resolve_slack_mentions(parent_text, token)
        reply_count = len(replies) - 1
        lines = [
            "[Thread context — this message is a reply in a Slack thread]",
            f"  @{parent_display}: {parent_text}",
            f"  ({reply_count} replies in thread)",
            "[/Thread context]",
            "",
        ]
        return "\n".join(lines)
    except Exception:
        logger.warning("Failed to fetch thread context for reply routing", exc_info=True)
        return ""


def sanitize_slack_reply(text: str, max_length: int = _MAX_REPLY_LENGTH) -> str:
    """Resolve Slack markup and mrkdwn formatting, then truncate for safe inbox delivery.

    User mentions are resolved to display names via the shared cache when available.
    URLs, channels, and HTML entities are converted by clean_slack_markup().
    """
    try:
        from core.notification.slack_names import resolve_slack_mentions

        text = resolve_slack_mentions(text, "")
    except Exception:
        from core.integrations._slack_markdown import clean_slack_markup

        text = clean_slack_markup(text)
    # Bold *text* → text, italic _text_ → text, strike ~text~ → text
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)~([^~]+)~(?!\w)", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text
