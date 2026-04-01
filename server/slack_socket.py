from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Slack Socket Mode integration for real-time message reception."""

import asyncio
import collections
import json
import logging
import os
import re
import threading
import time
from typing import Any

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.error import BoltUnhandledRequestError
from slack_bolt.response import BoltResponse

from core.config.models import load_config
from core.messenger import Messenger
from core.paths import get_data_dir
from core.tools._base import _lookup_shared_credentials, _lookup_vault_credential, get_credential

logger = logging.getLogger("animaworks.slack_socket")

# ── Dedup: prevent same Slack message from being processed twice ──
# When both message and app_mention events fire for a single @-mention,
# the SAME handler should only process it once.  The key is scoped per
# handler (anima name) so that different bots receiving the same event
# in a shared channel each process it independently.
_DEDUP_TTL_SEC = 10
_dedup_lock = threading.Lock()
_recent_ts: collections.OrderedDict[str, float] = collections.OrderedDict()

_USER_NAME_CACHE_MAX = 500
_cache_lock = threading.Lock()
_user_name_cache: dict[str, str] = {}


def _is_duplicate_ts(ts: str, handler_name: str = "") -> bool:
    """Return True if *ts* was already processed by *handler_name*.

    The dedup key is ``handler_name:ts`` so that different per-Anima
    handlers can each process the same Slack event independently.
    Within a single handler, duplicate ``message`` + ``app_mention``
    events for the same ts are still suppressed.
    """
    key = f"{handler_name}:{ts}" if handler_name else ts
    now = time.monotonic()
    with _dedup_lock:
        while _recent_ts and next(iter(_recent_ts.values())) < now - _DEDUP_TTL_SEC:
            _recent_ts.popitem(last=False)
        if key in _recent_ts:
            return True
        _recent_ts[key] = now
        return False


def _cache_user_name(uid: str, name: str) -> None:
    """Thread-safe bounded insert into the user-name cache."""
    with _cache_lock:
        if len(_user_name_cache) >= _USER_NAME_CACHE_MAX and uid not in _user_name_cache:
            try:
                _user_name_cache.pop(next(iter(_user_name_cache)))
            except StopIteration:
                pass
        _user_name_cache[uid] = name


def _get_cached_user_name(uid: str) -> str | None:
    """Thread-safe lookup from the user-name cache."""
    with _cache_lock:
        return _user_name_cache.get(uid)


def _detect_slack_intent(text: str, channel_id: str, bot_user_id: str) -> str:
    """Return ``"question"`` if the message is a DM or mentions the bot."""
    if channel_id.startswith("D"):
        return "question"
    if bot_user_id and f"<@{bot_user_id}>" in (text or ""):
        return "question"
    return ""


def _resolve_slack_mentions(text: str, token: str) -> str:
    """Resolve <@U...> mentions and Slack markup to human-readable text.

    Extracts all user IDs, resolves unknown ones via Slack API (cached),
    then applies clean_slack_markup() for full conversion.
    """
    if not text:
        return text
    user_ids = set(re.findall(r"<@(U[A-Z0-9]+)>", text))
    unknown = {uid for uid in user_ids if _get_cached_user_name(uid) is None}
    if unknown and token:
        try:
            from core.tools.slack import SlackClient

            client = SlackClient(token=token)
            for uid in unknown:
                _cache_user_name(uid, client.resolve_user_name(uid))
        except Exception:
            logger.debug("Failed to resolve Slack user mentions", exc_info=True)
    from core.tools._slack_markdown import clean_slack_markup

    with _cache_lock:
        snapshot = dict(_user_name_cache)
    return clean_slack_markup(text, cache=snapshot)


def _build_slack_annotation(channel_id: str, has_mention: bool) -> str:
    """Build a system annotation line for Slack message targeting."""
    if channel_id.startswith("D"):
        return "[slack:DM]\n"
    if has_mention:
        return "[slack:channel — あなたがメンションされています]\n"
    return "[slack:channel — あなたへの直接メンションはありません]\n"


def _detect_mention_intent(
    text: str,
    bot_user_id: str,
    alias_user_ids: set[str],
) -> str:
    """Return 'question' if text mentions the bot or any alias user ID."""
    if bot_user_id and f"<@{bot_user_id}>" in (text or ""):
        return "question"
    for uid in alias_user_ids:
        if f"<@{uid}>" in (text or ""):
            return "question"
    return ""


def _mentions_registered_bot(
    text: str,
    bot_user_ids: dict[str, str],
    *,
    exclude_names: set[str] | None = None,
) -> bool:
    """Return True when text mentions any registered bot user ID."""
    raw = text or ""
    excluded = exclude_names or set()
    for name, uid in bot_user_ids.items():
        if name in excluded or not uid:
            continue
        if f"<@{uid}>" in raw:
            return True
    return False


def _load_alias_user_ids() -> set[str]:
    """Load Slack user IDs from config user_aliases for mention detection."""
    try:
        cfg = load_config()
        return {alias.slack_user_id for alias in cfg.external_messaging.user_aliases.values() if alias.slack_user_id}
    except Exception:
        return set()


_THREAD_CTX_SUMMARY_LIMIT = 150


def _fetch_thread_context(token: str, channel_id: str, thread_ts: str, *, limit: int = 10) -> str:
    """Fetch Slack thread context and format as a concise summary block.

    Returns a ``[Thread context]`` block with the parent message's first line
    (truncated to *_THREAD_CTX_SUMMARY_LIMIT* chars) and the reply count,
    or an empty string when *thread_ts* is empty or the fetch fails.
    """
    if not thread_ts or not token:
        return ""
    try:
        from core.tools._slack_markdown import clean_slack_markup
        from core.tools.slack import SlackClient

        client = SlackClient(token=token)
        replies = client.thread_replies(channel_id, thread_ts)
        if len(replies) <= 1:
            return ""
        parent = replies[0]
        parent_user = parent.get("user", "unknown")
        parent_text = parent.get("text", "").replace("\n", " ")[:_THREAD_CTX_SUMMARY_LIMIT]
        # Resolve parent author display name
        if _get_cached_user_name(parent_user) is None:
            try:
                _cache_user_name(parent_user, client.resolve_user_name(parent_user))
            except Exception:
                pass
        parent_display = _get_cached_user_name(parent_user) or parent_user
        with _cache_lock:
            snapshot = dict(_user_name_cache)
        parent_text = clean_slack_markup(parent_text, cache=snapshot)
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
        logger.warning("Failed to fetch Slack thread context", exc_info=True)
        return ""


async def _resolve_bot_user_id(app: AsyncApp) -> str:
    """Call ``auth.test`` once and return the bot's Slack user ID."""
    try:
        resp = await app.client.auth_test()
        return resp.get("user_id", "") or ""
    except Exception:
        logger.warning("Failed to resolve bot user ID via auth.test", exc_info=True)
        return ""


# ── Board routing dedup: one board post per Slack message ──
# Multiple handlers (per-anima + shared) receive the same event; only
# the first one should forward to the AnimaWorks board.
_board_dedup_lock = threading.Lock()
_board_dedup_ts: collections.OrderedDict[str, float] = collections.OrderedDict()
_BOARD_DEDUP_TTL_SEC = 10


def _route_to_board(channel_id: str, text: str, user_name: str, *, ts: str = "") -> None:
    """Post a Slack message to the mapped AnimaWorks board (if any).

    Looks up the board_mapping from config.  If the channel has a
    corresponding board, the message is posted via Messenger with
    ``source="slack"`` to prevent echo loops.

    The *ts* parameter is used for global dedup — when multiple handlers
    receive the same Slack event, only the first call posts to the board.
    """
    if ts:
        now = time.monotonic()
        with _board_dedup_lock:
            # Expire old entries
            while _board_dedup_ts and next(iter(_board_dedup_ts.values())) < now - _BOARD_DEDUP_TTL_SEC:
                _board_dedup_ts.popitem(last=False)
            if ts in _board_dedup_ts:
                return
            _board_dedup_ts[ts] = now

    try:
        cfg = load_config()
        board_name = cfg.external_messaging.slack.board_mapping.get(channel_id)
        if not board_name:
            return
        shared_dir = get_data_dir() / "shared"
        messenger = Messenger(shared_dir, user_name or "slack")
        messenger.post_channel(board_name, text, source="slack", from_name=user_name or "slack")
    except Exception:
        logger.debug("Board routing failed for channel %s", channel_id, exc_info=True)


class SlackSocketModeManager:
    """Manages Slack Socket Mode WebSocket connections.

    Supports multiple per-Anima bots alongside an optional shared bot.
    Each per-Anima bot runs its own AsyncApp + AsyncSocketModeHandler,
    with messages routed directly to the corresponding Anima inbox.
    """

    def __init__(self) -> None:
        self._handler_map: dict[str, AsyncSocketModeHandler] = {}
        self._app_map: dict[str, AsyncApp] = {}
        self._bot_user_ids: dict[str, str] = {}

    @property
    def _handlers(self) -> list[AsyncSocketModeHandler]:
        """Backward-compatible list view of active handlers."""
        return list(self._handler_map.values())

    @property
    def _apps(self) -> list[AsyncApp]:
        """Backward-compatible list view of active apps."""
        return list(self._app_map.values())

    async def start(self) -> None:
        """Start Socket Mode connections if enabled in config."""
        config = load_config()
        slack_config = config.external_messaging.slack
        if not slack_config.enabled or slack_config.mode != "socket":
            logger.info("Slack Socket Mode is disabled")
            return

        # Register all per-Anima handlers without connecting yet
        for anima_name in self._discover_per_anima_bots():
            await self._add_per_anima_handler(anima_name, connect=False)

        # Shared bot (optional fallback for unmapped channels)
        try:
            shared_bot = get_credential("slack", "slack_socket", env_var="SLACK_BOT_TOKEN")
            shared_app_token = get_credential("slack_app", "slack_socket", env_var="SLACK_APP_TOKEN")
            app = AsyncApp(token=shared_bot, raise_error_for_unhandled_request=True)
            self._register_error_handler(app)
            shared_bot_uid = await _resolve_bot_user_id(app)
            self._bot_user_ids["__shared__"] = shared_bot_uid
            self._register_shared_handler(app, shared_bot_uid)
            handler = AsyncSocketModeHandler(app, shared_app_token)
            self._app_map["__shared__"] = app
            self._handler_map["__shared__"] = handler
            logger.info("Shared Slack bot registered (bot_uid=%s)", shared_bot_uid)
        except Exception:
            if not self._handler_map:
                raise
            logger.info("Shared Slack bot not configured; per-Anima bots only")

        # Connect all handlers in a single batch to avoid double-connect
        if self._handler_map:
            await asyncio.gather(*(h.connect_async() for h in self._handler_map.values()))
            logger.info(
                "Slack Socket Mode connected (%d handler(s))",
                len(self._handler_map),
            )

            # Post-connect health check: verify WebSocket sessions
            await asyncio.sleep(2)
            dead: list[str] = []
            for name, handler in self._handler_map.items():
                client = handler.client
                session = getattr(client, "current_session", None)
                receiver = getattr(client, "message_receiver", None)
                alive = (
                    session is not None
                    and not getattr(session, "closed", True)
                    and receiver is not None
                    and not receiver.done()
                )
                if not alive:
                    dead.append(name)
            if dead:
                logger.warning(
                    "Socket Mode: %d handler(s) NOT alive after connect: %s",
                    len(dead), dead,
                )

    async def reload(self) -> dict[str, Any]:
        """Diff-based handler reload: add new, remove deleted, keep existing."""
        config = load_config()
        slack_config = config.external_messaging.slack
        if not slack_config.enabled or slack_config.mode != "socket":
            if self._handler_map:
                await self.stop()
            return {"status": "disabled"}

        current_animas = {name for name in self._handler_map if name != "__shared__"}
        desired_animas = set(self._discover_per_anima_bots())

        added: list[str] = []
        removed: list[str] = []
        errors: list[dict[str, str]] = []

        # Add new handlers first (add-before-remove for safety)
        to_add = desired_animas - current_animas
        for name in sorted(to_add):
            try:
                ok = await self._add_per_anima_handler(name)
                if ok:
                    added.append(name)
            except Exception as exc:
                logger.exception("Failed to add per-Anima handler for '%s'", name)
                errors.append({"name": name, "error": str(exc)})

        # Remove handlers no longer desired
        to_remove = current_animas - desired_animas
        for name in sorted(to_remove):
            try:
                await self._remove_per_anima_handler(name)
                removed.append(name)
            except Exception as exc:
                logger.exception("Failed to remove per-Anima handler for '%s'", name)
                errors.append({"name": name, "error": str(exc)})

        result: dict[str, Any] = {
            "status": "ok",
            "added": added,
            "removed": removed,
            "active_handlers": len(self._handler_map),
        }
        if errors:
            result["errors"] = errors
        return result

    async def _add_per_anima_handler(self, anima_name: str, *, connect: bool = True) -> bool:
        """Create a per-Anima Socket Mode handler.

        When *connect* is False the handler is registered but not connected
        (caller is responsible for calling ``connect_async`` later).

        Returns True on success, False if credentials are missing.
        """
        bot_token = self._get_per_anima_credential("SLACK_BOT_TOKEN", anima_name)
        app_token = self._get_per_anima_credential("SLACK_APP_TOKEN", anima_name)
        if not bot_token or not app_token:
            logger.debug("Missing credentials for per-Anima bot '%s'", anima_name)
            return False

        try:
            app = AsyncApp(token=bot_token, raise_error_for_unhandled_request=True)
            self._register_error_handler(app)
            bot_uid = await _resolve_bot_user_id(app)
            self._bot_user_ids[anima_name] = bot_uid
            self._register_per_anima_handler(app, anima_name, bot_uid)
            handler = AsyncSocketModeHandler(app, app_token)
            self._app_map[anima_name] = app
            self._handler_map[anima_name] = handler
            if connect:
                await handler.connect_async()
            logger.info(
                "Per-Anima Slack bot registered: %s (bot_uid=%s)",
                anima_name,
                bot_uid,
            )
            return True
        except Exception:
            logger.exception(
                "Failed to set up per-Anima Slack bot for '%s'",
                anima_name,
            )
            # Clean up partial state
            self._app_map.pop(anima_name, None)
            self._handler_map.pop(anima_name, None)
            self._bot_user_ids.pop(anima_name, None)
            return False

    async def _remove_per_anima_handler(self, anima_name: str) -> None:
        """Disconnect and remove a per-Anima Socket Mode handler."""
        handler = self._handler_map.pop(anima_name, None)
        self._app_map.pop(anima_name, None)
        self._bot_user_ids.pop(anima_name, None)
        if handler is not None:
            try:
                await handler.close_async()
            except Exception:
                logger.exception(
                    "Error closing Socket Mode handler for '%s'",
                    anima_name,
                )
        logger.info("Per-Anima Slack bot removed: %s", anima_name)

    @staticmethod
    def _discover_per_anima_bots() -> list[str]:
        """Scan vault/shared credentials/env for SLACK_BOT_TOKEN__* keys."""
        found: set[str] = set()
        prefix = "SLACK_BOT_TOKEN__"

        try:
            from core.config.vault import get_vault_manager

            vm = get_vault_manager()
            data = vm.load_vault()
            shared_section = data.get("shared") or {}
            for key in shared_section:
                if key.startswith(prefix):
                    found.add(key[len(prefix) :])
        except Exception:
            pass

        try:
            cred_file = get_data_dir() / "shared" / "credentials.json"
            if cred_file.is_file():
                data = json.loads(cred_file.read_text(encoding="utf-8"))
                for key in data:
                    if key.startswith(prefix):
                        found.add(key[len(prefix) :])
        except Exception:
            pass

        # Also scan environment variables (populated from .env via dotenv).
        # On Windows os.environ uppercases keys, so normalise to lowercase.
        for key, val in os.environ.items():
            if key.startswith(prefix) and val:
                found.add(key[len(prefix) :].lower())

        return sorted(found)

    @staticmethod
    def _get_per_anima_credential(base_key: str, anima_name: str) -> str | None:
        """Resolve a per-Anima credential (e.g. SLACK_BOT_TOKEN__sumire).

        Cascade: vault → shared/credentials.json → environment variable.
        """
        key = f"{base_key}__{anima_name}"
        token = _lookup_vault_credential(key)
        if token:
            return token
        token = _lookup_shared_credentials(key)
        if token:
            return token
        return os.environ.get(key) or None

    def _register_per_anima_handler(self, app: AsyncApp, anima_name: str, bot_user_id: str = "") -> None:
        """Register event handler that routes messages to a specific Anima.

        Routing rules:
        - **Self-posted messages** (from any registered bot): Ignored to
          prevent infinite response loops.
        - **DM** (channel starts with ``D``): Always deliver to inbox.
        - **Channel message with @mention of this bot**: Deliver to inbox.
        - **Channel message without @mention**:
          - If this Anima is the ``default_anima`` (e.g. sakura/COO):
            deliver to inbox (she responds as the channel's primary contact).
          - Otherwise: board routing only (no inbox delivery).
        """

        # Reference to all known bot UIDs (shared dict, updated at runtime)
        all_bot_uids = self._bot_user_ids

        @app.event("message")
        async def handle_message(event: dict, say) -> None:  # noqa: ARG001
            if "subtype" in event:
                return

            sender = event.get("user", "")
            is_own_bot = sender and sender in all_bot_uids.values()

            ts = event.get("ts", "")
            channel_id = event.get("channel", "")

            # ── Own bot messages: forward to board only (no inbox) ──
            if is_own_bot:
                if not channel_id.startswith("D"):
                    # Resolve the anima name from the bot UID for the board post
                    bot_name = next(
                        (n for n, uid in all_bot_uids.items() if uid == sender),
                        "anima",
                    )
                    token_for_resolve = self._get_per_anima_credential("SLACK_BOT_TOKEN", anima_name) or ""
                    raw_text = event.get("text", "")
                    text = await asyncio.to_thread(_resolve_slack_mentions, raw_text, token_for_resolve)
                    _route_to_board(channel_id, text, bot_name, ts=ts)
                return

            if _is_duplicate_ts(ts, anima_name):
                return

            token = self._get_per_anima_credential("SLACK_BOT_TOKEN", anima_name) or ""

            try:
                from core.notification.reply_routing import route_thread_reply

                if route_thread_reply(event, get_data_dir() / "shared", slack_token=token):
                    return
            except Exception:
                logger.debug("Reply routing lookup failed", exc_info=True)

            text = event.get("text", "")
            channel_id = event.get("channel", "")
            thread_ts = event.get("thread_ts", "")
            is_dm = channel_id.startswith("D")
            mentioned_other_registered_bot = _mentions_registered_bot(
                text,
                all_bot_uids,
                exclude_names={anima_name, "__shared__"},
            )

            alias_ids = _load_alias_user_ids()
            mention_intent = _detect_mention_intent(text, bot_user_id, alias_ids)

            if thread_ts:
                ctx = await asyncio.to_thread(_fetch_thread_context, token, channel_id, thread_ts)
                if ctx:
                    text = ctx + text

            text = await asyncio.to_thread(_resolve_slack_mentions, text, token)

            # ── Decide whether to deliver to inbox ──
            # DM: always.  @mention: always.
            # Channel without mention: only if this anima is the default_anima
            # (e.g. sakura as COO — the primary responder for all channels).
            is_default = False
            if not is_dm and not mention_intent and not mentioned_other_registered_bot:
                try:
                    cfg = load_config()
                    is_default = (anima_name == cfg.external_messaging.slack.default_anima)
                except Exception:
                    pass

            should_deliver = is_dm or bool(mention_intent) or is_default

            # ── Board routing: forward channel message to board exactly once ──
            # Only the handler that "owns" this message forwards to the board:
            #   - @mention present → the mentioned bot's handler
            #   - no mention → default_anima's handler
            # This prevents N handlers from each posting the same message.
            if not is_dm and (bool(mention_intent) or is_default):
                user_name = _get_cached_user_name(sender) or sender
                _route_to_board(channel_id, text, user_name, ts=ts)

            if should_deliver:
                has_mention = bool(mention_intent)
                annotation = _build_slack_annotation(channel_id, has_mention)
                annotated = annotation + text
                intent = mention_intent if mention_intent else "question"

                shared_dir = get_data_dir() / "shared"
                messenger = Messenger(shared_dir, anima_name)
                messenger.receive_external(
                    content=annotated,
                    source="slack",
                    source_message_id=ts,
                    external_user_id=sender,
                    external_channel_id=channel_id,
                    external_thread_ts=thread_ts,
                    intent=intent,
                )

                logger.info(
                    "Per-Anima Socket Mode message routed: channel=%s -> anima=%s (intent=%s, dm=%s, default=%s)",
                    channel_id,
                    anima_name,
                    intent,
                    is_dm,
                    is_default,
                )
            else:
                logger.debug(
                    "Per-Anima %s: channel msg without mention, board-only: channel=%s",
                    anima_name,
                    channel_id,
                )

        @app.event("app_mention")
        async def handle_app_mention(event: dict, say) -> None:  # noqa: ARG001
            # Ignore self-posted mentions (prevent loops)
            sender = event.get("user", "")
            if sender and sender in all_bot_uids.values():
                return

            ts = event.get("ts", "")
            if _is_duplicate_ts(ts, anima_name):
                return

            text = event.get("text", "")
            channel_id = event.get("channel", "")
            thread_ts = event.get("thread_ts", "")

            if thread_ts:
                token = self._get_per_anima_credential("SLACK_BOT_TOKEN", anima_name) or ""
                ctx = await asyncio.to_thread(_fetch_thread_context, token, channel_id, thread_ts)
                if ctx:
                    text = ctx + text

            _mention_token = self._get_per_anima_credential("SLACK_BOT_TOKEN", anima_name) or ""
            text = await asyncio.to_thread(_resolve_slack_mentions, text, _mention_token)
            annotation = _build_slack_annotation(channel_id, True)
            text = annotation + text

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name)
            messenger.receive_external(
                content=text,
                source="slack",
                source_message_id=ts,
                external_user_id=event.get("user", ""),
                external_channel_id=channel_id,
                external_thread_ts=thread_ts,
                intent="question",
            )
            logger.info(
                "Per-Anima Socket Mode app_mention routed: channel=%s -> anima=%s",
                channel_id,
                anima_name,
            )

    def _register_shared_handler(
        self,
        app: AsyncApp,
        bot_user_id: str = "",
    ) -> None:
        """Register event handler for the shared bot (channel-based routing).

        The ``anima_mapping`` is resolved dynamically via ``load_config()``
        on every incoming message so that config changes take effect without
        reconnecting the Socket Mode handler.
        """

        # Reference to all known bot UIDs for self-message filtering
        all_bot_uids = self._bot_user_ids

        @app.event("message")
        async def handle_message(event: dict, say) -> None:  # noqa: ARG001
            if "subtype" in event:
                return

            # Ignore messages from any of our own bots (prevent loops)
            sender = event.get("user", "")
            if sender and sender in all_bot_uids.values():
                return

            ts = event.get("ts", "")
            if _is_duplicate_ts(ts, "__shared__"):
                return

            _shared_token = get_credential("slack", "slack_webhook", env_var="SLACK_BOT_TOKEN") or ""

            try:
                from core.notification.reply_routing import route_thread_reply

                if route_thread_reply(event, get_data_dir() / "shared", slack_token=_shared_token):
                    return
            except Exception:
                logger.debug("Reply routing lookup failed", exc_info=True)

            channel_id = event.get("channel", "")
            cfg = load_config()
            slack_cfg = cfg.external_messaging.slack
            anima_name = slack_cfg.anima_mapping.get(channel_id) or slack_cfg.default_anima
            if not anima_name:
                logger.debug(
                    "No anima mapping for channel %s and no default_anima; ignoring",
                    channel_id,
                )
                return

            text = event.get("text", "")
            thread_ts = event.get("thread_ts", "")
            mentioned_per_anima_bot = _mentions_registered_bot(
                text,
                all_bot_uids,
                exclude_names={"__shared__"},
            )

            # Let the explicitly mentioned per-Anima bot handle this channel
            # message so the shared default route does not steal it.
            if channel_id and not channel_id.startswith("D") and mentioned_per_anima_bot:
                logger.debug(
                    "Shared Slack handler skipped due to per-Anima bot mention: channel=%s",
                    channel_id,
                )
                return

            alias_ids = _load_alias_user_ids()
            mention_intent = _detect_mention_intent(text, bot_user_id, alias_ids)

            if thread_ts:
                ctx = await asyncio.to_thread(_fetch_thread_context, _shared_token, channel_id, thread_ts)
                if ctx:
                    text = ctx + text

            text = await asyncio.to_thread(_resolve_slack_mentions, text, _shared_token)

            # Route to AnimaWorks board BEFORE adding inbox annotations
            user_name = _get_cached_user_name(event.get("user", "")) or event.get("user", "")
            _route_to_board(channel_id, text, user_name, ts=ts)

            has_mention = bool(mention_intent)
            annotation = _build_slack_annotation(channel_id, has_mention)
            annotated = annotation + text
            # Shared-handler delivery already means this message was targeted by
            # channel mapping or default_anima selection, so treat it as
            # actionable even when it is not an explicit @mention/DM.
            intent = mention_intent or _detect_slack_intent(annotated, channel_id, bot_user_id) or "question"

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name)
            messenger.receive_external(
                content=annotated,
                source="slack",
                source_message_id=ts,
                external_user_id=event.get("user", ""),
                external_channel_id=channel_id,
                external_thread_ts=thread_ts,
                intent=intent,
            )

            logger.info(
                "Shared Socket Mode message routed: channel=%s -> anima=%s (intent=%s)",
                channel_id,
                anima_name,
                intent or "none",
            )

        @app.event("app_mention")
        async def handle_app_mention(event: dict, say) -> None:  # noqa: ARG001
            ts = event.get("ts", "")
            if _is_duplicate_ts(ts, "__shared__"):
                return

            channel_id = event.get("channel", "")
            cfg = load_config()
            slack_cfg = cfg.external_messaging.slack
            anima_name_resolved = slack_cfg.anima_mapping.get(channel_id) or slack_cfg.default_anima
            if not anima_name_resolved:
                logger.debug(
                    "No anima mapping for channel %s (app_mention); ignoring",
                    channel_id,
                )
                return

            text = event.get("text", "")
            thread_ts = event.get("thread_ts", "")

            if thread_ts:
                token = get_credential("slack", "slack_webhook", env_var="SLACK_BOT_TOKEN") or ""
                ctx = await asyncio.to_thread(_fetch_thread_context, token, channel_id, thread_ts)
                if ctx:
                    text = ctx + text

            _shared_tok = get_credential("slack", "slack_webhook", env_var="SLACK_BOT_TOKEN") or ""
            text = await asyncio.to_thread(_resolve_slack_mentions, text, _shared_tok)
            annotation = _build_slack_annotation(channel_id, True)
            text = annotation + text

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name_resolved)
            messenger.receive_external(
                content=text,
                source="slack",
                source_message_id=ts,
                external_user_id=event.get("user", ""),
                external_channel_id=channel_id,
                external_thread_ts=thread_ts,
                intent="question",
            )
            logger.info(
                "Shared Socket Mode app_mention routed: channel=%s -> anima=%s",
                channel_id,
                anima_name_resolved,
            )

    @staticmethod
    def _register_error_handler(app: AsyncApp) -> None:
        """Suppress Bolt 404 for Slack events we intentionally don't handle."""

        @app.error
        async def _handle_bolt_error(error, body) -> BoltResponse | None:
            if isinstance(error, BoltUnhandledRequestError):
                logger.debug(
                    "Ignoring unhandled Slack event: %s",
                    (body or {}).get("event", {}).get("type", "unknown"),
                )
                return BoltResponse(status=200, body="")
            raise error

    async def stop(self) -> None:
        """Disconnect all Socket Mode handlers gracefully."""
        for handler in self._handler_map.values():
            try:
                await handler.close_async()
            except Exception:
                logger.exception("Error closing Socket Mode handler")
        self._handler_map.clear()
        self._app_map.clear()
        self._bot_user_ids.clear()
        logger.info("Slack Socket Mode disconnected")

    async def health_check(self) -> dict[str, Any]:
        """Call ``auth.test`` on each active handler and return connection status."""
        results: dict[str, Any] = {}
        for name, app in self._app_map.items():
            try:
                resp = await asyncio.wait_for(app.client.auth_test(), timeout=10)
                results[name] = {
                    "ok": True,
                    "bot_user_id": self._bot_user_ids.get(name, ""),
                    "team": resp.get("team", ""),
                }
            except Exception as exc:
                results[name] = {"ok": False, "error": str(exc)}
        return results

    @property
    def is_connected(self) -> bool:
        """Return whether any Socket Mode handler is active."""
        return len(self._handler_map) > 0
