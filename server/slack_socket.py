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

# ── Dedup: prevent same Slack message from being delivered twice ──
# When both message and app_mention events fire for a single @-mention,
# the first handler to process stores the ts; the second skips it.
_DEDUP_TTL_SEC = 10
_recent_ts: collections.OrderedDict[str, float] = collections.OrderedDict()


def _is_duplicate_ts(ts: str) -> bool:
    """Return True if *ts* was already processed within the TTL window."""
    now = time.monotonic()
    while _recent_ts and next(iter(_recent_ts.values())) < now - _DEDUP_TTL_SEC:
        _recent_ts.popitem(last=False)
    if ts in _recent_ts:
        return True
    _recent_ts[ts] = now
    return False


def _detect_slack_intent(text: str, channel_id: str, bot_user_id: str) -> str:
    """Return ``"question"`` if the message is a DM or mentions the bot."""
    if channel_id.startswith("D"):
        return "question"
    if bot_user_id and f"<@{bot_user_id}>" in (text or ""):
        return "question"
    return ""


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
        from core.tools.slack import SlackClient

        client = SlackClient(token=token)
        replies = client.thread_replies(channel_id, thread_ts)
        if len(replies) <= 1:
            return ""
        parent = replies[0]
        parent_user = parent.get("user", "unknown")
        parent_text = parent.get("text", "").replace("\n", " ")[:_THREAD_CTX_SUMMARY_LIMIT]
        reply_count = len(replies) - 1
        lines = [
            "[Thread context — this message is a reply in a Slack thread]",
            f"  <@{parent_user}>: {parent_text}",
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

        for anima_name in self._discover_per_anima_bots():
            await self._add_per_anima_handler(anima_name)

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

        if self._handler_map:
            await asyncio.gather(*(h.connect_async() for h in self._handler_map.values()))
            logger.info(
                "Slack Socket Mode connected (%d handler(s))",
                len(self._handler_map),
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

    async def _add_per_anima_handler(self, anima_name: str) -> bool:
        """Create and connect a per-Anima Socket Mode handler.

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
        """Scan vault/shared credentials for SLACK_BOT_TOKEN__* keys."""
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

        return sorted(found)

    @staticmethod
    def _get_per_anima_credential(base_key: str, anima_name: str) -> str | None:
        """Resolve a per-Anima credential (e.g. SLACK_BOT_TOKEN__sumire)."""
        key = f"{base_key}__{anima_name}"
        token = _lookup_vault_credential(key)
        if token:
            return token
        return _lookup_shared_credentials(key)

    def _register_per_anima_handler(self, app: AsyncApp, anima_name: str, bot_user_id: str = "") -> None:
        """Register event handler that routes all messages to a specific Anima."""

        @app.event("message")
        async def handle_message(event: dict, say) -> None:  # noqa: ARG001
            if "subtype" in event:
                return

            ts = event.get("ts", "")
            if _is_duplicate_ts(ts):
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
            intent = _detect_slack_intent(text, channel_id, bot_user_id)

            if thread_ts:
                ctx = await asyncio.to_thread(_fetch_thread_context, token, channel_id, thread_ts)
                if ctx:
                    text = ctx + text

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name)
            messenger.receive_external(
                content=text,
                source="slack",
                source_message_id=ts,
                external_user_id=event.get("user", ""),
                external_channel_id=channel_id,
                external_thread_ts=thread_ts,
                intent=intent,
            )
            logger.info(
                "Per-Anima Socket Mode message routed: channel=%s -> anima=%s (intent=%s)",
                channel_id,
                anima_name,
                intent or "none",
            )

        @app.event("app_mention")
        async def handle_app_mention(event: dict, say) -> None:  # noqa: ARG001
            ts = event.get("ts", "")
            if _is_duplicate_ts(ts):
                return

            text = event.get("text", "")
            channel_id = event.get("channel", "")
            thread_ts = event.get("thread_ts", "")

            if thread_ts:
                token = self._get_per_anima_credential("SLACK_BOT_TOKEN", anima_name) or ""
                ctx = await asyncio.to_thread(_fetch_thread_context, token, channel_id, thread_ts)
                if ctx:
                    text = ctx + text

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

        @app.event("message")
        async def handle_message(event: dict, say) -> None:  # noqa: ARG001
            if "subtype" in event:
                return

            ts = event.get("ts", "")
            if _is_duplicate_ts(ts):
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
            intent = _detect_slack_intent(text, channel_id, bot_user_id)

            if thread_ts:
                ctx = await asyncio.to_thread(_fetch_thread_context, _shared_token, channel_id, thread_ts)
                if ctx:
                    text = ctx + text

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name)
            messenger.receive_external(
                content=text,
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
            if _is_duplicate_ts(ts):
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

    @property
    def is_connected(self) -> bool:
        """Return whether any Socket Mode handler is active."""
        return len(self._handler_map) > 0
