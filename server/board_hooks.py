from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Server-owned post-commit hooks for Board events and Slack sync."""

import asyncio
import logging
import threading
from typing import Any

from core.messaging.board import (
    BoardPost,
    PostResult,
    board_access,
    register_post_hook,
    unregister_post_hook,
)
from core.messaging.board_reads import register_read_hook, unregister_read_hook

logger = logging.getLogger("animaworks.server.board_hooks")

_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn(loop: asyncio.AbstractEventLoop, coro: Any) -> None:
    """Schedule *coro* on *loop* and keep a reference until it finishes."""
    task = loop.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def register_board_post_hooks(app: Any) -> None:
    loop = asyncio.get_running_loop()
    loop_thread_id = threading.get_ident()

    def _on_loop_thread() -> bool:
        # post_to_board is meant to run off the event loop, but a caller on the
        # loop thread must not block it waiting for its own coroutine.
        return threading.get_ident() == loop_thread_id

    def ws_event(post: BoardPost, result: PostResult) -> str:
        ws_manager = getattr(app.state, "ws_manager", None)
        if ws_manager is None:
            return "skipped"
        from server.events import emit_direct

        event = {
            "channel": post.channel,
            "from": post.from_person,
            "text": post.text,
            "source": post.source,
            "ts": post.ts,
            "id": post.id,
            "kind": post.kind,
        }
        if post.reply_to:
            event["reply_to"] = post.reply_to
        if result.notified or result.not_notified:
            event["notified"] = list(result.notified)
            event["not_notified"] = dict(result.not_notified)
        if _on_loop_thread():
            _spawn(loop, emit_direct(ws_manager, "board.post", event))
            return "sent"
        future = asyncio.run_coroutine_threadsafe(emit_direct(ws_manager, "board.post", event), loop)
        try:
            future.result(timeout=10)
        except Exception:
            future.cancel()
            logger.exception("Failed to emit board.post for #%s", post.channel)
            return "failed"
        return "sent"

    register_post_hook("ws_event", ws_event)

    def ws_read_event(payload: dict[str, Any]) -> str:
        ws_manager = getattr(app.state, "ws_manager", None)
        if ws_manager is None:
            return "skipped"
        from server.events import emit_direct

        event = {
            "channel": payload.get("channel", ""),
            "reader": payload.get("reader", ""),
            "upto_id": payload.get("upto_id", ""),
            "upto_ts": payload.get("upto_ts", ""),
            "picked_id": payload.get("picked_id", ""),
        }
        channel = event["channel"]
        if _on_loop_thread():
            _spawn(loop, emit_direct(ws_manager, "board.read", event))
            return "sent"
        future = asyncio.run_coroutine_threadsafe(emit_direct(ws_manager, "board.read", event), loop)
        try:
            future.result(timeout=10)
        except Exception:
            future.cancel()
            logger.exception("Failed to emit board.read for #%s", channel)
            return "failed"
        return "sent"

    register_read_hook("ws_read_event", ws_read_event)

    try:
        from core.config.models import load_config

        slack_enabled = bool(load_config().external_messaging.slack.enabled)
    except Exception:
        logger.warning("Could not read Slack settings while registering Board hooks", exc_info=True)
        slack_enabled = False

    if slack_enabled:

        def slack_outbound(post: BoardPost, _result: PostResult) -> str:
            from core.platform.env import get_env

            if post.source in {"slack", "discord"} or post.origin == "mirror":
                return "skipped"
            if get_env("ANIMAWORKS_DISABLE_EXTERNAL_SYNC") == "1" or bool(
                getattr(app.state, "disable_external_sync", False)
            ):
                return "skipped"
            shared_dir = app.state.shared_dir
            decision = board_access(shared_dir, post.channel, post.principal, "outbound")
            if not decision.allowed:
                return "skipped"

            from core.messaging.messenger import load_channel_meta_state

            state, meta = load_channel_meta_state(shared_dir, post.channel)
            if state == "unreadable":
                return "skipped"
            try:
                from core.config.models import load_config

                slack_cfg = load_config().external_messaging.slack
            except Exception:
                logger.exception("Could not read Slack settings for Board post #%s", post.channel)
                return "failed"
            if meta is not None and meta.members and post.channel not in slack_cfg.board_outbound_sync:
                return "skipped"

            from core.messaging.outbound_auto import BoardSlackSync

            coro = BoardSlackSync().sync_board_post(
                board_name=post.channel,
                text=post.text,
                from_person=post.from_person,
                source=post.source,
            )
            if _on_loop_thread():
                _spawn(loop, coro)
                return "sent"
            try:
                return asyncio.run(coro)
            except Exception:
                logger.exception("Board → Slack sync failed for #%s", post.channel)
                return "failed"

        register_post_hook("slack_outbound", slack_outbound)


def unregister_board_post_hooks() -> None:
    unregister_post_hook("ws_event")
    unregister_post_hook("slack_outbound")
    unregister_read_hook("ws_read_event")
