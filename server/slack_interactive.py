from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Slack Bolt interactive handlers for call_human interactive mode (Block Kit + modals)."""

import re
from typing import Any

try:
    from slack_bolt.app.async_app import AsyncApp
except ModuleNotFoundError:
    AsyncApp = Any  # type: ignore[misc, assignment]


def _build_comment_modal(callback_id: str) -> dict[str, Any]:
    """Slack modal view for optional comment on interactive approval."""
    from core.i18n import t

    return {
        "type": "modal",
        "callback_id": f"aw_interact_comment:{callback_id}",
        "title": {"type": "plain_text", "text": t("interactive.comment_modal_title")},
        "submit": {"type": "plain_text", "text": t("interactive.comment_modal_submit")},
        "blocks": [
            {
                "type": "input",
                "block_id": "comment_input",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "comment_text",
                    "multiline": True,
                },
                "label": {"type": "plain_text", "text": t("interactive.comment_modal_label")},
            }
        ],
    }


async def _handle_interactive_action(ack: Any, body: dict[str, Any], client: Any, log: Any) -> None:
    """Handle Slack Block Kit button presses for call_human interactive mode."""
    await ack()
    try:
        actions = body.get("actions") or []
        if not actions:
            log.warning("interactive block_actions missing actions: %s", body.get("type"))
            return
        action = actions[0]
        callback_id = str(action.get("value") or "")
        action_id = str(action.get("action_id") or "")
        if not action_id.startswith("aw_interact_"):
            return
        option = action_id.removeprefix("aw_interact_")

        user_blob = body.get("user") or {}
        user_id = str(user_blob.get("id") or "")
        user_name = str(user_blob.get("username") or user_blob.get("name") or user_id)

        channel_blob = body.get("channel") or {}
        channel_id = str(channel_blob.get("id") or "")

        from core.i18n import t
        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()

        try:
            req, status = await router.lookup_verbose(callback_id)
        except Exception:
            log.exception("InteractionRouter.lookup failed for callback_id=%s", callback_id)
            if channel_id and user_id:
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text=t("interactive.expired"),
                    )
                except Exception:
                    log.debug("chat_postEphemeral after lookup error failed", exc_info=True)
            return

        if req is None:
            log.warning(
                "Interactive request inactive: callback_id=%s status=%s",
                callback_id,
                status,
            )
            inactive_key = {
                "resolved": "interactive.already_resolved",
                "expired": "interactive.expired",
            }.get(status, "interactive.not_found")
            if channel_id and user_id:
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text=t(inactive_key),
                    )
                except Exception:
                    log.debug("chat_postEphemeral for inactive interaction failed", exc_info=True)
            return

        allowed = req.allowed_users.get("slack", [])
        if allowed and user_id not in allowed:
            if channel_id and user_id:
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text=t("interactive.unauthorized"),
                    )
                except Exception:
                    log.debug("chat_postEphemeral for unauthorized failed", exc_info=True)
            return

        if option == "comment":
            trigger_id = body.get("trigger_id")
            if not trigger_id:
                log.warning("comment option but missing trigger_id")
                return
            try:
                await client.views_open(
                    trigger_id=trigger_id,
                    view=_build_comment_modal(callback_id),
                )
            except Exception:
                log.exception("views_open for comment modal failed")
            return

        message_blob = body.get("message") or {}
        message_ts = str(message_blob.get("ts") or "")

        try:
            result = await router.resolve(
                callback_id,
                decision=option,
                actor=user_name,
                source="slack",
            )
        except Exception:
            log.exception("InteractionRouter.resolve failed for callback_id=%s", callback_id)
            if channel_id and user_id:
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text=t("interactive.already_resolved"),
                    )
                except Exception:
                    log.debug("chat_postEphemeral after resolve error failed", exc_info=True)
            return

        if result is None:
            if channel_id and user_id:
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text=t("interactive.already_resolved"),
                    )
                except Exception:
                    log.debug("chat_postEphemeral for already_resolved failed", exc_info=True)
            return

        resolved_text = t("interactive.resolved_by", actor=user_name, decision=option)
        if channel_id and message_ts:
            try:
                await client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=resolved_text,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": resolved_text,
                            },
                        }
                    ],
                )
            except Exception:
                log.exception("chat_update after interactive resolve failed")
    except Exception:
        log.exception("Unexpected error in _handle_interactive_action")


async def _handle_comment_submission(ack: Any, body: dict[str, Any], client: Any, log: Any) -> None:
    """Handle Slack modal submission for interactive comment."""
    await ack()
    try:
        view = body.get("view") or {}
        raw_cb = str(view.get("callback_id") or "")
        if not raw_cb.startswith("aw_interact_comment:"):
            return
        callback_id = raw_cb.removeprefix("aw_interact_comment:")

        user_blob = body.get("user") or {}
        user_id = str(user_blob.get("id") or "")
        user_name = str(user_blob.get("username") or user_blob.get("name") or user_id)

        comment_text = ""
        try:
            comment_text = str(view["state"]["values"]["comment_input"]["comment_text"].get("value") or "")
        except (KeyError, TypeError):
            log.warning("comment modal state shape unexpected for callback_id=%s", callback_id)

        from core.notification.interactive import get_interaction_router

        router = get_interaction_router()
        try:
            await router.resolve(
                callback_id,
                decision="comment",
                actor=user_name,
                source="slack",
                comment=comment_text,
            )
        except Exception:
            log.exception("InteractionRouter.resolve failed for comment callback_id=%s", callback_id)
    except Exception:
        log.exception("Unexpected error in _handle_comment_submission")


def register_interactive_handlers(app: AsyncApp) -> None:
    """Register Block Kit interactive handlers on *app* (per-Anima or shared)."""

    @app.action(re.compile(r"^aw_interact_"))
    async def _on_interactive_action(ack, body, client, logger) -> None:
        await _handle_interactive_action(ack, body, client, logger)

    @app.view(re.compile(r"^aw_interact_comment:"))
    async def _on_comment_submission(ack, body, client, logger) -> None:
        await _handle_comment_submission(ack, body, client, logger)
