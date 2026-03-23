from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Webhook endpoints for external messaging platform integration.

Receives push notifications from Slack and Chatwork, verifies signatures,
maps channels to Anima names, and delivers messages to Anima inboxes.
"""

import base64
import hashlib
import hmac
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from core.config.models import load_config
from core.messenger import Messenger
from core.paths import get_data_dir
from core.tools._base import ToolConfigError, get_credential

logger = logging.getLogger("animaworks.webhooks")

_slack_bot_user_ids: dict[str, str] = {}


def create_webhooks_router() -> APIRouter:
    """Create the webhooks API router."""
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    # ── Slack ──────────────────────────────────────────────

    def _verify_slack_signature(
        body: bytes,
        timestamp: str,
        signature: str,
        signing_secret: str | None = None,
    ) -> bool:
        """Verify Slack request signature using signing secret."""
        if signing_secret is None:
            try:
                signing_secret = get_credential(
                    "slack_signing",
                    "slack_webhook",
                    env_var="SLACK_SIGNING_SECRET",
                )
            except ToolConfigError:
                logger.error("SLACK_SIGNING_SECRET not configured")
                return False

        # Reject requests older than 5 minutes (replay attack prevention)
        try:
            if abs(time.time() - int(timestamp)) > 300:
                logger.warning("Slack request timestamp too old: %s", timestamp)
                return False
        except (ValueError, TypeError):
            return False

        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected = (
            "v0="
            + hmac.new(
                signing_secret.encode("utf-8"),
                sig_basestring.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )
        return hmac.compare_digest(expected, signature)

    def _resolve_per_anima_signing_secret(anima_name: str) -> str | None:
        """Resolve per-Anima signing secret from vault/shared credentials."""
        from core.tools._base import _lookup_shared_credentials, _lookup_vault_credential

        key = f"SLACK_SIGNING_SECRET__{anima_name}"
        secret = _lookup_vault_credential(key)
        if secret:
            return secret
        return _lookup_shared_credentials(key)

    @router.post("/slack/events")
    async def slack_events(request: Request) -> dict:
        """Handle Slack Event Subscriptions.

        Supports:
        - URL verification challenge (initial setup)
        - message events → Anima inbox delivery
        """
        body = await request.body()
        data = json.loads(body)

        # Challenge response for Slack URL verification (no signature check)
        if data.get("type") == "url_verification":
            return {"challenge": data.get("challenge", "")}

        signature = request.headers.get("X-Slack-Signature", "")
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")

        config = load_config()
        slack_config = config.external_messaging.slack

        api_app_id = data.get("api_app_id", "")
        anima_from_app = slack_config.app_id_mapping.get(api_app_id) if api_app_id else None

        if anima_from_app:
            per_secret = _resolve_per_anima_signing_secret(anima_from_app)
            if not _verify_slack_signature(body, timestamp, signature, per_secret):
                raise HTTPException(status_code=400, detail="Invalid signature")
        else:
            if not _verify_slack_signature(body, timestamp, signature):
                raise HTTPException(status_code=400, detail="Invalid signature")

        if not slack_config.enabled:
            logger.debug("Slack webhook received but external_messaging.slack is disabled")
            return {"ok": True}

        # Process message events
        event = data.get("event", {})
        if event.get("type") == "message" and "subtype" not in event:
            channel_id = event.get("channel", "")
            if anima_from_app:
                anima_name = anima_from_app
            else:
                anima_name = slack_config.anima_mapping.get(channel_id) or slack_config.default_anima

            text = event.get("text", "")
            user_id = event.get("user", "")
            message_ts = event.get("ts", "")
            thread_ts = event.get("thread_ts", "")

            # Resolve bot token early (needed for both reply routing and thread context)
            cache_key = anima_from_app or "__shared__"
            bot_user_id = _slack_bot_user_ids.get(cache_key, "")
            _token: str | None = None
            if not bot_user_id and api_app_id:
                try:
                    from slack_sdk.web.async_client import AsyncWebClient

                    if anima_from_app:
                        from server.slack_socket import SlackSocketModeManager

                        _token = SlackSocketModeManager._get_per_anima_credential(
                            "SLACK_BOT_TOKEN",
                            anima_from_app,
                        )
                    if not _token:
                        _token = get_credential("slack", "slack_webhook", env_var="SLACK_BOT_TOKEN")
                    client = AsyncWebClient(token=_token)
                    resp = await client.auth_test()
                    bot_user_id = resp.get("user_id", "") or ""
                    if bot_user_id:
                        _slack_bot_user_ids[cache_key] = bot_user_id
                except Exception:
                    logger.debug("Failed to resolve bot user ID for webhook", exc_info=True)

            # Resolve token if not yet resolved (for thread context / reply routing)
            if _token is None:
                if anima_from_app:
                    from server.slack_socket import SlackSocketModeManager

                    _token = SlackSocketModeManager._get_per_anima_credential(
                        "SLACK_BOT_TOKEN",
                        anima_from_app,
                    )
                if not _token:
                    try:
                        _token = get_credential("slack", "slack_webhook", env_var="SLACK_BOT_TOKEN")
                    except Exception:
                        _token = ""

            # Thread-reply routing for call_human notifications
            try:
                from core.notification.reply_routing import route_thread_reply

                if route_thread_reply(event, get_data_dir() / "shared", slack_token=_token or ""):
                    return {"ok": True}
            except Exception:
                logger.debug("Reply routing lookup failed", exc_info=True)

            if not anima_name:
                logger.warning("No anima mapping for Slack channel %s and no default_anima", channel_id)
                return {"ok": True}

            import asyncio

            from server.slack_socket import (
                _build_slack_annotation,
                _detect_mention_intent,
                _detect_slack_intent,
                _fetch_thread_context,
                _load_alias_user_ids,
                _resolve_slack_mentions,
            )

            alias_ids = _load_alias_user_ids()
            mention_intent = _detect_mention_intent(text, bot_user_id, alias_ids)

            if thread_ts:
                ctx = await asyncio.to_thread(_fetch_thread_context, _token or "", channel_id, thread_ts)
                if ctx:
                    text = ctx + text

            text = await asyncio.to_thread(_resolve_slack_mentions, text, _token or "")
            has_mention = bool(mention_intent)
            annotation = _build_slack_annotation(channel_id, has_mention)
            text = annotation + text
            intent = mention_intent or _detect_slack_intent(text, channel_id, bot_user_id)

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name)
            messenger.receive_external(
                content=text,
                source="slack",
                source_message_id=message_ts,
                external_user_id=user_id,
                external_channel_id=channel_id,
                external_thread_ts=thread_ts,
                intent=intent,
            )
            logger.info(
                "Slack message delivered: channel=%s user=%s -> anima=%s (intent=%s)",
                channel_id,
                user_id,
                anima_name,
                intent or "none",
            )

        return {"ok": True}

    # ── Chatwork ───────────────────────────────────────────

    def _verify_chatwork_signature(body: bytes, signature: str) -> bool:
        """Verify Chatwork webhook signature using HMAC-SHA256.

        Chatwork signs the request body with HMAC-SHA256 using the
        Base64-decoded webhook token as the secret key.  The signature
        in the header is the Base64-encoded HMAC digest.
        """
        try:
            token = get_credential(
                "chatwork_webhook",
                "chatwork_webhook",
                env_var="CHATWORK_WEBHOOK_TOKEN",
            )
        except ToolConfigError:
            logger.error("CHATWORK_WEBHOOK_TOKEN not configured")
            return False

        secret_key = base64.b64decode(token)
        expected = base64.b64encode(
            hmac.new(secret_key, body, hashlib.sha256).digest(),
        ).decode("utf-8")
        return hmac.compare_digest(expected, signature)

    @router.post("/chatwork")
    async def chatwork_webhook(request: Request) -> dict:
        """Handle Chatwork Webhook notifications.

        Supports message_created events → Anima inbox delivery.
        """
        body = await request.body()

        # Signature verification (HMAC-SHA256)
        signature = request.headers.get("X-ChatWorkWebhookSignature", "")
        if not _verify_chatwork_signature(body, signature):
            raise HTTPException(status_code=400, detail="Invalid signature")

        data = json.loads(body)

        # Check if external messaging is enabled for Chatwork
        config = load_config()
        cw_config = config.external_messaging.chatwork
        if not cw_config.enabled:
            logger.debug("Chatwork webhook received but external_messaging.chatwork is disabled")
            return {"ok": True}

        # Process message_created events
        webhook_event = data.get("webhook_event", {})
        if data.get("webhook_event_type") == "message_created":
            room_id = str(webhook_event.get("room_id", ""))
            anima_name = cw_config.anima_mapping.get(room_id)
            if not anima_name:
                logger.warning("No anima mapping for Chatwork room %s", room_id)
                return {"ok": True}

            body_text = webhook_event.get("body", "")
            account = webhook_event.get("account", {})
            account_id = str(account.get("account_id", ""))
            message_id = str(webhook_event.get("message_id", ""))

            shared_dir = get_data_dir() / "shared"
            messenger = Messenger(shared_dir, anima_name)
            messenger.receive_external(
                content=body_text,
                source="chatwork",
                source_message_id=message_id,
                external_user_id=account_id,
                external_channel_id=room_id,
            )
            logger.info(
                "Chatwork message delivered: room=%s account=%s -> anima=%s",
                room_id,
                account_id,
                anima_name,
            )

        return {"ok": True}

    return router
