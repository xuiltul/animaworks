"""Unit tests for server/routes/webhooks.py — Slack & Chatwork webhook endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config.models import (
    AnimaWorksConfig,
    ExternalMessagingChannelConfig,
    ExternalMessagingConfig,
)
from core.schemas import Message
from server.routes.webhooks import create_webhooks_router

# ── Test App Setup ────────────────────────────────────────


@pytest.fixture
def app():
    """Create a minimal FastAPI app with the webhooks router."""
    app = FastAPI()
    api_router = create_webhooks_router()
    app.include_router(api_router, prefix="/api")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ── Helpers ───────────────────────────────────────────────

SIGNING_SECRET = "test_signing_secret_1234"


def _make_slack_signature(body: bytes, timestamp: str) -> str:
    """Generate a valid Slack signature for testing."""
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    return "v0=" + hmac.new(
        SIGNING_SECRET.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _slack_headers(body: bytes) -> dict[str, str]:
    """Generate valid Slack request headers."""
    ts = str(int(time.time()))
    return {
        "X-Slack-Signature": _make_slack_signature(body, ts),
        "X-Slack-Request-Timestamp": ts,
    }


# ── Config Models ─────────────────────────────────────────


class TestExternalMessagingConfig:
    def test_default_disabled(self):
        config = ExternalMessagingConfig()
        assert config.slack.enabled is False
        assert config.chatwork.enabled is False

    def test_slack_enabled_with_mapping(self):
        config = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                enabled=True,
                anima_mapping={"C123": "sakura"},
            ),
        )
        assert config.slack.enabled is True
        assert config.slack.anima_mapping["C123"] == "sakura"

    def test_chatwork_enabled_with_mapping(self):
        config = ExternalMessagingConfig(
            chatwork=ExternalMessagingChannelConfig(
                enabled=True,
                anima_mapping={"456": "kotoha"},
            ),
        )
        assert config.chatwork.enabled is True
        assert config.chatwork.anima_mapping["456"] == "kotoha"

    def test_in_animaworks_config(self):
        config = AnimaWorksConfig()
        assert hasattr(config, "external_messaging")
        assert isinstance(config.external_messaging, ExternalMessagingConfig)

    def test_serialization_roundtrip(self):
        config = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                slack=ExternalMessagingChannelConfig(
                    enabled=True,
                    anima_mapping={"C123": "sakura"},
                ),
            ),
        )
        data = config.model_dump(mode="json")
        restored = AnimaWorksConfig.model_validate(data)
        assert restored.external_messaging.slack.enabled is True
        assert restored.external_messaging.slack.anima_mapping["C123"] == "sakura"


# ── Message Model Extension ──────────────────────────────


class TestMessageExternalFields:
    def test_default_source_is_anima(self):
        msg = Message(from_person="sakura", to_person="kotoha", content="hi")
        assert msg.source == "anima"
        assert msg.source_message_id == ""
        assert msg.external_user_id == ""
        assert msg.external_channel_id == ""

    def test_external_source_fields(self):
        msg = Message(
            from_person="slack:U123",
            to_person="sakura",
            content="hello",
            source="slack",
            source_message_id="1234567890.123456",
            external_user_id="U123",
            external_channel_id="C456",
        )
        assert msg.source == "slack"
        assert msg.source_message_id == "1234567890.123456"
        assert msg.external_user_id == "U123"
        assert msg.external_channel_id == "C456"

    def test_json_roundtrip(self):
        msg = Message(
            from_person="chatwork:12345",
            to_person="kotoha",
            content="task done",
            source="chatwork",
            source_message_id="msg_001",
            external_user_id="12345",
            external_channel_id="room_789",
        )
        data = json.loads(msg.model_dump_json())
        restored = Message(**data)
        assert restored.source == "chatwork"
        assert restored.source_message_id == "msg_001"
        assert restored.external_user_id == "12345"


# ── Slack Webhook ─────────────────────────────────────────


class TestSlackChallenge:
    def test_url_verification(self, client):
        """Slack URL verification challenge should be returned without signature check."""
        payload = {"type": "url_verification", "challenge": "abc123xyz"}
        resp = client.post("/api/webhooks/slack/events", json=payload)
        assert resp.status_code == 200
        assert resp.json() == {"challenge": "abc123xyz"}


class TestSlackSignatureVerification:
    @patch("server.routes.webhooks.get_credential", return_value=SIGNING_SECRET)
    @patch("server.routes.webhooks.load_config")
    def test_valid_signature_accepted(self, mock_config, mock_cred, client):
        mock_config.return_value = AnimaWorksConfig()
        payload = json.dumps({"type": "event_callback", "event": {"type": "reaction_added"}})
        body = payload.encode("utf-8")
        headers = _slack_headers(body)
        resp = client.post(
            "/api/webhooks/slack/events", content=body, headers=headers,
        )
        assert resp.status_code == 200

    @patch("server.routes.webhooks.get_credential", return_value=SIGNING_SECRET)
    def test_invalid_signature_rejected(self, mock_cred, client):
        payload = json.dumps({"type": "event_callback", "event": {"type": "message"}})
        headers = {
            "X-Slack-Signature": "v0=invalidsignature",
            "X-Slack-Request-Timestamp": str(int(time.time())),
        }
        resp = client.post(
            "/api/webhooks/slack/events",
            content=payload.encode("utf-8"),
            headers=headers,
        )
        assert resp.status_code == 400

    @patch("server.routes.webhooks.get_credential", return_value=SIGNING_SECRET)
    def test_old_timestamp_rejected(self, mock_cred, client):
        old_ts = str(int(time.time()) - 600)  # 10 minutes ago
        payload = json.dumps({"type": "event_callback", "event": {"type": "message"}})
        body = payload.encode("utf-8")
        sig = _make_slack_signature(body, old_ts)
        headers = {
            "X-Slack-Signature": sig,
            "X-Slack-Request-Timestamp": old_ts,
        }
        resp = client.post(
            "/api/webhooks/slack/events", content=body, headers=headers,
        )
        assert resp.status_code == 400


class TestSlackMessageDelivery:
    @patch("server.routes.webhooks.get_data_dir")
    @patch("server.routes.webhooks.get_credential", return_value=SIGNING_SECRET)
    @patch("server.routes.webhooks.load_config")
    def test_message_delivered_to_inbox(
        self, mock_config, mock_cred, mock_data_dir, client, tmp_path,
    ):
        # Setup data dir
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "sakura").mkdir(parents=True)
        mock_data_dir.return_value = tmp_path

        # Config with Slack enabled
        mock_config.return_value = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                slack=ExternalMessagingChannelConfig(
                    enabled=True,
                    anima_mapping={"C0ACT663B5L": "sakura"},
                ),
            ),
        )

        payload = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C0ACT663B5L",
                "user": "U999TEST",
                "text": "Hello Sakura!",
                "ts": "1234567890.123456",
            },
        })
        body = payload.encode("utf-8")
        headers = _slack_headers(body)

        resp = client.post(
            "/api/webhooks/slack/events", content=body, headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Verify message was written to inbox
        inbox_files = list((shared_dir / "inbox" / "sakura").glob("*.json"))
        assert len(inbox_files) == 1
        msg_data = json.loads(inbox_files[0].read_text(encoding="utf-8"))
        assert msg_data["content"] == "Hello Sakura!"
        assert msg_data["source"] == "slack"
        assert msg_data["external_user_id"] == "U999TEST"
        assert msg_data["external_channel_id"] == "C0ACT663B5L"
        assert msg_data["source_message_id"] == "1234567890.123456"

    @patch("server.routes.webhooks.get_credential", return_value=SIGNING_SECRET)
    @patch("server.routes.webhooks.load_config")
    def test_disabled_slack_ignores_message(self, mock_config, mock_cred, client):
        mock_config.return_value = AnimaWorksConfig()  # slack.enabled = False
        payload = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C123",
                "user": "U456",
                "text": "test",
                "ts": "1.1",
            },
        })
        body = payload.encode("utf-8")
        headers = _slack_headers(body)
        resp = client.post(
            "/api/webhooks/slack/events", content=body, headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    @patch("server.routes.webhooks.get_credential", return_value=SIGNING_SECRET)
    @patch("server.routes.webhooks.load_config")
    def test_unmapped_channel_ignored(self, mock_config, mock_cred, client):
        mock_config.return_value = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                slack=ExternalMessagingChannelConfig(
                    enabled=True,
                    anima_mapping={"C_OTHER": "kotoha"},
                ),
            ),
        )
        payload = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C_UNKNOWN",
                "user": "U456",
                "text": "test",
                "ts": "1.1",
            },
        })
        body = payload.encode("utf-8")
        headers = _slack_headers(body)
        resp = client.post(
            "/api/webhooks/slack/events", content=body, headers=headers,
        )
        assert resp.status_code == 200

    @patch("server.routes.webhooks.get_credential", return_value=SIGNING_SECRET)
    @patch("server.routes.webhooks.load_config")
    def test_message_with_subtype_ignored(self, mock_config, mock_cred, client):
        """Bot messages and edited messages (with subtype) should be ignored."""
        mock_config.return_value = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                slack=ExternalMessagingChannelConfig(
                    enabled=True,
                    anima_mapping={"C123": "sakura"},
                ),
            ),
        )
        payload = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "subtype": "bot_message",
                "channel": "C123",
                "text": "bot says hi",
                "ts": "1.1",
            },
        })
        body = payload.encode("utf-8")
        headers = _slack_headers(body)
        resp = client.post(
            "/api/webhooks/slack/events", content=body, headers=headers,
        )
        assert resp.status_code == 200


# ── Chatwork Webhook ──────────────────────────────────────

# Chatwork webhook token is Base64-encoded; the secret key is the decoded bytes.
# For tests, we use a known token and compute signatures accordingly.
CHATWORK_TOKEN_RAW = b"chatwork_test_secret_key_1234567"
CHATWORK_TOKEN = base64.b64encode(CHATWORK_TOKEN_RAW).decode("utf-8")


def _chatwork_signature(body: bytes) -> str:
    """Compute a valid Chatwork webhook signature for testing."""
    return base64.b64encode(
        hmac.new(CHATWORK_TOKEN_RAW, body, hashlib.sha256).digest(),
    ).decode("utf-8")


class TestChatworkSignatureVerification:
    @patch("server.routes.webhooks.get_credential", return_value=CHATWORK_TOKEN)
    @patch("server.routes.webhooks.load_config")
    def test_valid_signature_accepted(self, mock_config, mock_cred, client):
        mock_config.return_value = AnimaWorksConfig()
        body = json.dumps(
            {"webhook_event_type": "room_deleted", "webhook_event": {}},
        ).encode("utf-8")
        resp = client.post(
            "/api/webhooks/chatwork",
            content=body,
            headers={"X-ChatWorkWebhookSignature": _chatwork_signature(body)},
        )
        assert resp.status_code == 200

    @patch("server.routes.webhooks.get_credential", return_value=CHATWORK_TOKEN)
    def test_invalid_signature_rejected(self, mock_cred, client):
        body = json.dumps(
            {"webhook_event_type": "message_created", "webhook_event": {}},
        ).encode("utf-8")
        resp = client.post(
            "/api/webhooks/chatwork",
            content=body,
            headers={"X-ChatWorkWebhookSignature": "wrong_signature"},
        )
        assert resp.status_code == 400


class TestChatworkMessageDelivery:
    @patch("server.routes.webhooks.get_data_dir")
    @patch("server.routes.webhooks.get_credential", return_value=CHATWORK_TOKEN)
    @patch("server.routes.webhooks.load_config")
    def test_message_delivered_to_inbox(
        self, mock_config, mock_cred, mock_data_dir, client, tmp_path,
    ):
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "kotoha").mkdir(parents=True)
        mock_data_dir.return_value = tmp_path

        mock_config.return_value = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                chatwork=ExternalMessagingChannelConfig(
                    enabled=True,
                    anima_mapping={"12345": "kotoha"},
                ),
            ),
        )

        body = json.dumps({
            "webhook_event_type": "message_created",
            "webhook_event": {
                "room_id": 12345,
                "message_id": "msg_001",
                "body": "Hi Kotoha!",
                "account": {"account_id": 67890, "name": "taka"},
            },
        }).encode("utf-8")
        resp = client.post(
            "/api/webhooks/chatwork",
            content=body,
            headers={"X-ChatWorkWebhookSignature": _chatwork_signature(body)},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        inbox_files = list((shared_dir / "inbox" / "kotoha").glob("*.json"))
        assert len(inbox_files) == 1
        msg_data = json.loads(inbox_files[0].read_text(encoding="utf-8"))
        assert msg_data["content"] == "Hi Kotoha!"
        assert msg_data["source"] == "chatwork"
        assert msg_data["external_user_id"] == "67890"
        assert msg_data["external_channel_id"] == "12345"
        assert msg_data["source_message_id"] == "msg_001"

    @patch("server.routes.webhooks.get_credential", return_value=CHATWORK_TOKEN)
    @patch("server.routes.webhooks.load_config")
    def test_disabled_chatwork_ignores_message(self, mock_config, mock_cred, client):
        mock_config.return_value = AnimaWorksConfig()  # chatwork.enabled = False
        body = json.dumps({
            "webhook_event_type": "message_created",
            "webhook_event": {
                "room_id": 999,
                "body": "test",
                "account": {"account_id": 1},
            },
        }).encode("utf-8")
        resp = client.post(
            "/api/webhooks/chatwork",
            content=body,
            headers={"X-ChatWorkWebhookSignature": _chatwork_signature(body)},
        )
        assert resp.status_code == 200
