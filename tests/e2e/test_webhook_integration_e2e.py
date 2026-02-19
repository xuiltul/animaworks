# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for webhook integration — full flow from webhook to inbox.

Tests the complete pipeline: HTTP request → signature verification →
channel mapping → Messenger → inbox file creation, using real filesystem
and config (no mocks on Messenger or config loading).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config.models import AnimaWorksConfig, ExternalMessagingChannelConfig, ExternalMessagingConfig, save_config
from core.schemas import Message
from server.routes.webhooks import create_webhooks_router

SIGNING_SECRET = "e2e_test_signing_secret"
CHATWORK_TOKEN_RAW = b"e2e_chatwork_secret_key_for_test"
CHATWORK_TOKEN = base64.b64encode(CHATWORK_TOKEN_RAW).decode("utf-8")


@pytest.fixture
def e2e_app(data_dir, monkeypatch):
    """Create a full-stack test app with real data_dir and config."""
    # Patch get_data_dir to use test data_dir
    monkeypatch.setattr("server.routes.webhooks.get_data_dir", lambda: data_dir)
    # Patch get_credential to return test secrets
    def _mock_get_credential(credential_name, tool_name, **kwargs):
        creds = {
            "slack_signing": SIGNING_SECRET,
            "chatwork_webhook": CHATWORK_TOKEN,
        }
        if credential_name in creds:
            return creds[credential_name]
        raise Exception(f"Unknown credential: {credential_name}")
    monkeypatch.setattr("server.routes.webhooks.get_credential", _mock_get_credential)

    app = FastAPI()
    router = create_webhooks_router()
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
def e2e_client(e2e_app):
    return TestClient(e2e_app)


def _slack_sign(body: bytes, ts: str | None = None) -> dict[str, str]:
    ts = ts or str(int(time.time()))
    sig_base = f"v0:{ts}:{body.decode('utf-8')}"
    sig = "v0=" + hmac.new(
        SIGNING_SECRET.encode(), sig_base.encode(), hashlib.sha256,
    ).hexdigest()
    return {"X-Slack-Signature": sig, "X-Slack-Request-Timestamp": ts}


# ── Slack E2E ─────────────────────────────────────────────


class TestSlackWebhookE2E:
    def test_full_flow_message_to_inbox(self, e2e_client, data_dir, make_anima):
        """Full E2E: Slack event → inbox file with correct metadata."""
        # Setup: create anima and configure mapping
        make_anima("sakura")
        config = AnimaWorksConfig.model_validate(
            json.loads((data_dir / "config.json").read_text(encoding="utf-8")),
        )
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                enabled=True,
                anima_mapping={"C_E2E_TEST": "sakura"},
            ),
        )
        save_config(config, data_dir / "config.json")

        # Send Slack event
        payload = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C_E2E_TEST",
                "user": "U_E2E_USER",
                "text": "E2E test message from Slack",
                "ts": "9999999999.000001",
            },
        })
        body = payload.encode("utf-8")
        resp = e2e_client.post(
            "/api/webhooks/slack/events",
            content=body,
            headers=_slack_sign(body),
        )
        assert resp.status_code == 200

        # Verify inbox
        inbox = data_dir / "shared" / "inbox" / "sakura"
        files = list(inbox.glob("*.json"))
        assert len(files) == 1

        msg = Message.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert msg.content == "E2E test message from Slack"
        assert msg.source == "slack"
        assert msg.external_user_id == "U_E2E_USER"
        assert msg.external_channel_id == "C_E2E_TEST"
        assert msg.source_message_id == "9999999999.000001"
        assert msg.to_person == "sakura"
        assert "slack:" in msg.from_person

    def test_challenge_response(self, e2e_client):
        """Slack URL verification returns challenge value."""
        resp = e2e_client.post(
            "/api/webhooks/slack/events",
            json={"type": "url_verification", "challenge": "challenge_token_xyz"},
        )
        assert resp.status_code == 200
        assert resp.json()["challenge"] == "challenge_token_xyz"

    def test_disabled_slack_drops_silently(self, e2e_client, data_dir):
        """When slack is disabled, messages are accepted but not delivered."""
        # Default config has slack disabled
        payload = json.dumps({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C_ANY",
                "user": "U_ANY",
                "text": "should be dropped",
                "ts": "1.1",
            },
        })
        body = payload.encode("utf-8")
        resp = e2e_client.post(
            "/api/webhooks/slack/events",
            content=body,
            headers=_slack_sign(body),
        )
        assert resp.status_code == 200

    def test_multiple_messages_sequential(self, e2e_client, data_dir, make_anima):
        """Multiple messages create separate inbox files."""
        make_anima("kotoha")
        config = AnimaWorksConfig.model_validate(
            json.loads((data_dir / "config.json").read_text(encoding="utf-8")),
        )
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                enabled=True,
                anima_mapping={"C_MULTI": "kotoha"},
            ),
        )
        save_config(config, data_dir / "config.json")

        for i in range(3):
            payload = json.dumps({
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "channel": "C_MULTI",
                    "user": "U_MULTI",
                    "text": f"message {i}",
                    "ts": f"100000000{i}.00000{i}",
                },
            })
            body = payload.encode("utf-8")
            # Small delay to ensure unique message IDs
            import time as _t
            _t.sleep(0.01)
            resp = e2e_client.post(
                "/api/webhooks/slack/events",
                content=body,
                headers=_slack_sign(body),
            )
            assert resp.status_code == 200

        inbox = data_dir / "shared" / "inbox" / "kotoha"
        files = list(inbox.glob("*.json"))
        assert len(files) == 3


# ── Chatwork E2E ──────────────────────────────────────────


def _chatwork_sign(body: bytes) -> str:
    """Compute a valid Chatwork webhook signature for E2E testing."""
    return base64.b64encode(
        hmac.new(CHATWORK_TOKEN_RAW, body, hashlib.sha256).digest(),
    ).decode("utf-8")


class TestChatworkWebhookE2E:
    def test_full_flow_message_to_inbox(self, e2e_client, data_dir, make_anima):
        """Full E2E: Chatwork event → inbox file with correct metadata."""
        make_anima("kotoha")
        config = AnimaWorksConfig.model_validate(
            json.loads((data_dir / "config.json").read_text(encoding="utf-8")),
        )
        config.external_messaging = ExternalMessagingConfig(
            chatwork=ExternalMessagingChannelConfig(
                enabled=True,
                anima_mapping={"99999": "kotoha"},
            ),
        )
        save_config(config, data_dir / "config.json")

        body = json.dumps({
            "webhook_event_type": "message_created",
            "webhook_event": {
                "room_id": 99999,
                "message_id": "e2e_msg_001",
                "body": "E2E test from Chatwork",
                "account": {"account_id": 54321, "name": "taka"},
            },
        }).encode("utf-8")
        resp = e2e_client.post(
            "/api/webhooks/chatwork",
            content=body,
            headers={"X-ChatWorkWebhookSignature": _chatwork_sign(body)},
        )
        assert resp.status_code == 200

        inbox = data_dir / "shared" / "inbox" / "kotoha"
        files = list(inbox.glob("*.json"))
        assert len(files) == 1

        msg = Message.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert msg.content == "E2E test from Chatwork"
        assert msg.source == "chatwork"
        assert msg.external_user_id == "54321"
        assert msg.external_channel_id == "99999"
        assert msg.source_message_id == "e2e_msg_001"
        assert msg.to_person == "kotoha"

    def test_invalid_signature_rejected(self, e2e_client):
        """Chatwork requests with wrong signature are rejected."""
        body = json.dumps({
            "webhook_event_type": "message_created",
            "webhook_event": {"body": "test"},
        }).encode("utf-8")
        resp = e2e_client.post(
            "/api/webhooks/chatwork",
            content=body,
            headers={"X-ChatWorkWebhookSignature": "wrong_signature"},
        )
        assert resp.status_code == 400


# ── Messenger receive_external E2E ────────────────────────


class TestMessengerReceiveExternalE2E:
    def test_receive_external_creates_file(self, data_dir):
        """Messenger.receive_external writes a valid Message JSON to inbox."""
        from core.messenger import Messenger

        shared_dir = data_dir / "shared"
        messenger = Messenger(shared_dir, "test-anima")

        msg = messenger.receive_external(
            content="External test",
            source="slack",
            source_message_id="ts_123",
            external_user_id="U_EXT",
            external_channel_id="C_EXT",
        )

        assert msg.source == "slack"
        assert msg.to_person == "test-anima"
        assert "slack:" in msg.from_person

        # Verify file on disk
        inbox = shared_dir / "inbox" / "test-anima"
        files = list(inbox.glob("*.json"))
        assert len(files) == 1

        restored = Message.model_validate_json(
            files[0].read_text(encoding="utf-8"),
        )
        assert restored.id == msg.id
        assert restored.content == "External test"
        assert restored.source == "slack"
        assert restored.thread_id == msg.id  # new thread

    def test_receive_external_appears_in_receive(self, data_dir):
        """External messages should be readable via receive()."""
        from core.messenger import Messenger

        shared_dir = data_dir / "shared"
        messenger = Messenger(shared_dir, "inbox-test")

        messenger.receive_external(
            content="Check inbox",
            source="chatwork",
            external_user_id="12345",
            external_channel_id="room_1",
        )

        messages = messenger.receive()
        assert len(messages) == 1
        assert messages[0].content == "Check inbox"
        assert messages[0].source == "chatwork"
