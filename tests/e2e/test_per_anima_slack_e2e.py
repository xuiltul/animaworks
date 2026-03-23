"""E2E tests for Per-Anima Slack Bot feature.

Tests the full flow of per-Anima token resolution across:
- Slack tool dispatch (slack_send)
- Outbound send_external (sender prefix logic)
- Webhook routing (api_app_id → anima_name)
- Config backward compatibility (app_id_mapping)
"""
from __future__ import annotations

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
    save_config,
)
from core.outbound import ResolvedRecipient, send_external
from core.tools.slack import dispatch
from server.routes.webhooks import create_webhooks_router

SIGNING_SECRET = "e2e_per_anima_slack_secret"


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def webhook_app(data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Create FastAPI app with webhooks router for E2E tests."""
    monkeypatch.setattr("server.routes.webhooks.get_data_dir", lambda: data_dir)

    def _mock_get_credential(credential_name: str, tool_name: str, **kwargs):
        creds = {"slack_signing": SIGNING_SECRET}
        if credential_name in creds:
            return creds[credential_name]
        raise Exception(f"Unknown credential: {credential_name}")

    monkeypatch.setattr("server.routes.webhooks.get_credential", _mock_get_credential)

    app = FastAPI()
    router = create_webhooks_router()
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
def webhook_client(webhook_app):
    return TestClient(webhook_app)


# ── 1. Full Slack tool dispatch with per-Anima token ────────


class TestSlackToolDispatchPerAnimaToken:
    def test_dispatch_uses_per_anima_token_when_configured(self):
        """Set up per-Anima token in vault mock; verify SlackClient gets it."""
        with patch("core.tools.slack.SlackClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.resolve_channel.return_value = "C123"
            mock_client.post_message.return_value = {"ts": "1234", "channel": "C123"}

            with patch(
                "core.tools.slack._resolve_slack_token",
            ) as mock_resolve:
                mock_resolve.return_value = "xoxb-per-anima-token"

                result = dispatch(
                    "slack_send",
                    {
                        "channel": "C123",
                        "message": "test",
                        "anima_dir": "/fake/animas/sumire",
                    },
                )

        assert result == {"ts": "1234", "channel": "C123"}
        mock_cls.assert_called_once_with(token="xoxb-per-anima-token")
        mock_client.resolve_channel.assert_called_once_with("C123")
        mock_client.post_message.assert_called_once()
        call_args = mock_client.post_message.call_args
        assert call_args[0][0] == "C123"
        assert "test" in call_args[0][1]


# ── 2. Full Slack tool dispatch with shared token fallback ──


class TestSlackToolDispatchSharedTokenFallback:
    def test_dispatch_uses_none_token_when_no_per_anima(self):
        """No per-Anima token; SlackClient created with None (shared fallback)."""
        with patch("core.tools.slack.SlackClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.resolve_channel.return_value = "C123"
            mock_client.post_message.return_value = {"ts": "5678", "channel": "C123"}

            with patch(
                "core.tools.slack._resolve_slack_token",
            ) as mock_resolve:
                mock_resolve.return_value = None

                result = dispatch(
                    "slack_send",
                    {
                        "channel": "C123",
                        "message": "test",
                        "anima_dir": "/fake/animas/sumire",
                    },
                )

        assert result == {"ts": "5678", "channel": "C123"}
        mock_cls.assert_called_once_with(token=None)


# ── 3. Outbound flow: per-Anima token skips sender prefix ───


class TestOutboundPerAnimaSkipsPrefix:
    def test_send_external_with_per_anima_token_no_sender_prefix(self):
        """Per-Anima token: message text does NOT have [sender_name] prefix."""
        resolved = ResolvedRecipient(
            is_internal=False,
            name="user",
            channel="slack",
            slack_user_id="U0TEST000001",
        )

        with patch("core.tools.slack.SlackClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.post_message.return_value = {"ts": "1", "channel": "D1"}

            with patch(
                "core.tools._base._lookup_vault_credential",
                return_value="xoxb-per-anima-token",
            ), patch(
                "core.tools._base._lookup_shared_credentials",
                return_value=None,
            ):
                result = send_external(
                    resolved,
                    "hello from sumire",
                    sender_name="sakura",
                    anima_name="sumire",
                )

        data = json.loads(result)
        assert data["status"] == "sent"
        mock_client.post_message.assert_called_once()
        text_sent = mock_client.post_message.call_args[0][1]
        assert text_sent == "hello from sumire"
        assert "[sakura]" not in text_sent
        mock_cls.assert_called_once_with(token="xoxb-per-anima-token")


# ── 4. Outbound flow: shared token includes sender prefix ───


class TestOutboundSharedTokenIncludesPrefix:
    def test_send_external_with_shared_token_has_sender_prefix(self):
        """No per-Anima token; message text HAS [sender_name] prefix."""
        resolved = ResolvedRecipient(
            is_internal=False,
            name="user",
            channel="slack",
            slack_user_id="U0TEST000001",
        )

        with patch("core.tools.slack.SlackClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.post_message.return_value = {"ts": "1", "channel": "D1"}

            with patch(
                "core.tools._base._lookup_vault_credential",
                return_value=None,
            ), patch(
                "core.tools._base._lookup_shared_credentials",
                return_value=None,
            ):
                result = send_external(
                    resolved,
                    "hello",
                    sender_name="sakura",
                    anima_name="",
                )

        data = json.loads(result)
        assert data["status"] == "sent"
        mock_client.post_message.assert_called_once()
        text_sent = mock_client.post_message.call_args[0][1]
        assert text_sent == "[sakura] hello"
        mock_cls.assert_called_once_with(token=None)


# ── 5. Webhook routing: per-Anima app_id routing ────────────


class TestWebhookPerAnimaAppIdRouting:
    def test_webhook_routes_to_anima_by_api_app_id(
        self,
        webhook_client: TestClient,
        data_dir: Path,
        make_anima,
    ):
        """Configure app_id_mapping; webhook with api_app_id delivers to correct Anima."""
        make_anima("sumire")
        config = AnimaWorksConfig.model_validate(
            json.loads((data_dir / "config.json").read_text(encoding="utf-8")),
        )
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                enabled=True,
                app_id_mapping={"A0PERANIMA123": "sumire"},
                anima_mapping={"C_E2E": "sumire"},
            ),
        )
        save_config(config, data_dir / "config.json")

        # Per-Anima signing secret for sumire (used for signature verification)
        per_secret = "e2e_sumire_signing_secret"

        def _mock_vault(key: str):
            if key == "SLACK_SIGNING_SECRET__sumire":
                return per_secret
            return None

        payload = json.dumps({
            "type": "event_callback",
            "api_app_id": "A0PERANIMA123",
            "event": {
                "type": "message",
                "channel": "C_E2E",
                "user": "U_E2E_USER",
                "text": "Webhook test for sumire",
                "ts": "9999999999.000001",
            },
        })
        body = payload.encode("utf-8")
        ts = str(int(time.time()))
        sig_base = f"v0:{ts}:{body.decode('utf-8')}"
        sig = "v0=" + hmac.new(
            per_secret.encode(), sig_base.encode(), hashlib.sha256,
        ).hexdigest()

        with patch(
            "core.tools._base._lookup_vault_credential",
            side_effect=_mock_vault,
        ), patch(
            "core.tools._base._lookup_shared_credentials",
            return_value=None,
        ):
            resp = webhook_client.post(
                "/api/webhooks/slack/events",
                content=body,
                headers={
                    "X-Slack-Signature": sig,
                    "X-Slack-Request-Timestamp": ts,
                },
            )

        assert resp.status_code == 200

        # Verify message delivered to sumire inbox
        inbox = data_dir / "shared" / "inbox" / "sumire"
        inbox.mkdir(parents=True, exist_ok=True)
        files = list(inbox.glob("*.json"))
        assert len(files) >= 1
        msg_data = json.loads(files[0].read_text(encoding="utf-8"))
        content = msg_data.get("content", "")
        assert "Webhook test for sumire" in content
        assert msg_data.get("to_person") == "sumire"


# ── 6. Config backward compatibility ───────────────────────


class TestConfigBackwardCompatibility:
    def test_external_messaging_config_without_app_id_mapping(self):
        """Load ExternalMessagingConfig from dict without app_id_mapping; defaults work."""
        data = {
            "preferred_channel": "slack",
            "slack": {"enabled": True, "anima_mapping": {"C1": "sakura"}},
        }
        cfg = ExternalMessagingConfig(**data)
        assert cfg.slack.enabled is True
        assert cfg.slack.anima_mapping == {"C1": "sakura"}
        assert cfg.slack.app_id_mapping == {}

    def test_external_messaging_config_with_app_id_mapping(self):
        """Load ExternalMessagingConfig with app_id_mapping; works correctly."""
        data = {
            "preferred_channel": "slack",
            "slack": {
                "enabled": True,
                "anima_mapping": {"C1": "sakura"},
                "app_id_mapping": {"A01234": "sumire"},
            },
        }
        cfg = ExternalMessagingConfig(**data)
        assert cfg.slack.app_id_mapping == {"A01234": "sumire"}
        assert cfg.slack.anima_mapping == {"C1": "sakura"}
