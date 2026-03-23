"""Unit tests for Per-Anima Slack Bot feature.

Covers:
- core/tools/slack._resolve_slack_token
- core/outbound._send_via_slack, send_external anima_name passthrough
- core/config/models.ExternalMessagingChannelConfig.app_id_mapping
- server/routes/webhooks per-Anima signing secret and api_app_id routing
- server/slack_socket.SlackSocketModeManager per-Anima discovery and handlers
- core/notification/channels/slack per-Anima token resolution
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    ExternalMessagingChannelConfig,
    ExternalMessagingConfig,
)
from core.outbound import ResolvedRecipient, _send_via_slack, send_external


# ── 1. _resolve_slack_token tests ─────────────────────────────────────────


class TestResolveSlackToken:
    """Tests for core.tools.slack._resolve_slack_token."""

    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value="xoxb-vault-token")
    def test_returns_per_anima_token_from_vault(self, mock_vault, mock_shared):
        from core.tools.slack import _resolve_slack_token

        args = {"anima_dir": "/home/.animaworks/animas/sumire"}
        result = _resolve_slack_token(args)
        assert result == "xoxb-vault-token"
        mock_vault.assert_called_once_with("SLACK_BOT_TOKEN__sumire")
        mock_shared.assert_not_called()

    @patch("core.tools._base._lookup_shared_credentials", return_value="xoxb-shared-token")
    @patch("core.tools._base._lookup_vault_credential", return_value=None)
    def test_returns_per_anima_token_from_shared_credentials(self, mock_vault, mock_shared):
        from core.tools.slack import _resolve_slack_token

        args = {"anima_dir": "/home/.animaworks/animas/kotoha"}
        result = _resolve_slack_token(args)
        assert result == "xoxb-shared-token"
        mock_vault.assert_called_once_with("SLACK_BOT_TOKEN__kotoha")
        mock_shared.assert_called_once_with("SLACK_BOT_TOKEN__kotoha")

    def test_returns_none_when_no_per_anima_token_and_no_anima_dir(self):
        from core.tools.slack import _resolve_slack_token

        args = {}
        with patch("core.tools._base._lookup_vault_credential", return_value=None):
            with patch("core.tools._base._lookup_shared_credentials", return_value=None):
                result = _resolve_slack_token(args)
        assert result is None

    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value=None)
    def test_returns_none_when_anima_dir_present_but_no_per_anima_token(
        self, mock_vault, mock_shared
    ):
        from core.tools.slack import _resolve_slack_token

        args = {"anima_dir": "/home/.animaworks/animas/sakura"}
        result = _resolve_slack_token(args)
        assert result is None
        mock_vault.assert_called_once_with("SLACK_BOT_TOKEN__sakura")
        mock_shared.assert_called_once_with("SLACK_BOT_TOKEN__sakura")

    @patch("core.tools._base._lookup_shared_credentials", return_value="xoxb-shared")
    @patch("core.tools._base._lookup_vault_credential", return_value="xoxb-vault")
    def test_vault_takes_priority_over_shared_credentials(self, mock_vault, mock_shared):
        from core.tools.slack import _resolve_slack_token

        args = {"anima_dir": "/home/.animaworks/animas/sumire"}
        result = _resolve_slack_token(args)
        assert result == "xoxb-vault"
        mock_vault.assert_called_once()
        mock_shared.assert_not_called()


# ── 2. outbound._send_via_slack tests ─────────────────────────────────────


class TestSendViaSlackPerAnima:
    """Tests for core.outbound._send_via_slack per-Anima token behavior."""

    @patch("core.outbound._resolve_outbound_icon", return_value="https://example.com/sakura.png")
    @patch("core.tools.slack.SlackClient")
    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value="xoxb-per-anima")
    def test_uses_per_anima_token_when_anima_name_has_token(
        self, mock_vault, mock_shared, mock_client_cls, mock_icon
    ):
        mock_client = MagicMock()
        mock_client.post_message.return_value = {"ts": "123.456", "channel": "U1"}
        mock_client_cls.return_value = mock_client

        result = _send_via_slack("U1", "hello", "sakura", anima_name="sakura")

        mock_vault.assert_called_once_with("SLACK_BOT_TOKEN__sakura")
        mock_client_cls.assert_called_once_with(token="xoxb-per-anima")
        mock_client.post_message.assert_called_once_with(
            "U1", "hello", username="sakura", icon_url="https://example.com/sakura.png",
        )
        assert "sent" in result

    @patch("core.outbound._resolve_outbound_icon", return_value="")
    @patch("core.tools.slack.SlackClient")
    @patch("core.tools._base._lookup_shared_credentials", return_value="xoxb-per")
    @patch("core.tools._base._lookup_vault_credential", return_value=None)
    def test_omits_sender_prefix_when_per_anima_token_used(
        self, mock_vault, mock_shared, mock_client_cls, mock_icon
    ):
        mock_client = MagicMock()
        mock_client.post_message.return_value = {"ts": "1.1", "channel": "U1"}
        mock_client_cls.return_value = mock_client

        result = _send_via_slack("U1", "content", "sakura", anima_name="sakura")

        mock_client.post_message.assert_called_once_with(
            "U1", "content", username="sakura", icon_url="",
        )

    @patch("core.outbound._resolve_outbound_icon", return_value="")
    @patch("core.tools.slack.SlackClient")
    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value=None)
    def test_includes_sender_prefix_when_fallback_to_shared_token(
        self, mock_vault, mock_shared, mock_client_cls, mock_icon
    ):
        mock_client = MagicMock()
        mock_client.post_message.return_value = {"ts": "1.1", "channel": "U1"}
        mock_client_cls.return_value = mock_client

        result = _send_via_slack("U1", "content", "sakura", anima_name="")

        mock_client_cls.assert_called_once_with(token=None)
        mock_client.post_message.assert_called_once_with(
            "U1", "[sakura] content", username="sakura", icon_url="",
        )

    @patch("core.outbound._send_via_slack")
    def test_send_external_passes_anima_name_to_send_via_slack(self, mock_send):
        mock_send.return_value = json.dumps({"status": "sent", "channel": "slack"})
        r = ResolvedRecipient(
            is_internal=False, name="user", channel="slack", slack_user_id="U1",
        )
        send_external(r, "hello", sender_name="sumire", anima_name="sumire")
        mock_send.assert_called_once_with("U1", "hello", "sumire", "sumire")


# ── 3. ExternalMessagingChannelConfig.app_id_mapping tests ──────────────────


class TestExternalMessagingChannelConfigAppIdMapping:
    """Tests for app_id_mapping field on ExternalMessagingChannelConfig."""

    def test_app_id_mapping_defaults_to_empty_dict(self):
        cfg = ExternalMessagingChannelConfig()
        assert cfg.app_id_mapping == {}

    def test_app_id_mapping_round_trips_through_json_serialization(self):
        cfg = ExternalMessagingChannelConfig(
            enabled=True,
            app_id_mapping={"A01234ABCD": "sakura", "A05678EFGH": "kotoha"},
        )
        data = cfg.model_dump()
        restored = ExternalMessagingChannelConfig.model_validate(data)
        assert restored.app_id_mapping == {"A01234ABCD": "sakura", "A05678EFGH": "kotoha"}

    def test_backward_compatible_without_app_id_mapping(self):
        data = {"enabled": True, "anima_mapping": {"C1": "sakura"}}
        cfg = ExternalMessagingChannelConfig.model_validate(data)
        assert cfg.app_id_mapping == {}
        assert cfg.anima_mapping == {"C1": "sakura"}


# ── 4. Webhook per-Anima routing tests ────────────────────────────────────


def _make_slack_signature(body: bytes, timestamp: str, secret: str) -> str:
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    return "v0=" + hmac.new(
        secret.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class TestWebhookPerAnimaRouting:
    """Tests for server/routes/webhooks per-Anima api_app_id routing."""

    @pytest.fixture
    def app(self):
        from fastapi import FastAPI
        from server.routes.webhooks import create_webhooks_router
        app = FastAPI()
        app.include_router(create_webhooks_router(), prefix="/api")
        return app

    @pytest.fixture
    def client(self, app):
        from fastapi.testclient import TestClient
        return TestClient(app)

    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value="per_anima_secret")
    @patch("server.routes.webhooks.get_data_dir")
    @patch("server.routes.webhooks.load_config")
    def test_api_app_id_in_mapping_uses_per_anima_signing_secret(
        self, mock_config, mock_data_dir, mock_vault, mock_shared, client, tmp_path
    ):
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "sakura").mkdir(parents=True)
        mock_data_dir.return_value = tmp_path

        mock_config.return_value = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                slack=ExternalMessagingChannelConfig(
                    enabled=True,
                    app_id_mapping={"A01234PERANIMA": "sakura"},
                    anima_mapping={"C123": "sakura"},
                ),
            ),
        )

        payload = json.dumps({
            "type": "event_callback",
            "api_app_id": "A01234PERANIMA",
            "event": {
                "type": "message",
                "channel": "C123",
                "user": "U999",
                "text": "Hello Sakura",
                "ts": "1234567890.123456",
            },
        })
        body = payload.encode("utf-8")
        ts = str(int(time.time()))
        headers = {
            "X-Slack-Signature": _make_slack_signature(body, ts, "per_anima_secret"),
            "X-Slack-Request-Timestamp": ts,
        }

        resp = client.post("/api/webhooks/slack/events", content=body, headers=headers)
        assert resp.status_code == 200
        mock_vault.assert_any_call("SLACK_SIGNING_SECRET__sakura")

    @patch("server.routes.webhooks.get_data_dir")
    @patch("server.routes.webhooks.load_config")
    def test_api_app_id_not_in_mapping_uses_shared_signing_secret(
        self, mock_config, mock_data_dir, client, tmp_path
    ):
        shared_secret = "shared_signing_secret_xyz"
        mock_data_dir.return_value = tmp_path

        mock_config.return_value = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                slack=ExternalMessagingChannelConfig(
                    enabled=True,
                    app_id_mapping={},
                    anima_mapping={"C123": "sakura"},
                ),
            ),
        )

        with patch("server.routes.webhooks.get_credential", return_value=shared_secret):
            payload = json.dumps({
                "type": "event_callback",
                "api_app_id": "A_UNKNOWN_APP",
                "event": {
                    "type": "message",
                    "channel": "C123",
                    "user": "U999",
                    "text": "Hello",
                    "ts": "1.1",
                },
            })
            body = payload.encode("utf-8")
            ts = str(int(time.time()))
            headers = {
                "X-Slack-Signature": _make_slack_signature(body, ts, shared_secret),
                "X-Slack-Request-Timestamp": ts,
            }
            resp = client.post("/api/webhooks/slack/events", content=body, headers=headers)
        assert resp.status_code == 200

    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value="per_secret")
    @patch("server.routes.webhooks.get_data_dir")
    @patch("server.routes.webhooks.load_config")
    def test_api_app_id_routing_sets_correct_anima_name_for_delivery(
        self, mock_config, mock_data_dir, mock_vault, mock_shared, client, tmp_path
    ):
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "kotoha").mkdir(parents=True)
        mock_data_dir.return_value = tmp_path

        mock_config.return_value = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                slack=ExternalMessagingChannelConfig(
                    enabled=True,
                    app_id_mapping={"A_KOTOHA_APP": "kotoha"},
                    anima_mapping={"C_ANY": "fallback"},
                ),
            ),
        )

        payload = json.dumps({
            "type": "event_callback",
            "api_app_id": "A_KOTOHA_APP",
            "event": {
                "type": "message",
                "channel": "C_ANY",
                "user": "U_USER",
                "text": "Message for Kotoha",
                "ts": "9999999999.999999",
            },
        })
        body = payload.encode("utf-8")
        ts = str(int(time.time()))
        headers = {
            "X-Slack-Signature": _make_slack_signature(body, ts, "per_secret"),
            "X-Slack-Request-Timestamp": ts,
        }

        resp = client.post("/api/webhooks/slack/events", content=body, headers=headers)
        assert resp.status_code == 200

        inbox_files = list((shared_dir / "inbox" / "kotoha").glob("*.json"))
        assert len(inbox_files) == 1
        msg_data = json.loads(inbox_files[0].read_text(encoding="utf-8"))
        assert "Message for Kotoha" in msg_data["content"]
        assert msg_data["source"] == "slack"


# ── 5. SlackSocketModeManager tests ────────────────────────────────────────


class TestSlackSocketModeManagerPerAnima:
    """Tests for server/slack_socket.SlackSocketModeManager per-Anima support."""

    @patch("server.slack_socket.get_data_dir")
    def test_discover_per_anima_bots_finds_keys_in_vault(self, mock_get_data_dir, tmp_path):
        from server.slack_socket import SlackSocketModeManager

        mock_get_data_dir.return_value = tmp_path

        with patch("core.config.vault.get_vault_manager") as mock_vm:
            vm = MagicMock()
            vm.load_vault.return_value = {
                "shared": {
                    "SLACK_BOT_TOKEN__sakura": "xoxb-sakura",
                    "SLACK_BOT_TOKEN__kotoha": "xoxb-kotoha",
                },
            }
            mock_vm.return_value = vm

            found = SlackSocketModeManager._discover_per_anima_bots()

        assert set(found) == {"kotoha", "sakura"}

    def test_discover_per_anima_bots_finds_keys_in_shared_credentials(self, tmp_path):
        from server.slack_socket import SlackSocketModeManager

        cred_file = tmp_path / "shared" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(
            json.dumps({
                "SLACK_BOT_TOKEN__sumire": "xoxb-sumire",
                "OTHER_KEY": "ignored",
            }),
            encoding="utf-8",
        )

        with patch("server.slack_socket.get_data_dir", return_value=tmp_path):
            found = SlackSocketModeManager._discover_per_anima_bots()

        assert "sumire" in found

    @patch("server.slack_socket._lookup_shared_credentials", return_value=None)
    @patch("server.slack_socket._lookup_vault_credential", return_value="xoxb-from-vault")
    def test_get_per_anima_credential_resolves_from_vault_first(
        self, mock_vault, mock_shared
    ):
        from server.slack_socket import SlackSocketModeManager

        result = SlackSocketModeManager._get_per_anima_credential(
            "SLACK_BOT_TOKEN", "sakura",
        )
        assert result == "xoxb-from-vault"
        mock_vault.assert_called_once_with("SLACK_BOT_TOKEN__sakura")
        mock_shared.assert_not_called()

    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.SlackSocketModeManager._get_per_anima_credential")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_is_connected_returns_true_when_handlers_exist(
        self, mock_config, mock_cred, mock_get_per_anima, mock_app_cls, mock_handler_cls
    ):
        from server.slack_socket import SlackSocketModeManager

        mock_get_per_anima.return_value = "xoxb-bot"
        mock_cred.side_effect = lambda *a, **kw: "token"
        mock_handler_cls.return_value.connect_async = AsyncMock(return_value=None)
        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )

        with patch.object(
            SlackSocketModeManager, "_discover_per_anima_bots", return_value=["sakura"]
        ):
            mgr = SlackSocketModeManager()
            await mgr.start()

        assert mgr.is_connected is True

    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.SlackSocketModeManager._get_per_anima_credential")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_start_registers_per_anima_bots_plus_shared_bot(
        self, mock_config, mock_cred, mock_get_per_anima, mock_app_cls, mock_handler_cls
    ):
        from server.slack_socket import SlackSocketModeManager

        mock_get_per_anima.return_value = "xoxb-per"
        mock_cred.return_value = "xoxb-shared"
        mock_handler_cls.return_value.connect_async = AsyncMock(return_value=None)
        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={"C1": "sakura"})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )

        with patch.object(
            SlackSocketModeManager, "_discover_per_anima_bots", return_value=["sumire"]
        ):
            mgr = SlackSocketModeManager()
            await mgr.start()

        assert len(mgr._handlers) >= 1
        assert len(mgr._apps) >= 1

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.SlackSocketModeManager._get_per_anima_credential", return_value="xoxb-per")
    @patch("server.slack_socket.get_credential", side_effect=Exception("no shared"))
    @patch("server.slack_socket.load_config")
    async def test_per_anima_handler_routes_to_correct_anima(
        self,
        mock_config,
        mock_cred,
        mock_get_per_anima,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        from server.slack_socket import SlackSocketModeManager

        mock_get_data_dir.return_value = tmp_path
        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )

        captured_handlers: dict[str, list] = {}
        mock_async_app = MagicMock()

        def _capture_event(event_type):
            def decorator(func):
                captured_handlers.setdefault(event_type, []).append(func)
                return func
            return decorator

        mock_async_app.event = _capture_event
        mock_app_cls.return_value = mock_async_app
        mock_handler_cls.return_value = AsyncMock()

        with patch.object(
            SlackSocketModeManager, "_discover_per_anima_bots", return_value=["sumire"]
        ):
            mgr = SlackSocketModeManager()
            await mgr.start()

        assert "message" in captured_handlers
        handler_fn = captured_handlers["message"][0]

        event = {
            "channel": "C_SUMIRE_CHAN",
            "user": "U_USER",
            "text": "Hello Sumire",
            "ts": "9999.1",
        }
        await handler_fn(event=event, say=AsyncMock())

        mock_messenger_cls.assert_called_with(tmp_path / "shared", "sumire")
        call_kwargs = mock_messenger_cls.return_value.receive_external.call_args[1]
        assert "Hello Sumire" in call_kwargs["content"]
        assert call_kwargs["source"] == "slack"
        assert call_kwargs["source_message_id"] == "9999.1"
        assert call_kwargs["external_user_id"] == "U_USER"
        assert call_kwargs["external_channel_id"] == "C_SUMIRE_CHAN"


# ── 6. Notification channel per-Anima token tests ─────────────────────────


class TestSlackNotificationChannelPerAnima:
    """Tests for core/notification/channels/slack per-Anima token resolution."""

    @pytest.fixture
    def slack_channel(self):
        from core.notification.channels.slack import SlackChannel
        return SlackChannel(config={"channel": "C123", "bot_token": ""})

    @patch("core.notification.channels.slack.SlackChannel._send_via_bot")
    @patch("core.tools._base.get_credential", side_effect=Exception("no shared"))
    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value="xoxb-per")
    async def test_send_uses_per_anima_token_when_anima_name_set_and_token_exists(
        self, mock_vault, mock_shared, mock_cred, mock_send_bot, slack_channel
    ):
        mock_send_bot.return_value = "slack: OK"

        result = await slack_channel.send(
            "Subject", "Body", "normal", anima_name="sakura",
        )

        assert result == "slack: OK"
        mock_vault.assert_called_once_with("SLACK_BOT_TOKEN__sakura")
        mock_send_bot.assert_awaited_once()
        call_kwargs = mock_send_bot.call_args
        assert call_kwargs[0][0] == "xoxb-per"

    @patch("core.notification.channels.slack.SlackChannel._send_via_bot")
    @patch("core.tools._base.get_credential", return_value="xoxb-shared")
    @patch("core.tools._base._lookup_shared_credentials", return_value=None)
    @patch("core.tools._base._lookup_vault_credential", return_value=None)
    async def test_send_falls_back_to_shared_token_when_no_per_anima_token(
        self, mock_vault, mock_shared, mock_cred, mock_send_bot, slack_channel
    ):
        mock_send_bot.return_value = "slack: OK"

        result = await slack_channel.send(
            "Subject", "Body", "normal", anima_name="sakura",
        )

        assert result == "slack: OK"
        mock_vault.assert_called_once_with("SLACK_BOT_TOKEN__sakura")
        mock_shared.assert_called_once_with("SLACK_BOT_TOKEN__sakura")
        mock_cred.assert_called_once()
        mock_send_bot.assert_awaited_once()
        call_args = mock_send_bot.call_args[0]
        assert call_args[0] == "xoxb-shared"
