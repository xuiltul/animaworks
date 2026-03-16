"""Tests for Slack tool Anima identity (username / icon_url) in slack_send."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch


from core.config.models import (
    AnimaWorksConfig,
    HumanNotificationConfig,
    NotificationChannelConfig,
)
from core.tools.slack import _resolve_slack_identity


# ── _resolve_slack_identity ──────────────────────────────────────


class TestResolveSlackIdentity:
    """Tests for _resolve_slack_identity helper."""

    def test_returns_name_and_icon_with_template(self):
        cfg = AnimaWorksConfig(
            human_notification=HumanNotificationConfig(
                enabled=True,
                channels=[
                    NotificationChannelConfig(
                        type="slack",
                        enabled=True,
                        config={
                            "channel": "C123",
                            "icon_url_template": "https://cdn.example.com/{name}/icon.png",
                        },
                    ),
                ],
            ),
        )
        with patch("core.config.load_config", return_value=cfg):
            name, icon = _resolve_slack_identity(
                {"anima_dir": "/home/user/.animaworks/animas/sakura"}
            )
        assert name == "sakura"
        assert icon == "https://cdn.example.com/sakura/icon.png"

    def test_returns_name_only_without_template(self):
        cfg = AnimaWorksConfig(
            human_notification=HumanNotificationConfig(
                enabled=True,
                channels=[
                    NotificationChannelConfig(
                        type="slack",
                        enabled=True,
                        config={"channel": "C123"},
                    ),
                ],
            ),
        )
        with patch("core.config.load_config", return_value=cfg):
            name, icon = _resolve_slack_identity(
                {"anima_dir": "/home/user/.animaworks/animas/mei"}
            )
        assert name == "mei"
        assert icon == ""

    def test_returns_empty_when_no_anima_dir(self):
        name, icon = _resolve_slack_identity({})
        assert name == ""
        assert icon == ""

    def test_returns_name_when_no_slack_channel(self):
        cfg = AnimaWorksConfig(
            human_notification=HumanNotificationConfig(
                enabled=True,
                channels=[
                    NotificationChannelConfig(
                        type="line",
                        enabled=True,
                        config={"access_token": "tok"},
                    ),
                ],
            ),
        )
        with patch("core.config.load_config", return_value=cfg):
            name, icon = _resolve_slack_identity(
                {"anima_dir": "/home/user/.animaworks/animas/rin"}
            )
        assert name == "rin"
        assert icon == ""

    def test_returns_name_when_notification_disabled(self):
        cfg = AnimaWorksConfig(
            human_notification=HumanNotificationConfig(enabled=False, channels=[]),
        )
        with patch("core.config.load_config", return_value=cfg):
            name, icon = _resolve_slack_identity(
                {"anima_dir": "/home/user/.animaworks/animas/hina"}
            )
        assert name == "hina"
        assert icon == ""

    def test_returns_name_when_load_config_fails(self):
        with patch(
            "core.config.load_config",
            side_effect=RuntimeError("config broken"),
        ):
            name, icon = _resolve_slack_identity(
                {"anima_dir": "/home/user/.animaworks/animas/kotoha"}
            )
        assert name == "kotoha"
        assert icon == ""

    def test_skips_disabled_slack_channel(self):
        cfg = AnimaWorksConfig(
            human_notification=HumanNotificationConfig(
                enabled=True,
                channels=[
                    NotificationChannelConfig(
                        type="slack",
                        enabled=False,
                        config={
                            "channel": "C123",
                            "icon_url_template": "https://cdn.example.com/{name}.png",
                        },
                    ),
                    NotificationChannelConfig(
                        type="slack",
                        enabled=True,
                        config={
                            "channel": "C456",
                            "icon_url_template": "https://other.example.com/{name}.png",
                        },
                    ),
                ],
            ),
        )
        with patch("core.config.load_config", return_value=cfg):
            name, icon = _resolve_slack_identity(
                {"anima_dir": "/home/user/.animaworks/animas/kaede"}
            )
        assert name == "kaede"
        assert icon == "https://other.example.com/kaede.png"


# ── post_message with identity ───────────────────────────────────


class TestPostMessageIdentity:
    """Tests for SlackClient.post_message username/icon_url params."""

    def _make_client(self):
        from core.tools.slack import SlackClient

        client = SlackClient.__new__(SlackClient)
        client.client = MagicMock()
        client._user_name_cache = {}
        client._channel_cache = {}
        client.my_user_id = None
        return client

    def test_username_and_icon_in_kwargs(self):
        client = self._make_client()
        mock_method = MagicMock(return_value={"ok": True, "ts": "1.0"})
        client.client.chat_postMessage = mock_method

        client.post_message(
            "C123", "hello", username="sakura", icon_url="https://cdn/sakura.png"
        )

        mock_method.assert_called_once()
        kwargs = mock_method.call_args[1]
        assert kwargs["channel"] == "C123"
        assert kwargs["text"] == "hello"
        assert kwargs["username"] == "sakura"
        assert kwargs["icon_url"] == "https://cdn/sakura.png"

    def test_no_username_icon_when_empty(self):
        client = self._make_client()
        mock_method = MagicMock(return_value={"ok": True, "ts": "1.0"})
        client.client.chat_postMessage = mock_method

        client.post_message("C123", "hello")

        kwargs = mock_method.call_args[1]
        assert "username" not in kwargs
        assert "icon_url" not in kwargs

    def test_username_only_no_icon(self):
        client = self._make_client()
        mock_method = MagicMock(return_value={"ok": True, "ts": "1.0"})
        client.client.chat_postMessage = mock_method

        client.post_message("C123", "hello", username="mei")

        kwargs = mock_method.call_args[1]
        assert kwargs["username"] == "mei"
        assert "icon_url" not in kwargs

    def test_thread_ts_with_identity(self):
        client = self._make_client()
        mock_method = MagicMock(return_value={"ok": True, "ts": "1.0"})
        client.client.chat_postMessage = mock_method

        client.post_message(
            "C123",
            "reply",
            thread_ts="1.0",
            username="rin",
            icon_url="https://cdn/rin.png",
        )

        kwargs = mock_method.call_args[1]
        assert kwargs["thread_ts"] == "1.0"
        assert kwargs["username"] == "rin"
        assert kwargs["icon_url"] == "https://cdn/rin.png"


# ── dispatch slack_send with identity ────────────────────────────


class TestDispatchSlackSendIdentity:
    """Tests for dispatch('slack_send', ...) passing identity params."""

    def test_dispatch_passes_identity_to_post_message(self):
        cfg = AnimaWorksConfig(
            human_notification=HumanNotificationConfig(
                enabled=True,
                channels=[
                    NotificationChannelConfig(
                        type="slack",
                        enabled=True,
                        config={
                            "channel": "C999",
                            "icon_url_template": "https://cdn/{name}.png",
                        },
                    ),
                ],
            ),
        )

        mock_client_instance = MagicMock()
        mock_client_instance.resolve_channel.return_value = "C123"
        mock_client_instance.post_message.return_value = {"ok": True, "ts": "1.0"}

        with (
            patch("core.tools.slack.SlackClient", return_value=mock_client_instance),
            patch("core.tools.slack._resolve_slack_token", return_value="xoxb-test"),
            patch("core.config.load_config", return_value=cfg),
        ):
            from core.tools.slack import dispatch

            dispatch(
                "slack_send",
                {
                    "channel": "#general",
                    "message": "hello",
                    "anima_dir": "/home/user/.animaworks/animas/sakura",
                },
            )

        mock_client_instance.post_message.assert_called_once()
        call_kwargs = mock_client_instance.post_message.call_args
        assert call_kwargs[1]["username"] == "sakura"
        assert call_kwargs[1]["icon_url"] == "https://cdn/sakura.png"

    def test_dispatch_without_anima_dir_no_identity(self):
        mock_client_instance = MagicMock()
        mock_client_instance.resolve_channel.return_value = "C123"
        mock_client_instance.post_message.return_value = {"ok": True, "ts": "1.0"}

        with (
            patch("core.tools.slack.SlackClient", return_value=mock_client_instance),
            patch("core.tools.slack._resolve_slack_token", return_value="xoxb-test"),
        ):
            from core.tools.slack import dispatch

            dispatch(
                "slack_send",
                {
                    "channel": "#general",
                    "message": "hello",
                },
            )

        call_kwargs = mock_client_instance.post_message.call_args
        assert call_kwargs[1]["username"] == ""
        assert call_kwargs[1]["icon_url"] == ""
