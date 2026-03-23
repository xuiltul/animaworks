"""Unit tests for server/slack_socket.py — SlackSocketModeManager."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── SlackSocketModeManager ───────────────────────────────


class TestSlackSocketModeManagerStart:
    """Tests for SlackSocketModeManager.start()."""

    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_start_disabled_when_not_enabled(self, mock_config, mock_cred):
        """Does nothing when slack.enabled is False."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=False, mode="socket")
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )

        mgr = SlackSocketModeManager()
        await mgr.start()

        mock_cred.assert_not_called()
        assert not mgr.is_connected

    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_start_disabled_when_mode_is_webhook(self, mock_config, mock_cred):
        """Does nothing when slack.mode is 'webhook'."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="webhook")
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )

        mgr = SlackSocketModeManager()
        await mgr.start()

        mock_cred.assert_not_called()
        assert not mgr.is_connected

    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_start_connects_when_enabled(
        self, mock_config, mock_cred, mock_app_cls, mock_handler_cls
    ):
        """Connects when slack is enabled with socket mode."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={"C1": "sakura"})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.side_effect = lambda name, tool, **kw: f"token_{name}"
        mock_app_cls.return_value = MagicMock()
        mock_handler_cls.return_value = AsyncMock()

        mgr = SlackSocketModeManager()
        with patch.object(type(mgr), "_discover_per_anima_bots", return_value=[]):
            await mgr.start()

        # Verify credentials were fetched for shared bot
        mock_cred.assert_any_call("slack", "slack_socket", env_var="SLACK_BOT_TOKEN")
        mock_cred.assert_any_call("slack_app", "slack_socket", env_var="SLACK_APP_TOKEN")

        assert mgr.is_connected


class TestSlackSocketModeManagerStop:
    """Tests for SlackSocketModeManager.stop()."""

    async def test_stop_when_not_started(self):
        """stop() is a no-op when handler is None."""
        from server.slack_socket import SlackSocketModeManager

        mgr = SlackSocketModeManager()
        await mgr.stop()  # Should not raise

    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_stop_closes_handler(
        self, mock_config, mock_cred, mock_app_cls, mock_handler_cls
    ):
        """stop() calls close_async on the handler."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.return_value = "fake_token"
        mock_app_cls.return_value = MagicMock()
        mock_handler_cls.return_value = AsyncMock()

        mgr = SlackSocketModeManager()
        with patch.object(type(mgr), "_discover_per_anima_bots", return_value=[]):
            await mgr.start()
        await mgr.stop()

        mock_handler_cls.return_value.close_async.assert_awaited_once()


class TestSlackSocketModeManagerHandlers:
    """Tests for message handler registration and routing."""

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_message_handler_routes_to_anima(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """Message events with mapped channels are routed to the correct anima inbox."""
        from server.slack_socket import SlackSocketModeManager

        anima_mapping = {"C_TEST_CHAN": "sakura"}
        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping=anima_mapping)
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path

        # Capture the event handler registration
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

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        mgr = SlackSocketModeManager()
        await mgr.start()

        # Verify "message" event handler was registered
        assert "message" in captured_handlers
        handler_fn = captured_handlers["message"][0]

        # Simulate a message event
        event = {
            "channel": "C_TEST_CHAN",
            "user": "U_USER_123",
            "text": "Hello from Slack",
            "ts": "1234567890.123456",
        }
        await handler_fn(event=event, say=AsyncMock())

        # Verify Messenger was called and content includes message text
        mock_messenger.receive_external.assert_called_once()
        call_kwargs = mock_messenger.receive_external.call_args[1]
        assert "Hello from Slack" in call_kwargs["content"]
        assert call_kwargs["source"] == "slack"
        assert call_kwargs["source_message_id"] == "1234567890.123456"
        assert call_kwargs["external_user_id"] == "U_USER_123"
        assert call_kwargs["external_channel_id"] == "C_TEST_CHAN"

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_message_handler_ignores_unmapped_channel(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """Messages from unmapped channels are ignored when no default_anima."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={"C_KNOWN": "sakura"}, default_anima="")
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path

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

        mgr = SlackSocketModeManager()
        await mgr.start()

        handler_fn = captured_handlers["message"][0]

        # Send message from unmapped channel with no default_anima
        event = {
            "channel": "C_UNKNOWN",
            "user": "U_USER",
            "text": "This should be ignored",
            "ts": "1.1",
        }
        await handler_fn(event=event, say=AsyncMock())

        # Messenger should not be instantiated
        mock_messenger_cls.assert_not_called()

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_message_handler_ignores_subtype_events(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """Message events with subtypes (edits, bot messages) are ignored."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={"C_CHAN": "sakura"})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path

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

        mgr = SlackSocketModeManager()
        await mgr.start()

        handler_fn = captured_handlers["message"][0]

        # Send message with subtype (e.g. message_changed)
        event = {
            "channel": "C_CHAN",
            "user": "U_USER",
            "text": "edited message",
            "ts": "1.1",
            "subtype": "message_changed",
        }
        await handler_fn(event=event, say=AsyncMock())

        mock_messenger_cls.assert_not_called()


class TestUnhandledEventSuppression:
    """Tests for Bolt 404 suppression via error handler."""

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_error_handler_registered_on_shared_app(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """Error handler is registered on the shared AsyncApp."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path

        mock_async_app = MagicMock()
        captured_error_handler = []
        mock_async_app.error = lambda fn: captured_error_handler.append(fn) or fn
        mock_async_app.event = lambda et: (lambda fn: fn)
        mock_app_cls.return_value = mock_async_app
        mock_handler_cls.return_value = AsyncMock()

        mgr = SlackSocketModeManager()
        with patch.object(type(mgr), "_discover_per_anima_bots", return_value=[]):
            await mgr.start()

        assert len(captured_error_handler) == 1

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_raise_error_for_unhandled_request_flag(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """AsyncApp is created with raise_error_for_unhandled_request=True."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={})
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path
        mock_app_cls.return_value = MagicMock()
        mock_handler_cls.return_value = AsyncMock()

        mgr = SlackSocketModeManager()
        with patch.object(type(mgr), "_discover_per_anima_bots", return_value=[]):
            await mgr.start()

        mock_app_cls.assert_called_with(
            token="fake_token",
            raise_error_for_unhandled_request=True,
        )

    async def test_error_handler_returns_200_for_unhandled(self):
        """BoltUnhandledRequestError is caught and returns 200."""
        from slack_bolt.error import BoltUnhandledRequestError
        from slack_bolt.response import BoltResponse

        from server.slack_socket import SlackSocketModeManager

        app = MagicMock()
        captured_handler = []
        app.error = lambda fn: captured_handler.append(fn) or fn

        SlackSocketModeManager._register_error_handler(app)
        assert len(captured_handler) == 1
        handler = captured_handler[0]

        error = BoltUnhandledRequestError(request=MagicMock(), current_response=None)

        result = await handler(
            error=error,
            body={"event": {"type": "reaction_added"}},
        )

        assert isinstance(result, BoltResponse)
        assert result.status == 200

    async def test_error_handler_re_raises_other_errors(self):
        """Non-BoltUnhandledRequestError errors are re-raised."""
        from server.slack_socket import SlackSocketModeManager

        app = MagicMock()
        captured_handler = []
        app.error = lambda fn: captured_handler.append(fn) or fn

        SlackSocketModeManager._register_error_handler(app)
        handler = captured_handler[0]

        with pytest.raises(ValueError, match="something broke"):
            await handler(error=ValueError("something broke"), body={})

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_message_handler_still_works_with_error_handler(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """Normal message events are still routed correctly."""
        from server.slack_socket import SlackSocketModeManager

        anima_mapping = {"C_TEST": "sakura"}
        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping=anima_mapping)
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path

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

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        mgr = SlackSocketModeManager()
        await mgr.start()

        assert "message" in captured_handlers
        handler_fn = captured_handlers["message"][0]

        event = {
            "channel": "C_TEST",
            "user": "U_USER",
            "text": "Hello",
            "ts": "9999.9999",
        }
        await handler_fn(event=event, say=AsyncMock())

        mock_messenger.receive_external.assert_called_once()


class TestExternalMessagingChannelConfigMode:
    """Tests for the mode field on ExternalMessagingChannelConfig."""

    def test_default_mode_is_socket(self):
        from core.config.models import ExternalMessagingChannelConfig

        cfg = ExternalMessagingChannelConfig()
        assert cfg.mode == "socket"

    def test_mode_webhook(self):
        from core.config.models import ExternalMessagingChannelConfig

        cfg = ExternalMessagingChannelConfig(mode="webhook")
        assert cfg.mode == "webhook"

    def test_mode_roundtrip_json(self):
        from core.config.models import ExternalMessagingChannelConfig

        cfg = ExternalMessagingChannelConfig(enabled=True, mode="socket", anima_mapping={"C1": "sakura"})
        data = cfg.model_dump()
        restored = ExternalMessagingChannelConfig.model_validate(data)
        assert restored.mode == "socket"
        assert restored.anima_mapping == {"C1": "sakura"}
