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
    async def test_start_connects_when_enabled(self, mock_config, mock_cred, mock_app_cls, mock_handler_cls):
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
    async def test_stop_closes_handler(self, mock_config, mock_cred, mock_app_cls, mock_handler_cls):
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

    @patch("core.notification.reply_routing.route_thread_reply", return_value=False)
    @patch("server.slack_socket._load_alias_user_ids", return_value=set())
    @patch("server.slack_socket._resolve_slack_mentions", side_effect=lambda text, token: text)
    @patch("server.slack_socket._route_to_board")
    @patch("server.slack_socket.load_config")
    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    async def test_default_anima_does_not_steal_other_bot_mentions(
        self,
        mock_messenger_cls,
        mock_get_data_dir,
        mock_config,
        mock_route_to_board,
        mock_resolve_mentions,
        mock_alias_ids,
        mock_route_thread_reply,
        tmp_path,
    ):
        """Default anima should not receive channel messages mentioning another bot."""
        from server.slack_socket import SlackSocketModeManager

        mock_get_data_dir.return_value = tmp_path
        slack_cfg = MagicMock(default_anima="sakura")
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )

        captured_handlers: dict[str, list] = {}
        mock_app = MagicMock()

        def _capture_event(event_type):
            def decorator(func):
                captured_handlers.setdefault(event_type, []).append(func)
                return func

            return decorator

        mock_app.event = _capture_event

        mgr = SlackSocketModeManager()
        mgr._bot_user_ids = {
            "sakura": "U_SAKURA",
            "ayane": "U_AYANE",
            "__shared__": "U_SHARED",
        }

        with patch.object(mgr, "_get_per_anima_credential", return_value="fake_token"):
            mgr._register_per_anima_handler(mock_app, "sakura", "U_SAKURA")

        handler_fn = captured_handlers["message"][-1]
        event = {
            "channel": "C_FINANCE",
            "user": "U_USER",
            "text": "<@U_AYANE> where is the file?",
            "ts": "123.456",
        }
        await handler_fn(event=event, say=AsyncMock())

        mock_messenger_cls.assert_not_called()
        mock_route_to_board.assert_not_called()

    @patch("core.notification.reply_routing.route_thread_reply", return_value=False)
    @patch("server.slack_socket._load_alias_user_ids", return_value=set())
    @patch("server.slack_socket._resolve_slack_mentions", side_effect=lambda text, token: text)
    @patch("server.slack_socket._route_to_board")
    @patch("server.slack_socket.get_credential", return_value="xoxb-fake")
    @patch("server.slack_socket.load_config")
    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    async def test_shared_handler_skips_per_anima_mentions(
        self,
        mock_messenger_cls,
        mock_get_data_dir,
        mock_config,
        mock_get_credential,
        mock_route_to_board,
        mock_resolve_mentions,
        mock_alias_ids,
        mock_route_thread_reply,
        tmp_path,
    ):
        """Shared default route should not consume messages for a per-Anima bot mention."""
        from server.slack_socket import SlackSocketModeManager

        mock_get_data_dir.return_value = tmp_path
        slack_cfg = MagicMock(anima_mapping={}, default_anima="sakura")
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(slack=slack_cfg),
        )

        captured_handlers: dict[str, list] = {}
        mock_app = MagicMock()

        def _capture_event(event_type):
            def decorator(func):
                captured_handlers.setdefault(event_type, []).append(func)
                return func

            return decorator

        mock_app.event = _capture_event

        mgr = SlackSocketModeManager()
        mgr._bot_user_ids = {
            "sakura": "U_SAKURA",
            "ayane": "U_AYANE",
            "__shared__": "U_SHARED",
        }

        mgr._register_shared_handler(mock_app, "U_SHARED")

        handler_fn = captured_handlers["message"][-1]
        event = {
            "channel": "C_FINANCE",
            "user": "U_USER",
            "text": "<@U_AYANE> please respond",
            "ts": "123.789",
        }
        await handler_fn(event=event, say=AsyncMock())

        mock_messenger_cls.assert_not_called()
        mock_route_to_board.assert_not_called()

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

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        mgr = SlackSocketModeManager()
        mgr._bot_user_ids = {"__shared__": "U_SHARED"}
        mgr._register_shared_handler(mock_async_app, "U_SHARED")

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
        assert call_kwargs["intent"] == "observe"

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
        mock_async_app.event = lambda et: lambda fn: fn
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

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        mgr = SlackSocketModeManager()
        mgr._bot_user_ids = {"__shared__": "U_SHARED"}
        mgr._register_shared_handler(mock_async_app, "U_SHARED")

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


# ── Slack annotation and addressee detection ──────────────


class TestBuildSlackAnnotation:
    """Tests for _build_slack_annotation with channel name and external addressees."""

    def test_dm_annotation(self):
        from server.slack_socket import _build_slack_annotation

        result = _build_slack_annotation("D_DIRECT", False)
        assert "[slack:DM]" in result

    def test_channel_with_mention_includes_channel_name(self):
        from server.slack_socket import _build_slack_annotation

        result = _build_slack_annotation("C_TEST", True, channel_name="general")
        assert "#general" in result
        assert "メンションされています" in result or "mentioned" in result

    def test_channel_no_mention_includes_channel_name(self):
        from server.slack_socket import _build_slack_annotation

        result = _build_slack_annotation("C_TEST", False, channel_name="aiシュライバーの作成")
        assert "#aiシュライバーの作成" in result
        assert "メンションはありません" in result or "no direct mention" in result

    def test_external_addressees_warning(self):
        from server.slack_socket import _build_slack_annotation

        result = _build_slack_annotation(
            "C_TEST",
            False,
            channel_name="dev",
            external_addressees=["ホアン ティン (OMINEXT)"],
        )
        assert "@ホアン ティン (OMINEXT)" in result
        assert "宛先注意" in result or "Addressee notice" in result

    def test_external_addressees_not_shown_when_mentioned(self):
        from server.slack_socket import _build_slack_annotation

        result = _build_slack_annotation(
            "C_TEST",
            True,
            channel_name="dev",
            external_addressees=["someone"],
        )
        assert "宛先注意" not in result
        assert "Addressee notice" not in result

    def test_no_channel_name_still_works(self):
        from server.slack_socket import _build_slack_annotation

        result = _build_slack_annotation("C_TEST", False)
        assert "[slack:channel" in result
        assert "#" not in result.split("—")[0]


class TestDetectExternalAddressees:
    """Tests for _detect_external_addressees."""

    def test_no_mentions_returns_empty(self):
        from server.slack_socket import _detect_external_addressees

        result = _detect_external_addressees("hello world", {"bot": "U_BOT"}, set())
        assert result == []

    def test_bot_mention_excluded(self):
        from server.slack_socket import _detect_external_addressees

        result = _detect_external_addressees(
            "<@U_BOT> hello",
            {"bot": "U_BOT"},
            set(),
        )
        assert result == []

    def test_alias_mention_excluded(self):
        from server.slack_socket import _detect_external_addressees

        result = _detect_external_addressees(
            "<@U_ALIAS> hello",
            {},
            {"U_ALIAS"},
        )
        assert result == []

    def test_external_mention_detected(self):
        from server.slack_socket import _cache_user_name, _detect_external_addressees

        _cache_user_name("UEXTERNAL1", "External Person")
        result = _detect_external_addressees(
            "<@UEXTERNAL1> please review",
            {"bot": "UBOT123"},
            {"UALIAS12"},
        )
        assert len(result) == 1
        assert "External Person" in result[0]

    def test_uncached_external_returns_uid_without_token(self):
        from server.slack_socket import _detect_external_addressees

        result = _detect_external_addressees(
            "<@UUNKNOWN1> hi",
            {},
            set(),
        )
        assert result == ["UUNKNOWN1"]


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
