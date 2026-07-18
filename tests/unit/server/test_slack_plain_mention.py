"""Unit tests for plain-text @anima-name mention routing in Slack channels."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def known_names():
    with patch(
        "server.slack_socket._known_anima_names",
        return_value=frozenset({"mei", "sakura", "aoi"}),
    ):
        yield


class TestDetectPlainAnimaMentions:
    def _detect(self, text: str) -> list[str]:
        from server.slack_socket import _detect_plain_anima_mentions

        return _detect_plain_anima_mentions(text)

    def test_basic_mention(self):
        assert self._detect("@mei これお願い") == ["mei"]

    def test_case_insensitive(self):
        assert self._detect("@Mei お願い") == ["mei"]

    def test_multiple_mentions_deduped(self):
        assert self._detect("@mei と @aoi で対応して @mei") == ["mei", "aoi"]

    def test_unknown_name_ignored(self):
        assert self._detect("@unknown_person hello") == []

    def test_email_not_matched(self):
        assert self._detect("連絡先は info@mei.example です") == []

    def test_japanese_text_directly_before_mention(self):
        assert self._detect("これ@mei お願い") == ["mei"]

    def test_real_slack_mention_not_matched(self):
        assert self._detect("<@U0A61UPTMUZ> hello") == []

    def test_empty_and_no_at(self):
        assert self._detect("") == []
        assert self._detect("meiさんお願い") == []


class TestPlainMentionRouting:
    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_plain_mention_routes_to_named_anima(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """@mei in a sakura-mapped channel delivers a question to mei, not sakura."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(
            enabled=True,
            mode="socket",
            anima_mapping={"C_TEST_CHAN": "sakura"},
            board_mapping={},
        )
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
        handler_fn = captured_handlers["message"][0]

        event = {
            "channel": "C_TEST_CHAN",
            "user": "U_USER_123",
            "text": "@mei 経費精算の集計をお願い",
            "ts": "7770000001.000001",
        }
        await handler_fn(event=event, say=AsyncMock())

        # Delivered to mei (not the channel default sakura)
        assert mock_messenger_cls.call_args[0][1] == "mei"
        call_kwargs = mock_messenger.receive_external.call_args[1]
        assert call_kwargs["intent"] == "question"
        assert "経費精算" in call_kwargs["content"]
        # Fanout-safe dedup id; thread anchor stays a valid Slack ts
        assert call_kwargs["source_message_id"] == "7770000001.000001#mei"
        assert call_kwargs["external_thread_ts"] == "7770000001.000001"

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_multi_mention_fanout(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """@mei @aoi delivers to both animas with distinct dedup ids."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(
            enabled=True,
            mode="socket",
            anima_mapping={"C_TEST_CHAN": "sakura"},
            board_mapping={},
        )
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
        mock_messenger_cls.return_value = MagicMock()

        mgr = SlackSocketModeManager()
        mgr._bot_user_ids = {"__shared__": "U_SHARED"}
        mgr._register_shared_handler(mock_async_app, "U_SHARED")
        handler_fn = captured_handlers["message"][0]

        await handler_fn(
            event={
                "channel": "C_TEST_CHAN",
                "user": "U_USER_123",
                "text": "@mei @aoi 分担して対応して",
                "ts": "7770000002.000001",
            },
            say=AsyncMock(),
        )

        targets = [c[0][1] for c in mock_messenger_cls.call_args_list]
        assert targets == ["mei", "aoi"]

    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_no_plain_mention_falls_through_to_default(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        tmp_path,
    ):
        """Without a plain mention the channel default still gets observe."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(
            enabled=True,
            mode="socket",
            anima_mapping={"C_TEST_CHAN": "sakura"},
            board_mapping={},
        )
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
        handler_fn = captured_handlers["message"][0]

        await handler_fn(
            event={
                "channel": "C_TEST_CHAN",
                "user": "U_USER_123",
                "text": "ただの雑談です",
                "ts": "7770000003.000001",
            },
            say=AsyncMock(),
        )

        assert mock_messenger_cls.call_args[0][1] == "sakura"
        call_kwargs = mock_messenger.receive_external.call_args[1]
        assert call_kwargs["intent"] == "observe"
        assert call_kwargs["source_message_id"] == "7770000003.000001"
