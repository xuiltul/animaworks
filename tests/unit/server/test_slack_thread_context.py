"""Unit tests for Slack thread context injection (slack_socket + webhooks + inbox)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch



# ── _fetch_thread_context ────────────────────────────────


class TestFetchThreadContext:
    """Tests for the _fetch_thread_context helper in slack_socket."""

    def test_returns_empty_when_no_thread_ts(self):
        from server.slack_socket import _fetch_thread_context

        assert _fetch_thread_context("xoxb-token", "C123", "") == ""

    def test_returns_empty_when_no_token(self):
        from server.slack_socket import _fetch_thread_context

        assert _fetch_thread_context("", "C123", "1234.5678") == ""

    @patch("core.tools.slack.SlackClient", autospec=True)
    def test_returns_empty_when_single_reply(self, mock_cls):
        """Single message (root only, no replies) should return empty."""
        from server.slack_socket import _fetch_thread_context

        mock_client = mock_cls.return_value
        mock_client.thread_replies.return_value = [
            {"user": "U1", "text": "root message", "ts": "1000.0"},
        ]
        result = _fetch_thread_context("xoxb-token", "C123", "1000.0")
        assert result == ""

    @patch("core.tools.slack.SlackClient", autospec=True)
    def test_formats_thread_context(self, mock_cls):
        """Normal thread with root + 2 replies: parent summary + reply count."""
        from server.slack_socket import _fetch_thread_context

        mock_client = mock_cls.return_value
        mock_client.thread_replies.return_value = [
            {"user": "U_BOT", "text": "Zombie process detected", "ts": "1000.0"},
            {"user": "U_HUMAN", "text": "What should I run?", "ts": "1001.0"},
            {"user": "U_HUMAN", "text": "Tell me the commands", "ts": "1002.0"},
        ]
        result = _fetch_thread_context("xoxb-token", "C123", "1000.0")
        assert "[Thread context" in result
        assert "<@U_BOT>: Zombie process detected" in result
        assert "(2 replies in thread)" in result
        assert "[/Thread context]" in result

    @patch("core.tools.slack.SlackClient", autospec=True)
    def test_summary_includes_parent_and_reply_count(self, mock_cls):
        """Concise summary: parent message (truncated) + reply count only."""
        from server.slack_socket import _fetch_thread_context

        replies = [{"user": f"U{i}", "text": f"msg {i}", "ts": f"{i}.0"} for i in range(15)]
        mock_client = mock_cls.return_value
        mock_client.thread_replies.return_value = replies

        result = _fetch_thread_context("xoxb-token", "C123", "0.0", limit=5)
        assert "<@U0>: msg 0" in result
        assert "(14 replies in thread)" in result

    @patch("core.tools.slack.SlackClient", autospec=True)
    def test_api_failure_returns_empty(self, mock_cls):
        """API errors are caught and return empty string."""
        from server.slack_socket import _fetch_thread_context

        mock_client = mock_cls.return_value
        mock_client.thread_replies.side_effect = RuntimeError("API error")
        result = _fetch_thread_context("xoxb-token", "C123", "1000.0")
        assert result == ""


# ── Per-Anima handler thread injection ────────────────────


class TestPerAnimaHandlerThreadInjection:
    """Per-Anima Socket Mode message handler injects thread context."""

    @patch("server.slack_socket._fetch_thread_context", return_value="")
    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_top_level_message_no_thread_context(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        mock_fetch_ctx,
        tmp_path,
    ):
        """Top-level messages (no thread_ts) skip thread context fetch."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={"C1": "anima1"})
        mock_config.return_value = MagicMock(external_messaging=MagicMock(slack=slack_cfg))
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path

        captured: dict[str, list] = {}
        mock_app = MagicMock()
        mock_app.event = lambda t: (lambda f: captured.setdefault(t, []).append(f) or f)
        mock_app_cls.return_value = mock_app
        mock_handler_cls.return_value = AsyncMock()
        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        mgr = SlackSocketModeManager()
        await mgr.start()

        handler = captured["message"][0]
        event = {"channel": "C1", "user": "U1", "text": "hello", "ts": "1.0"}
        await handler(event=event, say=AsyncMock())

        mock_fetch_ctx.assert_not_called()
        mock_messenger.receive_external.assert_called_once()
        call_kw = mock_messenger.receive_external.call_args
        assert call_kw.kwargs.get("external_thread_ts", call_kw[1].get("external_thread_ts", "")) == ""

    @patch("server.slack_socket._fetch_thread_context")
    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_thread_reply_injects_context(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        mock_fetch_ctx,
        tmp_path,
    ):
        """Thread replies prepend thread context to content."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(enabled=True, mode="socket", anima_mapping={"C1": "anima1"})
        mock_config.return_value = MagicMock(external_messaging=MagicMock(slack=slack_cfg))
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path
        mock_fetch_ctx.return_value = "[Thread context]\n  <@U_BOT>: root\n[/Thread context]\n"

        captured: dict[str, list] = {}
        mock_app = MagicMock()
        mock_app.event = lambda t: (lambda f: captured.setdefault(t, []).append(f) or f)
        mock_app_cls.return_value = mock_app
        mock_handler_cls.return_value = AsyncMock()
        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        mgr = SlackSocketModeManager()
        await mgr.start()

        handler = captured["message"][0]
        event = {
            "channel": "C1",
            "user": "U1",
            "text": "tell me the command",
            "ts": "2.0",
            "thread_ts": "1.0",
        }
        await handler(event=event, say=AsyncMock())

        mock_fetch_ctx.assert_called_once()
        call_kw = mock_messenger.receive_external.call_args
        content = call_kw.kwargs.get("content") or call_kw[1].get("content")
        assert content.startswith("[Thread context]")
        assert "tell me the command" in content
        thread_ts = call_kw.kwargs.get("external_thread_ts") or call_kw[1].get("external_thread_ts")
        assert thread_ts == "1.0"


# ── Shared handler thread injection ──────────────────────


class TestSharedHandlerThreadInjection:
    """Shared Socket Mode handlers inject thread context."""

    @patch("server.slack_socket._fetch_thread_context")
    @patch("server.slack_socket.get_data_dir")
    @patch("server.slack_socket.Messenger")
    @patch("server.slack_socket.AsyncSocketModeHandler")
    @patch("server.slack_socket.AsyncApp")
    @patch("server.slack_socket.get_credential")
    @patch("server.slack_socket.load_config")
    async def test_shared_message_thread_injects_context(
        self,
        mock_config,
        mock_cred,
        mock_app_cls,
        mock_handler_cls,
        mock_messenger_cls,
        mock_get_data_dir,
        mock_fetch_ctx,
        tmp_path,
    ):
        """Shared message handler injects thread context for thread replies."""
        from server.slack_socket import SlackSocketModeManager

        slack_cfg = MagicMock(
            enabled=True, mode="socket", anima_mapping={"C1": "sakura"}, default_anima="sakura"
        )
        mock_config.return_value = MagicMock(external_messaging=MagicMock(slack=slack_cfg))
        mock_cred.return_value = "fake_token"
        mock_get_data_dir.return_value = tmp_path
        mock_fetch_ctx.return_value = "[Thread context]\n  <@U0>: root\n[/Thread context]\n"

        captured: dict[str, list] = {}
        mock_app = MagicMock()
        mock_app.event = lambda t: (lambda f: captured.setdefault(t, []).append(f) or f)
        mock_app_cls.return_value = mock_app
        mock_handler_cls.return_value = AsyncMock()
        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        mgr = SlackSocketModeManager()
        await mgr.start()

        handler = captured["message"][0]
        event = {
            "channel": "C1",
            "user": "U1",
            "text": "thread reply",
            "ts": "3.0",
            "thread_ts": "1.0",
        }
        await handler(event=event, say=AsyncMock())

        mock_fetch_ctx.assert_called_once()
        call_kw = mock_messenger.receive_external.call_args
        content = call_kw.kwargs.get("content") or call_kw[1].get("content")
        assert "[Thread context]" in content
        assert "thread reply" in content
        thread_ts = call_kw.kwargs.get("external_thread_ts") or call_kw[1].get("external_thread_ts")
        assert thread_ts == "1.0"


# ── receive_external passthrough ─────────────────────────


class TestReceiveExternalThreadTs:
    """Verify external_thread_ts passes through to Message."""

    def test_external_thread_ts_stored_in_message(self, tmp_path):
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        shared.mkdir()
        m = Messenger(shared, "test_anima")
        msg = m.receive_external(
            content="reply text",
            source="slack",
            source_message_id="2.0",
            external_user_id="U1",
            external_channel_id="C1",
            external_thread_ts="1.0",
            intent="question",
        )
        assert msg.external_thread_ts == "1.0"

    def test_external_thread_ts_defaults_empty(self, tmp_path):
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        shared.mkdir()
        m = Messenger(shared, "test_anima")
        msg = m.receive_external(
            content="top level",
            source="slack",
            source_message_id="1.0",
            external_user_id="U1",
            external_channel_id="C1",
        )
        assert msg.external_thread_ts == ""


# ── _build_reply_instruction ─────────────────────────────


class TestBuildReplyInstruction:
    """Tests for _build_reply_instruction with external_thread_ts."""

    def test_uses_external_thread_ts_when_present(self):
        from core._anima_inbox import _build_reply_instruction
        from core.schemas import Message

        m = Message(
            from_person="slack:U1",
            to_person="sakura",
            content="reply",
            source="slack",
            source_message_id="2.0",
            external_user_id="U1",
            external_channel_id="C1",
            external_thread_ts="1.0",
        )
        result = _build_reply_instruction(m)
        assert "--thread 1.0" in result
        assert "--thread 2.0" not in result

    def test_falls_back_to_source_message_id(self):
        from core._anima_inbox import _build_reply_instruction
        from core.schemas import Message

        m = Message(
            from_person="slack:U1",
            to_person="sakura",
            content="top-level",
            source="slack",
            source_message_id="1.0",
            external_user_id="U1",
            external_channel_id="C1",
            external_thread_ts="",
        )
        result = _build_reply_instruction(m)
        assert "--thread 1.0" in result

    def test_no_thread_when_both_empty(self):
        from core._anima_inbox import _build_reply_instruction
        from core.schemas import Message

        m = Message(
            from_person="slack:U1",
            to_person="sakura",
            content="no ts",
            source="slack",
            source_message_id="",
            external_user_id="U1",
            external_channel_id="C1",
            external_thread_ts="",
        )
        result = _build_reply_instruction(m)
        assert "--thread" not in result

    def test_chatwork_unaffected(self):
        from core._anima_inbox import _build_reply_instruction
        from core.schemas import Message

        m = Message(
            from_person="chatwork:123",
            to_person="sakura",
            content="msg",
            source="chatwork",
            external_channel_id="456",
        )
        result = _build_reply_instruction(m)
        assert "chatwork" in result
        assert "--thread" not in result
