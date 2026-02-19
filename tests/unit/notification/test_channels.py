"""Tests for notification channel implementations."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from core.notification.channels.slack import SlackChannel
from core.notification.channels.line import LineChannel
from core.notification.channels.telegram import TelegramChannel
from core.notification.channels.chatwork import ChatworkChannel
from core.notification.channels.ntfy import NtfyChannel


# ── Slack Channel ─────────────────────────────────────────────


class TestSlackChannel:
    @pytest.fixture
    def channel(self) -> SlackChannel:
        return SlackChannel({"webhook_url": "https://hooks.slack.com/test"})

    def test_channel_type(self, channel: SlackChannel):
        assert channel.channel_type == "slack"

    @pytest.mark.asyncio
    async def test_send_success(self, channel: SlackChannel):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Test", "Body")
            assert result == "slack: OK"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://hooks.slack.com/test"
            payload = call_args[1]["json"]
            assert "*Test*" in payload["text"]
            assert "Body" in payload["text"]

    @pytest.mark.asyncio
    async def test_send_no_webhook_url(self):
        channel = SlackChannel({})
        result = await channel.send("Test", "Body")
        assert "ERROR" in result
        assert "webhook_url" in result

    @pytest.mark.asyncio
    async def test_send_via_webhook_url_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SLACK_WEBHOOK", "https://hooks.slack.com/env-url")
        channel = SlackChannel({"webhook_url_env": "SLACK_WEBHOOK"})
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Test", "Body")
            assert result == "slack: OK"
            assert mock_client.post.call_args[0][0] == "https://hooks.slack.com/env-url"

    @pytest.mark.asyncio
    async def test_send_truncates_long_message(self, channel: SlackChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            long_body = "x" * 50000
            await channel.send("Test", long_body)
            payload = mock_client.post.call_args[1]["json"]
            assert len(payload["text"]) <= 40000

    @pytest.mark.asyncio
    async def test_send_with_priority_and_anima(self, channel: SlackChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send(
                "Alert", "Error occurred", "urgent", anima_name="alice"
            )
            assert result == "slack: OK"
            payload = mock_client.post.call_args[1]["json"]
            assert "[URGENT]" in payload["text"]
            assert "(from alice)" in payload["text"]

    @pytest.mark.asyncio
    async def test_send_http_error(self, channel: SlackChannel):
        with patch("core.notification.channels.slack.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response,
            )
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Test", "Body")
            assert "ERROR" in result
            assert "500" in result


# ── LINE Channel ──────────────────────────────────────────────


class TestLineChannel:
    @pytest.fixture
    def channel(self, monkeypatch: pytest.MonkeyPatch) -> LineChannel:
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        return LineChannel({
            "channel_access_token_env": "LINE_CHANNEL_ACCESS_TOKEN",
            "user_id": "U123456",
        })

    def test_channel_type(self, channel: LineChannel):
        assert channel.channel_type == "line"

    @pytest.mark.asyncio
    async def test_send_success(self, channel: LineChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.line.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Test", "Body")
            assert result == "line: OK"
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["to"] == "U123456"
            assert payload["messages"][0]["type"] == "text"
            assert "Test" in payload["messages"][0]["text"]

    @pytest.mark.asyncio
    async def test_send_no_token(self):
        channel = LineChannel({
            "channel_access_token_env": "NONEXISTENT_ENV_VAR",
            "user_id": "U123456",
        })
        result = await channel.send("Test", "Body")
        assert "ERROR" in result
        assert "channel_access_token_env" in result

    @pytest.mark.asyncio
    async def test_send_no_user_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
        channel = LineChannel({
            "channel_access_token_env": "LINE_CHANNEL_ACCESS_TOKEN",
        })
        result = await channel.send("Test", "Body")
        assert "ERROR" in result
        assert "user_id" in result


# ── Telegram Channel ─────────────────────────────────────────


class TestTelegramChannel:
    @pytest.fixture
    def channel(self, monkeypatch: pytest.MonkeyPatch) -> TelegramChannel:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot123:abc")
        return TelegramChannel({
            "bot_token_env": "TELEGRAM_BOT_TOKEN",
            "chat_id": "99999",
        })

    def test_channel_type(self, channel: TelegramChannel):
        assert channel.channel_type == "telegram"

    @pytest.mark.asyncio
    async def test_send_success(self, channel: TelegramChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Alert", "Details")
            assert result == "telegram: OK"
            call_args = mock_client.post.call_args
            assert "bot123:abc" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["chat_id"] == "99999"
            assert "Alert" in payload["text"]

    @pytest.mark.asyncio
    async def test_send_escapes_html(self, channel: TelegramChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("<script>alert('xss')</script>", "a < b & c > d")
            assert result == "telegram: OK"
            payload = mock_client.post.call_args[1]["json"]
            assert payload["parse_mode"] == "HTML"
            # HTML special chars should be escaped
            assert "<script>" not in payload["text"]
            assert "&lt;script&gt;" in payload["text"]
            assert "&amp;" in payload["text"]

    @pytest.mark.asyncio
    async def test_send_truncates_long_message(self, channel: TelegramChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            long_body = "x" * 5000
            await channel.send("Test", long_body)
            payload = mock_client.post.call_args[1]["json"]
            assert len(payload["text"]) <= 4096

    @pytest.mark.asyncio
    async def test_send_no_token(self):
        channel = TelegramChannel({
            "bot_token_env": "NONEXISTENT_VAR",
            "chat_id": "99999",
        })
        result = await channel.send("Test", "Body")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_send_no_chat_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot123:abc")
        channel = TelegramChannel({"bot_token_env": "TELEGRAM_BOT_TOKEN"})
        result = await channel.send("Test", "Body")
        assert "ERROR" in result
        assert "chat_id" in result


# ── Chatwork Channel ─────────────────────────────────────────


class TestChatworkChannel:
    @pytest.fixture
    def channel(self, monkeypatch: pytest.MonkeyPatch) -> ChatworkChannel:
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cw-token-123")
        return ChatworkChannel({
            "api_token_env": "CHATWORK_API_TOKEN",
            "room_id": "42",
        })

    def test_channel_type(self, channel: ChatworkChannel):
        assert channel.channel_type == "chatwork"

    @pytest.mark.asyncio
    async def test_send_success(self, channel: ChatworkChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.chatwork.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Test", "Body")
            assert result == "chatwork: OK"
            call_args = mock_client.post.call_args
            assert "/rooms/42/messages" in call_args[0][0]
            assert call_args[1]["headers"]["X-ChatWorkToken"] == "cw-token-123"
            assert "[info]" in call_args[1]["data"]["body"]

    @pytest.mark.asyncio
    async def test_send_no_token(self):
        channel = ChatworkChannel({
            "api_token_env": "NONEXISTENT_VAR",
            "room_id": "42",
        })
        result = await channel.send("Test", "Body")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_send_no_room_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cw-token-123")
        channel = ChatworkChannel({"api_token_env": "CHATWORK_API_TOKEN"})
        result = await channel.send("Test", "Body")
        assert "ERROR" in result
        assert "room_id" in result

    @pytest.mark.asyncio
    async def test_send_non_numeric_room_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cw-token-123")
        channel = ChatworkChannel({
            "api_token_env": "CHATWORK_API_TOKEN",
            "room_id": "abc-not-a-number",
        })
        result = await channel.send("Test", "Body")
        assert "ERROR" in result
        assert "numeric" in result

    @pytest.mark.asyncio
    async def test_send_truncates_long_message(self, channel: ChatworkChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.chatwork.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            long_body = "x" * 30000
            await channel.send("Test", long_body)
            sent_body = mock_client.post.call_args[1]["data"]["body"]
            assert len(sent_body) <= 28000


# ── ntfy Channel ──────────────────────────────────────────────


class TestNtfyChannel:
    @pytest.fixture
    def channel(self) -> NtfyChannel:
        return NtfyChannel({
            "server_url": "https://ntfy.sh",
            "topic": "test-topic",
        })

    def test_channel_type(self, channel: NtfyChannel):
        assert channel.channel_type == "ntfy"

    @pytest.mark.asyncio
    async def test_send_success(self, channel: NtfyChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Test", "Body")
            assert result == "ntfy: OK"
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://ntfy.sh/test-topic"
            assert call_args[1]["content"] == "Body"
            assert call_args[1]["headers"]["Title"] == "Test"
            assert call_args[1]["headers"]["Priority"] == "3"

    @pytest.mark.asyncio
    async def test_send_with_priority_mapping(self, channel: NtfyChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await channel.send("Alert", "Critical!", "urgent")
            headers = mock_client.post.call_args[1]["headers"]
            assert headers["Priority"] == "5"

    @pytest.mark.asyncio
    async def test_send_with_auth_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NTFY_TOKEN", "ntfy-secret")
        channel = NtfyChannel({
            "server_url": "https://ntfy.example.com",
            "topic": "test",
            "token_env": "NTFY_TOKEN",
        })
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await channel.send("Test", "Body")
            headers = mock_client.post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer ntfy-secret"

    @pytest.mark.asyncio
    async def test_send_no_topic(self):
        channel = NtfyChannel({"server_url": "https://ntfy.sh"})
        result = await channel.send("Test", "Body")
        assert "ERROR" in result
        assert "topic" in result

    @pytest.mark.asyncio
    async def test_send_default_server_url(self):
        channel = NtfyChannel({"topic": "test-topic"})
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await channel.send("Test", "Body")
            assert result == "ntfy: OK"
            assert mock_client.post.call_args[0][0] == "https://ntfy.sh/test-topic"

    @pytest.mark.asyncio
    async def test_send_with_anima_name(self, channel: NtfyChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await channel.send("Test", "Body", anima_name="bob")
            headers = mock_client.post.call_args[1]["headers"]
            assert "(from bob)" in headers["Title"]

    @pytest.mark.asyncio
    async def test_send_truncates_body_and_title(self, channel: NtfyChannel):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            long_subject = "A" * 300
            long_body = "B" * 5000
            await channel.send(long_subject, long_body)
            call_args = mock_client.post.call_args
            assert len(call_args[1]["headers"]["Title"]) <= 256
            assert len(call_args[1]["content"]) <= 4096
