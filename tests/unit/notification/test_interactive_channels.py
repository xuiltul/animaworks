# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for interactive payloads on notification channels."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.schemas import AnimaWorksConfig, InteractionConfig
from core.notification.channels.chatwork import ChatworkChannel
from core.notification.channels.discord import DiscordChannel, _build_discord_components
from core.notification.channels.line import LineChannel
from core.notification.channels.ntfy import NtfyChannel
from core.notification.channels.slack import SlackChannel, _build_interactive_blocks
from core.notification.channels.telegram import TelegramChannel
from core.notification.interactive import InteractionRequest


def _sample_request() -> InteractionRequest:
    return InteractionRequest(
        callback_id="cb-interactive-1",
        anima_name="sakura",
        category="approval",
        options=["approve", "reject"],
        allowed_users={},
        metadata={},
        created_at=datetime.now(tz=UTC),
        approval_token="test-token",
        message_ts={},
    )


class TestSlackInteractiveBlocks:
    """Slack Block Kit structure for interactive notifications."""

    def test_build_interactive_blocks_structure(self):
        req = _sample_request()
        text = "*Subject*\nbody"
        blocks = _build_interactive_blocks(text, req)
        assert blocks[0]["type"] == "section"
        assert blocks[1]["type"] == "actions"
        assert blocks[1]["block_id"] == f"aw_interact:{req.callback_id}"
        elements = blocks[1]["elements"]
        assert len(elements) == 2
        assert elements[0]["action_id"] == "aw_interact_approve"
        assert elements[0]["value"] == req.callback_id
        assert elements[0]["style"] == "primary"
        assert elements[1]["action_id"] == "aw_interact_reject"
        assert elements[1]["style"] == "danger"

    @pytest.mark.asyncio
    async def test_send_includes_blocks_in_payload(self):
        req = _sample_request()
        channel = SlackChannel({"bot_token": "xoxb-test", "channel": "C123456"})

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678", "channel": "C123456"}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("core.notification.channels.slack.httpx.AsyncClient", return_value=mock_client),
            patch(
                "core.notification.interactive.update_interaction_message_ts_resilient",
            ) as mock_update_ts,
            patch("core.notification.reply_routing.save_notification_mapping"),
        ):
            result = await channel.send(
                "Hello",
                "World",
                anima_name="sakura",
                interaction=req,
            )

        assert "slack" in result.lower()
        mock_client.post.assert_called_once()
        call_kw = mock_client.post.call_args.kwargs
        payload = call_kw["json"]
        assert "blocks" in payload
        assert payload["blocks"][1]["type"] == "actions"
        mock_update_ts.assert_called_once_with(
            req.callback_id,
            "slack",
            "1234.5678",
        )


class TestDiscordComponents:
    """Discord Message Components for interactive notifications."""

    def test_build_discord_components_structure(self):
        req = _sample_request()
        rows = _build_discord_components(req)
        assert len(rows) == 1
        assert rows[0]["type"] == 1
        comps = rows[0]["components"]
        assert len(comps) == 2
        assert comps[0]["type"] == 2
        assert comps[0]["custom_id"] == f"aw_interact:{req.callback_id}:approve"
        assert comps[1]["custom_id"] == f"aw_interact:{req.callback_id}:reject"

    @pytest.mark.asyncio
    async def test_send_via_webhook_includes_components(self):
        req = _sample_request()
        channel = DiscordChannel({"webhook_url": "https://discord.com/api/webhooks/1/token"})

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"id": "999888777"}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "core.notification.interactive.update_interaction_message_ts_resilient",
            ) as mock_update_ts,
            patch(
                "core.tools._anima_icon_url.resolve_anima_icon_url",
                return_value="",
            ),
        ):
            await channel.send("Hi", "Body", anima_name="sakura", interaction=req)

        mock_client.post.assert_called_once()
        payload = mock_client.post.call_args.kwargs["json"]
        assert "components" in payload
        assert payload["components"][0]["components"][0]["custom_id"].endswith(":approve")
        mock_update_ts.assert_called_once_with(
            req.callback_id,
            "discord",
            "999888777",
        )


def _mock_config_with_web_base(web_base: str) -> AnimaWorksConfig:
    cfg = AnimaWorksConfig()
    cfg.interaction = InteractionConfig(web_base_url=web_base)
    return cfg


class TestTextFallbackChannels:
    """Channels append :func:`build_text_fallback` to the body."""

    @pytest.fixture
    def interaction_req(self) -> InteractionRequest:
        return _sample_request()

    @pytest.mark.asyncio
    async def test_chatwork_appends_fallback(self, interaction_req):
        channel = ChatworkChannel({"room_id": "12345", "api_token_env": "CW_TOK"})
        with (
            patch.object(channel, "_resolve_credential_with_vault", return_value="tok"),
            patch("core.notification.channels.chatwork.httpx.AsyncClient") as mock_ac_cls,
            patch(
                "core.config.models.load_config",
                return_value=_mock_config_with_web_base("https://app.example.com"),
            ),
            patch("core.tools.chatwork.md_to_chatwork", side_effect=lambda x: x),
        ):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ac_cls.return_value = mock_client

            await channel.send("S", "B", interaction=interaction_req)

        posted = mock_client.post.call_args.kwargs["data"]["body"]
        assert "[1]" in posted
        assert "https://app.example.com/api/approve/cb-interactive-1" in posted

    @pytest.mark.asyncio
    async def test_line_appends_fallback(self, interaction_req):
        channel = LineChannel({"user_id": "U1", "channel_access_token_env": "LINE_TOK"})
        with (
            patch.object(channel, "_resolve_credential_with_vault", return_value="tok"),
            patch("core.notification.channels.line.httpx.AsyncClient") as mock_ac_cls,
            patch(
                "core.config.models.load_config",
                return_value=_mock_config_with_web_base("https://line.example.com"),
            ),
        ):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ac_cls.return_value = mock_client

            await channel.send("S", "B", interaction=interaction_req)

        payload = mock_client.post.call_args.kwargs["json"]
        text = payload["messages"][0]["text"]
        assert "[1]" in text
        assert "https://line.example.com/api/approve/cb-interactive-1" in text

    @pytest.mark.asyncio
    async def test_telegram_appends_fallback(self, interaction_req):
        channel = TelegramChannel({"chat_id": "999", "bot_token_env": "TG_TOK"})
        with (
            patch.object(channel, "_resolve_credential_with_vault", return_value="123:ABC"),
            patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_ac_cls,
            patch(
                "core.config.models.load_config",
                return_value=_mock_config_with_web_base("https://tg.example.com"),
            ),
        ):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ac_cls.return_value = mock_client

            await channel.send("S", "B", interaction=interaction_req)

        payload = mock_client.post.call_args.kwargs["json"]
        text = payload["text"]
        assert "[1]" in text
        assert "https://tg.example.com/api/approve/cb-interactive-1" in text

    @pytest.mark.asyncio
    async def test_ntfy_appends_fallback(self, interaction_req):
        channel = NtfyChannel({"server_url": "https://ntfy.example.com", "topic": "alerts"})
        with (
            patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_ac_cls,
            patch(
                "core.config.models.load_config",
                return_value=_mock_config_with_web_base("https://ntfy.example.com"),
            ),
        ):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ac_cls.return_value = mock_client

            await channel.send("S", "B", interaction=interaction_req)

        body = mock_client.post.call_args.kwargs["content"]
        assert "[1]" in body
        assert "https://ntfy.example.com/api/approve/cb-interactive-1" in body
