"""Tests for BoardSlackSync outbound allow-list behavior."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    ExternalMessagingChannelConfig,
    ExternalMessagingConfig,
)
from core.messaging.outbound_auto import BoardSlackSync


class _FakeSlackResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True, "ts": "123.456"}


class _FakeAsyncClient:
    def __init__(self):
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers, json):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return _FakeSlackResponse()


def _board_sync_config(
    *,
    enabled: bool = True,
    outbound_sync: list[str] | None = None,
    outbound_sync_all: bool = False,
) -> AnimaWorksConfig:
    return AnimaWorksConfig(
        external_messaging=ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                enabled=enabled,
                board_mapping={"CGENERAL": "general", "COPS": "ops"},
                board_outbound_sync=outbound_sync or [],
                board_outbound_sync_all=outbound_sync_all,
            )
        )
    )


class TestBoardSlackSync:
    @pytest.mark.asyncio
    async def test_empty_outbound_sync_denies_by_default(self):
        cfg = _board_sync_config(outbound_sync=[], outbound_sync_all=False)

        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.messaging.outbound_auto._resolve_bot_token") as mock_token,
            patch("core.channels.slack.httpx.AsyncClient") as mock_client,
        ):
            result = await BoardSlackSync().sync_board_post("general", "hello", "sakura")

        assert result == "skipped"
        mock_token.assert_not_called()
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_outbound_sync_with_all_flag_allows_mapped_board(self):
        cfg = _board_sync_config(outbound_sync=[], outbound_sync_all=True)
        client = _FakeAsyncClient()

        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.messaging.outbound_auto._resolve_bot_token", return_value="xoxb-test"),
            patch("core.messaging.outbound_auto._resolve_avatar_url", return_value=""),
            patch("core.channels.slack.httpx.AsyncClient", return_value=client),
        ):
            result = await BoardSlackSync().sync_board_post("general", "hello", "sakura")

        assert result == "sent"
        assert client.posts[0]["json"]["channel"] == "CGENERAL"

    @pytest.mark.asyncio
    async def test_non_empty_outbound_sync_allows_only_whitelisted_board(self):
        cfg = _board_sync_config(outbound_sync=["ops"], outbound_sync_all=False)
        client = _FakeAsyncClient()

        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.messaging.outbound_auto._resolve_bot_token", return_value="xoxb-test"),
            patch("core.messaging.outbound_auto._resolve_avatar_url", return_value=""),
            patch("core.channels.slack.httpx.AsyncClient", return_value=client),
        ):
            result = await BoardSlackSync().sync_board_post("ops", "hello", "sakura")

        assert result == "sent"
        assert client.posts[0]["json"]["channel"] == "COPS"

    @pytest.mark.asyncio
    async def test_non_empty_outbound_sync_takes_precedence_over_all_flag(self):
        cfg = _board_sync_config(outbound_sync=["ops"], outbound_sync_all=True)

        with (
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.messaging.outbound_auto._resolve_bot_token") as mock_token,
            patch("core.channels.slack.httpx.AsyncClient") as mock_client,
        ):
            result = await BoardSlackSync().sync_board_post("general", "hello", "sakura")

        assert result == "skipped"
        mock_token.assert_not_called()
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_slack_source_prevents_echo_sync(self):
        cfg = _board_sync_config(outbound_sync=[], outbound_sync_all=True)

        with (
            patch("core.config.models.load_config", return_value=cfg) as mock_config,
            patch("core.messaging.outbound_auto._resolve_bot_token") as mock_token,
        ):
            result = await BoardSlackSync().sync_board_post(
                "general",
                "hello",
                "sakura",
                source="slack",
            )

        assert result == "skipped"
        mock_config.assert_not_called()
        mock_token.assert_not_called()
