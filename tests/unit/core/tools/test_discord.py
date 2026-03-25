# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Unit tests for Discord integration (client, markdown, cache, dispatch)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.tools._discord_cache import MessageCache
from core.tools._discord_client import DiscordAPIError, DiscordClient
from core.tools._discord_markdown import (
    DISCORD_MESSAGE_LIMIT,
    clean_discord_markup,
    format_discord_timestamp,
    md_to_discord,
    truncate,
)
from core.tools.discord import EXECUTION_PROFILE, dispatch, get_tool_schemas

# ── Helpers ──────────────────────────────────────────────────


def _mock_response(
    status_code: int = 200,
    json_data: object | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = json.dumps(json_data).encode() if json_data is not None else b""
    resp.text = text or (json.dumps(json_data) if json_data is not None else "")
    resp.json.return_value = json_data
    return resp


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def _patch_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.tools._discord_cache.get_data_dir", lambda: tmp_path)


# ── TestDiscordMarkdown ─────────────────────────────────────


class TestDiscordMarkdown:
    """Tests for core.tools._discord_markdown."""

    def test_clean_user_mention(self) -> None:
        assert clean_discord_markup("<@123456>") == "@123456"

    def test_clean_user_mention_with_cache(self) -> None:
        cache = {"123456": "alice"}
        assert clean_discord_markup("<@123456>", cache=cache) == "@alice"

    def test_clean_channel_mention(self) -> None:
        assert clean_discord_markup("<#789>") == "#789"

    def test_clean_emoji(self) -> None:
        assert clean_discord_markup("<:smile:123>") == ":smile:"

    def test_clean_animated_emoji(self) -> None:
        assert clean_discord_markup("<a:wave:456>") == ":wave:"

    def test_clean_timestamp(self) -> None:
        assert clean_discord_markup("<t:1700000000:F>") == "2023-11-15 07:13:20 JST"

    def test_clean_empty(self) -> None:
        assert clean_discord_markup("") == ""

    def test_md_to_discord_truncates(self) -> None:
        long_text = "a" * (DISCORD_MESSAGE_LIMIT + 50)
        out = md_to_discord(long_text)
        assert len(out) == DISCORD_MESSAGE_LIMIT
        assert out.endswith("...")
        assert out.startswith("a" * (DISCORD_MESSAGE_LIMIT - 3))

    def test_md_to_discord_passthrough(self) -> None:
        text = "**bold** and `code`"
        assert md_to_discord(text) == truncate(text, limit=DISCORD_MESSAGE_LIMIT)

    def test_truncate_short(self) -> None:
        assert truncate("hello world", limit=100) == "hello world"

    def test_truncate_newlines(self) -> None:
        assert truncate("a\nb\nc", limit=100) == "a b c"

    def test_format_discord_timestamp(self) -> None:
        snowflake = "1174109840998400000"
        assert format_discord_timestamp(snowflake) == "2023-11-15 07:13:20 JST"

    def test_format_discord_timestamp_invalid(self) -> None:
        assert format_discord_timestamp("not-a-number") == "not-a-number"


# ── TestDiscordCache ────────────────────────────────────────


@pytest.mark.usefixtures("_patch_data_dir")
class TestDiscordCache:
    """Tests for MessageCache (SQLite)."""

    def test_upsert_and_get_recent(self) -> None:
        cache = MessageCache()
        try:
            cid = "channel-1"
            messages = [
                {
                    "id": "m2",
                    "content": "second",
                    "timestamp": "2024-01-02T00:00:00+00:00",
                    "author": {"id": "u1", "username": "bob"},
                },
                {
                    "id": "m1",
                    "content": "first",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "author": {"id": "u1", "username": "bob"},
                },
            ]
            cache.upsert_messages(cid, messages)
            recent = cache.get_recent(cid, limit=10)
            assert len(recent) == 2
            assert recent[0]["id"] == "m2"
            assert recent[1]["id"] == "m1"
        finally:
            cache.close()

    def test_search(self) -> None:
        cache = MessageCache()
        try:
            cid = "ch-search"
            cache.upsert_messages(
                cid,
                [
                    {
                        "id": "a",
                        "content": "hello world",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "author": {"id": "u1", "username": "a"},
                    },
                    {
                        "id": "b",
                        "content": "other",
                        "timestamp": "2024-01-02T00:00:00+00:00",
                        "author": {"id": "u2", "username": "b"},
                    },
                ],
            )
            hits = cache.search("hello", limit=10)
            assert len(hits) == 1
            assert hits[0]["id"] == "a"
        finally:
            cache.close()

    def test_search_with_channel_filter(self) -> None:
        cache = MessageCache()
        try:
            cache.upsert_messages(
                "c1",
                [
                    {
                        "id": "x",
                        "content": "needle",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "author": {"id": "u1", "username": "a"},
                    },
                ],
            )
            cache.upsert_messages(
                "c2",
                [
                    {
                        "id": "y",
                        "content": "needle here",
                        "timestamp": "2024-01-02T00:00:00+00:00",
                        "author": {"id": "u1", "username": "a"},
                    },
                ],
            )
            only_c2 = cache.search("needle", channel_id="c2", limit=10)
            assert len(only_c2) == 1
            assert only_c2[0]["channel_id"] == "c2"
        finally:
            cache.close()

    def test_upsert_guild(self) -> None:
        cache = MessageCache()
        try:
            cache.upsert_guild({"id": "g1", "name": "Test Guild", "icon": "abc"})
            row = cache.conn.execute("SELECT * FROM guilds WHERE id = ?", ("g1",)).fetchone()
            assert row is not None
            assert dict(row)["name"] == "Test Guild"
        finally:
            cache.close()

    def test_upsert_channel(self) -> None:
        cache = MessageCache()
        try:
            cache.upsert_channel(
                {
                    "id": "ch1",
                    "guild_id": "g1",
                    "name": "general",
                    "type": 0,
                    "position": 1,
                },
            )
            row = cache.conn.execute("SELECT * FROM channels WHERE id = ?", ("ch1",)).fetchone()
            assert row is not None
            assert dict(row)["name"] == "general"
        finally:
            cache.close()

    def test_update_sync_state(self) -> None:
        cache = MessageCache()
        try:
            cache.update_sync_state("ch-sync", last_message_id="m-last")
            row = cache.conn.execute(
                "SELECT * FROM sync_state WHERE channel_id = ?",
                ("ch-sync",),
            ).fetchone()
            assert row is not None
            d = dict(row)
            assert d["last_message_id"] == "m-last"
            assert d["synced_at"]
        finally:
            cache.close()

    def test_get_user_name_cache(self) -> None:
        cache = MessageCache()
        try:
            cache.upsert_messages(
                "c-u",
                [
                    {
                        "id": "m1",
                        "content": "hi",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "author": {
                            "id": "42",
                            "username": "bob",
                            "global_name": "BobDisplay",
                        },
                    },
                ],
            )
            mapping = cache.get_user_name_cache()
            assert mapping.get("42") == "BobDisplay"
        finally:
            cache.close()

    def test_get_user_name(self) -> None:
        cache = MessageCache()
        try:
            cache.upsert_messages(
                "c-u2",
                [
                    {
                        "id": "m1",
                        "content": "x",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "author": {"id": "99", "username": "solo"},
                    },
                ],
            )
            assert cache.get_user_name("99") == "solo"
        finally:
            cache.close()


# ── TestDiscordClient ───────────────────────────────────────


class TestDiscordClient:
    """Tests for DiscordClient and DiscordAPIError."""

    def test_guilds(self) -> None:
        mock_http = MagicMock()
        mock_http.request.return_value = _mock_response(
            json_data=[{"id": "1", "name": "G"}],
        )
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                assert client.guilds() == [{"id": "1", "name": "G"}]
                mock_http.request.assert_called()
                call = mock_http.request.call_args
                assert call[0][0] == "GET"
                assert str(call[0][1]).endswith("/users/@me/guilds")
            finally:
                client.close()

    def test_channels(self) -> None:
        mock_http = MagicMock()
        payload = [
            {"id": "t1", "name": "text", "type": 0},
            {"id": "v1", "name": "voice", "type": 2},
            {"id": "a1", "name": "announce", "type": 5},
        ]
        mock_http.request.return_value = _mock_response(json_data=payload)
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                chans = client.channels("guild-x")
                ids = {c["id"] for c in chans}
                assert ids == {"t1", "a1"}
            finally:
                client.close()

    def test_send_message(self) -> None:
        mock_http = MagicMock()
        mock_http.request.return_value = _mock_response(json_data={"id": "msg1", "content": "hi"})
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                out = client.send_message("ch1", "hello")
                assert out["id"] == "msg1"
                call = mock_http.request.call_args
                assert call[0][0] == "POST"
                assert call[1]["json"] == {"content": "hello"}
            finally:
                client.close()

    def test_send_message_with_reply(self) -> None:
        mock_http = MagicMock()
        mock_http.request.return_value = _mock_response(json_data={"id": "m2"})
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                client.send_message("ch1", "reply text", reply_to="orig")
                call = mock_http.request.call_args
                body = call[1]["json"]
                assert body["content"] == "reply text"
                assert body["message_reference"] == {
                    "message_id": "orig",
                    "channel_id": "ch1",
                }
            finally:
                client.close()

    def test_channel_history(self) -> None:
        mock_http = MagicMock()
        mock_http.request.return_value = _mock_response(json_data=[{"id": "h1"}])
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                assert client.channel_history("ch1", limit=10) == [{"id": "h1"}]
                call = mock_http.request.call_args
                assert call[1]["params"] == {"limit": 10}
            finally:
                client.close()

    def test_add_reaction(self) -> None:
        mock_http = MagicMock()
        mock_http.request.return_value = _mock_response(json_data=None)
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                client.add_reaction("c1", "m1", "👍")
                call = mock_http.request.call_args
                assert call[0][0] == "PUT"
                path = call[0][1]
                assert "/reactions/" in path
                assert "%F0%9F%91%8D" in path or "👍" in path
            finally:
                client.close()

    def test_rate_limit_retry(self) -> None:
        mock_http = MagicMock()
        mock_http.request.side_effect = [
            _mock_response(
                status_code=429,
                json_data={"message": "rate limited", "retry_after": 0.01},
            ),
            _mock_response(json_data=[{"id": "ok"}]),
        ]
        with (
            patch("core.tools._discord_client.httpx.Client", return_value=mock_http),
            patch("core.tools._retry.time.sleep", lambda _s: None),
        ):
            client = DiscordClient(token="fake-token")
            try:
                assert client.guilds() == [{"id": "ok"}]
                assert mock_http.request.call_count == 2
            finally:
                client.close()

    def test_api_error(self) -> None:
        mock_http = MagicMock()
        mock_http.request.return_value = _mock_response(
            status_code=404,
            json_data={"message": "Unknown Channel", "code": 10003},
        )
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                with pytest.raises(DiscordAPIError) as exc_info:
                    client.guilds()
                err = exc_info.value
                assert err.status == 404
                assert err.code == 10003
                assert "404" in str(err)
            finally:
                client.close()

    def test_resolve_channel_by_id(self) -> None:
        client = DiscordClient(token="fake-token")
        try:
            assert client.resolve_channel("guild-z", "123456789012345678") == "123456789012345678"
        finally:
            client.close()

    def test_resolve_channel_by_name(self) -> None:
        mock_http = MagicMock()
        mock_http.request.return_value = _mock_response(
            json_data=[
                {"id": "99", "name": "general", "type": 0},
                {"id": "voice", "name": "VC", "type": 2},
            ],
        )
        with patch("core.tools._discord_client.httpx.Client", return_value=mock_http):
            client = DiscordClient(token="fake-token")
            try:
                cid = client.resolve_channel("guild-a", "#general")
                assert cid == "99"
            finally:
                client.close()


# ── TestDiscordDispatch ─────────────────────────────────────


class TestDiscordDispatch:
    """Tests for core.tools.discord dispatch and metadata."""

    def test_dispatch_discord_guilds(self) -> None:
        with patch("core.tools.discord.DiscordClient") as MockClient:
            mock_inst = MockClient.return_value
            mock_inst.guilds.return_value = [{"id": "g"}]
            out = dispatch("discord_guilds", {})
            assert out == [{"id": "g"}]
            mock_inst.guilds.assert_called_once_with()

    def test_dispatch_discord_channel_post(self) -> None:
        with patch("core.tools.discord.DiscordClient") as MockClient:
            mock_inst = MockClient.return_value
            mock_inst.send_message.return_value = {"id": "mid"}
            result = dispatch(
                "discord_channel_post",
                {"channel_id": "ch1", "text": "hello **world**"},
            )
            assert result == {"status": "ok", "channel_id": "ch1", "message_id": "mid"}
            mock_inst.send_message.assert_called_once()
            args, kwargs = mock_inst.send_message.call_args
            assert args[0] == "ch1"
            assert args[1] == md_to_discord("hello **world**")

    def test_dispatch_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("discord_nope", {})

    def test_get_tool_schemas(self) -> None:
        schemas = get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert "discord_channel_post" in names

    def test_execution_profile(self) -> None:
        assert "channel_post" in EXECUTION_PROFILE
        assert EXECUTION_PROFILE["channel_post"].get("gated") is True
