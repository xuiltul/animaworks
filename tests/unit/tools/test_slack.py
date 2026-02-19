"""Tests for core/tools/slack.py — Slack integration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from core.tools._base import ToolConfigError
from core.tools.slack import (
    JST,
    MessageCache,
    clean_slack_markup,
    format_slack_ts,
    get_tool_schemas,
    truncate,
)


# ── format_slack_ts ───────────────────────────────────────────────


class TestFormatSlackTs:
    def test_valid_timestamp(self):
        result = format_slack_ts("1707123456.789012")
        assert "2024-02-05" in result  # rough check

    def test_invalid_timestamp(self):
        assert format_slack_ts("not-a-ts") == "not-a-ts"

    def test_empty_string(self):
        assert format_slack_ts("") == ""


# ── clean_slack_markup ────────────────────────────────────────────


class TestCleanSlackMarkup:
    def test_empty(self):
        assert clean_slack_markup("") == ""

    def test_user_mention_no_cache(self):
        result = clean_slack_markup("Hello <@U06MJKLV0TG>!")
        assert "@U06MJKLV0TG" in result
        assert "<@U06MJKLV0TG>" not in result

    def test_user_mention_with_cache(self):
        cache = {"U06MJKLV0TG": "Alice"}
        result = clean_slack_markup("Hello <@U06MJKLV0TG>!", cache=cache)
        assert "@Alice" in result

    def test_channel_ref_with_name(self):
        result = clean_slack_markup("See <#C01234|general>")
        assert "#general" in result

    def test_channel_ref_without_name(self):
        result = clean_slack_markup("See <#C01234>")
        assert "#C01234" in result

    def test_url_with_label(self):
        result = clean_slack_markup("Click <https://example.com|here>")
        assert "here (https://example.com)" in result

    def test_url_without_label(self):
        result = clean_slack_markup("Link: <https://example.com>")
        assert "https://example.com" in result
        assert "<" not in result

    def test_html_entities(self):
        result = clean_slack_markup("a &amp; b &lt; c &gt; d")
        assert result == "a & b < c > d"


# ── truncate ──────────────────────────────────────────────────────


class TestTruncate:
    def test_short_string(self):
        assert truncate("hello", 80) == "hello"

    def test_long_string(self):
        result = truncate("a" * 100, 10)
        assert result == "a" * 10 + "..."

    def test_newlines_replaced(self):
        result = truncate("line1\nline2", 80)
        assert "\n" not in result
        assert "line1 line2" in result


# ── MessageCache ──────────────────────────────────────────────────


class TestMessageCache:
    def test_init_creates_tables(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        cache = MessageCache(db_path=db_path)
        try:
            # Verify tables exist
            rows = cache.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {r["name"] for r in rows}
            assert "channels" in table_names
            assert "messages" in table_names
            assert "users" in table_names
            assert "sync_state" in table_names
        finally:
            cache.close()

    def test_upsert_channel(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_channel({
                "id": "C123",
                "name": "general",
                "is_member": True,
            })
            row = cache.conn.execute(
                "SELECT * FROM channels WHERE channel_id = 'C123'"
            ).fetchone()
            assert row["name"] == "general"
            assert row["type"] == "public_channel"
        finally:
            cache.close()

    def test_upsert_channel_dm(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_channel({"id": "D123", "is_im": True, "user": "U999"})
            row = cache.conn.execute(
                "SELECT * FROM channels WHERE channel_id = 'D123'"
            ).fetchone()
            assert row["type"] == "im"
            assert "DM:" in row["name"]
        finally:
            cache.close()

    def test_upsert_user(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_user({
                "id": "U001",
                "profile": {"display_name": "Alice"},
                "real_name": "Alice Smith",
                "name": "alice",
            })
            row = cache.conn.execute(
                "SELECT * FROM users WHERE user_id = 'U001'"
            ).fetchone()
            assert row["name"] == "Alice"
        finally:
            cache.close()

    def test_upsert_messages_and_search(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            messages = [
                {
                    "ts": "1707123456.000001",
                    "user": "U001",
                    "text": "Hello world from Slack",
                    "thread_ts": "",
                    "reply_count": 0,
                },
                {
                    "ts": "1707123457.000002",
                    "user": "U002",
                    "text": "A different message",
                    "thread_ts": "",
                    "reply_count": 0,
                },
            ]
            cache.upsert_messages("C123", messages)

            results = cache.search("Hello")
            assert len(results) == 1
            assert "Hello world" in results[0]["text"]
        finally:
            cache.close()

    def test_search_with_channel_filter(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("C1", [
                {"ts": "1.0", "user": "U1", "text": "match this", "thread_ts": "", "reply_count": 0},
            ])
            cache.upsert_messages("C2", [
                {"ts": "2.0", "user": "U1", "text": "match this too", "thread_ts": "", "reply_count": 0},
            ])
            results = cache.search("match", channel_id="C1")
            assert len(results) == 1
        finally:
            cache.close()

    def test_get_recent(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("C1", [
                {"ts": "1.0", "user": "U1", "text": "first", "thread_ts": "", "reply_count": 0},
                {"ts": "2.0", "user": "U1", "text": "second", "thread_ts": "", "reply_count": 0},
                {"ts": "3.0", "user": "U1", "text": "third", "thread_ts": "", "reply_count": 0},
            ])
            results = cache.get_recent("C1", limit=2)
            assert len(results) == 2
            # Most recent first
            assert results[0]["text"] == "third"
        finally:
            cache.close()

    def test_find_mentions(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("C1", [
                {"ts": "1.0", "user": "U2", "text": "Hey <@U001>!", "thread_ts": "", "reply_count": 0},
                {"ts": "2.0", "user": "U001", "text": "my own msg", "thread_ts": "", "reply_count": 0},
            ])
            mentions = cache.find_mentions("U001")
            assert len(mentions) == 1
            assert mentions[0]["text"] == "Hey <@U001>!"
        finally:
            cache.close()

    def test_find_unreplied_threaded(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("C1", [
                {"ts": "1.0", "user": "U2", "text": "<@U001> help", "thread_ts": "1.0", "reply_count": 1},
                # U001 replied in thread
                {"ts": "1.5", "user": "U001", "text": "sure", "thread_ts": "1.0", "reply_count": 0},
            ])
            unreplied = cache.find_unreplied("U001")
            assert len(unreplied) == 0
        finally:
            cache.close()

    def test_find_unreplied_not_replied(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("C1", [
                {"ts": "1.0", "user": "U2", "text": "<@U001> need help", "thread_ts": "", "reply_count": 0},
            ])
            unreplied = cache.find_unreplied("U001")
            assert len(unreplied) == 1
        finally:
            cache.close()

    def test_get_channel_name(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_channel({"id": "C1", "name": "general"})
            assert cache.get_channel_name("C1") == "general"
            assert cache.get_channel_name("CUNKNOWN") == "CUNKNOWN"
        finally:
            cache.close()

    def test_get_user_name(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_user({"id": "U1", "profile": {"display_name": "Bob"}, "real_name": "Bob", "name": "bob"})
            assert cache.get_user_name("U1") == "Bob"
            assert cache.get_user_name("UUNKNOWN") == "UUNKNOWN"
        finally:
            cache.close()

    def test_get_user_name_cache(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_user({"id": "U1", "profile": {"display_name": "A"}, "real_name": "A", "name": "a"})
            cache.upsert_user({"id": "U2", "profile": {"display_name": "B"}, "real_name": "B", "name": "b"})
            name_map = cache.get_user_name_cache()
            assert name_map == {"U1": "A", "U2": "B"}
        finally:
            cache.close()

    def test_get_stats(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_channel({"id": "C1", "name": "ch1"})
            cache.upsert_user({"id": "U1", "profile": {"display_name": "A"}, "real_name": "A", "name": "a"})
            cache.upsert_messages("C1", [
                {"ts": "1.0", "user": "U1", "text": "msg", "thread_ts": "", "reply_count": 0},
            ])
            stats = cache.get_stats()
            assert stats["channels"] == 1
            assert stats["users"] == 1
            assert stats["messages"] == 1
        finally:
            cache.close()

    def test_update_sync_state(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("C1", [
                {"ts": "1.0", "user": "U1", "text": "a", "thread_ts": "", "reply_count": 0},
                {"ts": "2.0", "user": "U1", "text": "b", "thread_ts": "", "reply_count": 0},
            ])
            cache.update_sync_state("C1")
            row = cache.conn.execute(
                "SELECT * FROM sync_state WHERE channel_id = 'C1'"
            ).fetchone()
            assert row is not None
            assert row["oldest_ts"] == "1.0"
            assert row["newest_ts"] == "2.0"
        finally:
            cache.close()


# ── SlackClient (mocked SDK) ─────────────────────────────────────


class TestSlackClient:
    """Test SlackClient with mocked slack_sdk."""

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        # We need to mock the slack_sdk import first
        mock_wc = MagicMock()
        mock_sae = type("SlackApiError", (Exception,), {})
        with patch.dict("core.tools.slack.__dict__", {"WebClient": mock_wc, "SlackApiError": mock_sae}):
            with patch("core.tools.slack.get_credential", side_effect=ToolConfigError("no token")):
                with pytest.raises(ToolConfigError):
                    from core.tools.slack import SlackClient
                    SlackClient()


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 5
        names = {s["name"] for s in schemas}
        assert names == {"slack_send", "slack_messages", "slack_search", "slack_unreplied", "slack_channels"}

    def test_slack_send_requires_channel_and_message(self):
        schemas = get_tool_schemas()
        send_schema = [s for s in schemas if s["name"] == "slack_send"][0]
        assert set(send_schema["input_schema"]["required"]) == {"channel", "message"}
