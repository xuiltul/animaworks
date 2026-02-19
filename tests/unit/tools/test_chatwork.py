"""Tests for core/tools/chatwork.py — Chatwork integration."""
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
from core.tools.chatwork import (
    ChatworkClient,
    MessageCache,
    clean_chatwork_tags,
    get_tool_schemas,
    _format_timestamp,
    JST,
)


# ── clean_chatwork_tags ───────────────────────────────────────────


class TestCleanChatworkTags:
    def test_removes_to_tag(self):
        text = "[To:12345]Alice\nHello there"
        result = clean_chatwork_tags(text)
        assert "[To:" not in result
        assert "Hello there" in result

    def test_removes_toall(self):
        assert "[toall]" not in clean_chatwork_tags("[toall]Hey everyone")

    def test_removes_info_block(self):
        text = "Before [info]some info[/info] After"
        result = clean_chatwork_tags(text)
        assert "[info block]" in result
        assert "[info]" not in result.replace("[info block]", "")

    def test_removes_other_tags(self):
        text = "[hr]line[code]x = 1[/code]"
        result = clean_chatwork_tags(text)
        assert "[hr]" not in result
        assert "[code]" not in result

    def test_empty_text(self):
        assert clean_chatwork_tags("") == ""


# ── _format_timestamp ─────────────────────────────────────────────


class TestFormatTimestamp:
    def test_formats_unix_timestamp(self):
        # 2026-01-15 00:00:00 UTC ~= 2026-01-15 09:00:00 JST
        ts = int(datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        result = _format_timestamp(ts)
        assert "2026-01-15" in result


# ── ChatworkClient (mocked requests) ─────────────────────────────


class TestChatworkClient:
    @pytest.fixture(autouse=True)
    def _mock_requests(self, monkeypatch: pytest.MonkeyPatch):
        """Pre-mock the requests library and set token."""
        monkeypatch.setenv("CHATWORK_API_TOKEN", "test-cw-token")
        mock_requests = MagicMock()
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        with patch.dict("core.tools.chatwork.__dict__", {"requests": mock_requests}):
            self._mock_session = mock_session
            self._mock_requests = mock_requests
            yield

    def _make_response(self, status_code: int = 200, json_data=None, text=""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text or ("" if json_data is None else "json")
        resp.json.return_value = json_data
        resp.headers = {}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return resp

    def test_init_with_token(self):
        client = ChatworkClient(api_token="my-token")
        assert client.api_token == "my-token"

    def test_init_from_env(self):
        with patch("core.tools.chatwork.get_credential", return_value="test-cw-token"):
            client = ChatworkClient()
        assert client.api_token == "test-cw-token"

    def test_init_missing_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CHATWORK_API_TOKEN", raising=False)
        with patch("core.tools.chatwork.get_credential", side_effect=ToolConfigError("no token")):
            with pytest.raises(ToolConfigError):
                ChatworkClient()

    def test_me(self):
        self._mock_session.request.return_value = self._make_response(
            json_data={"account_id": 123, "name": "Bot"}
        )
        client = ChatworkClient()
        result = client.me()
        assert result["account_id"] == 123

    def test_rooms(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 1, "name": "Room1"}]
        )
        client = ChatworkClient()
        result = client.rooms()
        assert len(result) == 1

    def test_post_message(self):
        self._mock_session.request.return_value = self._make_response(
            json_data={"message_id": "m1"}
        )
        client = ChatworkClient()
        result = client.post_message("123", "Hello")
        assert result["message_id"] == "m1"

    def test_post_message_too_long(self):
        client = ChatworkClient()
        with pytest.raises(ValueError, match="10,000"):
            client.post_message("123", "x" * 10001)

    def test_get_room_by_name_exact(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[
                {"room_id": 1, "name": "alpha"},
                {"room_id": 2, "name": "beta"},
            ]
        )
        client = ChatworkClient()
        result = client.get_room_by_name("alpha")
        assert result["room_id"] == 1

    def test_get_room_by_name_partial(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[
                {"room_id": 1, "name": "alpha-project"},
                {"room_id": 2, "name": "beta-team"},
            ]
        )
        client = ChatworkClient()
        result = client.get_room_by_name("beta")
        assert result["room_id"] == 2

    def test_get_room_by_name_not_found(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 1, "name": "only-room"}]
        )
        client = ChatworkClient()
        result = client.get_room_by_name("nonexistent")
        assert result is None

    def test_resolve_room_id_numeric(self):
        client = ChatworkClient()
        assert client.resolve_room_id("12345") == "12345"

    def test_resolve_room_id_by_name(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 999, "name": "target"}]
        )
        client = ChatworkClient()
        assert client.resolve_room_id("target") == "999"

    def test_resolve_room_id_not_found(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 1, "name": "other"}]
        )
        client = ChatworkClient()
        with pytest.raises(ToolConfigError):
            client.resolve_room_id("missing")

    def test_rate_limit_retry(self):
        rate_resp = self._make_response(status_code=429)
        rate_resp.raise_for_status = MagicMock()  # 429 doesn't raise, it retries
        rate_resp.headers = {"Retry-After": "0"}
        ok_resp = self._make_response(status_code=200, json_data={"ok": True})
        self._mock_session.request.side_effect = [rate_resp, ok_resp]

        client = ChatworkClient()
        with patch("core.tools.chatwork.time.sleep"):
            result = client.get("/me")
        assert result == {"ok": True}

    def test_204_returns_none(self):
        resp = self._make_response(status_code=204)
        self._mock_session.request.return_value = resp
        client = ChatworkClient()
        result = client.get("/rooms/123/messages")
        assert result is None

    def test_my_tasks(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"task_id": 1, "body": "Do something"}]
        )
        client = ChatworkClient()
        tasks = client.my_tasks()
        assert len(tasks) == 1


# ── MessageCache ──────────────────────────────────────────────────


class TestChatworkMessageCache:
    def test_init_creates_tables(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            tables = {r["name"] for r in cache.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            assert "rooms" in tables
            assert "messages" in tables
            assert "sync_state" in tables
        finally:
            cache.close()

    def test_upsert_room(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_room({"room_id": 1, "name": "TestRoom"})
            row = cache.conn.execute(
                "SELECT * FROM rooms WHERE room_id = '1'"
            ).fetchone()
            assert row["name"] == "TestRoom"
        finally:
            cache.close()

    def test_upsert_messages_and_search(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            messages = [
                {
                    "message_id": "m1",
                    "send_time": 1707123456,
                    "account": {"account_id": "a1", "name": "Alice"},
                    "body": "Hello from Chatwork",
                },
            ]
            cache.upsert_messages("100", messages)
            results = cache.search("Hello")
            assert len(results) == 1
            assert "Hello from Chatwork" in results[0]["body"]
        finally:
            cache.close()

    def test_get_recent(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("100", [
                {"message_id": "m1", "send_time": 1, "account": {"account_id": "a1", "name": "A"}, "body": "first"},
                {"message_id": "m2", "send_time": 2, "account": {"account_id": "a1", "name": "A"}, "body": "second"},
            ])
            results = cache.get_recent("100", limit=1)
            assert len(results) == 1
            assert results[0]["body"] == "second"
        finally:
            cache.close()

    def test_find_mentions(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("100", [
                {
                    "message_id": "m1",
                    "send_time": 1,
                    "account": {"account_id": "other", "name": "Other"},
                    "body": "[To:myid]Alice\nPlease check",
                },
                {
                    "message_id": "m2",
                    "send_time": 2,
                    "account": {"account_id": "myid", "name": "Me"},
                    "body": "my own msg",
                },
            ])
            mentions = cache.find_mentions("myid")
            assert len(mentions) == 1
            assert "Please check" in mentions[0]["body"]
        finally:
            cache.close()

    def test_find_unreplied(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_messages("100", [
                {
                    "message_id": "m1",
                    "send_time": 1,
                    "account": {"account_id": "other", "name": "Other"},
                    "body": "[To:myid]Check this",
                },
            ])
            unreplied = cache.find_unreplied("myid")
            assert len(unreplied) == 1

            # Now add a reply
            cache.upsert_messages("100", [
                {
                    "message_id": "m2",
                    "send_time": 2,
                    "account": {"account_id": "myid", "name": "Me"},
                    "body": "Done",
                },
            ])
            unreplied = cache.find_unreplied("myid")
            assert len(unreplied) == 0
        finally:
            cache.close()

    def test_get_room_name(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_room({"room_id": 1, "name": "Dev Room"})
            assert cache.get_room_name("1") == "Dev Room"
            assert cache.get_room_name("999") == "999"
        finally:
            cache.close()

    def test_get_stats(self, tmp_path: Path):
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_room({"room_id": 1, "name": "R"})
            cache.upsert_messages("1", [
                {"message_id": "m1", "send_time": 1, "account": {"account_id": "a", "name": "A"}, "body": "msg"},
            ])
            stats = cache.get_stats()
            assert stats["rooms"] == 1
            assert stats["messages"] == 1
        finally:
            cache.close()


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 5
        names = {s["name"] for s in schemas}
        assert names == {
            "chatwork_send", "chatwork_messages", "chatwork_search",
            "chatwork_unreplied", "chatwork_rooms",
        }

    def test_chatwork_send_requires_room_and_message(self):
        schemas = get_tool_schemas()
        send = [s for s in schemas if s["name"] == "chatwork_send"][0]
        assert set(send["input_schema"]["required"]) == {"room", "message"}
