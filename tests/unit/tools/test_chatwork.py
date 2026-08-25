"""Tests for core/tools/chatwork.py — Chatwork integration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools._base import ToolConfigError
from core.tools.chatwork import (
    ChatworkClient,
    MessageCache,
    _format_timestamp,
    _sync_rooms,
    clean_chatwork_tags,
    get_tool_schemas,
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
        ts = int(datetime(2026, 1, 15, 0, 0, 0, tzinfo=UTC).timestamp())
        result = _format_timestamp(ts)
        assert "2026-01-15" in result


# ── ChatworkClient (mocked requests) ─────────────────────────────


class TestChatworkClient:
    @pytest.fixture(autouse=True)
    def _mock_requests(self, monkeypatch: pytest.MonkeyPatch):
        """Pre-mock the requests library and set token."""
        mock_requests = MagicMock()
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        with patch.dict("core.tools._chatwork_client.__dict__", {"requests": mock_requests}):
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

    def test_init_requires_token(self):
        with pytest.raises(TypeError):
            ChatworkClient()  # type: ignore[call-arg]

    def test_me(self):
        self._mock_session.request.return_value = self._make_response(
            json_data={"account_id": 123, "name": "Bot"}
        )
        client = ChatworkClient(api_token="test-cw-token")
        result = client.me()
        assert result["account_id"] == 123

    def test_rooms(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 1, "name": "Room1"}]
        )
        client = ChatworkClient(api_token="test-cw-token")
        result = client.rooms()
        assert len(result) == 1

    def test_get_room_by_name_exact(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[
                {"room_id": 1, "name": "alpha"},
                {"room_id": 2, "name": "beta"},
            ]
        )
        client = ChatworkClient(api_token="test-cw-token")
        result = client.get_room_by_name("alpha")
        assert result["room_id"] == 1

    def test_get_room_by_name_partial(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[
                {"room_id": 1, "name": "alpha-project"},
                {"room_id": 2, "name": "beta-team"},
            ]
        )
        client = ChatworkClient(api_token="test-cw-token")
        result = client.get_room_by_name("beta")
        assert result["room_id"] == 2

    def test_get_room_by_name_not_found(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 1, "name": "only-room"}]
        )
        client = ChatworkClient(api_token="test-cw-token")
        result = client.get_room_by_name("nonexistent")
        assert result is None

    def test_resolve_room_id_numeric(self):
        client = ChatworkClient(api_token="test-cw-token")
        assert client.resolve_room_id("12345") == "12345"

    def test_resolve_room_id_by_name(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 999, "name": "target"}]
        )
        client = ChatworkClient(api_token="test-cw-token")
        assert client.resolve_room_id("target") == "999"

    def test_resolve_room_id_not_found(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"room_id": 1, "name": "other"}]
        )
        client = ChatworkClient(api_token="test-cw-token")
        with pytest.raises(ToolConfigError):
            client.resolve_room_id("missing")

    def test_rate_limit_retry(self):
        rate_resp = self._make_response(status_code=429)
        rate_resp.raise_for_status = MagicMock()  # 429 doesn't raise, it retries
        rate_resp.headers = {"Retry-After": "0"}
        ok_resp = self._make_response(status_code=200, json_data={"ok": True})
        self._mock_session.request.side_effect = [rate_resp, ok_resp]

        client = ChatworkClient(api_token="test-cw-token")
        with patch("core.tools._retry.time.sleep"):
            result = client.get("/me")
        assert result == {"ok": True}

    def test_204_returns_none(self):
        resp = self._make_response(status_code=204)
        self._mock_session.request.return_value = resp
        client = ChatworkClient(api_token="test-cw-token")
        result = client.get("/rooms/123/messages")
        assert result is None

    def test_my_tasks(self):
        self._mock_session.request.return_value = self._make_response(
            json_data=[{"task_id": 1, "body": "Do something"}]
        )
        client = ChatworkClient(api_token="test-cw-token")
        tasks = client.my_tasks()
        assert len(tasks) == 1

    def test_upload_file(self, tmp_path: Path):
        self._mock_session.request.return_value = self._make_response(
            json_data={"file_id": 1}
        )
        file_path = tmp_path / "report.pdf"
        file_path.write_bytes(b"%PDF-1.4 fake content")
        client = ChatworkClient(api_token="test-cw-token")
        result = client.upload_file("123", file_path, message="here it is")
        assert result == {"file_id": 1}
        self._mock_session.request.assert_called_once()
        args, kwargs = self._mock_session.request.call_args
        assert args[0] == "POST"
        assert args[1] == "https://api.chatwork.com/v2/rooms/123/files"
        files = kwargs["files"]
        assert "file" in files
        assert files["file"][0] == "report.pdf"
        assert kwargs["data"] == {"message": "here it is"}

    def test_upload_file_too_large(self, tmp_path: Path):
        file_path = tmp_path / "big.pdf"
        with file_path.open("wb") as f:
            f.truncate(5 * 1024 * 1024 + 1)
        client = ChatworkClient(api_token="test-cw-token")
        with pytest.raises(ValueError):
            client.upload_file("123", file_path)
        self._mock_session.request.assert_not_called()

    def test_upload_file_missing(self, tmp_path: Path):
        client = ChatworkClient(api_token="test-cw-token")
        with pytest.raises(FileNotFoundError):
            client.upload_file("123", tmp_path / "nope.pdf")
        self._mock_session.request.assert_not_called()


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
    def test_returns_empty_list(self):
        """External tool modules' get_tool_schemas() returns empty list."""
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert schemas == []


# ── _sync_rooms ───────────────────────────────────────────────────


class TestSyncRooms:
    def test_sync_populates_rooms_and_messages(self, tmp_path: Path):
        """_sync_rooms fetches rooms, upserts metadata, then syncs messages."""
        client = MagicMock(spec=ChatworkClient)
        client.rooms.return_value = [
            {"room_id": 100, "name": "Room A", "type": "group", "last_update_time": 2},
            {"room_id": 200, "name": "Room B", "type": "direct", "last_update_time": 1},
        ]
        client.get_messages.side_effect = [
            [{"message_id": "m1", "send_time": 10, "account": {"account_id": "a1", "name": "A"}, "body": "hello"}],
            [{"message_id": "m2", "send_time": 20, "account": {"account_id": "a2", "name": "B"}, "body": "world"}],
        ]

        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            with patch("core.tools._chatwork_cli.time.sleep"):
                result = _sync_rooms(client, cache, sync_limit=10)

            # Both rooms saved
            stats = cache.get_stats()
            assert stats["rooms"] == 2
            assert stats["messages"] == 2
            assert result["rooms"] == 2
            assert result["messages"] == 2

            # Room types preserved
            row = cache.conn.execute(
                "SELECT type FROM rooms WHERE room_id = '200'"
            ).fetchone()
            assert row["type"] == "direct"
        finally:
            cache.close()

    def test_sync_respects_limit(self, tmp_path: Path):
        """Only top N rooms by last_update_time get messages synced."""
        client = MagicMock(spec=ChatworkClient)
        client.rooms.return_value = [
            {"room_id": 1, "name": "Old", "type": "group", "last_update_time": 1},
            {"room_id": 2, "name": "New", "type": "group", "last_update_time": 100},
            {"room_id": 3, "name": "Mid", "type": "group", "last_update_time": 50},
        ]
        client.get_messages.return_value = [
            {"message_id": "m1", "send_time": 1, "account": {"account_id": "a", "name": "A"}, "body": "x"},
        ]

        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            with patch("core.tools._chatwork_cli.time.sleep"):
                result = _sync_rooms(client, cache, sync_limit=1)

            # All 3 rooms have metadata
            assert result["rooms"] == 3
            # But only 1 room's messages were fetched
            assert client.get_messages.call_count == 1
            # The room synced should be room_id=2 (highest last_update_time)
            client.get_messages.assert_called_once_with("2", force=True)
        finally:
            cache.close()

    def test_sync_continues_on_error(self, tmp_path: Path):
        """If get_messages fails for one room, sync continues with others."""
        client = MagicMock(spec=ChatworkClient)
        client.rooms.return_value = [
            {"room_id": 1, "name": "Fail", "type": "group", "last_update_time": 2},
            {"room_id": 2, "name": "OK", "type": "group", "last_update_time": 1},
        ]
        client.get_messages.side_effect = [
            Exception("API error"),
            [{"message_id": "m1", "send_time": 1, "account": {"account_id": "a", "name": "A"}, "body": "ok"}],
        ]

        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            with patch("core.tools._chatwork_cli.time.sleep"):
                result = _sync_rooms(client, cache, sync_limit=10)

            assert result["rooms"] == 2
            assert result["messages"] == 1
        finally:
            cache.close()

    def test_sync_handles_none_messages(self, tmp_path: Path):
        """get_messages returning None (no unread) is handled gracefully."""
        client = MagicMock(spec=ChatworkClient)
        client.rooms.return_value = [
            {"room_id": 1, "name": "Empty", "type": "group", "last_update_time": 1},
        ]
        client.get_messages.return_value = None

        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            with patch("core.tools._chatwork_cli.time.sleep"):
                result = _sync_rooms(client, cache, sync_limit=10)

            assert result["rooms"] == 1
            assert result["messages"] == 0
        finally:
            cache.close()

    def test_sync_empty_rooms(self, tmp_path: Path):
        """Rooms API returning empty list is handled gracefully."""
        client = MagicMock(spec=ChatworkClient)
        client.rooms.return_value = []

        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            with patch("core.tools._chatwork_cli.time.sleep"):
                result = _sync_rooms(client, cache, sync_limit=10)

            assert result["rooms"] == 0
            assert result["messages"] == 0
        finally:
            cache.close()


# ── find_mentions with DM rooms ──────────────────────────────────


class TestFindMentionsWithDMRooms:
    """Test that find_mentions detects DM messages after rooms are synced."""

    def test_dm_room_messages_detected(self, tmp_path: Path):
        """Messages in type='direct' rooms are treated as mentions."""
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            # Sync a direct room
            cache.upsert_room({"room_id": 500, "name": "DM with Alice", "type": "direct"})
            cache.upsert_messages("500", [
                {
                    "message_id": "dm1",
                    "send_time": 100,
                    "account": {"account_id": "alice", "name": "Alice"},
                    "body": "Hey, can you check this?",
                },
            ])

            # Should find as mention without [To:] tag
            mentions = cache.find_mentions("myid", config={"unreplied": {"include_direct_messages": True}})
            assert len(mentions) == 1
            assert mentions[0]["message_id"] == "dm1"
        finally:
            cache.close()

    def test_watch_room_messages_detected(self, tmp_path: Path):
        """Messages in watch_rooms are treated as mentions."""
        cache = MessageCache(db_path=tmp_path / "test.db")
        try:
            cache.upsert_room({"room_id": 600, "name": "Ops Channel", "type": "group"})
            cache.upsert_messages("600", [
                {
                    "message_id": "w1",
                    "send_time": 100,
                    "account": {"account_id": "bob", "name": "Bob"},
                    "body": "Server is down",
                },
            ])

            config = {"unreplied": {"watch_rooms": [{"room_id": "600"}]}}
            mentions = cache.find_mentions("myid", config=config)
            assert len(mentions) == 1
        finally:
            cache.close()


# ── dispatch ──────────────────────────────────────────────────────


class TestDispatch:
    def test_dispatch_chatwork_sync(self, tmp_path: Path):
        """dispatch('chatwork_sync') calls _sync_rooms."""
        mock_requests = MagicMock()
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        mock_session.request.return_value = MagicMock(
            status_code=200,
            text="json",
            json=MagicMock(return_value=[
                {"room_id": 1, "name": "R1", "type": "group", "last_update_time": 1},
            ]),
            headers={},
            raise_for_status=MagicMock(),
        )

        from core.tools.chatwork import dispatch

        with (
            patch.dict("core.tools._chatwork_client.__dict__", {"requests": mock_requests}),
            patch("core.tools.chatwork.resolve_identity", return_value=MagicMock(token="test-token")),
            patch("core.tools.chatwork.resolve_cache_db_path", return_value=tmp_path / "1" / "messages.db"),
            patch("core.tools.chatwork.MessageCache") as MockCache,
        ):
            mock_cache = MagicMock()
            MockCache.return_value = mock_cache
            with patch("core.tools._chatwork_cli.time.sleep"):
                result = dispatch("chatwork_sync", {"limit": 5})

            assert result["rooms"] == 1

    def test_dispatch_chatwork_mentions(self, tmp_path: Path):
        """dispatch('chatwork_mentions') calls find_mentions."""
        mock_requests = MagicMock()
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session

        # me() call
        me_resp = MagicMock(
            status_code=200, text="json",
            json=MagicMock(return_value={"account_id": 123, "name": "Bot"}),
            headers={}, raise_for_status=MagicMock(),
        )
        mock_session.request.return_value = me_resp

        from core.tools.chatwork import dispatch

        with (
            patch.dict("core.tools._chatwork_client.__dict__", {"requests": mock_requests}),
            patch("core.tools.chatwork.resolve_identity", return_value=MagicMock(token="test-token")),
            patch("core.tools.chatwork.resolve_cache_db_path", return_value=tmp_path / "123" / "messages.db"),
            patch("core.tools.chatwork._load_chatwork_tool_config", return_value={}),
            patch("core.tools.chatwork.MessageCache") as MockCache,
        ):
            mock_cache = MagicMock()
            mock_cache.find_mentions.return_value = [{"message_id": "m1"}]
            MockCache.return_value = mock_cache

            result = dispatch("chatwork_mentions", {"include_toall": False, "limit": 50})

            assert len(result) == 1
            mock_cache.find_mentions.assert_called_once_with(
                "123",
                exclude_toall=True,
                limit=50,
                config={},
            )

    def test_dispatch_chatwork_unreplied_uses_resolved_identity(self, tmp_path: Path):
        """unreplied uses the delegated account's token and account ID."""
        from core.tools.chatwork import dispatch

        client = MagicMock()
        client.me.return_value = {"account_id": 987, "name": "Delegated"}
        identity = MagicMock(token="delegated-token")
        with (
            patch("core.tools.chatwork.resolve_identity", return_value=identity) as resolver,
            patch("core.tools.chatwork.ChatworkClient", return_value=client) as client_class,
            patch("core.tools.chatwork.resolve_cache_db_path", return_value=tmp_path / "987" / "messages.db"),
            patch("core.tools.chatwork._load_chatwork_tool_config", return_value={}),
            patch("core.tools.chatwork.MessageCache") as cache_class,
        ):
            cache = cache_class.return_value
            cache.find_unreplied.return_value = [{"message_id": "m2"}]
            result = dispatch(
                "chatwork_unreplied",
                {"as": "owner", "anima_dir": "/srv/animas/mei"},
            )

        assert result == [{"message_id": "m2"}]
        resolver.assert_called_once_with("owner", anima_dir="/srv/animas/mei")
        client_class.assert_called_once_with(api_token="delegated-token")
        cache.find_unreplied.assert_called_once_with(
            "987",
            exclude_toall=True,
            config={},
        )


# ── EXECUTION_PROFILE ─────────────────────────────────────────────


class TestExecutionProfile:
    def test_profile_covers_all_commands(self):
        from core.tools.chatwork import EXECUTION_PROFILE

        expected = {
            "rooms", "messages", "send", "upload", "search", "unreplied",
            "sync", "me", "members", "contacts", "task",
            "mytasks", "tasks", "mentions", "stats", "files", "download",
            "delete",
        }
        assert set(EXECUTION_PROFILE.keys()) == expected

    def test_sync_is_background_eligible(self):
        from core.tools.chatwork import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["sync"]["background_eligible"] is True

    def test_other_commands_not_background_eligible(self):
        """Commands other than sync and download are not background-eligible."""
        from core.tools.chatwork import EXECUTION_PROFILE

        non_eligible = (
            "rooms", "messages", "send", "search", "unreplied",
            "me", "members", "contacts", "task", "mytasks",
            "tasks", "mentions", "stats", "files", "delete",
        )
        for key in non_eligible:
            assert EXECUTION_PROFILE[key]["background_eligible"] is False, key
