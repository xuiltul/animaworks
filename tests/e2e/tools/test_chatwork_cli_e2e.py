"""E2E tests for chatwork CLI subcommands — no real API calls."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools.chatwork import cli_main, ChatworkClient, MessageCache


@pytest.fixture(autouse=True)
def _mock_chatwork_env(monkeypatch, tmp_path):
    """Provide mocked Chatwork environment for all E2E tests."""
    monkeypatch.setenv("CHATWORK_API_TOKEN", "test-token")
    mock_requests = MagicMock()
    mock_session = MagicMock()
    mock_requests.Session.return_value = mock_session
    with patch.dict("core.tools.chatwork.__dict__", {"requests": mock_requests}):
        yield mock_session


@pytest.fixture
def mock_cache(tmp_path, monkeypatch):
    """Provide a real MessageCache in tmp_path.

    close() is replaced with a no-op so that CLI handlers'
    ``finally: cache.close()`` does not invalidate the connection
    before test assertions run.
    """
    cache = MessageCache(db_path=tmp_path / "test.db")
    _real_close = cache.close
    cache.close = lambda: None  # no-op during test
    monkeypatch.setattr(
        "core.tools.chatwork.MessageCache",
        lambda db_path=None: cache,
    )
    yield cache
    _real_close()


def _make_resp(json_data):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "json"
    resp.json.return_value = json_data
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


# ── sync subcommand ──────────────────────────────────────────────


class TestSyncCLI:
    def test_sync_command_populates_cache(
        self, _mock_chatwork_env, mock_cache, capsys
    ):
        session = _mock_chatwork_env
        rooms_resp = _make_resp([
            {"room_id": 10, "name": "Room10", "type": "group", "last_update_time": 100},
            {"room_id": 20, "name": "Room20", "type": "direct", "last_update_time": 50},
        ])
        msgs_resp = _make_resp([
            {"message_id": "m1", "send_time": 1000,
             "account": {"account_id": "a1", "name": "Alice"}, "body": "hello"},
        ])
        session.request.side_effect = [rooms_resp, msgs_resp, msgs_resp]

        with patch("core.tools.chatwork.time.sleep"):
            cli_main(["sync", "--limit", "2"])

        stats = mock_cache.get_stats()
        assert stats["rooms"] == 2
        assert stats["messages"] >= 1

        captured = capsys.readouterr()
        assert "Sync complete" in captured.out

    def test_sync_single_room(self, _mock_chatwork_env, mock_cache, capsys):
        session = _mock_chatwork_env
        # resolve_room_id("42") returns immediately (numeric) — no API call.
        # Only get_messages is called.
        msgs_resp = _make_resp([
            {"message_id": "m1", "send_time": 100,
             "account": {"account_id": "a1", "name": "A"}, "body": "test"},
        ])
        session.request.side_effect = [msgs_resp]

        cli_main(["sync", "42"])

        captured = capsys.readouterr()
        assert "1 messages" in captured.out


# ── unreplied --sync subcommand ──────────────────────────────────


class TestUnrepliedSyncCLI:
    def test_unreplied_with_sync_flag(
        self, _mock_chatwork_env, mock_cache, capsys
    ):
        session = _mock_chatwork_env

        # me() response
        me_resp = _make_resp({"account_id": 999, "name": "TestBot"})
        # rooms() for sync
        rooms_resp = _make_resp([
            {"room_id": 100, "name": "Office", "type": "group", "last_update_time": 1},
        ])
        # messages for sync
        msgs_resp = _make_resp([
            {"message_id": "m1", "send_time": 500,
             "account": {"account_id": "other", "name": "Other"},
             "body": "[To:999]TestBot\nPlease review"},
        ])

        session.request.side_effect = [me_resp, rooms_resp, msgs_resp]

        with patch("core.tools.chatwork.time.sleep"):
            cli_main(["unreplied", "--sync", "--sync-limit", "1"])

        captured = capsys.readouterr()
        assert "Unreplied: 1" in captured.out or "Please review" in captured.out


# ── rooms caching ────────────────────────────────────────────────


class TestRoomsCLI:
    def test_rooms_command_caches_metadata(
        self, _mock_chatwork_env, mock_cache, capsys
    ):
        session = _mock_chatwork_env
        session.request.return_value = _make_resp([
            {"room_id": 1, "name": "Alpha", "type": "direct", "last_update_time": 100},
            {"room_id": 2, "name": "Beta", "type": "group", "last_update_time": 50},
        ])

        cli_main(["rooms"])

        # Rooms should be cached in DB
        stats = mock_cache.get_stats()
        assert stats["rooms"] == 2

        # Check type is preserved
        row = mock_cache.conn.execute(
            "SELECT type FROM rooms WHERE room_id = '1'"
        ).fetchone()
        assert row["type"] == "direct"


# ── me subcommand ────────────────────────────────────────────────


class TestMeCLI:
    def test_me_command(self, _mock_chatwork_env, capsys):
        session = _mock_chatwork_env
        session.request.return_value = _make_resp({
            "account_id": 12345,
            "name": "Test User",
            "mail": "test@example.com",
            "organization_name": "Test Org",
        })

        cli_main(["me"])

        captured = capsys.readouterr()
        assert "12345" in captured.out
        assert "Test User" in captured.out


# ── members subcommand ───────────────────────────────────────────


class TestMembersCLI:
    def test_members_command(self, _mock_chatwork_env, capsys):
        session = _mock_chatwork_env
        session.request.return_value = _make_resp([
            {"account_id": 1, "name": "Alice", "role": "admin"},
            {"account_id": 2, "name": "Bob", "role": "member"},
        ])

        cli_main(["members", "123"])

        captured = capsys.readouterr()
        assert "Alice" in captured.out
        assert "Bob" in captured.out


# ── contacts subcommand ──────────────────────────────────────────


class TestContactsCLI:
    def test_contacts_command(self, _mock_chatwork_env, capsys):
        session = _mock_chatwork_env
        session.request.return_value = _make_resp([
            {"account_id": 10, "name": "Contact A"},
        ])

        cli_main(["contacts"])

        captured = capsys.readouterr()
        assert "Contact A" in captured.out


# ── mytasks subcommand ───────────────────────────────────────────


class TestMyTasksCLI:
    def test_mytasks_command(self, _mock_chatwork_env, capsys):
        session = _mock_chatwork_env
        session.request.return_value = _make_resp([
            {
                "task_id": 1,
                "body": "Review PR",
                "room": {"name": "Dev"},
                "limit_time": 0,
                "assigned_by_account": {"name": "Manager"},
            },
        ])

        cli_main(["mytasks"])

        captured = capsys.readouterr()
        assert "Review PR" in captured.out

    def test_mytasks_empty(self, _mock_chatwork_env, capsys):
        session = _mock_chatwork_env
        session.request.return_value = _make_resp([])

        cli_main(["mytasks"])

        captured = capsys.readouterr()
        assert "No open tasks" in captured.out


# ── stats subcommand ─────────────────────────────────────────────


class TestStatsCLI:
    def test_stats_command(self, _mock_chatwork_env, mock_cache, capsys):
        mock_cache.upsert_room({"room_id": 1, "name": "R"})
        mock_cache.upsert_messages("1", [
            {"message_id": "m1", "send_time": 1,
             "account": {"account_id": "a", "name": "A"}, "body": "x"},
        ])

        cli_main(["stats"])

        captured = capsys.readouterr()
        assert "Rooms: 1" in captured.out
        assert "Messages: 1" in captured.out


# ── mentions subcommand ──────────────────────────────────────────


class TestMentionsCLI:
    def test_mentions_json_output(self, _mock_chatwork_env, mock_cache, capsys):
        session = _mock_chatwork_env
        me_resp = _make_resp({"account_id": 999, "name": "Bot"})
        session.request.return_value = me_resp

        # Pre-populate cache
        mock_cache.upsert_room({"room_id": 100, "name": "Office", "type": "group"})
        mock_cache.upsert_messages("100", [
            {
                "message_id": "m1", "send_time": 1,
                "account": {"account_id": "other", "name": "Other"},
                "body": "[To:999]Bot\nPlease check",
            },
        ])

        cli_main(["mentions", "--json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["unreplied"] is True
