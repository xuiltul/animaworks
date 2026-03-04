"""Tests for _handle_manage_channel — Board channel ACL management."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.messenger import (
    ChannelMeta,
    Messenger,
    load_channel_meta,
    save_channel_meta,
)


# ── Helpers ──────────────────────────────────────────────


def _make_handler(tmp_path: Path, anima_name: str = "alice"):
    """Create a ToolHandler with minimal mocked dependencies and a real Messenger."""
    from core.tooling.handler import ToolHandler

    with patch("core.tooling.handler.ToolHandler.__init__", lambda self, **kw: None):
        handler = ToolHandler.__new__(ToolHandler)

    handler._anima_dir = tmp_path / "animas" / anima_name
    handler._anima_dir.mkdir(parents=True, exist_ok=True)
    handler._anima_name = anima_name
    handler._memory = MagicMock()
    handler._on_message_sent = None
    handler._on_schedule_changed = None
    handler._human_notifier = None
    handler._background_manager = None
    handler._pending_notifications = []
    handler._replied_to = {"chat": set(), "background": set()}
    handler._posted_channels = {"chat": set(), "background": set()}
    handler._subordinate_activity_dirs = []
    handler._subordinate_management_files = []
    handler._session_origin = ""
    handler._session_origin_chain = []

    import uuid
    handler._session_id = uuid.uuid4().hex[:12]

    from core.memory.activity import ActivityLogger
    handler._activity = MagicMock(spec=ActivityLogger)

    from core.tooling.dispatch import ExternalToolDispatcher
    handler._external = MagicMock(spec=ExternalToolDispatcher)

    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "channels").mkdir(exist_ok=True)
    handler._messenger = Messenger(shared_dir, anima_name)

    return handler


# ── create action ────────────────────────────────────────


class TestManageChannelCreate:
    def test_create_new_channel(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        result = handler._handle_manage_channel({
            "action": "create",
            "channel": "team-chat",
            "members": ["bob"],
            "description": "Team discussion",
        })
        assert "team-chat" in result
        shared_dir = tmp_path / "shared"
        channel_file = shared_dir / "channels" / "team-chat.jsonl"
        assert channel_file.exists()
        meta = load_channel_meta(shared_dir, "team-chat")
        assert meta is not None
        assert "alice" in meta.members
        assert "bob" in meta.members
        assert meta.created_by == "alice"
        assert meta.description == "Team discussion"

    def test_create_auto_includes_creator(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        handler._handle_manage_channel({
            "action": "create",
            "channel": "my-ch",
            "members": ["bob"],
        })
        meta = load_channel_meta(tmp_path / "shared", "my-ch")
        assert meta is not None
        assert meta.members[0] == "alice"

    def test_create_duplicate_channel_rejected(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        (tmp_path / "shared" / "channels" / "existing.jsonl").write_text("", encoding="utf-8")
        result = handler._handle_manage_channel({
            "action": "create",
            "channel": "existing",
        })
        assert "Error" in result or "既に存在" in result

    def test_create_with_no_members_includes_self(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        handler._handle_manage_channel({
            "action": "create",
            "channel": "solo",
        })
        meta = load_channel_meta(tmp_path / "shared", "solo")
        assert meta is not None
        assert "alice" in meta.members

    def test_create_invalid_name_rejected(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        result = handler._handle_manage_channel({
            "action": "create",
            "channel": "../exploit",
        })
        assert "error" in result.lower() or "Error" in result


# ── add_member action ────────────────────────────────────


class TestManageChannelAddMember:
    def test_add_member_to_existing(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice"]))
        result = handler._handle_manage_channel({
            "action": "add_member",
            "channel": "team",
            "members": ["bob", "charlie"],
        })
        assert "bob" in result
        meta = load_channel_meta(shared_dir, "team")
        assert meta is not None
        assert set(meta.members) == {"alice", "bob", "charlie"}

    def test_add_member_no_duplicates(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice", "bob"]))
        handler._handle_manage_channel({
            "action": "add_member",
            "channel": "team",
            "members": ["bob", "charlie"],
        })
        meta = load_channel_meta(shared_dir, "team")
        assert meta is not None
        assert meta.members.count("bob") == 1

    def test_add_member_to_open_channel_rejected(self, tmp_path: Path):
        """add_member on open/legacy channel should be rejected to prevent accidental restriction."""
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "general.jsonl").write_text("", encoding="utf-8")
        result = handler._handle_manage_channel({
            "action": "add_member",
            "channel": "general",
            "members": ["bob"],
        })
        assert "Error" in result or "オープン" in result
        # Meta should NOT have been created
        meta = load_channel_meta(shared_dir, "general")
        assert meta is None

    def test_add_member_non_member_denied(self, tmp_path: Path):
        """Non-member cannot add members to a restricted channel."""
        handler = _make_handler(tmp_path, anima_name="eve")
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "secret.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "secret", ChannelMeta(members=["alice", "bob"]))
        result = handler._handle_manage_channel({
            "action": "add_member",
            "channel": "secret",
            "members": ["charlie"],
        })
        assert "Error" in result or "メンバーではない" in result
        meta = load_channel_meta(shared_dir, "secret")
        assert meta is not None
        assert "charlie" not in meta.members

    def test_add_member_nonexistent_channel(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        result = handler._handle_manage_channel({
            "action": "add_member",
            "channel": "ghost",
            "members": ["bob"],
        })
        assert "見つかりません" in result or "not found" in result.lower()

    def test_add_member_missing_members_arg(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        result = handler._handle_manage_channel({
            "action": "add_member",
            "channel": "team",
        })
        assert "error" in result.lower() or "Error" in result


# ── remove_member action ─────────────────────────────────


class TestManageChannelRemoveMember:
    def test_remove_member(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice", "bob", "charlie"]))
        result = handler._handle_manage_channel({
            "action": "remove_member",
            "channel": "team",
            "members": ["charlie"],
        })
        assert "charlie" in result
        meta = load_channel_meta(shared_dir, "team")
        assert meta is not None
        assert "charlie" not in meta.members
        assert "alice" in meta.members

    def test_remove_member_from_open_channel(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "general.jsonl").write_text("", encoding="utf-8")
        result = handler._handle_manage_channel({
            "action": "remove_member",
            "channel": "general",
            "members": ["bob"],
        })
        assert "オープン" in result or "open" in result.lower()

    def test_remove_member_non_member_denied(self, tmp_path: Path):
        """Non-member cannot remove members from a restricted channel."""
        handler = _make_handler(tmp_path, anima_name="eve")
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "secret.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "secret", ChannelMeta(members=["alice", "bob"]))
        result = handler._handle_manage_channel({
            "action": "remove_member",
            "channel": "secret",
            "members": ["bob"],
        })
        assert "Error" in result or "メンバーではない" in result
        meta = load_channel_meta(shared_dir, "secret")
        assert meta is not None
        assert "bob" in meta.members  # bob should NOT have been removed


# ── info action ──────────────────────────────────────────


class TestManageChannelInfo:
    def test_info_restricted_channel(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "team", ChannelMeta(
            members=["alice", "bob"],
            created_by="alice",
            created_at="2026-03-03T00:00:00+09:00",
            description="Team channel",
        ))
        result = handler._handle_manage_channel({
            "action": "info",
            "channel": "team",
        })
        data = json.loads(result)
        assert data["members"] == ["alice", "bob"]
        assert data["created_by"] == "alice"
        assert data["description"] == "Team channel"

    def test_info_open_channel(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "general.jsonl").write_text("", encoding="utf-8")
        result = handler._handle_manage_channel({
            "action": "info",
            "channel": "general",
        })
        assert "オープン" in result or "open" in result.lower()

    def test_info_nonexistent_channel(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        result = handler._handle_manage_channel({
            "action": "info",
            "channel": "nope",
        })
        assert "見つかりません" in result or "not found" in result.lower()


# ── invalid action ───────────────────────────────────────


class TestManageChannelInvalidAction:
    def test_unknown_action(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        result = handler._handle_manage_channel({
            "action": "delete",
            "channel": "test",
        })
        assert "error" in result.lower() or "Unknown" in result

    def test_missing_action(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        result = handler._handle_manage_channel({
            "channel": "test",
        })
        assert "error" in result.lower() or "Error" in result

    def test_missing_channel(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        result = handler._handle_manage_channel({
            "action": "info",
        })
        assert "error" in result.lower() or "Error" in result


# ── ACL integration: post/read with handler ──────────────


class TestHandlerPostReadACL:
    def test_post_channel_acl_denied(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "secret.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "secret", ChannelMeta(members=["bob"]))
        result = handler._handle_post_channel({
            "channel": "secret",
            "text": "Should be denied",
        })
        assert "アクセス権" in result or "access" in result.lower()

    def test_read_channel_acl_denied(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "secret.jsonl").write_text(
            json.dumps({"ts": "2026-03-03T00:00:00", "from": "bob", "text": "hi", "source": "anima"}) + "\n",
            encoding="utf-8",
        )
        save_channel_meta(shared_dir, "secret", ChannelMeta(members=["bob"]))
        result = handler._handle_read_channel({
            "channel": "secret",
        })
        assert "アクセス権" in result or "access" in result.lower()

    def test_post_channel_acl_allowed(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        shared_dir = tmp_path / "shared"
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice", "bob"]))
        result = handler._handle_post_channel({
            "channel": "team",
            "text": "Hello team",
        })
        assert "Posted" in result or "team" in result
