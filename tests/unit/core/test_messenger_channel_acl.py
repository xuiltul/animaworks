"""Unit tests for Board channel ACL in core/messenger.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.messenger import (
    ChannelMeta,
    Messenger,
    is_channel_member,
    load_channel_meta,
    save_channel_meta,
)


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir()
    (d / "channels").mkdir()
    return d


@pytest.fixture
def messenger(shared_dir: Path) -> Messenger:
    return Messenger(shared_dir, "alice")


# ── load_channel_meta ────────────────────────────────────


class TestLoadChannelMeta:
    def test_returns_none_when_no_meta_file(self, shared_dir: Path):
        result = load_channel_meta(shared_dir, "general")
        assert result is None

    def test_loads_meta_with_members(self, shared_dir: Path):
        meta_path = shared_dir / "channels" / "private.meta.json"
        meta_path.write_text(
            json.dumps({"members": ["alice", "bob"], "created_by": "alice"}),
            encoding="utf-8",
        )
        result = load_channel_meta(shared_dir, "private")
        assert result is not None
        assert result.members == ["alice", "bob"]
        assert result.created_by == "alice"

    def test_loads_meta_with_empty_members(self, shared_dir: Path):
        meta_path = shared_dir / "channels" / "open-ch.meta.json"
        meta_path.write_text(
            json.dumps({"members": []}),
            encoding="utf-8",
        )
        result = load_channel_meta(shared_dir, "open-ch")
        assert result is not None
        assert result.members == []

    def test_returns_none_on_malformed_json(self, shared_dir: Path):
        meta_path = shared_dir / "channels" / "broken.meta.json"
        meta_path.write_text("not valid json{{{", encoding="utf-8")
        result = load_channel_meta(shared_dir, "broken")
        assert result is None

    def test_description_field(self, shared_dir: Path):
        meta_path = shared_dir / "channels" / "team.meta.json"
        meta_path.write_text(
            json.dumps({
                "members": ["alice"],
                "description": "Team discussion",
            }),
            encoding="utf-8",
        )
        result = load_channel_meta(shared_dir, "team")
        assert result is not None
        assert result.description == "Team discussion"


# ── save_channel_meta ────────────────────────────────────


class TestSaveChannelMeta:
    def test_creates_meta_file(self, shared_dir: Path):
        meta = ChannelMeta(
            members=["alice", "bob"],
            created_by="alice",
            created_at="2026-03-03T00:00:00+09:00",
            description="test",
        )
        save_channel_meta(shared_dir, "new-ch", meta)
        meta_path = shared_dir / "channels" / "new-ch.meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["members"] == ["alice", "bob"]
        assert data["created_by"] == "alice"

    def test_overwrites_existing_meta(self, shared_dir: Path):
        meta1 = ChannelMeta(members=["alice"])
        save_channel_meta(shared_dir, "ch", meta1)
        meta2 = ChannelMeta(members=["alice", "bob"])
        save_channel_meta(shared_dir, "ch", meta2)
        loaded = load_channel_meta(shared_dir, "ch")
        assert loaded is not None
        assert loaded.members == ["alice", "bob"]


# ── is_channel_member ────────────────────────────────────


class TestIsChannelMember:
    def test_open_channel_no_meta(self, shared_dir: Path):
        assert is_channel_member(shared_dir, "general", "anyone") is True

    def test_open_channel_empty_members(self, shared_dir: Path):
        save_channel_meta(shared_dir, "open-ch", ChannelMeta(members=[]))
        assert is_channel_member(shared_dir, "open-ch", "anyone") is True

    def test_member_has_access(self, shared_dir: Path):
        save_channel_meta(shared_dir, "private", ChannelMeta(members=["alice", "bob"]))
        assert is_channel_member(shared_dir, "private", "alice") is True
        assert is_channel_member(shared_dir, "private", "bob") is True

    def test_non_member_denied(self, shared_dir: Path):
        save_channel_meta(shared_dir, "private", ChannelMeta(members=["alice"]))
        assert is_channel_member(shared_dir, "private", "charlie") is False

    def test_human_source_always_allowed(self, shared_dir: Path):
        save_channel_meta(shared_dir, "private", ChannelMeta(members=["alice"]))
        assert is_channel_member(shared_dir, "private", "charlie", source="human") is True

    def test_human_source_bypasses_even_strict_channel(self, shared_dir: Path):
        save_channel_meta(shared_dir, "secret", ChannelMeta(members=["alice"]))
        assert is_channel_member(shared_dir, "secret", "unknown", source="human") is True


# ── post_channel with ACL ────────────────────────────────


class TestPostChannelACL:
    def test_open_channel_allows_post(self, shared_dir: Path, messenger: Messenger):
        messenger.post_channel("general", "Hello!")
        channel_file = shared_dir / "channels" / "general.jsonl"
        assert channel_file.exists()

    def test_member_can_post(self, shared_dir: Path, messenger: Messenger):
        save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice", "bob"]))
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        messenger.post_channel("team", "Hello team!")
        lines = (shared_dir / "channels" / "team.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_non_member_blocked(self, shared_dir: Path):
        save_channel_meta(shared_dir, "private", ChannelMeta(members=["bob"]))
        (shared_dir / "channels" / "private.jsonl").write_text("", encoding="utf-8")
        alice = Messenger(shared_dir, "alice")
        alice.post_channel("private", "Should not appear")
        content = (shared_dir / "channels" / "private.jsonl").read_text(encoding="utf-8").strip()
        assert content == ""

    def test_human_source_bypasses_acl(self, shared_dir: Path):
        save_channel_meta(shared_dir, "restricted", ChannelMeta(members=["bob"]))
        (shared_dir / "channels" / "restricted.jsonl").write_text("", encoding="utf-8")
        alice = Messenger(shared_dir, "alice")
        alice.post_channel("restricted", "From human", source="human", from_name="human")
        lines = (shared_dir / "channels" / "restricted.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1


# ── read_channel with ACL ────────────────────────────────


class TestReadChannelACL:
    def test_open_channel_readable(self, shared_dir: Path, messenger: Messenger):
        messenger.post_channel("general", "msg1")
        result = messenger.read_channel("general")
        assert len(result) == 1

    def test_member_can_read(self, shared_dir: Path, messenger: Messenger):
        save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice"]))
        (shared_dir / "channels" / "team.jsonl").write_text("", encoding="utf-8")
        messenger.post_channel("team", "msg1")
        result = messenger.read_channel("team")
        assert len(result) == 1

    def test_non_member_gets_empty(self, shared_dir: Path):
        save_channel_meta(shared_dir, "private", ChannelMeta(members=["bob"]))
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(exist_ok=True)
        (channels_dir / "private.jsonl").write_text(
            json.dumps({"ts": "2026-03-03T00:00:00", "from": "bob", "text": "secret", "source": "anima"}) + "\n",
            encoding="utf-8",
        )
        alice = Messenger(shared_dir, "alice")
        result = alice.read_channel("private")
        assert result == []

    def test_human_source_reads_restricted(self, shared_dir: Path):
        save_channel_meta(shared_dir, "restricted", ChannelMeta(members=["bob"]))
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(exist_ok=True)
        (channels_dir / "restricted.jsonl").write_text(
            json.dumps({"ts": "2026-03-03T00:00:00", "from": "bob", "text": "data", "source": "anima"}) + "\n",
            encoding="utf-8",
        )
        alice = Messenger(shared_dir, "alice")
        result = alice.read_channel("restricted", source="human")
        assert len(result) == 1


# ── Backward compatibility: legacy channels stay open ──────


class TestBackwardCompatibility:
    def test_general_without_meta_is_open(self, shared_dir: Path):
        assert is_channel_member(shared_dir, "general", "any-anima") is True

    def test_ops_without_meta_is_open(self, shared_dir: Path):
        assert is_channel_member(shared_dir, "ops", "any-anima") is True

    def test_post_and_read_work_without_meta(self, shared_dir: Path, messenger: Messenger):
        messenger.post_channel("general", "Legacy works!")
        result = messenger.read_channel("general")
        assert len(result) == 1
        assert result[0]["text"] == "Legacy works!"


# ── read_channel_mentions with source param ─────────────


class TestReadChannelMentionsACL:
    def test_non_member_gets_no_mentions(self, shared_dir: Path):
        save_channel_meta(shared_dir, "private", ChannelMeta(members=["bob"]))
        (shared_dir / "channels" / "private.jsonl").write_text(
            json.dumps({"ts": "2026-03-03T00:00:00", "from": "bob", "text": "hello @alice", "source": "anima"}) + "\n",
            encoding="utf-8",
        )
        alice = Messenger(shared_dir, "alice")
        result = alice.read_channel_mentions("private", name="alice")
        assert result == []

    def test_human_source_reads_mentions(self, shared_dir: Path):
        save_channel_meta(shared_dir, "private", ChannelMeta(members=["bob"]))
        (shared_dir / "channels" / "private.jsonl").write_text(
            json.dumps({"ts": "2026-03-03T00:00:00", "from": "bob", "text": "hello @alice", "source": "anima"}) + "\n",
            encoding="utf-8",
        )
        human_messenger = Messenger(shared_dir, "human")
        result = human_messenger.read_channel_mentions("private", name="alice", source="human")
        assert len(result) == 1

    def test_member_reads_own_mentions(self, shared_dir: Path):
        save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice", "bob"]))
        (shared_dir / "channels" / "team.jsonl").write_text(
            json.dumps({"ts": "2026-03-03T00:00:00", "from": "bob", "text": "hi @alice", "source": "anima"}) + "\n",
            encoding="utf-8",
        )
        alice = Messenger(shared_dir, "alice")
        result = alice.read_channel_mentions("team", name="alice")
        assert len(result) == 1
