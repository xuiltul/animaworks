"""Unit tests for shared channel messaging in core/messenger.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.messenger import Messenger


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir()
    return d


@pytest.fixture
def messenger(shared_dir: Path) -> Messenger:
    return Messenger(shared_dir, "alice")


# ── post_channel ─────────────────────────────────────────


class TestPostChannel:
    def test_creates_channel_file(self, shared_dir, messenger):
        messenger.post_channel("general", "Hello everyone!")
        channel_file = shared_dir / "channels" / "general.jsonl"
        assert channel_file.exists()

    def test_appends_jsonl_entry(self, shared_dir, messenger):
        messenger.post_channel("general", "First post")
        messenger.post_channel("general", "Second post")
        channel_file = shared_dir / "channels" / "general.jsonl"
        lines = channel_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["from"] == "alice"
        assert first["text"] == "First post"
        assert first["source"] == "anima"
        assert "ts" in first

    def test_custom_source(self, shared_dir, messenger):
        messenger.post_channel("general", "Human message", source="human")
        channel_file = shared_dir / "channels" / "general.jsonl"
        entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
        assert entry["source"] == "human"

    def test_creates_channels_dir(self, shared_dir, messenger):
        messenger.post_channel("ops", "test")
        assert (shared_dir / "channels").is_dir()

    def test_multiple_channels(self, shared_dir, messenger):
        messenger.post_channel("general", "gen msg")
        messenger.post_channel("ops", "ops msg")
        assert (shared_dir / "channels" / "general.jsonl").exists()
        assert (shared_dir / "channels" / "ops.jsonl").exists()


# ── read_channel ─────────────────────────────────────────


class TestReadChannel:
    def test_empty_channel(self, shared_dir, messenger):
        result = messenger.read_channel("general")
        assert result == []

    def test_nonexistent_channel(self, shared_dir, messenger):
        result = messenger.read_channel("nonexistent")
        assert result == []

    def test_reads_messages(self, shared_dir, messenger):
        messenger.post_channel("general", "msg1")
        messenger.post_channel("general", "msg2")
        result = messenger.read_channel("general")
        assert len(result) == 2
        assert result[0]["text"] == "msg1"
        assert result[1]["text"] == "msg2"

    def test_limit(self, shared_dir, messenger):
        for i in range(10):
            messenger.post_channel("general", f"msg{i}")
        result = messenger.read_channel("general", limit=3)
        assert len(result) == 3
        # Should return the LAST 3 in chronological order
        assert result[0]["text"] == "msg7"
        assert result[1]["text"] == "msg8"
        assert result[2]["text"] == "msg9"

    def test_human_only(self, shared_dir, messenger):
        messenger.post_channel("general", "anima msg", source="anima")
        messenger.post_channel("general", "human msg", source="human")
        messenger.post_channel("general", "another anima", source="anima")
        result = messenger.read_channel("general", human_only=True)
        assert len(result) == 1
        assert result[0]["text"] == "human msg"

    def test_chronological_order(self, shared_dir, messenger):
        messenger.post_channel("general", "first")
        messenger.post_channel("general", "second")
        messenger.post_channel("general", "third")
        result = messenger.read_channel("general")
        assert [m["text"] for m in result] == ["first", "second", "third"]


# ── read_channel_mentions ─────────────────────────────────


class TestReadChannelMentions:
    def test_finds_mentions(self, shared_dir, messenger):
        messenger.post_channel("general", "Hello @alice, please check this")
        messenger.post_channel("general", "No mention here")
        messenger.post_channel("general", "@alice another one")
        result = messenger.read_channel_mentions("general")
        assert len(result) == 2
        assert "@alice" in result[0]["text"]

    def test_no_mentions(self, shared_dir, messenger):
        messenger.post_channel("general", "Hello everyone")
        result = messenger.read_channel_mentions("general")
        assert result == []

    def test_custom_name(self, shared_dir, messenger):
        messenger.post_channel("general", "Hey @bob check this")
        result = messenger.read_channel_mentions("general", name="bob")
        assert len(result) == 1

    def test_limit(self, shared_dir, messenger):
        for i in range(5):
            messenger.post_channel("general", f"@alice msg{i}")
        result = messenger.read_channel_mentions("general", limit=2)
        assert len(result) == 2
        # Should return the last 2
        assert result[0]["text"] == "@alice msg3"
        assert result[1]["text"] == "@alice msg4"


# ── read_dm_history ─────────────────────────────────────


class TestReadDmHistory:
    """read_dm_history() reads existing dm_logs files.

    send() writes to dm_logs/ again for legacy fallback support.
    These tests verify that read_dm_history() can parse dm_logs.
    """

    def _write_dm_log(self, shared_dir: Path, pair: str, entries: list[dict]) -> None:
        """Helper: write JSONL entries to dm_logs/{pair}.jsonl."""
        dm_dir = shared_dir / "dm_logs"
        dm_dir.mkdir(parents=True, exist_ok=True)
        filepath = dm_dir / f"{pair}.jsonl"
        with filepath.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def test_no_history(self, shared_dir, messenger):
        result = messenger.read_dm_history("bob")
        assert result == []

    def test_reads_legacy_log(self, shared_dir, messenger):
        self._write_dm_log(shared_dir, "alice-bob", [
            {"ts": "2026-02-17T10:00:00", "from": "alice", "text": "Hello Bob!", "source": "anima"},
        ])
        result = messenger.read_dm_history("bob")
        assert len(result) == 1
        assert result[0]["from"] == "alice"
        assert result[0]["text"] == "Hello Bob!"

    def test_dm_log_path_sorted(self, shared_dir, messenger):
        path = messenger._get_dm_log_path("bob")
        assert path.name == "alice-bob.jsonl"
        # From bob's perspective, path should be the same
        bob = Messenger(shared_dir, "bob")
        path_bob = bob._get_dm_log_path("alice")
        assert path_bob.name == "alice-bob.jsonl"

    def test_bidirectional_history(self, shared_dir):
        alice = Messenger(shared_dir, "alice")
        bob = Messenger(shared_dir, "bob")
        # Manually write bidirectional entries (pair is always alice-bob)
        dm_dir = shared_dir / "dm_logs"
        dm_dir.mkdir(parents=True, exist_ok=True)
        filepath = dm_dir / "alice-bob.jsonl"
        entries = [
            {"ts": "2026-02-17T10:00:00", "from": "alice", "text": "Hi Bob", "source": "anima"},
            {"ts": "2026-02-17T10:00:01", "from": "bob", "text": "Hi Alice", "source": "anima"},
        ]
        with filepath.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Both should see the full history
        history_alice = alice.read_dm_history("bob")
        history_bob = bob.read_dm_history("alice")
        assert len(history_alice) == 2
        assert len(history_bob) == 2

    def test_limit(self, shared_dir, messenger):
        entries = [
            {"ts": f"2026-02-17T10:{i:02d}:00", "from": "alice", "text": f"msg{i}", "source": "anima"}
            for i in range(10)
        ]
        self._write_dm_log(shared_dir, "alice-bob", entries)
        result = messenger.read_dm_history("bob", limit=3)
        assert len(result) == 3

    def test_send_creates_dm_log(self, shared_dir, messenger):
        """send() writes to dm_logs/ again for legacy fallback support."""
        messenger.send("bob", "Hello!")
        assert (shared_dir / "dm_logs").exists()

    def test_send_no_message_log(self, shared_dir, messenger):
        """send() should NOT create message_log anymore."""
        messenger.send("bob", "Hello!")
        assert not (shared_dir / "message_log").exists()


# ── receive_external() channel mirroring ──────────────────


class TestReceiveExternalChannelMirroring:
    def test_human_at_all_mirrors_to_general(self, shared_dir, messenger):
        messenger.receive_external(
            "@all Server error is resolved",
            source="human",
            external_user_id="taka",
        )
        channel_file = shared_dir / "channels" / "general.jsonl"
        assert channel_file.exists()
        entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
        assert entry["source"] == "human"
        assert entry["from"] == "taka"  # Human identity, not anima name
        assert "@all" in entry["text"]

    def test_human_without_at_all_no_mirror(self, shared_dir, messenger):
        messenger.receive_external(
            "Just a normal message",
            source="human",
            external_user_id="taka",
        )
        channel_file = shared_dir / "channels" / "general.jsonl"
        assert not channel_file.exists()

    def test_anima_at_all_no_mirror(self, shared_dir, messenger):
        """Only human messages with @all should be mirrored."""
        messenger.receive_external(
            "@all announcement", source="anima",
        )
        channel_file = shared_dir / "channels" / "general.jsonl"
        assert not channel_file.exists()

    def test_receive_external_still_creates_inbox(self, shared_dir, messenger):
        messenger.receive_external(
            "@all resolve notice", source="human", external_user_id="taka",
        )
        # inbox should still get the message
        messages = messenger.receive()
        assert len(messages) == 1

    def test_receive_external_no_message_log(self, shared_dir, messenger):
        """receive_external() should NOT create message_log anymore."""
        messenger.receive_external(
            "Test", source="human", external_user_id="taka",
        )
        assert not (shared_dir / "message_log").exists()


# ── Name validation (path traversal prevention) ─────────


class TestNameValidation:
    def test_post_channel_rejects_path_traversal(self, messenger):
        with pytest.raises(ValueError, match="Invalid channel name"):
            messenger.post_channel("../../etc", "exploit")

    def test_read_channel_rejects_path_traversal(self, messenger):
        with pytest.raises(ValueError, match="Invalid channel name"):
            messenger.read_channel("../secrets")

    def test_read_dm_history_rejects_path_traversal(self, messenger):
        with pytest.raises(ValueError, match="Invalid peer name"):
            messenger.read_dm_history("../../root")

    def test_valid_channel_names_accepted(self, shared_dir, messenger):
        # These should not raise
        messenger.post_channel("general", "ok")
        messenger.post_channel("ops", "ok")
        messenger.post_channel("my-channel-1", "ok")

    def test_uppercase_channel_rejected(self, messenger):
        with pytest.raises(ValueError, match="Invalid channel name"):
            messenger.post_channel("General", "not allowed")
