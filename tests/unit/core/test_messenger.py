"""Unit tests for core/messenger.py — file-system messaging."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.messenger import Messenger
from core.schemas import Message


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir()
    return d


@pytest.fixture
def messenger(shared_dir: Path) -> Messenger:
    return Messenger(shared_dir, "alice")


# ── Initialization ────────────────────────────────────────


class TestMessengerInit:
    def test_creates_inbox_dir(self, shared_dir):
        m = Messenger(shared_dir, "bob")
        assert (shared_dir / "inbox" / "bob").is_dir()

    def test_person_name(self, messenger):
        assert messenger.person_name == "alice"

    def test_shared_dir(self, shared_dir, messenger):
        assert messenger.shared_dir == shared_dir


# ── send ──────────────────────────────────────────────────


class TestSend:
    def test_sends_message_to_recipient_inbox(self, shared_dir, messenger):
        msg = messenger.send("bob", "Hello Bob!")
        assert msg.from_person == "alice"
        assert msg.to_person == "bob"
        assert msg.content == "Hello Bob!"
        # File should exist in bob's inbox
        bob_inbox = shared_dir / "inbox" / "bob"
        assert bob_inbox.exists()
        files = list(bob_inbox.glob("*.json"))
        assert len(files) == 1

    def test_send_creates_target_dir(self, shared_dir, messenger):
        messenger.send("charlie", "Hi Charlie")
        assert (shared_dir / "inbox" / "charlie").is_dir()

    def test_auto_thread_id(self, shared_dir, messenger):
        msg = messenger.send("bob", "test")
        assert msg.thread_id == msg.id

    def test_explicit_thread_id(self, shared_dir, messenger):
        msg = messenger.send("bob", "test", thread_id="thread-1")
        assert msg.thread_id == "thread-1"

    def test_message_type(self, shared_dir, messenger):
        msg = messenger.send("bob", "test", msg_type="delegation")
        assert msg.type == "delegation"

    def test_reply_to(self, shared_dir, messenger):
        msg = messenger.send("bob", "test", reply_to="msg-original")
        assert msg.reply_to == "msg-original"

    def test_message_file_content(self, shared_dir, messenger):
        msg = messenger.send("bob", "Content check")
        bob_inbox = shared_dir / "inbox" / "bob"
        files = list(bob_inbox.glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["from_person"] == "alice"
        assert data["to_person"] == "bob"
        assert data["content"] == "Content check"


# ── reply ─────────────────────────────────────────────────


class TestReply:
    def test_reply_inherits_thread(self, shared_dir, messenger):
        original = Message(
            from_person="bob", to_person="alice",
            content="original", thread_id="thread-abc",
        )
        reply = messenger.reply(original, "Got it!")
        assert reply.to_person == "bob"
        assert reply.thread_id == "thread-abc"
        assert reply.reply_to == original.id

    def test_reply_uses_id_as_thread_when_no_thread(self, shared_dir, messenger):
        original = Message(
            from_person="bob", to_person="alice",
            content="original",
        )
        original.thread_id = ""  # no thread_id
        reply = messenger.reply(original, "Got it!")
        assert reply.thread_id == original.id


# ── receive ───────────────────────────────────────────────


class TestReceive:
    def test_receive_empty(self, messenger):
        messages = messenger.receive()
        assert messages == []

    def test_receive_messages(self, shared_dir, messenger):
        # Send a message to alice (from another person)
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "Hello Alice!")
        bob.send("alice", "Second message")

        messages = messenger.receive()
        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)

    def test_receive_sorted_by_filename(self, shared_dir, messenger):
        # Manually write messages with controlled names
        inbox = shared_dir / "inbox" / "alice"
        for i, content in enumerate(["first", "second", "third"]):
            msg = Message(from_person="bob", to_person="alice", content=content)
            (inbox / f"msg_{i:03d}.json").write_text(
                msg.model_dump_json(indent=2), encoding="utf-8"
            )
        messages = messenger.receive()
        assert len(messages) == 3

    def test_receive_skips_malformed(self, shared_dir, messenger):
        inbox = shared_dir / "inbox" / "alice"
        (inbox / "bad.json").write_text("not json", encoding="utf-8")
        messages = messenger.receive()
        assert len(messages) == 0


# ── receive_and_archive ───────────────────────────────────


class TestReceiveAndArchive:
    def test_archives_after_receive(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "Message 1")
        messages = messenger.receive_and_archive()
        assert len(messages) == 1
        # Inbox should be empty
        assert not messenger.has_unread()
        # Processed dir should have the file
        processed = shared_dir / "inbox" / "alice" / "processed"
        assert processed.is_dir()
        assert len(list(processed.glob("*.json"))) == 1

    def test_no_archive_when_empty(self, shared_dir, messenger):
        messages = messenger.receive_and_archive()
        assert messages == []


# ── archive_all ───────────────────────────────────────────


class TestArchiveAll:
    def test_moves_all_to_processed(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "msg1")
        bob.send("alice", "msg2")
        count = messenger.archive_all()
        assert count == 2
        assert messenger.unread_count() == 0

    def test_archive_empty_inbox(self, messenger):
        count = messenger.archive_all()
        assert count == 0

    def test_cleans_non_json_files(self, shared_dir, messenger):
        """A .txt file in inbox is moved to processed/ by archive_all()."""
        inbox = shared_dir / "inbox" / "alice"
        (inbox / "stray.txt").write_text("not json", encoding="utf-8")

        count = messenger.archive_all()
        # Return value counts only .json files
        assert count == 0
        # The .txt file should have been moved to processed/
        processed = inbox / "processed"
        assert (processed / "stray.txt").exists()
        assert not (inbox / "stray.txt").exists()

    def test_ignores_hidden_files(self, shared_dir, messenger):
        """Files starting with '.' are left untouched by archive_all()."""
        inbox = shared_dir / "inbox" / "alice"
        (inbox / ".tmp").write_text("hidden", encoding="utf-8")

        count = messenger.archive_all()
        assert count == 0
        # Hidden file should remain in inbox, not moved
        assert (inbox / ".tmp").exists()
        processed = inbox / "processed"
        assert not (processed / ".tmp").exists()

    def test_cleans_non_json_alongside_json(self, shared_dir, messenger):
        """.json and .txt coexist — both are correctly processed."""
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "real message")
        inbox = shared_dir / "inbox" / "alice"
        (inbox / "leftover.txt").write_text("stray", encoding="utf-8")

        count = messenger.archive_all()
        # Only the .json file is counted
        assert count == 1
        processed = inbox / "processed"
        # Both files should be in processed/
        assert (processed / "leftover.txt").exists()
        assert len(list(processed.glob("*.json"))) == 1
        # Inbox should have neither
        assert not (inbox / "leftover.txt").exists()
        assert len(list(inbox.glob("*.json"))) == 0


# ── archive_from ──────────────────────────────────────────


class TestArchiveFrom:
    def test_archives_only_from_sender(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        charlie = Messenger(shared_dir, "charlie")
        bob.send("alice", "from bob")
        charlie.send("alice", "from charlie")

        count = messenger.archive_from("bob")
        assert count == 1
        # charlie's message remains
        remaining = messenger.receive()
        assert len(remaining) == 1
        assert remaining[0].from_person == "charlie"

    def test_archive_from_nonexistent_sender(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "from bob")
        count = messenger.archive_from("unknown")
        assert count == 0
        assert messenger.unread_count() == 1


# ── has_unread / unread_count ─────────────────────────────


class TestUnread:
    def test_has_unread_false_when_empty(self, messenger):
        assert messenger.has_unread() is False

    def test_has_unread_true_when_messages(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "hello")
        assert messenger.has_unread() is True

    def test_unread_count(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "1")
        bob.send("alice", "2")
        bob.send("alice", "3")
        assert messenger.unread_count() == 3


# ── send_async ────────────────────────────────────────────


class TestSendAsync:
    async def test_falls_back_to_filesystem(self, shared_dir, messenger):
        msg = await messenger.send_async("bob", "async hello")
        assert msg.from_person == "alice"
        assert msg.to_person == "bob"
        # Should be in bob's inbox
        bob_inbox = shared_dir / "inbox" / "bob"
        assert len(list(bob_inbox.glob("*.json"))) == 1

    async def test_auto_thread_id(self, shared_dir, messenger):
        msg = await messenger.send_async("bob", "test")
        assert msg.thread_id == msg.id

    async def test_delegates_to_sync_send(self, shared_dir, messenger):
        msg = await messenger.send_async("bob", "async test")
        assert msg.from_person == "alice"
        bob_inbox = shared_dir / "inbox" / "bob"
        assert len(list(bob_inbox.glob("*.json"))) == 1
