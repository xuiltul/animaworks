"""Unit tests for core/messenger.py — file-system messaging."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.messenger import InboxItem, Messenger
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

    def test_anima_name(self, messenger):
        assert messenger.anima_name == "alice"

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
        # Send a message to alice (from another anima)
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


# ── send() writes DM log (restored) ────────────────────


class TestSendCreatesDmLog:
    """send() writes to dm_logs/ again for legacy fallback support."""

    def test_send_creates_dm_log(self, shared_dir, messenger):
        messenger.send("bob", "Hello Bob!")
        log_dir = shared_dir / "dm_logs"
        assert log_dir.exists()

    def test_reply_creates_dm_log(self, shared_dir, messenger):
        original = Message(
            from_person="bob", to_person="alice",
            content="original", thread_id="thread-abc",
        )
        messenger.reply(original, "Got it!")
        log_dir = shared_dir / "dm_logs"
        assert log_dir.exists()

    async def test_send_async_creates_dm_log(self, shared_dir, messenger):
        await messenger.send_async("bob", "async msg")
        log_dir = shared_dir / "dm_logs"
        assert log_dir.exists()


# ── receive_with_paths ───────────────────────────────────


class TestReceiveWithPaths:
    def test_empty_inbox_returns_empty_list(self, messenger):
        items = messenger.receive_with_paths()
        assert items == []

    def test_returns_inbox_items_with_paths(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "Hello from bob 1")
        bob.send("alice", "Hello from bob 2")

        items = messenger.receive_with_paths()

        assert len(items) == 2
        for item in items:
            assert isinstance(item, InboxItem)
            assert isinstance(item.msg, Message)
            assert isinstance(item.path, Path)
            assert item.path.exists()
            assert item.msg.from_person == "bob"

    def test_items_sorted_by_filename(self, shared_dir, messenger):
        inbox = shared_dir / "inbox" / "alice"
        for i, content in enumerate(["first", "second", "third"]):
            msg = Message(from_person="bob", to_person="alice", content=content)
            (inbox / f"msg_{i:03d}.json").write_text(
                msg.model_dump_json(indent=2), encoding="utf-8",
            )

        items = messenger.receive_with_paths()

        assert len(items) == 3
        assert items[0].msg.content == "first"
        assert items[1].msg.content == "second"
        assert items[2].msg.content == "third"

    def test_skips_malformed_json(self, shared_dir, messenger):
        inbox = shared_dir / "inbox" / "alice"
        (inbox / "bad.json").write_text("not valid json", encoding="utf-8")

        items = messenger.receive_with_paths()

        assert items == []

    def test_does_not_archive_messages(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "persistent message")

        messenger.receive_with_paths()

        assert messenger.has_unread() is True


# ── archive_paths ────────────────────────────────────────


class TestArchivePaths:
    def test_archives_specified_items_only(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "msg1")
        bob.send("alice", "msg2")
        bob.send("alice", "msg3")

        items = messenger.receive_with_paths()
        assert len(items) == 3

        count = messenger.archive_paths(items[0:2])

        assert count == 2
        # Third message should still be in inbox
        remaining = messenger.receive_with_paths()
        assert len(remaining) == 1
        assert remaining[0].msg.content == items[2].msg.content
        # First two should be in processed/
        processed = shared_dir / "inbox" / "alice" / "processed"
        assert len(list(processed.glob("*.json"))) == 2

    def test_archives_nothing_when_empty(self, messenger):
        count = messenger.archive_paths([])
        assert count == 0

    def test_skips_already_archived(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "will be moved manually")

        items = messenger.receive_with_paths()
        assert len(items) == 1

        # Manually move the file to processed/ before calling archive_paths
        processed = shared_dir / "inbox" / "alice" / "processed"
        processed.mkdir(exist_ok=True)
        items[0].path.rename(processed / items[0].path.name)

        count = messenger.archive_paths(items)
        assert count == 0

    def test_new_messages_survive_archive(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "msg1")
        bob.send("alice", "msg2")

        items = messenger.receive_with_paths()
        assert len(items) == 2

        # Send a 3rd message AFTER receive_with_paths()
        bob.send("alice", "msg3")

        messenger.archive_paths(items)

        # The 3rd message should still be in inbox
        assert messenger.unread_count() == 1

    def test_creates_processed_dir(self, shared_dir, messenger):
        bob = Messenger(shared_dir, "bob")
        bob.send("alice", "trigger processed dir creation")

        items = messenger.receive_with_paths()
        messenger.archive_paths(items)

        processed = shared_dir / "inbox" / "alice" / "processed"
        assert processed.is_dir()


# ── Delivery verification ────────────────────────────────


class TestSendDeliveryVerification:
    """Verify that send() checks file was actually written."""

    def test_send_verifies_file_exists(self, shared_dir, messenger):
        """Normal send should succeed (file written and verified)."""
        msg = messenger.send("bob", "verified message")
        bob_inbox = shared_dir / "inbox" / "bob"
        assert (bob_inbox / f"{msg.id}.json").exists()

    def test_send_raises_on_write_failure(self, shared_dir, messenger):
        """If file write silently fails, OSError should be raised."""
        from unittest.mock import patch

        # Make filepath.exists() return False after write_text
        original_exists = Path.exists

        call_count = 0

        def fake_exists(self_path):
            nonlocal call_count
            # The delivery verification check is on the specific inbox file
            if "inbox" in str(self_path) and str(self_path).endswith(".json"):
                call_count += 1
                # Only fail the first .json exists check (the verification)
                if call_count == 1:
                    return False
            return original_exists(self_path)

        with patch.object(Path, "exists", fake_exists):
            with pytest.raises(OSError, match="Message delivery failed"):
                messenger.send("bob", "will fail verification")


# ── Activity logging warning ─────────────────────────────


class TestActivityLoggingWarning:
    """Verify that ActivityLogger failures produce warning logs."""

    def test_activity_logger_failure_logs_warning(self, shared_dir, messenger, caplog):
        """When ActivityLogger.log() raises, a warning should be logged."""
        import logging
        from unittest.mock import patch, MagicMock

        # Create anima directory so the activity log path check passes
        anima_dir = shared_dir.parent / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        mock_activity = MagicMock()
        mock_activity.log.side_effect = RuntimeError("disk full")

        with patch("core.memory.activity.ActivityLogger", return_value=mock_activity):
            with caplog.at_level(logging.WARNING, logger="animaworks.messenger"):
                messenger.send("bob", "activity will fail")

        assert any("Activity logging failed" in r.message for r in caplog.records)
        assert any("disk full" in r.message for r in caplog.records)

    def test_send_succeeds_despite_activity_failure(self, shared_dir, messenger):
        """send() should succeed even when ActivityLogger fails."""
        from unittest.mock import patch, MagicMock

        anima_dir = shared_dir.parent / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        mock_activity = MagicMock()
        mock_activity.log.side_effect = RuntimeError("boom")

        with patch("core.memory.activity.ActivityLogger", return_value=mock_activity):
            msg = messenger.send("bob", "should still work")

        assert msg.from_person == "alice"
        bob_inbox = shared_dir / "inbox" / "bob"
        assert len(list(bob_inbox.glob("*.json"))) == 1
