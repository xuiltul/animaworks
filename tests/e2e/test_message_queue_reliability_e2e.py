"""E2E tests for message queue reliability.

Verifies that the selective archiving mechanism guarantees no message
loss, even when new messages arrive during heartbeat processing.
"""
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
def alice(shared_dir: Path) -> Messenger:
    return Messenger(shared_dir, "alice")


@pytest.fixture
def bob(shared_dir: Path) -> Messenger:
    return Messenger(shared_dir, "bob")


@pytest.fixture
def charlie(shared_dir: Path) -> Messenger:
    return Messenger(shared_dir, "charlie")


# ── Selective archiving guarantees ──────────────────────


class TestSelectiveArchiving:
    """Verify that only processed messages are archived."""

    def test_new_messages_survive_selective_archive(self, alice, bob):
        """Messages arriving after receive_with_paths are not archived.

        Simulates: heartbeat reads inbox → agent processes → new message
        arrives during processing → archive_paths only archives original batch.
        """
        # Phase 1: Bob sends 2 messages to Alice
        bob.send("alice", "Message 1")
        bob.send("alice", "Message 2")

        # Phase 2: Alice reads inbox (snapshot)
        items = alice.receive_with_paths()
        assert len(items) == 2

        # Phase 3: During processing, Bob sends a 3rd message
        bob.send("alice", "Message 3 - arrived during processing")

        # Phase 4: Alice archives only the snapshot
        archived = alice.archive_paths(items)
        assert archived == 2

        # Phase 5: The 3rd message must survive
        assert alice.has_unread() is True
        assert alice.unread_count() == 1
        remaining = alice.receive()
        assert len(remaining) == 1
        assert "Message 3" in remaining[0].content

    def test_multi_sender_selective_archive(self, alice, bob, charlie):
        """Messages from multiple senders are correctly handled.

        Only the messages captured by receive_with_paths are archived;
        later messages from any sender survive.
        """
        # Initial messages from both senders
        bob.send("alice", "Bob's initial message")
        charlie.send("alice", "Charlie's initial message")

        # Snapshot
        items = alice.receive_with_paths()
        assert len(items) == 2

        # New messages arrive from both senders
        bob.send("alice", "Bob's late message")
        charlie.send("alice", "Charlie's late message")

        # Archive only the snapshot
        alice.archive_paths(items)

        # Both late messages must survive
        assert alice.unread_count() == 2
        remaining = alice.receive()
        contents = {m.content for m in remaining}
        assert "Bob's late message" in contents
        assert "Charlie's late message" in contents

    def test_empty_inbox_archive_is_noop(self, alice):
        """Archiving empty items list does nothing."""
        archived = alice.archive_paths([])
        assert archived == 0
        assert alice.unread_count() == 0

    def test_partial_archive(self, alice, bob):
        """Can archive a subset of received messages."""
        bob.send("alice", "Keep this")
        bob.send("alice", "Archive this")
        bob.send("alice", "Keep this too")

        items = alice.receive_with_paths()
        assert len(items) == 3

        # Only archive the middle message
        alice.archive_paths([items[1]])

        # Two messages remain
        assert alice.unread_count() == 2
        remaining = alice.receive()
        contents = [m.content for m in remaining]
        assert "Keep this" in contents
        assert "Keep this too" in contents
        assert "Archive this" not in contents


# ── InboxItem data integrity ────────────────────────────


class TestInboxItemIntegrity:
    """Verify InboxItem correctly pairs messages with file paths."""

    def test_paths_point_to_real_files(self, alice, bob):
        """Each InboxItem.path points to an existing file."""
        bob.send("alice", "Test message")
        items = alice.receive_with_paths()

        assert len(items) == 1
        assert items[0].path.exists()
        assert items[0].path.suffix == ".json"

    def test_message_content_matches_file(self, alice, bob):
        """InboxItem.msg content matches the file on disk."""
        bob.send("alice", "Verify content")
        items = alice.receive_with_paths()

        file_data = json.loads(items[0].path.read_text(encoding="utf-8"))
        assert file_data["content"] == items[0].msg.content
        assert file_data["from_person"] == items[0].msg.from_person

    def test_paths_are_in_inbox_dir(self, alice, bob, shared_dir):
        """All paths are within the anima's inbox directory."""
        bob.send("alice", "Location check")
        items = alice.receive_with_paths()

        inbox = shared_dir / "inbox" / "alice"
        for item in items:
            assert item.path.parent == inbox

    def test_archive_moves_to_processed(self, alice, bob, shared_dir):
        """After archive_paths, files move to processed/ subdirectory."""
        bob.send("alice", "To be archived")
        items = alice.receive_with_paths()
        original_name = items[0].path.name

        alice.archive_paths(items)

        processed = shared_dir / "inbox" / "alice" / "processed"
        assert processed.is_dir()
        assert (processed / original_name).exists()
        assert not items[0].path.exists()


# ── Backward compatibility ──────────────────────────────


class TestBackwardCompatibility:
    """Verify archive_all still works for non-heartbeat callers."""

    def test_archive_all_still_sweeps_everything(self, alice, bob):
        """archive_all continues to move all messages (for receive_and_archive)."""
        bob.send("alice", "Message A")
        bob.send("alice", "Message B")

        count = alice.archive_all()
        assert count == 2
        assert alice.unread_count() == 0

    def test_receive_and_archive_ack_flow(self, alice, bob, shared_dir):
        """receive_and_archive sends ACK and archives all messages."""
        bob.send("alice", "Greetings")

        messages = alice.receive_and_archive()
        assert len(messages) == 1
        assert messages[0].content == "Greetings"
        assert alice.unread_count() == 0

        # ACK should have been sent to bob
        bob_inbox = shared_dir / "inbox" / "bob"
        ack_files = list(bob_inbox.glob("*.json"))
        assert len(ack_files) >= 1
        ack_data = json.loads(ack_files[0].read_text(encoding="utf-8"))
        assert ack_data["type"] == "ack"
