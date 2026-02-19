"""Unit tests for messenger read ACK feature."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

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


def _place_message(
    inbox_dir: Path,
    from_person: str,
    content: str,
    msg_type: str = "message",
) -> None:
    """Write a Message JSON file directly into an inbox directory."""
    msg = Message(
        from_person=from_person,
        to_person="alice",
        content=content,
        type=msg_type,
    )
    if not msg.thread_id:
        msg.thread_id = msg.id
    (inbox_dir / f"{msg.id}.json").write_text(
        msg.model_dump_json(indent=2), encoding="utf-8"
    )


# ── receive_and_archive ACK ─────────────────────────────


class TestReceiveAndArchiveACK:
    """Tests for read-ACK behaviour in receive_and_archive()."""

    def test_sends_ack_to_sender(self, shared_dir: Path, messenger: Messenger) -> None:
        """When alice receives a message from bob, an ACK is sent to bob's inbox."""
        alice_inbox = shared_dir / "inbox" / "alice"
        _place_message(alice_inbox, "bob", "Hello Alice!")

        messenger.receive_and_archive()

        bob_inbox = shared_dir / "inbox" / "bob"
        ack_files = list(bob_inbox.glob("*.json"))
        assert len(ack_files) == 1
        data = json.loads(ack_files[0].read_text(encoding="utf-8"))
        assert data["type"] == "ack"
        assert data["from_person"] == "alice"
        assert data["to_person"] == "bob"

    def test_no_ack_for_ack_messages(
        self, shared_dir: Path, messenger: Messenger
    ) -> None:
        """Receiving an ack-type message should NOT generate another ACK (loop prevention)."""
        alice_inbox = shared_dir / "inbox" / "alice"
        _place_message(alice_inbox, "bob", "[既読通知] 1件", msg_type="ack")

        messenger.receive_and_archive()

        bob_inbox = shared_dir / "inbox" / "bob"
        if bob_inbox.exists():
            ack_files = list(bob_inbox.glob("*.json"))
            assert len(ack_files) == 0, "ACK messages must not generate further ACKs"
        # bob_inbox may not even be created — that is also correct

    def test_ack_groups_by_sender(
        self, shared_dir: Path, messenger: Messenger
    ) -> None:
        """Multiple messages from the same sender produce exactly one ACK."""
        alice_inbox = shared_dir / "inbox" / "alice"
        _place_message(alice_inbox, "bob", "msg-1")
        _place_message(alice_inbox, "bob", "msg-2")
        _place_message(alice_inbox, "bob", "msg-3")

        messenger.receive_and_archive()

        bob_inbox = shared_dir / "inbox" / "bob"
        ack_files = list(bob_inbox.glob("*.json"))
        assert len(ack_files) == 1

        data = json.loads(ack_files[0].read_text(encoding="utf-8"))
        assert data["type"] == "ack"
        assert "3件" in data["content"]

    def test_ack_multiple_senders(
        self, shared_dir: Path, messenger: Messenger
    ) -> None:
        """Messages from different senders generate separate ACKs."""
        alice_inbox = shared_dir / "inbox" / "alice"
        _place_message(alice_inbox, "bob", "from bob")
        _place_message(alice_inbox, "charlie", "from charlie")

        messenger.receive_and_archive()

        bob_inbox = shared_dir / "inbox" / "bob"
        charlie_inbox = shared_dir / "inbox" / "charlie"

        bob_acks = list(bob_inbox.glob("*.json"))
        charlie_acks = list(charlie_inbox.glob("*.json"))

        assert len(bob_acks) == 1
        assert len(charlie_acks) == 1

        bob_data = json.loads(bob_acks[0].read_text(encoding="utf-8"))
        charlie_data = json.loads(charlie_acks[0].read_text(encoding="utf-8"))
        assert bob_data["type"] == "ack"
        assert charlie_data["type"] == "ack"

    def test_ack_content_includes_summary(
        self, shared_dir: Path, messenger: Messenger
    ) -> None:
        """ACK message contains '[既読通知]' and a truncated summary of the received message."""
        alice_inbox = shared_dir / "inbox" / "alice"
        _place_message(alice_inbox, "bob", "Important project update")

        messenger.receive_and_archive()

        bob_inbox = shared_dir / "inbox" / "bob"
        ack_files = list(bob_inbox.glob("*.json"))
        assert len(ack_files) == 1

        data = json.loads(ack_files[0].read_text(encoding="utf-8"))
        assert "[既読通知]" in data["content"]
        assert "Important project update" in data["content"]

    def test_messages_still_archived(
        self, shared_dir: Path, messenger: Messenger
    ) -> None:
        """After ACK is sent, original messages are moved to processed/."""
        alice_inbox = shared_dir / "inbox" / "alice"
        _place_message(alice_inbox, "bob", "archive me")

        messenger.receive_and_archive()

        # Inbox should be empty (no unread .json files at top level)
        assert not messenger.has_unread()

        # Processed dir should contain the original message
        processed = alice_inbox / "processed"
        assert processed.is_dir()
        assert len(list(processed.glob("*.json"))) >= 1
