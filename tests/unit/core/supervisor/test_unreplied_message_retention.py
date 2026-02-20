from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for unreplied message retention in heartbeat archive logic.

Covers Change 4: unreplied messages remain in inbox after heartbeat.

The archive logic only archives messages from senders who were replied to.
Messages from unreplied senders remain in the inbox for future processing.
System/non-sender messages are always archived.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

import pytest


class FakeMsg:
    """Minimal message mock."""
    def __init__(self, from_person: str, content: str = "test", type: str = "dm"):
        self.from_person = from_person
        self.content = content
        self.type = type
        self.id = f"msg_{from_person}"
        self.thread_id = ""


class FakeInboxItem:
    """Minimal InboxItem mock."""
    def __init__(self, msg: FakeMsg, path: str = ""):
        self.msg = msg
        self.path = Path(path) if path else Path("/tmp/fake")


class TestUnrepliedMessageRetention:
    """Test that heartbeat only archives messages from replied-to senders."""

    def _build_archive_logic(self, inbox_items, replied_to_set, senders):
        """Replicate the archive logic from anima.py for unit testing."""
        items_to_archive = [
            item for item in inbox_items
            if item.msg.from_person in replied_to_set
            or item.msg.from_person not in senders
        ]
        items_to_keep = [
            item for item in inbox_items
            if item not in items_to_archive
        ]
        return items_to_archive, items_to_keep

    def test_replied_messages_are_archived(self):
        """Messages from senders who were replied to should be archived."""
        msg_alice = FakeMsg("alice")
        msg_bob = FakeMsg("bob")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_bob)]
        senders = {"alice", "bob"}
        replied_to = {"alice", "bob"}

        to_archive, to_keep = self._build_archive_logic(items, replied_to, senders)
        assert len(to_archive) == 2
        assert len(to_keep) == 0

    def test_unreplied_messages_are_kept(self):
        """Messages from senders NOT replied to should remain in inbox."""
        msg_alice = FakeMsg("alice")
        msg_bob = FakeMsg("bob")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_bob)]
        senders = {"alice", "bob"}
        replied_to = {"alice"}  # Only replied to alice

        to_archive, to_keep = self._build_archive_logic(items, replied_to, senders)
        assert len(to_archive) == 1
        assert to_archive[0].msg.from_person == "alice"
        assert len(to_keep) == 1
        assert to_keep[0].msg.from_person == "bob"

    def test_no_replies_keeps_all_in_inbox(self):
        """When no replies were sent, all messages remain in inbox."""
        msg_alice = FakeMsg("alice")
        msg_bob = FakeMsg("bob")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_bob)]
        senders = {"alice", "bob"}
        replied_to = set()  # No replies

        to_archive, to_keep = self._build_archive_logic(items, replied_to, senders)
        assert len(to_archive) == 0
        assert len(to_keep) == 2

    def test_system_messages_always_archived(self):
        """Messages from senders NOT in the senders set (system messages)
        should always be archived."""
        msg_alice = FakeMsg("alice")
        msg_system = FakeMsg("system_bot")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_system)]
        senders = {"alice"}  # system_bot not in senders
        replied_to = set()  # No replies

        to_archive, to_keep = self._build_archive_logic(items, replied_to, senders)
        assert len(to_archive) == 1
        assert to_archive[0].msg.from_person == "system_bot"
        assert len(to_keep) == 1
        assert to_keep[0].msg.from_person == "alice"

    def test_mixed_scenario(self):
        """Mixed scenario: some replied, some unreplied, one system."""
        msg_alice = FakeMsg("alice")
        msg_bob = FakeMsg("bob")
        msg_charlie = FakeMsg("charlie")
        msg_system = FakeMsg("system")
        items = [
            FakeInboxItem(msg_alice),
            FakeInboxItem(msg_bob),
            FakeInboxItem(msg_charlie),
            FakeInboxItem(msg_system),
        ]
        senders = {"alice", "bob", "charlie"}
        replied_to = {"alice", "charlie"}

        to_archive, to_keep = self._build_archive_logic(items, replied_to, senders)
        archived_names = {i.msg.from_person for i in to_archive}
        kept_names = {i.msg.from_person for i in to_keep}
        assert archived_names == {"alice", "charlie", "system"}
        assert kept_names == {"bob"}
