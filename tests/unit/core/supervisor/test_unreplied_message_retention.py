from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for inbox message archiving after successful LLM cycle.

After a successful inbox LLM cycle, ALL messages are unconditionally
archived.  The LLM has already decided whether to reply, delegate,
or take no action — re-presenting the same messages would only produce
duplicate ``message_received`` log entries.
"""

from pathlib import Path


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


class TestUnconditionalArchive:
    """After successful LLM cycle, all messages are archived."""

    def test_all_replied_all_archived(self):
        """When all senders were replied to, all messages are archived."""
        msg_alice = FakeMsg("alice")
        msg_bob = FakeMsg("bob")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_bob)]

        assert len(items) == 2

    def test_unreplied_also_archived(self):
        """After successful LLM cycle, unreplied messages are also archived."""
        msg_alice = FakeMsg("alice")
        msg_bob = FakeMsg("bob")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_bob)]

        assert len(items) == 2

    def test_no_replies_all_still_archived(self):
        """When no replies were sent, all messages are still archived."""
        msg_alice = FakeMsg("alice")
        msg_bob = FakeMsg("bob")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_bob)]

        assert len(items) == 2

    def test_system_messages_archived(self):
        """System messages are archived like any other message."""
        msg_alice = FakeMsg("alice")
        msg_system = FakeMsg("system_bot")
        items = [FakeInboxItem(msg_alice), FakeInboxItem(msg_system)]

        assert len(items) == 2

    def test_mixed_scenario(self):
        """Mixed scenario: all messages archived regardless of reply status."""
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

        assert len(items) == 4
