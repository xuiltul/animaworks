from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for Messenger.send() depth limit integration.

Covers:
- send() blocks when depth limit exceeded for internal Anima
- send() returns error Message with type="error"
- send() allows ack/error/system_alert messages even when depth exceeded
- send() blocks board_mention when depth exceeded (praise-loop-prevention)
- send() skips check for external recipients
- send() allows when depth not exceeded
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.cascade_limiter import ConversationDepthLimiter
from core.messenger import Messenger


@pytest.fixture()
def shared_dir(tmp_path: Path) -> Path:
    """Create a shared directory structure."""
    d = tmp_path / "shared"
    d.mkdir()
    (d / "inbox" / "alice").mkdir(parents=True)
    (d / "inbox" / "bob").mkdir(parents=True)
    return d


@pytest.fixture()
def animas_dir(tmp_path: Path) -> Path:
    """Create an animas directory with known Anima directories."""
    d = tmp_path / "animas"
    d.mkdir()
    (d / "bob").mkdir()
    return d


class TestDepthLimitBlocking:
    """Test that Messenger.send blocks when depth limit is exceeded."""

    def test_blocked_returns_error_message(self, shared_dir, animas_dir):
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=1)
        limiter.check_and_record("alice", "bob")  # use up the one allowed

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            result = messenger.send("bob", "hello")

        assert result.type == "error"
        assert result.from_person == "system"
        assert result.to_person == "alice"
        assert "ConversationDepthExceeded" in result.content

    def test_blocked_does_not_write_file(self, shared_dir, animas_dir):
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=1)
        limiter.check_and_record("alice", "bob")

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            messenger.send("bob", "hello")

        # No file should be written to bob's inbox
        inbox = shared_dir / "inbox" / "bob"
        assert len(list(inbox.glob("*.json"))) == 0


class TestDepthLimitAllowed:
    """Test that send succeeds when depth is not exceeded."""

    def test_allowed_writes_file(self, shared_dir, animas_dir):
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=6)

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            result = messenger.send("bob", "hello")

        assert result.type == "message"
        assert result.from_person == "alice"
        inbox = shared_dir / "inbox" / "bob"
        assert len(list(inbox.glob("*.json"))) == 1


class TestDepthLimitBypass:
    """Test that ack/error/system_alert messages bypass the depth check."""

    def test_ack_bypasses_depth_check(self, shared_dir, animas_dir):
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=1)
        limiter.check_and_record("alice", "bob")  # exhaust

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            result = messenger.send("bob", "ok", msg_type="ack")

        assert result.type == "ack"
        assert result.from_person == "alice"

    def test_error_bypasses_depth_check(self, shared_dir, animas_dir):
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=1)
        limiter.check_and_record("alice", "bob")

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            result = messenger.send("bob", "fail", msg_type="error")

        assert result.type == "error"
        assert result.from_person == "alice"

    def test_system_alert_bypasses_depth_check(self, shared_dir, animas_dir):
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=1)
        limiter.check_and_record("alice", "bob")  # exhaust

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            result = messenger.send("bob", "alert content", msg_type="system_alert")

        assert result.type == "system_alert"
        assert result.from_person == "alice"

    def test_board_mention_does_not_bypass_depth_check(self, shared_dir, animas_dir):
        """board_mention no longer bypasses depth check (praise-loop-prevention)."""
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=1)
        limiter.check_and_record("alice", "bob")  # exhaust

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            result = messenger.send("bob", "mentioned you", msg_type="board_mention")

        assert result.type == "error"
        assert "ConversationDepthExceeded" in result.content


class TestExternalRecipientSkip:
    """Test that external (non-Anima) recipients skip the depth check."""

    def test_external_recipient_not_checked(self, shared_dir, animas_dir):
        messenger = Messenger(shared_dir, "alice")
        limiter = ConversationDepthLimiter(max_depth=1)
        limiter.check_and_record("alice", "external-user")  # exhaust

        # "external-user" directory does not exist in animas_dir
        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.depth_limiter", limiter),
        ):
            result = messenger.send("external-user", "hello")

        # Should succeed because external-user is not a known Anima
        assert result.type == "message"
        assert result.from_person == "alice"
