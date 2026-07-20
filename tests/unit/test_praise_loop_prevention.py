from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for praise-loop prevention changes.

Tests cover two code changes:
1. Messenger.send() — board_mention no longer exempt from depth limiter
2. ToolHandler._handle_post_channel() — _suppress_board_fanout flag
"""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from core.messenger import Messenger
    from core.tooling.handler import ToolHandler


# ── Test 1: board_mention is no longer exempt from depth check ────


@pytest.mark.unit
class TestBoardMentionDepthCheck:
    """Verify that board_mention goes through the depth limiter,
    while ack/error/system_alert remain exempt."""

    @pytest.fixture
    def shared_dir(self, tmp_path: Path) -> Path:
        """Create a shared directory with inbox structure."""
        d = tmp_path / "shared"
        d.mkdir()
        return d

    @pytest.fixture
    def animas_dir(self, tmp_path: Path) -> Path:
        """Create animas directory with a target Anima."""
        d = tmp_path / "animas"
        d.mkdir()
        (d / "target-anima").mkdir()
        return d

    @pytest.fixture
    def messenger(self, shared_dir: Path) -> Messenger:
        from core.messenger import Messenger

        return Messenger(shared_dir=shared_dir, anima_name="sender-anima")

    def test_board_mention_calls_depth_limiter(self, messenger: Messenger, animas_dir: Path) -> None:
        """board_mention should NOT be exempt — depth_limiter.check_depth must be called."""
        mock_limiter = MagicMock()
        mock_limiter.check_global_outbound.return_value = True  # pass before depth check
        mock_limiter.check_depth.return_value = True

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.get_depth_limiter", return_value=mock_limiter),
        ):
            messenger.send(to="target-anima", content="Great job!", msg_type="board_mention")

        mock_limiter.check_depth.assert_called_once_with(
            "sender-anima",
            "target-anima",
            animas_dir / "sender-anima",
        )

    def test_board_mention_blocked_when_depth_exceeded(self, messenger: Messenger, animas_dir: Path) -> None:
        """board_mention should be blocked when depth limiter returns False."""
        mock_limiter = MagicMock()
        mock_limiter.check_global_outbound.return_value = True  # pass before depth check
        mock_limiter.check_depth.return_value = False

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.get_depth_limiter", return_value=mock_limiter),
        ):
            result = messenger.send(
                to="target-anima",
                content="Great job!",
                msg_type="board_mention",
            )

        assert result.type == "error"
        assert result.from_person == "system"
        assert "ConversationDepthExceeded" in result.content

    def test_ack_exempt_from_depth_limiter(self, messenger: Messenger, animas_dir: Path) -> None:
        """msg_type='ack' should bypass the depth limiter entirely."""
        mock_limiter = MagicMock()

        # Ensure inbox exists for target
        inbox = messenger.shared_dir / "inbox" / "target-anima"
        inbox.mkdir(parents=True, exist_ok=True)

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.get_depth_limiter", return_value=mock_limiter),
        ):
            result = messenger.send(to="target-anima", content="ok", msg_type="ack")

        mock_limiter.check_depth.assert_not_called()
        assert result.type == "ack"

    def test_error_exempt_from_depth_limiter(self, messenger: Messenger, animas_dir: Path) -> None:
        """msg_type='error' should bypass the depth limiter entirely."""
        mock_limiter = MagicMock()

        inbox = messenger.shared_dir / "inbox" / "target-anima"
        inbox.mkdir(parents=True, exist_ok=True)

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.get_depth_limiter", return_value=mock_limiter),
        ):
            result = messenger.send(to="target-anima", content="err", msg_type="error")

        mock_limiter.check_depth.assert_not_called()
        assert result.type == "error"

    def test_system_alert_exempt_from_depth_limiter(self, messenger: Messenger, animas_dir: Path) -> None:
        """msg_type='system_alert' should bypass the depth limiter entirely."""
        mock_limiter = MagicMock()

        inbox = messenger.shared_dir / "inbox" / "target-anima"
        inbox.mkdir(parents=True, exist_ok=True)

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.get_depth_limiter", return_value=mock_limiter),
        ):
            result = messenger.send(
                to="target-anima",
                content="alert",
                msg_type="system_alert",
            )

        mock_limiter.check_depth.assert_not_called()
        assert result.type == "system_alert"

    def test_regular_message_calls_depth_limiter(self, messenger: Messenger, animas_dir: Path) -> None:
        """Regular 'message' type should also go through the depth limiter."""
        mock_limiter = MagicMock()
        mock_limiter.check_global_outbound.return_value = True  # pass before depth check
        mock_limiter.check_depth.return_value = True

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.cascade_limiter.get_depth_limiter", return_value=mock_limiter),
        ):
            messenger.send(to="target-anima", content="hello", msg_type="message")

        mock_limiter.check_depth.assert_called_once_with(
            "sender-anima",
            "target-anima",
            animas_dir / "sender-anima",
        )


# ── Test 2: _suppress_board_fanout flag in handler ────────────────


@pytest.mark.unit
class TestSuppressBoardFanout:
    """Verify that _suppress_board_fanout flag controls fanout in _handle_post_channel."""

    @pytest.fixture(autouse=True)
    def _bypass_acl(self):
        """Bypass channel ACL checks — these tests use MagicMock messenger."""
        with patch("core.messenger.is_channel_member", return_value=True):
            yield

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test-anima"
        d.mkdir(parents=True)
        (d / "permissions.md").write_text("", encoding="utf-8")
        return d

    @pytest.fixture
    def memory(self) -> MagicMock:
        m = MagicMock()
        m.read_permissions.return_value = ""
        m.search_memory_text.return_value = []
        return m

    @pytest.fixture
    def messenger(self, tmp_path: Path) -> MagicMock:
        m = MagicMock()
        m.anima_name = "test-anima"
        m.shared_dir = tmp_path / "shared"
        (m.shared_dir / "channels").mkdir(parents=True)
        msg = MagicMock()
        msg.id = "msg_001"
        msg.thread_id = "thread_001"
        m.send.return_value = msg
        return m

    @pytest.fixture
    def handler(
        self,
        anima_dir: Path,
        memory: MagicMock,
        messenger: MagicMock,
    ) -> ToolHandler:
        from core.tooling.handler import ToolHandler

        return ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=[],
        )

    def test_post_channel_calls_fanout_when_flag_not_set(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """Without _suppress_board_fanout, _fanout_board_mentions should be called."""
        with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
            handler._handle_post_channel({"channel": "general", "text": "@all hello"})

        mock_fanout.assert_called_once_with("general", "@all hello")

    def test_post_channel_calls_fanout_when_flag_explicitly_false(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """suppress_board_fanout=False should still call fanout normally."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(False)
        try:
            with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
                handler._handle_post_channel({"channel": "dev", "text": "@bob check"})

            mock_fanout.assert_called_once_with("dev", "@bob check")
        finally:
            suppress_board_fanout.reset(token)

    def test_post_channel_suppresses_fanout_when_flag_true(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """suppress_board_fanout=True should skip _fanout_board_mentions."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(True)
        try:
            with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
                handler._handle_post_channel({"channel": "general", "text": "@all thanks"})

            mock_fanout.assert_not_called()
        finally:
            suppress_board_fanout.reset(token)

    def test_post_channel_still_posts_when_fanout_suppressed(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """Even with fanout suppressed, the channel post itself should still succeed."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(True)
        try:
            result = handler._handle_post_channel({"channel": "general", "text": "hello"})

            messenger.post_channel.assert_called_once_with("general", "hello")
            assert "Posted to #general" in result
        finally:
            suppress_board_fanout.reset(token)

    def test_post_channel_logs_suppression(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """Suppressed fanout should be logged."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(True)
        try:
            with patch("core.tooling.handler_comms.logger") as mock_logger:
                handler._handle_post_channel({"channel": "ops", "text": "@all acknowledged"})

            # Look for the suppression log message
            log_messages = [call.args[0] if call.args else "" for call in mock_logger.info.call_args_list]
            assert any("Suppressed board fanout" in msg for msg in log_messages), (
                f"Expected suppression log message, got: {log_messages}"
            )
        finally:
            suppress_board_fanout.reset(token)

    def test_suppress_flag_defaults_to_false_via_contextvar(
        self,
        handler: ToolHandler,
    ) -> None:
        """suppress_board_fanout ContextVar should default to False."""
        from core.tooling.handler import suppress_board_fanout

        assert suppress_board_fanout.get() is False
