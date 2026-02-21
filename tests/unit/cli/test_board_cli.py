# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for cli/commands/board.py — Board CLI commands.

Covers:
- cmd_board_read: reads channel messages via Messenger.read_channel
- cmd_board_post: posts to channel via Messenger.post_channel + fanout
- cmd_board_dm_history: reads DM history via Messenger.read_dm_history
- _fanout_board_mentions: @name and @all mention fanout logic
- Board wrapper script template exists and is executable
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── cmd_board_read ───────────────────────────────────────


class TestCmdBoardRead:
    """Tests for cmd_board_read — reading shared channel messages."""

    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_read_prints_json(
        self, mock_ensure, mock_shared, mock_messenger_cls, capsys,
    ):
        """cmd_board_read prints JSON when messages exist."""
        from cli.commands.board import cmd_board_read

        mock_messenger = MagicMock()
        mock_messenger.read_channel.return_value = [
            {"ts": "2026-02-19T10:00:00", "from": "alice", "text": "hello"},
            {"ts": "2026-02-19T10:01:00", "from": "bob", "text": "hi"},
        ]
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(channel="general", limit=20, human_only=False)
        cmd_board_read(args)

        mock_messenger.read_channel.assert_called_once_with(
            "general", limit=20, human_only=False,
        )

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert len(parsed) == 2
        assert parsed[0]["from"] == "alice"

    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_read_no_messages(
        self, mock_ensure, mock_shared, mock_messenger_cls, capsys,
    ):
        """cmd_board_read prints 'No messages' when channel is empty."""
        from cli.commands.board import cmd_board_read

        mock_messenger = MagicMock()
        mock_messenger.read_channel.return_value = []
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(channel="ops", limit=10, human_only=False)
        cmd_board_read(args)

        captured = capsys.readouterr()
        assert "No messages in #ops" in captured.out

    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_read_passes_human_only(
        self, mock_ensure, mock_shared, mock_messenger_cls, capsys,
    ):
        """cmd_board_read forwards human_only flag to Messenger."""
        from cli.commands.board import cmd_board_read

        mock_messenger = MagicMock()
        mock_messenger.read_channel.return_value = []
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(channel="general", limit=5, human_only=True)
        cmd_board_read(args)

        mock_messenger.read_channel.assert_called_once_with(
            "general", limit=5, human_only=True,
        )

    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_read_messenger_constructed_with_cli(
        self, mock_ensure, mock_shared, mock_messenger_cls,
    ):
        """cmd_board_read constructs Messenger with 'cli' as anima_name."""
        from cli.commands.board import cmd_board_read

        mock_messenger = MagicMock()
        mock_messenger.read_channel.return_value = []
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(channel="general", limit=20, human_only=False)
        cmd_board_read(args)

        mock_messenger_cls.assert_called_once_with(Path("/tmp/shared"), "cli")


# ── cmd_board_post ───────────────────────────────────────


class TestCmdBoardPost:
    """Tests for cmd_board_post — posting to shared channels."""

    @patch("cli.commands.board._notify_server_board_posted")
    @patch("cli.commands.board._fanout_board_mentions")
    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_post_success(
        self, mock_ensure, mock_shared, mock_messenger_cls,
        mock_fanout, mock_notify, capsys,
    ):
        """cmd_board_post calls post_channel and prints confirmation."""
        from cli.commands.board import cmd_board_post

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(
            from_anima="alice", channel="general", text="Hello everyone!",
        )
        cmd_board_post(args)

        mock_messenger.post_channel.assert_called_once_with(
            "general", "Hello everyone!",
        )
        captured = capsys.readouterr()
        assert "Posted to #general" in captured.out

    @patch("cli.commands.board._notify_server_board_posted")
    @patch("cli.commands.board._fanout_board_mentions")
    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_post_triggers_fanout(
        self, mock_ensure, mock_shared, mock_messenger_cls,
        mock_fanout, mock_notify,
    ):
        """cmd_board_post calls _fanout_board_mentions with correct args."""
        from cli.commands.board import cmd_board_post

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(
            from_anima="alice", channel="dev", text="Hey @bob check this",
        )
        cmd_board_post(args)

        mock_fanout.assert_called_once_with(
            mock_messenger, "alice", "dev", "Hey @bob check this",
        )

    @patch("cli.commands.board._notify_server_board_posted")
    @patch("cli.commands.board._fanout_board_mentions")
    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_post_triggers_server_notification(
        self, mock_ensure, mock_shared, mock_messenger_cls,
        mock_fanout, mock_notify,
    ):
        """cmd_board_post notifies the running server."""
        from cli.commands.board import cmd_board_post

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(
            from_anima="alice", channel="general", text="Hello!",
        )
        cmd_board_post(args)

        mock_notify.assert_called_once_with("alice", "general", "Hello!")

    @patch("cli.commands.board._notify_server_board_posted")
    @patch("cli.commands.board._fanout_board_mentions")
    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_post_messenger_constructed_with_from_anima(
        self, mock_ensure, mock_shared, mock_messenger_cls,
        mock_fanout, mock_notify,
    ):
        """cmd_board_post constructs Messenger with from_anima name."""
        from cli.commands.board import cmd_board_post

        mock_messenger = MagicMock()
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(
            from_anima="sakura", channel="general", text="test",
        )
        cmd_board_post(args)

        mock_messenger_cls.assert_called_once_with(Path("/tmp/shared"), "sakura")


# ── cmd_board_dm_history ─────────────────────────────────


class TestCmdBoardDmHistory:
    """Tests for cmd_board_dm_history — reading DM history."""

    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_dm_history_prints_json(
        self, mock_ensure, mock_shared, mock_messenger_cls, capsys,
    ):
        """cmd_board_dm_history prints JSON when DM history exists."""
        from cli.commands.board import cmd_board_dm_history

        mock_messenger = MagicMock()
        mock_messenger.read_dm_history.return_value = [
            {"ts": "2026-02-19T10:00:00", "from": "alice", "text": "hi bob"},
            {"ts": "2026-02-19T10:01:00", "from": "bob", "text": "hello alice"},
        ]
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(from_anima="alice", peer="bob", limit=20)
        cmd_board_dm_history(args)

        mock_messenger.read_dm_history.assert_called_once_with("bob", limit=20)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert len(parsed) == 2
        assert parsed[1]["from"] == "bob"

    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_dm_history_no_messages(
        self, mock_ensure, mock_shared, mock_messenger_cls, capsys,
    ):
        """cmd_board_dm_history prints 'No DM history' when empty."""
        from cli.commands.board import cmd_board_dm_history

        mock_messenger = MagicMock()
        mock_messenger.read_dm_history.return_value = []
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(from_anima="alice", peer="bob", limit=10)
        cmd_board_dm_history(args)

        captured = capsys.readouterr()
        assert "No DM history with bob" in captured.out

    @patch("core.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.init.ensure_runtime_dir")
    def test_dm_history_respects_limit(
        self, mock_ensure, mock_shared, mock_messenger_cls,
    ):
        """cmd_board_dm_history passes limit to Messenger.read_dm_history."""
        from cli.commands.board import cmd_board_dm_history

        mock_messenger = MagicMock()
        mock_messenger.read_dm_history.return_value = []
        mock_messenger_cls.return_value = mock_messenger

        args = argparse.Namespace(from_anima="alice", peer="bob", limit=50)
        cmd_board_dm_history(args)

        mock_messenger.read_dm_history.assert_called_once_with("bob", limit=50)


# ── _fanout_board_mentions ───────────────────────────────


class TestFanoutBoardMentions:
    """Tests for _fanout_board_mentions — @mention fanout logic."""

    def test_at_name_sends_dm_to_running_anima(self, tmp_path):
        """@sakura sends a DM to sakura if she has a running socket."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "sakura.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "alice", "general", "Hey @sakura, check this",
            )

        # Should send exactly one board_mention DM
        assert mock_messenger.send.call_count == 1
        call_kwargs = mock_messenger.send.call_args.kwargs
        assert call_kwargs["to"] == "sakura"
        assert call_kwargs["msg_type"] == "board_mention"
        assert "sakura" in call_kwargs["content"] or "@sakura" in call_kwargs["content"]

    def test_at_all_sends_to_all_running_except_sender(self, tmp_path):
        """@all sends DMs to all running animas except the posting anima."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()
        (sockets_dir / "charlie.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "alice", "general", "Hello @all!",
            )

        # alice is excluded (sender); bob and charlie should receive DMs
        assert mock_messenger.send.call_count == 2
        targets = {
            c.kwargs["to"] for c in mock_messenger.send.call_args_list
        }
        assert targets == {"bob", "charlie"}
        assert "alice" not in targets

    def test_no_mentions_does_nothing(self, tmp_path):
        """Text without @mentions should not trigger any fanout."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "alice", "general", "No mentions here",
            )

        mock_messenger.send.assert_not_called()

    def test_mentioned_anima_not_running(self, tmp_path):
        """@bob when bob has no socket file should not send any DM."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        # Only alice is running, not bob
        (sockets_dir / "alice.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "charlie", "dev", "Hey @bob, are you there?",
            )

        mock_messenger.send.assert_not_called()

    def test_at_all_excludes_sender_socket(self, tmp_path):
        """@all must not send a DM to the posting anima even if its socket exists."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "alice", "general", "@all standup time",
            )

        targets = {
            c.kwargs["to"] for c in mock_messenger.send.call_args_list
        }
        assert "alice" not in targets
        assert "bob" in targets

    def test_no_sockets_dir_does_nothing(self, tmp_path):
        """If sockets directory does not exist, no fanout should occur."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        # Do NOT create sockets_dir
        mock_messenger = MagicMock(spec=Messenger)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "alice", "general", "@all hello",
            )

        mock_messenger.send.assert_not_called()

    def test_fanout_dm_content_format(self, tmp_path):
        """Fanout DM content includes board_reply tag, original text, and reply instructions."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "bob.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)

        original_text = "Hey @bob, please review the PR"
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "alice", "dev", original_text,
            )

        assert mock_messenger.send.call_count == 1
        content = mock_messenger.send.call_args.kwargs["content"]

        # Check board_reply tag with channel and from fields
        assert "[board_reply:channel=dev,from=alice]" in content
        # Check original text is included
        assert original_text in content
        # Check reply instructions with post_channel guidance
        assert "post_channel" in content
        assert 'channel="dev"' in content

    def test_fanout_multiple_named_mentions(self, tmp_path):
        """@bob @charlie sends DMs to both but not to unmentioned alice."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()
        (sockets_dir / "charlie.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            _fanout_board_mentions(
                mock_messenger, "dave", "project",
                "@bob @charlie please review",
            )

        targets = {
            c.kwargs["to"] for c in mock_messenger.send.call_args_list
        }
        assert targets == {"bob", "charlie"}
        assert "alice" not in targets

    def test_fanout_send_failure_is_silent(self, tmp_path):
        """If messenger.send raises, _fanout_board_mentions should not propagate."""
        from core.messenger import Messenger
        from cli.commands.board import _fanout_board_mentions

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "bob.sock").touch()

        mock_messenger = MagicMock(spec=Messenger)
        mock_messenger.send.side_effect = RuntimeError("IPC failure")

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            # Should not raise
            _fanout_board_mentions(
                mock_messenger, "alice", "general", "Hey @bob",
            )

        mock_messenger.send.assert_called_once()


