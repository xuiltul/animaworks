"""Tests for board mention fanout in ToolHandler._handle_post_channel / _fanout_board_mentions."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    return m


@pytest.fixture
def messenger() -> MagicMock:
    m = MagicMock()
    m.anima_name = "test-anima"
    msg = MagicMock()
    msg.id = "msg_001"
    msg.thread_id = "thread_001"
    m.send.return_value = msg
    return m


@pytest.fixture
def handler_with_messenger(
    anima_dir: Path, memory: MagicMock, messenger: MagicMock,
) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
        tool_registry=[],
    )


@pytest.fixture
def sockets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run" / "sockets"
    d.mkdir(parents=True)
    return d


# ── Board mention fanout tests ───────────────────────────────


@pytest.mark.unit
class TestBoardMentionFanout:
    """Tests for _handle_post_channel() and _fanout_board_mentions()."""

    def test_fanout_at_all_sends_to_running_animas(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """@all mention posts DM to all running Animas except the poster."""
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel", {"channel": "general", "text": "Hello @all!"},
            )

        # messenger.send is called for post_channel internals + fanout DMs.
        # Fanout targets: alice and bob (test-anima excluded as poster).
        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
            or (len(c.args) == 0 and c.kwargs.get("msg_type") == "board_mention")
        ]
        targets = {c.kwargs["to"] for c in fanout_calls}
        assert targets == {"alice", "bob"}

    def test_fanout_at_name_sends_to_specific_anima(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """@bob mention sends DM only to bob."""
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel", {"channel": "dev", "text": "Hey @bob, check this"},
            )

        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
        ]
        targets = {c.kwargs["to"] for c in fanout_calls}
        assert targets == {"bob"}

    def test_fanout_no_mention_no_dm(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """Text without any @mention should not trigger any fanout DM."""
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel", {"channel": "general", "text": "No mentions here"},
            )

        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
        ]
        assert len(fanout_calls) == 0

    def test_fanout_excludes_self(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """@all should not send a DM to the posting Anima itself (test-anima)."""
        # Include the poster's own socket
        (sockets_dir / "test-anima.sock").touch()
        (sockets_dir / "alice.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel", {"channel": "general", "text": "@all standup time"},
            )

        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
        ]
        targets = {c.kwargs["to"] for c in fanout_calls}
        assert "test-anima" not in targets
        assert "alice" in targets

    def test_fanout_only_running_animas(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """@bob when bob has no socket file should not send any DM."""
        # Only alice is running; bob is not
        (sockets_dir / "alice.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel", {"channel": "dev", "text": "Hey @bob, are you there?"},
            )

        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
        ]
        assert len(fanout_calls) == 0

    def test_fanout_dm_content_format(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """Fanout DM content should contain board_reply tag, original text, and reply instructions."""
        (sockets_dir / "bob.sock").touch()

        original_text = "Hey @bob, please review the PR"
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel", {"channel": "dev", "text": original_text},
            )

        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
        ]
        assert len(fanout_calls) == 1
        content = fanout_calls[0].kwargs["content"]

        # Check board_reply tag with channel and from fields
        assert "[board_reply:channel=dev,from=test-anima]" in content
        # Check original text is included
        assert original_text in content
        # Check reply instructions with post_channel guidance
        assert "post_channel" in content
        assert 'channel="dev"' in content

    def test_fanout_dm_type_is_board_mention(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """Fanout DM must use msg_type='board_mention'."""
        (sockets_dir / "alice.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel", {"channel": "general", "text": "@all hello"},
            )

        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
        ]
        assert len(fanout_calls) >= 1
        for call in fanout_calls:
            assert call.kwargs["msg_type"] == "board_mention"

    def test_fanout_multiple_named_mentions(
        self,
        handler_with_messenger: ToolHandler,
        messenger: MagicMock,
        sockets_dir: Path,
        tmp_path: Path,
    ):
        """@bob @charlie should send DM to both bob and charlie."""
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()
        (sockets_dir / "charlie.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler_with_messenger.handle(
                "post_channel",
                {"channel": "project", "text": "@bob @charlie please review"},
            )

        fanout_calls = [
            c for c in messenger.send.call_args_list
            if c.kwargs.get("msg_type") == "board_mention"
        ]
        targets = {c.kwargs["to"] for c in fanout_calls}
        assert targets == {"bob", "charlie"}
        # alice should NOT receive a DM (not mentioned)
        assert "alice" not in targets
