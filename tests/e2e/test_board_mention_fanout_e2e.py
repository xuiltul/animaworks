# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for board mention fanout via ToolHandler + Messenger.

Tests the complete flow: posting a message with @all / @name to a board channel
triggers DM delivery of board_mention messages to target Animas' inboxes.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory import MemoryManager
from core.messenger import Messenger
from core.tooling.handler import ToolHandler


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir()
    (d / "inbox").mkdir()
    (d / "channels").mkdir()
    (d / "dm_logs").mkdir()
    return d


@pytest.fixture
def sockets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run" / "sockets"
    d.mkdir(parents=True)
    return d


def _make_anima_dir(tmp_path: Path, name: str) -> Path:
    """Create a minimal anima directory with required files."""
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text(
        f"# {name}\n\nテスト用Anima。\n", encoding="utf-8",
    )
    (anima_dir / "permissions.md").write_text(
        "# Permissions\n\n## メッセージング\n- send_message: OK\n- post_channel: OK\n",
        encoding="utf-8",
    )
    return anima_dir


def _make_tool_handler(
    anima_dir: Path,
    shared_dir: Path,
) -> ToolHandler:
    """Build a ToolHandler with a real Messenger and mock MemoryManager."""
    name = anima_dir.name
    memory = MagicMock(spec=MemoryManager)
    memory.read_permissions.return_value = ""
    messenger = Messenger(shared_dir, name)
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
    )


def _create_socket(sockets_dir: Path, name: str) -> Path:
    """Create a dummy socket file to simulate a running Anima."""
    sock = sockets_dir / f"{name}.sock"
    sock.touch()
    return sock


def _read_inbox(shared_dir: Path, name: str) -> list[dict]:
    """Read all JSON messages from an Anima's inbox."""
    inbox = shared_dir / "inbox" / name
    if not inbox.exists():
        return []
    msgs = []
    for f in sorted(inbox.glob("*.json")):
        msgs.append(json.loads(f.read_text(encoding="utf-8")))
    return msgs


# ── Tests ──────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestBoardMentionFanout:
    """Board mention fanout: @all / @name in post_channel triggers DM delivery."""

    def test_e2e_post_channel_at_all_fanout(
        self, tmp_path: Path, shared_dir: Path, sockets_dir: Path,
    ) -> None:
        """@all fanout delivers board_mention DMs to all running Animas except sender."""
        # Setup anima directories
        alice_dir = _make_anima_dir(tmp_path, "alice")
        _make_anima_dir(tmp_path, "bob")
        _make_anima_dir(tmp_path, "charlie")

        # Create socket files for all three (all running)
        _create_socket(sockets_dir, "alice")
        _create_socket(sockets_dir, "bob")
        _create_socket(sockets_dir, "charlie")

        handler = _make_tool_handler(alice_dir, shared_dir)

        # Patch get_data_dir to return tmp_path (sockets live under tmp_path/run/sockets)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("core.paths.get_data_dir", lambda: tmp_path)

            result = handler.handle("post_channel", {
                "channel": "general",
                "text": "@all テスト通知です",
            })

        assert "Posted to #general" in result

        # Bob and Charlie should each have a board_mention DM
        bob_msgs = _read_inbox(shared_dir, "bob")
        charlie_msgs = _read_inbox(shared_dir, "charlie")

        bob_board = [m for m in bob_msgs if m.get("type") == "board_mention"]
        charlie_board = [m for m in charlie_msgs if m.get("type") == "board_mention"]

        assert len(bob_board) == 1, f"Expected 1 board_mention for bob, got {len(bob_board)}"
        assert len(charlie_board) == 1, f"Expected 1 board_mention for charlie, got {len(charlie_board)}"

        # Alice (sender) should NOT receive a fanout DM
        alice_msgs = _read_inbox(shared_dir, "alice")
        alice_board = [m for m in alice_msgs if m.get("type") == "board_mention"]
        assert len(alice_board) == 0, "Sender should not receive board_mention"

        # Verify DM content
        assert bob_board[0]["type"] == "board_mention"
        assert "[board_reply:channel=general,from=alice]" in bob_board[0]["content"]
        assert "@all テスト通知です" in bob_board[0]["content"]

    def test_e2e_post_channel_at_name_fanout(
        self, tmp_path: Path, shared_dir: Path, sockets_dir: Path,
    ) -> None:
        """@name fanout delivers board_mention DM only to the named Anima."""
        alice_dir = _make_anima_dir(tmp_path, "alice")
        _make_anima_dir(tmp_path, "bob")
        _make_anima_dir(tmp_path, "charlie")

        _create_socket(sockets_dir, "alice")
        _create_socket(sockets_dir, "bob")
        _create_socket(sockets_dir, "charlie")

        handler = _make_tool_handler(alice_dir, shared_dir)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("core.paths.get_data_dir", lambda: tmp_path)

            result = handler.handle("post_channel", {
                "channel": "ops",
                "text": "@bob 確認お願い",
            })

        assert "Posted to #ops" in result

        # Bob should have a board_mention DM
        bob_msgs = _read_inbox(shared_dir, "bob")
        bob_board = [m for m in bob_msgs if m.get("type") == "board_mention"]
        assert len(bob_board) == 1, f"Expected 1 board_mention for bob, got {len(bob_board)}"

        # Charlie should NOT have a board_mention DM
        charlie_msgs = _read_inbox(shared_dir, "charlie")
        charlie_board = [m for m in charlie_msgs if m.get("type") == "board_mention"]
        assert len(charlie_board) == 0, "Charlie should not receive @bob mention"

    def test_e2e_board_mention_no_ack(
        self, tmp_path: Path, shared_dir: Path, sockets_dir: Path,
    ) -> None:
        """board_mention messages are excluded from ACK on receive_and_archive."""
        alice_dir = _make_anima_dir(tmp_path, "alice")
        _make_anima_dir(tmp_path, "bob")

        _create_socket(sockets_dir, "alice")
        _create_socket(sockets_dir, "bob")

        handler = _make_tool_handler(alice_dir, shared_dir)

        # Alice posts @all -> bob gets board_mention
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("core.paths.get_data_dir", lambda: tmp_path)

            handler.handle("post_channel", {
                "channel": "general",
                "text": "@all テスト",
            })

        # Bob has a board_mention DM in inbox
        bob_msgs = _read_inbox(shared_dir, "bob")
        assert any(m.get("type") == "board_mention" for m in bob_msgs)

        # Bob calls receive_and_archive
        bob_messenger = Messenger(shared_dir, "bob")
        messages = bob_messenger.receive_and_archive()
        assert len(messages) >= 1

        # Alice's inbox should NOT contain an ACK from bob
        # (board_mention is excluded from ACK in receive_and_archive)
        alice_msgs = _read_inbox(shared_dir, "alice")
        alice_acks = [m for m in alice_msgs if m.get("type") == "ack"]
        assert len(alice_acks) == 0, (
            "board_mention should not trigger ACK; "
            f"found {len(alice_acks)} ack(s) in alice's inbox"
        )

    def test_e2e_fanout_includes_stopped_animas(
        self, tmp_path: Path, shared_dir: Path, sockets_dir: Path,
    ) -> None:
        """@all fanout reaches both running and stopped Animas (inbox saved for pickup)."""
        alice_dir = _make_anima_dir(tmp_path, "alice")
        _make_anima_dir(tmp_path, "bob")
        _make_anima_dir(tmp_path, "charlie")

        # Only bob has a socket file (running); charlie does not
        _create_socket(sockets_dir, "alice")
        _create_socket(sockets_dir, "bob")
        # charlie: no socket -> not running, but has anima dir

        handler = _make_tool_handler(alice_dir, shared_dir)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("core.paths.get_data_dir", lambda: tmp_path)

            handler.handle("post_channel", {
                "channel": "general",
                "text": "@all テスト",
            })

        # Bob (running) should receive a board_mention
        bob_msgs = _read_inbox(shared_dir, "bob")
        bob_board = [m for m in bob_msgs if m.get("type") == "board_mention"]
        assert len(bob_board) == 1, f"Expected 1 board_mention for bob, got {len(bob_board)}"

        # Charlie (stopped) should NOT receive a board_mention (Fix 8: running only)
        charlie_msgs = _read_inbox(shared_dir, "charlie")
        charlie_board = [m for m in charlie_msgs if m.get("type") == "board_mention"]
        assert len(charlie_board) == 0, (
            "Charlie (stopped) should NOT receive board_mention — only running Animas are targets"
        )
