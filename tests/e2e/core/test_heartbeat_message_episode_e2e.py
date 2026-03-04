# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.messenger import Messenger
from core.memory.manager import MemoryManager
from core.tooling.handler import active_session_type


class TestInboxMessageEpisodeE2E:
    """Full integration tests for message episode recording during inbox processing.

    Since the 3-path execution separation, inbox processing is handled by
    process_inbox_message() (Path A), not run_heartbeat() (Path B).
    """

    async def test_message_content_persists_in_episodes(self, data_dir, make_anima):
        """Full integration: message sent → process_inbox → activity_log contains message content.

        Episodes are written by _process_inbox_messages; activity_log records message_received.
        """
        alice_dir = make_anima("alice")
        make_anima("mio")  # Sender must be in config.animas for inbox to accept
        shared_dir = data_dir / "shared"

        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "AWS監視タスクを追加しました。30分間隔で確認してください。")

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": trigger,
                        "action": "responded",
                        "summary": "Processed mio message",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.process_inbox_message()

        # Activity log records message_received (primary record of inbox processing)
        activity_dir = alice_dir / "activity_log"
        today_file = activity_dir / f"{date.today().isoformat()}.jsonl"
        assert today_file.exists()
        lines = [l for l in today_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 1
        entries = [json.loads(l) for l in lines]
        msg_entries = [e for e in entries if e.get("type") == "message_received"]
        assert len(msg_entries) >= 1
        content_str = " ".join(e.get("content", "") or "" for e in msg_entries)
        assert "AWS監視タスク" in content_str

    async def test_messages_archived_after_episode_recording(self, data_dir, make_anima):
        """Verify messages are archived after being recorded to activity_log."""
        alice_dir = make_anima("alice")
        make_anima("mio")
        shared_dir = data_dir / "shared"

        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "テストメッセージ: ログ確認をお願いします。")

        inbox_dir = shared_dir / "inbox" / "alice"
        assert len(list(inbox_dir.glob("*.json"))) == 1

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": trigger,
                        "action": "responded",
                        "summary": "Processed message",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            original_archive = dp.messenger.archive_paths
            archive_called = False

            def mock_archive(items):
                nonlocal archive_called
                archive_called = True
                return original_archive(items)

            dp.messenger.archive_paths = mock_archive

            await dp.process_inbox_message()

            # Activity log records message_received
            activity_file = alice_dir / "activity_log" / f"{date.today().isoformat()}.jsonl"
            assert activity_file.exists()
            lines = [l for l in activity_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            msg_entries = [e for e in (json.loads(l) for l in lines) if e.get("type") == "message_received"]
            assert len(msg_entries) >= 1
            content_str = " ".join(e.get("content", "") or "" for e in msg_entries)
            assert "ログ確認" in content_str

            assert archive_called

    async def test_ack_messages_not_in_episodes(self, data_dir, make_anima):
        """Verify ACK messages are filtered out and not recorded in activity_log."""
        alice_dir = make_anima("alice")
        make_anima("mio")
        shared_dir = data_dir / "shared"

        from core.schemas import Message

        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "重要なタスク: DB バックアップを実行してください。")
        ack_msg = Message(
            from_person="mio",
            to_person="alice",
            content="了解しました",
            type="ack",
        )
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        ack_file = inbox_dir / f"ack_{date.today().isoformat()}.json"
        ack_file.write_text(ack_msg.model_dump_json(), encoding="utf-8")

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": trigger,
                        "action": "responded",
                        "summary": "Processed messages",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.process_inbox_message()

        activity_file = alice_dir / "activity_log" / f"{date.today().isoformat()}.jsonl"
        assert activity_file.exists()
        lines = [l for l in activity_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        msg_entries = [e for e in (json.loads(l) for l in lines) if e.get("type") == "message_received"]
        content_str = " ".join(e.get("content", "") or "" for e in msg_entries)

        assert "DB バックアップ" in content_str
        assert "了解しました" not in content_str

    async def test_multiple_messages_recorded_in_order(self, data_dir, make_anima):
        """Verify multiple messages are all recorded to activity_log in order."""
        alice_dir = make_anima("alice")
        make_anima("mio")
        make_anima("bob")
        make_anima("charlie")
        shared_dir = data_dir / "shared"

        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "第一のタスク: ログ解析")

        bob_messenger = Messenger(shared_dir, "bob")
        bob_messenger.send("alice", "第二のタスク: セキュリティ監査")

        charlie_messenger = Messenger(shared_dir, "charlie")
        charlie_messenger.send("alice", "第三のタスク: パフォーマンス最適化")

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": trigger,
                        "action": "responded",
                        "summary": "Processed all messages",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.process_inbox_message()

        activity_file = alice_dir / "activity_log" / f"{date.today().isoformat()}.jsonl"
        assert activity_file.exists()
        lines = [l for l in activity_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        msg_entries = [e for e in (json.loads(l) for l in lines) if e.get("type") == "message_received"]
        assert len(msg_entries) >= 3
        content_str = " ".join(e.get("content", "") or "" for e in msg_entries)
        from_str = " ".join(e.get("from_person", e.get("from", "")) or "" for e in msg_entries)

        assert "ログ解析" in content_str
        assert "セキュリティ監査" in content_str
        assert "パフォーマンス最適化" in content_str
        assert "mio" in from_str or "bob" in from_str or "charlie" in from_str

    async def test_episode_recording_failure_does_not_crash_inbox(self, data_dir, make_anima):
        """Verify inbox processing continues even if episode recording fails."""
        alice_dir = make_anima("alice")
        make_anima("mio")
        shared_dir = data_dir / "shared"

        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "テストメッセージ")

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            dp.memory.append_episode = MagicMock(side_effect=OSError("Disk full"))

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": trigger,
                        "action": "responded",
                        "summary": "Processed despite error",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            result = await dp.process_inbox_message()
            # Cycle returns "responded"; status may be "idle" after processing
            assert result.action in ("responded", "idle")


class TestHeartbeatNoInboxProcessing:
    """Verify heartbeat no longer processes inbox messages (3-path separation)."""

    async def test_heartbeat_does_not_process_inbox(self, data_dir, make_anima):
        """Heartbeat should warn about unread messages but not process them."""
        alice_dir = make_anima("alice")
        make_anima("mio")
        shared_dir = data_dir / "shared"

        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "テストメッセージ")

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "HEARTBEAT_OK",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        # Messages should still be in inbox (not archived by heartbeat)
        inbox_dir = shared_dir / "inbox" / "alice"
        assert len(list(inbox_dir.glob("*.json"))) == 1
