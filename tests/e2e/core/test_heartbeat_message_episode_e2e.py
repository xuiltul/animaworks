# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.messenger import Messenger
from core.memory.manager import MemoryManager


class TestHeartbeatMessageEpisodeE2E:
    """Full integration tests for message episode recording during heartbeat."""

    async def test_message_content_persists_in_episodes(self, data_dir, make_anima):
        """Full integration: message sent → heartbeat → episode contains message content."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Send a real message from mio to alice
        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "AWS監視タスクを追加しました。30分間隔で確認してください。")

        # Create DigitalAnima with real MemoryManager and Messenger, but mock AgentCore
        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Processed mio message",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        # Verify episode file contains the message
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")
        assert "mioからのメッセージ受信" in content
        assert "AWS監視タスク" in content

    async def test_messages_archived_after_episode_recording(self, data_dir, make_anima):
        """Verify messages are archived after being recorded to episodes."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Send a message
        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "テストメッセージ: ログ確認をお願いします。")

        # Verify message is in inbox before heartbeat
        inbox_dir = shared_dir / "inbox" / "alice"
        assert len(list(inbox_dir.glob("*.json"))) == 1

        # Mock and run heartbeat
        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Processed message",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            # Mock archive_paths to verify it's called after episode recording
            original_archive = dp.messenger.archive_paths
            archive_called = False

            def mock_archive(items):
                nonlocal archive_called
                archive_called = True
                return original_archive(items)

            dp.messenger.archive_paths = mock_archive

            await dp.run_heartbeat()

            # Verify episode was created with message content
            episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
            assert episode_file.exists()
            content = episode_file.read_text(encoding="utf-8")
            assert "mioからのメッセージ受信" in content
            assert "ログ確認" in content

            # Verify archive was called (messages moved to processed/)
            assert archive_called

    async def test_ack_messages_not_in_episodes(self, data_dir, make_anima):
        """Verify ACK messages are filtered out and not recorded in episodes."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Send both a normal message and an ACK message
        from core.schemas import Message

        mio_messenger = Messenger(shared_dir, "mio")
        # Normal message
        mio_messenger.send("alice", "重要なタスク: DB バックアップを実行してください。")
        # ACK message (should be filtered out)
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

        # Mock and run heartbeat
        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Processed messages",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        # Verify episode file contains only the normal message, not the ACK
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")

        # Should contain the normal message
        assert "mioからのメッセージ受信" in content
        assert "DB バックアップ" in content

        # Should NOT contain the ACK message content
        assert "了解しました" not in content

    async def test_multiple_messages_recorded_in_order(self, data_dir, make_anima):
        """Verify multiple messages are all recorded to episodes in order."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Send multiple messages from different senders
        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "第一のタスク: ログ解析")

        bob_messenger = Messenger(shared_dir, "bob")
        bob_messenger.send("alice", "第二のタスク: セキュリティ監査")

        charlie_messenger = Messenger(shared_dir, "charlie")
        charlie_messenger.send("alice", "第三のタスク: パフォーマンス最適化")

        # Mock and run heartbeat
        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Processed all messages",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        # Verify all messages are recorded in episodes
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")

        # Check all messages are present
        assert "mioからのメッセージ受信" in content
        assert "ログ解析" in content
        assert "bobからのメッセージ受信" in content
        assert "セキュリティ監査" in content
        assert "charlieからのメッセージ受信" in content
        assert "パフォーマンス最適化" in content

        # Verify count - should have 3 message episodes
        assert content.count("からのメッセージ受信") == 3

    async def test_episode_recording_failure_does_not_crash_heartbeat(self, data_dir, make_anima):
        """Verify heartbeat continues even if episode recording fails."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Send a message
        mio_messenger = Messenger(shared_dir, "mio")
        mio_messenger.send("alice", "テストメッセージ")

        # Mock and run heartbeat with episode recording failure
        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            # Make append_episode raise an error
            original_append = dp.memory.append_episode
            dp.memory.append_episode = MagicMock(side_effect=OSError("Disk full"))

            async def mock_stream(prompt, trigger="manual"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Processed despite error",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            # Heartbeat should not crash despite episode recording failure
            result = await dp.run_heartbeat()
            assert result.trigger == "heartbeat"
            assert result.action == "responded"
