# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for heartbeat message episode recording.

Covers:
- Inter-Anima messages received during heartbeat are recorded to episodic memory
- ACK messages are excluded from recording
- Message content is truncated at 1000 chars
- Maximum 50 messages are recorded per heartbeat
- Heartbeat summary includes message count
- Episode recording failures do not block heartbeat completion
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.messenger import InboxItem
from core.schemas import Message


# ══════════════════════════════════════════════════════════
# TestHeartbeatMessageEpisodeRecording
# ══════════════════════════════════════════════════════════


class TestHeartbeatMessageEpisodeRecording:
    """Tests that run_heartbeat records inter-Anima messages to episodes."""

    async def test_message_content_recorded_to_episode(self, data_dir, make_anima):
        """When heartbeat processes 1 message, verify message content is recorded to episodes."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # Create a message from "mio"
            msg = Message(from_person="mio", to_person="alice", content="Hello from mio")
            MockMsg.return_value.has_unread.return_value = True
            MockMsg.return_value.receive_with_paths.return_value = [InboxItem(msg=msg, path=Path(f"/fake/{msg.id}.json"))]
            MockMsg.return_value.archive_paths.return_value = 1

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
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

            result = await dp.run_heartbeat()

            # append_episode should be called at least 2 times:
            # 1. For the message episode
            # 2. For the heartbeat summary episode
            assert MockMM.return_value.append_episode.call_count >= 2

            # Check that one of the calls contains the message content
            all_calls = [call[0][0] for call in MockMM.return_value.append_episode.call_args_list]
            message_episodes = [ep for ep in all_calls if "mioからのメッセージ受信" in ep]
            assert len(message_episodes) == 1
            assert "Hello from mio" in message_episodes[0]
            assert "**送信者**: mio" in message_episodes[0]

    async def test_multiple_messages_recorded_separately(self, data_dir, make_anima):
        """When 3 messages from different senders arrive, each gets its own episode entry."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # Create 3 messages from different senders
            msg1 = Message(from_person="mio", to_person="alice", content="Message from mio")
            msg2 = Message(from_person="bob", to_person="alice", content="Message from bob")
            msg3 = Message(from_person="carol", to_person="alice", content="Message from carol")
            MockMsg.return_value.has_unread.return_value = True
            MockMsg.return_value.receive_with_paths.return_value = [
                InboxItem(msg=msg1, path=Path(f"/fake/{msg1.id}.json")),
                InboxItem(msg=msg2, path=Path(f"/fake/{msg2.id}.json")),
                InboxItem(msg=msg3, path=Path(f"/fake/{msg3.id}.json")),
            ]
            MockMsg.return_value.archive_paths.return_value = 3

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
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

            result = await dp.run_heartbeat()

            # append_episode should be called at least 4 times:
            # 3 for message episodes + 1 for heartbeat summary
            assert MockMM.return_value.append_episode.call_count >= 4

            # Check that each message has its own episode
            all_calls = [call[0][0] for call in MockMM.return_value.append_episode.call_args_list]
            mio_episodes = [ep for ep in all_calls if "mioからのメッセージ受信" in ep]
            bob_episodes = [ep for ep in all_calls if "bobからのメッセージ受信" in ep]
            carol_episodes = [ep for ep in all_calls if "carolからのメッセージ受信" in ep]

            assert len(mio_episodes) == 1
            assert len(bob_episodes) == 1
            assert len(carol_episodes) == 1
            assert "Message from mio" in mio_episodes[0]
            assert "Message from bob" in bob_episodes[0]
            assert "Message from carol" in carol_episodes[0]

    async def test_ack_messages_excluded_from_episode(self, data_dir, make_anima):
        """ACK type messages should NOT be recorded to episodes."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # Create 2 regular messages and 1 ACK message
            msg1 = Message(from_person="mio", to_person="alice", type="message", content="Regular message")
            msg2 = Message(from_person="bob", to_person="alice", type="ack", content="ACK message")
            msg3 = Message(from_person="carol", to_person="alice", type="message", content="Another regular")
            MockMsg.return_value.has_unread.return_value = True
            MockMsg.return_value.receive_with_paths.return_value = [
                InboxItem(msg=msg1, path=Path(f"/fake/{msg1.id}.json")),
                InboxItem(msg=msg2, path=Path(f"/fake/{msg2.id}.json")),
                InboxItem(msg=msg3, path=Path(f"/fake/{msg3.id}.json")),
            ]
            MockMsg.return_value.archive_paths.return_value = 3

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
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

            result = await dp.run_heartbeat()

            # Should only record 2 message episodes (not the ACK) + 1 heartbeat summary
            all_calls = [call[0][0] for call in MockMM.return_value.append_episode.call_args_list]
            message_episodes = [ep for ep in all_calls if "からのメッセージ受信" in ep]

            # Only 2 message episodes should be recorded (ACK excluded)
            assert len(message_episodes) == 2
            # Verify ACK content is not in any episode
            all_episodes_text = "\n".join(all_calls)
            assert "ACK message" not in all_episodes_text
            # Verify regular messages are recorded
            assert "Regular message" in all_episodes_text
            assert "Another regular" in all_episodes_text

    async def test_message_content_truncated_at_1000_chars(self, data_dir, make_anima):
        """Message content longer than 1000 chars should be truncated in episode."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # Create a message with 1500 chars
            long_content = "A" * 1500
            msg = Message(from_person="mio", to_person="alice", content=long_content)
            MockMsg.return_value.has_unread.return_value = True
            MockMsg.return_value.receive_with_paths.return_value = [InboxItem(msg=msg, path=Path(f"/fake/{msg.id}.json"))]
            MockMsg.return_value.archive_paths.return_value = 1

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
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

            result = await dp.run_heartbeat()

            # Find the message episode
            all_calls = [call[0][0] for call in MockMM.return_value.append_episode.call_args_list]
            message_episodes = [ep for ep in all_calls if "mioからのメッセージ受信" in ep]
            assert len(message_episodes) == 1

            # Verify content is truncated at 1000 chars
            episode = message_episodes[0]
            # Extract the content section (after "**内容**:\n")
            content_start = episode.index("**内容**:\n") + len("**内容**:\n")
            content_part = episode[content_start:]
            # The content should be exactly 1000 A's (plus potential newline)
            assert content_part.strip() == "A" * 1000

    async def test_max_50_messages_recorded(self, data_dir, make_anima):
        """When 60 messages arrive, only the first 50 should be recorded to episodes."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # Create 60 messages
            messages = [
                Message(from_person=f"sender{i}", to_person="alice", content=f"Message {i}")
                for i in range(60)
            ]
            MockMsg.return_value.has_unread.return_value = True
            MockMsg.return_value.receive_with_paths.return_value = [
                InboxItem(msg=m, path=Path(f"/fake/{m.id}.json")) for m in messages
            ]
            MockMsg.return_value.archive_paths.return_value = 60

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
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

            result = await dp.run_heartbeat()

            # Count message episodes (excluding heartbeat summary)
            all_calls = [call[0][0] for call in MockMM.return_value.append_episode.call_args_list]
            message_episodes = [ep for ep in all_calls if "からのメッセージ受信" in ep]

            # Should only record 50 message episodes (not all 60)
            assert len(message_episodes) == 50

            # Verify messages 0-49 are recorded
            for i in range(50):
                assert any(f"Message {i}" in ep for ep in message_episodes)

            # Verify messages 50-59 are NOT recorded
            all_episodes_text = "\n".join(all_calls)
            for i in range(50, 60):
                assert f"Message {i}" not in all_episodes_text

    async def test_no_messages_no_episode_recording(self, data_dir, make_anima):
        """When heartbeat has no unread messages, message episodes should not be recorded."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # No unread messages
            MockMsg.return_value.has_unread.return_value = False

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "checked",
                        "summary": "HEARTBEAT_OK",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            result = await dp.run_heartbeat()

            # append_episode should NOT be called for HEARTBEAT_OK
            MockMM.return_value.append_episode.assert_not_called()

    async def test_heartbeat_summary_includes_message_count(self, data_dir, make_anima):
        """Heartbeat summary episode should include message count when messages are processed."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # Create 3 messages
            msg1 = Message(from_person="mio", to_person="alice", content="Message 1")
            msg2 = Message(from_person="bob", to_person="alice", content="Message 2")
            msg3 = Message(from_person="carol", to_person="alice", content="Message 3")
            MockMsg.return_value.has_unread.return_value = True
            MockMsg.return_value.receive_with_paths.return_value = [
                InboxItem(msg=msg1, path=Path(f"/fake/{msg1.id}.json")),
                InboxItem(msg=msg2, path=Path(f"/fake/{msg2.id}.json")),
                InboxItem(msg=msg3, path=Path(f"/fake/{msg3.id}.json")),
            ]
            MockMsg.return_value.archive_paths.return_value = 3

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
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

            result = await dp.run_heartbeat()

            # Find the heartbeat summary episode (should be the last call)
            all_calls = [call[0][0] for call in MockMM.return_value.append_episode.call_args_list]
            # The heartbeat summary episode contains "ハートビート活動"
            heartbeat_episodes = [ep for ep in all_calls if "ハートビート活動" in ep]
            assert len(heartbeat_episodes) == 1

            # Verify it includes the message count
            summary_episode = heartbeat_episodes[0]
            assert "（3件のメッセージを処理）" in summary_episode

    async def test_episode_recording_failure_does_not_block_heartbeat(self, data_dir, make_anima):
        """When append_episode raises exception, heartbeat should still complete normally."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            # Make append_episode raise an exception
            MockMM.return_value.append_episode = MagicMock(side_effect=Exception("Episode write failed"))
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            # Create a message
            msg = Message(from_person="mio", to_person="alice", content="Test message")
            MockMsg.return_value.has_unread.return_value = True
            MockMsg.return_value.receive_with_paths.return_value = [InboxItem(msg=msg, path=Path(f"/fake/{msg.id}.json"))]
            MockMsg.return_value.archive_paths.return_value = 1

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
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

            # Heartbeat should complete without raising exception
            result = await dp.run_heartbeat()

            # Verify heartbeat completed successfully
            assert result.summary == "Processed messages"
            assert result.action == "responded"

            # Verify messages were still archived despite episode recording failure
            MockMsg.return_value.archive_paths.assert_called_once()
