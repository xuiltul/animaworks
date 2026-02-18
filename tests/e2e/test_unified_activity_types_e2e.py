"""E2E tests for unified activity type mapping — backend summary fields.

Verifies that DigitalAnima methods produce activity log entries
with correct summary fields, using real ActivityLogger on disk.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.activity import ActivityLogger, ActivityEntry
from core.messenger import InboxItem
from core.schemas import CycleResult


# ── Helpers ───────────────────────────────────────────────

def _read_activity_entries(anima_dir: Path) -> list[dict]:
    """Read all JSONL entries from today's activity log."""
    log_dir = anima_dir / "activity_log"
    today_file = log_dir / f"{date.today().isoformat()}.jsonl"
    if not today_file.exists():
        return []
    entries = []
    for line in today_file.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _find_entries_by_type(entries: list[dict], event_type: str) -> list[dict]:
    """Filter entries by event type."""
    return [e for e in entries if e.get("type") == event_type]


def _make_cycle_result(**kwargs) -> CycleResult:
    defaults = dict(trigger="test", action="responded", summary="done", duration_ms=100)
    defaults.update(kwargs)
    return CycleResult(**defaults)


# ── Heartbeat Start Summary ──────────────────────────────

class TestHeartbeatStartSummary:
    """Verify heartbeat_start activity entry includes summary."""

    @pytest.mark.asyncio
    async def test_heartbeat_start_has_summary(self, data_dir, make_anima) -> None:
        """run_heartbeat should log heartbeat_start with summary='定期巡回開始'."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsger, \
             patch("core.anima.ConversationMemory") as MockConvMem, \
             patch("core.anima.StreamingJournal") as MockJournal:

            # Setup mocks
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "チェック項目"
            MockMM.return_value.append_episode = MagicMock()
            MockMM.return_value.append_cron_log = MagicMock()
            MockMsger.return_value.has_unread.return_value = False
            MockMsger.return_value.unread_count.return_value = 0

            mock_conv = MagicMock()
            mock_conv.load.return_value = MagicMock(turns=[])
            mock_conv.compress_if_needed = AsyncMock()
            mock_conv.finalize_if_session_ended = AsyncMock()
            MockConvMem.return_value = mock_conv

            mock_journal = MagicMock()
            mock_journal.open = MagicMock()
            mock_journal.write_text = MagicMock()
            mock_journal.finalize = MagicMock()
            mock_journal.close = MagicMock()
            MockJournal.return_value = mock_journal

            # Agent streaming yields cycle_done
            async def mock_stream(*args, **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "HEARTBEAT_OK: すべて正常",
                        "duration_ms": 50,
                    },
                }

            mock_agent = MockAgent.return_value
            mock_agent.run_cycle_streaming = mock_stream
            mock_agent.reset_reply_tracking = MagicMock()
            mock_agent.replied_to = set()
            mock_agent.background_manager = None
            mock_agent.drain_notifications.return_value = []

            from core.anima import DigitalAnima
            anima = DigitalAnima(anima_dir, shared_dir)
            await anima.run_heartbeat()

            # Verify activity log
            entries = _read_activity_entries(anima_dir)
            hb_starts = _find_entries_by_type(entries, "heartbeat_start")
            assert len(hb_starts) >= 1, "Expected at least 1 heartbeat_start entry"
            assert hb_starts[0].get("summary") == "定期巡回開始", (
                f"heartbeat_start summary should be '定期巡回開始', got: {hb_starts[0].get('summary')}"
            )


# ── Message Received Summary ─────────────────────────────

class TestMessageReceivedSummary:
    """Verify message_received activity entry includes summary."""

    @pytest.mark.asyncio
    async def test_process_message_logs_summary(self, data_dir, make_anima) -> None:
        """process_message should log message_received with summary=content[:100]."""
        anima_dir = make_anima("bob")
        shared_dir = data_dir / "shared"

        test_content = "こんにちは、テストメッセージです。"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsger, \
             patch("core.anima.ConversationMemory") as MockConvMem:

            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMsger.return_value.unread_count.return_value = 0

            mock_conv = MagicMock()
            mock_conv.build_chat_prompt.return_value = "prompt"
            mock_conv.compress_if_needed = AsyncMock()
            mock_conv.append_turn = MagicMock()
            mock_conv.save = MagicMock()
            MockConvMem.return_value = mock_conv

            mock_agent = MockAgent.return_value
            mock_agent.run_cycle = AsyncMock(return_value=_make_cycle_result())
            mock_agent.background_manager = None
            mock_agent.drain_notifications.return_value = []

            from core.anima import DigitalAnima
            anima = DigitalAnima(anima_dir, shared_dir)
            await anima.process_message(test_content, from_person="user")

            entries = _read_activity_entries(anima_dir)
            msg_received = _find_entries_by_type(entries, "message_received")
            assert len(msg_received) >= 1, "Expected at least 1 message_received entry"
            assert msg_received[0].get("summary") == test_content[:100], (
                f"message_received summary should be content[:100], got: {msg_received[0].get('summary')}"
            )


# ── DM Received Summary ─────────────────────────────────

class TestDmReceivedSummary:
    """Verify dm_received activity entry includes summary."""

    @pytest.mark.asyncio
    async def test_heartbeat_with_dm_logs_summary(self, data_dir, make_anima) -> None:
        """run_heartbeat processing DMs should log dm_received with summary."""
        anima_dir = make_anima("carol")
        shared_dir = data_dir / "shared"

        dm_content = "緊急の報告です。システムに問題が発生しています。"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsger, \
             patch("core.anima.ConversationMemory") as MockConvMem, \
             patch("core.anima.StreamingJournal") as MockJournal:

            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "チェック項目"
            MockMM.return_value.append_episode = MagicMock()
            MockMsger.return_value.unread_count.return_value = 1

            # Mock unread messages
            mock_msg = MagicMock()
            mock_msg.from_person = "dave"
            mock_msg.content = dm_content
            mock_msg.type = "message"
            MockMsger.return_value.has_unread.return_value = True
            MockMsger.return_value.receive_with_paths.return_value = [InboxItem(msg=mock_msg, path=Path("/fake/msg.json"))]
            MockMsger.return_value.archive_paths.return_value = 1

            mock_conv = MagicMock()
            mock_conv.load.return_value = MagicMock(turns=[])
            mock_conv.compress_if_needed = AsyncMock()
            mock_conv.finalize_if_session_ended = AsyncMock()
            MockConvMem.return_value = mock_conv

            mock_journal = MagicMock()
            mock_journal.open = MagicMock()
            mock_journal.write_text = MagicMock()
            mock_journal.finalize = MagicMock()
            mock_journal.close = MagicMock()
            MockJournal.return_value = mock_journal

            async def mock_stream(*args, **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "メッセージ処理完了",
                        "duration_ms": 50,
                    },
                }

            mock_agent = MockAgent.return_value
            mock_agent.run_cycle_streaming = mock_stream
            mock_agent.reset_reply_tracking = MagicMock()
            mock_agent.replied_to = {"dave"}
            mock_agent.background_manager = None
            mock_agent.drain_notifications.return_value = []

            from core.anima import DigitalAnima
            anima = DigitalAnima(anima_dir, shared_dir)
            await anima.run_heartbeat()

            entries = _read_activity_entries(anima_dir)
            dm_entries = _find_entries_by_type(entries, "dm_received")
            assert len(dm_entries) >= 1, "Expected at least 1 dm_received entry"
            assert dm_entries[0].get("summary") == dm_content[:100], (
                f"dm_received summary should be content[:100], got: {dm_entries[0].get('summary')}"
            )


# ── ActivityLogger Integration ───────────────────────────

class TestActivityLoggerSummaryField:
    """Verify ActivityLogger correctly stores and retrieves summary field."""

    def test_log_with_summary(self, tmp_path: Path) -> None:
        """activity.log() with summary param should persist it in JSONL."""
        anima_dir = tmp_path / "animas" / "test"
        (anima_dir / "activity_log").mkdir(parents=True)

        activity = ActivityLogger(anima_dir)
        activity.log("heartbeat_start", summary="定期巡回開始")

        entries = _read_activity_entries(anima_dir)
        assert len(entries) == 1
        assert entries[0]["type"] == "heartbeat_start"
        assert entries[0]["summary"] == "定期巡回開始"

    def test_log_with_content_and_summary(self, tmp_path: Path) -> None:
        """Both content and summary should be stored."""
        anima_dir = tmp_path / "animas" / "test"
        (anima_dir / "activity_log").mkdir(parents=True)

        activity = ActivityLogger(anima_dir)
        activity.log(
            "message_received",
            content="full message content here",
            summary="full message co",
            from_person="user",
        )

        entries = _read_activity_entries(anima_dir)
        assert len(entries) == 1
        assert entries[0]["content"] == "full message content here"
        assert entries[0]["summary"] == "full message co"
        assert entries[0]["from"] == "user"

    def test_recent_returns_summary(self, tmp_path: Path) -> None:
        """ActivityLogger.recent() should return entries with summary."""
        anima_dir = tmp_path / "animas" / "test"
        (anima_dir / "activity_log").mkdir(parents=True)

        activity = ActivityLogger(anima_dir)
        activity.log("heartbeat_start", summary="定期巡回開始")
        activity.log("dm_received", content="hi", summary="hi", from_person="alice")

        entries = activity.recent(days=1)
        assert len(entries) == 2
        summaries = [e.summary for e in entries]
        assert "定期巡回開始" in summaries
        assert "hi" in summaries
