from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""E2E integration tests for the activity log degradation fix.

These tests use REAL file I/O operations (no mocks for files) and test
actual end-to-end workflows for:
  - Shared channel visibility in priming (cross-Anima posts)
  - Messenger.send() creating dm_sent + dm_logs entries
  - Full-content preservation in dm_received activity log entries
  - format_for_priming() content_trim parameter
  - read_dm_history() type filtering (exclude message_received/response_sent)
  - read_dm_history() 30-day range retrieval

References:
  - docs/issues/20260219_activity-log-degradation-fix.md
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger
from core.memory.priming import PrimingEngine
from core.messenger import Messenger

logger = logging.getLogger(__name__)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory with required subdirs."""
    d = tmp_path / "animas" / "test-anima"
    for subdir in ("activity_log", "knowledge", "episodes", "skills"):
        (d / subdir).mkdir(parents=True)
    return d


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    """Create a shared directory with channels and users subdirs."""
    d = tmp_path / "shared"
    (d / "channels").mkdir(parents=True)
    (d / "users").mkdir(parents=True)
    (d / "inbox").mkdir(parents=True)
    return d


# ── Helpers ───────────────────────────────────────────────────


def _write_channel_posts(channels_dir: Path, channel: str, posts: list[dict]) -> Path:
    """Write JSONL channel posts to a shared channel file."""
    filepath = channels_dir / f"{channel}.jsonl"
    with filepath.open("a", encoding="utf-8") as f:
        for post in posts:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
    return filepath


def _write_activity_entries(
    anima_dir: Path,
    date_str: str,
    entries: list[dict],
) -> Path:
    """Write raw JSONL entries to an activity log file for a specific date."""
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / f"{date_str}.jsonl"
    with filepath.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return filepath


def _read_jsonl(filepath: Path) -> list[dict]:
    """Read and parse a JSONL file, returning a list of dicts."""
    entries = []
    if not filepath.exists():
        return entries
    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


# ── Tests ─────────────────────────────────────────────────────


class TestSharedChannelVisibleInPriming:
    """Test 1: Shared channel posts appear in priming output."""

    async def test_shared_channel_visible_in_priming(
        self,
        anima_dir: Path,
        shared_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Cross-Anima shared channel posts appear in Channel B priming output.

        After the degradation fix, PrimingEngine._channel_b_recent_activity()
        reads shared/channels/*.jsonl to restore cross-Anima visibility.
        """
        now = datetime.now()

        # Write some channel posts from other animas
        posts = [
            {
                "ts": (now - timedelta(minutes=30)).isoformat(),
                "from": "yuki",
                "text": "boto3の新しいバージョンで問題が発生しました",
                "source": "anima",
            },
            {
                "ts": (now - timedelta(minutes=20)).isoformat(),
                "from": "sakura",
                "text": "レビュー依頼: PR #42 のマージをお願いします",
                "source": "anima",
            },
            {
                "ts": (now - timedelta(minutes=10)).isoformat(),
                "from": "admin",
                "text": "全員に周知: 明日のメンテナンス予定 @test-anima",
                "source": "human",
            },
        ]
        _write_channel_posts(shared_dir / "channels", "general", posts)

        # Also write one entry to the anima's own activity log so the
        # code path enters the "entries exist" branch (not fallback).
        al = ActivityLogger(anima_dir)
        al.log("heartbeat_start", summary="巡回開始")

        common_skills = tmp_path / "common_skills"
        common_skills.mkdir(exist_ok=True)

        with (
            patch("core.paths.get_shared_dir", return_value=shared_dir),
            patch("core.paths.get_common_skills_dir", return_value=common_skills),
        ):
            engine = PrimingEngine(anima_dir, shared_dir)
            result = await engine._channel_b_recent_activity(
                sender_name="human",
                keywords=[],
            )

        # Shared channel posts should appear in the priming output
        assert result, "Channel B should return non-empty priming text"
        assert "boto3" in result or "yuki" in result, (
            "Channel post from yuki should appear in priming output"
        )
        assert "sakura" in result or "レビュー" in result, (
            "Channel post from sakura should appear in priming output"
        )
        # The human @mention should definitely appear
        assert "メンテナンス" in result or "admin" in result, (
            "Human @mention post should appear in priming output"
        )


class TestMessengerSendCreatesDmSentAndDmLogs:
    """Test 2: Messenger.send() creates both dm_sent and dm_logs entries."""

    def test_messenger_send_creates_dm_sent_and_dm_logs(
        self,
        tmp_path: Path,
    ) -> None:
        """Messenger.send() records dm_sent in activity log and dm_logs.

        After the fix, send() writes to:
        1. animas/{sender}/activity_log/{date}.jsonl (dm_sent)
        2. shared/dm_logs/{pair}.jsonl (legacy fallback)
        """
        # Build directory structure
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "recipient").mkdir(parents=True)
        sender_anima_dir = tmp_path / "animas" / "sender"
        (sender_anima_dir / "activity_log").mkdir(parents=True)

        messenger = Messenger(shared_dir, "sender")
        messenger.send("recipient", "hello from sender")

        # 1. Check activity log has dm_sent entry
        today = date.today().isoformat()
        activity_log_path = sender_anima_dir / "activity_log" / f"{today}.jsonl"
        assert activity_log_path.exists(), "Activity log file should be created"

        activity_entries = _read_jsonl(activity_log_path)
        dm_sent_entries = [e for e in activity_entries if e.get("type") == "dm_sent"]
        assert len(dm_sent_entries) >= 1, "At least one dm_sent entry should exist"

        dm_entry = dm_sent_entries[0]
        assert dm_entry.get("to") == "recipient"
        assert dm_entry.get("content") == "hello from sender"

        # 2. Check dm_logs has an entry
        pair = sorted(["sender", "recipient"])
        dm_log_path = shared_dir / "dm_logs" / f"{pair[0]}-{pair[1]}.jsonl"
        assert dm_log_path.exists(), "DM log file should be created"

        dm_log_entries = _read_jsonl(dm_log_path)
        assert len(dm_log_entries) >= 1, "At least one dm_log entry should exist"
        assert dm_log_entries[0].get("from") == "sender"
        assert dm_log_entries[0].get("text") == "hello from sender"


class TestDmReceivedFullContentInActivityLog:
    """Test 4: ActivityLogger stores full content in dm_received entries."""

    def test_dm_received_full_content_in_activity_log(
        self,
        anima_dir: Path,
    ) -> None:
        """ActivityLogger.log('dm_received', content=long_content) stores full content.

        After the fix, content is no longer truncated to 200 chars at write time.
        The full content is preserved in the JSONL file.
        """
        al = ActivityLogger(anima_dir)

        # Generate a 500-character content string
        long_content = "これは500文字のテストメッセージです。" * 25  # ~400+ chars
        long_content = long_content[:500]
        assert len(long_content) == 500

        al.log(
            "dm_received",
            content=long_content,
            from_person="yuki",
            summary=long_content[:200],
        )

        # Read back the raw JSONL
        today = date.today().isoformat()
        log_path = anima_dir / "activity_log" / f"{today}.jsonl"
        assert log_path.exists()

        entries = _read_jsonl(log_path)
        dm_entries = [e for e in entries if e.get("type") == "dm_received"]
        assert len(dm_entries) == 1

        stored_content = dm_entries[0].get("content", "")
        assert len(stored_content) == 500, (
            f"Full 500-char content should be stored, got {len(stored_content)} chars"
        )
        assert stored_content == long_content


class TestFormatForPrimingContentTrim:
    """Test 5: format_for_priming() respects content_trim parameter."""

    def test_format_for_priming_with_content_trim(
        self,
        anima_dir: Path,
    ) -> None:
        """format_for_priming(entries, content_trim=100) trims long content."""
        al = ActivityLogger(anima_dir)

        # Create entries with long content (500+ chars)
        long_text = "A" * 600
        entries = [
            ActivityEntry(
                ts=datetime.now().isoformat(),
                type="message_received",
                content=long_text,
                from_person="alice",
                channel="chat",
            ),
        ]

        # With content_trim=100
        result_trimmed = al.format_for_priming(
            entries, budget_tokens=5000, content_trim=100,
        )
        assert result_trimmed, "Should produce non-empty output"
        # The entry should not contain the full 600-char content
        assert "A" * 600 not in result_trimmed, (
            "Content should be trimmed when content_trim=100"
        )
        # Should contain the truncation marker
        assert "..." in result_trimmed, "Trimmed content should contain '...'"

    def test_format_for_priming_no_trim(
        self,
        anima_dir: Path,
    ) -> None:
        """format_for_priming(entries, content_trim=0) preserves full content."""
        al = ActivityLogger(anima_dir)

        long_text = "B" * 300
        entries = [
            ActivityEntry(
                ts=datetime.now().isoformat(),
                type="message_received",
                content=long_text,
                from_person="bob",
                channel="chat",
            ),
        ]

        # With content_trim=0 (no trimming)
        result_full = al.format_for_priming(
            entries, budget_tokens=5000, content_trim=0,
        )
        assert result_full, "Should produce non-empty output"
        # Full content should be preserved (within budget)
        assert "B" * 300 in result_full, (
            "Full content should be preserved when content_trim=0"
        )


class TestReadDmHistoryExcludesMessageTypes:
    """Test 6: read_dm_history() excludes message_received/response_sent."""

    def test_read_dm_history_excludes_message_types(
        self,
        tmp_path: Path,
    ) -> None:
        """read_dm_history() only returns dm_sent and dm_received entries.

        After the fix, the type filter no longer includes message_received
        and response_sent, which polluted DM history with chat conversations.
        """
        # Build directory structure
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "test-anima").mkdir(parents=True)
        anima_dir = tmp_path / "animas" / "test-anima"
        (anima_dir / "activity_log").mkdir(parents=True)

        now = datetime.now()
        today = date.today().isoformat()

        # Write mixed activity log entries involving "peer-anima"
        entries = [
            {
                "ts": (now - timedelta(minutes=40)).isoformat(),
                "type": "dm_sent",
                "content": "DM送信: タスクの確認です",
                "to": "peer-anima",
            },
            {
                "ts": (now - timedelta(minutes=35)).isoformat(),
                "type": "dm_received",
                "content": "DM受信: 了解しました",
                "from": "peer-anima",
            },
            {
                "ts": (now - timedelta(minutes=30)).isoformat(),
                "type": "message_received",
                "content": "MSG受信: これはチャットメッセージ",
                "from": "peer-anima",
                "channel": "chat",
            },
            {
                "ts": (now - timedelta(minutes=25)).isoformat(),
                "type": "response_sent",
                "content": "MSG送信: チャット応答",
                "to": "peer-anima",
                "channel": "chat",
            },
        ]
        _write_activity_entries(anima_dir, today, entries)

        messenger = Messenger(shared_dir, "test-anima")
        history = messenger.read_dm_history("peer-anima", limit=20)

        # Should only contain dm_sent and dm_received
        for entry in history:
            text = entry.get("text", "")
            assert "DM送信" in text or "DM受信" in text, (
                f"Only dm_sent/dm_received should appear, got: {text}"
            )

        # Should NOT contain message_received or response_sent content
        all_texts = " ".join(e.get("text", "") for e in history)
        assert "MSG受信" not in all_texts, (
            "message_received content should not appear in DM history"
        )
        assert "MSG送信" not in all_texts, (
            "response_sent content should not appear in DM history"
        )

        # Should have exactly 2 entries (dm_sent + dm_received)
        assert len(history) == 2, f"Expected 2 DM entries, got {len(history)}"


class TestReadDmHistory30DayRange:
    """Test 7: read_dm_history() retrieves DMs from 8+ days ago."""

    def test_read_dm_history_30_day_range(
        self,
        tmp_path: Path,
    ) -> None:
        """read_dm_history() can retrieve DMs from 10 days ago.

        After the fix, days=30 (instead of days=7) allows retrieval of
        older DM conversations.
        """
        # Build directory structure
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "test-anima").mkdir(parents=True)
        anima_dir = tmp_path / "animas" / "test-anima"
        (anima_dir / "activity_log").mkdir(parents=True)

        # Write a dm_sent entry dated 10 days ago
        old_date = date.today() - timedelta(days=10)
        old_date_str = old_date.isoformat()
        old_ts = datetime.combine(old_date, datetime.min.time().replace(hour=14))

        entries = [
            {
                "ts": old_ts.isoformat(),
                "type": "dm_sent",
                "content": "OLD_DM_FROM_10_DAYS_AGO",
                "to": "peer-anima",
            },
            {
                "ts": (old_ts + timedelta(minutes=5)).isoformat(),
                "type": "dm_received",
                "content": "OLD_DM_REPLY_FROM_10_DAYS_AGO",
                "from": "peer-anima",
            },
        ]
        _write_activity_entries(anima_dir, old_date_str, entries)

        messenger = Messenger(shared_dir, "test-anima")
        history = messenger.read_dm_history("peer-anima", limit=20)

        # The 10-day-old entries should be retrieved (days=30 covers this)
        assert len(history) >= 2, (
            f"Expected at least 2 entries from 10 days ago, got {len(history)}"
        )

        all_texts = " ".join(e.get("text", "") for e in history)
        assert "OLD_DM_FROM_10_DAYS_AGO" in all_texts, (
            "10-day-old dm_sent should be retrievable with days=30"
        )
        assert "OLD_DM_REPLY_FROM_10_DAYS_AGO" in all_texts, (
            "10-day-old dm_received should be retrievable with days=30"
        )
