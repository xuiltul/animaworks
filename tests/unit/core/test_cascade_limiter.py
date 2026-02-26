from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for ConversationDepthLimiter (file-based implementation).

Covers:
- check_depth blocks when exchange count exceeds max_depth
- check_depth allows when under limit
- check_depth fail-closed when activity_log is missing
- check_global_outbound hourly/daily limits and fail-closed on error
- current_depth reporting
- Legacy check_and_record always returns True (no-op stub)
- Module-level singleton exists
"""

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.cascade_limiter import ConversationDepthLimiter
from core.time_utils import now_jst


def _write_activity_entries(
    anima_dir: Path,
    entries: list[dict],
) -> None:
    """Write activity log entries for testing."""
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    today = now_jst().strftime("%Y-%m-%d")
    log_file = log_dir / f"{today}.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _make_dm_entry(
    event_type: str = "dm_sent",
    from_person: str = "alice",
    to_person: str = "bob",
    content: str = "hello",
    ts: str | None = None,
) -> dict:
    """Create a DM activity log entry."""
    return {
        "ts": ts or now_jst().isoformat(),
        "type": event_type,
        "content": content,
        "from_person": from_person,
        "to_person": to_person,
    }


@pytest.fixture
def _patch_config():
    """Patch load_config for ConversationDepthLimiter.__init__."""
    with patch("core.cascade_limiter.load_config") as mock_cfg:
        mock_cfg.return_value.heartbeat.depth_window_s = 600
        mock_cfg.return_value.heartbeat.max_depth = 6
        yield mock_cfg


class TestCheckDepth:
    """Test file-based check_depth method."""

    def test_blocks_on_exceeded(self, tmp_path: Path, _patch_config):
        """depth超過でFalseを返す。"""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        limiter = ConversationDepthLimiter(window_s=600, max_depth=3)

        entries = [
            _make_dm_entry("dm_sent", "alice", "bob"),
            _make_dm_entry("dm_received", "bob", "alice"),
            _make_dm_entry("dm_sent", "alice", "bob"),
        ]
        _write_activity_entries(anima_dir, entries)

        assert limiter.check_depth("alice", "bob", anima_dir) is False

    def test_allows_under_limit(self, tmp_path: Path, _patch_config):
        """limit内でTrueを返す。"""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        limiter = ConversationDepthLimiter(window_s=600, max_depth=6)

        entries = [
            _make_dm_entry("dm_sent", "alice", "bob"),
            _make_dm_entry("dm_received", "bob", "alice"),
        ]
        _write_activity_entries(anima_dir, entries)

        assert limiter.check_depth("alice", "bob", anima_dir) is True

    def test_fail_closed_on_read_error(self, tmp_path: Path, _patch_config):
        """activity_log読み取りエラー時はFalseを返す（fail-closed）。"""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        limiter = ConversationDepthLimiter(window_s=600, max_depth=3)
        with patch("core.memory.activity.ActivityLogger.recent", side_effect=OSError("disk error")):
            assert limiter.check_depth("alice", "bob", anima_dir) is False

    def test_old_entries_not_counted(self, tmp_path: Path, _patch_config):
        """ウィンドウ外の古いエントリはカウントされない。"""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        old_ts = (now_jst() - timedelta(seconds=700)).isoformat()
        entries = [
            _make_dm_entry("dm_sent", "alice", "bob", ts=old_ts),
            _make_dm_entry("dm_received", "bob", "alice", ts=old_ts),
            _make_dm_entry("dm_sent", "alice", "bob", ts=old_ts),
        ]
        _write_activity_entries(anima_dir, entries)

        limiter = ConversationDepthLimiter(window_s=600, max_depth=3)
        assert limiter.check_depth("alice", "bob", anima_dir) is True


class TestCheckGlobalOutbound:
    """Test check_global_outbound hourly/daily limits."""

    def test_allows_under_hourly_limit(self, tmp_path: Path, _patch_config):
        """Under hourly limit → returns True."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        entries = [
            _make_dm_entry("dm_sent", "alice", "bob"),
            _make_dm_entry("dm_sent", "alice", "charlie"),
        ]
        _write_activity_entries(anima_dir, entries)

        limiter = ConversationDepthLimiter(max_per_hour=3, max_per_day=5)
        assert limiter.check_global_outbound("alice", anima_dir) is True

    def test_blocks_on_hourly_limit(self, tmp_path: Path, _patch_config):
        """At/above hourly limit → returns False."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        entries = [
            _make_dm_entry("dm_sent", "alice", "bob"),
            _make_dm_entry("dm_sent", "alice", "charlie"),
            _make_dm_entry("dm_sent", "alice", "dave"),
        ]
        _write_activity_entries(anima_dir, entries)

        limiter = ConversationDepthLimiter(max_per_hour=3, max_per_day=5)
        result = limiter.check_global_outbound("alice", anima_dir)
        assert result is not True
        assert isinstance(result, str)
        assert "GlobalOutboundLimitExceeded" in result

    def test_blocks_on_daily_limit(self, tmp_path: Path, _patch_config):
        """At/above daily limit → returns False."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        entries = [
            _make_dm_entry("dm_sent", "alice", f"user{i}")
            for i in range(5)
        ]
        _write_activity_entries(anima_dir, entries)

        limiter = ConversationDepthLimiter(max_per_hour=10, max_per_day=5)
        result = limiter.check_global_outbound("alice", anima_dir)
        assert result is not True
        assert isinstance(result, str)
        assert "GlobalOutboundLimitExceeded" in result

    def test_fail_closed_on_error(self, tmp_path: Path, _patch_config):
        """When activity log read fails → returns False."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        limiter = ConversationDepthLimiter(max_per_hour=3, max_per_day=5)
        with patch(
            "core.memory.activity.ActivityLogger",
            side_effect=OSError("read failed"),
        ):
            result = limiter.check_global_outbound("alice", anima_dir)
            assert result is not True
            assert isinstance(result, str)
            assert "GlobalOutboundLimitExceeded" in result

    def test_old_entries_not_counted_hourly(self, tmp_path: Path, _patch_config):
        """Entries older than 1 hour don't count for hourly."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        old_ts = (now_jst() - timedelta(hours=2)).isoformat()
        entries = [
            _make_dm_entry("dm_sent", "alice", "bob", ts=old_ts),
            _make_dm_entry("dm_sent", "alice", "charlie", ts=old_ts),
            _make_dm_entry("dm_sent", "alice", "dave"),
        ]
        _write_activity_entries(anima_dir, entries)

        limiter = ConversationDepthLimiter(max_per_hour=3, max_per_day=5)
        assert limiter.check_global_outbound("alice", anima_dir) is True

    def test_old_entries_not_counted_daily(self, tmp_path: Path, _patch_config):
        """Entries older than 24 hours don't count for daily."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        old_ts = (now_jst() - timedelta(hours=25)).isoformat()
        entries = [
            _make_dm_entry("dm_sent", "alice", f"user{i}", ts=old_ts)
            for i in range(5)
        ]
        entries.append(_make_dm_entry("dm_sent", "alice", "bob"))
        _write_activity_entries(anima_dir, entries)

        limiter = ConversationDepthLimiter(max_per_hour=10, max_per_day=5)
        assert limiter.check_global_outbound("alice", anima_dir) is True


class TestCurrentDepth:
    """Test current_depth reporting."""

    def test_zero_when_empty(self, tmp_path: Path, _patch_config):
        """ログなしで0を返す。"""
        anima_dir = tmp_path / "animas" / "alice"
        limiter = ConversationDepthLimiter(window_s=600, max_depth=6)
        assert limiter.current_depth("alice", "bob", anima_dir) == 0

    def test_counts_exchanges(self, tmp_path: Path, _patch_config):
        """交換数を正しくカウントする。"""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        entries = [
            _make_dm_entry("dm_sent", "alice", "bob"),
            _make_dm_entry("dm_received", "bob", "alice"),
        ]
        _write_activity_entries(anima_dir, entries)

        limiter = ConversationDepthLimiter(window_s=600, max_depth=6)
        assert limiter.current_depth("alice", "bob", anima_dir) == 2


class TestLegacyCheckAndRecord:
    """Legacy check_and_record is a no-op that always returns True."""

    def test_always_returns_true(self, _patch_config):
        """後方互換性スタブは常にTrueを返す。"""
        limiter = ConversationDepthLimiter(max_depth=1)
        assert limiter.check_and_record("alice", "bob") is True
        assert limiter.check_and_record("alice", "bob") is True
        assert limiter.check_and_record("alice", "bob") is True


class TestModuleSingleton:
    """Test the module-level singleton is shared."""

    def test_singleton_exists(self):
        """シングルトンが存在する。"""
        from core.cascade_limiter import depth_limiter
        assert isinstance(depth_limiter, ConversationDepthLimiter)
