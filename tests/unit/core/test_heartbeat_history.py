# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for heartbeat history loading in core/anima.py.

Note: _save_heartbeat_history and _purge_old_heartbeat_logs were removed
as part of the unified activity log migration.  Writing is now handled by
ActivityLogger.  Only the read-side (_load_heartbeat_history) remains for
backward compatibility with existing heartbeat_history/ files.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────


def _make_digital_anima(anima_dir: Path, shared_dir: Path):
    """Create a DigitalAnima with all heavy deps mocked."""
    with patch("core.anima.AgentCore"), \
         patch("core.anima.MemoryManager") as MockMM, \
         patch("core.anima.Messenger"):
        MockMM.return_value.read_model_config.return_value = MagicMock()
        from core.anima import DigitalAnima
        return DigitalAnima(anima_dir, shared_dir)


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "alice"
    d.mkdir(parents=True)
    (d / "identity.md").write_text("# Alice", encoding="utf-8")
    return d


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def dp(anima_dir: Path, shared_dir: Path):
    """A DigitalAnima instance with mocked dependencies."""
    return _make_digital_anima(anima_dir, shared_dir)


# ── TestHeartbeatHistoryLoad ─────────────────────────────


class TestHeartbeatHistoryLoad:
    """Tests for _load_heartbeat_history (reads from unified activity log)."""

    def _write_activity_entries(self, anima_dir, entries_by_date):
        """Write activity log entries in the unified format.

        entries_by_date: dict mapping date string to list of entry dicts.
        """
        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        for date_str, entries in entries_by_date.items():
            lines = [json.dumps(e, ensure_ascii=False) for e in entries]
            (log_dir / f"{date_str}.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8",
            )

    def test_load_from_activity_log(self, dp, anima_dir):
        """Loading reads heartbeat_end entries from unified activity log."""
        today = date.today().isoformat()
        entries = []
        for i in range(5):
            entries.append({
                "ts": f"{today}T{10 + i:02d}:00:00",
                "type": "heartbeat_end",
                "summary": f"Entry {i}",
            })
        self._write_activity_entries(anima_dir, {today: entries})

        text = dp._load_heartbeat_history()
        assert text != ""
        # Should contain at most _HEARTBEAT_HISTORY_N entries (default 3)
        lines = text.strip().splitlines()
        assert len(lines) == dp._HEARTBEAT_HISTORY_N
        # Should be the last N entries
        assert "Entry 4" in lines[-1]
        assert "Entry 3" in lines[-2]
        assert "Entry 2" in lines[-3]

    def test_load_ignores_non_heartbeat_entries(self, dp, anima_dir):
        """Loading only picks up heartbeat_end type entries."""
        today = date.today().isoformat()
        entries = [
            {"ts": f"{today}T08:00:00", "type": "message_received", "summary": "A message"},
            {"ts": f"{today}T09:00:00", "type": "heartbeat_end", "summary": "Heartbeat result"},
        ]
        self._write_activity_entries(anima_dir, {today: entries})

        text = dp._load_heartbeat_history()
        assert text != ""
        assert "Heartbeat result" in text
        assert "A message" not in text

    def test_load_returns_empty_when_no_files(self, dp, anima_dir):
        """Loading returns empty string when no activity log or legacy files exist."""
        # With legacy fallback, load_recent_heartbeat_summary is called when
        # activity log is empty. Ensure it returns "" for the empty case.
        dp.memory.load_recent_heartbeat_summary.return_value = ""
        text = dp._load_heartbeat_history()
        assert text == ""

    def test_load_across_multiple_days(self, dp, anima_dir):
        """Loading reads entries from multiple recent day files."""
        entries_by_date = {}
        for days_ago in range(2):
            file_date = date.today() - timedelta(days=days_ago)
            date_str = file_date.isoformat()
            entries_by_date[date_str] = [{
                "ts": f"{date_str}T10:00:00",
                "type": "heartbeat_end",
                "summary": f"Day {days_ago} ago",
            }]
        self._write_activity_entries(anima_dir, entries_by_date)

        text = dp._load_heartbeat_history()
        assert text != ""
        lines = text.strip().splitlines()
        assert len(lines) == 2
