# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for catch-up logic in SchedulerMixin.

Verifies that missed daily/weekly/monthly jobs are detected on startup
and that marker files are read/written correctly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.supervisor._mgr_scheduler import (
    _JST,
    _marker_dir,
    _read_marker,
    _write_marker,
)


# ── Marker helpers ──────────────────────────────────────────────────


class TestMarkerHelpers:

    def test_write_and_read_marker(self, tmp_path: Path) -> None:
        marker = tmp_path / "test_marker"
        ts = datetime(2026, 3, 4, 2, 0, tzinfo=_JST)
        _write_marker(marker, ts)

        result = _read_marker(marker)
        assert result == ts

    def test_read_marker_missing_file(self, tmp_path: Path) -> None:
        result = _read_marker(tmp_path / "nonexistent")
        assert result is None

    def test_read_marker_corrupt_file(self, tmp_path: Path) -> None:
        marker = tmp_path / "bad"
        marker.write_text("not-a-date")
        result = _read_marker(marker)
        assert result is None

    def test_write_marker_default_timestamp(self, tmp_path: Path) -> None:
        marker = tmp_path / "default_ts"
        _write_marker(marker)
        result = _read_marker(marker)
        assert result is not None
        assert (datetime.now(_JST) - result).total_seconds() < 5

    def test_marker_dir_creates_directory(self, tmp_path: Path) -> None:
        d = _marker_dir(tmp_path / "data")
        assert d.exists()
        assert d.name == "run"


# ── Catch-up detection logic ────────────────────────────────────────


class TestCatchupDetection:
    """Test the time-delta thresholds used in _catchup_missed_jobs."""

    def test_daily_missed_when_marker_old(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_daily_consolidation"
        _write_marker(marker, datetime.now(_JST) - timedelta(hours=40))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_JST) - last) > timedelta(hours=36)

    def test_daily_not_missed_when_recent(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_daily_consolidation"
        _write_marker(marker, datetime.now(_JST) - timedelta(hours=10))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_JST) - last) <= timedelta(hours=36)

    def test_weekly_missed_when_marker_old(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_weekly_integration"
        _write_marker(marker, datetime.now(_JST) - timedelta(days=10))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_JST) - last) > timedelta(days=9)

    def test_weekly_not_missed_when_recent(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_weekly_integration"
        _write_marker(marker, datetime.now(_JST) - timedelta(days=5))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_JST) - last) <= timedelta(days=9)

    def test_monthly_missed_when_no_marker(self, tmp_path: Path) -> None:
        last = _read_marker(tmp_path / "last_monthly_forgetting")
        assert last is None

    def test_monthly_not_missed_when_recent(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_monthly_forgetting"
        _write_marker(marker, datetime.now(_JST) - timedelta(days=20))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_JST) - last) <= timedelta(days=35)
