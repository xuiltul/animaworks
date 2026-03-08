# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for catch-up logic in SchedulerMixin.

Verifies that missed daily/weekly/monthly jobs are detected on startup
and that marker files are read/written correctly.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from core.supervisor._mgr_scheduler import (
    _marker_dir,
    _read_marker,
    _write_marker,
)

# Fixed TZ for deterministic tests (production uses get_app_timezone())
_TEST_TZ = ZoneInfo("Asia/Tokyo")

# ── Marker helpers ──────────────────────────────────────────────────


class TestMarkerHelpers:
    def test_write_and_read_marker(self, tmp_path: Path) -> None:
        marker = tmp_path / "test_marker"
        ts = datetime(2026, 3, 4, 2, 0, tzinfo=_TEST_TZ)
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
        assert (datetime.now(_TEST_TZ) - result).total_seconds() < 5

    def test_marker_dir_creates_directory(self, tmp_path: Path) -> None:
        d = _marker_dir(tmp_path / "data")
        assert d.exists()
        assert d.name == "run"


# ── Catch-up detection logic ────────────────────────────────────────


class TestCatchupDetection:
    """Test the time-delta thresholds used in _catchup_missed_jobs."""

    def test_daily_missed_when_marker_old(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_daily_consolidation"
        _write_marker(marker, datetime.now(_TEST_TZ) - timedelta(hours=40))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_TEST_TZ) - last) > timedelta(hours=36)

    def test_daily_not_missed_when_recent(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_daily_consolidation"
        _write_marker(marker, datetime.now(_TEST_TZ) - timedelta(hours=10))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_TEST_TZ) - last) <= timedelta(hours=36)

    def test_weekly_missed_when_marker_old(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_weekly_integration"
        _write_marker(marker, datetime.now(_TEST_TZ) - timedelta(days=10))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_TEST_TZ) - last) > timedelta(days=9)

    def test_weekly_not_missed_when_recent(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_weekly_integration"
        _write_marker(marker, datetime.now(_TEST_TZ) - timedelta(days=5))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_TEST_TZ) - last) <= timedelta(days=9)

    def test_monthly_missed_when_no_marker(self, tmp_path: Path) -> None:
        last = _read_marker(tmp_path / "last_monthly_forgetting")
        assert last is None

    def test_monthly_not_missed_when_recent(self, tmp_path: Path) -> None:
        marker = tmp_path / "last_monthly_forgetting"
        _write_marker(marker, datetime.now(_TEST_TZ) - timedelta(days=20))
        last = _read_marker(marker)
        assert last is not None
        assert (datetime.now(_TEST_TZ) - last) <= timedelta(days=35)


# ── Catch-up execution ──────────────────────────────────────────────


def _make_supervisor(tmp_path: Path):
    from core.supervisor.manager import ProcessSupervisor

    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
    )


def _create_anima_dir(animas_dir: Path, name: str, *, enabled: bool = True) -> None:
    d = animas_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.md").write_text(f"# {name}", encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"enabled": enabled}), encoding="utf-8")


@pytest.mark.asyncio
async def test_catchup_daily_indexing_when_missed(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    _create_anima_dir(sup.animas_dir, "sakura")

    mdir = _marker_dir(sup._get_data_dir())
    _write_marker(mdir / "last_daily_consolidation", datetime.now(_TEST_TZ) - timedelta(hours=1))
    _write_marker(mdir / "last_weekly_integration", datetime.now(_TEST_TZ) - timedelta(days=1))
    _write_marker(mdir / "last_monthly_forgetting", datetime.now(_TEST_TZ) - timedelta(days=1))
    _write_marker(mdir / "last_housekeeping", datetime.now(_TEST_TZ) - timedelta(hours=1))
    _write_marker(mdir / "last_daily_indexing", datetime.now(_TEST_TZ) - timedelta(hours=40))

    with (
        patch.object(sup, "_run_daily_indexing", new_callable=AsyncMock) as mock_indexing,
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "core.config.load_config",
            return_value=type(
                "Cfg",
                (),
                {
                    "consolidation": type(
                        "CC",
                        (),
                        {
                            "daily_enabled": True,
                            "weekly_enabled": True,
                            "monthly_enabled": True,
                            "indexing_enabled": True,
                        },
                    )()
                },
            )(),
        ),
    ):
        await sup._catchup_missed_jobs()

    mock_indexing.assert_called_once()
