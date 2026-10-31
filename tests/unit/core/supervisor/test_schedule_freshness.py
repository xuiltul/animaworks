"""Tests for stale schedule detection via mtime reconciliation."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.runtime.scheduler_manager import SchedulerManager


@pytest.fixture
def scheduler_mgr(tmp_path: Path) -> SchedulerManager:
    """Create a SchedulerManager with a temp anima dir."""
    anima = MagicMock()
    anima.memory.read_heartbeat_config.return_value = ""
    anima.memory.read_cron_config.return_value = ""
    mgr = SchedulerManager(
        anima=anima,
        anima_name="test",
        anima_dir=tmp_path,
        emit_event=MagicMock(),
    )
    return mgr


class TestRecordScheduleMtimes:
    def test_records_existing_files(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        (tmp_path / "cron.md").write_text("# test")
        (tmp_path / "HEARTBEAT.md").write_text("# test")
        scheduler_mgr._record_schedule_mtimes()
        assert scheduler_mgr._cron_md_mtime > 0
        assert scheduler_mgr._heartbeat_md_mtime > 0

    def test_records_zero_for_missing_files(self, scheduler_mgr: SchedulerManager) -> None:
        scheduler_mgr._record_schedule_mtimes()
        assert scheduler_mgr._cron_md_mtime == 0.0
        assert scheduler_mgr._heartbeat_md_mtime == 0.0


class TestCheckScheduleFreshness:
    def test_no_change_returns_false(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        (tmp_path / "cron.md").write_text("# v1")
        scheduler_mgr._record_schedule_mtimes()
        assert scheduler_mgr._check_schedule_freshness() is False

    def test_cron_change_triggers_reload(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        (tmp_path / "cron.md").write_text("# v1")
        scheduler_mgr._record_schedule_mtimes()

        # Simulate file modification (ensure mtime changes)
        time.sleep(0.05)
        (tmp_path / "cron.md").write_text("# v2")

        with patch.object(scheduler_mgr, "reload_schedule") as mock_reload:
            result = scheduler_mgr._check_schedule_freshness()
        assert result is True
        mock_reload.assert_called_once_with("test")

    def test_heartbeat_change_triggers_reload(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        (tmp_path / "HEARTBEAT.md").write_text("# v1")
        scheduler_mgr._record_schedule_mtimes()

        time.sleep(0.05)
        (tmp_path / "HEARTBEAT.md").write_text("# v2")

        with patch.object(scheduler_mgr, "reload_schedule") as mock_reload:
            result = scheduler_mgr._check_schedule_freshness()
        assert result is True
        mock_reload.assert_called_once()

    def test_deleted_cron_triggers_reload(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        (tmp_path / "cron.md").write_text("# v1")
        scheduler_mgr._record_schedule_mtimes()

        (tmp_path / "cron.md").unlink()

        with patch.object(scheduler_mgr, "reload_schedule") as mock_reload:
            result = scheduler_mgr._check_schedule_freshness()
        assert result is True
        mock_reload.assert_called_once()

    def test_missing_files_initially_no_reload(self, scheduler_mgr: SchedulerManager) -> None:
        """When files never existed, no reload needed."""
        scheduler_mgr._record_schedule_mtimes()
        assert scheduler_mgr._check_schedule_freshness() is False
