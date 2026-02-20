"""Unit tests for SchedulerManager (extracted from AnimaRunner)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.scheduler_manager import SchedulerManager


class TestSchedulerManagerSetup:
    """Tests for SchedulerManager initialization."""

    def _make_scheduler_mgr(self, tmp_path: Path, mock_anima: MagicMock | None = None):
        if mock_anima is None:
            mock_anima = MagicMock()
        return SchedulerManager(
            anima=mock_anima,
            anima_name="test-anima",
            anima_dir=tmp_path / "animas" / "test-anima",
            emit_event=MagicMock(),
        )

    def test_scheduler_initially_none(self, tmp_path):
        mgr = self._make_scheduler_mgr(tmp_path)
        assert mgr.scheduler is None

    @pytest.mark.asyncio
    async def test_setup_creates_scheduler(self, tmp_path):
        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = "5分ごと\n8:00 - 23:00"
        mock_anima.memory.read_cron_config.return_value = ""
        mock_anima.set_on_schedule_changed = MagicMock()
        mgr = self._make_scheduler_mgr(tmp_path, mock_anima)

        mgr.setup()

        assert mgr.scheduler is not None
        assert mgr.scheduler.running
        # Should have heartbeat job registered
        jobs = mgr.scheduler.get_jobs()
        assert len(jobs) >= 1
        assert any("heartbeat" in j.id for j in jobs)

        mgr.shutdown()

    @pytest.mark.asyncio
    async def test_setup_with_cron_tasks(self, tmp_path):
        mock_anima = MagicMock()
        mock_anima.name = "test-anima"
        mock_anima.memory.read_heartbeat_config.return_value = "30分ごと\n9:00 - 22:00"
        mock_anima.memory.read_cron_config.return_value = """## Task A
schedule: 0 9 * * *
type: llm
Do something
"""
        mock_anima.set_on_schedule_changed = MagicMock()
        mgr = self._make_scheduler_mgr(tmp_path, mock_anima)

        mgr.setup()

        jobs = mgr.scheduler.get_jobs()
        job_ids = [j.id for j in jobs]
        assert "test-anima_heartbeat" in job_ids
        assert "test-anima_cron_0" in job_ids

        mgr.shutdown()

    @pytest.mark.asyncio
    async def test_setup_no_heartbeat_config(self, tmp_path):
        """Scheduler should start even without heartbeat.md content."""
        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = ""
        mock_anima.memory.read_cron_config.return_value = ""
        mock_anima.set_on_schedule_changed = MagicMock()
        mgr = self._make_scheduler_mgr(tmp_path, mock_anima)

        mgr.setup()

        assert mgr.scheduler is not None
        assert mgr.scheduler.running
        # No jobs since no config
        assert len(mgr.scheduler.get_jobs()) == 0

        mgr.shutdown()

    @pytest.mark.asyncio
    async def test_setup_handles_error_gracefully(self, tmp_path):
        """Scheduler setup should not crash on errors."""
        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.side_effect = Exception("file error")
        mgr = self._make_scheduler_mgr(tmp_path, mock_anima)

        # Should not raise
        mgr.setup()
        # Scheduler should be None after error
        assert mgr.scheduler is None

    @pytest.mark.asyncio
    async def test_reload_schedule(self, tmp_path):
        """Test hot-reload of schedule from disk."""
        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = "5分ごと\n8:00 - 23:00"
        mock_anima.memory.read_cron_config.return_value = ""
        mock_anima.set_on_schedule_changed = MagicMock()
        mgr = self._make_scheduler_mgr(tmp_path, mock_anima)

        mgr.setup()
        initial_jobs = len(mgr.scheduler.get_jobs())
        assert initial_jobs == 1  # heartbeat only

        # Now add cron config for reload
        mock_anima.memory.read_cron_config.return_value = """## New Task
schedule: 0 10 * * *
type: llm
New task description
"""
        result = mgr.reload_schedule("test-anima")
        assert result["removed"] == 1
        assert len(result["new_jobs"]) == 2  # heartbeat + cron

        mgr.shutdown()

    @pytest.mark.asyncio
    async def test_get_status_includes_scheduler_info(self, tmp_path):
        """AnimaRunner.get_status should include scheduler_running and scheduler_jobs."""
        from core.supervisor.runner import AnimaRunner

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(exist_ok=True)
        anima_dir = animas_dir / "test-anima"
        anima_dir.mkdir(exist_ok=True)
        (anima_dir / "identity.md").write_text("test")
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(exist_ok=True)
        socket_path = tmp_path / "test.sock"

        runner = AnimaRunner(
            anima_name="test-anima",
            socket_path=socket_path,
            animas_dir=animas_dir,
            shared_dir=shared_dir,
        )

        mock_anima = MagicMock()
        mock_anima._status = "idle"
        mock_anima._current_task = None
        mock_anima.needs_bootstrap = False
        mock_anima.memory.read_heartbeat_config.return_value = "5分ごと\n8:00 - 23:00"
        mock_anima.memory.read_cron_config.return_value = ""
        mock_anima.set_on_schedule_changed = MagicMock()
        runner.anima = mock_anima

        # Create and setup SchedulerManager
        runner._scheduler_mgr = SchedulerManager(
            anima=mock_anima,
            anima_name="test-anima",
            anima_dir=anima_dir,
            emit_event=runner._emit_event,
        )
        runner._scheduler_mgr.setup()

        result = await runner._handle_get_status({})
        assert result["scheduler_running"] is True
        assert result["scheduler_jobs"] == 1

        runner._scheduler_mgr.shutdown()


class TestSchedulerManagerCleanup:
    """Tests for scheduler cleanup on shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_scheduler(self, tmp_path):
        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = "5分ごと\n8:00 - 23:00"
        mock_anima.memory.read_cron_config.return_value = ""
        mock_anima.set_on_schedule_changed = MagicMock()

        mgr = SchedulerManager(
            anima=mock_anima,
            anima_name="test-anima",
            anima_dir=tmp_path / "animas" / "test-anima",
            emit_event=MagicMock(),
        )
        mgr.setup()
        assert mgr.scheduler.running

        mgr.shutdown()
        # AsyncIOScheduler.shutdown(wait=False) defers actual stop to the
        # event loop; yield control so the scheduler can finalize.
        await asyncio.sleep(0.1)
        assert not mgr.scheduler.running
