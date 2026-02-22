"""Unit tests for ProcessSupervisor system cron scheduling."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSupervisorSchedulerInit:
    """Tests for ProcessSupervisor scheduler initialization."""

    def _make_supervisor(self, tmp_path: Path):
        from core.supervisor.manager import ProcessSupervisor

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(exist_ok=True)
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(exist_ok=True)
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)

        return ProcessSupervisor(
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            run_dir=run_dir,
            log_dir=log_dir,
        )

    def test_scheduler_initially_none(self, tmp_path):
        sup = self._make_supervisor(tmp_path)
        assert sup.scheduler is None
        assert sup.is_scheduler_running() is False

    @pytest.mark.asyncio
    async def test_start_system_scheduler(self, tmp_path):
        sup = self._make_supervisor(tmp_path)

        mock_cfg = MagicMock()
        mock_cfg.consolidation = MagicMock(
            daily_enabled=True,
            daily_time="02:00",
            weekly_enabled=True,
            weekly_time="sun:03:00",
        )
        mock_load = MagicMock(return_value=mock_cfg)

        with patch.dict(
            "sys.modules",
            {"core.config": MagicMock(load_config=mock_load)},
        ):
            sup._start_system_scheduler()

        assert sup.scheduler is not None
        assert sup.scheduler.running
        assert sup.is_scheduler_running() is True

        jobs = sup.scheduler.get_jobs()
        job_ids = [j.id for j in jobs]
        assert "system_daily_consolidation" in job_ids
        assert "system_weekly_integration" in job_ids

        sup.scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_system_crons_disabled(self, tmp_path):
        sup = self._make_supervisor(tmp_path)

        mock_cfg = MagicMock()
        mock_cfg.consolidation = MagicMock(
            daily_enabled=False,
            weekly_enabled=False,
        )
        mock_load = MagicMock(return_value=mock_cfg)

        with patch.dict(
            "sys.modules",
            {"core.config": MagicMock(load_config=mock_load)},
        ):
            sup._start_system_scheduler()

        assert sup.scheduler.running
        jobs = sup.scheduler.get_jobs()
        job_ids = {j.id for j in jobs}
        # Monthly forgetting and activity log rotation are always present
        assert "system_monthly_forgetting" in job_ids
        assert "system_activity_log_rotation" in job_ids
        # Daily and weekly consolidation should be disabled
        assert "system_daily_consolidation" not in job_ids
        assert "system_weekly_integration" not in job_ids

        sup.scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_system_crons_config_error_handled(self, tmp_path):
        sup = self._make_supervisor(tmp_path)

        mock_mod = MagicMock()
        mock_mod.load_config.side_effect = Exception("config error")

        with patch.dict("sys.modules", {"core.config": mock_mod}):
            sup._start_system_scheduler()

        # Should still start scheduler even without config (uses defaults)
        assert sup.scheduler is not None
        assert sup.scheduler.running

        sup.scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_shutdown_stops_scheduler(self, tmp_path):
        sup = self._make_supervisor(tmp_path)

        mock_cfg = MagicMock()
        mock_cfg.consolidation = None
        mock_load = MagicMock(return_value=mock_cfg)

        with patch.dict(
            "sys.modules",
            {"core.config": MagicMock(load_config=mock_load)},
        ):
            sup._start_system_scheduler()

        assert sup.is_scheduler_running() is True
        await sup.shutdown_all()
        assert sup.is_scheduler_running() is False
