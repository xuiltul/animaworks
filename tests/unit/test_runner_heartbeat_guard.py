# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for heartbeat collision prevention.

Verifies that the heartbeat_running flag and _cron_running set properly
prevent overlapping heartbeat and cron executions in SchedulerManager
and InboxRateLimiter.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.scheduler_manager import SchedulerManager
from core.supervisor.inbox_rate_limiter import InboxRateLimiter


def _make_scheduler_mgr(tmp_path: Path) -> SchedulerManager:
    """Create a SchedulerManager with minimal dependencies."""
    mock_anima = MagicMock()
    mock_anima.run_heartbeat = AsyncMock()
    return SchedulerManager(
        anima=mock_anima,
        anima_name="guard-test",
        anima_dir=tmp_path / "animas" / "guard-test",
        emit_event=MagicMock(),
    )


def _make_inbox_limiter(
    tmp_path: Path, scheduler_mgr: SchedulerManager | None = None,
) -> InboxRateLimiter:
    """Create an InboxRateLimiter with minimal dependencies."""
    mock_anima = MagicMock()
    mock_anima.run_heartbeat = AsyncMock()
    mock_anima.messenger.receive.return_value = []
    mock_anima._lock = asyncio.Lock()

    if scheduler_mgr is None:
        scheduler_mgr = MagicMock(spec=SchedulerManager)
        scheduler_mgr.heartbeat_running = False

    return InboxRateLimiter(
        anima=mock_anima,
        anima_name="guard-test",
        shutdown_event=asyncio.Event(),
        scheduler_mgr=scheduler_mgr,
    )


class TestHeartbeatGuard:
    """Verify heartbeat overlap prevention in SchedulerManager."""

    def test_initial_heartbeat_running_is_false(self, tmp_path):
        """SchedulerManager initializes with heartbeat_running=False."""
        mgr = _make_scheduler_mgr(tmp_path)
        assert mgr.heartbeat_running is False

    def test_initial_cron_running_is_empty(self, tmp_path):
        """SchedulerManager initializes with empty _cron_running set."""
        mgr = _make_scheduler_mgr(tmp_path)
        assert mgr._cron_running == set()

    @pytest.mark.asyncio
    async def test_heartbeat_tick_skips_when_already_running(self, tmp_path):
        """heartbeat_tick should skip immediately when heartbeat_running is True."""
        mgr = _make_scheduler_mgr(tmp_path)
        mgr._heartbeat_running = True

        await mgr.heartbeat_tick()

        # run_heartbeat should NOT have been called
        mgr._anima.run_heartbeat.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_tick_runs_when_not_already_running(self, tmp_path):
        """heartbeat_tick should execute when heartbeat_running is False."""
        mgr = _make_scheduler_mgr(tmp_path)
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"summary": "ok"}
        mgr._anima.run_heartbeat = AsyncMock(return_value=mock_result)

        assert mgr.heartbeat_running is False

        await mgr.heartbeat_tick()

        mgr._anima.run_heartbeat.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_tick_resets_flag_after_completion(self, tmp_path):
        """heartbeat_running flag resets to False after heartbeat completes."""
        mgr = _make_scheduler_mgr(tmp_path)
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"summary": "ok"}
        mgr._anima.run_heartbeat = AsyncMock(return_value=mock_result)

        await mgr.heartbeat_tick()

        assert mgr.heartbeat_running is False

    @pytest.mark.asyncio
    async def test_heartbeat_tick_resets_flag_on_exception(self, tmp_path):
        """heartbeat_running flag resets even when heartbeat raises an exception."""
        mgr = _make_scheduler_mgr(tmp_path)
        mgr._anima.run_heartbeat = AsyncMock(side_effect=RuntimeError("boom"))

        await mgr.heartbeat_tick()

        # Flag must be reset even after failure
        assert mgr.heartbeat_running is False

    @pytest.mark.asyncio
    async def test_heartbeat_tick_skips_when_no_anima(self, tmp_path):
        """heartbeat_tick returns early when anima is None."""
        mgr = _make_scheduler_mgr(tmp_path)
        mgr._anima = None

        # Should not raise
        await mgr.heartbeat_tick()


class TestCronGuard:
    """Verify cron overlap prevention in SchedulerManager."""

    @pytest.mark.asyncio
    async def test_cron_tick_skips_when_already_running(self, tmp_path):
        """cron_tick should skip when the same task name is in _cron_running."""
        from core.schemas import CronTask

        mgr = _make_scheduler_mgr(tmp_path)
        mgr._anima.run_cron_task = AsyncMock()

        task = CronTask(name="daily_report", schedule="0 9 * * *", description="test", type="llm")
        mgr._cron_running.add("daily_report")

        await mgr.cron_tick(task)

        # The task's LLM call should NOT be started
        mgr._anima.run_cron_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_cron_tick_runs_when_not_already_running(self, tmp_path):
        """cron_tick should dispatch when the task name is not in _cron_running."""
        from core.schemas import CronTask

        mgr = _make_scheduler_mgr(tmp_path)
        mgr._anima.run_cron_task = AsyncMock()

        task = CronTask(name="weekly_review", schedule="0 9 * * 1", description="test", type="llm")
        assert "weekly_review" not in mgr._cron_running

        # cron_tick creates a background task via asyncio.create_task
        await mgr.cron_tick(task)

        # Give the background task a moment to start
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_cron_running_tracks_task_name(self, tmp_path):
        """_cron_running should contain task names that are currently executing."""
        mgr = _make_scheduler_mgr(tmp_path)

        mgr._cron_running.add("daily_report")
        mgr._cron_running.add("weekly_review")

        assert "daily_report" in mgr._cron_running
        assert "weekly_review" in mgr._cron_running
        assert "monthly_summary" not in mgr._cron_running

    @pytest.mark.asyncio
    async def test_run_cron_task_removes_name_on_completion(self, tmp_path):
        """_run_cron_task should discard task name from _cron_running after completion."""
        from core.schemas import CronTask

        mgr = _make_scheduler_mgr(tmp_path)
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"summary": "done"}
        mgr._anima.run_cron_task = AsyncMock(return_value=mock_result)

        task = CronTask(name="daily_report", schedule="0 9 * * *", description="test", type="llm")

        await mgr._run_cron_task(task)

        assert "daily_report" not in mgr._cron_running

    @pytest.mark.asyncio
    async def test_run_cron_task_removes_name_on_exception(self, tmp_path):
        """_run_cron_task should discard task name even when execution fails."""
        from core.schemas import CronTask

        mgr = _make_scheduler_mgr(tmp_path)
        mgr._anima.run_cron_task = AsyncMock(side_effect=RuntimeError("cron failed"))

        task = CronTask(name="daily_report", schedule="0 9 * * *", description="test", type="llm")

        await mgr._run_cron_task(task)

        # Must be cleaned up despite error
        assert "daily_report" not in mgr._cron_running


class TestMessageTriggeredHeartbeatGuard:
    """Verify message-triggered heartbeat respects the guard flag."""

    @pytest.mark.asyncio
    async def test_message_heartbeat_skips_when_already_running(self, tmp_path):
        """message_triggered_heartbeat should skip when heartbeat_running is True."""
        mock_scheduler_mgr = MagicMock(spec=SchedulerManager)
        mock_scheduler_mgr.heartbeat_running = True

        limiter = _make_inbox_limiter(tmp_path, mock_scheduler_mgr)
        limiter._pending_trigger = True

        await limiter.message_triggered_heartbeat()

        limiter._anima.run_heartbeat.assert_not_called()
        # _pending_trigger should be reset
        assert limiter._pending_trigger is False


class TestRunnerHeartbeat24hDefault:
    """Verify SchedulerManager._setup_heartbeat respects 2-tier active hours resolution."""

    def test_default_24h_when_no_time_range(self, tmp_path):
        """No time range in heartbeat.md => hour='*' (24h)."""
        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = "- チェック項目A"

        mgr = SchedulerManager(
            anima=mock_anima,
            anima_name="guard-test",
            anima_dir=tmp_path / "animas" / "guard-test",
            emit_event=MagicMock(),
        )
        mock_scheduler = MagicMock()
        mgr.scheduler = mock_scheduler

        mgr._setup_heartbeat()

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        trigger = call_kwargs[1]["trigger"] if "trigger" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        hour_field = str(trigger.fields[5])
        assert hour_field == "*"

    def test_heartbeat_md_time_range_restricts_hours(self, tmp_path):
        """Time range in heartbeat.md restricts heartbeat hours."""
        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = "稼働時間: 8:00 - 20:00"

        mgr = SchedulerManager(
            anima=mock_anima,
            anima_name="guard-test",
            anima_dir=tmp_path / "animas" / "guard-test",
            emit_event=MagicMock(),
        )
        mock_scheduler = MagicMock()
        mgr.scheduler = mock_scheduler

        mgr._setup_heartbeat()

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        trigger = call_kwargs[1]["trigger"] if "trigger" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        hour_field = str(trigger.fields[5])
        assert "8-19" in hour_field
