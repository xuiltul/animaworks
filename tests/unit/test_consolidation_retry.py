"""Unit tests for consolidation retry mechanism."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_lifecycle(tmp_path: Path):
    """Create a minimal LifecycleManager with mocked dependencies."""
    from core.lifecycle import LifecycleManager

    mgr = LifecycleManager.__new__(LifecycleManager)
    mgr.scheduler = MagicMock()
    mgr.animas = {}
    mgr._ws_broadcast = None
    return mgr


class TestScheduleConsolidationRetry:
    """Tests for _schedule_consolidation_retry."""

    @patch("core.time_utils.now_local")
    def test_schedules_one_shot_job(self, mock_now, tmp_path):
        from datetime import datetime

        mock_now.return_value = datetime(2026, 3, 12, 2, 0, 0, tzinfo=UTC)
        mgr = _make_lifecycle(tmp_path)

        mgr._schedule_consolidation_retry("kotoha", max_turns=30)

        mgr.scheduler.add_job.assert_called_once()
        call_kwargs = mgr.scheduler.add_job.call_args
        assert call_kwargs.kwargs["kwargs"]["anima_name"] == "kotoha"
        assert call_kwargs.kwargs["kwargs"]["max_turns"] == 30
        assert call_kwargs.kwargs["id"] == "consolidation_retry_kotoha"
        assert call_kwargs.kwargs["replace_existing"] is True

    @patch("core.time_utils.now_local")
    def test_replace_existing_prevents_duplicates(self, mock_now, tmp_path):
        from datetime import datetime

        mock_now.return_value = datetime(2026, 3, 12, 2, 0, 0, tzinfo=UTC)
        mgr = _make_lifecycle(tmp_path)

        mgr._schedule_consolidation_retry("kotoha", max_turns=30)
        mgr._schedule_consolidation_retry("kotoha", max_turns=30)

        assert mgr.scheduler.add_job.call_count == 2
        for call in mgr.scheduler.add_job.call_args_list:
            assert call.kwargs["replace_existing"] is True


class TestRunConsolidationRetry:
    """Tests for _run_consolidation_retry."""

    @pytest.mark.asyncio
    async def test_retry_success(self, tmp_path):
        mgr = _make_lifecycle(tmp_path)

        mock_anima = MagicMock()
        mock_result = MagicMock()
        mock_result.duration_ms = 25000
        mock_anima.run_consolidation = AsyncMock(return_value=mock_result)
        mock_anima.memory.anima_dir = tmp_path / "animas" / "kotoha"
        mgr.animas["kotoha"] = mock_anima

        with patch("core.memory.consolidation.ConsolidationEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine
            await mgr._run_consolidation_retry("kotoha", max_turns=30)

        mock_anima.run_consolidation.assert_awaited_once_with(
            consolidation_type="daily",
            max_turns=30,
        )

    @pytest.mark.asyncio
    async def test_retry_skips_missing_anima(self, tmp_path):
        mgr = _make_lifecycle(tmp_path)
        await mgr._run_consolidation_retry("nonexistent", max_turns=30)

    @pytest.mark.asyncio
    async def test_retry_failure_does_not_raise(self, tmp_path):
        mgr = _make_lifecycle(tmp_path)

        mock_anima = MagicMock()
        mock_anima.run_consolidation = AsyncMock(side_effect=RuntimeError("still limited"))
        mgr.animas["kotoha"] = mock_anima

        await mgr._run_consolidation_retry("kotoha", max_turns=30)


class TestDailyConsolidationRetryTrigger:
    """Tests for retry triggering within _handle_daily_consolidation."""

    @staticmethod
    def _passing_gate() -> SimpleNamespace:
        return SimpleNamespace(
            should_run=True,
            activity_count=5,
            episode_count=0,
            carryover_count=0,
            threshold=1,
        )

    @pytest.mark.asyncio
    async def test_short_duration_triggers_retry(self, tmp_path):
        mgr = _make_lifecycle(tmp_path)

        mock_anima = MagicMock()
        mock_result = MagicMock()
        mock_result.duration_ms = 4600  # < 10_000 threshold
        mock_result.summary = "You've hit your limit"
        mock_anima.run_consolidation = AsyncMock(return_value=mock_result)
        mock_anima.memory.anima_dir = tmp_path / "animas" / "kotoha"
        mgr.animas["kotoha"] = mock_anima
        mgr._schedule_consolidation_retry = MagicMock()

        with (
            patch("core.config.load_config") as mock_cfg,
            patch(
                "core.lifecycle.system_consolidation.evaluate_daily_consolidation_gate",
                return_value=self._passing_gate(),
            ),
            patch("core.lifecycle.system_consolidation.run_daily_consolidation_post_processing", AsyncMock()),
        ):
            mock_cfg.return_value = MagicMock(consolidation=None)
            await mgr._handle_daily_consolidation()

        mgr._schedule_consolidation_retry.assert_called_once_with("kotoha", 30)

    @pytest.mark.asyncio
    async def test_normal_duration_no_retry(self, tmp_path):
        mgr = _make_lifecycle(tmp_path)

        mock_anima = MagicMock()
        mock_result = MagicMock()
        mock_result.duration_ms = 45000  # Normal
        mock_result.summary = "Consolidation complete"
        mock_anima.run_consolidation = AsyncMock(return_value=mock_result)
        mock_anima.memory.anima_dir = tmp_path / "animas" / "test"
        mgr.animas["test"] = mock_anima
        mgr._schedule_consolidation_retry = MagicMock()

        with (
            patch("core.config.load_config") as mock_cfg,
            patch(
                "core.lifecycle.system_consolidation.evaluate_daily_consolidation_gate",
                return_value=self._passing_gate(),
            ),
            patch("core.lifecycle.system_consolidation.run_daily_consolidation_post_processing", AsyncMock()),
        ):
            mock_cfg.return_value = MagicMock(consolidation=None)
            await mgr._handle_daily_consolidation()

        mgr._schedule_consolidation_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_triggers_retry(self, tmp_path):
        mgr = _make_lifecycle(tmp_path)

        mock_anima = MagicMock()
        mock_anima.run_consolidation = AsyncMock(side_effect=RuntimeError("boom"))
        mock_anima.memory.anima_dir = tmp_path / "animas" / "kotoha"
        mgr.animas["kotoha"] = mock_anima
        mgr._schedule_consolidation_retry = MagicMock()

        with (
            patch("core.config.load_config") as mock_cfg,
            patch(
                "core.lifecycle.system_consolidation.evaluate_daily_consolidation_gate",
                return_value=self._passing_gate(),
            ),
            patch("core.lifecycle.system_consolidation.run_daily_consolidation_post_processing", AsyncMock()),
        ):
            mock_cfg.return_value = MagicMock(consolidation=None)
            await mgr._handle_daily_consolidation()

        mgr._schedule_consolidation_retry.assert_called_once_with("kotoha", 30)
