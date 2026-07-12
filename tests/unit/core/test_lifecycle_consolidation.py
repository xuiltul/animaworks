from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Integration tests for lifecycle consolidation setup."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLifecycleConsolidationIntegration:
    """Test lifecycle consolidation setup."""

    @pytest.mark.asyncio
    async def test_system_crons_registered_on_supervisor(self, tmp_path: Path):
        """System-wide crons are registered by ProcessSupervisor."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        from core.supervisor.manager import ProcessSupervisor
        from core.time_utils import get_app_timezone

        supervisor = ProcessSupervisor(
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
            run_dir=tmp_path / "run",
        )
        supervisor.scheduler = AsyncIOScheduler(timezone=get_app_timezone())
        supervisor._setup_system_crons()

        jobs = supervisor.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert "system_daily_consolidation" in job_ids
        assert "system_weekly_integration" in job_ids

        daily_job = next(j for j in jobs if j.id == "system_daily_consolidation")
        assert daily_job.name == "System: Daily Consolidation"

        weekly_job = next(j for j in jobs if j.id == "system_weekly_integration")
        assert weekly_job.name == "System: Weekly Integration"

    @pytest.mark.asyncio
    async def test_handle_daily_consolidation_with_config(self, tmp_path: Path):
        """Test daily consolidation handler respects config settings."""
        from core.anima import DigitalAnima
        from core.lifecycle import LifecycleManager

        manager = LifecycleManager()

        # Create mock anima
        anima_dir = tmp_path / "test_anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Test Anima", encoding="utf-8")

        # Mock DigitalAnima
        mock_anima = MagicMock(spec=DigitalAnima)
        mock_anima.name = "test_anima"
        mock_anima.memory = MagicMock()
        mock_anima.memory.anima_dir = anima_dir

        manager.animas["test_anima"] = mock_anima

        # Mock config to disable consolidation
        mock_config = MagicMock()
        mock_consolidation_cfg = MagicMock()
        mock_consolidation_cfg.daily_enabled = False
        mock_config.consolidation = mock_consolidation_cfg

        with patch("core.config.load_config", return_value=mock_config):
            await manager._handle_daily_consolidation()

        # Consolidation should be skipped when disabled
        # (No exception should be raised)

    @pytest.mark.asyncio
    async def test_handle_daily_consolidation_with_anima(self, tmp_path: Path):
        """Test daily consolidation runs for registered anima via run_consolidation."""
        from core.anima import DigitalAnima
        from core.lifecycle import LifecycleManager

        manager = LifecycleManager()

        # Create anima directory structure
        anima_dir = tmp_path / "test_anima"
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True)

        # Mock anima with run_consolidation() (new Anima-driven flow)
        mock_anima = MagicMock(spec=DigitalAnima)
        mock_anima.name = "test_anima"
        mock_anima.memory = MagicMock()
        mock_anima.memory.anima_dir = anima_dir
        # run_consolidation side-effect: simulate knowledge file creation
        mock_result = MagicMock()
        mock_result.duration_ms = 1234
        mock_result.summary = "Consolidated 3 episodes"

        async def _fake_run_consolidation(**kwargs):
            # Simulate the Anima writing a knowledge file during consolidation
            kf = knowledge_dir / "test-knowledge.md"
            kf.write_text(
                "---\nauto_consolidated: true\n---\n# Test Knowledge\n\nTest content from consolidation\n",
                encoding="utf-8",
            )
            return mock_result

        mock_anima.run_consolidation = AsyncMock(side_effect=_fake_run_consolidation)

        manager.animas["test_anima"] = mock_anima

        # Mock config to enable consolidation
        mock_config = MagicMock()
        mock_consolidation_cfg = MagicMock()
        mock_consolidation_cfg.daily_enabled = True
        mock_consolidation_cfg.min_episodes_threshold = 1
        mock_consolidation_cfg.max_turns = 30
        mock_config.consolidation = mock_consolidation_cfg

        gate = SimpleNamespace(
            should_run=True,
            activity_count=3,
            episode_count=0,
            carryover_count=0,
            threshold=1,
        )

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.lifecycle.system_consolidation.should_skip_inactive_consolidation", return_value=False),
            patch("core.lifecycle.system_consolidation.evaluate_daily_consolidation_gate", return_value=gate),
            patch("core.lifecycle.system_consolidation.run_daily_consolidation_post_processing", AsyncMock()),
        ):
            await manager._handle_daily_consolidation()

        # Verify anima.run_consolidation was called
        mock_anima.run_consolidation.assert_called_once()

        # Verify knowledge file was created by the (mocked) Anima
        knowledge_file = knowledge_dir / "test-knowledge.md"
        assert knowledge_file.exists()
        content = knowledge_file.read_text(encoding="utf-8")
        assert "Test Knowledge" in content
        assert "auto_consolidated: true" in content


def test_resolve_post_processing_cooldown_seconds() -> None:
    from core.lifecycle.system_consolidation import resolve_post_processing_cooldown_seconds

    assert resolve_post_processing_cooldown_seconds(SimpleNamespace(post_processing_cooldown_seconds=12)) == 12.0
    assert resolve_post_processing_cooldown_seconds(SimpleNamespace(post_processing_cooldown_seconds=-1)) == 0.0
    assert resolve_post_processing_cooldown_seconds(SimpleNamespace(post_processing_cooldown_seconds="bad")) == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
