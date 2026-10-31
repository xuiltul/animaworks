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
from unittest.mock import patch

import pytest


class TestLifecycleConsolidationIntegration:
    """Test lifecycle consolidation setup."""

    @pytest.mark.asyncio
    async def test_system_crons_registered_on_supervisor(self, tmp_path: Path):
        """System-wide crons are registered by ProcessSupervisor."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        from core.time_utils import get_app_timezone
        from server.supervisor.manager import ProcessSupervisor

        supervisor = ProcessSupervisor(
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
            run_dir=tmp_path / "run",
        )
        supervisor.scheduler = AsyncIOScheduler(timezone=get_app_timezone())
        from core.config.models import AnimaWorksConfig

        config = AnimaWorksConfig()
        config.consolidation.weekly_enabled = True
        with patch("core.config.load_config", return_value=config):
            supervisor._setup_system_crons()

        jobs = supervisor.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert "system_daily_consolidation" in job_ids
        assert "system_weekly_integration" in job_ids

        daily_job = next(j for j in jobs if j.id == "system_daily_consolidation")
        assert daily_job.name == "System: Daily Consolidation"

        weekly_job = next(j for j in jobs if j.id == "system_weekly_integration")
        assert weekly_job.name == "System: Weekly Integration"

def test_resolve_post_processing_cooldown_seconds() -> None:
    from core.lifecycle.system_consolidation import resolve_post_processing_cooldown_seconds

    assert resolve_post_processing_cooldown_seconds(SimpleNamespace(post_processing_cooldown_seconds=12)) == 12.0
    assert resolve_post_processing_cooldown_seconds(SimpleNamespace(post_processing_cooldown_seconds=-1)) == 0.0
    assert resolve_post_processing_cooldown_seconds(SimpleNamespace(post_processing_cooldown_seconds="bad")) == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
