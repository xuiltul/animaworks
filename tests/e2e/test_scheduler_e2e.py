# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for scheduler integration.

Tests that AnimaRunner and ProcessSupervisor correctly set up schedulers
with real config files and APScheduler instances. Does NOT trigger actual
heartbeat/cron execution (would require LLM).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory with heartbeat and cron config."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    (tmp_path / "shared").mkdir()

    # Identity
    (anima_dir / "identity.md").write_text("# Test Anima\nA test anima.")

    # Heartbeat config
    (anima_dir / "heartbeat.md").write_text(
        "# Heartbeat: test-anima\n\n"
        "## 実行間隔\n10分ごと\n\n"
        "## 活動時間\n8:00 - 22:00（JST）\n\n"
        "## チェックリスト\n- Inboxをチェック\n"
    )

    # Cron config (schedule: must be a valid 5-field cron expression)
    (anima_dir / "cron.md").write_text(
        "## 毎朝の業務計画（毎日 9:00 JST）\n"
        "schedule: 0 9 * * *\n"
        "type: llm\n"
        "長期記憶から昨日の進捗を確認する。\n\n"
        "## 週次振り返り（毎週金曜 17:00 JST）\n"
        "schedule: 0 17 * * 5\n"
        "type: llm\n"
        "今週のepisodesを振り返る。\n"
    )

    return anima_dir


class TestAnimaRunnerSchedulerE2E:
    """E2E: AnimaRunner reads real config files and sets up APScheduler."""

    @pytest.mark.asyncio
    async def test_runner_starts_scheduler_with_real_config(self, anima_dir, tmp_path):
        """AnimaRunner should read heartbeat.md and cron.md and register jobs."""
        from core.supervisor.runner import AnimaRunner

        animas_dir = anima_dir.parent
        shared_dir = tmp_path / "shared"
        socket_path = tmp_path / "test.sock"

        runner = AnimaRunner(
            anima_name="test-anima",
            socket_path=socket_path,
            animas_dir=animas_dir,
            shared_dir=shared_dir,
        )

        # Create a mock DigitalAnima that uses real memory for config reading
        mock_anima = MagicMock()
        mock_anima.name = "test-anima"

        # Use real file reading for config
        mock_anima.memory.read_heartbeat_config.return_value = (
            anima_dir / "heartbeat.md"
        ).read_text()
        mock_anima.memory.read_cron_config.return_value = (
            anima_dir / "cron.md"
        ).read_text()
        mock_anima.set_on_schedule_changed = MagicMock()

        runner.anima = mock_anima
        runner._setup_scheduler()

        # Verify scheduler is running
        assert runner.scheduler is not None
        assert runner.scheduler.running

        # Verify jobs
        jobs = runner.scheduler.get_jobs()
        job_ids = [j.id for j in jobs]
        assert "test-anima_heartbeat" in job_ids
        assert "test-anima_cron_0" in job_ids
        assert "test-anima_cron_1" in job_ids
        assert len(jobs) == 3  # 1 heartbeat + 2 cron tasks

        # Verify heartbeat interval
        heartbeat_job = runner.scheduler.get_job("test-anima_heartbeat")
        assert heartbeat_job is not None

        runner.scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_runner_hot_reload_on_config_change(self, anima_dir, tmp_path):
        """After config change, reload_schedule should update jobs."""
        from core.supervisor.runner import AnimaRunner

        animas_dir = anima_dir.parent
        shared_dir = tmp_path / "shared"
        socket_path = tmp_path / "test.sock"

        runner = AnimaRunner(
            anima_name="test-anima",
            socket_path=socket_path,
            animas_dir=animas_dir,
            shared_dir=shared_dir,
        )

        mock_anima = MagicMock()
        mock_anima.name = "test-anima"
        mock_anima.memory.read_heartbeat_config.return_value = (
            anima_dir / "heartbeat.md"
        ).read_text()
        mock_anima.memory.read_cron_config.return_value = (
            anima_dir / "cron.md"
        ).read_text()
        mock_anima.set_on_schedule_changed = MagicMock()
        runner.anima = mock_anima

        runner._setup_scheduler()
        assert len(runner.scheduler.get_jobs()) == 3

        # Simulate adding a new cron task
        new_cron = (
            "## 毎朝の業務計画（毎日 9:00 JST）\n"
            "schedule: 0 9 * * *\n"
            "type: llm\n"
            "Description A\n\n"
            "## 週次振り返り（毎週金曜 17:00 JST）\n"
            "schedule: 0 17 * * 5\n"
            "type: llm\n"
            "Description B\n\n"
            "## 新しいタスク（毎日 18:00 JST）\n"
            "schedule: 0 18 * * *\n"
            "type: llm\n"
            "New task added\n"
        )
        mock_anima.memory.read_cron_config.return_value = new_cron

        result = runner._reload_schedule("test-anima")
        assert result["removed"] == 3  # old jobs removed
        assert len(result["new_jobs"]) == 4  # 1 heartbeat + 3 cron

        runner.scheduler.shutdown(wait=False)


class TestProcessSupervisorSystemCronE2E:
    """E2E: ProcessSupervisor sets up system cron with real config."""

    def _make_supervisor(self, tmp_path: Path):
        from core.supervisor.manager import ProcessSupervisor

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(exist_ok=True)
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(exist_ok=True)
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)

        return ProcessSupervisor(
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            run_dir=run_dir,
        )

    @pytest.mark.asyncio
    async def test_system_scheduler_starts_with_config(self, tmp_path):
        """System scheduler should register consolidation jobs."""
        sup = self._make_supervisor(tmp_path)

        with patch("core.config.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.consolidation = MagicMock(
                daily_enabled=True,
                daily_time="02:00",
                weekly_enabled=True,
                weekly_time="sun:03:00",
            )
            mock_config.return_value = mock_cfg

            sup._start_system_scheduler()

        assert sup.is_scheduler_running() is True
        jobs = sup.scheduler.get_jobs()
        assert len(jobs) == 3

        # Check job details
        daily = sup.scheduler.get_job("system_daily_consolidation")
        assert daily is not None
        assert "Daily" in daily.name

        weekly = sup.scheduler.get_job("system_weekly_integration")
        assert weekly is not None
        assert "Weekly" in weekly.name

        monthly = sup.scheduler.get_job("system_monthly_forgetting")
        assert monthly is not None
        assert "Monthly" in monthly.name

        sup.scheduler.shutdown(wait=False)


class TestSchedulerAPIE2E:
    """E2E: Test that API endpoints reflect actual scheduler state."""

    @pytest.mark.asyncio
    async def test_system_status_reflects_scheduler(self, tmp_path):
        """GET /api/system/status should show scheduler_running from supervisor."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from server.routes.system import create_system_router

        app = FastAPI()
        app.state.animas_dir = tmp_path / "animas"
        app.state.animas_dir.mkdir()
        app.state.shared_dir = tmp_path / "shared"
        app.state.anima_names = []

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.is_scheduler_running.return_value = True
        app.state.supervisor = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        app.state.ws_manager = ws_manager

        router = create_system_router()
        app.include_router(router, prefix="/api")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")

        data = resp.json()
        assert data["scheduler_running"] is True

    @pytest.mark.asyncio
    async def test_system_scheduler_shows_jobs(self, tmp_path):
        """GET /api/system/scheduler should show system and anima jobs."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from server.routes.system import create_system_router

        app = FastAPI()
        animas_dir = tmp_path / "animas" / "alice"
        animas_dir.mkdir(parents=True)
        (animas_dir / "cron.md").write_text(
            "## Task（毎日 9:00 JST）\ntype: llm\nDo thing\n"
        )
        app.state.animas_dir = tmp_path / "animas"
        app.state.shared_dir = tmp_path / "shared"
        app.state.anima_names = ["alice"]

        mock_scheduler = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "system_daily_consolidation"
        mock_job.name = "System: Daily Consolidation"
        mock_job.trigger = MagicMock(__str__=lambda s: "cron[hour='2', minute='0']")
        mock_job.next_run_time = None
        mock_scheduler.get_jobs.return_value = [mock_job]

        supervisor = MagicMock()
        supervisor.is_scheduler_running.return_value = True
        supervisor.scheduler = mock_scheduler
        app.state.supervisor = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        app.state.ws_manager = ws_manager

        router = create_system_router()
        app.include_router(router, prefix="/api")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")

        data = resp.json()
        assert data["running"] is True
        assert len(data["system_jobs"]) == 1
        assert data["system_jobs"][0]["id"] == "system_daily_consolidation"
        assert len(data["anima_jobs"]) >= 1
