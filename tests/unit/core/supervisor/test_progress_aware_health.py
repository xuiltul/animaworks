"""Tests for engine-owned stream timeouts and supervisor process liveness."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.time_utils import now_jst
from server.supervisor._mgr_health import HealthMixin
from server.supervisor.manager import HealthConfig
from server.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats


def _make_handle(tmp_path: Path, *, started_minutes_ago: int = 5) -> ProcessHandle:
    handle = ProcessHandle(
        anima_name="test-anima",
        socket_path=tmp_path / "test.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )
    handle.state = ProcessState.RUNNING
    handle.stats = ProcessStats(started_at=now_jst() - timedelta(minutes=started_minutes_ago))
    process = MagicMock()
    process.poll.return_value = None
    process.returncode = None
    process.pid = 12345
    handle.process = process
    return handle


def _make_supervisor(health_config: HealthConfig | None = None) -> HealthMixin:
    supervisor = object.__new__(HealthMixin)
    supervisor.health_config = health_config or HealthConfig()
    supervisor._shutdown = False
    supervisor._restarting = set()
    from server.supervisor.restart_state import RestartController

    supervisor._restart_ctl = RestartController(
        failed_threshold=5,
        base_delay_sec=2.0,
        max_delay_sec=60.0,
        stable_reset_sec=300.0,
    )
    supervisor.restart_policy = MagicMock()
    supervisor.restart_policy.max_retries = 5
    supervisor.restart_policy.backoff_base_sec = 2.0
    supervisor.restart_policy.backoff_max_sec = 60.0
    supervisor.restart_policy.reset_after_sec = 300.0
    supervisor.processes = {}
    return supervisor


class TestEngineOwnsBusyTimeout:
    @pytest.mark.asyncio
    async def test_busy_engine_turn_is_not_killed_by_supervisor_idle_threshold(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path)
        handle.ping = AsyncMock(
            return_value={
                "success": True,
                "is_busy": True,
                "last_progress_at": (now_jst() - timedelta(hours=4)).isoformat(),
            }
        )
        supervisor = _make_supervisor()
        supervisor._handle_process_hang = AsyncMock()

        await supervisor._check_process_health("test-anima", handle)

        supervisor._handle_process_hang.assert_not_called()
        assert handle.stats.last_busy_since is not None

    @pytest.mark.asyncio
    async def test_long_stream_is_not_killed_when_runner_process_is_alive(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path, started_minutes_ago=240)
        handle._streaming = True
        handle._streaming_started_at = now_jst() - timedelta(hours=4)
        supervisor = _make_supervisor()
        supervisor._handle_process_hang = AsyncMock()
        supervisor._handle_process_failure = AsyncMock()

        await supervisor._check_process_health("test-anima", handle)

        supervisor._handle_process_hang.assert_not_called()
        supervisor._handle_process_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_dead_streaming_process_is_still_reconciled(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path)
        handle._streaming = True
        handle.is_alive = MagicMock(return_value=False)
        supervisor = _make_supervisor()
        supervisor._handle_process_failure = AsyncMock()

        await supervisor._check_process_health("test-anima", handle)
        await asyncio.sleep(0)

        supervisor._handle_process_failure.assert_awaited_once_with("test-anima", handle)

    @pytest.mark.asyncio
    async def test_busy_to_idle_resets_last_busy_since(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path)
        handle.stats.last_busy_since = now_jst() - timedelta(minutes=30)
        handle.ping = AsyncMock(return_value={"success": True, "is_busy": False})
        supervisor = _make_supervisor()

        await supervisor._check_process_health("test-anima", handle)

        assert handle.stats.last_busy_since is None

    @pytest.mark.asyncio
    async def test_missed_ping_limit_still_detects_unresponsive_runner(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path)
        handle.stats.missed_pings = HealthConfig().max_missed_pings
        handle.ping = AsyncMock(return_value={"success": False})
        supervisor = _make_supervisor()
        supervisor._handle_process_hang = AsyncMock()

        await supervisor._check_process_health("test-anima", handle)
        await asyncio.sleep(0)

        supervisor._handle_process_hang.assert_awaited_once_with("test-anima", handle)


class TestServerRunnerLivenessTimeout:
    def test_default_is_900_seconds(self) -> None:
        from core.config.models import ServerConfig

        assert ServerConfig().runner_liveness_timeout == 900

    def test_custom_value(self) -> None:
        from core.config.models import ServerConfig

        assert ServerConfig(runner_liveness_timeout=1200).runner_liveness_timeout == 1200


class TestProcessStatsLastBusySince:
    def test_default_is_none(self) -> None:
        stats = ProcessStats(started_at=now_jst())
        assert stats.last_busy_since is None
