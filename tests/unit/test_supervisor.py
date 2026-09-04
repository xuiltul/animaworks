"""
Unit tests for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import (
    HealthConfig,
    ProcessSupervisor,
    RestartPolicy,
)
from core.supervisor.process_handle import ProcessHandle, ProcessState
from core.time_utils import now_jst


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        yield {
            "animas_dir": tmp / "animas",
            "shared_dir": tmp / "shared",
            "run_dir": tmp / "run"
        }


@pytest.fixture
def supervisor(temp_dirs):
    """Create a ProcessSupervisor instance."""
    return ProcessSupervisor(
        animas_dir=temp_dirs["animas_dir"],
        shared_dir=temp_dirs["shared_dir"],
        run_dir=temp_dirs["run_dir"],
        restart_policy=RestartPolicy(
            max_retries=3,
            backoff_base_sec=0.1,
            backoff_max_sec=1.0
        ),
        health_config=HealthConfig(
            ping_interval_sec=0.5,
            ping_timeout_sec=0.2,
            max_missed_pings=2,
            startup_grace_sec=0.5
        )
    )


@pytest.mark.asyncio
async def test_supervisor_initialization(supervisor, temp_dirs):
    """Test supervisor initialization."""
    assert supervisor.animas_dir == temp_dirs["animas_dir"]
    assert supervisor.shared_dir == temp_dirs["shared_dir"]
    assert supervisor.run_dir == temp_dirs["run_dir"]
    assert len(supervisor.processes) == 0


@pytest.mark.asyncio
async def test_get_process_status_not_found(supervisor):
    """Test status for non-existent process."""
    status = supervisor.get_process_status("nonexistent")
    assert status["status"] == "not_found"


@pytest.mark.asyncio
async def test_restart_policy_backoff():
    """Test exponential backoff calculation."""
    policy = RestartPolicy(
        backoff_base_sec=2.0,
        backoff_max_sec=60.0
    )

    # Calculate backoffs for different retry counts
    backoffs = []
    for retry in range(7):
        backoff = min(
            policy.backoff_base_sec * (2 ** retry),
            policy.backoff_max_sec
        )
        backoffs.append(backoff)

    assert backoffs == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


@pytest.mark.asyncio
async def test_health_config_defaults():
    """Test health check configuration defaults."""
    config = HealthConfig()

    assert config.ping_interval_sec == 10.0
    assert config.ping_timeout_sec == 5.0
    assert config.max_missed_pings == 6
    assert config.startup_grace_sec == 30.0


@pytest.mark.asyncio
async def test_supervisor_process_tracking(supervisor):
    """Test process handle tracking."""
    # Mock handle
    handle = MagicMock(spec=ProcessHandle)
    handle.anima_name = "test_anima"
    handle.state = ProcessState.RUNNING
    handle.get_pid.return_value = 12345
    handle.stats = MagicMock()
    handle.stats.started_at = now_jst()
    handle.stats.restart_count = 0
    handle.stats.missed_pings = 0
    handle.stats.last_ping_at = None

    supervisor.processes["test_anima"] = handle

    # Get status
    status = supervisor.get_process_status("test_anima")
    assert status["pid"] == 12345
    assert status["status"] == "running"


@pytest.mark.asyncio
async def test_get_all_status(supervisor):
    """Test getting status of all processes."""
    # Add mock handles
    for name in ["alice", "bob"]:
        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = name
        handle.state = ProcessState.RUNNING
        handle.get_pid.return_value = 12345
        handle.stats = MagicMock()
        handle.stats.started_at = now_jst()
        handle.stats.restart_count = 0
        handle.stats.missed_pings = 0
        handle.stats.last_ping_at = None
        supervisor.processes[name] = handle

    all_status = supervisor.get_all_status()

    assert len(all_status) == 2
    assert "alice" in all_status
    assert "bob" in all_status
    assert all_status["alice"]["status"] == "running"
    assert all_status["bob"]["status"] == "running"


@pytest.mark.asyncio
async def test_supervisor_shutdown(supervisor):
    """Test graceful shutdown."""
    # Add mock handle
    handle = AsyncMock(spec=ProcessHandle)
    handle.anima_name = "test_anima"
    supervisor.processes["test_anima"] = handle

    # Shutdown
    await supervisor.shutdown_all()

    # Verify stop was called
    handle.stop.assert_called_once()
    assert len(supervisor.processes) == 0
    assert supervisor._shutdown is True


@pytest.mark.asyncio
async def test_configured_stop_timeout_is_used_for_normal_stop(temp_dirs):
    """Normal stops use the server-configured graceful-stop budget."""
    config = SimpleNamespace(server=SimpleNamespace(anima_stop_timeout=45.5))
    with patch("core.config.load_config", return_value=config):
        supervisor = ProcessSupervisor(
            animas_dir=temp_dirs["animas_dir"],
            shared_dir=temp_dirs["shared_dir"],
            run_dir=temp_dirs["run_dir"],
            restart_policy=RestartPolicy(),
            health_config=HealthConfig(),
        )

    handle = AsyncMock(spec=ProcessHandle)
    supervisor.processes["test_anima"] = handle

    await supervisor.stop_anima("test_anima")

    handle.stop.assert_awaited_once_with(
        timeout=45.5,
        drain_streams=True,
        drain_background=True,
        drain_timeout=None,
    )


@pytest.mark.asyncio
async def test_restart_uses_configured_stop_timeout(supervisor):
    """Restart delegates its normal stop to the configured stop budget."""
    supervisor._anima_stop_timeout = 45.5
    handle = AsyncMock(spec=ProcessHandle)
    supervisor.processes["test_anima"] = handle
    supervisor.start_anima = AsyncMock()

    await supervisor.restart_anima("test_anima")

    handle.stop.assert_awaited_once_with(
        timeout=45.5,
        drain_streams=True,
        drain_background=True,
        drain_timeout=None,
    )
    supervisor.start_anima.assert_awaited_once_with("test_anima")


@pytest.mark.asyncio
async def test_shutdown_retains_short_stop_timeout(supervisor):
    """Full-server shutdown remains bounded to the legacy 10-second stop."""
    handle = AsyncMock(spec=ProcessHandle)
    supervisor.processes["test_anima"] = handle

    await supervisor.shutdown_all()

    handle.stop.assert_awaited_once_with(
        timeout=10.0,
        drain_streams=False,
        drain_background=False,
        drain_timeout=None,
    )


@pytest.mark.asyncio
async def test_restart_anima_preserves_failure_state_during_shutdown(supervisor):
    """restart_anima's counter reset must not touch failure-tracking state mid-shutdown (#243)."""
    supervisor._permanently_failed.add("test_anima")
    supervisor._restart_counts["test_anima"] = 2
    supervisor._shutdown = True
    handle = AsyncMock(spec=ProcessHandle)
    supervisor.processes["test_anima"] = handle
    supervisor.start_anima = AsyncMock()

    await supervisor.restart_anima("test_anima")

    assert "test_anima" in supervisor._permanently_failed
    assert supervisor._restart_counts["test_anima"] == 2


@pytest.mark.asyncio
async def test_get_process_status_starting_timeout_skips_write_during_shutdown(supervisor):
    """A spawn-timeout observed while querying status must not record a new failure mid-shutdown (#243)."""
    supervisor._starting_since["test_anima"] = time.monotonic() - supervisor._spawn_timeout_sec - 1
    supervisor._shutdown = True

    status = supervisor.get_process_status("test_anima")

    assert status["status"] == "error"
    assert "test_anima" not in supervisor._failure_reasons
    assert "test_anima" not in supervisor._permanently_failed


@pytest.mark.asyncio
async def test_get_process_status_running_timeout_skips_write_during_shutdown(supervisor):
    """A stuck RESTARTING handle observed while querying status must not record a new failure mid-shutdown (#243)."""
    handle = MagicMock(spec=ProcessHandle)
    handle.anima_name = "test_anima"
    handle.state = ProcessState.RESTARTING
    handle.get_pid.return_value = 12345
    handle.stats = MagicMock()
    handle.stats.started_at = now_jst() - timedelta(seconds=supervisor._spawn_timeout_sec + 1)
    handle.stats.restart_count = 0
    handle.stats.missed_pings = 0
    handle.stats.last_ping_at = None
    handle.stats.last_busy_since = None
    supervisor.processes["test_anima"] = handle
    supervisor._shutdown = True

    status = supervisor.get_process_status("test_anima")

    assert status["status"] == "error"
    assert "test_anima" not in supervisor._failure_reasons
    assert "test_anima" not in supervisor._permanently_failed


@pytest.mark.asyncio
async def test_reconcile_recovery_loop_stops_at_shutdown(supervisor):
    """The permanently-failed recovery loop must not run once shutdown has started (#243)."""
    supervisor.animas_dir.mkdir(parents=True, exist_ok=True)
    anima_dir = supervisor.animas_dir / "test_anima"
    anima_dir.mkdir()
    (anima_dir / "identity.md").write_text("x")
    (anima_dir / "status.json").write_text('{"enabled": true}')

    handle = AsyncMock(spec=ProcessHandle)
    supervisor.processes["test_anima"] = handle
    supervisor._permanently_failed.add("test_anima")
    supervisor._failed_log_times["test_anima"] = 0.0
    supervisor._shutdown = True
    supervisor.start_anima = AsyncMock()
    supervisor._reconcile_assets = AsyncMock()

    await supervisor._reconcile()

    assert "test_anima" in supervisor._permanently_failed
    supervisor.start_anima.assert_not_awaited()
    supervisor._reconcile_assets.assert_not_awaited()


# Note: Full integration tests with actual process spawning
# are better suited for integration test suite.
# These unit tests focus on logic and state management.
