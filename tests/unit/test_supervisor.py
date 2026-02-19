"""
Unit tests for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from core.supervisor.manager import (
    ProcessSupervisor,
    RestartPolicy,
    HealthConfig
)
from core.supervisor.process_handle import ProcessHandle, ProcessState


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
    assert config.max_missed_pings == 3
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
    handle.stats.started_at = datetime.now()
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
        handle.stats.started_at = datetime.now()
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


# Note: Full integration tests with actual process spawning
# are better suited for integration test suite.
# These unit tests focus on logic and state management.
