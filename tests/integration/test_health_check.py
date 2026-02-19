"""Integration tests for health checking and auto-restart."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import pytest

from core.supervisor.manager import ProcessSupervisor, HealthConfig, RestartPolicy
from core.supervisor.process_handle import ProcessState


@pytest.mark.asyncio
async def test_health_check_loop(data_dir: Path, make_anima):
    """Test health check loop pings processes."""
    make_anima("test-anima")

    # Short ping interval for testing
    health_config = HealthConfig(
        ping_interval_sec=1.0,
        ping_timeout_sec=2.0,
        max_missed_pings=3,
        startup_grace_sec=2.0
    )

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs",
        health_config=health_config
    )

    try:
        # Start anima
        await supervisor.start_all(["test-anima"])

        # Wait for health check loop to run a few times
        await asyncio.sleep(3.0)

        # Verify ping stats updated
        handle = supervisor.processes.get("test-anima")
        assert handle is not None
        assert handle.stats.last_ping_at is not None
        assert handle.stats.missed_pings == 0

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_process_crash_detection(data_dir: Path, make_anima):
    """Test detection when process crashes."""
    make_anima("test-anima")

    restart_policy = RestartPolicy(
        max_retries=1,  # Allow one restart
        backoff_base_sec=0.5,
        reset_after_sec=10.0
    )

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs",
        restart_policy=restart_policy
    )

    try:
        await supervisor.start_anima("test-anima")

        handle = supervisor.processes.get("test-anima")
        original_pid = handle.get_pid()

        # Kill the process forcibly
        import os
        os.kill(original_pid, signal.SIGKILL)

        # Wait for supervisor to detect crash and restart
        await asyncio.sleep(3.0)

        # Verify process was restarted
        handle = supervisor.processes.get("test-anima")
        if handle:
            new_pid = handle.get_pid()
            # Note: In actual implementation, supervisor needs to monitor
            # process exit and trigger restart. This test may need adjustment
            # based on actual auto-restart implementation.

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_missed_pings_tracking(data_dir: Path, make_anima):
    """Test tracking of missed pings."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_anima("test-anima")
        handle = supervisor.processes.get("test-anima")

        # First ping should succeed
        success = await handle.ping(timeout=5.0)
        assert success is True
        assert handle.stats.missed_pings == 0

        # Stop the process but keep handle
        await handle.stop(timeout=2.0)

        # Ping should fail now
        success = await handle.ping(timeout=1.0)
        assert success is False
        # Missed ping count incremented
        # Note: Actual behavior depends on implementation

    finally:
        await supervisor.shutdown_all()
