"""E2E tests for streaming crash recovery."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor.manager import ProcessSupervisor, HealthConfig, RestartPolicy
from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats


@pytest.mark.asyncio
async def test_health_check_detects_crash_during_streaming(tmp_path: Path):
    """Verify complete flow: streaming handle with dead process is detected."""
    supervisor = ProcessSupervisor(
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        run_dir=tmp_path / "run",
        health_config=HealthConfig(
            ping_interval_sec=0.5,
            startup_grace_sec=0,
        ),
    )
    supervisor._max_streaming_duration_sec = 60

    # Create handle simulating streaming + dead process
    handle = ProcessHandle(
        anima_name="crash-test",
        socket_path=tmp_path / "crash.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )
    handle.state = ProcessState.RUNNING
    handle._streaming = True
    handle._streaming_started_at = datetime.now()
    handle.stats = ProcessStats(started_at=datetime.now())
    handle.process = MagicMock()
    handle.process.poll.return_value = 1  # Process died
    handle.process.pid = 99999
    handle.ipc_client = MagicMock()
    handle.ipc_client.writer = MagicMock()
    handle.ipc_client.writer.is_closing.return_value = False

    supervisor.processes["crash-test"] = handle

    # Run health check
    failure_detected = asyncio.Event()
    original_handler = supervisor._handle_process_failure

    async def capture_failure(name, h):
        failure_detected.set()

    supervisor._handle_process_failure = capture_failure

    await supervisor._check_process_health("crash-test", handle)
    await asyncio.sleep(0.2)

    assert failure_detected.is_set(), (
        "Health check should detect process death during streaming"
    )


@pytest.mark.asyncio
async def test_streaming_timestamp_lifecycle(tmp_path: Path):
    """Verify _streaming_started_at is set and cleared correctly."""
    handle = ProcessHandle(
        anima_name="ts-test",
        socket_path=tmp_path / "test.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )

    # Initially None
    assert handle._streaming_started_at is None

    # Simulate streaming start
    handle._streaming = True
    handle._streaming_started_at = datetime.now()
    assert handle._streaming_started_at is not None
    assert handle._streaming is True

    # Simulate streaming end
    handle._streaming = False
    handle._streaming_started_at = None
    assert handle._streaming_started_at is None
    assert handle._streaming is False
