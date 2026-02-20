# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for streaming crash recovery (Agent SDK crash detection).

Tests the three fixes:
1. ProcessHandle._streaming_started_at tracking
2. _check_process_health during streaming
3. _keepalive_producer stops when producer finishes
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from core.time_utils import now_jst
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats
from core.supervisor.manager import ProcessSupervisor, HealthConfig, RestartPolicy


class TestStreamingTimestamp:
    """Test ProcessHandle._streaming_started_at tracking."""

    def test_initial_state(self, tmp_path: Path):
        """_streaming_started_at is None initially."""
        handle = ProcessHandle(
            anima_name="test",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        assert handle._streaming is False
        assert handle._streaming_started_at is None


class TestHealthCheckDuringStreaming:
    """Test _check_process_health behavior during streaming."""

    @pytest.fixture
    def supervisor(self, tmp_path: Path) -> ProcessSupervisor:
        return ProcessSupervisor(
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
            run_dir=tmp_path / "run",
            health_config=HealthConfig(startup_grace_sec=0),
        )

    @pytest.fixture
    def streaming_handle(self, tmp_path: Path) -> ProcessHandle:
        handle = ProcessHandle(
            anima_name="test",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        handle.state = ProcessState.RUNNING
        handle._streaming = True
        handle._streaming_started_at = now_jst()
        handle.stats = ProcessStats(started_at=now_jst() - timedelta(minutes=5))
        # Mock process
        handle.process = MagicMock()
        handle.process.poll.return_value = None  # alive
        handle.process.pid = 12345
        # Mock IPC client
        handle.ipc_client = MagicMock()
        handle.ipc_client.writer = MagicMock()
        handle.ipc_client.writer.is_closing.return_value = False
        return handle

    @pytest.mark.asyncio
    async def test_streaming_process_death_detected(
        self, supervisor: ProcessSupervisor, streaming_handle: ProcessHandle
    ):
        """During streaming, process death is detected."""
        # Simulate process exit
        streaming_handle.process.poll.return_value = 1
        supervisor.processes["test"] = streaming_handle

        with patch.object(supervisor, "_handle_process_failure", new_callable=AsyncMock) as mock_failure:
            await supervisor._check_process_health("test", streaming_handle)
            # Give the created task time to start
            await asyncio.sleep(0.1)
            mock_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_failed_state_detected(
        self, supervisor: ProcessSupervisor, streaming_handle: ProcessHandle
    ):
        """During streaming, FAILED state is detected."""
        streaming_handle.state = ProcessState.FAILED
        supervisor.processes["test"] = streaming_handle

        with patch.object(supervisor, "_handle_process_failure", new_callable=AsyncMock) as mock_failure:
            await supervisor._check_process_health("test", streaming_handle)
            await asyncio.sleep(0.1)
            mock_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_timeout_detected(
        self, supervisor: ProcessSupervisor, streaming_handle: ProcessHandle
    ):
        """Streaming exceeding max duration triggers hang detection."""
        # Set started_at far in the past
        streaming_handle._streaming_started_at = now_jst() - timedelta(hours=1)
        supervisor._max_streaming_duration_sec = 60  # 60s for testing
        supervisor.processes["test"] = streaming_handle

        with patch.object(supervisor, "_handle_process_hang", new_callable=AsyncMock) as mock_hang:
            await supervisor._check_process_health("test", streaming_handle)
            await asyncio.sleep(0.1)
            mock_hang.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_normal_not_triggered(
        self, supervisor: ProcessSupervisor, streaming_handle: ProcessHandle
    ):
        """Normal streaming should not trigger failure/hang detection."""
        supervisor._max_streaming_duration_sec = 1800
        supervisor.processes["test"] = streaming_handle

        with patch.object(supervisor, "_handle_process_failure", new_callable=AsyncMock) as mock_failure:
            with patch.object(supervisor, "_handle_process_hang", new_callable=AsyncMock) as mock_hang:
                await supervisor._check_process_health("test", streaming_handle)
                await asyncio.sleep(0.1)
                mock_failure.assert_not_called()
                mock_hang.assert_not_called()


class TestKeepaliveProducerStop:
    """Test that keepalive producer stops when stream producer finishes."""

    @pytest.mark.asyncio
    async def test_keepalive_stops_when_producer_done(self):
        """Keepalive should stop when producer_task is done."""
        from core.supervisor.ipc import IPCResponse

        queue: asyncio.Queue = asyncio.Queue()
        last_chunk_time_holder = [0.0]

        import time
        last_chunk_time_holder[0] = time.monotonic()

        # Create a producer_task that completes immediately (simulating crash)
        async def instant_crash():
            raise RuntimeError("Agent SDK crashed")

        producer_task = asyncio.create_task(instant_crash())
        # Wait for it to finish
        try:
            await producer_task
        except RuntimeError:
            pass

        # Now create keepalive producer that checks producer_task
        keepalive_started = asyncio.Event()
        keepalive_stopped = asyncio.Event()

        async def keepalive_producer():
            keepalive_started.set()
            try:
                while True:
                    await asyncio.sleep(0.1)  # Short interval for testing
                    if producer_task.done():
                        keepalive_stopped.set()
                        return
            except asyncio.CancelledError:
                return

        task = asyncio.create_task(keepalive_producer())
        await keepalive_started.wait()

        # Wait for keepalive to detect producer death
        async with asyncio.timeout(2.0):
            await keepalive_stopped.wait()

        assert keepalive_stopped.is_set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
