# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for process liveness during streams and keepalive behavior."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.time_utils import now_jst
from server.supervisor.manager import HealthConfig, ProcessSupervisor
from server.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats


class TestStreamingTimestamp:
    def test_initial_state(self, tmp_path: Path) -> None:
        handle = ProcessHandle(
            anima_name="test",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        assert handle._streaming is False
        assert handle._streaming_started_at is None


class TestHealthCheckDuringStreaming:
    @pytest.fixture
    def supervisor(self, tmp_path: Path) -> ProcessSupervisor:
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True, exist_ok=True)
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
        handle._streaming_started_at = now_jst() - timedelta(hours=4)
        handle.stats = ProcessStats(started_at=now_jst() - timedelta(minutes=5))
        handle.process = MagicMock()
        handle.process.poll.return_value = None
        handle.process.pid = 12345
        handle.ipc_client = MagicMock()
        handle.ipc_client.writer = MagicMock()
        handle.ipc_client.writer.is_closing.return_value = False
        return handle

    @pytest.mark.asyncio
    async def test_streaming_process_death_detected(
        self, supervisor: ProcessSupervisor, streaming_handle: ProcessHandle
    ) -> None:
        streaming_handle.process.poll.return_value = 1
        supervisor.processes["test"] = streaming_handle

        with patch.object(supervisor, "_handle_process_failure", new_callable=AsyncMock) as failure:
            await supervisor._check_process_health("test", streaming_handle)
            await asyncio.sleep(0)
            failure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_streaming_failed_state_detected(
        self, supervisor: ProcessSupervisor, streaming_handle: ProcessHandle
    ) -> None:
        streaming_handle.state = ProcessState.FAILED
        supervisor.processes["test"] = streaming_handle

        with patch.object(supervisor, "_handle_process_failure", new_callable=AsyncMock) as failure:
            await supervisor._check_process_health("test", streaming_handle)
            await asyncio.sleep(0)
            failure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_four_hour_stream_is_not_killed_while_process_is_alive(
        self, supervisor: ProcessSupervisor, streaming_handle: ProcessHandle
    ) -> None:
        supervisor.processes["test"] = streaming_handle

        with (
            patch.object(supervisor, "_handle_process_failure", new_callable=AsyncMock) as failure,
            patch.object(supervisor, "_handle_process_hang", new_callable=AsyncMock) as hang,
        ):
            await supervisor._check_process_health("test", streaming_handle)
            await asyncio.sleep(0)
            failure.assert_not_called()
            hang.assert_not_called()


class TestKeepaliveProducerStop:
    @pytest.mark.asyncio
    async def test_keepalive_stops_when_producer_done(self) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        last_chunk_time_holder = [time.monotonic()]
        assert queue.empty()
        assert last_chunk_time_holder[0] > 0

        async def instant_crash() -> None:
            raise RuntimeError("Agent SDK crashed")

        producer_task = asyncio.create_task(instant_crash())
        with pytest.raises(RuntimeError, match="Agent SDK crashed"):
            await producer_task

        keepalive_started = asyncio.Event()
        keepalive_stopped = asyncio.Event()

        async def keepalive_producer() -> None:
            keepalive_started.set()
            try:
                while True:
                    await asyncio.sleep(0.1)
                    if producer_task.done():
                        keepalive_stopped.set()
                        return
            except asyncio.CancelledError:
                return

        task = asyncio.create_task(keepalive_producer())
        await keepalive_started.wait()
        async with asyncio.timeout(2.0):
            await keepalive_stopped.wait()
        await task
