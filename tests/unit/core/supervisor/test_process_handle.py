"""Unit tests for ProcessHandle bug fixes: ping counter and is_alive IPC detection."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats


@pytest.fixture
def handle(tmp_path: Path) -> ProcessHandle:
    """Create a ProcessHandle instance for testing."""
    h = ProcessHandle(
        anima_name="test-anima",
        socket_path=tmp_path / "test.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
    )
    return h


class TestPingCounter:
    """Tests for Bug #2: ping() must increment missed_pings when state != RUNNING."""

    @pytest.mark.asyncio
    async def test_ping_increments_missed_pings_when_failed(self, handle: ProcessHandle):
        """ping() should increment missed_pings when state is FAILED."""
        handle.state = ProcessState.FAILED
        handle.stats.missed_pings = 0

        result = await handle.ping()

        assert result is False
        assert handle.stats.missed_pings == 1

    @pytest.mark.asyncio
    async def test_ping_increments_missed_pings_when_stopped(self, handle: ProcessHandle):
        """ping() should increment missed_pings when state is STOPPED."""
        handle.state = ProcessState.STOPPED
        handle.stats.missed_pings = 0

        result = await handle.ping()

        assert result is False
        assert handle.stats.missed_pings == 1

    @pytest.mark.asyncio
    async def test_ping_accumulates_missed_pings_across_calls(self, handle: ProcessHandle):
        """ping() should accumulate missed_pings across multiple calls."""
        handle.state = ProcessState.FAILED
        handle.stats.missed_pings = 0

        await handle.ping()
        await handle.ping()
        await handle.ping()

        assert handle.stats.missed_pings == 3

    @pytest.mark.asyncio
    async def test_ping_reaches_hang_detection_threshold(self, handle: ProcessHandle):
        """ping() missed_pings should reach 3 (hang detection threshold) after 3 calls."""
        handle.state = ProcessState.FAILED
        handle.stats.missed_pings = 0

        for _ in range(3):
            await handle.ping()

        assert handle.stats.missed_pings >= 3


class TestIsAlive:
    """Tests for Bug #3: is_alive() must detect dead IPC connections."""

    def test_is_alive_returns_false_when_no_process(self, handle: ProcessHandle):
        """is_alive() should return False when process is None."""
        handle.process = None
        assert handle.is_alive() is False

    def test_is_alive_returns_false_when_process_exited(self, handle: ProcessHandle):
        """is_alive() should return False when process has exited."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # exited with code 1
        handle.process = mock_process

        assert handle.is_alive() is False

    def test_is_alive_returns_true_when_process_running_no_ipc(self, handle: ProcessHandle):
        """is_alive() should return True when process is running and no IPC client."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # still running
        handle.process = mock_process
        handle.ipc_client = None

        assert handle.is_alive() is True

    def test_is_alive_returns_true_when_ipc_connection_healthy(self, handle: ProcessHandle):
        """is_alive() should return True when process is running and IPC is healthy."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        handle.process = mock_process

        mock_client = MagicMock()
        mock_writer = MagicMock()
        mock_writer.is_closing.return_value = False
        mock_client.writer = mock_writer
        handle.ipc_client = mock_client

        assert handle.is_alive() is True

    def test_is_alive_returns_true_when_process_running_with_ipc_client(self, handle: ProcessHandle):
        """is_alive() returns True when OS process is running, regardless of IPC client state.

        With per-request dedicated connections, there is no persistent IPC
        writer to check.  Liveness is determined solely by OS process status.
        """
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        handle.process = mock_process

        mock_client = MagicMock()
        handle.ipc_client = mock_client

        assert handle.is_alive() is True
