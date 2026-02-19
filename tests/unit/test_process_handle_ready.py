"""Unit tests for ProcessHandle._wait_for_ready and socket early-exit detection."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.ipc import IPCResponse
from core.supervisor.process_handle import ProcessHandle, ProcessState


@pytest.fixture
def handle(tmp_path: Path) -> ProcessHandle:
    """Create a ProcessHandle with mock paths."""
    socket_path = tmp_path / "test.sock"
    return ProcessHandle(
        anima_name="test-anima",
        socket_path=socket_path,
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
    )


class TestWaitForReady:
    """Tests for _wait_for_ready polling."""

    @pytest.mark.asyncio
    async def test_ready_on_first_ping(self, handle: ProcessHandle):
        """Should return immediately when first ping returns ok."""
        handle.process = MagicMock()
        handle.process.poll.return_value = None  # Process alive

        mock_client = AsyncMock()
        mock_client.send_request.return_value = IPCResponse(
            id="ready_check",
            result={"status": "ok", "anima": "test-anima"},
        )
        handle.ipc_client = mock_client

        await handle._wait_for_ready(timeout=5.0)
        # Should succeed without timeout

    @pytest.mark.asyncio
    async def test_ready_after_initializing(self, handle: ProcessHandle):
        """Should poll until status changes from initializing to ok."""
        handle.process = MagicMock()
        handle.process.poll.return_value = None

        call_count = 0

        async def mock_send(request, timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return IPCResponse(
                    id="ready_check",
                    result={"status": "initializing"},
                )
            return IPCResponse(
                id="ready_check",
                result={"status": "ok"},
            )

        mock_client = AsyncMock()
        mock_client.send_request = mock_send
        handle.ipc_client = mock_client

        await handle._wait_for_ready(timeout=10.0)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_raises(self, handle: ProcessHandle):
        """Should raise TimeoutError when anima never becomes ready."""
        handle.process = MagicMock()
        handle.process.poll.return_value = None

        mock_client = AsyncMock()
        mock_client.send_request.return_value = IPCResponse(
            id="ready_check",
            result={"status": "initializing"},
        )
        handle.ipc_client = mock_client

        with pytest.raises(TimeoutError, match="not ready"):
            await handle._wait_for_ready(timeout=2.0)

    @pytest.mark.asyncio
    async def test_process_exit_during_init_raises(self, handle: ProcessHandle):
        """Should raise RuntimeError if process exits during init."""
        handle.process = MagicMock()
        handle.process.poll.return_value = 1  # Exited with error
        handle.process.returncode = 1

        mock_client = AsyncMock()
        handle.ipc_client = mock_client

        with pytest.raises(RuntimeError, match="exited with code 1"):
            await handle._wait_for_ready(timeout=5.0)


class TestWaitForSocketEarlyExit:
    """Tests for _wait_for_socket early exit detection."""

    @pytest.mark.asyncio
    async def test_process_crash_before_socket(self, handle: ProcessHandle):
        """Should raise RuntimeError if process exits before socket is created."""
        handle.process = MagicMock()
        handle.process.poll.return_value = 1
        handle.process.returncode = 1

        with pytest.raises(RuntimeError, match="exited with code 1"):
            await handle._wait_for_socket(timeout=5.0)

    @pytest.mark.asyncio
    async def test_socket_created_successfully(self, handle: ProcessHandle, tmp_path):
        """Should return when socket file is created."""
        handle.process = MagicMock()
        handle.process.poll.return_value = None

        # Create socket file after a short delay
        async def create_socket():
            await asyncio.sleep(0.2)
            handle.socket_path.touch()

        task = asyncio.create_task(create_socket())
        await handle._wait_for_socket(timeout=5.0)
        await task
