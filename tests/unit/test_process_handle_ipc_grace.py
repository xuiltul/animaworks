"""Unit tests for IPC grace period: do not set FAILED on transient IPC errors when process is alive."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor.ipc import IPCResponse
from core.supervisor.process_handle import ProcessHandle, ProcessState


@pytest.fixture
def handle(tmp_path: Path) -> ProcessHandle:
    """Create a ProcessHandle in RUNNING state with mocked IPC client."""
    socket_path = tmp_path / "test.sock"
    h = ProcessHandle(
        anima_name="test-anima",
        socket_path=socket_path,
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
    )
    h.state = ProcessState.RUNNING
    h.process = MagicMock()
    h.process.pid = 12345
    h.process.poll.return_value = None  # Process alive
    h.process.returncode = None
    h.ipc_client = AsyncMock()
    socket_path.touch()
    return h


class TestSendRequestIPCGrace:
    """Tests for send_request: only set FAILED when process is actually dead."""

    @pytest.mark.asyncio
    async def test_send_request_does_not_set_failed_when_process_alive(
        self,
        handle: ProcessHandle,
    ):
        """RuntimeError when process alive (poll() is None) must NOT set FAILED."""
        handle.ipc_client.send_request.side_effect = RuntimeError("IPC connection lost")

        with pytest.raises(RuntimeError, match="IPC connection lost"):
            await handle.send_request("ping", {}, timeout=5.0)

        assert handle.state == ProcessState.RUNNING

    @pytest.mark.asyncio
    async def test_send_request_does_set_failed_when_process_dead(
        self,
        handle: ProcessHandle,
    ):
        """RuntimeError when process dead (poll() returns exit code) must set FAILED."""
        handle.process.poll.return_value = 1
        handle.process.returncode = 1
        handle.ipc_client.send_request.side_effect = RuntimeError("IPC connection lost")

        with pytest.raises(RuntimeError, match="IPC connection lost"):
            await handle.send_request("ping", {}, timeout=5.0)

        assert handle.state == ProcessState.FAILED


class TestSendRequestStreamIPCGrace:
    """Tests for send_request_stream: only set FAILED when process is actually dead."""

    @pytest.mark.asyncio
    async def test_send_request_stream_does_not_set_failed_when_process_alive(
        self,
        handle: ProcessHandle,
    ):
        """RuntimeError when process alive must NOT set FAILED."""

        async def failing_stream(*args, **kwargs):
            yield IPCResponse(id="req_1", stream=True, chunk="{}")
            raise RuntimeError("IPC stream broken")

        handle.ipc_client.send_request_stream = failing_stream

        chunks = []
        with pytest.raises(RuntimeError, match="IPC stream broken"):
            async for resp in handle.send_request_stream("chat", {"stream": True}):
                chunks.append(resp)

        assert handle.state == ProcessState.RUNNING

    @pytest.mark.asyncio
    async def test_send_request_stream_does_set_failed_when_process_dead(
        self,
        handle: ProcessHandle,
    ):
        """RuntimeError when process dead must set FAILED."""
        handle.process.poll.return_value = 1
        handle.process.returncode = 1

        async def failing_stream(*args, **kwargs):
            yield IPCResponse(id="req_1", stream=True, chunk="{}")
            raise RuntimeError("IPC stream broken")

        handle.ipc_client.send_request_stream = failing_stream

        with pytest.raises(RuntimeError, match="IPC stream broken"):
            async for _ in handle.send_request_stream("chat", {"stream": True}):
                pass

        assert handle.state == ProcessState.FAILED

    @pytest.mark.asyncio
    async def test_send_request_stream_raises_runtime_error_in_all_cases(
        self,
        handle: ProcessHandle,
    ):
        """RuntimeError must always be re-raised regardless of process state."""

        async def failing_stream(*args, **kwargs):
            yield IPCResponse(id="req_1", stream=True, chunk="{}")
            raise RuntimeError("IPC stream broken")

        handle.ipc_client.send_request_stream = failing_stream

        with pytest.raises(RuntimeError):
            async for _ in handle.send_request_stream("chat", {"stream": True}):
                pass

        handle.process.poll.return_value = 1
        handle.process.returncode = 1

        with pytest.raises(RuntimeError):
            async for _ in handle.send_request_stream("chat", {"stream": True}):
                pass
