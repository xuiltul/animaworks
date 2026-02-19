# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for IPC health check and recovery flow.

Tests the full chain:
1. IPC buffer limit allows large messages
2. IPC connection death is detected by is_alive()
3. ping() increments missed_pings even when FAILED
4. Health check detects FAILED state and triggers recovery
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from core.supervisor.ipc import IPCClient, IPCServer, IPCRequest, IPCResponse, IPC_BUFFER_LIMIT
from core.supervisor.process_handle import ProcessHandle, ProcessState


@pytest.mark.asyncio
async def test_large_message_roundtrip():
    """E2E: A 128KB message should successfully round-trip through IPC."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "large_e2e.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            msg = request.params.get("message", "")
            return IPCResponse(
                id=request.id,
                result={"length": len(msg), "echo": msg[:10]}
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            # 128KB message — would have caused LimitOverrunError before fix
            large_msg = "X" * (128 * 1024)
            request = IPCRequest(
                id="e2e_large_001",
                method="echo",
                params={"message": large_msg}
            )
            response = await client.send_request(request, timeout=10.0)

            assert response.error is None
            assert response.result["length"] == 128 * 1024

            await client.close()
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_connection_death_detected_by_is_alive():
    """E2E: When IPC connection is forcefully closed, is_alive() should detect it."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "death_e2e.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            return IPCResponse(id=request.id, result={"status": "ok"})

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            handle = ProcessHandle(
                anima_name="test-anima",
                socket_path=socket_path,
                animas_dir=Path(tmpdir) / "animas",
                shared_dir=Path(tmpdir) / "shared",
            )

            # Simulate a running process with IPC connected
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # OS-level alive
            handle.process = mock_process

            handle.ipc_client = IPCClient(socket_path)
            await handle.ipc_client.connect()
            handle.state = ProcessState.RUNNING

            # is_alive() should be True initially
            assert handle.is_alive() is True

            # Forcefully close the writer (simulates IPC death)
            handle.ipc_client.writer.close()
            await handle.ipc_client.writer.wait_closed()

            # is_alive() should now detect the dead IPC connection
            assert handle.is_alive() is False

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ping_counter_reaches_threshold_when_failed():
    """E2E: ping() should reach hang detection threshold (3) when state is FAILED."""
    with TemporaryDirectory() as tmpdir:
        handle = ProcessHandle(
            anima_name="test-anima",
            socket_path=Path(tmpdir) / "test.sock",
            animas_dir=Path(tmpdir) / "animas",
            shared_dir=Path(tmpdir) / "shared",
        )
        handle.state = ProcessState.FAILED
        handle.stats.missed_pings = 0

        # Simulate 3 health check cycles
        for i in range(3):
            result = await handle.ping()
            assert result is False
            assert handle.stats.missed_pings == i + 1

        # After 3 pings, missed_pings should be at threshold
        assert handle.stats.missed_pings == 3


@pytest.mark.asyncio
async def test_health_check_detects_failed_state():
    """E2E: _check_process_health() should detect FAILED state directly."""
    from core.supervisor.manager import ProcessSupervisor, HealthConfig

    with TemporaryDirectory() as tmpdir:
        supervisor = ProcessSupervisor(
            animas_dir=Path(tmpdir) / "animas",
            shared_dir=Path(tmpdir) / "shared",
            run_dir=Path(tmpdir) / "run",
            health_config=HealthConfig(startup_grace_sec=0),
        )

        handle = ProcessHandle(
            anima_name="test-anima",
            socket_path=Path(tmpdir) / "test.sock",
            animas_dir=Path(tmpdir) / "animas",
            shared_dir=Path(tmpdir) / "shared",
        )
        handle.state = ProcessState.FAILED

        # Set started_at far enough back to pass startup grace period
        from datetime import datetime, timedelta
        handle.stats.started_at = datetime.now() - timedelta(seconds=60)

        supervisor.processes["test-anima"] = handle

        # Track if _handle_process_failure was called
        failure_called = False
        original_handle_failure = supervisor._handle_process_failure

        async def mock_handle_failure(name, h):
            nonlocal failure_called
            failure_called = True

        supervisor._handle_process_failure = mock_handle_failure

        await supervisor._check_process_health("test-anima", handle)

        # Allow the asyncio.create_task to run
        await asyncio.sleep(0)

        assert failure_called, "_handle_process_failure should be called for FAILED state"
