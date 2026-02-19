# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for ProcessHandle / IPC bug fixes.

Tests three bug fixes:
1. stop() sends shutdown via IPC before state changes to STOPPING
2. Health check recovers handles stuck in STOPPING state
3. IPC stream connection close raises RuntimeError
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor.ipc import IPCClient, IPCRequest, IPCResponse, IPCServer
from core.supervisor.process_handle import ProcessHandle, ProcessState


# ── Test 1: stop() sends shutdown via IPC while still RUNNING ────────


@pytest.mark.asyncio
async def test_stop_sends_shutdown_via_ipc_e2e():
    """E2E: stop() must send 'shutdown' IPC request before changing state to STOPPING.

    The bug was that state was changed to STOPPING *before* send_request(),
    but send_request() requires state == RUNNING. This test proves the fix
    by verifying the server actually receives the shutdown request.
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "shutdown_e2e.sock"

        # Track whether "shutdown" method was received by the server
        shutdown_received = asyncio.Event()

        async def handler(request: IPCRequest) -> IPCResponse:
            if request.method == "shutdown":
                shutdown_received.set()
                return IPCResponse(id=request.id, result={"status": "shutting_down"})
            return IPCResponse(id=request.id, result={"status": "ok"})

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            handle = ProcessHandle(
                anima_name="test-shutdown",
                socket_path=socket_path,
                animas_dir=Path(tmpdir) / "animas",
                shared_dir=Path(tmpdir) / "shared",
            )

            # Simulate a running process with IPC connected
            mock_process = MagicMock()
            # poll() returns None while "alive", then 0 after shutdown
            mock_process.poll.side_effect = [None, 0, 0, 0, 0]
            mock_process.returncode = 0
            mock_process.pid = 12345
            handle.process = mock_process

            handle.ipc_client = IPCClient(socket_path)
            await handle.ipc_client.connect()
            handle.state = ProcessState.RUNNING

            # Call stop() — this should send "shutdown" via IPC while still RUNNING
            await handle.stop(timeout=5.0)

            # Verify the server received the shutdown request
            assert shutdown_received.is_set(), (
                "Server never received 'shutdown' request — "
                "send_request() was likely blocked by premature state change"
            )

            # Verify the handle ended in STOPPED state
            assert handle.state == ProcessState.STOPPED

        finally:
            await server.stop()


# ── Test 2: Health check recovers stuck STOPPING handles ─────────────


@pytest.mark.asyncio
async def test_health_check_recovers_stuck_stopping_e2e():
    """E2E: _check_process_health() transitions a handle stuck in STOPPING to FAILED.

    When a process gets stuck in STOPPING state for more than 30 seconds
    (e.g. after a failed shutdown), the health check should detect this,
    change the state to FAILED, and trigger _handle_process_failure for
    automatic recovery.
    """
    from core.supervisor.manager import HealthConfig, ProcessSupervisor

    with TemporaryDirectory() as tmpdir:
        supervisor = ProcessSupervisor(
            animas_dir=Path(tmpdir) / "animas",
            shared_dir=Path(tmpdir) / "shared",
            run_dir=Path(tmpdir) / "run",
            health_config=HealthConfig(startup_grace_sec=0),
        )

        handle = ProcessHandle(
            anima_name="test-stuck",
            socket_path=Path(tmpdir) / "test.sock",
            animas_dir=Path(tmpdir) / "animas",
            shared_dir=Path(tmpdir) / "shared",
        )

        # Set the handle to STOPPING with stopping_since > 30 seconds ago
        handle.state = ProcessState.STOPPING
        handle.stopping_since = datetime.now() - timedelta(seconds=60)

        supervisor.processes["test-stuck"] = handle

        # Track if _handle_process_failure was called
        failure_called = False
        failure_name = None

        async def mock_handle_failure(name, h):
            nonlocal failure_called, failure_name
            failure_called = True
            failure_name = name

        supervisor._handle_process_failure = mock_handle_failure

        # Run health check
        await supervisor._check_process_health("test-stuck", handle)

        # Allow the asyncio.create_task to run
        await asyncio.sleep(0)

        # Verify state transitioned to FAILED
        assert handle.state == ProcessState.FAILED, (
            f"Expected FAILED but got {handle.state} — "
            "health check did not detect stuck STOPPING state"
        )

        # Verify recovery was triggered
        assert failure_called, (
            "_handle_process_failure should be called for a handle stuck in STOPPING"
        )
        assert failure_name == "test-stuck"


# ── Test 3: IPC stream connection close raises RuntimeError ──────────


@pytest.mark.asyncio
async def test_ipc_stream_connection_close_raises_runtime_error_e2e():
    """E2E: Connection close during streaming raises RuntimeError.

    When the server drops the connection mid-stream (e.g. process crash),
    the client's send_request_stream() must raise
    RuntimeError("Connection closed during stream") instead of silently
    returning incomplete data. This is the underlying mechanism that
    triggers the ProcessHandle to transition to FAILED state.
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "stream_close_e2e.sock"

        # We need the server-side writer to forcibly close the connection
        # mid-stream. Capture the writer from the server's connection handler.
        server_writer_ref: list[asyncio.StreamWriter] = []
        chunk_sent = asyncio.Event()

        async def _stream_gen(request_id: str):
            """Async generator: sends one chunk, then waits for connection to be killed."""
            yield IPCResponse(
                id=request_id,
                stream=True,
                chunk="partial data",
            )
            chunk_sent.set()
            # Block until the test kills the connection
            await asyncio.sleep(10)

        async def streaming_handler(request: IPCRequest):
            """Handler that returns an async iterator for streaming."""
            return _stream_gen(request.id)

        server = IPCServer(socket_path, streaming_handler)

        # Monkey-patch _handle_connection to capture the server-side writer
        original_handle = server._handle_connection

        async def patched_handle(reader, writer):
            server_writer_ref.append(writer)
            await original_handle(reader, writer)

        server._handle_connection = patched_handle
        await server.start()

        client = IPCClient(socket_path)
        await client.connect()

        try:
            request = IPCRequest(
                id="stream_close_001",
                method="stream_test",
                params={"stream": True},
            )

            chunks_received = []

            async def kill_connection_after_chunk():
                """Wait for the first chunk to be sent, then close the server-side writer."""
                await chunk_sent.wait()
                # Give a tiny bit of time for the chunk to propagate to the client
                await asyncio.sleep(0.05)
                # Forcibly close the server-side writer (simulates process crash)
                if server_writer_ref:
                    # Close the dedicated streaming connection (last one opened),
                    # not the shared connection (first one from client.connect())
                    server_writer_ref[-1].close()
                    await server_writer_ref[-1].wait_closed()

            kill_task = asyncio.create_task(kill_connection_after_chunk())

            with pytest.raises(RuntimeError, match="Connection closed"):
                async for response in client.send_request_stream(
                    request, timeout=5.0
                ):
                    chunks_received.append(response)

            await kill_task

            # Verify we got at least the first chunk before the error
            assert len(chunks_received) >= 1, (
                "Should have received at least one chunk before connection close"
            )
            assert chunks_received[0].chunk == "partial data"

        finally:
            await client.close()
            # Ensure server is stopped
            try:
                await server.stop()
            except Exception:
                pass
