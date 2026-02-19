"""
Unit tests for IPC dedicated stream connection and ID validation.

Tests the dedicated connection per streaming request, response ID validation,
and _reconnect() recovery mechanism.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import pytest
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
from unittest.mock import AsyncMock, MagicMock, patch

from core.supervisor.ipc import (
    IPC_BUFFER_LIMIT,
    IPC_CHUNK_MAX,
    IPCClient,
    IPCRequest,
    IPCResponse,
    IPCServer,
)
from core.supervisor.process_handle import ProcessHandle, ProcessState


# ── Helpers ──────────────────────────────────────────────────


class MockStreamWriter:
    """Minimal mock for asyncio.StreamWriter used in unit tests."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self._closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        pass

    def get_extra_info(self, key: str, default=None):
        return default

    def is_closing(self) -> bool:
        return self._closed


# ── Test 1: Stream uses dedicated connection ──────────────────


@pytest.mark.asyncio
async def test_stream_uses_dedicated_connection():
    """Streaming opens a DEDICATED UDS connection separate from the shared one.

    Start a real IPCServer, connect a client (shared connection), then call
    send_request_stream. The server should see 2 distinct connections: one
    from connect() and one from the dedicated stream connection opened by
    send_request_stream().
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "dedicated.sock"

        connection_count = 0
        original_handle = None

        chunks = ["alpha", "beta", "gamma"]

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            if request.method == "stream_test":
                async def _gen() -> AsyncIterator[IPCResponse]:
                    for c in chunks:
                        yield IPCResponse(
                            id=request.id, stream=True, chunk=c
                        )
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        done=True,
                        result={"total": len(chunks)},
                    )
                return _gen()
            return IPCResponse(
                id=request.id, result={"pong": True}
            )

        server = IPCServer(socket_path, handler)

        # Wrap _handle_connection to count connections
        original_handle = server._handle_connection

        async def counting_handle(reader, writer):
            nonlocal connection_count
            connection_count += 1
            await original_handle(reader, writer)

        server._handle_connection = counting_handle

        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()
            # Allow event loop to run the connection handler callback
            await asyncio.sleep(0.05)
            # Connection 1: the shared connection from connect()
            assert connection_count == 1

            request = IPCRequest(
                id="ded_001",
                method="stream_test",
                params={"stream": True},
            )

            received_chunks: list[str] = []
            async for response in client.send_request_stream(request, timeout=5.0):
                if response.chunk and not response.done:
                    received_chunks.append(response.chunk)

            # Verify the stream worked correctly
            assert received_chunks == chunks

            # Connection 2: the dedicated connection from send_request_stream
            assert connection_count == 2

            await client.close()

        finally:
            await server.stop()


# ── Test 2: Dedicated connection closes on completion ─────────


@pytest.mark.asyncio
async def test_stream_dedicated_connection_closes_on_completion():
    """The dedicated stream connection is closed after the stream completes normally.

    Track server-side writers to verify the dedicated connection is closed
    after all chunks have been consumed.
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "close_ok.sock"

        server_writers: list[asyncio.StreamWriter] = []

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            async def _gen() -> AsyncIterator[IPCResponse]:
                yield IPCResponse(
                    id=request.id, stream=True, chunk="data"
                )
                yield IPCResponse(
                    id=request.id, stream=True, done=True, result={"ok": True}
                )
            return _gen()

        server = IPCServer(socket_path, handler)

        original_handle = server._handle_connection

        async def tracking_handle(reader, writer):
            server_writers.append(writer)
            await original_handle(reader, writer)

        server._handle_connection = tracking_handle

        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="close_001",
                method="stream_test",
                params={"stream": True},
            )
            async for _ in client.send_request_stream(request, timeout=5.0):
                pass

            # Allow server-side cleanup to finish
            await asyncio.sleep(0.1)

            # server_writers[0] = shared connection, server_writers[1] = dedicated
            assert len(server_writers) == 2
            dedicated_writer = server_writers[1]
            # The dedicated connection writer should be closing or closed
            assert dedicated_writer.is_closing()

            await client.close()

        finally:
            await server.stop()


# ── Test 3: Dedicated connection closes on cancel ─────────────


@pytest.mark.asyncio
async def test_stream_dedicated_connection_closes_on_cancel():
    """The dedicated stream connection is closed even when iteration is cancelled.

    Uses mock-based approach to avoid server cleanup issues. Feed one chunk
    to a mock reader, break out of the loop, and verify the mock writer
    was closed by the finally block in send_request_stream.
    """
    client = IPCClient(Path("/tmp/test_cancel.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    mock_reader = asyncio.StreamReader()
    mock_writer = MockStreamWriter()

    # Feed one streaming chunk (not done) — no more data after this
    chunk = IPCResponse(id="cancel_001", stream=True, chunk="first")
    mock_reader.feed_data((chunk.to_json() + "\n").encode("utf-8"))

    async def mock_open_unix(*args, **kwargs):
        return mock_reader, mock_writer

    with patch("asyncio.open_unix_connection", side_effect=mock_open_unix):
        stream_gen = client.send_request_stream(
            IPCRequest(id="cancel_001", method="slow_stream"), timeout=5.0
        )
        async for response in stream_gen:
            assert response.chunk == "first"
            break  # Cancel after first chunk
        # Explicitly close the async generator to trigger finally block
        await stream_gen.aclose()

    # Verify dedicated connection was closed despite cancellation
    assert mock_writer._closed, "Dedicated connection should be closed after break"


# ── Test 4: Dedicated connection closes on timeout ────────────


@pytest.mark.asyncio
async def test_stream_dedicated_connection_closes_on_timeout():
    """The dedicated stream connection is closed when a timeout fires.

    Patches asyncio.open_unix_connection to return mock objects and
    asyncio.wait_for to immediately raise TimeoutError, verifying the
    finally block closes the dedicated writer even on timeout.
    """
    client = IPCClient(Path("/tmp/test_timeout.sock"))
    # Set up the shared connection so the "not connected" guard passes
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    mock_reader = asyncio.StreamReader()
    mock_writer = MockStreamWriter()
    # Do NOT feed any data to mock_reader -- readline() would block forever

    async def mock_open_unix(*args, **kwargs):
        return mock_reader, mock_writer

    original_wait_for = asyncio.wait_for

    async def mock_wait_for(coro, *, timeout):
        # Close the coroutine to avoid RuntimeWarning
        coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.open_unix_connection", side_effect=mock_open_unix):
        with patch.object(asyncio, "wait_for", side_effect=mock_wait_for):
            with pytest.raises(asyncio.TimeoutError):
                async for _ in client.send_request_stream(
                    IPCRequest(id="t_001", method="slow"),
                    timeout=1.0,
                ):
                    pass

    # Verify the dedicated connection was closed despite the timeout
    assert mock_writer._closed


# ── Test 5: send_request ID validation (matching) ────────────


@pytest.mark.asyncio
async def test_send_request_id_validation():
    """send_request succeeds when response ID matches request ID."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "id_ok.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            return IPCResponse(id=request.id, result={"echo": True})

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="req_match_001",
                method="echo",
                params={},
            )
            response = await client.send_request(request, timeout=5.0)

            assert response.id == "req_match_001"
            assert response.result == {"echo": True}
            assert response.error is None

            await client.close()

        finally:
            await server.stop()


# ── Test 6: send_request ID mismatch raises ──────────────────


@pytest.mark.asyncio
async def test_send_request_id_mismatch_raises():
    """send_request raises RuntimeError when response ID does not match.

    The handler intentionally returns a response with a WRONG ID to
    simulate a stale response read from the buffer. After the error,
    the client should have called _reconnect().
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "id_mismatch.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            # Intentionally return wrong ID to simulate stale data
            return IPCResponse(id="stale_ping", result={"stale": True})

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            old_writer = client.writer

            request = IPCRequest(
                id="req_123",
                method="test",
                params={},
            )

            with pytest.raises(RuntimeError, match="IPC protocol error"):
                await client.send_request(request, timeout=5.0)

            # After ID mismatch, _reconnect() should have created new
            # reader/writer objects (the connection was re-established)
            assert client.writer is not old_writer

            await client.close()

        finally:
            await server.stop()


# ── Test 7: stream ID mismatch raises ─────────────────────────


@pytest.mark.asyncio
async def test_stream_id_mismatch_raises():
    """send_request_stream raises RuntimeError on response ID mismatch.

    Patches asyncio.open_unix_connection to return a mock reader that
    feeds back a response with a WRONG ID.
    """
    client = IPCClient(Path("/tmp/test_stream_mismatch.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    mock_reader = asyncio.StreamReader()
    mock_writer = MockStreamWriter()

    # Feed a streaming response with WRONG ID
    wrong_response = IPCResponse(
        id="wrong_id_999",
        stream=True,
        chunk="data",
    )
    mock_reader.feed_data((wrong_response.to_json() + "\n").encode("utf-8"))

    async def mock_open_unix(*args, **kwargs):
        return mock_reader, mock_writer

    with patch("asyncio.open_unix_connection", side_effect=mock_open_unix):
        with pytest.raises(RuntimeError, match="IPC protocol error: response ID mismatch"):
            async for _ in client.send_request_stream(
                IPCRequest(id="req_correct_id", method="stream_test"),
                timeout=5.0,
            ):
                pass

    # Verify the dedicated connection was closed
    assert mock_writer._closed


# ── Test 8: stream non-streaming response with wrong ID ──────


@pytest.mark.asyncio
async def test_stream_non_streaming_response_with_wrong_id():
    """send_request_stream raises RuntimeError for a non-streaming response with wrong ID.

    When the server returns a plain (non-streaming) response with a mismatched
    ID, the stream should raise a protocol error rather than yielding the
    stale response.
    """
    client = IPCClient(Path("/tmp/test_nonstream_mismatch.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    mock_reader = asyncio.StreamReader()
    mock_writer = MockStreamWriter()

    # Feed a non-streaming response with WRONG ID
    wrong_response = IPCResponse(
        id="stale_old_ping",
        result={"status": "ok"},
    )
    mock_reader.feed_data((wrong_response.to_json() + "\n").encode("utf-8"))

    async def mock_open_unix(*args, **kwargs):
        return mock_reader, mock_writer

    with patch("asyncio.open_unix_connection", side_effect=mock_open_unix):
        with pytest.raises(RuntimeError, match="IPC protocol error"):
            async for _ in client.send_request_stream(
                IPCRequest(id="req_fresh_001", method="process_message"),
                timeout=5.0,
            ):
                pass

    # Verify the dedicated connection was closed
    assert mock_writer._closed


# ── Test 9: _reconnect after ID mismatch ─────────────────────


@pytest.mark.asyncio
async def test_reconnect_after_id_mismatch():
    """After ID mismatch triggers _reconnect, the next request succeeds.

    Uses a real server where the handler alternates: first returns a wrong ID,
    second returns the correct ID. The first send_request should raise
    RuntimeError, and the second should succeed (proving _reconnect worked).
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "reconnect.sock"

        call_count = 0

        async def handler(request: IPCRequest) -> IPCResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return WRONG ID to trigger mismatch
                return IPCResponse(
                    id="stale_response",
                    result={"wrong": True},
                )
            # All subsequent calls: return correct ID
            return IPCResponse(
                id=request.id,
                result={"correct": True},
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            old_writer = client.writer

            # First request: should raise RuntimeError due to ID mismatch
            req1 = IPCRequest(id="req_first", method="test", params={})
            with pytest.raises(RuntimeError, match="IPC protocol error"):
                await client.send_request(req1, timeout=5.0)

            # After mismatch, writer should have been replaced by _reconnect
            assert client.writer is not old_writer
            new_writer = client.writer

            # Second request: should succeed (proves _reconnect worked)
            req2 = IPCRequest(id="req_second", method="test", params={})
            response = await client.send_request(req2, timeout=5.0)

            assert response.id == "req_second"
            assert response.result == {"correct": True}

            # Writer should still be the reconnected one
            assert client.writer is new_writer

            await client.close()

        finally:
            await server.stop()


# ── Test 10: Concurrent ping and stream ───────────────────────


@pytest.mark.asyncio
async def test_concurrent_ping_and_stream():
    """Pings on the shared connection and streaming on a dedicated connection
    work concurrently without ID mismatch errors.

    The dedicated stream connection ensures that ping responses never
    contaminate the stream reader buffer, and vice versa.
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "concurrent.sock"

        num_stream_chunks = 5

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            if request.method == "ping":
                return IPCResponse(
                    id=request.id,
                    result={"status": "ok"},
                )
            elif request.method == "stream_long":
                async def _gen() -> AsyncIterator[IPCResponse]:
                    for i in range(num_stream_chunks):
                        await asyncio.sleep(0.05)
                        yield IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=f"chunk_{i}",
                        )
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        done=True,
                        result={"total": num_stream_chunks},
                    )
                return _gen()

            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD"},
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            # Collect results from both tasks
            stream_chunks: list[str] = []
            stream_result = None
            ping_results: list[bool] = []
            errors: list[Exception] = []

            async def do_stream():
                nonlocal stream_result
                try:
                    req = IPCRequest(
                        id="stream_conc_001",
                        method="stream_long",
                        params={"stream": True},
                    )
                    async for response in client.send_request_stream(
                        req, timeout=10.0
                    ):
                        if response.done:
                            stream_result = response.result
                        elif response.chunk:
                            stream_chunks.append(response.chunk)
                except Exception as e:
                    errors.append(e)

            async def do_pings():
                try:
                    for i in range(3):
                        await asyncio.sleep(0.03)
                        req = IPCRequest(
                            id=f"ping_conc_{i:03d}",
                            method="ping",
                            params={},
                        )
                        resp = await client.send_request(req, timeout=5.0)
                        ping_results.append(
                            resp.result is not None
                            and resp.result.get("status") == "ok"
                        )
                except Exception as e:
                    errors.append(e)

            await asyncio.gather(do_stream(), do_pings())

            # No errors should have occurred
            assert errors == [], f"Unexpected errors: {errors}"

            # All stream chunks received correctly
            expected_chunks = [f"chunk_{i}" for i in range(num_stream_chunks)]
            assert stream_chunks == expected_chunks
            assert stream_result == {"total": num_stream_chunks}

            # All pings succeeded
            assert len(ping_results) == 3
            assert all(ping_results)

            await client.close()

        finally:
            await server.stop()


# ── Test 11: ready_check uses unique ID ───────────────────────


@pytest.mark.asyncio
async def test_ready_check_uses_unique_id():
    """ProcessHandle._wait_for_ready uses a unique ping ID per call,
    not the old fixed "ready_check" string.

    Creates a ProcessHandle with a mock IPC client, patches send_request
    to capture requests, and verifies the IDs start with "ping_" and
    are unique across calls.
    """
    socket_path = Path("/tmp/test_ready_check.sock")
    animas_dir = Path("/tmp/test_animas")
    shared_dir = Path("/tmp/test_shared")

    handle = ProcessHandle(
        anima_name="test-anima",
        socket_path=socket_path,
        animas_dir=animas_dir,
        shared_dir=shared_dir,
    )

    captured_requests: list[IPCRequest] = []

    async def mock_send_request(request: IPCRequest, timeout: float = 60.0):
        captured_requests.append(request)
        return IPCResponse(id=request.id, result={"status": "ok"})

    mock_client = MagicMock(spec=IPCClient)
    mock_client.send_request = AsyncMock(side_effect=mock_send_request)

    handle.ipc_client = mock_client
    # Mock process to prevent "exited during initialization" check
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    handle.process = mock_process

    # Call _wait_for_ready twice (it polls until "ok" is returned)
    await handle._wait_for_ready(timeout=5.0)

    # At least one request should have been captured
    assert len(captured_requests) >= 1
    first_id = captured_requests[0].id

    # Verify ID format: starts with "ping_", not the old fixed "ready_check"
    assert first_id.startswith("ping_"), (
        f"Expected ID to start with 'ping_', got '{first_id}'"
    )
    assert first_id != "ready_check", (
        "Expected unique ID, not the old fixed 'ready_check'"
    )
    assert captured_requests[0].method == "ping"

    # Call _wait_for_ready again with a fresh capture to verify uniqueness
    captured_requests.clear()
    await handle._wait_for_ready(timeout=5.0)

    assert len(captured_requests) >= 1
    second_id = captured_requests[0].id

    assert second_id.startswith("ping_")
    assert second_id != first_id, (
        f"Expected unique IDs across calls, but both were '{first_id}'"
    )
