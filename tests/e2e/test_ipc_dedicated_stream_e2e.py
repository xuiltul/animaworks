"""
E2E tests for IPC dedicated stream connection.

Tests verify that streaming requests use a dedicated connection,
isolating them from concurrent unary traffic on the shared connection.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from core.supervisor.ipc import IPCClient, IPCRequest, IPCResponse, IPCServer

pytestmark = [pytest.mark.asyncio, pytest.mark.e2e]


# ── Test 1: Stream with interleaved pings ─────────────────────────


async def test_e2e_stream_with_interleaved_pings() -> None:
    """Streaming on a dedicated connection should not interfere with
    concurrent unary (ping) requests on the shared connection.

    Start a server that handles both 'ping' and 'stream_test'.  Kick off
    a streaming request in one task and send 3 pings on the shared
    connection in a second task.  Both should complete without errors.
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "interleaved.sock"

        async def handler(request: IPCRequest) -> IPCResponse | AsyncIterator[IPCResponse]:
            if request.method == "ping":
                return IPCResponse(id=request.id, result={"status": "ok"})

            if request.method == "stream_test":
                async def _stream() -> AsyncIterator[IPCResponse]:
                    for i in range(5):
                        yield IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=f'{{"type": "text_delta", "text": "chunk_{i}"}}',
                        )
                        await asyncio.sleep(0.05)
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        done=True,
                        result={"response": "complete"},
                    )
                return _stream()

            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD", "message": f"Unknown: {request.method}"},
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        client = IPCClient(socket_path)
        try:
            await client.connect()

            # Task 1: collect streaming responses on a dedicated connection
            stream_results: list[IPCResponse] = []

            async def _do_stream() -> None:
                req = IPCRequest(id="stream_001", method="stream_test")
                async for resp in client.send_request_stream(req, timeout=5.0):
                    stream_results.append(resp)

            # Task 2: send 3 pings on the shared connection
            ping_results: list[IPCResponse] = []

            async def _do_pings() -> None:
                for i in range(3):
                    await asyncio.sleep(0.03)
                    req = IPCRequest(id=f"ping_{i}", method="ping")
                    resp = await client.send_request(req, timeout=5.0)
                    ping_results.append(resp)

            await asyncio.gather(_do_stream(), _do_pings())

            # Verify streaming: 5 chunks + 1 done = 6 responses
            assert len(stream_results) == 6, (
                f"Expected 6 stream responses (5 chunks + done), got {len(stream_results)}"
            )
            assert stream_results[-1].done is True
            assert stream_results[-1].result == {"response": "complete"}

            # Verify pings: all 3 should succeed
            assert len(ping_results) == 3, (
                f"Expected 3 ping responses, got {len(ping_results)}"
            )
            for i, resp in enumerate(ping_results):
                assert resp.error is None, f"Ping {i} returned error: {resp.error}"
                assert resp.result == {"status": "ok"}

        finally:
            await client.close()
            await server.stop()


# ── Test 2: Heartbeat relay over dedicated connection ─────────────


async def test_e2e_heartbeat_relay_over_dedicated_connection() -> None:
    """A heartbeat relay streams start -> processing -> result -> done.

    All chunks should arrive in order on the dedicated connection.
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "heartbeat_relay.sock"

        relay_sequence = [
            {"type": "heartbeat_relay_start", "message": "処理中です"},
            {"type": "heartbeat_relay", "text": "チェック1完了"},
            {"type": "heartbeat_relay", "text": "チェック2完了"},
            {"type": "heartbeat_relay_done"},
        ]

        async def handler(request: IPCRequest) -> IPCResponse | AsyncIterator[IPCResponse]:
            if request.method != "heartbeat_relay":
                return IPCResponse(
                    id=request.id,
                    error={"code": "UNKNOWN_METHOD", "message": "Unknown"},
                )

            async def _stream() -> AsyncIterator[IPCResponse]:
                import json
                for item in relay_sequence:
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        chunk=json.dumps(item, ensure_ascii=False),
                    )
                    await asyncio.sleep(0.02)
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={"response": "heartbeat complete"},
                )
            return _stream()

        server = IPCServer(socket_path, handler)
        await server.start()

        client = IPCClient(socket_path)
        try:
            await client.connect()

            collected: list[IPCResponse] = []
            req = IPCRequest(id="hb_001", method="heartbeat_relay")
            async for resp in client.send_request_stream(req, timeout=5.0):
                collected.append(resp)

            # 4 chunk responses + 1 done response = 5 total
            assert len(collected) == 5, (
                f"Expected 5 responses (4 relay chunks + done), got {len(collected)}"
            )

            # Verify chunks are in correct order
            import json
            for i, expected in enumerate(relay_sequence):
                chunk_data = json.loads(collected[i].chunk)
                assert chunk_data["type"] == expected["type"], (
                    f"Chunk {i}: expected type={expected['type']}, "
                    f"got type={chunk_data['type']}"
                )

            # Verify done
            assert collected[-1].done is True
            assert collected[-1].result == {"response": "heartbeat complete"}

        finally:
            await client.close()
            await server.stop()


# ── Test 3: Stale response isolation ─────────────────────────────


async def test_e2e_stale_response_isolation() -> None:
    """KEY TEST: A delayed unary response must NOT contaminate the
    dedicated stream connection.

    Before the dedicated-connection fix, if a slow_ping response arrived
    after a streaming request started on the *same* connection, the stream
    would receive the slow_ping response (wrong ID) and raise an
    IPC protocol error.

    With dedicated connections:
    - slow_ping travels on the shared connection
    - stream_test travels on its own dedicated connection
    - They never cross.
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "stale_isolation.sock"

        async def handler(request: IPCRequest) -> IPCResponse | AsyncIterator[IPCResponse]:
            if request.method == "slow_ping":
                # Simulate a delayed response
                await asyncio.sleep(0.2)
                return IPCResponse(id=request.id, result={"status": "slow_ok"})

            if request.method == "stream_test":
                async def _stream() -> AsyncIterator[IPCResponse]:
                    for i in range(3):
                        yield IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=f'{{"type": "text_delta", "text": "part_{i}"}}',
                        )
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        done=True,
                        result={"response": "stream done"},
                    )
                return _stream()

            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD", "message": "Unknown"},
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        client = IPCClient(socket_path)
        try:
            await client.connect()

            # Step 1: Send slow_ping on the shared connection (takes 0.2s).
            # We do NOT await it yet — launch it as a task.
            slow_ping_req = IPCRequest(id="slow_ping_001", method="slow_ping")
            ping_task = asyncio.create_task(
                client.send_request(slow_ping_req, timeout=5.0)
            )

            # Step 2: Give the request time to reach the server, then
            # immediately start streaming on a dedicated connection.
            await asyncio.sleep(0.05)

            stream_results: list[IPCResponse] = []
            stream_req = IPCRequest(id="stream_002", method="stream_test")
            async for resp in client.send_request_stream(stream_req, timeout=5.0):
                stream_results.append(resp)

            # Step 3: Await the slow ping
            ping_resp = await ping_task

            # Verify stream: 3 chunks + 1 done = 4 responses
            assert len(stream_results) == 4, (
                f"Expected 4 stream responses (3 chunks + done), got {len(stream_results)}"
            )

            # All stream response IDs must match the stream request ID
            for i, resp in enumerate(stream_results):
                assert resp.id == "stream_002", (
                    f"Stream response {i} has id={resp.id}, "
                    f"expected 'stream_002' — stale response leaked!"
                )

            assert stream_results[-1].done is True

            # Verify slow ping also succeeded (on the shared connection)
            assert ping_resp.error is None, (
                f"Slow ping returned error: {ping_resp.error}"
            )
            assert ping_resp.result == {"status": "slow_ok"}
            assert ping_resp.id == "slow_ping_001"

        finally:
            await client.close()
            await server.stop()


# ── Test 4: Multiple sequential streams ──────────────────────────


async def test_e2e_multiple_sequential_streams() -> None:
    """Each sequential streaming request should open a new dedicated
    connection and close it when the stream ends.

    The server tracks how many connections have been handled.  After N
    streaming requests, there should be N+1 connections total (1 shared
    + N dedicated).
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "multi_stream.sock"

        connection_count = 0

        async def handler(request: IPCRequest) -> IPCResponse | AsyncIterator[IPCResponse]:
            if request.method == "stream_test":
                async def _stream() -> AsyncIterator[IPCResponse]:
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        chunk='{"type": "text_delta", "text": "hello"}',
                    )
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        done=True,
                        result={"response": "ok"},
                    )
                return _stream()

            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD", "message": "Unknown"},
            )

        # Wrap _handle_connection to count connections
        original_handle = IPCServer._handle_connection

        async def _counting_handle(self_server, reader, writer):
            nonlocal connection_count
            connection_count += 1
            return await original_handle(self_server, reader, writer)

        server = IPCServer(socket_path, handler)
        server._handle_connection = lambda r, w: _counting_handle(server, r, w)
        await server.start()

        client = IPCClient(socket_path)
        try:
            await client.connect()
            # The shared connection is connection #1
            # Give the server a moment to register the connection
            await asyncio.sleep(0.05)
            initial_connections = connection_count

            # Perform 3 sequential streaming requests
            for seq in range(3):
                results: list[IPCResponse] = []
                req = IPCRequest(id=f"seq_{seq}", method="stream_test")
                async for resp in client.send_request_stream(req, timeout=5.0):
                    results.append(resp)

                # Each stream: 1 chunk + 1 done = 2 responses
                assert len(results) == 2, (
                    f"Stream {seq}: expected 2 responses, got {len(results)}"
                )
                assert results[0].chunk is not None
                assert results[-1].done is True

                # Verify response IDs match
                for resp in results:
                    assert resp.id == f"seq_{seq}", (
                        f"Stream {seq}: response id={resp.id}, expected 'seq_{seq}'"
                    )

            # Allow server to process connection closes
            await asyncio.sleep(0.1)

            # Verify connection count: initial (shared) + 3 dedicated
            dedicated_connections = connection_count - initial_connections
            assert dedicated_connections == 3, (
                f"Expected 3 dedicated connections, got {dedicated_connections} "
                f"(total={connection_count}, initial={initial_connections})"
            )

        finally:
            await client.close()
            await server.stop()
