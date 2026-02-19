"""
Unit tests for IPC communication layer.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.supervisor.ipc import (
    IPCClient,
    IPCServer,
    IPCRequest,
    IPCResponse,
    IPCEvent
)


# ── Protocol Tests ────────────────────────────────────────────────

def test_ipc_request_serialization():
    """Test IPCRequest JSON serialization."""
    request = IPCRequest(
        id="req_001",
        method="test_method",
        params={"key": "value"}
    )

    json_str = request.to_json()
    data = json.loads(json_str)

    assert data["id"] == "req_001"
    assert data["method"] == "test_method"
    assert data["params"] == {"key": "value"}


def test_ipc_request_deserialization():
    """Test IPCRequest JSON deserialization."""
    json_str = json.dumps({
        "id": "req_002",
        "method": "another_method",
        "params": {"foo": "bar"}
    })

    request = IPCRequest.from_json(json_str)

    assert request.id == "req_002"
    assert request.method == "another_method"
    assert request.params == {"foo": "bar"}


def test_ipc_response_serialization():
    """Test IPCResponse JSON serialization."""
    # Normal response
    response = IPCResponse(
        id="req_001",
        result={"status": "ok"}
    )
    json_str = response.to_json()
    data = json.loads(json_str)
    assert data["id"] == "req_001"
    assert data["result"] == {"status": "ok"}
    assert "error" not in data

    # Error response
    error_response = IPCResponse(
        id="req_002",
        error={"code": "ERROR", "message": "Something went wrong"}
    )
    json_str = error_response.to_json()
    data = json.loads(json_str)
    assert data["id"] == "req_002"
    assert data["error"]["code"] == "ERROR"
    assert "result" not in data


def test_ipc_response_streaming():
    """Test streaming IPCResponse."""
    # Chunk
    chunk_response = IPCResponse(
        id="req_003",
        stream=True,
        chunk="Hello "
    )
    json_str = chunk_response.to_json()
    data = json.loads(json_str)
    assert data["stream"] is True
    assert data["chunk"] == "Hello "
    assert "done" not in data

    # Final chunk
    final_response = IPCResponse(
        id="req_003",
        stream=True,
        done=True,
        result={"total_chunks": 3}
    )
    json_str = final_response.to_json()
    data = json.loads(json_str)
    assert data["stream"] is True
    assert data["done"] is True
    assert data["result"]["total_chunks"] == 3


def test_ipc_event_serialization():
    """Test IPCEvent JSON serialization."""
    event = IPCEvent(
        event="status_changed",
        data={"status": "thinking"}
    )

    json_str = event.to_json()
    data = json.loads(json_str)

    assert data["event"] == "status_changed"
    assert data["data"] == {"status": "thinking"}


# ── Server/Client Tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ipc_server_client_communication():
    """Test basic server-client communication."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "test.sock"

        # Define a simple request handler
        async def handler(request: IPCRequest) -> IPCResponse:
            if request.method == "echo":
                return IPCResponse(
                    id=request.id,
                    result={"echo": request.params.get("message")}
                )
            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD", "message": "Unknown method"}
            )

        # Start server
        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            # Connect client
            client = IPCClient(socket_path)
            await client.connect()

            # Send request
            request = IPCRequest(
                id="test_001",
                method="echo",
                params={"message": "Hello, World!"}
            )
            response = await client.send_request(request, timeout=5.0)

            # Verify response
            assert response.id == "test_001"
            assert response.result is not None
            assert response.result["echo"] == "Hello, World!"
            assert response.error is None

            # Clean up
            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_ping_pong():
    """Test ping-pong pattern."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "ping.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            if request.method == "ping":
                return IPCResponse(
                    id=request.id,
                    result={"pong": True}
                )
            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD"}
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            # Send ping
            request = IPCRequest(id="ping_001", method="ping", params={})
            response = await client.send_request(request, timeout=5.0)

            assert response.result is not None
            assert response.result["pong"] is True

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_error_handling():
    """Test error response handling."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "error.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            if request.method == "error":
                return IPCResponse(
                    id=request.id,
                    error={"code": "TEST_ERROR", "message": "Test error"}
                )
            return IPCResponse(id=request.id, result={})

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(id="err_001", method="error", params={})
            response = await client.send_request(request, timeout=5.0)

            assert response.error is not None
            assert response.error["code"] == "TEST_ERROR"
            assert response.result is None

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_timeout():
    """Test request timeout."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "timeout.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            # Simulate slow handler
            await asyncio.sleep(10.0)
            return IPCResponse(id=request.id, result={})

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(id="timeout_001", method="slow", params={})

            # Should timeout
            with pytest.raises(asyncio.TimeoutError):
                await client.send_request(request, timeout=1.0)

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_large_message():
    """Test that messages larger than 64KB can be transmitted (buffer limit raised to 16MB)."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "large.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            # Echo back the length of the received message
            msg_len = len(request.params.get("message", ""))
            return IPCResponse(
                id=request.id,
                result={"length": msg_len}
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            # Create a message larger than 64KB (the old default limit)
            large_message = "A" * (128 * 1024)  # 128KB
            request = IPCRequest(
                id="large_001",
                method="echo",
                params={"message": large_message}
            )
            response = await client.send_request(request, timeout=10.0)

            assert response.result is not None
            assert response.result["length"] == 128 * 1024

            await client.close()
        finally:
            await server.stop()


def test_ipc_buffer_limit_constant():
    """Test that IPC_BUFFER_LIMIT is set to 16MB."""
    from core.supervisor.ipc import IPC_BUFFER_LIMIT
    assert IPC_BUFFER_LIMIT == 16 * 1024 * 1024
