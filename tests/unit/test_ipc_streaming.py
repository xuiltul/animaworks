# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for IPC streaming support.

Tests the streaming protocol where a handler returns an AsyncIterator[IPCResponse]
and the client reads multiple JSON lines until done=True.
"""

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import AsyncIterator
from typing import Union

from core.supervisor.ipc import (
    IPCClient,
    IPCServer,
    IPCRequest,
    IPCResponse,
)


# ── Streaming Protocol Serialization Tests ────────────────────────────


def test_streaming_chunk_serialization():
    """Test that a streaming chunk serializes correctly."""
    chunk = IPCResponse(
        id="req_001",
        stream=True,
        chunk="Hello "
    )
    data = json.loads(chunk.to_json())
    assert data["id"] == "req_001"
    assert data["stream"] is True
    assert data["chunk"] == "Hello "
    assert "done" not in data
    assert "result" not in data


def test_streaming_done_serialization():
    """Test that a final streaming response serializes correctly."""
    done = IPCResponse(
        id="req_001",
        stream=True,
        done=True,
        result={"response": "Hello world", "replied_to": []}
    )
    data = json.loads(done.to_json())
    assert data["id"] == "req_001"
    assert data["stream"] is True
    assert data["done"] is True
    assert data["result"]["response"] == "Hello world"


def test_streaming_chunk_deserialization():
    """Test that a streaming chunk deserializes correctly."""
    line = json.dumps({
        "id": "req_001",
        "stream": True,
        "chunk": "partial text"
    })
    resp = IPCResponse.from_json(line)
    assert resp.id == "req_001"
    assert resp.stream is True
    assert resp.chunk == "partial text"
    assert resp.done is False


def test_streaming_done_deserialization():
    """Test that a final streaming response deserializes correctly."""
    line = json.dumps({
        "id": "req_001",
        "stream": True,
        "done": True,
        "result": {"response": "complete"}
    })
    resp = IPCResponse.from_json(line)
    assert resp.id == "req_001"
    assert resp.stream is True
    assert resp.done is True
    assert resp.result == {"response": "complete"}


# ── Server/Client Streaming Tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_ipc_streaming_basic():
    """Test basic streaming: server yields multiple chunks then done."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "stream.sock"

        chunks = ["Hello", ", ", "world", "!"]

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            if request.method == "stream_echo":

                async def _gen() -> AsyncIterator[IPCResponse]:
                    for c in chunks:
                        yield IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=c
                        )
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        done=True,
                        result={"response": "".join(chunks)}
                    )

                return _gen()

            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD", "message": "Unknown"}
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="stream_001",
                method="stream_echo",
                params={"stream": True}
            )

            received_chunks: list[str] = []
            final_result = None

            async for response in client.send_request_stream(request, timeout=5.0):
                if response.done:
                    final_result = response.result
                elif response.chunk:
                    received_chunks.append(response.chunk)

            assert received_chunks == chunks
            assert final_result is not None
            assert final_result["response"] == "Hello, world!"

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_streaming_single_done():
    """Test streaming with only a final done response (no intermediate chunks)."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "stream_done.sock"

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            async def _gen() -> AsyncIterator[IPCResponse]:
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={"response": "immediate result"}
                )

            return _gen()

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="done_001",
                method="instant",
                params={}
            )

            results: list[IPCResponse] = []
            async for response in client.send_request_stream(request, timeout=5.0):
                results.append(response)

            assert len(results) == 1
            assert results[0].done is True
            assert results[0].result == {"response": "immediate result"}

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_streaming_error_from_handler():
    """Test that a non-streaming error response terminates the stream."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "stream_err.sock"

        async def handler(
            request: IPCRequest,
        ) -> IPCResponse:
            # Return a plain error (not streaming)
            return IPCResponse(
                id=request.id,
                error={"code": "SOME_ERROR", "message": "Something went wrong"}
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="err_001",
                method="failing",
                params={"stream": True}
            )

            results: list[IPCResponse] = []
            async for response in client.send_request_stream(request, timeout=5.0):
                results.append(response)

            # Should receive a single non-streaming error response
            assert len(results) == 1
            assert results[0].stream is False
            assert results[0].error is not None
            assert results[0].error["code"] == "SOME_ERROR"

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_streaming_mixed_with_non_streaming():
    """Test that non-streaming requests still work alongside streaming."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "mixed.sock"

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            if request.method == "echo":
                # Non-streaming response
                return IPCResponse(
                    id=request.id,
                    result={"echo": request.params.get("message")}
                )
            elif request.method == "stream_count":
                # Streaming response

                async def _gen() -> AsyncIterator[IPCResponse]:
                    for i in range(3):
                        yield IPCResponse(
                            id=request.id,
                            stream=True,
                            chunk=str(i)
                        )
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        done=True,
                        result={"count": 3}
                    )

                return _gen()

            return IPCResponse(
                id=request.id,
                error={"code": "UNKNOWN_METHOD", "message": "Unknown"}
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            # 1) Non-streaming request
            req1 = IPCRequest(
                id="ns_001",
                method="echo",
                params={"message": "hello"}
            )
            resp1 = await client.send_request(req1, timeout=5.0)
            assert resp1.result == {"echo": "hello"}

            # 2) Streaming request
            req2 = IPCRequest(
                id="s_001",
                method="stream_count",
                params={"stream": True}
            )
            stream_chunks: list[str] = []
            stream_result = None
            async for response in client.send_request_stream(req2, timeout=5.0):
                if response.done:
                    stream_result = response.result
                elif response.chunk:
                    stream_chunks.append(response.chunk)

            assert stream_chunks == ["0", "1", "2"]
            assert stream_result == {"count": 3}

            # 3) Another non-streaming request to verify connection still works
            req3 = IPCRequest(
                id="ns_002",
                method="echo",
                params={"message": "still alive"}
            )
            resp3 = await client.send_request(req3, timeout=5.0)
            assert resp3.result == {"echo": "still alive"}

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_streaming_many_chunks():
    """Test streaming with many chunks to exercise the protocol."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "many_chunks.sock"

        num_chunks = 100

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            async def _gen() -> AsyncIterator[IPCResponse]:
                for i in range(num_chunks):
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        chunk=f"chunk_{i}"
                    )
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={"total": num_chunks}
                )

            return _gen()

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="many_001",
                method="stream_many",
                params={"stream": True}
            )

            received: list[str] = []
            final = None
            async for response in client.send_request_stream(request, timeout=10.0):
                if response.done:
                    final = response.result
                elif response.chunk:
                    received.append(response.chunk)

            assert len(received) == num_chunks
            assert received[0] == "chunk_0"
            assert received[-1] == f"chunk_{num_chunks - 1}"
            assert final == {"total": num_chunks}

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_streaming_json_chunks():
    """Test streaming with JSON-encoded chunks (as used by AnimaRunner)."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "json_chunks.sock"

        events = [
            {"type": "text_delta", "text": "Hello"},
            {"type": "text_delta", "text": " world"},
            {"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"},
            {"type": "tool_end", "tool_id": "t1", "tool_name": "web_search"},
        ]

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            async def _gen() -> AsyncIterator[IPCResponse]:
                for event in events:
                    yield IPCResponse(
                        id=request.id,
                        stream=True,
                        chunk=json.dumps(event, ensure_ascii=False)
                    )
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={"response": "Hello world", "replied_to": []}
                )

            return _gen()

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="json_001",
                method="process_message",
                params={"message": "hi", "stream": True}
            )

            received_events: list[dict] = []
            final_result = None

            async for response in client.send_request_stream(request, timeout=5.0):
                if response.done:
                    final_result = response.result
                elif response.chunk:
                    received_events.append(json.loads(response.chunk))

            assert len(received_events) == 4
            assert received_events[0] == {"type": "text_delta", "text": "Hello"}
            assert received_events[1] == {"type": "text_delta", "text": " world"}
            assert received_events[2]["type"] == "tool_start"
            assert received_events[3]["type"] == "tool_end"
            assert final_result["response"] == "Hello world"

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_ipc_streaming_timeout():
    """Test that streaming times out if the handler is too slow."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "slow_stream.sock"

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            async def _gen() -> AsyncIterator[IPCResponse]:
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    chunk="first"
                )
                # Simulate a very slow stream
                await asyncio.sleep(10.0)
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={"response": "late"}
                )

            return _gen()

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="timeout_001",
                method="slow",
                params={"stream": True}
            )

            with pytest.raises(asyncio.TimeoutError):
                async for _ in client.send_request_stream(request, timeout=1.0):
                    pass

            await client.close()

        finally:
            await server.stop()
