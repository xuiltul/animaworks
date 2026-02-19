"""
Unit tests for IPC chunk splitting and first-chunk timeout logic.

Tests the _chunked_write() static method and the first_chunk_timeout
behaviour introduced to handle large RAG payloads.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from core.supervisor.ipc import (
    IPC_CHUNK_MAX,
    IPCClient,
    IPCRequest,
    IPCResponse,
    IPCServer,
)


# ── Mock Writer ──────────────────────────────────────────────────


class MockWriter:
    """Mock asyncio.StreamWriter that records write calls."""

    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data: bytes):
        self.writes.append(data)

    async def drain(self):
        pass


class MockStreamWriter:
    """Mock StreamWriter for client-side tests (supports close/wait_closed)."""

    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data: bytes):
        self.writes.append(data)

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass

    def get_extra_info(self, key: str, default=None):
        return default


# ── IPC_CHUNK_MAX Constant ───────────────────────────────────────


def test_ipc_chunk_max_is_1mb():
    """IPC_CHUNK_MAX should be exactly 1MB."""
    assert IPC_CHUNK_MAX == 1 * 1024 * 1024


# ── _chunked_write Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_chunked_write_small_data():
    """Data <= IPC_CHUNK_MAX should be written in a single call."""
    writer = MockWriter()
    data = b"x" * 100  # Well under 1MB

    await IPCServer._chunked_write(writer, data)

    assert len(writer.writes) == 1
    assert writer.writes[0] == data


@pytest.mark.asyncio
async def test_chunked_write_exact_boundary():
    """Data == IPC_CHUNK_MAX should be written in a single call."""
    writer = MockWriter()
    data = b"A" * IPC_CHUNK_MAX  # Exactly 1MB

    await IPCServer._chunked_write(writer, data)

    assert len(writer.writes) == 1
    assert writer.writes[0] == data


@pytest.mark.asyncio
async def test_chunked_write_large_data():
    """Data > IPC_CHUNK_MAX should be split into multiple write calls."""
    writer = MockWriter()
    # 2.5 MB → should be split into 3 chunks (1MB + 1MB + 0.5MB)
    total_size = IPC_CHUNK_MAX * 2 + IPC_CHUNK_MAX // 2
    data = b"B" * total_size

    await IPCServer._chunked_write(writer, data)

    assert len(writer.writes) == 3
    assert len(writer.writes[0]) == IPC_CHUNK_MAX
    assert len(writer.writes[1]) == IPC_CHUNK_MAX
    assert len(writer.writes[2]) == IPC_CHUNK_MAX // 2
    # Concatenated writes should equal the original data
    assert b"".join(writer.writes) == data


@pytest.mark.asyncio
async def test_chunked_write_just_over_boundary():
    """Data that is 1 byte over IPC_CHUNK_MAX should produce exactly 2 writes."""
    writer = MockWriter()
    data = b"C" * (IPC_CHUNK_MAX + 1)

    await IPCServer._chunked_write(writer, data)

    assert len(writer.writes) == 2
    assert len(writer.writes[0]) == IPC_CHUNK_MAX
    assert len(writer.writes[1]) == 1
    assert b"".join(writer.writes) == data


@pytest.mark.asyncio
async def test_chunked_write_empty_data():
    """Empty data should be written in a single call (fast path)."""
    writer = MockWriter()
    data = b""

    await IPCServer._chunked_write(writer, data)

    # len(b"") == 0 <= IPC_CHUNK_MAX, so single write path
    assert len(writer.writes) == 1
    assert writer.writes[0] == b""


# ── First Chunk Timeout Tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_first_chunk_uses_generous_timeout():
    """First readline should use max(timeout, 120.0) — fast response succeeds."""
    client = IPCClient(Path("/tmp/test_first_chunk.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    # Feed a single non-streaming response to the reader
    response = IPCResponse(id="1", result={"ok": True})
    response_bytes = (response.to_json() + "\n").encode("utf-8")
    client.reader.feed_data(response_bytes)

    # With timeout=30.0, effective first_chunk_timeout = max(30.0, 120.0) = 120.0
    # The fast response should be read without hitting any timeout
    chunks = []
    async for chunk in client.send_request_stream(
        IPCRequest(id="1", method="test"), timeout=30.0
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].result == {"ok": True}


@pytest.mark.asyncio
async def test_first_chunk_timeout_respects_higher_value():
    """When user timeout > 120s, first_chunk_timeout should use that larger value."""
    client = IPCClient(Path("/tmp/test_higher_timeout.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    # Feed a non-streaming response
    response = IPCResponse(id="2", result={"data": "large"})
    response_bytes = (response.to_json() + "\n").encode("utf-8")
    client.reader.feed_data(response_bytes)

    # With timeout=300.0, effective first_chunk_timeout = max(300.0, 120.0) = 300.0
    chunks = []
    async for chunk in client.send_request_stream(
        IPCRequest(id="2", method="test"), timeout=300.0
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].result == {"data": "large"}


@pytest.mark.asyncio
async def test_subsequent_chunks_use_normal_timeout():
    """After the first chunk, subsequent reads should use the normal timeout."""
    client = IPCClient(Path("/tmp/test_subsequent.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    # Feed a streaming sequence: two chunks then a final done chunk
    chunk1 = IPCResponse(id="3", stream=True, chunk="Hello ")
    chunk2 = IPCResponse(id="3", stream=True, chunk="World")
    final = IPCResponse(id="3", stream=True, done=True, result={"total": 2})

    for resp in [chunk1, chunk2, final]:
        client.reader.feed_data((resp.to_json() + "\n").encode("utf-8"))

    chunks = []
    async for chunk in client.send_request_stream(
        IPCRequest(id="3", method="stream_test"), timeout=30.0
    ):
        chunks.append(chunk)

    # All three responses should be received
    assert len(chunks) == 3
    assert chunks[0].chunk == "Hello "
    assert chunks[1].chunk == "World"
    assert chunks[2].done is True
    assert chunks[2].result == {"total": 2}


@pytest.mark.asyncio
async def test_first_chunk_timeout_fires_on_no_data():
    """If no data arrives within the effective timeout, TimeoutError is raised.

    We monkeypatch asyncio.wait_for to capture the timeout values used for
    each readline call, then raise TimeoutError immediately so the test
    completes quickly.
    """
    from unittest.mock import patch

    client = IPCClient(Path("/tmp/test_timeout_fire.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    # Do NOT feed any data — readline will block forever without mock.
    captured_timeouts: list[float] = []

    async def mock_wait_for(coro, *, timeout):
        """Capture the timeout and raise TimeoutError immediately."""
        captured_timeouts.append(timeout)
        # Cancel the coroutine to avoid it dangling
        coro.close()
        raise asyncio.TimeoutError()

    with patch.object(asyncio, "wait_for", side_effect=mock_wait_for):
        with pytest.raises(asyncio.TimeoutError):
            async for _ in client.send_request_stream(
                IPCRequest(id="4", method="slow"), timeout=30.0
            ):
                pass  # pragma: no cover

    # The first (and only) call should use first_chunk_timeout = max(30.0, 120.0) = 120.0
    assert len(captured_timeouts) == 1
    assert captured_timeouts[0] == 120.0


@pytest.mark.asyncio
async def test_subsequent_chunk_timeout_value():
    """Verify that after the first chunk, the normal timeout is used for subsequent reads.

    We feed one streaming chunk, then monkeypatch wait_for to capture
    the timeout on the second read.
    """
    from unittest.mock import patch

    client = IPCClient(Path("/tmp/test_subsequent_timeout_val.sock"))
    client.reader = asyncio.StreamReader()
    client.writer = MockStreamWriter()

    # Feed one streaming chunk (not done) so the loop continues
    chunk1 = IPCResponse(id="5", stream=True, chunk="data")
    client.reader.feed_data((chunk1.to_json() + "\n").encode("utf-8"))

    captured_timeouts: list[float] = []
    call_count = 0
    original_wait_for = asyncio.wait_for

    async def mock_wait_for(coro, *, timeout):
        """First call: pass through. Second call: capture timeout and raise."""
        nonlocal call_count
        call_count += 1
        captured_timeouts.append(timeout)
        if call_count == 1:
            # Let the first call through (it will read chunk1)
            return await original_wait_for(coro, timeout=timeout)
        else:
            # Second call: capture and raise
            coro.close()
            raise asyncio.TimeoutError()

    with patch.object(asyncio, "wait_for", side_effect=mock_wait_for):
        chunks = []
        with pytest.raises(asyncio.TimeoutError):
            async for chunk in client.send_request_stream(
                IPCRequest(id="5", method="stream"), timeout=30.0
            ):
                chunks.append(chunk)

    # First chunk received successfully
    assert len(chunks) == 1
    assert chunks[0].chunk == "data"

    # Verify timeout values:
    # - First call (chunk_count==0): max(30.0, 120.0) = 120.0
    # - Second call (chunk_count==1): normal timeout = 30.0
    assert len(captured_timeouts) == 2
    assert captured_timeouts[0] == 120.0   # first_chunk_timeout
    assert captured_timeouts[1] == 30.0    # normal timeout
