# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for IPC keep-alive functionality.

Tests keep-alive chunk emission in AnimaRunner._handle_process_message_stream()
and per-chunk timeout behaviour in IPCClient.send_request_stream().
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.ipc import IPCClient, IPCRequest, IPCResponse
from core.supervisor.runner import (
    AnimaRunner,
    _DEFAULT_KEEPALIVE_INTERVAL,
    _SENTINEL,
    _Sentinel,
)


# ── Helpers ──────────────────────────────────────────────────


def _make_runner() -> AnimaRunner:
    """Create a AnimaRunner with minimal config for unit testing."""
    runner = AnimaRunner(
        anima_name="test",
        socket_path=Path("/tmp/test.sock"),
        animas_dir=Path("/tmp/animas"),
        shared_dir=Path("/tmp/shared"),
    )
    runner.anima = MagicMock()
    runner.anima.needs_bootstrap = False
    runner.anima._lock = asyncio.Lock()
    return runner


async def _collect_responses(
    aiter,
) -> list[IPCResponse]:
    """Drain an async iterator into a list."""
    results: list[IPCResponse] = []
    async for item in aiter:
        results.append(item)
    return results


# ── TestRunnerKeepalive ──────────────────────────────────────


class TestRunnerKeepalive:
    """Tests for keep-alive chunk emission in _handle_process_message_stream()."""

    async def test_keepalive_emitted_during_silence(self):
        """Keep-alive chunks are emitted when the Agent SDK stream is silent.

        Simulates a stream that produces no chunks for 3 seconds, with
        keepalive_interval set to 1 second. At least one keep-alive chunk
        with ``{"type": "keepalive", "elapsed_s": ...}`` should be yielded.
        """
        runner = _make_runner()

        async def _slow_stream(message, from_person="human", **kwargs):
            """Async generator that is silent for ~1.5s then emits cycle_done."""
            await asyncio.sleep(1.5)
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "done"},
            }

        runner.anima.process_message_stream = _slow_stream

        request = IPCRequest(
            id="ka_001",
            method="process_message",
            params={"message": "hello", "stream": True},
        )

        # Patch load_config to return keepalive_interval=0.5
        mock_config = MagicMock()
        mock_config.server.keepalive_interval = 0.5

        with patch(
            "core.config.load_config", return_value=mock_config
        ):
            responses = await _collect_responses(
                runner._handle_process_message_stream(request)
            )

        # Extract keep-alive chunks
        keepalive_chunks = []
        for resp in responses:
            if resp.chunk:
                data = json.loads(resp.chunk)
                if data.get("type") == "keepalive":
                    keepalive_chunks.append(data)

        assert len(keepalive_chunks) >= 1, (
            f"Expected at least 1 keepalive chunk, got {len(keepalive_chunks)}"
        )
        # Verify keepalive chunk structure
        for ka in keepalive_chunks:
            assert "elapsed_s" in ka
            assert isinstance(ka["elapsed_s"], (int, float))

    async def test_no_keepalive_when_chunks_flowing(self):
        """No keep-alive chunks when the Agent SDK stream is actively producing.

        Simulates a stream that emits 5 chunks at 0.1s intervals, with
        keepalive_interval set to 2 seconds. Since chunks arrive faster
        than the keepalive interval, no keep-alive should be emitted.
        """
        runner = _make_runner()

        async def _fast_stream(message, from_person="human", **kwargs):
            """Async generator that produces chunks rapidly."""
            for i in range(5):
                await asyncio.sleep(0.1)
                yield {"type": "text_delta", "text": f"chunk{i}"}
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "fast"},
            }

        runner.anima.process_message_stream = _fast_stream

        request = IPCRequest(
            id="ka_002",
            method="process_message",
            params={"message": "hi", "stream": True},
        )

        mock_config = MagicMock()
        mock_config.server.keepalive_interval = 2

        with patch(
            "core.config.load_config", return_value=mock_config
        ):
            responses = await _collect_responses(
                runner._handle_process_message_stream(request)
            )

        # Count keep-alive chunks
        keepalive_count = 0
        for resp in responses:
            if resp.chunk:
                data = json.loads(resp.chunk)
                if data.get("type") == "keepalive":
                    keepalive_count += 1

        assert keepalive_count == 0, (
            f"Expected 0 keepalive chunks when data is flowing, got {keepalive_count}"
        )

    async def test_keepalive_stops_after_stream_done(self):
        """Keep-alive task is cancelled after the stream completes.

        Simulates a stream that completes instantly. After the stream is
        done, no additional keep-alive chunks should arrive.
        """
        runner = _make_runner()

        async def _instant_stream(message, from_person="human", **kwargs):
            """Async generator that completes immediately."""
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "instant"},
            }

        runner.anima.process_message_stream = _instant_stream

        request = IPCRequest(
            id="ka_003",
            method="process_message",
            params={"message": "quick", "stream": True},
        )

        mock_config = MagicMock()
        mock_config.server.keepalive_interval = 0.3

        with patch(
            "core.config.load_config", return_value=mock_config
        ):
            responses = await _collect_responses(
                runner._handle_process_message_stream(request)
            )

        # The stream should have completed with a done response
        done_responses = [r for r in responses if r.done]
        assert len(done_responses) == 1

        # Wait a bit longer than keepalive_interval to ensure no more come
        # (If the keepalive task was not cancelled, it would still be running
        # and would have tried to put items on the queue, but since the
        # generator has exited, the finally block should have cancelled it.)
        await asyncio.sleep(0.5)

        # No keepalive chunks should be present at all for an instant stream
        keepalive_count = sum(
            1
            for r in responses
            if r.chunk and json.loads(r.chunk).get("type") == "keepalive"
        )
        assert keepalive_count == 0


# ── TestIPCPerChunkTimeout ───────────────────────────────────


class TestIPCPerChunkTimeout:
    """Tests for per-chunk timeout in IPCClient.send_request_stream()."""

    async def test_per_chunk_timeout_resets_on_data(self):
        """Per-chunk timeout resets on each received chunk.

        Uses a mock reader that returns 3 chunks at 0.3s intervals, with
        a per-chunk timeout of 0.8s. The total elapsed time (~0.9s) exceeds
        any single-shot timeout of 0.8s, proving the timeout resets per chunk.
        """
        client = IPCClient(socket_path=Path("/tmp/test.sock"))

        # Build mock reader/writer
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)
        client.reader = reader
        client.writer = writer
        client._lock = asyncio.Lock()

        # Prepare response lines: 3 stream chunks + 1 done
        chunk_responses = [
            IPCResponse(id="req_1", stream=True, chunk="c1"),
            IPCResponse(id="req_1", stream=True, chunk="c2"),
            IPCResponse(id="req_1", stream=True, chunk="c3"),
            IPCResponse(id="req_1", stream=True, done=True, result={"ok": True}),
        ]

        call_count = 0

        async def _readline_with_delay():
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx < len(chunk_responses):
                # Simulate 0.3s delay between each chunk
                await asyncio.sleep(0.3)
                return (chunk_responses[idx].to_json() + "\n").encode("utf-8")
            return b""

        reader.readline = _readline_with_delay

        request = IPCRequest(id="req_1", method="test", params={})

        # Per-chunk timeout of 0.8s — total time ~1.2s would fail with a
        # single-shot timeout of 0.8s, but per-chunk should succeed.
        collected: list[IPCResponse] = []
        async for resp in client.send_request_stream(request, timeout=0.8):
            collected.append(resp)

        assert len(collected) == 4
        assert collected[-1].done is True
        assert collected[-1].result == {"ok": True}

    async def test_per_chunk_timeout_fires_on_silence(self):
        """Per-chunk timeout fires when no data arrives.

        Uses a mock reader that returns one chunk, then blocks forever.
        With timeout=0.5s, a TimeoutError should be raised.
        """
        client = IPCClient(socket_path=Path("/tmp/test.sock"))

        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)
        client.reader = reader
        client.writer = writer
        client._lock = asyncio.Lock()

        first_response = IPCResponse(id="req_2", stream=True, chunk="first")
        call_count = 0

        async def _readline_block_after_first():
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return (first_response.to_json() + "\n").encode("utf-8")
            # Block forever (simulates a hung stream)
            await asyncio.sleep(999)
            return b""

        reader.readline = _readline_block_after_first

        request = IPCRequest(id="req_2", method="test", params={})

        with pytest.raises(TimeoutError):
            async for _ in client.send_request_stream(request, timeout=0.5):
                pass
