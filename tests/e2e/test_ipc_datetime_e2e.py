# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
E2E tests for IPC datetime serialization fix.

Tests the full IPC server/client round-trip with datetime objects
in streaming responses, verifying the actual crash path is fixed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import pytest

from core.schemas import CycleResult
from core.supervisor.ipc import (
    IPCClient,
    IPCEvent,
    IPCRequest,
    IPCResponse,
    IPCServer,
)


@pytest.mark.asyncio
async def test_streaming_with_cycle_result_datetime():
    """E2E: Streaming response with CycleResult containing datetime survives IPC round-trip.

    This reproduces the original crash:
    - Agent yields CycleResult.model_dump() (with native datetime)
    - Runner puts it into IPCResponse.result
    - IPCResponse.to_json() serializes it
    - Client reads and deserializes
    """
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "cycle_result.sock"

        # Simulate the runner's behavior: build a response with cycle_result
        cycle_result = CycleResult(
            trigger="message:human",
            action="responded",
            summary="Hello from IPC",
            duration_ms=250,
            context_usage_ratio=0.3,
            session_chained=False,
            total_turns=1,
        ).model_dump()  # Note: mode="json" NOT used here to test defensive serialization

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            async def _gen() -> AsyncIterator[IPCResponse]:
                # Stream text chunks
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    chunk=json.dumps(
                        {"type": "text_delta", "text": "Hello "}, ensure_ascii=False
                    ),
                )
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    chunk=json.dumps(
                        {"type": "text_delta", "text": "from IPC"}, ensure_ascii=False
                    ),
                )
                # Final response with cycle_result containing datetime
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={
                        "response": "Hello from IPC",
                        "replied_to": [],
                        "cycle_result": cycle_result,
                    },
                )

            return _gen()

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="cycle_001",
                method="process_message",
                params={"message": "hi", "stream": True},
            )

            chunks: list[str] = []
            final_result = None

            async for response in client.send_request_stream(request, timeout=5.0):
                if response.done:
                    final_result = response.result
                elif response.chunk:
                    event = json.loads(response.chunk)
                    if event.get("type") == "text_delta":
                        chunks.append(event["text"])

            # Verify chunks came through
            assert chunks == ["Hello ", "from IPC"]

            # Verify final result
            assert final_result is not None
            assert final_result["response"] == "Hello from IPC"
            assert final_result["cycle_result"]["trigger"] == "message:human"
            assert final_result["cycle_result"]["summary"] == "Hello from IPC"
            # timestamp should be a string (serialized by default=str)
            assert isinstance(final_result["cycle_result"]["timestamp"], str)

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_streaming_with_model_dump_json_mode():
    """E2E: Streaming with CycleResult.model_dump(mode='json') — the recommended fix path."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "json_mode.sock"

        # This is the fixed path: model_dump(mode="json") converts datetime to ISO string
        cycle_result = CycleResult(
            trigger="message:user",
            action="responded",
            summary="Fixed response",
            duration_ms=100,
        ).model_dump(mode="json")

        async def handler(
            request: IPCRequest,
        ) -> Union[IPCResponse, AsyncIterator[IPCResponse]]:
            async def _gen() -> AsyncIterator[IPCResponse]:
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    chunk=json.dumps(
                        {"type": "text_delta", "text": "Fixed"}, ensure_ascii=False
                    ),
                )
                yield IPCResponse(
                    id=request.id,
                    stream=True,
                    done=True,
                    result={
                        "response": "Fixed response",
                        "replied_to": [],
                        "cycle_result": cycle_result,
                    },
                )

            return _gen()

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="fixed_001",
                method="process_message",
                params={"message": "test", "stream": True},
            )

            final_result = None
            async for response in client.send_request_stream(request, timeout=5.0):
                if response.done:
                    final_result = response.result

            assert final_result is not None
            assert final_result["cycle_result"]["trigger"] == "message:user"
            # With mode="json", timestamp is already an ISO string
            ts = final_result["cycle_result"]["timestamp"]
            assert isinstance(ts, str)
            # Should be a valid ISO 8601 datetime
            datetime.fromisoformat(ts)

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_non_streaming_response_with_datetime():
    """E2E: Non-streaming response with datetime in result dict."""
    with TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "non_stream_dt.sock"

        async def handler(request: IPCRequest) -> IPCResponse:
            return IPCResponse(
                id=request.id,
                result={
                    "response": "done",
                    "completed_at": datetime.now(),
                },
            )

        server = IPCServer(socket_path, handler)
        await server.start()

        try:
            client = IPCClient(socket_path)
            await client.connect()

            request = IPCRequest(
                id="ns_dt_001",
                method="get_status",
                params={},
            )

            response = await client.send_request(request, timeout=5.0)

            assert response.result is not None
            assert response.result["response"] == "done"
            assert isinstance(response.result["completed_at"], str)

            await client.close()

        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_event_with_datetime():
    """E2E: IPCEvent.to_json() with datetime in data should serialize correctly."""
    now = datetime.now()
    event = IPCEvent(
        event="anima_activity",
        data={
            "anima": "sakura",
            "last_activity": now,
            "status": "active",
        },
    )

    # Serialize and deserialize
    json_str = event.to_json()
    data = json.loads(json_str)

    assert data["event"] == "anima_activity"
    assert data["data"]["anima"] == "sakura"
    assert data["data"]["status"] == "active"
    assert isinstance(data["data"]["last_activity"], str)
    assert str(now) == data["data"]["last_activity"]
