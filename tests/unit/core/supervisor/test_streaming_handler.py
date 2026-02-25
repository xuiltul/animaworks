"""Unit tests for StreamingIPCHandler."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.ipc import IPCRequest, IPCResponse
from core.supervisor.streaming_handler import StreamingIPCHandler


def _make_handler() -> StreamingIPCHandler:
    """Create a StreamingIPCHandler with minimal dependencies."""
    mock_anima = MagicMock()
    mock_anima.needs_bootstrap = False
    return StreamingIPCHandler(
        anima=mock_anima,
        anima_name="test-anima",
        anima_dir="/tmp/test",
    )


class TestStreamingIPCHandlerInit:
    """Test StreamingIPCHandler initialization."""

    def test_creates_instance(self):
        handler = _make_handler()
        assert handler._anima_name == "test-anima"

    def test_independent_instantiation(self):
        """StreamingIPCHandler can be instantiated without AnimaRunner."""
        handler = StreamingIPCHandler(
            anima=MagicMock(),
            anima_name="standalone",
            anima_dir="/tmp/standalone",
        )
        assert handler._anima_name == "standalone"


class TestStreamHandleNotInitialized:
    """Test handling when anima is not initialized."""

    @pytest.mark.asyncio
    async def test_yields_error_when_no_anima(self):
        """Should yield error response when anima is None."""
        handler = _make_handler()
        handler._anima = None

        request = IPCRequest(id="req-1", method="process_message", params={})

        responses = []
        async for resp in handler.handle_stream(request):
            responses.append(resp)

        assert len(responses) == 1
        assert responses[0].error is not None
        assert responses[0].error["code"] == "NOT_INITIALIZED"


class TestStreamHandleBasicFlow:
    """Test basic streaming flow."""

    @pytest.mark.asyncio
    async def test_text_delta_chunks(self):
        """Should yield text_delta chunks from stream."""
        handler = _make_handler()

        async def mock_stream(*args, **kwargs):
            yield {"type": "text_delta", "text": "Hello "}
            yield {"type": "text_delta", "text": "World"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Hello World"}}

        handler._anima.process_message_stream = mock_stream
        handler._anima.needs_bootstrap = False

        request = IPCRequest(
            id="req-1",
            method="process_message",
            params={"message": "test", "stream": True},
        )

        responses = []
        with patch("core.config.load_config") as mock_config:
            mock_config.return_value.server.keepalive_interval = 30
            async for resp in handler.handle_stream(request):
                responses.append(resp)

        # Should have text_delta chunks + done
        assert len(responses) >= 2
        # Last response should be done
        assert responses[-1].done is True
        assert responses[-1].result["response"] == "Hello World"

    @pytest.mark.asyncio
    async def test_error_in_stream(self):
        """Should handle errors in stream producer."""
        handler = _make_handler()

        async def mock_stream(*args, **kwargs):
            raise RuntimeError("stream error")
            yield  # noqa: unreachable - make it an async generator

        handler._anima.process_message_stream = mock_stream
        handler._anima.needs_bootstrap = False

        request = IPCRequest(
            id="req-1",
            method="process_message",
            params={"message": "test", "stream": True},
        )

        responses = []
        with patch("core.config.load_config") as mock_config:
            mock_config.return_value.server.keepalive_interval = 30
            async for resp in handler.handle_stream(request):
                responses.append(resp)

        assert len(responses) >= 1
        # Should have an error response
        error_responses = [r for r in responses if r.error is not None]
        assert len(error_responses) == 1
        assert error_responses[0].error["code"] == "STREAM_ERROR"

    @pytest.mark.asyncio
    async def test_forwards_intent_and_normalizes_none(self):
        """intent should be passed through and None should normalize to empty string."""
        handler = _make_handler()
        captured_kwargs = {}

        async def mock_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"type": "cycle_done", "cycle_result": {"summary": "ok"}}

        handler._anima.process_message_stream = mock_stream
        handler._anima.needs_bootstrap = False

        request = IPCRequest(
            id="req-1",
            method="process_message",
            params={"message": "test", "stream": True, "intent": None},
        )

        with patch("core.config.load_config") as mock_config:
            mock_config.return_value.server.keepalive_interval = 30
            async for _ in handler.handle_stream(request):
                pass

        assert captured_kwargs["intent"] == ""


class TestStreamHandleToolEndSerialization:
    """Test that tool_end events with ToolCallRecord are serialized safely."""

    @pytest.mark.asyncio
    async def test_tool_end_with_dataclass_record(self):
        """tool_end events containing a ToolCallRecord dataclass must not raise."""
        from core.execution.base import ToolCallRecord

        handler = _make_handler()

        async def mock_stream(*args, **kwargs):
            yield {"type": "tool_start", "tool_name": "read_memory_file"}
            yield {
                "type": "tool_end",
                "tool_id": "tc_1",
                "tool_name": "read_memory_file",
                "record": ToolCallRecord(
                    tool_name="read_memory_file",
                    tool_id="tc_1",
                    input_summary='{"path": "episodes/2026-02-25.md"}',
                    result_summary="(content of the file)",
                ),
            }
            yield {"type": "text_delta", "text": "Done"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Done"}}

        handler._anima.process_message_stream = mock_stream
        handler._anima.needs_bootstrap = False

        request = IPCRequest(
            id="req-1",
            method="process_message",
            params={"message": "read my memories", "stream": True},
        )

        responses: list[IPCResponse] = []
        with patch("core.config.load_config") as mock_config:
            mock_config.return_value.server.keepalive_interval = 30
            async for resp in handler.handle_stream(request):
                responses.append(resp)

        # Verify tool_end chunk was serialized successfully
        tool_end_chunks = [
            r for r in responses
            if r.chunk and '"tool_end"' in r.chunk
        ]
        assert len(tool_end_chunks) == 1

        parsed = json.loads(tool_end_chunks[0].chunk)
        assert parsed["record"]["tool_name"] == "read_memory_file"
        assert parsed["record"]["tool_id"] == "tc_1"

        # Final response must be done
        assert responses[-1].done is True
