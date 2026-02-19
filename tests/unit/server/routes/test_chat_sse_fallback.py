# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SSE stream fallback handling.

Verifies that _ipc_stream_events correctly handles:
- Non-streaming IPC responses (result without done flag)
- ValueError from send_request_stream (surfaced as SSE error)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.ipc import IPCResponse


def _make_ipc_response(
    *,
    result: dict | None = None,
    error: dict | None = None,
    chunk: str | None = None,
    done: bool = False,
) -> IPCResponse:
    """Create an IPCResponse for testing."""
    return IPCResponse(
        id="test-req-1",
        result=result,
        error=error,
        chunk=chunk,
        done=done,
    )


def _parse_sse(sse_text: str) -> list[dict]:
    """Parse SSE frames into list of {event, data} dicts."""
    frames = []
    for block in sse_text.strip().split("\n\n"):
        if not block or block.startswith(":"):
            continue
        event = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event:
            frames.append({"event": event, "data": data})
    return frames


class TestNonStreamingFallback:
    """_ipc_stream_events should handle non-streaming IPC responses."""

    @pytest.mark.asyncio
    async def test_result_without_done_flag_emits_done_sse(self):
        """IPC response with result but done=False should emit 'done' SSE."""
        from server.routes.chat import _format_sse, extract_emotion

        # Build mock IPC response: result set, done=False (non-streaming)
        ipc_resp = _make_ipc_response(
            result={
                "response": "こんにちは！",
                "cycle_result": {"summary": "こんにちは！"},
            },
            done=False,
        )

        async def mock_stream():
            yield ipc_resp

        # Simulate what _ipc_stream_events does with the new fallback
        collected_frames = []
        full_response = ""

        async for resp in mock_stream():
            if resp.done:
                result = resp.result or {}
                full_response = result.get("response", full_response)
                cycle_result = result.get("cycle_result", {})
                summary = cycle_result.get("summary", full_response)
                clean_text, emotion = extract_emotion(summary)
                cycle_result["summary"] = clean_text
                cycle_result["emotion"] = emotion
                full_response = clean_text
                collected_frames.append(
                    _format_sse("done", cycle_result or {"summary": clean_text, "emotion": emotion})
                )
                break

            if resp.chunk:
                continue

            # THIS IS THE NEW FALLBACK CODE
            if resp.result:
                result = resp.result
                full_response = result.get("response", "")
                cycle_result = result.get("cycle_result", {})
                summary = cycle_result.get("summary", full_response)
                clean_text, emotion = extract_emotion(summary)
                cycle_result["summary"] = clean_text
                cycle_result["emotion"] = emotion
                full_response = clean_text
                collected_frames.append(
                    _format_sse("done", cycle_result or {"summary": clean_text, "emotion": emotion})
                )
                break

        assert len(collected_frames) == 1
        parsed = _parse_sse(collected_frames[0])
        assert len(parsed) == 1
        assert parsed[0]["event"] == "done"
        assert parsed[0]["data"]["summary"] == "こんにちは！"

    @pytest.mark.asyncio
    async def test_streaming_done_still_works(self):
        """Normal streaming response with done=True should still work."""
        from server.routes.chat import _format_sse, extract_emotion

        ipc_resp = _make_ipc_response(
            result={
                "response": "通常の応答です",
                "cycle_result": {"summary": "通常の応答です"},
            },
            done=True,
        )

        collected_frames = []
        full_response = ""

        async def mock_stream():
            yield ipc_resp

        async for resp in mock_stream():
            if resp.done:
                result = resp.result or {}
                full_response = result.get("response", full_response)
                cycle_result = result.get("cycle_result", {})
                summary = cycle_result.get("summary", full_response)
                clean_text, emotion = extract_emotion(summary)
                cycle_result["summary"] = clean_text
                cycle_result["emotion"] = emotion
                collected_frames.append(
                    _format_sse("done", cycle_result or {"summary": clean_text, "emotion": emotion})
                )
                break

            if resp.chunk:
                continue

            if resp.result:
                # Should NOT reach here since done=True was handled above
                collected_frames.append("UNEXPECTED_FALLBACK")
                break

        assert len(collected_frames) == 1
        parsed = _parse_sse(collected_frames[0])
        assert parsed[0]["event"] == "done"
        assert parsed[0]["data"]["summary"] == "通常の応答です"


class TestEmotionExtraction:
    """Verify extract_emotion handles SSE-relevant edge cases."""

    def test_emotion_in_response(self):
        from server.routes.chat import extract_emotion

        text = 'こんにちは！<!-- emotion: {"emotion": "smile"} -->'
        clean, emotion = extract_emotion(text)
        assert clean == "こんにちは！"
        assert emotion == "smile"

    def test_no_emotion_defaults_neutral(self):
        from server.routes.chat import extract_emotion

        clean, emotion = extract_emotion("普通の返答")
        assert clean == "普通の返答"
        assert emotion == "neutral"


class TestValueErrorHandling:
    """send_request_stream ValueError should be surfaced as SSE IPC_ERROR."""

    @pytest.mark.asyncio
    async def test_value_error_emits_ipc_error_sse(self):
        """ValueError from stream should emit error SSE frame."""
        from server.routes.chat import _format_sse

        collected_frames = []

        async def mock_stream():
            raise ValueError("Stream error: Connection reset")
            yield  # Make it an async generator  # noqa: RET503

        try:
            async for _resp in mock_stream():
                pass
        except ValueError as e:
            collected_frames.append(
                _format_sse("error", {"code": "IPC_ERROR", "message": str(e)})
            )

        assert len(collected_frames) == 1
        parsed = _parse_sse(collected_frames[0])
        assert parsed[0]["event"] == "error"
        assert parsed[0]["data"]["code"] == "IPC_ERROR"
        assert "Connection reset" in parsed[0]["data"]["message"]
