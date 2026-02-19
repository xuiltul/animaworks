# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for IPC datetime serialization fix.

Verifies that IPC protocol types correctly serialize non-primitive Python objects
(datetime, Path, etc.) via the `default=str` fallback in json.dumps().
Also verifies that CycleResult.model_dump(mode="json") produces ISO 8601 strings.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from core.schemas import CycleResult
from core.supervisor.ipc import IPCRequest, IPCResponse, IPCEvent


# ── CycleResult model_dump(mode="json") Tests ────────────────────────


class TestCycleResultJsonMode:
    """Verify CycleResult.model_dump(mode='json') serializes datetime to ISO string."""

    def test_model_dump_json_mode_timestamp_is_string(self):
        """timestamp should be an ISO 8601 string when mode='json'."""
        result = CycleResult(
            trigger="message:human",
            action="responded",
            summary="Hello",
            duration_ms=100,
        )
        dumped = result.model_dump(mode="json")

        assert isinstance(dumped["timestamp"], str)
        # Should be parseable as ISO 8601
        datetime.fromisoformat(dumped["timestamp"])

    def test_model_dump_without_json_mode_timestamp_is_datetime(self):
        """Without mode='json', timestamp should be a native datetime (baseline)."""
        result = CycleResult(
            trigger="test",
            action="test",
        )
        dumped = result.model_dump()

        assert isinstance(dumped["timestamp"], datetime)

    def test_model_dump_json_mode_is_json_serializable(self):
        """model_dump(mode='json') output should be fully JSON-serializable."""
        result = CycleResult(
            trigger="message:human",
            action="responded",
            summary="Test response",
            duration_ms=500,
            context_usage_ratio=0.42,
            session_chained=True,
            total_turns=3,
        )
        dumped = result.model_dump(mode="json")

        # This must not raise TypeError
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)

        assert parsed["trigger"] == "message:human"
        assert parsed["action"] == "responded"
        assert parsed["summary"] == "Test response"
        assert parsed["duration_ms"] == 500
        assert isinstance(parsed["timestamp"], str)


# ── IPCResponse.to_json() with datetime Tests ────────────────────────


class TestIPCResponseDatetimeSerialization:
    """Verify IPCResponse.to_json() handles datetime in result dict."""

    def test_response_with_datetime_in_result(self):
        """to_json() should not raise when result contains a datetime object."""
        now = datetime.now()
        response = IPCResponse(
            id="test_001",
            result={"timestamp": now, "value": "ok"},
        )

        # This must not raise TypeError
        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["id"] == "test_001"
        assert data["result"]["value"] == "ok"
        # datetime should be serialized as string
        assert isinstance(data["result"]["timestamp"], str)
        assert str(now) == data["result"]["timestamp"]

    def test_streaming_done_with_datetime_in_cycle_result(self):
        """Simulate the actual crash path: streaming done with CycleResult containing datetime."""
        cycle_result = CycleResult(
            trigger="message:human",
            action="responded",
            summary="Hello",
            duration_ms=100,
        ).model_dump()  # Without mode="json" — datetime remains native

        response = IPCResponse(
            id="stream_001",
            stream=True,
            done=True,
            result={
                "response": "Hello",
                "replied_to": [],
                "cycle_result": cycle_result,
            },
        )

        # This was the crash site. With default=str, it should work.
        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["stream"] is True
        assert data["done"] is True
        assert data["result"]["response"] == "Hello"
        assert isinstance(data["result"]["cycle_result"]["timestamp"], str)

    def test_response_with_path_in_result(self):
        """to_json() should handle Path objects via default=str."""
        response = IPCResponse(
            id="test_002",
            result={"path": Path("/tmp/test"), "status": "ok"},
        )

        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["result"]["path"] == "/tmp/test"
        assert data["result"]["status"] == "ok"

    def test_normal_response_unchanged(self):
        """Normal responses with primitive types should still work correctly."""
        response = IPCResponse(
            id="test_003",
            result={"count": 42, "name": "test", "active": True},
        )

        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["result"] == {"count": 42, "name": "test", "active": True}

    def test_error_response_unchanged(self):
        """Error responses should still work correctly."""
        response = IPCResponse(
            id="test_004",
            error={"code": "ERR", "message": "fail"},
        )

        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["error"] == {"code": "ERR", "message": "fail"}
        assert "result" not in data


# ── IPCRequest.to_json() with non-primitive params Tests ─────────────


class TestIPCRequestDatetimeSerialization:
    """Verify IPCRequest.to_json() handles non-primitive params."""

    def test_request_with_datetime_in_params(self):
        """to_json() should not raise when params contain a datetime object."""
        now = datetime.now()
        request = IPCRequest(
            id="req_001",
            method="test",
            params={"since": now},
        )

        json_str = request.to_json()
        data = json.loads(json_str)

        assert data["params"]["since"] == str(now)

    def test_normal_request_unchanged(self):
        """Normal requests with primitive types should still work correctly."""
        request = IPCRequest(
            id="req_002",
            method="echo",
            params={"message": "hello", "count": 5},
        )

        json_str = request.to_json()
        data = json.loads(json_str)

        assert data["params"] == {"message": "hello", "count": 5}


# ── IPCEvent.to_json() with non-primitive data Tests ─────────────────


class TestIPCEventDatetimeSerialization:
    """Verify IPCEvent.to_json() handles non-primitive data."""

    def test_event_with_datetime_in_data(self):
        """to_json() should not raise when data contains a datetime object."""
        now = datetime.now()
        event = IPCEvent(
            event="status_changed",
            data={"status": "active", "since": now},
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["data"]["status"] == "active"
        assert isinstance(data["data"]["since"], str)

    def test_normal_event_unchanged(self):
        """Normal events with primitive types should still work correctly."""
        event = IPCEvent(
            event="heartbeat",
            data={"status": "ok"},
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data == {"event": "heartbeat", "data": {"status": "ok"}}
