"""Tests for core.execution._streaming — shared streaming helpers."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.execution._streaming import (
    accumulate_tool_call_chunks,
    parse_accumulated_tool_calls,
    stream_error_boundary,
)
from core.execution.base import StreamDisconnectedError


# ── Fake objects for LiteLLM delta simulation ─────────────────


class FakeFunction:
    """Mimics ``delta.tool_calls[].function`` from LiteLLM streaming chunks."""

    def __init__(self, name: str | None = None, arguments: str = "") -> None:
        self.name = name
        self.arguments = arguments


class FakeDeltaToolCall:
    """Mimics a single entry in ``delta.tool_calls`` from LiteLLM streaming."""

    def __init__(
        self,
        index: int,
        id: str | None = None,
        function: FakeFunction | None = None,
    ) -> None:
        self.index = index
        self.id = id
        self.function = function or FakeFunction()


# ── accumulate_tool_call_chunks ───────────────────────────────


class TestAccumulateToolCallChunksSingleTool:
    """First chunk creates entry with id/name; subsequent chunks accumulate arguments."""

    def test_first_chunk_creates_entry(self) -> None:
        acc: dict[int, dict] = {}
        delta = [FakeDeltaToolCall(index=0, id="call_1", function=FakeFunction(name="read_file"))]
        new = accumulate_tool_call_chunks(acc, delta)

        assert 0 in acc
        assert acc[0]["id"] == "call_1"
        assert acc[0]["name"] == "read_file"
        assert acc[0]["arguments"] == ""
        assert new == ["read_file"]

    def test_subsequent_chunks_accumulate_arguments(self) -> None:
        acc: dict[int, dict] = {}

        # First chunk: name + id
        accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, id="call_1", function=FakeFunction(name="read_file")),
        ])
        # Second chunk: partial arguments
        accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, function=FakeFunction(arguments='{"path":')),
        ])
        # Third chunk: rest of arguments
        accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, function=FakeFunction(arguments=' "/tmp/f"}')),
        ])

        assert acc[0]["arguments"] == '{"path": "/tmp/f"}'

    def test_returns_name_only_on_first_encounter(self) -> None:
        acc: dict[int, dict] = {}

        new1 = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, id="call_1", function=FakeFunction(name="search")),
        ])
        new2 = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, function=FakeFunction(arguments='{"q":"x"}')),
        ])

        assert new1 == ["search"]
        assert new2 == []


class TestAccumulateToolCallChunksMultipleTools:
    """Multiple tool_calls with different indices are tracked independently."""

    def test_two_tools_different_indices(self) -> None:
        acc: dict[int, dict] = {}

        # First tool
        new1 = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, id="call_A", function=FakeFunction(name="read_file")),
        ])
        # Second tool
        new2 = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=1, id="call_B", function=FakeFunction(name="write_file")),
        ])
        # Arguments for first
        accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, function=FakeFunction(arguments='{"a":1}')),
        ])
        # Arguments for second
        accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=1, function=FakeFunction(arguments='{"b":2}')),
        ])

        assert len(acc) == 2
        assert acc[0]["name"] == "read_file"
        assert acc[0]["arguments"] == '{"a":1}'
        assert acc[1]["name"] == "write_file"
        assert acc[1]["arguments"] == '{"b":2}'
        assert new1 == ["read_file"]
        assert new2 == ["write_file"]


class TestAccumulateToolCallChunksReturnsNewToolNames:
    """Only newly discovered tool names are returned each call."""

    def test_already_known_tool_not_returned(self) -> None:
        acc: dict[int, dict] = {}

        # First encounter
        new = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, id="c1", function=FakeFunction(name="tool_a")),
            FakeDeltaToolCall(index=1, id="c2", function=FakeFunction(name="tool_b")),
        ])
        assert sorted(new) == ["tool_a", "tool_b"]

        # Subsequent chunks for same indices — no new names
        new2 = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, function=FakeFunction(arguments="{}")),
            FakeDeltaToolCall(index=1, function=FakeFunction(arguments="{}")),
        ])
        assert new2 == []


class TestAccumulateToolCallChunksEmptyName:
    """An empty name is not included in the new_tools list."""

    def test_empty_name_excluded(self) -> None:
        acc: dict[int, dict] = {}
        new = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, id="call_x", function=FakeFunction(name="")),
        ])
        assert new == []
        assert acc[0]["name"] == ""

    def test_none_name_excluded(self) -> None:
        acc: dict[int, dict] = {}
        new = accumulate_tool_call_chunks(acc, [
            FakeDeltaToolCall(index=0, id="call_y", function=FakeFunction(name=None)),
        ])
        assert new == []
        assert acc[0]["name"] == ""


# ── parse_accumulated_tool_calls ──────────────────────────────


class TestParseAccumulatedToolCallsValidJson:
    """Parses arguments JSON correctly into Python dicts."""

    def test_valid_json_parsed(self) -> None:
        acc = {
            0: {"id": "c1", "name": "read_file", "arguments": '{"path": "/tmp/test.txt"}'},
        }
        result = parse_accumulated_tool_calls(acc)

        assert len(result) == 1
        assert result[0]["id"] == "c1"
        assert result[0]["name"] == "read_file"
        assert result[0]["arguments"] == {"path": "/tmp/test.txt"}
        assert result[0]["raw_arguments"] is None


class TestParseAccumulatedToolCallsInvalidJson:
    """Sets raw_arguments when JSON parse fails."""

    def test_invalid_json_sets_raw_arguments(self) -> None:
        acc = {
            0: {"id": "c1", "name": "broken", "arguments": "{not valid json"},
        }
        result = parse_accumulated_tool_calls(acc)

        assert len(result) == 1
        assert result[0]["arguments"] is None
        assert result[0]["raw_arguments"] == "{not valid json"

    def test_empty_string_arguments_sets_raw(self) -> None:
        acc = {
            0: {"id": "c1", "name": "empty_args", "arguments": ""},
        }
        result = parse_accumulated_tool_calls(acc)

        assert result[0]["arguments"] is None
        assert result[0]["raw_arguments"] == ""


class TestParseAccumulatedToolCallsSortedByIndex:
    """Results are sorted by index, not insertion order."""

    def test_sorted_output(self) -> None:
        acc = {
            2: {"id": "c3", "name": "third", "arguments": '{"n":3}'},
            0: {"id": "c1", "name": "first", "arguments": '{"n":1}'},
            1: {"id": "c2", "name": "second", "arguments": '{"n":2}'},
        }
        result = parse_accumulated_tool_calls(acc)

        assert len(result) == 3
        assert result[0]["name"] == "first"
        assert result[1]["name"] == "second"
        assert result[2]["name"] == "third"
        assert result[0]["arguments"]["n"] == 1
        assert result[1]["arguments"]["n"] == 2
        assert result[2]["arguments"]["n"] == 3


# ── stream_error_boundary ─────────────────────────────────────


class TestStreamErrorBoundaryPassesOnSuccess:
    """No exception means no effect — the context manager is transparent."""

    @pytest.mark.asyncio
    async def test_success_passthrough(self) -> None:
        parts: list[str] = ["hello"]
        async with stream_error_boundary(parts, executor_name="test"):
            parts.append("world")

        assert parts == ["hello", "world"]


class TestStreamErrorBoundaryWrapsGenericError:
    """Converts a plain Exception to StreamDisconnectedError with partial_text."""

    @pytest.mark.asyncio
    async def test_wraps_exception(self) -> None:
        parts = ["partial", "response"]

        with pytest.raises(StreamDisconnectedError) as exc_info:
            async with stream_error_boundary(parts, executor_name="A2"):
                raise RuntimeError("connection reset")

        err = exc_info.value
        assert err.partial_text == "partial\nresponse"
        assert "A2 stream error" in str(err)
        assert "connection reset" in str(err)
        # Original exception is chained
        assert isinstance(err.__cause__, RuntimeError)


class TestStreamErrorBoundaryPassesStreamDisconnected:
    """An existing StreamDisconnectedError is re-raised without wrapping."""

    @pytest.mark.asyncio
    async def test_passthrough_stream_disconnected(self) -> None:
        parts = ["some", "data"]
        original = StreamDisconnectedError(
            "already wrapped", partial_text="original_partial",
        )

        with pytest.raises(StreamDisconnectedError) as exc_info:
            async with stream_error_boundary(parts, executor_name="A2"):
                raise original

        caught = exc_info.value
        # The exact same exception object is re-raised
        assert caught is original
        assert caught.partial_text == "original_partial"
        assert str(caught) == "already wrapped"
