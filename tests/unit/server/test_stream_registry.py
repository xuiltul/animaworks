# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Unit tests for server.stream_registry module.

Covers ResponseStream, StreamRegistry, and format_sse_with_id.
"""

import asyncio
import json
import time

import pytest

from server.stream_registry import (
    MAX_EVENTS,
    ResponseStream,
    SSEEvent,
    StreamRegistry,
    format_sse_with_id,
)


# ── Helpers ──────────────────────────────────────────────


def _make_stream(
    response_id: str = "abc12345",
    anima_name: str = "alice",
    **kwargs,
) -> ResponseStream:
    """Create a ResponseStream with deterministic defaults."""
    return ResponseStream(response_id=response_id, anima_name=anima_name, **kwargs)


# ── ResponseStream ───────────────────────────────────────


class TestResponseStreamAddEvent:
    """Tests for ResponseStream.add_event."""

    def test_appends_event_with_correct_seq(self):
        stream = _make_stream()
        e0 = stream.add_event("text_delta", {"text": "hi"})
        e1 = stream.add_event("text_delta", {"text": " there"})

        assert e0.seq == 0
        assert e1.seq == 1
        assert len(stream.events) == 2

    def test_event_id_format(self):
        stream = _make_stream(response_id="resp0001")
        e0 = stream.add_event("text_delta", {"text": "a"})
        e1 = stream.add_event("text_delta", {"text": "b"})

        assert e0.event_id == "resp0001:0"
        assert e1.event_id == "resp0001:1"

    def test_tracks_full_text_for_text_delta(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "Hello"})
        stream.add_event("text_delta", {"text": ", world!"})

        assert stream.full_text == "Hello, world!"

    def test_full_text_ignores_non_text_delta_events(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "start"})
        stream.add_event("tool_start", {"tool_name": "web_search"})
        stream.add_event("tool_end", {"tool_name": "web_search"})
        stream.add_event("text_delta", {"text": " end"})

        assert stream.full_text == "start end"

    def test_tracks_active_tool_on_tool_start(self):
        stream = _make_stream()
        assert stream.active_tool is None

        stream.add_event("tool_start", {"tool_name": "web_search"})
        assert stream.active_tool == "web_search"

    def test_clears_active_tool_on_tool_end(self):
        stream = _make_stream()
        stream.add_event("tool_start", {"tool_name": "web_search"})
        assert stream.active_tool == "web_search"

        stream.add_event("tool_end", {"tool_name": "web_search"})
        assert stream.active_tool is None

    def test_marks_complete_on_done_event(self):
        stream = _make_stream()
        assert stream.complete is False

        stream.add_event("done", {"status": "ok"})
        assert stream.complete is True

    def test_done_event_captures_emotion(self):
        stream = _make_stream()
        stream.add_event("done", {"emotion": "happy"})

        assert stream.emotion == "happy"

    def test_done_event_replaces_full_text_with_summary(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "long accumulated text"})
        stream.add_event("done", {"summary": "short summary"})

        assert stream.full_text == "short summary"

    def test_done_event_without_summary_preserves_full_text(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "accumulated"})
        stream.add_event("done", {})

        assert stream.full_text == "accumulated"

    def test_returns_sse_event_object(self):
        stream = _make_stream()
        result = stream.add_event("text_delta", {"text": "x"})

        assert isinstance(result, SSEEvent)
        assert result.event == "text_delta"
        assert result.payload == {"text": "x"}


class TestResponseStreamEventsAfter:
    """Tests for ResponseStream.events_after."""

    def test_returns_events_after_given_seq(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "a"})
        stream.add_event("text_delta", {"text": "b"})
        stream.add_event("text_delta", {"text": "c"})

        after = stream.events_after(0)
        assert len(after) == 2
        assert after[0].seq == 1
        assert after[1].seq == 2

    def test_returns_all_events_when_after_negative(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "a"})
        stream.add_event("text_delta", {"text": "b"})

        after = stream.events_after(-1)
        assert len(after) == 2

    def test_returns_empty_when_no_events_after(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "a"})

        after = stream.events_after(0)
        assert after == []

    def test_returns_empty_on_empty_stream(self):
        stream = _make_stream()
        assert stream.events_after(0) == []


class TestResponseStreamProperties:
    """Tests for ResponseStream properties: last_event_id, event_count, status."""

    def test_last_event_id_none_when_empty(self):
        stream = _make_stream()
        assert stream.last_event_id is None

    def test_last_event_id_returns_most_recent(self):
        stream = _make_stream(response_id="r1")
        stream.add_event("text_delta", {"text": "a"})
        stream.add_event("text_delta", {"text": "b"})

        assert stream.last_event_id == "r1:1"

    def test_event_count_zero_initially(self):
        stream = _make_stream()
        assert stream.event_count == 0

    def test_event_count_increments(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "a"})
        stream.add_event("text_delta", {"text": "b"})
        stream.add_event("done", {})

        assert stream.event_count == 3

    def test_status_streaming_by_default(self):
        stream = _make_stream()
        assert stream.status == "streaming"

    def test_status_complete_after_done(self):
        stream = _make_stream()
        stream.add_event("done", {})
        assert stream.status == "complete"

    def test_current_seq_minus_one_when_empty(self):
        stream = _make_stream()
        assert stream.current_seq == -1

    def test_current_seq_tracks_last_event(self):
        stream = _make_stream()
        stream.add_event("text_delta", {"text": "a"})
        assert stream.current_seq == 0
        stream.add_event("text_delta", {"text": "b"})
        assert stream.current_seq == 1

    def test_stores_from_person(self):
        stream = _make_stream(from_person="taro")
        assert stream.from_person == "taro"

    def test_from_person_defaults_to_human(self):
        stream = _make_stream()
        assert stream.from_person == "human"


class TestResponseStreamBufferLimit:
    """Tests for MAX_EVENTS buffer eviction."""

    def test_evicts_oldest_events_when_exceeding_limit(self):
        stream = _make_stream()
        for i in range(MAX_EVENTS + 100):
            stream.add_event("text_delta", {"text": str(i)})

        assert len(stream.events) == MAX_EVENTS
        assert stream.events[0].seq == 100
        assert stream.events[-1].seq == MAX_EVENTS + 99

    def test_no_eviction_when_under_limit(self):
        stream = _make_stream()
        for i in range(10):
            stream.add_event("text_delta", {"text": str(i)})

        assert len(stream.events) == 10
        assert stream.events[0].seq == 0

    def test_events_after_works_with_evicted_buffer(self):
        stream = _make_stream()
        for i in range(MAX_EVENTS + 50):
            stream.add_event("text_delta", {"text": str(i)})

        result = stream.events_after(49)
        assert len(result) == MAX_EVENTS
        assert result[0].seq == 50

        result = stream.events_after(MAX_EVENTS + 40)
        assert len(result) == 9


class TestResponseStreamWaitNewEvent:
    """Tests for wait_new_event async method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_event_arrives(self):
        stream = _make_stream()

        async def _add_after_delay():
            await asyncio.sleep(0.01)
            stream.add_event("text_delta", {"text": "x"})

        task = asyncio.create_task(_add_after_delay())
        result = await stream.wait_new_event(timeout=2.0)
        assert result is True
        await task

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        stream = _make_stream()
        result = await stream.wait_new_event(timeout=0.01)
        assert result is False


# ── StreamRegistry ───────────────────────────────────────


class TestStreamRegistryRegister:
    """Tests for StreamRegistry.register."""

    def test_creates_new_response_stream(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        assert isinstance(stream, ResponseStream)
        assert stream.anima_name == "alice"
        assert len(stream.response_id) > 8

    def test_stores_from_person(self):
        registry = StreamRegistry()
        stream = registry.register("alice", from_person="taro")

        assert stream.from_person == "taro"

    def test_from_person_defaults_to_human(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        assert stream.from_person == "human"

    def test_unique_ids_for_multiple_registrations(self):
        registry = StreamRegistry()
        s1 = registry.register("alice")
        s2 = registry.register("bob")

        assert s1.response_id != s2.response_id

    def test_registered_stream_retrievable_by_get(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        assert registry.get(stream.response_id) is stream


class TestStreamRegistryGet:
    """Tests for StreamRegistry.get."""

    def test_returns_registered_stream(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        result = registry.get(stream.response_id)
        assert result is stream

    def test_returns_none_for_unknown_id(self):
        registry = StreamRegistry()
        assert registry.get("nonexistent") is None


class TestStreamRegistryGetActive:
    """Tests for StreamRegistry.get_active."""

    def test_returns_active_stream_for_anima(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        active = registry.get_active("alice")
        assert active is stream

    def test_returns_none_for_unknown_anima(self):
        registry = StreamRegistry()
        assert registry.get_active("unknown") is None

    def test_returns_latest_stream_when_multiple_registered(self):
        registry = StreamRegistry()
        _s1 = registry.register("alice")
        s2 = registry.register("alice")

        active = registry.get_active("alice")
        assert active is s2

    def test_returns_none_after_mark_complete(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        registry.mark_complete(stream.response_id)
        assert registry.get_active("alice") is None


class TestStreamRegistryMarkComplete:
    """Tests for StreamRegistry.mark_complete."""

    def test_sets_complete_flag(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        registry.mark_complete(stream.response_id)
        assert stream.complete is True

    def test_clears_active_mapping(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        registry.mark_complete(stream.response_id)
        assert registry.get_active("alice") is None

    def test_does_not_clear_active_if_newer_stream_registered(self):
        registry = StreamRegistry()
        s1 = registry.register("alice")
        s2 = registry.register("alice")

        registry.mark_complete(s1.response_id)
        assert registry.get_active("alice") is s2

    def test_no_error_for_unknown_response_id(self):
        registry = StreamRegistry()
        registry.mark_complete("nonexistent")

    def test_stream_still_retrievable_by_get_after_complete(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        registry.mark_complete(stream.response_id)
        assert registry.get(stream.response_id) is stream


class TestStreamRegistryCleanup:
    """Tests for StreamRegistry.cleanup (TTL-based expiration)."""

    def test_removes_expired_streams(self, monkeypatch):
        registry = StreamRegistry()
        stream = registry.register("alice")

        monkeypatch.setattr(stream, "created_at", time.time() - registry.TTL - 1)

        removed = registry.cleanup()
        assert removed == 1
        assert registry.get(stream.response_id) is None

    def test_keeps_non_expired_streams(self, monkeypatch):
        registry = StreamRegistry()
        stream = registry.register("alice")

        removed = registry.cleanup()
        assert removed == 0
        assert registry.get(stream.response_id) is stream

    def test_clears_active_mapping_for_expired_stream(self, monkeypatch):
        registry = StreamRegistry()
        stream = registry.register("alice")

        monkeypatch.setattr(stream, "created_at", time.time() - registry.TTL - 1)

        registry.cleanup()
        assert registry.get_active("alice") is None

    def test_mixed_expired_and_fresh(self, monkeypatch):
        registry = StreamRegistry()
        old_stream = registry.register("alice")
        fresh_stream = registry.register("bob")

        monkeypatch.setattr(
            old_stream, "created_at", time.time() - registry.TTL - 1,
        )

        removed = registry.cleanup()
        assert removed == 1
        assert registry.get(old_stream.response_id) is None
        assert registry.get(fresh_stream.response_id) is fresh_stream

    def test_returns_zero_when_nothing_to_clean(self):
        registry = StreamRegistry()
        assert registry.cleanup() == 0


class TestStreamRegistryFormatSSE:
    """Tests for StreamRegistry.format_sse."""

    def test_adds_event_to_stream_buffer(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        registry.format_sse(stream, "text_delta", {"text": "hello"})

        assert stream.event_count == 1
        assert stream.events[0].event == "text_delta"

    def test_returns_formatted_sse_string(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        result = registry.format_sse(stream, "text_delta", {"text": "hi"})

        assert result.startswith("id: ")
        assert "event: text_delta" in result
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_sse_id_matches_event_id(self):
        registry = StreamRegistry()
        stream = registry.register("alice")

        result = registry.format_sse(stream, "text_delta", {"text": "x"})
        event_id = stream.events[0].event_id

        assert result.startswith(f"id: {event_id}\n")


# ── format_sse_with_id ─────────────────────────────────


class TestFormatSSEWithId:
    """Tests for the format_sse_with_id helper function."""

    def test_correct_sse_format(self):
        result = format_sse_with_id("text_delta", {"text": "hello"}, "resp:0")

        lines = result.split("\n")
        assert lines[0] == "id: resp:0"
        assert lines[1] == "event: text_delta"
        assert lines[2].startswith("data: ")
        assert result.endswith("\n\n")

    def test_data_is_valid_json(self):
        payload = {"text": "hello", "count": 42}
        result = format_sse_with_id("test_event", payload, "id:1")

        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data_json = data_line[len("data: "):]
        parsed = json.loads(data_json)

        assert parsed == payload

    def test_handles_unicode_payload(self):
        payload = {"text": "こんにちは世界"}
        result = format_sse_with_id("text_delta", payload, "id:0")

        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data_json = data_line[len("data: "):]
        parsed = json.loads(data_json)

        assert parsed["text"] == "こんにちは世界"

    def test_handles_empty_payload(self):
        result = format_sse_with_id("done", {}, "resp:5")

        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data_json = data_line[len("data: "):]
        parsed = json.loads(data_json)

        assert parsed == {}

    def test_different_event_types(self):
        for event_type in ("text_delta", "tool_start", "tool_end", "done", "error"):
            result = format_sse_with_id(event_type, {}, "id:0")
            assert f"event: {event_type}" in result


# ── _handle_resume integration ────────────────────────


class TestHandleResumeLastEventId:
    """Tests for _handle_resume correctly parsing last_event_id into after_seq."""

    def _make_registry_with_events(self, n_events: int = 5) -> tuple:
        """Create a registry with a stream containing n_events."""
        registry = StreamRegistry()
        stream = registry.register("alice", from_person="human")
        for i in range(n_events):
            stream.add_event("text_delta", {"text": f"chunk{i}"})
        return registry, stream

    @pytest.mark.asyncio
    async def test_last_event_id_parsed_to_after_seq(self):
        """Verify that last_event_id 'resp:2' results in replaying only events after seq 2."""
        from server.routes.chat import _handle_resume

        registry, stream = self._make_registry_with_events(5)
        response_id = stream.response_id
        stream.add_event("done", {"summary": "done"})

        # Resume from after seq 2 — should replay seq 3, 4, and done (seq 5)
        result = _handle_resume(
            registry, response_id, f"{response_id}:2", "alice", from_person="human",
        )
        assert result.status_code == 200
        assert result.media_type == "text/event-stream"

        frames = await _collect_sse(result)

        # Should contain events with seq > 2 (i.e., seq 3, 4, 5)
        event_ids = _extract_event_ids(frames)
        assert all(int(eid.split(":")[1]) > 2 for eid in event_ids)
        assert len(event_ids) == 3  # seq 3, 4, 5 (done)

    @pytest.mark.asyncio
    async def test_empty_last_event_id_replays_all(self):
        """With empty last_event_id, all events should be replayed."""
        from server.routes.chat import _handle_resume

        registry, stream = self._make_registry_with_events(3)
        stream.add_event("done", {})

        result = _handle_resume(
            registry, stream.response_id, "", "alice", from_person="human",
        )

        frames = await _collect_sse(result)
        event_ids = _extract_event_ids(frames)
        assert len(event_ids) == 4  # seq 0, 1, 2, 3 (done)

    @pytest.mark.asyncio
    async def test_wrong_from_person_returns_error(self):
        """Resume with wrong from_person should return error SSE."""
        from server.routes.chat import _handle_resume

        registry, stream = self._make_registry_with_events(2)

        result = _handle_resume(
            registry, stream.response_id, "", "alice", from_person="attacker",
        )

        frames = await _collect_sse(result)
        combined = "".join(frames)
        assert "STREAM_NOT_FOUND" in combined

    @pytest.mark.asyncio
    async def test_unknown_response_id_returns_error(self):
        """Resume with unknown response_id should return error SSE."""
        from server.routes.chat import _handle_resume

        registry = StreamRegistry()

        result = _handle_resume(
            registry, "nonexistent", "", "alice", from_person="human",
        )

        frames = await _collect_sse(result)
        combined = "".join(frames)
        assert "STREAM_NOT_FOUND" in combined

    def test_chat_request_last_event_id_field_recognized(self):
        """Verify ChatRequest.last_event_id is a recognized Pydantic field."""
        from server.routes.chat import ChatRequest

        req = ChatRequest(
            message="", resume="resp123", last_event_id="resp123:42",
        )
        assert req.last_event_id == "resp123:42"
        assert "last_event_id" in ChatRequest.model_fields


async def _collect_sse(response) -> list[str]:
    """Collect all SSE frames from a StreamingResponse."""
    frames = []
    async for chunk in response.body_iterator:
        frames.append(chunk)
    return frames


def _extract_event_ids(frames: list[str]) -> list[str]:
    """Extract event IDs from SSE frames."""
    ids = []
    for frame in frames:
        for line in frame.split("\n"):
            if line.startswith("id: "):
                ids.append(line[4:].strip())
    return ids
