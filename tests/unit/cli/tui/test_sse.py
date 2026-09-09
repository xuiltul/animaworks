# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cli.tui.sse import aiter_sse, parse_sse_lines


def _events(lines):
    return list(parse_sse_lines(lines))


def test_multiple_frames_with_ids():
    lines = [
        "id: abc:1",
        "event: stream_start",
        'data: {"response_id": "abc"}',
        "",
        "id: abc:2",
        "event: text_delta",
        'data: {"text": "hi"}',
        "",
    ]
    events = _events(lines)
    assert len(events) == 2
    assert events[0].id == "abc:1"
    assert events[0].event == "stream_start"
    assert events[0].data == {"response_id": "abc"}
    assert events[1].id == "abc:2"
    assert events[1].event == "text_delta"
    assert events[1].data == {"text": "hi"}


def test_keepalive_comments_ignored():
    lines = [
        ": keepalive",
        "event: done",
        'data: {"summary": "ok"}',
        "",
    ]
    events = _events(lines)
    assert len(events) == 1
    assert events[0].event == "done"
    assert events[0].data == {"summary": "ok"}


def test_frame_without_id():
    lines = [
        "event: error",
        'data: {"message": "boom"}',
        "",
    ]
    events = _events(lines)
    assert len(events) == 1
    assert events[0].id is None
    assert events[0].event == "error"


def test_non_json_data_wrapped_as_raw():
    lines = [
        "event: text_delta",
        "data: just some text",
        "",
    ]
    events = _events(lines)
    assert len(events) == 1
    assert events[0].data == {"raw": "just some text"}


def test_non_object_json_wrapped_as_raw():
    lines = [
        "event: x",
        "data: [1, 2, 3]",
        "",
    ]
    events = _events(lines)
    assert len(events) == 1
    assert events[0].data == {"raw": "[1, 2, 3]"}


def test_multiline_data_lines_joined():
    lines = [
        "event: tool_detail",
        'data: {"tool_id": "1",',
        'data:  "detail": "two"}',
        "",
    ]
    events = _events(lines)
    assert len(events) == 1
    assert events[0].data == {"tool_id": "1", "detail": "two"}


def test_lines_split_across_chunks_concat():
    # Frame parsed when the blank line arrives, even if fed incrementally.
    all_events = []
    buffer = []
    for line in [
        "event: text_delta",
        'data: {"text": "a"',
        "",
    ]:
        buffer.append(line)
        if line == "":
            all_events.extend(parse_sse_lines(buffer))
            buffer.clear()
    assert len(all_events) == 1
    assert all_events[0].event == "text_delta"
    assert all_events[0].data == {"raw": '{"text": "a"'}


@pytest.mark.asyncio
async def test_aiter_sse_buffers_partial_lines():
    class FakeResponse:
        def __init__(self, lines):
            self._lines = lines

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    resp = FakeResponse(
        [
            "event: text_delta",
            'data: {"text": "he',
            "",
            "event: done",
            'data: {"summary": "hi"}',
            "",
        ]
    )
    events = [ev async for ev in aiter_sse(resp)]
    assert len(events) == 2
    assert events[0].event == "text_delta"
    assert events[0].data == {"raw": '{"text": "he'}
    assert events[1].event == "done"
    assert events[1].data == {"summary": "hi"}
