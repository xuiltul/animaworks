# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class SseEvent:
    """A single parsed Server-Sent Event frame.

    ``data`` is always a ``dict``. If the raw data payload was not valid
    JSON (or was not a JSON object), it is wrapped as ``{"raw": <str>}``.
    """

    event: str
    data: dict[str, Any]
    id: str | None = None


def _parse_data(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"raw": raw}


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SseEvent]:
    """Parse raw SSE lines into :class:`SseEvent` frames.

    ``lines`` is an iterable of individual lines (with or without
    trailing newlines). Frames are delimited by a blank line. Lines
    starting with ``:`` are comments and are ignored.
    """
    event_id: str | None = None
    event_name: str = ""
    data_lines: list[str] = []

    def flush() -> Iterator[SseEvent]:
        nonlocal event_id, event_name, data_lines
        if event_name or data_lines:
            yield SseEvent(
                event=event_name or "message",
                data=_parse_data("\n".join(data_lines)),
                id=event_id,
            )
        event_id = None
        event_name = ""
        data_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n").rstrip("\r")
        if not line:
            yield from flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip() or None
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        # Other SSE fields (retry, etc.) are ignored.

    yield from flush()


async def aiter_sse(response: httpx.Response) -> AsyncIterator[SseEvent]:
    """Asynchronously iterate :class:`SseEvent` frames from an httpx response.

    Buffers partial lines so a single frame split across multiple
    ``aiter_lines()`` chunks is still parsed correctly.
    """
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            for ev in parse_sse_lines(buffer):
                yield ev
            buffer.clear()
        else:
            buffer.append(line)
    if buffer:
        for ev in parse_sse_lines(buffer):
            yield ev
