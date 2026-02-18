from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("animaworks.stream_registry")


# ── Data Models ──────────────────────────────────────────


@dataclass
class SSEEvent:
    """A single buffered SSE event."""

    seq: int
    event: str
    payload: dict[str, Any]
    event_id: str  # "{response_id}:{seq}"


MAX_EVENTS = 2000


@dataclass
class ResponseStream:
    """Per-response stream state and event buffer."""

    response_id: str
    anima_name: str
    from_person: str = "human"
    created_at: float = field(default_factory=time.time)
    events: list[SSEEvent] = field(default_factory=list)
    complete: bool = False
    full_text: str = ""
    active_tool: str | None = None
    emotion: str | None = None
    _seq_counter: int = field(default=0, repr=False)
    _new_event: asyncio.Event = field(
        default_factory=asyncio.Event, repr=False,
    )

    def add_event(self, event: str, payload: dict[str, Any]) -> SSEEvent:
        """Append an event to the buffer and return it."""
        seq = self._seq_counter
        self._seq_counter += 1
        event_id = f"{self.response_id}:{seq}"

        sse_event = SSEEvent(
            seq=seq,
            event=event,
            payload=payload,
            event_id=event_id,
        )
        self.events.append(sse_event)

        # Evict oldest events if buffer exceeds limit
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]

        # Track accumulated state
        if event == "text_delta":
            self.full_text += payload.get("text", "")
        elif event == "tool_start":
            self.active_tool = payload.get("tool_name")
        elif event == "tool_end":
            self.active_tool = None
        elif event == "done":
            self.complete = True
            self.emotion = payload.get("emotion")
            summary = payload.get("summary")
            if summary:
                self.full_text = summary

        # Notify waiters
        self._new_event.set()
        self._new_event.clear()

        return sse_event

    def events_after(self, after_seq: int) -> list[SSEEvent]:
        """Return all events with seq > after_seq."""
        return [e for e in self.events if e.seq > after_seq]

    @property
    def current_seq(self) -> int:
        """The sequence number of the most recent event, or -1 if empty."""
        return self._seq_counter - 1 if self.events else -1

    async def wait_new_event(self, timeout: float = 30.0) -> bool:
        """Wait for a new event. Returns False on timeout."""
        try:
            await asyncio.wait_for(self._new_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def last_event_id(self) -> str | None:
        """The event ID of the most recent event, or None."""
        if not self.events:
            return None
        return self.events[-1].event_id

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def status(self) -> str:
        return "complete" if self.complete else "streaming"


# ── StreamRegistry ───────────────────────────────────────


class StreamRegistry:
    """In-memory registry of active and recently completed response streams.

    Stores per-response event buffers for SSE replay and progress polling.
    Entries are kept for TTL seconds after creation, then cleaned up.
    """

    TTL = 3600  # 1 hour

    def __init__(self) -> None:
        self._streams: dict[str, ResponseStream] = {}
        self._anima_active: dict[str, str] = {}  # anima_name -> response_id
        self._cleanup_task: asyncio.Task[None] | None = None

    def register(
        self, anima_name: str, *, from_person: str = "human",
    ) -> ResponseStream:
        """Create a new ResponseStream and register it."""
        response_id = secrets.token_urlsafe(24)
        stream = ResponseStream(
            response_id=response_id,
            anima_name=anima_name,
            from_person=from_person,
        )
        self._streams[response_id] = stream
        self._anima_active[anima_name] = response_id
        logger.debug(
            "Registered stream %s for anima=%s", response_id, anima_name,
        )
        return stream

    def get(self, response_id: str) -> ResponseStream | None:
        """Look up a stream by response ID."""
        return self._streams.get(response_id)

    def get_active(self, anima_name: str) -> ResponseStream | None:
        """Return the most recent (possibly still active) stream for an anima."""
        response_id = self._anima_active.get(anima_name)
        if response_id is None:
            return None
        return self._streams.get(response_id)

    def mark_complete(self, response_id: str) -> None:
        """Mark a stream as complete."""
        stream = self._streams.get(response_id)
        if stream:
            stream.complete = True
            # Clear the active mapping if this is still the active stream
            if self._anima_active.get(stream.anima_name) == response_id:
                self._anima_active.pop(stream.anima_name, None)

    def cleanup(self) -> int:
        """Remove expired streams. Returns count of removed entries."""
        now = time.time()
        expired = [
            rid for rid, s in self._streams.items()
            if now - s.created_at > self.TTL
        ]
        for rid in expired:
            stream = self._streams.pop(rid, None)
            if stream:
                # Clean up active mapping if it points to this expired stream
                if self._anima_active.get(stream.anima_name) == rid:
                    self._anima_active.pop(stream.anima_name, None)
        if expired:
            logger.info("Cleaned up %d expired stream(s)", len(expired))
        return len(expired)

    async def start_cleanup_loop(self, interval: float = 300) -> None:
        """Background loop that periodically cleans up expired streams."""
        async def _loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    self.cleanup()
                except Exception:
                    logger.exception("Stream cleanup error")

        self._cleanup_task = asyncio.create_task(_loop())

    async def stop_cleanup_loop(self) -> None:
        """Stop the background cleanup loop."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    def format_sse(
        self,
        stream: ResponseStream,
        event: str,
        payload: dict[str, Any],
    ) -> str:
        """Add an event to the stream buffer and return the formatted SSE frame."""
        sse_event = stream.add_event(event, payload)
        return format_sse_with_id(event, payload, sse_event.event_id)


def format_sse_with_id(
    event: str,
    payload: dict[str, Any],
    event_id: str,
) -> str:
    """Format a single SSE frame with id field."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"id: {event_id}\nevent: {event}\ndata: {data}\n\n"
