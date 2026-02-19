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
    _notify_seq: int = field(default=0, repr=False)

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
            logger.info(
                "[SSE-BUF] evict oldest events stream=%s buffer=%d->%d",
                self.response_id, len(self.events), MAX_EVENTS,
            )
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

        # Log every event added (non-text_delta at INFO, text_delta at DEBUG to avoid flood)
        if event != "text_delta":
            logger.info(
                "[SSE-BUF] add_event stream=%s seq=%d event=%s buf_size=%d",
                self.response_id, seq, event, len(self.events),
            )
        else:
            logger.debug(
                "[SSE-BUF] add_event stream=%s seq=%d event=text_delta delta_len=%d total_text=%d",
                self.response_id, seq, len(payload.get("text", "")), len(self.full_text),
            )

        # Notify waiters — increment seq THEN set event (waiter checks seq)
        self._notify_seq += 1
        self._new_event.set()
        self._new_event.clear()

        return sse_event

    def events_after(self, after_seq: int) -> list[SSEEvent]:
        """Return all events with seq > after_seq."""
        result = [e for e in self.events if e.seq > after_seq]
        if result:
            logger.info(
                "[SSE-BUF] events_after stream=%s after_seq=%d found=%d",
                self.response_id, after_seq, len(result),
            )
        return result

    @property
    def current_seq(self) -> int:
        """The sequence number of the most recent event, or -1 if empty."""
        return self._seq_counter - 1 if self.events else -1

    async def wait_new_event(self, timeout: float = 30.0) -> bool:
        """Wait for a new event. Returns False on timeout.

        Uses a sequence counter to avoid the race where set()+clear()
        fires before the waiter's await runs.  Even if the Event is
        already cleared, the changed ``_notify_seq`` tells us a new
        event arrived.
        """
        logger.info(
            "[SSE-WAIT] wait_new_event stream=%s timeout=%.1fs complete=%s seq=%d",
            self.response_id, timeout, self.complete, self._seq_counter - 1,
        )
        seen_seq = self._notify_seq
        deadline = asyncio.get_event_loop().time() + timeout
        try:
            while self._notify_seq == seen_seq:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.info(
                        "[SSE-WAIT] timeout stream=%s after=%.1fs complete=%s",
                        self.response_id, timeout, self.complete,
                    )
                    return False
                await asyncio.wait_for(self._new_event.wait(), timeout=remaining)
            logger.info(
                "[SSE-WAIT] got_event stream=%s new_seq=%d",
                self.response_id, self._seq_counter - 1,
            )
            return True
        except asyncio.TimeoutError:
            logger.info(
                "[SSE-WAIT] timeout stream=%s after=%.1fs complete=%s",
                self.response_id, timeout, self.complete,
            )
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
        logger.info(
            "[SSE-REG] register stream=%s anima=%s from=%s total_streams=%d",
            response_id, anima_name, from_person, len(self._streams),
        )
        return stream

    def get(self, response_id: str) -> ResponseStream | None:
        """Look up a stream by response ID."""
        return self._streams.get(response_id)

    def get_active(self, anima_name: str) -> ResponseStream | None:
        """Return the most recent (possibly still active) stream for an anima."""
        response_id = self._anima_active.get(anima_name)
        if response_id is None:
            logger.info("[SSE-REG] get_active anima=%s -> no active stream", anima_name)
            return None
        stream = self._streams.get(response_id)
        if stream:
            logger.info(
                "[SSE-REG] get_active anima=%s -> stream=%s status=%s events=%d",
                anima_name, response_id, stream.status, stream.event_count,
            )
        return stream

    def mark_complete(self, response_id: str) -> None:
        """Mark a stream as complete."""
        stream = self._streams.get(response_id)
        if stream:
            stream.complete = True
            logger.info(
                "[SSE-REG] mark_complete stream=%s anima=%s events=%d text_len=%d",
                response_id, stream.anima_name, stream.event_count,
                len(stream.full_text),
            )
            # Clear the active mapping if this is still the active stream
            if self._anima_active.get(stream.anima_name) == response_id:
                self._anima_active.pop(stream.anima_name, None)
        else:
            logger.info(
                "[SSE-REG] mark_complete stream=%s NOT_FOUND", response_id,
            )

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
        frame = format_sse_with_id(event, payload, sse_event.event_id)
        if event != "text_delta":
            logger.info(
                "[SSE-FRAME] yield stream=%s event=%s id=%s frame_len=%d",
                stream.response_id, event, sse_event.event_id, len(frame),
            )
        return frame


def format_sse_with_id(
    event: str,
    payload: dict[str, Any],
    event_id: str,
) -> str:
    """Format a single SSE frame with id field."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"id: {event_id}\nevent: {event}\ndata: {data}\n\n"
