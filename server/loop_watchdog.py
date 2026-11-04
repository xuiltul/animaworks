# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Report event loop stalls in the log, with the stack that caused them.

The loop updates a timestamp on a timer. A separate thread reads it, because
code running on a stalled loop cannot report anything until the stall is over.
When the timestamp goes stale the thread logs where the loop thread is stuck,
which is what py-spy would otherwise be needed for.

Logging goes through the normal loggers. In the server those are behind the
non-blocking queue, so a stall caused by the disk can still be reported.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time

logger = logging.getLogger("animaworks.loop_watchdog")

DEFAULT_THRESHOLD_S = 1.0
DEFAULT_INTERVAL_S = 0.25
_MAX_STACK_FRAMES = 12


def _format_stack(thread_id: int | None) -> str:
    """Describe a thread's innermost frames, without reading any source files."""
    frame = sys._current_frames().get(thread_id) if thread_id is not None else None
    lines: list[str] = []
    while frame is not None and len(lines) < _MAX_STACK_FRAMES:
        code = frame.f_code
        lines.append(f"{code.co_filename}:{frame.f_lineno} {code.co_name}")
        frame = frame.f_back
    return " <- ".join(lines) if lines else "unavailable"


class LoopLagWatchdog:
    """Log a WARNING when the event loop stops running for ``threshold_s``."""

    def __init__(self, *, threshold_s: float = DEFAULT_THRESHOLD_S, interval_s: float = DEFAULT_INTERVAL_S) -> None:
        self._threshold_s = threshold_s
        self._interval_s = interval_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._last_tick = 0.0
        self._timer: asyncio.TimerHandle | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stalls_detected = 0

    def start(self) -> None:
        """Start watching the running loop. Call from a coroutine on that loop."""
        self._loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._tick()
        self._thread = threading.Thread(target=self._watch, name="loop-lag-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _tick(self) -> None:
        self._last_tick = time.monotonic()
        if not self._stop.is_set() and self._loop is not None:
            self._timer = self._loop.call_later(self._interval_s, self._tick)

    def _watch(self) -> None:
        # A stall is a gap beyond the expected wait for the next tick.
        limit = self._threshold_s + self._interval_s
        stall_started: float | None = None
        while not self._stop.wait(self._interval_s):
            last_tick = self._last_tick
            now = time.monotonic()
            if now - last_tick > limit:
                if stall_started is None:
                    stall_started = last_tick
                    self.stalls_detected += 1
                    logger.warning(
                        "Event loop stalled for %.1fs and counting; loop thread at: %s",
                        now - last_tick - self._interval_s,
                        _format_stack(self._loop_thread_id),
                    )
            elif stall_started is not None:
                logger.warning(
                    "Event loop stall ended after %.1fs",
                    last_tick - stall_started - self._interval_s,
                )
                stall_started = None
