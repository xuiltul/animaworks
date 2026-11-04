# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Non-blocking log delivery for the server process.

A write to the log file, or to stdout/stderr when launchd redirects them to a
file, can stall for many seconds when the disk is busy. The server's event loop
must not wait for that, so while this queue is active every logging handler and
both standard streams only enqueue work, and one background thread (the "pump")
does the actual writing.

Behaviour worth knowing:

- Ordering: one queue and one thread, so items are written in the order they
  were enqueued, across all loggers and both streams.
- Timestamps and context: a record keeps the time it was created, and handlers
  run inside a copy of the caller's context, so ``request_id`` and ``cycle_id``
  are unchanged however late the record is written.
- Overflow: the queue holds at most ``max_items``. The last ``reserved_items``
  slots are kept for WARNING and above, so a long stall loses routine lines
  first. Dropped items are counted and reported as a WARNING once the pump can
  write again, and the total is available from :func:`get_log_queue_stats`.
- Shutdown: :func:`stop_log_queue` writes out what is queued, waiting at most
  ``timeout`` seconds, then reconnects the handlers directly. uvicorn re-raises
  SIGTERM after its graceful shutdown, which skips ``finally`` and atexit, so
  :func:`flush_on_signal` covers that path.

Only the server enables this. Short-lived CLI commands keep direct handlers.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import io
import logging
import logging.handlers
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple, TextIO

MAX_QUEUED_ITEMS = 10_000
RESERVED_FOR_WARNINGS = 1_000
DROP_REPORT_INTERVAL_S = 5.0
SHUTDOWN_FLUSH_TIMEOUT_S = 10.0

_REPORT_LOGGER_NAME = "animaworks.log_queue"
_IMMUTABLE_ARG_TYPES = (str, bytes, int, float, bool, type(None))


class _RecordItem(NamedTuple):
    record: logging.LogRecord
    targets: tuple[logging.Handler, ...]
    context: contextvars.Context


class _WriteItem(NamedTuple):
    stream: TextIO
    text: str


class _FlushItem(NamedTuple):
    done: threading.Event


_STOP = object()


def _deliver(record: logging.LogRecord, targets: Iterable[logging.Handler]) -> None:
    for handler in targets:
        if record.levelno >= handler.level:
            handler.handle(record)


class LogPump:
    """Bounded queue of log work plus the thread that performs it.

    Uses ``SimpleQueue``, whose ``put`` is lock-free and safe to call from a
    signal handler. Its size check is therefore approximate: the bound can be
    overshot by at most the number of threads enqueueing at that instant.
    """

    def __init__(self, *, max_items: int = MAX_QUEUED_ITEMS, reserved_items: int = RESERVED_FOR_WARNINGS) -> None:
        self.max_items = max_items
        self._routine_limit = max(1, max_items - reserved_items)
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._drop_lock = threading.RLock()
        self._dropped_routine = 0
        self._dropped_warnings = 0
        self._unreported_routine = 0
        self._unreported_warnings = 0
        self._last_report = time.monotonic()
        self.report_targets: tuple[logging.Handler, ...] = ()
        self._thread = threading.Thread(target=self._run, name="log-pump", daemon=True)

    def start(self) -> None:
        self._thread.start()

    # ── Producer side (never blocks) ───────────────────────────

    def submit_record(self, record: logging.LogRecord, targets: tuple[logging.Handler, ...]) -> None:
        is_warning = record.levelno >= logging.WARNING
        if self._queue.qsize() >= (self.max_items if is_warning else self._routine_limit):
            self._count_drop(is_warning)
            return
        self._queue.put(_RecordItem(record, targets, contextvars.copy_context()))

    def submit_write(self, stream: TextIO, text: str) -> None:
        if self._queue.qsize() >= self._routine_limit:
            self._count_drop(False)
            return
        self._queue.put(_WriteItem(stream, text))

    def _count_drop(self, is_warning: bool) -> None:
        with self._drop_lock:
            if is_warning:
                self._dropped_warnings += 1
                self._unreported_warnings += 1
            else:
                self._dropped_routine += 1
                self._unreported_routine += 1

    # ── Consumer side ──────────────────────────────────────────

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=DROP_REPORT_INTERVAL_S)
            except queue.Empty:
                item = None
            if item is _STOP:
                return
            self._process(item)
            self._report_drops()

    def _process(self, item: Any) -> None:
        # An exception here must never end the thread, or logging would stop
        # for the rest of the process and the queue would only ever fill up.
        try:
            if isinstance(item, _RecordItem):
                item.context.run(_deliver, item.record, item.targets)
            elif isinstance(item, _WriteItem):
                item.stream.write(item.text)
                item.stream.flush()
            elif isinstance(item, _FlushItem):
                item.done.set()
        except Exception as exc:  # noqa: BLE001 - see comment above
            # Not through logging: that could re-enter the handler that failed.
            with contextlib.suppress(OSError):
                os.write(2, f"log queue: failed to write an item: {exc!r}\n".encode(errors="replace"))

    def _report_drops(self, *, force: bool = False) -> None:
        if not force and time.monotonic() - self._last_report < DROP_REPORT_INTERVAL_S:
            return
        with self._drop_lock:
            routine, warnings = self._unreported_routine, self._unreported_warnings
            self._unreported_routine = self._unreported_warnings = 0
        if not routine and not warnings:
            return
        self._last_report = time.monotonic()
        record = logging.LogRecord(
            _REPORT_LOGGER_NAME,
            logging.WARNING,
            __file__,
            0,
            "Log queue overflow: dropped %d routine item(s) and %d WARNING+ record(s) since the last report",
            (routine, warnings),
            None,
        )
        self._process(_RecordItem(record, self.report_targets, contextvars.copy_context()))

    # ── Control ────────────────────────────────────────────────

    def flush(self, timeout: float) -> bool:  # noqa: ASYNC109 - this is synchronous; call it off the loop
        """Wait until everything enqueued so far has been written."""
        if not self._thread.is_alive():
            return True
        done = threading.Event()
        self._queue.put(_FlushItem(done))
        return done.wait(timeout)

    def stop(self, timeout: float) -> bool:  # noqa: ASYNC109 - this is synchronous; call it off the loop
        """Write out the queue and end the thread. Return False on timeout."""
        self._queue.put(_STOP)
        self._thread.join(timeout)
        if self._thread.is_alive():
            return False
        # Items queued behind the stop marker by other threads.
        while True:
            try:
                self._process(self._queue.get_nowait())
            except queue.Empty:
                break
        self._report_drops(force=True)
        return True

    def stats(self) -> dict[str, int]:
        with self._drop_lock:
            routine, warnings = self._dropped_routine, self._dropped_warnings
        return {
            "queued": self._queue.qsize(),
            "max_items": self.max_items,
            "dropped_routine": routine,
            "dropped_warnings": warnings,
            "dropped_total": routine + warnings,
        }


class QueuedHandler(logging.handlers.QueueHandler):
    """Handler that hands records to the pump, which runs ``targets``."""

    def __init__(self, pump: LogPump, targets: Sequence[logging.Handler]) -> None:
        super().__init__(pump._queue)
        self.pump = pump
        self.targets = tuple(targets)

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # Keep the record structured: formatters on the pump thread still need
        # the structlog event dict and exc_info, and uvicorn's access formatter
        # unpacks record.args. Fold %-args into the message only if one of them
        # could change before the line is written.
        if isinstance(record.msg, str) and record.args:
            args = record.args
            if not (isinstance(args, tuple) and all(isinstance(arg, _IMMUTABLE_ARG_TYPES) for arg in args)):
                record.msg = record.getMessage()
                record.args = None
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        self.pump.submit_record(record, self.targets)


class _QueuedStream(io.TextIOBase):
    """Text stream that enqueues writes; everything else goes to the real stream."""

    def __init__(self, pump: LogPump, real: TextIO) -> None:
        self._pump = pump
        self._real = real

    def write(self, text: str) -> int:
        self._pump.submit_write(self._real, text)
        return len(text)

    def flush(self) -> None:
        """Do nothing: the pump flushes after every write."""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return self._real.isatty()

    def fileno(self) -> int:
        return self._real.fileno()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._real.encoding

    @property
    def errors(self) -> str | None:  # type: ignore[override]
        return self._real.errors

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# ── Process-wide installation ──────────────────────────────────

_pump: LogPump | None = None
_real_stdout: TextIO | None = None
_real_stderr: TextIO | None = None
_install_lock = threading.RLock()


def _all_loggers() -> list[logging.Logger]:
    registered = list(logging.root.manager.loggerDict.values())
    return [logging.getLogger(), *(logger for logger in registered if isinstance(logger, logging.Logger))]


def _route_logger(pump: LogPump, logger: logging.Logger) -> None:
    direct = [handler for handler in logger.handlers if not isinstance(handler, QueuedHandler)]
    if direct:
        queued = [handler for handler in logger.handlers if isinstance(handler, QueuedHandler)]
        logger.handlers[:] = [*queued, QueuedHandler(pump, direct)]


def _unroute_logger(logger: logging.Logger) -> None:
    if any(isinstance(handler, QueuedHandler) for handler in logger.handlers):
        restored: list[logging.Handler] = []
        for handler in logger.handlers:
            restored.extend(handler.targets if isinstance(handler, QueuedHandler) else (handler,))
        logger.handlers[:] = restored


def start_log_queue(*, max_items: int = MAX_QUEUED_ITEMS, reserved_items: int = RESERVED_FOR_WARNINGS) -> LogPump:
    """Route all configured handlers plus stdout and stderr through one pump.

    Handlers already attached to any logger are moved behind the queue. Add
    handlers created later with :func:`route_handler`. A stream handler created
    later binds to the wrapped ``sys.stderr``, so it is non-blocking anyway.
    """
    global _pump, _real_stdout, _real_stderr
    with _install_lock:
        if _pump is not None:
            return _pump
        pump = LogPump(max_items=max_items, reserved_items=reserved_items)
        pump.report_targets = tuple(logging.getLogger().handlers)
        pump.start()
        for logger in _all_loggers():
            _route_logger(pump, logger)
        _real_stdout, _real_stderr = sys.stdout, sys.stderr
        sys.stdout = _QueuedStream(pump, _real_stdout)  # type: ignore[assignment]
        sys.stderr = _QueuedStream(pump, _real_stderr)  # type: ignore[assignment]
        _pump = pump
        atexit.register(stop_log_queue)
        return pump


def stop_log_queue(timeout: float = SHUTDOWN_FLUSH_TIMEOUT_S) -> bool:  # noqa: ASYNC109 - this is synchronous
    """Write out queued items and reconnect handlers directly.

    Returns False if the disk did not let the queue drain within ``timeout``;
    the count of what was left is then written to stderr. Safe to call twice.
    """
    global _pump, _real_stdout, _real_stderr
    with _install_lock:
        pump = _pump
        if pump is None:
            return True
        _pump = None
        for logger in _all_loggers():
            _unroute_logger(logger)
        if isinstance(sys.stdout, _QueuedStream) and _real_stdout is not None:
            sys.stdout = _real_stdout
        if isinstance(sys.stderr, _QueuedStream) and _real_stderr is not None:
            sys.stderr = _real_stderr
        _real_stdout = _real_stderr = None
        atexit.unregister(stop_log_queue)
    finished = pump.stop(timeout)
    if not finished:
        left = max(0, pump.stats()["queued"] - 1)  # not counting the stop marker
        os.write(2, f"log queue: gave up after {timeout:.0f}s with {left} item(s) unwritten\n".encode())
    return finished


def route_handler(handler: logging.Handler) -> logging.Handler:
    """Return ``handler`` behind the queue when it is active, else unchanged.

    For handlers created after :func:`start_log_queue`, such as a lazily
    attached dedicated file handler.
    """
    return QueuedHandler(_pump, [handler]) if _pump is not None else handler


def flush_log_queue(timeout: float = SHUTDOWN_FLUSH_TIMEOUT_S) -> bool:  # noqa: ASYNC109 - this is synchronous
    """Wait until everything enqueued so far has been written."""
    pump = _pump
    return True if pump is None else pump.flush(timeout)


def get_log_queue_stats() -> dict[str, int] | None:
    """Return queue depth and drop counters, or None when the queue is inactive."""
    pump = _pump
    return None if pump is None else pump.stats()


def flush_on_signal(signum: int = signal.SIGTERM) -> None:
    """Flush the queue when ``signum`` ends the process, then let it end as usual.

    Only replaces the default disposition, so the process still exits by signal
    and its exit status does not change. uvicorn saves this handler, runs its
    graceful shutdown, restores it and re-raises the signal, which makes this
    the last thing to run on that path.
    """

    def _flush_then_exit(received: int, _frame: object) -> None:
        stop_log_queue()
        signal.signal(received, signal.SIG_DFL)
        signal.raise_signal(received)

    is_main_thread = threading.current_thread() is threading.main_thread()
    if is_main_thread and signal.getsignal(signum) == signal.SIG_DFL:
        signal.signal(signum, _flush_then_exit)
