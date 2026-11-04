"""Tests for the server's non-blocking log queue."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import logging
import logging.config
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import structlog
import uvicorn.config
import uvicorn.logging

from core.infra.log_queue import (
    QueuedHandler,
    flush_log_queue,
    get_log_queue_stats,
    route_handler,
    start_log_queue,
    stop_log_queue,
)
from core.infra.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _Sink(logging.Handler):
    """Collects formatted messages, optionally held back by a gate like a stalled disk."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        super().__init__()
        self.gate = gate
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if self.gate is not None:
            self.gate.wait(timeout=30.0)
        self.messages.append(record.getMessage())


@contextmanager
def _queued(*handlers: logging.Handler, **kwargs: int) -> Iterator[logging.Logger]:
    """Attach only ``handlers`` to the root logger and route them through the queue."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers[:] = list(handlers)
    root.setLevel(logging.DEBUG)
    start_log_queue(**kwargs)
    try:
        yield logging.getLogger("animaworks.test")
    finally:
        stop_log_queue(timeout=5.0)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_logging_returns_while_the_destination_is_stalled() -> None:
    gate = threading.Event()
    sink = _Sink(gate)
    try:
        with _queued(sink) as log:
            started = time.monotonic()
            for index in range(100):
                log.info("line %d", index)
            elapsed = time.monotonic() - started
            assert sink.messages == []
            gate.set()
    finally:
        gate.set()
    assert elapsed < 1.0
    assert sink.messages == [f"line {index}" for index in range(100)]


def test_stdout_stderr_and_logs_stay_in_order() -> None:
    written = io.StringIO()
    sink = logging.StreamHandler(written)
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = written
    try:
        with _queued(sink) as log:
            log.info("first")
            print("second")  # noqa: T201
            log.info("third")
            sys.stderr.write("fourth\n")
            assert flush_log_queue(timeout=5.0)
            assert sys.stdout is not written
    finally:
        sys.stdout, sys.stderr = saved
    assert written.getvalue().split() == ["first", "second", "third", "fourth"]


def test_queued_stream_behaves_like_a_text_stream() -> None:
    real = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    saved = sys.stdout
    sys.stdout = real
    try:
        with _queued(_Sink()):
            assert sys.stdout.encoding == "utf-8"
            assert sys.stdout.isatty() is False
            assert sys.stdout.writable() is True
            assert sys.stdout.write("abc") == 3
            sys.stdout.flush()
            assert sys.stdout.buffer is real.buffer
    finally:
        sys.stdout = saved


def test_args_are_captured_when_the_log_call_happens() -> None:
    gate = threading.Event()
    sink = _Sink(gate)
    try:
        with _queued(sink) as log:
            data = {"state": "before"}
            log.info("value=%s", data)
            data["state"] = "after"
            gate.set()
    finally:
        gate.set()
    assert sink.messages == ["value={'state': 'before'}"]


def test_formatters_that_read_args_still_get_them() -> None:
    written = io.StringIO()
    handler = logging.StreamHandler(written)
    handler.setFormatter(uvicorn.logging.AccessFormatter('%(client_addr)s "%(request_line)s" %(status_code)s'))
    # A logger of its own: the uvicorn loggers carry state left by other tests.
    access = logging.getLogger("animaworks.test.access")
    access.handlers[:] = [handler]
    access.propagate = False
    access.setLevel(logging.INFO)
    try:
        with _queued(_Sink()):
            assert isinstance(access.handlers[0], QueuedHandler)
            access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/x", "1.1", 200)
            assert flush_log_queue(timeout=5.0)
    finally:
        access.handlers[:] = []
        access.propagate = True
        access.setLevel(logging.NOTSET)
    assert '1.2.3.4:5 "GET /x HTTP/1.1"' in written.getvalue()


def test_overflow_is_bounded_counted_and_reported() -> None:
    gate = threading.Event()
    sink = _Sink(gate)
    try:
        with _queued(sink, max_items=10, reserved_items=2) as log:
            log.info("held in the handler")
            time.sleep(0.2)  # let the pump pick it up and block on the gate
            for index in range(50):
                log.info("routine %d", index)
            for index in range(50):
                log.warning("warning %d", index)
            stats = get_log_queue_stats()
            gate.set()
    finally:
        gate.set()

    assert stats is not None
    assert stats["queued"] <= 10
    assert stats["dropped_routine"] == 42  # the routine limit is 8 queued items
    assert stats["dropped_warnings"] == 48  # the reserve holds the last 2
    reports = [message for message in sink.messages if "Log queue overflow" in message]
    assert reports == ["Log queue overflow: dropped 42 routine item(s) and 48 WARNING+ record(s) since the last report"]
    assert "warning 0" in sink.messages
    assert "warning 2" not in sink.messages


def test_a_failing_handler_does_not_end_the_pump() -> None:
    class Broken(logging.Handler):
        def filter(self, record: logging.LogRecord) -> bool:
            raise RuntimeError("broken filter")

    with _queued(Broken()) as log:
        log.info("one")
        assert flush_log_queue(timeout=5.0)
        log.info("two")
        # Answering the second flush proves the thread outlived both failures.
        assert flush_log_queue(timeout=5.0)


def test_stop_writes_out_the_queue_and_reconnects_handlers() -> None:
    sink = _Sink()
    root = logging.getLogger()
    with _queued(sink) as log:
        assert any(isinstance(handler, QueuedHandler) for handler in root.handlers)
        real_stdout = sys.stdout
        for index in range(500):
            log.info("line %d", index)
    assert len(sink.messages) == 500
    assert sink.messages[-1] == "line 499"
    assert sys.stdout is not real_stdout


def test_stop_gives_up_after_the_timeout_when_the_disk_never_answers() -> None:
    gate = threading.Event()
    sink = _Sink(gate)
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers[:] = [sink]
    root.setLevel(logging.DEBUG)
    start_log_queue()
    try:
        logging.getLogger("animaworks.test").info("stuck")
        time.sleep(0.1)
        started = time.monotonic()
        assert stop_log_queue(timeout=0.2) is False
        assert time.monotonic() - started < 2.0
        assert root.handlers == [sink]
    finally:
        gate.set()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_route_handler_is_a_no_op_when_the_queue_is_off() -> None:
    sink = _Sink()
    assert route_handler(sink) is sink
    with _queued(_Sink()):
        routed = route_handler(sink)
        assert isinstance(routed, QueuedHandler)
        assert routed.targets == (sink,)


def test_uvicorn_loggers_are_routed_through_the_queue() -> None:
    uvicorn_logger = logging.getLogger("uvicorn")
    access_logger = logging.getLogger("uvicorn.access")
    saved = (uvicorn_logger.handlers[:], access_logger.handlers[:], uvicorn_logger.propagate, access_logger.propagate)
    root_handlers = logging.getLogger().handlers[:]
    # dictConfig closes handlers it did not create; run it before ours exist.
    logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
    try:
        with _queued(_Sink()):
            assert all(isinstance(handler, QueuedHandler) for handler in access_logger.handlers)
            assert all(isinstance(handler, QueuedHandler) for handler in uvicorn_logger.handlers)
            assert access_logger.handlers
        assert not any(isinstance(handler, QueuedHandler) for handler in access_logger.handlers)
    finally:
        uvicorn_logger.handlers[:], access_logger.handlers[:] = saved[0], saved[1]
        uvicorn_logger.propagate, access_logger.propagate = saved[2], saved[3]
        logging.getLogger().handlers[:] = root_handlers


@pytest.fixture
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    stop_log_queue(timeout=5.0)
    for handler in root.handlers:
        handler.close()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


@pytest.mark.usefixtures("_restore_logging")
def test_context_timestamp_traceback_and_redaction_survive_the_queue(tmp_path: Path) -> None:
    """Delivery happens on another thread, later. The log line must not show that."""
    setup_logging(level="INFO", log_dir=tmp_path)
    log = logging.getLogger("animaworks.test")
    gate = threading.Event()
    log_file = next(
        h for h in logging.getLogger().handlers if getattr(h, "baseFilename", "").endswith("animaworks.log")
    )
    real_emit = log_file.emit

    def stalled_emit(record: logging.LogRecord) -> None:
        gate.wait(timeout=30.0)
        real_emit(record)

    log_file.emit = stalled_emit  # type: ignore[method-assign]
    start_log_queue()

    structlog.contextvars.bind_contextvars(request_id="req-1")
    logged_at = time.time()
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed with token=abcdefgh12345678")
    structlog.contextvars.clear_contextvars()

    time.sleep(1.2)  # the record now waits behind a stalled disk
    gate.set()
    assert stop_log_queue(timeout=5.0)

    entry = json.loads((tmp_path / "animaworks.log").read_text(encoding="utf-8").splitlines()[-1])
    assert entry["request_id"] == "req-1"
    assert entry["logger"] == "animaworks.test"
    stamped = entry["timestamp"].replace("Z", "+00:00")
    from datetime import datetime

    assert abs(datetime.fromisoformat(stamped).timestamp() - logged_at) < 1.0
    assert "ValueError: boom" in entry["exception"]
    assert "abcdefgh12345678" not in entry["event"]


@pytest.mark.usefixtures("_restore_logging")
def test_structlog_events_go_through_the_queue(tmp_path: Path) -> None:
    setup_logging(level="INFO", log_dir=tmp_path)
    start_log_queue()
    structlog.get_logger("animaworks.test").warning("structured", answer=42)
    assert stop_log_queue(timeout=5.0)

    entry = json.loads((tmp_path / "errors.log").read_text(encoding="utf-8").splitlines()[-1])
    assert entry["event"] == "structured"
    assert entry["answer"] == 42


def test_real_uvicorn_logs_go_through_the_queue_and_survive_sigterm(tmp_path: Path) -> None:
    """Run actual uvicorn with stdout and stderr redirected, as launchd does.

    Checks the access log still formats (its formatter unpacks record.args), and
    that lines written during uvicorn's shutdown, after which it re-raises SIGTERM
    and so skips finally and atexit, reach their destination.
    """
    log_path = tmp_path / "root.log"
    script = textwrap.dedent(
        f"""
        import logging, logging.config, os, signal, socket, threading
        import httpx, uvicorn, uvicorn.config
        from core.infra.log_queue import flush_on_signal, start_log_queue, stop_log_queue

        logging.getLogger().handlers[:] = [logging.FileHandler({str(log_path)!r})]
        logging.getLogger().setLevel(logging.INFO)
        logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
        start_log_queue()
        flush_on_signal()

        async def app(scope, receive, send):
            logging.getLogger("animaworks.test").info("handled %s", scope["path"])
            await send({{"type": "http.response.start", "status": 200, "headers": []}})
            await send({{"type": "http.response.body", "body": b"ok"}})

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        config = uvicorn.Config(app, lifespan="off", log_level="info", log_config=None)
        server = uvicorn.Server(config)

        def client():
            while not server.started:
                threading.Event().wait(0.02)
            httpx.get(f"http://127.0.0.1:{{port}}/ping", timeout=10)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=client, daemon=True).start()
        try:
            server.run(sockets=[sock])
        finally:
            stop_log_queue()
            print("FINALLY RAN", flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == -signal.SIGTERM, result.stderr
    assert '"GET /ping HTTP/1.1" 200' in result.stdout, result.stderr
    assert "Finished server process" in result.stderr
    assert "handled /ping" in log_path.read_text(encoding="utf-8")
    # This is why flush_on_signal exists: uvicorn's re-raise skips the finally.
    assert "FINALLY RAN" not in result.stdout


def test_sigterm_writes_out_the_queue_before_the_process_ends(tmp_path: Path) -> None:
    """uvicorn re-raises SIGTERM after its shutdown, which skips finally and atexit."""
    log_path = tmp_path / "out.log"
    script = textwrap.dedent(
        f"""
        import logging, signal, sys
        from core.infra.log_queue import flush_on_signal, start_log_queue

        class SlowFile(logging.FileHandler):
            def emit(self, record):
                import time
                time.sleep(0.01)
                super().emit(record)

        root = logging.getLogger()
        root.handlers[:] = [SlowFile({str(log_path)!r})]
        root.setLevel(logging.INFO)
        start_log_queue()
        flush_on_signal()
        for index in range(50):
            root.info("line %d", index)
        # What uvicorn does at the end of its graceful shutdown.
        signal.raise_signal(signal.SIGTERM)
        sys.exit(3)  # not reached
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == -signal.SIGTERM, result.stderr
    assert log_path.read_text(encoding="utf-8").splitlines() == [f"line {index}" for index in range(50)]
