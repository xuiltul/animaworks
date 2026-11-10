"""The event loop must keep running while disk writes stall.

On a busy disk any write can block for many seconds. These tests make the
relevant disk operations artificially slow and assert that the loop thread
never waits on them and ``/api/system/health`` keeps answering through an ASGI
client.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, FrameType
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.auth.manager import create_session, hash_password, save_auth
from core.auth.models import AuthConfig, AuthUser
from core.infra import startup_progress
from core.infra.log_queue import start_log_queue, stop_log_queue

STALL_S = 2.0
MAX_LOOP_GAP_S = 1.0
MAX_HEALTH_S = 1.0


def _stall() -> None:
    """Block the calling thread the way a stalled disk operation does.

    Every artificial stall goes through here, so the loop probe can tell
    whether the loop thread itself was the one waiting on the disk.
    """
    time.sleep(STALL_S)


@pytest.fixture(autouse=True)
def _ready_server():
    startup_progress._reset_for_testing()
    startup_progress.set_phase("ready")
    yield
    startup_progress._reset_for_testing()


def _make_app(data_dir: Path):
    from core.config.models import AnimaWorksConfig, invalidate_cache, save_config
    from server.app import create_app

    save_config(AnimaWorksConfig(setup_complete=True), data_dir / "config.json")
    invalidate_cache()
    return create_app(data_dir / "animas", data_dir / "shared")


class _SlowStream(io.StringIO):
    """A log destination whose writes stall like a busy disk."""

    stalled = True

    def write(self, text: str) -> int:
        if self.stalled:
            _stall()
        return super().write(text)


@contextmanager
def _logging_through_slow_stream() -> Iterator[_SlowStream]:
    """Give the root logger one slow destination, set up as the server does."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    stream = _SlowStream()
    root.handlers[:] = [logging.StreamHandler(stream)]
    root.setLevel(logging.INFO)
    start_log_queue()
    try:
        yield stream
    finally:
        stream.stalled = False  # the test is over, so let the queue drain quickly
        stop_log_queue(timeout=30.0)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


@contextmanager
def _stalling_path_ops(matches: Callable[[Path], bool]) -> Iterator[None]:
    """Make ``read_text``, ``unlink`` and ``stat`` stall for the matching paths."""
    real_read_text, real_unlink, real_stat = Path.read_text, Path.unlink, os.stat

    def slow_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if matches(self):
            _stall()
        return real_read_text(self, *args, **kwargs)

    def slow_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if matches(self):
            _stall()
        return real_unlink(self, *args, **kwargs)

    def slow_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if isinstance(path, str | Path) and matches(Path(path)):
            _stall()
        return real_stat(path, *args, **kwargs)

    with (
        patch.object(Path, "read_text", slow_read_text),
        patch.object(Path, "unlink", slow_unlink),
        patch("os.stat", slow_stat),
    ):
        yield


class _LoopGapProbe:
    """Watch the event loop while disk operations stall.

    ``max_gap`` is the longest time the loop went without running a coroutine.
    A watcher thread samples the loop thread's stack whenever the loop has not
    ticked for ``SAMPLE_AFTER_S``. :meth:`check` fails when a sample shows the
    loop thread inside :func:`_stall`, that is, the loop itself waiting on a
    stalled disk operation. A long gap with the loop thread anywhere else, such
    as a paused CI runner, becomes a warning that carries the sampled stacks.
    """

    SAMPLE_AFTER_S = 0.25
    MAX_KEPT_STACKS = 3

    def __init__(self) -> None:
        self.max_gap = 0.0
        self.stall_stack: str | None = None
        self.other_stacks: list[str] = []
        self._task: asyncio.Task[None] | None = None
        self._last_tick = time.monotonic()
        self._loop_thread_id = threading.get_ident()
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None

    async def _run(self) -> None:
        last = time.monotonic()
        self._last_tick = last
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            self.max_gap = max(self.max_gap, now - last)
            last = now
            self._last_tick = now

    def _watch(self) -> None:
        sampled_tick = None
        while not self._stop.wait(0.05):
            tick = self._last_tick
            if time.monotonic() - tick < self.SAMPLE_AFTER_S:
                continue
            frame = sys._current_frames().get(self._loop_thread_id)
            if frame is None:
                continue
            if _runs_code(frame, _stall.__code__):
                if self.stall_stack is None:
                    self.stall_stack = "".join(traceback.format_stack(frame))
            elif tick != sampled_tick and len(self.other_stacks) < self.MAX_KEPT_STACKS:
                self.other_stacks.append("".join(traceback.format_stack(frame)))
            sampled_tick = tick

    def check(self) -> None:
        """Fail if the loop waited on a stalled disk operation; warn on other long gaps."""
        if self.stall_stack is not None:
            pytest.fail(
                f"The event loop waited on a stalled disk operation "
                f"(longest gap {self.max_gap:.2f}s). Loop thread:\n{self.stall_stack}"
            )
        if self.max_gap >= MAX_LOOP_GAP_S:
            stacks = "\n".join(self.other_stacks) or "(no sample)"
            warnings.warn(
                f"Event loop gap of {self.max_gap:.2f}s without a stalled disk operation "
                f"on the loop thread. Loop thread during the gap:\n{stacks}",
                stacklevel=2,
            )

    async def __aenter__(self) -> _LoopGapProbe:
        self._loop_thread_id = threading.get_ident()
        self._task = asyncio.create_task(self._run())
        # Let the probe take its first timestamp. Otherwise code that blocks
        # before its first real suspension would run before the probe starts,
        # and the stall would go unmeasured.
        await asyncio.sleep(0)
        self._watcher = threading.Thread(target=self._watch, name="loop-gap-probe", daemon=True)
        self._watcher.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        # Let the probe wake once more, or a body that never suspended goes unseen.
        await asyncio.sleep(0.05)
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._stop.set()
        if self._watcher is not None:
            self._watcher.join(timeout=1.0)


def _runs_code(frame: FrameType | None, code: CodeType) -> bool:
    """Return True if ``code`` is running anywhere in ``frame``'s call chain."""
    while frame is not None:
        if frame.f_code is code:
            return True
        frame = frame.f_back
    return False


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _timed_health(client: AsyncClient) -> float:
    started = time.monotonic()
    response = await client.get("/api/system/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    return time.monotonic() - started


def _write_event_files(data_dir: Path, count: int) -> Path:
    events_dir = data_dir / "run" / "events" / "alice"
    events_dir.mkdir(parents=True)
    for index in range(count):
        event = {"event": "anima.test", "data": {"index": index}}
        (events_dir / f"{index:04d}.json").write_text(json.dumps(event), encoding="utf-8")
    return events_dir


# ── Request logging ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_logging_does_not_block_loop(data_dir: Path) -> None:
    save_auth(AuthConfig(auth_mode="local_trust"))
    app = _make_app(data_dir)

    with _logging_through_slow_stream():
        async with _client(app) as client, _LoopGapProbe() as probe:
            await client.get("/health")  # logged by RequestLoggingMiddleware
            await client.get("/health")
            await asyncio.sleep(0.2)

    probe.check()


@pytest.mark.asyncio
async def test_print_to_stdout_does_not_block_loop(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    real_write = capsys._capture.out.tmpfile.write  # type: ignore[union-attr]

    def slow_write(text: str) -> int:
        _stall()
        return real_write(text)

    with (
        _logging_through_slow_stream(),
        patch.object(capsys._capture.out.tmpfile, "write", slow_write),  # type: ignore[union-attr]
    ):
        async with _LoopGapProbe() as probe:
            print("written by a library")  # noqa: T201
            await asyncio.sleep(0.2)

    probe.check()


# ── Event file drain ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_file_drain_does_not_block_loop(data_dir: Path) -> None:
    app = _make_app(data_dir)
    events_dir = _write_event_files(data_dir, count=2)
    broadcast: list[dict] = []

    async def record_event(event_type: str, data: dict) -> None:
        broadcast.append(data)

    supervisor = app.state.supervisor
    with (
        _stalling_path_ops(lambda path: path.is_relative_to(events_dir)),
        patch.object(supervisor, "_broadcast_event", record_event),
    ):
        async with _LoopGapProbe() as probe:
            await supervisor._poll_anima_events()

    probe.check()
    assert broadcast == [{"index": 0}, {"index": 1}]
    assert list(events_dir.glob("*.json")) == []


# ── Polled routes ────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/api/animas/alice/sessions", "/api/activity/running-tasks"])
async def test_polled_routes_do_not_block_loop(data_dir: Path, make_anima: Callable, route: str) -> None:
    save_auth(AuthConfig(auth_mode="local_trust"))
    anima_dir = make_anima("alice")
    (anima_dir / "state" / "conversations").mkdir(parents=True, exist_ok=True)
    (anima_dir / "state" / "conversations" / "t1.json").write_text('{"turns": []}', encoding="utf-8")
    sidecar_dir = data_dir / "run" / "animas"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_dir / "alice.busy.json"
    sidecar.write_text('{"is_busy": false}', encoding="utf-8")
    app = _make_app(data_dir)
    app.state.anima_names = ["alice"]

    def is_polled_file(path: Path) -> bool:
        return path == sidecar or path.name == "t1.json"

    with _stalling_path_ops(is_polled_file):
        async with _client(app) as client, _LoopGapProbe() as probe:
            response = await client.get(route)

    assert response.status_code == 200
    probe.check()


# ── Authentication ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_guard_does_not_block_loop(data_dir: Path) -> None:
    owner = AuthUser(username="owner", role="owner", password_hash=hash_password("secret"))
    config = AuthConfig(auth_mode="password", trust_localhost=False, owner=owner)
    token = create_session(config, "owner")
    save_auth(config)
    app = _make_app(data_dir)
    auth_path = data_dir / "auth.json"

    with _stalling_path_ops(lambda path: path == auth_path):
        async with _client(app) as client, _LoopGapProbe() as probe:
            client.cookies.set("session_token", token)
            response = await client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json()["username"] == "owner"
    probe.check()


# ── Health, end to end ───────────────────────────────────────


@pytest.mark.asyncio
async def test_health_stays_fast_while_all_disk_writes_stall(data_dir: Path) -> None:
    """Request logging, the drain loop and auth stall together, as they do in production."""
    owner = AuthUser(username="owner", role="owner", password_hash=hash_password("secret"))
    config = AuthConfig(auth_mode="password", trust_localhost=False, owner=owner)
    token = create_session(config, "owner")
    save_auth(config)
    app = _make_app(data_dir)
    auth_path = data_dir / "auth.json"
    events_dir = _write_event_files(data_dir, count=3)

    def stalls(path: Path) -> bool:
        return path == auth_path or path.is_relative_to(events_dir)

    supervisor = app.state.supervisor
    with _logging_through_slow_stream(), _stalling_path_ops(stalls):
        async with _client(app) as client, _LoopGapProbe() as probe:
            client.cookies.set("session_token", token)
            drain = asyncio.create_task(supervisor._poll_anima_events())
            protected = asyncio.create_task(client.get("/api/auth/me"))
            logged = asyncio.create_task(client.get("/health"))

            health_times = []
            for _ in range(6):
                health_times.append(await _timed_health(client))
                await asyncio.sleep(0.25)

            await asyncio.wait_for(asyncio.gather(drain, protected, logged), timeout=30.0)

    assert max(health_times) < MAX_HEALTH_S
    probe.check()
    assert protected.result().status_code == 200
    assert list(events_dir.glob("*.json")) == []
