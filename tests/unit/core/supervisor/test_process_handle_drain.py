# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for in-flight stream drain protection in ProcessHandle.stop().

A rolling restart / RAG repair / reconcile must not cut a user-facing chat
response off mid-generation; ``_drain_active_stream`` waits (bounded) for the
stream to finish first. Hang recovery bypasses this via ``kill()``.
"""

from __future__ import annotations

import asyncio

import pytest

from core.supervisor.process_handle import ProcessHandle


class _FakeProc:
    """Stands in for a child process: ``poll()`` is None while alive."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive

    def poll(self) -> int | None:
        return None if self._alive else 0


def _make_handle(*, streaming: bool, proc: _FakeProc | None) -> ProcessHandle:
    handle = ProcessHandle.__new__(ProcessHandle)
    handle.anima_name = "aoi"
    handle._streaming = streaming
    handle._streaming_started_at = None
    handle.process = proc
    return handle


@pytest.mark.asyncio
async def test_drain_returns_immediately_when_not_streaming():
    handle = _make_handle(streaming=False, proc=_FakeProc())
    # Must not block even with a large drain timeout.
    await asyncio.wait_for(handle._drain_active_stream(60.0), timeout=1.0)


@pytest.mark.asyncio
async def test_drain_waits_until_stream_finishes():
    handle = _make_handle(streaming=True, proc=_FakeProc())

    async def finish_soon() -> None:
        await asyncio.sleep(0.2)
        handle._streaming = False

    task = asyncio.create_task(finish_soon())
    await asyncio.wait_for(handle._drain_active_stream(5.0), timeout=2.0)
    assert handle._streaming is False
    await task


@pytest.mark.asyncio
async def test_drain_times_out_when_stream_stuck():
    handle = _make_handle(streaming=True, proc=_FakeProc())
    # Stream never finishes: drain gives up after its own timeout and returns
    # so the stop can proceed to SIGTERM/SIGKILL.
    await asyncio.wait_for(handle._drain_active_stream(0.3), timeout=2.0)
    assert handle._streaming is True


@pytest.mark.asyncio
async def test_drain_stops_when_process_dies():
    proc = _FakeProc(alive=True)
    handle = _make_handle(streaming=True, proc=proc)

    async def kill_soon() -> None:
        await asyncio.sleep(0.2)
        proc._alive = False

    task = asyncio.create_task(kill_soon())
    await asyncio.wait_for(handle._drain_active_stream(5.0), timeout=2.0)
    await task


@pytest.mark.asyncio
async def test_drain_noop_with_zero_timeout():
    handle = _make_handle(streaming=True, proc=_FakeProc())
    await asyncio.wait_for(handle._drain_active_stream(0.0), timeout=1.0)
