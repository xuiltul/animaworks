"""Tests for the event loop stall watchdog."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from server.loop_watchdog import LoopLagWatchdog


def _a_blocking_function_on_the_loop(seconds: float) -> None:
    time.sleep(seconds)


@pytest.mark.asyncio
async def test_stall_is_logged_with_the_stack_and_its_end(caplog: pytest.LogCaptureFixture) -> None:
    watchdog = LoopLagWatchdog(threshold_s=0.2, interval_s=0.05)
    with caplog.at_level(logging.WARNING, logger="animaworks.loop_watchdog"):
        watchdog.start()
        await asyncio.sleep(0.1)
        _a_blocking_function_on_the_loop(0.6)
        await asyncio.sleep(0.2)
        watchdog.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert watchdog.stalls_detected == 1
    assert any("Event loop stalled" in m and "_a_blocking_function_on_the_loop" in m for m in messages)
    assert any("Event loop stall ended" in m for m in messages)


@pytest.mark.asyncio
async def test_a_healthy_loop_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    watchdog = LoopLagWatchdog(threshold_s=0.5, interval_s=0.05)
    with caplog.at_level(logging.WARNING, logger="animaworks.loop_watchdog"):
        watchdog.start()
        await asyncio.sleep(0.3)
        watchdog.stop()

    assert watchdog.stalls_detected == 0
    assert caplog.records == []


@pytest.mark.asyncio
async def test_stop_ends_the_thread_and_timer() -> None:
    watchdog = LoopLagWatchdog(threshold_s=0.5, interval_s=0.05)
    watchdog.start()
    thread = watchdog._thread
    watchdog.stop()
    assert thread is not None
    assert not thread.is_alive()
    assert watchdog._timer is None
