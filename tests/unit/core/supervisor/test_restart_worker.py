# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified restart worker (single worker, backoff loop).

The ProcessSupervisor's ``start_anima`` / ``stop_anima`` are mocked; time is
advanced deterministically via an injected fake monotonic clock and a patched
``asyncio.sleep`` so backoff waits converge quickly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.supervisor.manager import ProcessSupervisor, RestartPolicy


class FakeTime:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


class FakeWS:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def broadcast(self, data: dict) -> None:
        self.events.append(data)


@pytest.fixture
def supervisor(tmp_path: Path):
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()

    s = ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
        restart_policy=RestartPolicy(
            max_retries=3,
            backoff_base_sec=1.0,
            backoff_max_sec=10.0,
            reset_after_sec=300.0,
        ),
    )
    ft = FakeTime()
    s._restart_ctl._clock = ft  # inject fake monotonic clock
    ws = FakeWS()
    s.ws_manager = ws
    return s, ft, ws


def _patch_sleep(ft: FakeTime):
    async def _fast_sleep(duration: float) -> None:
        ft.t += duration

    return patch("server.supervisor._mgr_health.asyncio.sleep", _fast_sleep)


@pytest.mark.asyncio
async def test_worker_retries_then_recovers(supervisor):
    s, ft, ws = supervisor
    start = AsyncMock(
        side_effect=[
            RuntimeError("boom1"),
            RuntimeError("boom2"),
            RuntimeError("boom3"),
            None,  # 4th attempt succeeds
        ]
    )
    s.start_anima = start  # type: ignore[method-assign]

    with _patch_sleep(ft):
        await s._restart_worker("a")

    assert start.await_count == 4
    assert "a" not in s._restarting
    assert s._restart_ctl.is_failed("a") is False
    assert s._restart_ctl.get("a").phase.value == "healthy"

    # FAILED (status=error with restart_state) broadcast happened before recovery.
    err_events = [e for e in ws.events if e.get("data", {}).get("status") == "error"]
    assert err_events, "expected a FAILED broadcast"
    assert err_events[0]["data"]["restart_state"] == "failed"
    assert s._restart_ctl.get("a").attempts == 3


@pytest.mark.asyncio
async def test_worker_is_singleton(supervisor):
    s, ft, ws = supervisor
    start = AsyncMock(side_effect=RuntimeError("boom"))
    s.start_anima = start  # type: ignore[method-assign]

    with (
        _patch_sleep(ft),
        patch("server.supervisor._mgr_health.asyncio.create_task") as mk,
    ):

        def _consume(coro):
            coro.close()  # never started; discard cleanly
            return MagicMock()

        mk.side_effect = _consume
        # Two concurrent failure handlers must not spawn two workers.
        await s._handle_process_failure("a", MagicMock())
        await s._handle_process_failure("a", MagicMock())
        assert s._restarting == {"a"}
        assert mk.call_count == 1
        s._restarting.discard("a")  # clean up for this synthetic test


@pytest.mark.asyncio
async def test_worker_stops_and_forgets_on_disable(supervisor):
    s, ft, ws = supervisor
    (s.animas_dir / "a").mkdir()
    (s.animas_dir / "a" / "status.json").write_text('{"enabled": false}\n', encoding="utf-8")
    s._restart_ctl.record_failure("a", "e1")
    s.processes["a"] = MagicMock()
    stop = AsyncMock(side_effect=lambda *a, **k: s.processes.pop("a", None))
    s.stop_anima = stop  # type: ignore[method-assign]
    start = AsyncMock()
    s.start_anima = start  # type: ignore[method-assign]

    with _patch_sleep(ft):
        await s._restart_worker("a")

    stop.assert_awaited()
    assert s._restart_ctl.get("a") is None  # forgotten
    assert "a" not in s._restarting


def test_reconcile_does_not_clear_failed_budget(supervisor):
    s, ft, _ = supervisor
    for i in range(3):
        s._restart_ctl.record_failure("a", f"e{i}")
    assert s._restart_ctl.is_failed("a")
    # 60 seconds elapse — the old behaviour cleared the budget here.
    ft.t += 60
    assert s._restart_ctl.get("a").attempts == 3
    assert s._restart_ctl.is_failed("a")


@pytest.mark.asyncio
async def test_failure_broadcast_error_still_ensures_one_worker(supervisor):
    s, _ft, _ws = supervisor
    s._restart_ctl.failed_threshold = 1
    s._broadcast_event = AsyncMock(side_effect=RuntimeError("broadcast unavailable"))  # type: ignore[method-assign]

    with patch("server.supervisor._mgr_health.asyncio.create_task") as create_task:
        create_task.side_effect = lambda coro: coro.close() or MagicMock()
        with pytest.raises(RuntimeError, match="broadcast unavailable"):
            await s._handle_process_failure("a", MagicMock(), reason="crash")

        assert s._restart_ctl.get("a").attempts == 1
        assert create_task.call_count == 1
        assert "a" in s._restarting

        # The guard is already reserved while the ensured worker is pending.
        await s._handle_process_failure("a", MagicMock(), reason="duplicate")
        assert s._restart_ctl.get("a").attempts == 1
        assert create_task.call_count == 1

    s._restart_worker_tasks.clear()
    s._restarting.clear()


@pytest.mark.asyncio
async def test_manual_restart_cancels_worker_waiting_in_backoff(supervisor):
    s, _ft, _ws = supervisor
    anima_dir = s.animas_dir / "a"
    anima_dir.mkdir()
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    s._restart_ctl.record_failure("a", "initial failure")
    s.processes["a"] = MagicMock()

    async def remove_process(*_args, **_kwargs):
        s.processes.pop("a", None)

    stop = AsyncMock(side_effect=remove_process)
    start = AsyncMock()
    s.stop_anima = stop  # type: ignore[method-assign]
    s.start_anima = start  # type: ignore[method-assign]
    waiting = asyncio.Event()

    async def blocked_sleep(_duration: float) -> None:
        waiting.set()
        await asyncio.Future()

    with patch("server.supervisor._mgr_health.asyncio.sleep", blocked_sleep):
        s._ensure_restart_worker("a")
        worker = s._restart_worker_tasks["a"]
        await asyncio.wait_for(waiting.wait(), timeout=1.0)

        await s.restart_anima("a")
        assert worker.done()

    stop.assert_awaited_once_with("a", quiesce=True)
    start.assert_awaited_once_with("a")
    assert "a" not in s._restarting
    assert "a" not in s._restart_worker_tasks


@pytest.mark.asyncio
async def test_worker_forgets_disabled_anima_during_backoff(supervisor):
    s, _ft, _ws = supervisor
    anima_dir = s.animas_dir / "a"
    anima_dir.mkdir()
    status_file = anima_dir / "status.json"
    status_file.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    s._restart_ctl.record_failure("a", "initial failure")
    start = AsyncMock()
    s.start_anima = start  # type: ignore[method-assign]
    waiting = asyncio.Event()
    real_sleep = asyncio.sleep

    async def polling_sleep(_duration: float) -> None:
        waiting.set()
        await real_sleep(0.01)

    with patch("server.supervisor._mgr_health.asyncio.sleep", polling_sleep):
        s._ensure_restart_worker("a")
        worker = s._restart_worker_tasks["a"]
        await asyncio.wait_for(waiting.wait(), timeout=1.0)
        status_file.write_text(json.dumps({"enabled": False}), encoding="utf-8")
        await asyncio.wait_for(worker, timeout=1.0)

    start.assert_not_awaited()
    assert s._restart_ctl.get("a") is None
    assert "a" not in s._restarting


@pytest.mark.asyncio
async def test_reconcile_forgets_disabled_and_deleted_records_without_process(supervisor):
    s, _ft, _ws = supervisor
    disabled_dir = s.animas_dir / "disabled"
    disabled_dir.mkdir()
    (disabled_dir / "IDENTITY.md").write_text("identity", encoding="utf-8")
    (disabled_dir / "status.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")
    s._restart_ctl.record_failure("disabled", "failure")
    s._restart_ctl.record_failure("deleted", "failure")
    s._check_config_freshness = MagicMock()  # type: ignore[method-assign]
    s._reconcile_assets = AsyncMock()  # type: ignore[method-assign]

    await s._reconcile()

    assert s._restart_ctl.get("disabled") is None
    assert s._restart_ctl.get("deleted") is None


def test_get_process_status_reports_failed_without_mutation(supervisor):
    s, ft, _ = supervisor
    for _i in range(3):
        s._restart_ctl.record_failure("a", "boom")
    assert s._restart_ctl.is_failed("a")

    status = s.get_process_status("a")
    assert status["status"] == "error"
    assert status["restart_state"] == "failed"
    assert status["last_error"] == "boom"
    assert status["restart_count"] == 3
    assert status["next_retry_at"]

    # Calling the getter must not mutate restart state.
    before = s._restart_ctl.get("a").attempts
    s.get_process_status("a")
    assert s._restart_ctl.get("a").attempts == before
    assert s._restart_ctl.is_failed("a")
