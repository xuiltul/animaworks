"""Crash semantics for heartbeat process isolation (always isolated)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.runtime.ipc_v2 import IPCV2ConnectionState, IPCV2Identity
from core.runtime.scheduler_manager import SchedulerManager
from core.runtime.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor
from core.schemas import CycleResult


def _manager(tmp_path: Path) -> tuple[SchedulerManager, MagicMock]:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    (tmp_path / "shared").mkdir()
    anima = MagicMock()
    anima.shared_dir = tmp_path / "shared"
    anima.run_heartbeat = AsyncMock(return_value=CycleResult(trigger="heartbeat", action="completed", summary="done"))
    emit = MagicMock()
    manager = SchedulerManager(anima, "sakura", anima_dir, emit)
    return manager, anima


@pytest.mark.asyncio
async def test_root_memory_service_is_always_provided(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    assert manager._task_runner_supervisor is not None
    assert manager._task_runner_supervisor._memory_service is not None
    await manager.shutdown_task_runners()


@pytest.mark.asyncio
async def test_flag_true_uses_child_result_without_root_llm(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path)
    assert manager._task_runner_supervisor is not None
    manager._task_runner_supervisor.run_heartbeat = AsyncMock(
        return_value={
            "task_type": "heartbeat",
            "result": {"action": "completed", "summary": "child result"},
            "success": True,
            "usage": {"input_tokens": 10},
        }
    )

    await manager.heartbeat_tick()

    anima.run_heartbeat.assert_not_awaited()
    manager._task_runner_supervisor.run_heartbeat.assert_awaited_once_with()
    manager._emit_event.assert_called_once()
    assert manager._emit_event.call_args[0][0] == "anima.heartbeat"
    assert manager._emit_event.call_args[0][1]["result"]["summary"] == "child result"


@pytest.mark.asyncio
async def test_heartbeat_skips_while_already_running_serializes_same_anima(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path)
    assert manager._task_runner_supervisor is not None

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_heartbeat() -> dict:
        started.set()
        await release.wait()
        return {
            "task_type": "heartbeat",
            "result": {"action": "completed", "summary": "first"},
            "success": True,
        }

    manager._task_runner_supervisor.run_heartbeat = AsyncMock(side_effect=_slow_heartbeat)

    first = asyncio.create_task(manager.heartbeat_tick())
    await started.wait()

    # Second fire while first is in flight must be skipped (same-anima serial HB).
    await manager.heartbeat_tick()
    anima.run_heartbeat.assert_not_awaited()
    assert manager._task_runner_supervisor.run_heartbeat.await_count == 1

    release.set()
    await first
    assert manager.heartbeat_running is False

    # After completion a new cycle can run.
    manager._task_runner_supervisor.run_heartbeat = AsyncMock(
        return_value={
            "task_type": "heartbeat",
            "result": {"action": "completed", "summary": "second"},
            "success": True,
        }
    )
    await manager.heartbeat_tick()
    manager._task_runner_supervisor.run_heartbeat.assert_awaited_once()


@pytest.mark.asyncio
async def test_sigkill_only_reaps_task_group_and_root_can_continue(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    anima_dir = tmp_path / "animas" / "sakura"
    shared_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    supervisor = TaskRunnerSupervisor("sakura", anima_dir, shared_dir)
    identity = IPCV2Identity(
        job_id="job-kill-hb",
        root_epoch=supervisor.root_epoch,
        attempt=1,
        lane="heartbeat",
        display_lane="background",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        start_new_session=True,
    )
    job = TaskRunnerJob(
        identity=identity,
        request_id="run-kill-hb",
        params={},
        result=asyncio.get_running_loop().create_future(),
        peer_state=IPCV2ConnectionState(identity),
        process=process,
        pid=process.pid,
        pgid=process.pid,
    )
    supervisor.jobs[identity.job_id] = job
    root_pid = os.getpid()

    os.killpg(process.pid, 9)
    return_code = await asyncio.wait_for(process.wait(), timeout=2)
    supervisor.jobs.pop(identity.job_id)

    assert return_code == -9
    assert os.getpid() == root_pid
    assert supervisor._accepting is True
    assert not supervisor.jobs


@pytest.mark.asyncio
async def test_child_crash_is_logged_and_next_tick_can_run(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path)
    assert manager._task_runner_supervisor is not None
    manager._task_runner_supervisor.run_heartbeat = AsyncMock(
        side_effect=RuntimeError("child exited before returning a result")
    )

    await manager.heartbeat_tick()

    anima.run_heartbeat.assert_not_awaited()
    assert manager.heartbeat_running is False

    manager._task_runner_supervisor.run_heartbeat = AsyncMock(
        return_value={
            "task_type": "heartbeat",
            "result": {"action": "completed", "summary": "recovered"},
            "success": True,
        }
    )
    await manager.heartbeat_tick()
    manager._task_runner_supervisor.run_heartbeat.assert_awaited_once()
    assert manager.heartbeat_running is False


@pytest.mark.asyncio
async def test_periodic_work_waits_for_initial_profile(tmp_path: Path) -> None:
    from core.schemas import CronTask

    manager, anima = _manager(tmp_path)
    directory = manager._anima_dir
    (directory / "IDENTITY.md").write_text("未定義", encoding="utf-8")
    (directory / "AGENTS.md").write_text("未定義", encoding="utf-8")
    (directory / "BOOTSTRAP.md").write_text("First conversation", encoding="utf-8")
    runner = manager._task_runner_supervisor
    if runner:
        runner.run_heartbeat = AsyncMock()
    manager._run_cron_task = AsyncMock()
    await manager.heartbeat_tick()
    await manager.cron_tick(CronTask(name="daily", schedule="0 9 * * *", type="llm", description="plan"))
    anima.run_heartbeat.assert_not_awaited()
    manager._run_cron_task.assert_not_awaited()
    if runner:
        runner.run_heartbeat.assert_not_awaited()
    # Completing the profile re-enables the existing schedule automatically.
    (directory / "IDENTITY.md").write_text("# Identity\nA helpful manager.", encoding="utf-8")
    (directory / "AGENTS.md").write_text("# Role\nCoordinate the team.", encoding="utf-8")
    (directory / "BOOTSTRAP.md").unlink()
    assert not manager._awaiting_initial_setup()
