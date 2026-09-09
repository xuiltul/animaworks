"""Feature flag and crash semantics for heartbeat process isolation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.schemas import CycleResult
from core.supervisor.ipc_v2 import IPCV2ConnectionState, IPCV2Identity
from core.supervisor.scheduler_manager import SchedulerManager
from core.supervisor.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor


def _manager(tmp_path: Path, *, isolated: bool) -> tuple[SchedulerManager, MagicMock]:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    (tmp_path / "shared").mkdir()
    (anima_dir / "status.json").write_text(
        json.dumps(
            {
                "process_model": "phase2",
                "task_process_isolation": {"heartbeat": isolated},
            }
        ),
        encoding="utf-8",
    )
    anima = MagicMock()
    anima.shared_dir = tmp_path / "shared"
    anima.run_heartbeat = AsyncMock(return_value=CycleResult(trigger="heartbeat", action="completed", summary="done"))
    emit = MagicMock()
    manager = SchedulerManager(anima, "sakura", anima_dir, emit)
    return manager, anima


@pytest.mark.asyncio
async def test_phase3_starts_root_memory_service_but_phase2_does_not(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path, isolated=True)
    assert manager._task_runner_supervisor is not None
    assert manager._task_runner_supervisor._memory_service is None
    await manager.shutdown_task_runners()

    phase3_dir = tmp_path / "phase3" / "animas" / "sakura"
    phase3_dir.mkdir(parents=True)
    shared = tmp_path / "phase3" / "shared"
    shared.mkdir()
    (phase3_dir / "status.json").write_text('{"process_model":"phase3"}', encoding="utf-8")
    anima = MagicMock(shared_dir=shared)
    phase3 = SchedulerManager(anima, "sakura", phase3_dir, MagicMock())

    assert phase3._task_runner_supervisor is not None
    assert phase3._task_runner_supervisor._memory_service is not None
    await phase3.shutdown_task_runners()


@pytest.mark.asyncio
async def test_flag_false_preserves_legacy_path_without_spawn(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path, isolated=False)

    await manager.heartbeat_tick()

    assert manager._task_runner_supervisor is None
    assert manager._heartbeat_isolated is False
    anima.run_heartbeat.assert_awaited_once()


@pytest.mark.asyncio
async def test_flag_true_uses_child_result_without_root_llm(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path, isolated=True)
    assert manager._task_runner_supervisor is not None
    assert manager._heartbeat_isolated is True
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
    manager, anima = _manager(tmp_path, isolated=True)
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
    manager, anima = _manager(tmp_path, isolated=True)
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
async def test_cron_only_flag_does_not_isolate_heartbeat(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    (tmp_path / "shared").mkdir()
    (anima_dir / "status.json").write_text(
        json.dumps(
            {
                "process_model": "phase2",
                "task_process_isolation": {"cron": True, "heartbeat": False},
            }
        ),
        encoding="utf-8",
    )
    anima = MagicMock()
    anima.shared_dir = tmp_path / "shared"
    anima.run_heartbeat = AsyncMock(return_value=CycleResult(trigger="heartbeat", action="completed", summary="legacy"))
    manager = SchedulerManager(anima, "sakura", anima_dir, MagicMock())

    assert manager._task_runner_supervisor is not None
    assert manager._cron_isolated is True
    assert manager._heartbeat_isolated is False

    await manager.heartbeat_tick()
    anima.run_heartbeat.assert_awaited_once()


@pytest.mark.parametrize("isolated", [False, True])
async def test_periodic_work_waits_for_initial_profile(tmp_path, isolated):
    from core.schemas import CronTask

    manager, anima = _manager(tmp_path, isolated=isolated)
    directory = manager._anima_dir
    (directory / "identity.md").write_text("未定義", encoding="utf-8")
    (directory / "injection.md").write_text("未定義", encoding="utf-8")
    (directory / "bootstrap.md").write_text("First conversation", encoding="utf-8")
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
    (directory / "identity.md").write_text("# Identity\nA helpful manager.", encoding="utf-8")
    (directory / "injection.md").write_text("# Role\nCoordinate the team.", encoding="utf-8")
    (directory / "bootstrap.md").unlink()
    assert not manager._awaiting_initial_setup()
