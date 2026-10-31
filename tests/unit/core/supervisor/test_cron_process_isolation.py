"""Cron command root dispatch, follow-up isolation, and process-group cleanup."""

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
from core.schemas import CronTask, CycleResult


def _manager(tmp_path: Path) -> tuple[SchedulerManager, MagicMock]:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    (tmp_path / "shared").mkdir()
    anima = MagicMock()
    anima.shared_dir = tmp_path / "shared"
    anima.run_cron_task = AsyncMock(return_value=CycleResult(trigger="cron:daily", action="completed", summary="done"))
    anima.run_cron_command = AsyncMock()
    emit = MagicMock()
    manager = SchedulerManager(anima, "sakura", anima_dir, emit)
    return manager, anima


@pytest.mark.asyncio
async def test_uses_child_result_without_root_llm(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path)
    task = CronTask(name="daily", schedule="0 9 * * *", description="daily work")
    assert manager._task_runner_supervisor is not None
    manager._task_runner_supervisor.run_cron = AsyncMock(
        return_value={
            "task_type": "llm",
            "result": {"action": "completed", "summary": "child result"},
            "success": True,
            "usage": {"input_tokens": 10},
        }
    )

    await manager._run_cron_task(task)

    anima.run_cron_task.assert_not_awaited()
    manager._task_runner_supervisor.run_cron.assert_awaited_once_with(task)
    manager._emit_event.assert_called_once()


@pytest.mark.asyncio
async def test_command_cron_runs_in_root_without_starting_a_runner(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path)
    task = CronTask(
        name="daily",
        schedule="0 9 * * *",
        type="command",
        command="echo hi",
        trigger_heartbeat=False,
    )
    assert manager._task_runner_supervisor is not None
    anima.run_cron_command = AsyncMock(
        return_value={"task": "daily", "exit_code": 0, "stdout": "hi", "stderr": "", "duration_ms": 1}
    )
    manager._task_runner_supervisor.run_cron = AsyncMock()
    manager._task_runner_supervisor.run_cron_followup = AsyncMock()

    await manager._run_cron_task(task)

    anima.run_cron_command.assert_awaited_once()
    assert anima.run_cron_command.await_args.args == ("daily",)
    assert anima.run_cron_command.await_args.kwargs["command"] == "echo hi"
    assert anima.run_cron_command.await_args.kwargs["serialize"] is False
    assert "ANIMAWORKS_ANIMA_DIR" in anima.run_cron_command.await_args.kwargs["env"]
    manager._task_runner_supervisor.run_cron.assert_not_awaited()
    manager._task_runner_supervisor.run_cron_followup.assert_not_awaited()
    manager._emit_event.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_cancels_root_command_before_closing_runners(tmp_path: Path) -> None:
    manager, anima = _manager(tmp_path)
    task = CronTask(
        name="daily",
        schedule="0 9 * * *",
        type="command",
        command="echo hi",
        trigger_heartbeat=False,
    )
    started = asyncio.Event()

    async def block_command(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    anima.run_cron_command = AsyncMock(side_effect=block_command)
    manager._task_runner_supervisor.close = AsyncMock()
    command_task = asyncio.create_task(manager._run_cron_task(task))
    await started.wait()

    await manager.shutdown_task_runners()

    assert command_task.cancelled()
    manager._task_runner_supervisor.close.assert_awaited_once()
    assert not manager._direct_cron_tasks


@pytest.mark.asyncio
async def test_sigkill_only_reaps_task_group_and_root_can_continue(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    anima_dir = tmp_path / "animas" / "sakura"
    shared_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    supervisor = TaskRunnerSupervisor("sakura", anima_dir, shared_dir)
    identity = IPCV2Identity(
        job_id="job-kill",
        root_epoch=supervisor.root_epoch,
        attempt=1,
        lane="cron",
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
        request_id="run-kill",
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
