"""External (queue) cancel watch for task-runner processes.

Verifies that ``TaskRunnerSupervisor._watch_job`` kills the child process group of
a ``task``-lane job whose queue entry is externally cancelled, and that
``PendingTaskExecutor._fail_task_terminal`` skips the failure notification when the
entry is already cancelled.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor import task_runner_supervisor
from core.supervisor.ipc_v2 import IPCV2ConnectionState, IPCV2Identity
from core.supervisor.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor


def _make_supervisor() -> TaskRunnerSupervisor:
    supervisor = object.__new__(TaskRunnerSupervisor)
    supervisor.anima_name = "sumire"
    supervisor.anima_dir = Path("/tmp/animas/sumire")
    supervisor._jobs = {}
    supervisor._hang_check_interval = 0.01
    supervisor._busy_hang_threshold_sec = 99999.0
    supervisor._terminate_hung_job = AsyncMock()
    return supervisor


def _make_job(supervisor: TaskRunnerSupervisor, *, lane: str, task_id: str) -> TaskRunnerJob:
    identity = IPCV2Identity(
        job_id=f"job-{lane}-{task_id}",
        root_epoch="epoch",
        attempt=1,
        lane=lane,
        display_lane="background",
    )
    job = TaskRunnerJob(
        identity=identity,
        request_id=f"run-{task_id}",
        params={"task_desc": {"task_id": task_id}},
        result=asyncio.get_running_loop().create_future(),
        peer_state=IPCV2ConnectionState(identity),
        process=SimpleNamespace(returncode=None),
        pid=999_999,
        pgid=999_999,
        last_progress_at=asyncio.get_running_loop().time(),
    )
    supervisor._jobs[job.identity.job_id] = job
    return job


def _patch_queue(monkeypatch, status: str):
    queue = MagicMock()
    queue.get_task_by_id.return_value = SimpleNamespace(status=status)
    monkeypatch.setattr(
        "core.memory.task_queue.TaskQueueManager",
        lambda *a, **k: queue,
    )
    return queue


def test_watch_job_terminates_on_external_cancel(monkeypatch) -> None:
    supervisor = _make_supervisor()

    async def main() -> TaskRunnerJob:
        job = _make_job(supervisor, lane="task", task_id="t1")
        await supervisor._watch_job(job)
        return job

    _patch_queue(monkeypatch, "cancelled")
    monkeypatch.setattr(task_runner_supervisor, "_CANCEL_CHECK_INTERVAL", 0.0)
    finished_job = asyncio.run(main())
    supervisor._terminate_hung_job.assert_awaited_once()
    assert finished_job.hang_kill_started is True


def test_watch_job_does_not_terminate_non_task_lane(monkeypatch) -> None:
    supervisor = _make_supervisor()

    async def main() -> None:
        job = _make_job(supervisor, lane="cron", task_id="t1")
        watch = asyncio.create_task(supervisor._watch_job(job))
        await asyncio.sleep(0.05)
        assert not supervisor._terminate_hung_job.called
        watch.cancel()
        await asyncio.gather(watch, return_exceptions=True)

    _patch_queue(monkeypatch, "cancelled")
    monkeypatch.setattr(task_runner_supervisor, "_CANCEL_CHECK_INTERVAL", 0.0)
    asyncio.run(main())


def test_watch_job_does_not_terminate_when_not_cancelled(monkeypatch) -> None:
    supervisor = _make_supervisor()

    async def main() -> None:
        job = _make_job(supervisor, lane="task", task_id="t1")
        watch = asyncio.create_task(supervisor._watch_job(job))
        await asyncio.sleep(0.05)
        assert not supervisor._terminate_hung_job.called
        watch.cancel()
        await asyncio.gather(watch, return_exceptions=True)

    _patch_queue(monkeypatch, "in_progress")
    monkeypatch.setattr(task_runner_supervisor, "_CANCEL_CHECK_INTERVAL", 0.0)
    asyncio.run(main())


def test_fail_task_terminal_skips_notify_when_cancelled(tmp_path) -> None:
    from core.supervisor.pending_executor import PendingTaskExecutor

    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    executor = PendingTaskExecutor(
        anima=MagicMock(),
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )
    executor._anima.messenger = MagicMock()
    queue = MagicMock()
    queue.get_task_by_id.return_value = SimpleNamespace(status="cancelled")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "core.memory.task_queue.TaskQueueManager",
            lambda *a, **k: queue,
        )
        executor._fail_task_terminal(
            {"task_id": "t1", "title": "T", "description": "D"},
            "FAILED: interrupted",
        )
    executor._anima.messenger.send.assert_not_called()
