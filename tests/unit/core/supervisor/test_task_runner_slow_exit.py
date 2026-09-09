"""結果送信後にexitが遅い子を殺しても、受領済み結果を失わないこと（2026-08-12事故の回帰）。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from core.supervisor import task_runner_supervisor as trs
from core.supervisor.task_runner_supervisor import TaskRunnerError, TaskRunnerSupervisor


def _supervisor(tmp_path: Path) -> TaskRunnerSupervisor:
    shared_dir = tmp_path / "shared"
    anima_dir = tmp_path / "animas" / "rin"
    shared_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    return TaskRunnerSupervisor("rin", anima_dir, shared_dir)


async def _spawn_with_result(
    supervisor: TaskRunnerSupervisor,
    monkeypatch: pytest.MonkeyPatch,
    terminal: dict,
) -> dict:
    """Drive _spawn_and_await against a child that never exits on its own."""
    real_exec = asyncio.create_subprocess_exec

    async def fake_exec(*_args, **kwargs):  # the "child" just sleeps forever
        kwargs.pop("env", None)
        return await real_exec(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(trs, "_TASK_RUNNER_EXIT_TIMEOUT", 0.3)

    run = asyncio.create_task(
        supervisor._spawn_and_await(
            lane="task",
            job_prefix="task",
            params_builder=lambda url_env: {"environment": {"urls": url_env}},
            log_context="slow-exit-test",
            attempt=1,
            display_lane="background",
            on_spawned=None,
            url_env={"ANIMAWORKS_EMBED_URL": "http://localhost:0"},
        )
    )
    # Deliver the terminal result once the job is registered.
    while not supervisor.jobs:
        await asyncio.sleep(0.01)
    job = next(iter(supervisor.jobs.values()))
    job.result.set_result(terminal)
    return await asyncio.wait_for(run, timeout=10)


@pytest.mark.asyncio
async def test_slow_exit_after_result_returns_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = _supervisor(tmp_path)
    result = await _spawn_with_result(
        supervisor, monkeypatch, {"result": {"task_type": "llm", "result": "ok", "success": True}}
    )
    assert result == {"task_type": "llm", "result": "ok", "success": True}
    assert not supervisor.jobs  # group reaped


@pytest.mark.asyncio
async def test_slow_exit_after_error_terminal_still_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = _supervisor(tmp_path)
    with pytest.raises(TaskRunnerError, match="EXECUTION_ERROR"):
        await _spawn_with_result(
            supervisor,
            monkeypatch,
            {"error": {"code": "EXECUTION_ERROR", "message": "boom"}},
        )


@pytest.mark.asyncio
async def test_spawn_callback_failure_reaps_child_before_releasing_job(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path)
    real_exec = asyncio.create_subprocess_exec
    spawned = []

    async def fake_exec(*args, **kwargs):
        kwargs.pop("env", None)
        process = await real_exec(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        spawned.append(process)
        return process

    def failed_callback(job):
        raise RuntimeError("identity storage failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(trs, "_TASK_RUNNER_TERM_TIMEOUT", 0.2)
    with pytest.raises(RuntimeError, match="identity storage failed"):
        await asyncio.wait_for(
            supervisor._spawn_and_await(
                lane="task",
                job_prefix="task",
                params_builder=lambda urls: {"environment": {"urls": urls}},
                log_context="callback-failure",
                attempt=1,
                display_lane="background",
                on_spawned=failed_callback,
                url_env={"ANIMAWORKS_EMBED_URL": "http://localhost:0"},
            ),
            timeout=5,
        )
    assert spawned[0].returncode is not None
    assert not supervisor.jobs
