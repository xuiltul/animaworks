"""Feature flag and crash semantics for TaskExec / background process isolation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.streaming_journal import StreamingJournal
from core.platform.processing_lease import (
    read_processing_lease,
    write_processing_lease,
)
from core.supervisor.ipc_v2 import IPCV2ConnectionState, IPCV2Identity
from core.supervisor.pending_executor import PendingTaskExecutor
from core.supervisor.task_runner_supervisor import (
    TaskRunnerError,
    TaskRunnerJob,
    TaskRunnerSupervisor,
)


def _anima_double(tmp_path: Path, *, pool_size: int = 1) -> MagicMock:
    anima = MagicMock()
    anima.shared_dir = tmp_path / "shared"
    anima.shared_dir.mkdir(parents=True, exist_ok=True)
    anima._background_worker_pool_size = pool_size
    anima._background_lock = asyncio.Lock()
    anima._mark_busy_start = MagicMock()
    anima._clear_busy_status_sidecar_if_idle = MagicMock()
    anima._status_slots = {"background": "idle"}
    anima._task_slots = {"background": ""}
    anima._active_background_workers = {}
    anima._active_parallel_tasks = {}
    anima._task_semaphore = None
    anima._keepalive_while_busy = None
    return anima


def _executor(
    tmp_path: Path,
    *,
    task_isolated: bool = False,
    background_isolated: bool = False,
    pool_size: int = 1,
) -> tuple[PendingTaskExecutor, MagicMock, Path]:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    flags: dict[str, bool] = {}
    if task_isolated:
        flags["task"] = True
    if background_isolated:
        flags["background"] = True
    (anima_dir / "status.json").write_text(
        json.dumps(
            {
                "process_model": "phase2",
                "task_process_isolation": flags,
            }
        ),
        encoding="utf-8",
    )
    anima = _anima_double(tmp_path, pool_size=pool_size)
    supervisor = None
    if task_isolated or background_isolated:
        supervisor = TaskRunnerSupervisor(
            "sakura",
            anima_dir,
            anima.shared_dir,
            max_concurrent=pool_size,
        )
    shutdown = asyncio.Event()
    executor = PendingTaskExecutor(
        anima,
        "sakura",
        anima_dir,
        shutdown,
        task_runner_supervisor=supervisor,
    )
    return executor, anima, anima_dir


async def test_isolated_task_journal_recovery_attaches_checkpoint_before_delete(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    journal = StreamingJournal(anima_dir, session_type="task", thread_id="task-1")
    journal.open(trigger="task:task-1")
    journal.write_text("partial task output")
    journal.close()
    pending_executor = MagicMock()
    owner = SimpleNamespace(_pending_executor=pending_executor)
    supervisor = TaskRunnerSupervisor(
        "sakura",
        anima_dir,
        tmp_path / "shared",
        busy_status_owner=owner,
    )

    await supervisor._recover_task_journals(("task",))

    pending_executor.add_recovered_task_checkpoint.assert_called_once_with(
        "task-1",
        "partial task output",
        [],
    )
    assert not StreamingJournal.has_orphan(anima_dir, "task")


@pytest.mark.asyncio
async def test_task_flag_false_preserves_legacy_path_without_spawn(tmp_path: Path) -> None:
    executor, anima, _ = _executor(tmp_path, task_isolated=False)
    assert executor._task_isolated is False
    assert executor._task_runner_supervisor is None

    task_desc = {"task_id": "t-legacy", "title": "legacy", "description": "work", "task_type": "llm"}
    with (
        patch.object(executor, "_run_llm_task", new=AsyncMock(return_value="done")) as run_llm,
        patch.object(executor, "_sync_task_queue"),
        patch.object(executor, "_handle_goal_completion", new=AsyncMock()),
    ):
        await executor._execute_llm_task(task_desc)

    run_llm.assert_awaited_once()
    # No supervisor means no isolated spawn path.
    assert executor._task_runner_supervisor is None


@pytest.mark.asyncio
async def test_task_flag_true_uses_child_result_without_root_llm(tmp_path: Path) -> None:
    executor, anima, anima_dir = _executor(tmp_path, task_isolated=True)
    assert executor._task_isolated is True
    assert executor._task_runner_supervisor is not None

    processing = anima_dir / "state" / "pending" / "processing"
    processing.mkdir(parents=True)
    processing_path = processing / "t-iso.json"
    processing_path.write_text('{"task_id":"t-iso"}', encoding="utf-8")

    executor._task_runner_supervisor.run_task = AsyncMock(
        return_value={"task_type": "llm", "result": "child-done", "success": True}
    )
    # Avoid real worker pool acquire on MagicMock.
    anima._acquire_background_worker = None
    type(anima)._acquire_background_worker = None  # type: ignore[attr-defined]

    task_desc = {
        "task_id": "t-iso",
        "title": "isolated",
        "description": "work",
        "task_type": "llm",
    }
    with (
        patch.object(executor, "_run_llm_task", new=AsyncMock()) as run_llm,
        patch.object(executor, "_sync_task_queue") as sync,
        patch.object(executor, "_handle_goal_completion", new=AsyncMock()),
    ):
        await executor._execute_llm_task(task_desc, processing_path=processing_path)

    run_llm.assert_not_awaited()
    executor._task_runner_supervisor.run_task.assert_awaited_once()
    sync.assert_called()
    # Result path classified as done.
    assert sync.call_args[0][1] == "done"


@pytest.mark.asyncio
async def test_child_crash_records_failed_retryable_and_root_continues(tmp_path: Path) -> None:
    executor, anima, anima_dir = _executor(tmp_path, task_isolated=True)
    assert executor._task_runner_supervisor is not None
    type(anima)._acquire_background_worker = None  # type: ignore[attr-defined]

    executor._task_runner_supervisor.run_task = AsyncMock(
        side_effect=TaskRunnerError("task runner exited before returning a result (exit=-9)")
    )
    task_desc = {
        "task_id": "t-crash",
        "title": "crash",
        "description": "work",
        "task_type": "llm",
    }
    with (
        patch.object(executor, "_write_failed_result") as write_failed,
        patch.object(executor, "_sync_task_queue") as sync,
    ):
        await executor._execute_llm_task(task_desc)

    write_failed.assert_called_once()
    sync.assert_called()
    assert sync.call_args[0][1] == "failed"
    summary = str(sync.call_args.kwargs.get("summary") or sync.call_args[1].get("summary", ""))
    assert "INTERRUPTED" in summary
    # The original TaskRunnerError must survive into the summary (no flattening).
    assert "exit=-9" in summary


@pytest.mark.asyncio
async def test_child_crash_during_shutdown_stays_for_startup_recovery(tmp_path: Path) -> None:
    executor, anima, anima_dir = _executor(tmp_path, task_isolated=True)
    assert executor._task_runner_supervisor is not None
    type(anima)._acquire_background_worker = None  # type: ignore[attr-defined]
    executor._task_runner_supervisor.run_task = AsyncMock(
        side_effect=TaskRunnerError("task runner exited during shutdown")
    )
    task_desc = {
        "task_id": "t-shutdown-crash",
        "title": "shutdown crash",
        "description": "work",
        "task_type": "llm",
    }
    processing = anima_dir / "state" / "pending" / "processing"
    failed = anima_dir / "state" / "pending" / "failed"
    processing.mkdir(parents=True)
    failed.mkdir()
    processing_path = processing / "t-shutdown-crash.json"
    processing_path.write_text(json.dumps(task_desc), encoding="utf-8")
    executor._shutdown_event.set()

    await executor._execute_claimed_llm_task(task_desc, processing_path, failed, None)

    assert processing_path.exists()
    assert not list(failed.glob("*.json"))
    anima.messenger.send.assert_not_called()


@pytest.mark.asyncio
async def test_same_attempt_not_reclaimed_while_lease_live(tmp_path: Path) -> None:
    executor, _, anima_dir = _executor(tmp_path, task_isolated=True)
    processing = anima_dir / "state" / "pending" / "processing"
    processing.mkdir(parents=True)
    failed = anima_dir / "state" / "pending" / "failed"
    failed.mkdir(parents=True)
    path = processing / "t-attempt.json"
    path.write_text('{"task_id":"t-attempt"}', encoding="utf-8")

    attempt = executor._next_attempt("t-attempt")
    write_processing_lease(
        path,
        anima="sakura",
        task_id="t-attempt",
        pid=os.getpid(),
        job_id="job-1",
        task_pid=os.getpid(),
        pgid=os.getpid(),
        root_epoch="epoch",
        attempt=attempt,
        process_start_time=1.0,
    )
    with patch(
        "core.supervisor.pending_executor.is_processing_lease_live",
        return_value=True,
    ):
        # Force attempt tracker to same attempt as lease.
        executor._attempt_by_task_id["t-attempt"] = attempt
        claimed = executor._claim_processing_task(path, failed, {"task_id": "t-attempt"})
    assert claimed is None


@pytest.mark.asyncio
async def test_lease_v2_written_on_spawn_callback(tmp_path: Path) -> None:
    executor, anima, anima_dir = _executor(tmp_path, task_isolated=True)
    type(anima)._acquire_background_worker = None  # type: ignore[attr-defined]
    processing = anima_dir / "state" / "pending" / "processing"
    processing.mkdir(parents=True)
    processing_path = processing / "t-lease.json"
    processing_path.write_text('{"task_id":"t-lease"}', encoding="utf-8")

    async def _fake_run_task(task_desc, *, attempt=1, display_lane="background", on_spawned=None):
        identity = IPCV2Identity(
            job_id="job-lease",
            root_epoch="epoch-lease",
            attempt=attempt,
            lane="task",
            display_lane=display_lane,
        )
        job = SimpleNamespace(
            identity=identity,
            pid=4242,
            pgid=4242,
            process_start_time=123.45,
        )
        if on_spawned is not None:
            await on_spawned(job)
        return {"task_type": "llm", "result": "ok", "success": True}

    executor._task_runner_supervisor.run_task = AsyncMock(side_effect=_fake_run_task)
    with (
        patch.object(executor, "_sync_task_queue"),
        patch.object(executor, "_handle_goal_completion", new=AsyncMock()),
    ):
        await executor._execute_llm_task(
            {"task_id": "t-lease", "title": "x", "description": "y", "task_type": "llm"},
            processing_path=processing_path,
        )

    payload = read_processing_lease(processing_path)
    assert payload is not None
    assert payload.get("schema_version") == 2
    assert payload.get("task_pid") == 4242
    assert payload.get("job_id") == "job-lease"
    assert payload.get("attempt") == 1


@pytest.mark.asyncio
async def test_background_flag_false_uses_legacy_command_path(tmp_path: Path) -> None:
    executor, anima, _ = _executor(tmp_path, background_isolated=False)
    assert executor._background_isolated is False

    # Without isolation, command path needs BackgroundTaskManager.
    bg_mgr = MagicMock()
    bg_mgr.submit = MagicMock(return_value="bg-1")
    bg_mgr._async_tasks = {}
    agent = MagicMock()
    agent.background_manager = bg_mgr
    anima.agent = agent
    type(anima)._agent_for_lane = None  # type: ignore[attr-defined]

    result = await executor.execute_pending_task(
        {
            "task_id": "cmd-1",
            "task_type": "command",
            "tool_name": "echo",
            "subcommand": "",
            "raw_args": [],
        }
    )
    assert result is None
    bg_mgr.submit.assert_called_once()


@pytest.mark.asyncio
async def test_background_flag_true_spawns_child(tmp_path: Path) -> None:
    executor, _, _ = _executor(tmp_path, background_isolated=True)
    assert executor._background_isolated is True
    assert executor._task_runner_supervisor is not None
    executor._task_runner_supervisor.run_background = AsyncMock(
        return_value={"task_type": "command", "result": "ok", "success": True}
    )
    await executor.execute_pending_task(
        {
            "task_id": "cmd-iso",
            "task_type": "command",
            "tool_name": "echo",
            "subcommand": "hi",
            "raw_args": [],
        }
    )
    executor._task_runner_supervisor.run_background.assert_awaited_once()
    kwargs = executor._task_runner_supervisor.run_background.await_args
    assert (
        kwargs.kwargs["kind"] == "command"
        or kwargs[1].get("kind") == "command"
        or (kwargs.args and kwargs.args[0] == "command")
        or kwargs.kwargs.get("kind") == "command"
    )
    # Prefer kwargs form
    assert executor._task_runner_supervisor.run_background.await_args.kwargs["kind"] == "command"


@pytest.mark.asyncio
async def test_background_pool_limit_caps_concurrent_children(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    supervisor = TaskRunnerSupervisor("sakura", anima_dir, shared, max_concurrent=1)

    started = asyncio.Event()
    release = asyncio.Event()
    in_flight = 0
    max_seen = 0

    async def _slow_spawn_and_await(**kwargs):
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        started.set()
        await release.wait()
        in_flight -= 1
        return {"ok": True}

    with (
        patch.object(supervisor, "_spawn_and_await", new=AsyncMock(side_effect=_slow_spawn_and_await)),
        patch.object(
            TaskRunnerSupervisor,
            "_required_url_environment",
            return_value={"ANIMAWORKS_EMBED_URL": "http://embed.test"},
        ),
    ):
        first = asyncio.create_task(supervisor.run_background(kind="command", payload={"tool_name": "a"}))
        await started.wait()
        second = asyncio.create_task(supervisor.run_background(kind="command", payload={"tool_name": "b"}))
        await asyncio.sleep(0.05)
        assert max_seen == 1
        assert not second.done()
        release.set()
        await asyncio.gather(first, second)
        assert max_seen == 1


@pytest.mark.asyncio
async def test_sigkill_only_reaps_task_group_and_root_survives(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    anima_dir = tmp_path / "animas" / "sakura"
    shared.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    supervisor = TaskRunnerSupervisor("sakura", anima_dir, shared)
    identity = IPCV2Identity(
        job_id="job-kill-task",
        root_epoch=supervisor.root_epoch,
        attempt=1,
        lane="task",
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
        request_id="run-kill-task",
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
async def test_grace_sends_event_and_waits_for_ack(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    anima_dir = tmp_path / "animas" / "sakura"
    shared.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    supervisor = TaskRunnerSupervisor("sakura", anima_dir, shared)
    identity = IPCV2Identity(
        job_id="job-grace",
        root_epoch=supervisor.root_epoch,
        attempt=1,
        lane="task",
        display_lane="background",
    )
    connection = AsyncMock()
    connection.send_event = AsyncMock()
    job = TaskRunnerJob(
        identity=identity,
        request_id="run-grace",
        params={},
        result=asyncio.get_running_loop().create_future(),
        peer_state=IPCV2ConnectionState(identity),
        connection=connection,
        process=None,
    )
    # Pre-set grace_acked so close() does not wait forever.
    job.grace_acked.set()
    supervisor.jobs[identity.job_id] = job

    await supervisor.close()

    connection.send_event.assert_awaited()
    assert connection.send_event.await_args.args[0] == "grace"
    assert supervisor._accepting is False
