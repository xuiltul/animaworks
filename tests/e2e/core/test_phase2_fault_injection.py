"""Phase 2 process-isolation fault-injection acceptance tests."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import pytest

from core.anima import DigitalAnima
from core.memory.rag.sqlite_health import quick_check_chroma_sqlite
from core.memory.streaming_journal import StreamingJournal
from core.memory.task_queue import TaskQueueManager
from core.platform.processing_lease import read_processing_lease, write_processing_lease
from core.schemas import CronTask
from core.supervisor import task_runner_supervisor
from core.supervisor.pending_executor import PendingTaskExecutor
from core.supervisor.scheduler_manager import SchedulerManager
from core.supervisor.task_runner_supervisor import TaskRunnerError, TaskRunnerJob, TaskRunnerSupervisor

pytestmark = [
    pytest.mark.timeout(60),
    pytest.mark.skipif(os.name != "posix", reason="fault injection requires POSIX process groups"),
]

_ANIMA = "fault-e2e"


@pytest.fixture
def phase2_anima(
    data_dir: Path,
    make_anima: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    anima_dir = make_anima(_ANIMA)
    monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://127.0.0.1:9")
    return anima_dir, data_dir / "shared"


def _cron(name: str, command: str) -> CronTask:
    return CronTask(
        name=name,
        schedule="0 0 1 1 *",
        type="command",
        command=command,
        trigger_heartbeat=False,
    )


async def _wait_for(
    probe: Callable[[], Any],
    *,
    timeout: float = 10.0,
) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = probe()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting for fault-injection condition")


def _job(supervisor: TaskRunnerSupervisor, name: str) -> TaskRunnerJob | None:
    return next(
        (job for job in supervisor.jobs.values() if (job.params.get("task") or {}).get("name") == name),
        None,
    )


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _events(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _record_event(path: Path, event: str, **data: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **data}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


async def _grace_root_harness(anima_dir: Path, event_path: Path) -> int:
    supervisor = TaskRunnerSupervisor(_ANIMA, anima_dir, anima_dir.parents[1] / "shared")
    running = asyncio.create_task(supervisor.run_cron(_cron("grace", "exec sleep 30")))
    job = await _wait_for(lambda: _job(supervisor, "grace"))
    await _wait_for(lambda: job.connection is not None and job.last_progress)
    journal = StreamingJournal(anima_dir, session_type="cron")
    journal.open("cron:grace", session_id="fault-e2e")
    journal.write_text("recover me")
    journal.close()
    stopping = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopping.set)
    _record_event(event_path, "ready", root_pid=os.getpid(), child_pid=job.pid, child_pgid=job.pgid)
    await stopping.wait()
    shutdown = asyncio.create_task(supervisor.close())
    await job.grace_acked.wait()
    _record_event(event_path, "grace_ack")
    await shutdown
    await asyncio.gather(running, return_exceptions=True)
    _record_event(event_path, "root_exit")
    return 0


def _seed_health_db(anima_dir: Path) -> tuple[Path, tuple[frozenset[str], int, str, str]]:
    persist_dir = anima_dir / "vectordb"
    persist_dir.mkdir()
    with sqlite3.connect(persist_dir / "chroma.sqlite3") as connection:
        journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        connection.execute("CREATE TABLE collections (name TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE chunks (collection TEXT, content TEXT)")
        connection.executemany("INSERT INTO collections VALUES (?)", [("episodes",), ("knowledge",)])
        connection.execute("INSERT INTO chunks VALUES (?, ?)", ("knowledge", "fault-safe sentinel"))
    return persist_dir, _db_snapshot(persist_dir, expected_mode=journal_mode)


def _db_snapshot(persist_dir: Path, *, expected_mode: str = "wal") -> tuple[frozenset[str], int, str, str]:
    health = quick_check_chroma_sqlite(persist_dir)
    assert health.status == "ok"
    with sqlite3.connect(persist_dir / "chroma.sqlite3") as connection:
        collections = frozenset(row[0] for row in connection.execute("SELECT name FROM collections"))
        chunk_count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        hit = connection.execute(
            "SELECT content FROM chunks WHERE content LIKE ?",
            ("%fault-safe%",),
        ).fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode == expected_mode == "wal"
    return collections, chunk_count, hit, journal_mode


@pytest.mark.asyncio
async def test_cron_sigkill_preserves_root_next_cycle_and_db(phase2_anima: tuple[Path, Path]) -> None:
    anima_dir, shared_dir = phase2_anima
    supervisor = TaskRunnerSupervisor(_ANIMA, anima_dir, shared_dir)
    persist_dir, db_before = _seed_health_db(anima_dir)
    root_pid = os.getpid()
    running = asyncio.create_task(supervisor.run_cron(_cron("killed", "exec sleep 30")))

    try:
        job = await _wait_for(lambda: _job(supervisor, "killed"))
        await _wait_for(lambda: job.connection is not None and job.last_progress)
        assert job.pid and job.pgid and job.pid != root_pid

        os.killpg(job.pgid, signal.SIGKILL)
        with pytest.raises(TaskRunnerError, match="exited before returning"):
            await asyncio.wait_for(running, timeout=5)

        await _wait_for(lambda: not _group_exists(job.pgid))
        result = await asyncio.wait_for(
            supervisor.run_cron(_cron("next-cycle", "printf next-cycle-ok")),
            timeout=15,
        )

        assert os.getpid() == root_pid
        assert result["success"] is True
        assert result["result"]["stdout"] == "next-cycle-ok"
        assert _db_snapshot(persist_dir) == db_before
    finally:
        await supervisor.close()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_root_restart_recovers_background_lease_once(
    phase2_anima: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir, shared_dir = phase2_anima
    monkeypatch.setattr(task_runner_supervisor, "_TASK_RUNNER_GRACE_TIMEOUT", 0.2)
    old_root = TaskRunnerSupervisor(_ANIMA, anima_dir, shared_dir)
    queue = TaskQueueManager(anima_dir)
    entry = queue.add_task(
        source="anima",
        original_instruction="fault injection",
        assignee=_ANIMA,
        summary="background interrupted by restart",
        status="in_progress",
    )
    processing_dir = anima_dir / "state" / "background_tasks" / "pending" / "processing"
    failed_dir = processing_dir.parent / "failed"
    processing_dir.mkdir(parents=True)
    failed_dir.mkdir()
    descriptor = processing_dir / f"{entry.task_id}.json"
    descriptor.write_text(json.dumps({"task_id": entry.task_id}), encoding="utf-8")
    spawned = asyncio.Event()
    captured: list[TaskRunnerJob] = []

    async def on_spawned(job: TaskRunnerJob) -> None:
        captured.append(job)
        write_processing_lease(
            descriptor,
            anima=_ANIMA,
            task_id=entry.task_id,
            pid=os.getpid(),
            job_id=job.identity.job_id,
            task_pid=job.pid,
            pgid=job.pgid,
            root_epoch=job.identity.root_epoch,
            attempt=1,
            process_start_time=job.process_start_time,
        )
        os.kill(job.pid or 0, signal.SIGSTOP)
        spawned.set()

    background = asyncio.create_task(
        old_root.run_background(
            kind="command",
            payload={"tool_name": "fault-injection", "raw_args": []},
            on_spawned=on_spawned,
        )
    )
    try:
        await asyncio.wait_for(spawned.wait(), timeout=5)
        lease = read_processing_lease(descriptor)
        assert lease and lease["attempt"] == 1 and lease["task_pid"] == captured[0].pid
    finally:
        await old_root.close()
    with pytest.raises(TaskRunnerError):
        await background

    new_root = TaskRunnerSupervisor(_ANIMA, anima_dir, shared_dir)
    try:
        PendingTaskExecutor._recover_processing(processing_dir, anima_dir)
        failed = failed_dir / descriptor.name
        failed_lease = read_processing_lease(failed)

        assert not descriptor.exists()
        assert not failed.exists()
        assert failed_lease is None
        recovered = queue.get_task_by_id(entry.task_id)
        assert recovered.status == "pending"
        assert recovered.meta["last_run_stop_kind"] == "crash"
        assert "INTERRUPTED" in recovered.summary and "PARTIALLY EXECUTED" in recovered.summary
        assert len(captured) == 1
        assert new_root.root_epoch != old_root.root_epoch
        assert not new_root.jobs
    finally:
        await new_root.close()


@pytest.mark.asyncio
async def test_hang_kills_only_stalled_group_while_other_lane_continues(
    phase2_anima: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir, shared_dir = phase2_anima
    threshold = 6.0
    supervisor = TaskRunnerSupervisor(
        _ANIMA,
        anima_dir,
        shared_dir,
        busy_hang_threshold_sec=30,
    )
    supervisor._hang_check_interval = 0.25
    monkeypatch.setattr(task_runner_supervisor, "_TASK_RUNNER_TERM_TIMEOUT", 0.1)
    root_pid = os.getpid()
    stalled = asyncio.create_task(supervisor.run_cron(_cron("stalled", "exec sleep 30")))

    try:
        job = await _wait_for(lambda: _job(supervisor, "stalled"))
        await _wait_for(lambda: job.connection is not None and job.last_progress)
        supervisor._busy_hang_threshold_sec = threshold
        job.last_progress_at = asyncio.get_running_loop().time()
        stopped_at = asyncio.get_running_loop().time()
        os.kill(job.pid or 0, signal.SIGSTOP)

        healthy = await asyncio.wait_for(
            supervisor.run_cron(_cron("healthy", "printf lane-ok")),
            timeout=15,
        )
        with pytest.raises(TaskRunnerError):
            await asyncio.wait_for(stalled, timeout=8)
        elapsed = asyncio.get_running_loop().time() - stopped_at

        assert healthy["success"] is True
        assert healthy["result"]["stdout"] == "lane-ok"
        assert job.hang_kill_started
        assert elapsed <= threshold + (supervisor._hang_check_interval * 2) + 0.5
        assert not _group_exists(job.pgid or 0)
        assert os.getpid() == root_pid
    finally:
        await supervisor.close()
        await asyncio.gather(stalled, return_exceptions=True)


@pytest.mark.asyncio
async def test_parallel_spawn_finishes_and_busy_sidecar_stays_root_owned(
    phase2_anima: tuple[Path, Path],
) -> None:
    anima_dir, shared_dir = phase2_anima
    owner = DigitalAnima(anima_dir, shared_dir)
    supervisor = TaskRunnerSupervisor(_ANIMA, anima_dir, shared_dir, busy_status_owner=owner)
    root_pid = os.getpid()
    tasks = [
        asyncio.create_task(supervisor.run_cron(_cron(f"load-{index}", f"sleep 0.3; printf load-{index}")))
        for index in range(4)
    ]

    try:
        await _wait_for(lambda: len(supervisor.jobs) == 4 and all(job.pid for job in supervisor.jobs.values()))
        sidecar = shared_dir.parent / "run" / "animas" / f"{_ANIMA}.busy.json"
        await _wait_for(sidecar.exists)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        child_pids = {job.pid for job in supervisor.jobs.values()}
        root_epochs = {job.identity.root_epoch for job in supervisor.jobs.values()}
        busy_pids = [payload["pid"]]
        while not all(task.done() for task in tasks):
            try:
                busy_pids.append(json.loads(sidecar.read_text(encoding="utf-8"))["pid"])
            except FileNotFoundError:
                pass
            await asyncio.sleep(0.02)
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=20)

        assert all(result["success"] for result in results)
        assert len(child_pids) == 4
        assert root_pid not in child_pids
        assert set(busy_pids) == {root_pid}
        assert payload["anima"] == _ANIMA
        assert root_epochs == {supervisor.root_epoch}
        assert not supervisor.jobs
    finally:
        await supervisor.close()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_root_grace_ack_precedes_exit_and_recovers_journal(
    phase2_anima: tuple[Path, Path],
) -> None:
    anima_dir, _ = phase2_anima
    event_path = anima_dir.parents[1] / "grace-events.jsonl"
    env = {
        **os.environ,
        "ANIMAWORKS_FAULT_ROOT_HARNESS": str(anima_dir),
        "ANIMAWORKS_FAULT_EVENT_PATH": str(event_path),
    }
    root = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__)),
        env=env,
        start_new_session=True,
    )
    ready: dict[str, Any] | None = None

    try:
        ready = await _wait_for(
            lambda: next((event for event in _events(event_path) if event["event"] == "ready"), None),
            timeout=15,
        )
        assert ready["root_pid"] == root.pid
        assert ready["child_pid"] != root.pid

        os.kill(root.pid, signal.SIGTERM)
        assert await asyncio.wait_for(root.wait(), timeout=15) == 0
    finally:
        if root.returncode is None:
            os.killpg(root.pid, signal.SIGKILL)
            await root.wait()
        if ready and _group_exists(ready["child_pgid"]):
            os.killpg(ready["child_pgid"], signal.SIGKILL)

    assert [event["event"] for event in _events(event_path)] == ["ready", "grace_ack", "root_exit"]
    assert not _group_exists(ready["child_pgid"])
    assert not StreamingJournal.has_orphan(anima_dir, session_type="cron")


@pytest.mark.asyncio
async def test_disabled_flag_runs_legacy_without_task_runner(
    phase2_anima: tuple[Path, Path],
) -> None:
    anima_dir, shared_dir = phase2_anima
    status_path = anima_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "process_model": "phase2",
            "task_process_isolation": {
                "cron": False,
                "heartbeat": False,
                "task": False,
                "background": False,
            },
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")
    anima = DigitalAnima(anima_dir, shared_dir)
    events: list[tuple[str, dict[str, Any]]] = []
    manager = SchedulerManager(anima, _ANIMA, anima_dir, lambda event, data: events.append((event, data)))

    running = asyncio.create_task(manager._run_cron_task(_cron("legacy", "sleep 0.3; printf legacy-ok")))
    await asyncio.sleep(0.1)
    descendants = psutil.Process(os.getpid()).children(recursive=True)
    task_runners = []
    for process in descendants:
        try:
            if "core.supervisor.task_runner" in " ".join(process.cmdline()):
                task_runners.append(process.pid)
        except psutil.Error:
            pass
    await asyncio.wait_for(running, timeout=10)

    assert manager._task_runner_supervisor is None
    assert task_runners == []
    assert events[0][1]["result"]["stdout"] == "legacy-ok"


if __name__ == "__main__" and os.environ.get("ANIMAWORKS_FAULT_ROOT_HARNESS"):
    raise SystemExit(
        asyncio.run(
            _grace_root_harness(
                Path(os.environ["ANIMAWORKS_FAULT_ROOT_HARNESS"]),
                Path(os.environ["ANIMAWORKS_FAULT_EVENT_PATH"]),
            )
        )
    )
