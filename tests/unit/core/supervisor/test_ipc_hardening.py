"""IPC hardening: keep received results through nonzero exit, retry ack waits,
reap leaked descendants, and keep file recovery off the event loop."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

from core.memory import streaming_journal
from core.supervisor import task_runner
from core.supervisor import task_runner_supervisor as trs
from core.supervisor.ipc_v2 import (
    IPCV2BackpressureTimeout,
    IPCV2ConnectionState,
    IPCV2Identity,
)
from core.supervisor.task_runner_supervisor import TaskRunnerSupervisor


def _supervisor(tmp_path: Path) -> TaskRunnerSupervisor:
    shared_dir = tmp_path / "shared"
    anima_dir = tmp_path / "animas" / "rin"
    shared_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    return TaskRunnerSupervisor("rin", anima_dir, shared_dir)


def _identity(job_id: str = "job-h") -> IPCV2Identity:
    return IPCV2Identity(
        job_id=job_id,
        root_epoch="epoch",
        attempt=1,
        lane="task",
        display_lane="background",
    )


# ── Task 1: received result is kept even when the child exits nonzero ───


async def _spawn_with_nonzero_exit(
    supervisor: TaskRunnerSupervisor,
    monkeypatch: pytest.MonkeyPatch,
    terminal: dict,
) -> dict:
    """Drive _spawn_and_await against a child that delivers a result but exits 3."""
    real_exec = asyncio.create_subprocess_exec

    async def fake_exec(*_args, **kwargs):  # exits nonzero after delivering result
        kwargs.pop("env", None)
        return await real_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(0.1); sys.exit(3)",
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(trs, "_TASK_RUNNER_EXIT_TIMEOUT", 5.0)

    run = asyncio.create_task(
        supervisor._spawn_and_await(
            lane="task",
            job_prefix="task",
            params_builder=lambda url_env: {"environment": {"urls": url_env}},
            log_context="harden-test",
            attempt=1,
            display_lane="background",
            on_spawned=None,
            url_env={"ANIMAWORKS_EMBED_URL": "http://localhost:0"},
        )
    )
    while not supervisor.jobs:
        await asyncio.sleep(0.01)
    job = next(iter(supervisor.jobs.values()))
    job.result.set_result(terminal)
    return await asyncio.wait_for(run, timeout=10)


@pytest.mark.asyncio
async def test_nonzero_exit_result_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    supervisor = _supervisor(tmp_path)
    terminal = {"result": {"task_type": "llm", "result": "ok", "success": True}}

    with caplog.at_level(logging.WARNING):
        result = await _spawn_with_nonzero_exit(supervisor, monkeypatch, terminal)

    assert result == {"task_type": "llm", "result": "ok", "success": True}
    assert any("result kept despite nonzero exit" in r.message for r in caplog.records)


# ── Task 2: terminal ack wait retries through backpressure timeouts ────


class _AckBackpressureConnection:
    """A connection whose wait_for_ack fails the first N times then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_remaining = fail_times
        self.ack_waits = 0
        self.closed = False

    async def send_response(self, request_id: str, *, result=None, error=None) -> int:
        return 42

    async def receive(self):
        await asyncio.Event().wait()  # cancellable, never yields a frame

    async def wait_for_ack(self, _seq: int) -> None:
        self.ack_waits += 1
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            raise IPCV2BackpressureTimeout("backpressure")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_send_terminal_retries_backpressured_ack(monkeypatch) -> None:
    monkeypatch.setattr(task_runner, "_TERMINAL_ACK_TIMEOUT", 2.0)
    monkeypatch.setattr(task_runner, "_TERMINAL_ACK_RETRY_SLEEP_SECONDS", 0.0)
    identity = _identity("job-ack")
    state = IPCV2ConnectionState(identity)
    connection = _AckBackpressureConnection(fail_times=2)

    result = await task_runner._send_terminal(
        connection,
        Path("/tmp/sock"),
        state,
        "run-1",
        None,
        result={"task_type": "llm", "result": "ok", "success": True},
    )

    assert result is connection
    assert connection.ack_waits == 3  # 2 backpressure retries, then success
    assert not connection.closed


# ── Task 3: descendant cleanup runs TERM then KILL, in order ──────────


class _FakeChildProcess:
    def __init__(self) -> None:
        self.signals: list[str] = []

    def terminate(self) -> None:
        self.signals.append("TERM")

    def kill(self) -> None:
        self.signals.append("KILL")


def test_cleanup_descendants_terms_then_kills(monkeypatch) -> None:
    import psutil as real_psutil

    children = [_FakeChildProcess(), _FakeChildProcess()]
    monkeypatch.setattr(
        real_psutil.Process,
        "children",
        lambda self, recursive=True: children,
    )
    # Everything survives the grace window → all get killed after TERM.
    monkeypatch.setattr(real_psutil, "wait_procs", lambda procs, timeout: ([], procs))

    task_runner._cleanup_descendants()

    for child in children:
        assert child.signals == ["TERM", "KILL"], child.signals


# ── Task 4: journal recovery uses a thread, not the event loop ────────


@pytest.mark.asyncio
async def test_recover_task_journals_runs_file_io_in_thread(
    tmp_path: Path, monkeypatch
) -> None:
    supervisor = _supervisor(tmp_path)
    real_to_thread = asyncio.to_thread
    thread_calls: list[str] = []

    async def fake_to_thread(fn, *args, **kwargs):
        thread_calls.append(getattr(fn, "__name__", repr(fn)))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        streaming_journal.StreamingJournal,
        "has_orphan",
        lambda *a, **k: False,
    )

    await supervisor._recover_task_journals(("task",))

    assert thread_calls, "disk recovery must run in a thread, not the event loop"
