from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import _SENTINEL_CANCELLED, PendingTaskExecutor


def _make_executor(tmp_path: Path, stop_kind: str = "normal") -> PendingTaskExecutor:
    anima_dir = tmp_path / "animas" / "test-anima"
    (anima_dir / "state").mkdir(parents=True)
    anima = MagicMock()
    anima._background_lock = asyncio.Lock()
    anima._task_semaphore = None
    anima._status_slots = {"background": "idle"}
    anima._task_slots = {"background": ""}
    anima._active_parallel_tasks = {}
    anima._active_background_workers = {}

    async def stream(*_args, **_kwargs):
        yield {"type": "text_delta", "text": "output"}
        yield {
            "type": "cycle_done",
            "cycle_result": {
                "summary": "result",
                "action": "skipped" if stop_kind == "budget_skipped" else "responded",
                "stop_kind": stop_kind,
                "tool_call_records": [],
            },
        }

    anima.agent.run_cycle_streaming = stream
    return PendingTaskExecutor(
        anima=anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _task(task_id: str, **overrides) -> dict:
    task = {
        "task_type": "llm",
        "task_id": task_id,
        "title": "Stop kind test",
        "description": "finish the task",
        "context": "original context",
        "reply_to": None,
        "submitted_at": "",
    }
    task.update(overrides)
    return task


def _queue_task(executor: PendingTaskExecutor, task_id: str, *, status: str = "in_progress") -> TaskQueueManager:
    manager = TaskQueueManager(executor._anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="finish the task",
        assignee="test-anima",
        summary="stop kind test",
        task_id=task_id,
        status=status,
    )
    return manager


@contextmanager
def _execution_patches():
    with (
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger"),
    ):
        yield


@pytest.mark.asyncio
async def test_interrupted_without_declaration_returns_to_pending(tmp_path: Path) -> None:
    """An interrupted run that never declared goes back to pending, not re-enqueued."""
    executor = _make_executor(tmp_path, "interrupted")
    manager = _queue_task(executor, "interrupted")

    with _execution_patches():
        await executor._execute_llm_task(_task("interrupted"))

    entry = manager.get_task_by_id("interrupted")
    assert entry is not None
    assert entry.status == "pending"
    assert entry.meta["last_run_stop_kind"] == "interrupted"
    assert entry.meta["last_run_ended_at"]
    assert not (executor._anima_dir / "state" / "pending" / "interrupted.json").exists()


@pytest.mark.asyncio
async def test_normal_stop_without_declaration_returns_to_pending(tmp_path: Path) -> None:
    """A run that finishes normally but never declares is handed back as pending."""
    executor = _make_executor(tmp_path)
    manager = _queue_task(executor, "undeclared")

    with _execution_patches():
        await executor._execute_llm_task(_task("undeclared"))

    entry = manager.get_task_by_id("undeclared")
    assert entry is not None
    assert entry.status == "pending"
    assert entry.summary == "stop kind test"
    assert entry.meta["last_run_note"] == "run ended without a completion declaration"
    assert entry.meta["last_run_stop_kind"] == "normal"
    assert entry.meta["last_run_ended_at"]
    # No descriptor is regenerated: nothing re-runs it on the anima's behalf.
    assert not list((executor._anima_dir / "state" / "pending").glob("*.json"))
    executor._anima.messenger.send.assert_not_called()
    result_path = executor._anima_dir / "state" / "task_results" / "undeclared.md"
    assert result_path.read_text() == "(undeclared)\n\nresult"


@pytest.mark.asyncio
async def test_undeclared_result_is_saved_for_its_attempt_without_completing(tmp_path: Path) -> None:
    from core.taskboard.tasks import process_identity
    from core.tasks_dispatch import publish_tasks

    executor = _make_executor(tmp_path)
    publish_tasks(executor._anima_dir, [_task("attempt-result")])
    manager = TaskQueueManager(executor._anima_dir)
    claim = manager.store.claim("test-anima", "attempt-result", process_identity())
    assert claim is not None
    token = claim["_attempt_token"]

    with _execution_patches():
        await executor._execute_canonical_task(claim)

    result_dir = executor._anima_dir / "state" / "task_results"
    assert (result_dir / "attempt-result" / f"{token}.md").read_text() == "(undeclared)\n\nresult"
    assert not (result_dir / "attempt-result.md").exists()
    assert manager.get_task_by_id("attempt-result").status == "pending"
    executor._anima.messenger.send.assert_not_called()
    with manager.store.reader() as db:
        attempt = db.execute("SELECT * FROM task_attempts WHERE token=?", (token,)).fetchone()
    assert attempt["stop_kind"] == "normal"
    assert attempt["result_ref"] == f"state/task_results/attempt-result/{token}.md"


@pytest.mark.asyncio
@pytest.mark.parametrize("has_partial_artifact", [False, True])
async def test_external_cancel_keeps_terminal_status_and_only_references_real_artifact(
    tmp_path: Path, has_partial_artifact: bool
) -> None:
    """A SIGTERM before child result must not become a successful empty run."""
    from core.taskboard.tasks import process_identity
    from core.tasks_dispatch import publish_tasks

    executor = _make_executor(tmp_path)
    executor._task_isolated = True
    executor._task_runner_supervisor = MagicMock()
    payload = _task("external-cancel")
    publish_tasks(executor._anima_dir, [payload])
    manager = TaskQueueManager(executor._anima_dir)
    claim = manager.store.claim("test-anima", "external-cancel", process_identity())
    assert claim is not None
    token = claim["_attempt_token"]
    result_ref = f"state/task_results/external-cancel/{token}.md"

    async def cancelled_child(*_args, **_kwargs):
        if has_partial_artifact:
            executor._save_task_result("external-cancel", "observed partial result")
        manager.update_status("external-cancel", "cancelled", summary="work no longer needed")
        raise RuntimeError("task runner exited before returning a result (exit=-15)")

    executor._run_task_in_worker = AsyncMock(side_effect=cancelled_child)
    with _execution_patches():
        await executor._execute_canonical_task(claim)

    executor._run_task_in_worker.assert_awaited_once()
    entry = manager.get_task_by_id("external-cancel")
    assert entry.status == "cancelled"
    assert entry.summary == "work no longer needed"
    assert entry.meta["last_run_stop_kind"] == "interrupted"
    assert manager.store.active_attempts("test-anima") == []
    assert manager.store.pending("test-anima") == []
    assert manager.store.wakeups("test-anima") == []
    assert manager.store.get_input("test-anima", "external-cancel")["description"] == payload["description"]
    executor._anima.messenger.send.assert_not_called()
    with manager.store.reader() as db:
        attempt = db.execute("SELECT * FROM task_attempts WHERE token=?", (token,)).fetchone()
    assert attempt["ended_at"]
    assert attempt["stop_kind"] == "interrupted"
    assert attempt["result_ref"] == (result_ref if has_partial_artifact else "")
    if has_partial_artifact:
        assert (executor._anima_dir / result_ref).read_text() == "observed partial result"
    else:
        assert not (executor._anima_dir / result_ref).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_kind", "declare_done", "expected_status"),
    [
        ("normal", True, "done"),
        ("interrupted", False, "pending"),
        ("interrupted", True, "done"),
        ("budget_skipped", False, "pending"),
    ],
)
async def test_resumed_attempt_records_its_own_stop_kind(
    tmp_path: Path, stop_kind: str, declare_done: bool, expected_status: str
) -> None:
    """A previous crash must not override a resumed run's actual outcome."""
    from core.taskboard.tasks import process_identity
    from core.tasks_dispatch import publish_tasks

    executor = _make_executor(tmp_path, stop_kind)
    task_id = "resumed"
    payload = _task(task_id)
    publish_tasks(executor._anima_dir, [payload])
    manager = TaskQueueManager(executor._anima_dir)
    first = manager.store.claim("test-anima", task_id, process_identity())
    assert first is not None
    manager.update_meta(
        task_id,
        {"last_run_ended_at": "2026-09-08T20:25:28+09:00", "last_run_note": "old failure", "business_tag": "keep"},
    )
    manager.store.finish(first["_attempt_token"], status="pending", stop_kind="crash")
    assert manager.store.pending("test-anima") == []  # No implicit retry.
    publish_tasks(executor._anima_dir, [{"task_id": task_id, "resume": True}])
    second = manager.store.claim("test-anima", task_id, process_identity())
    assert second is not None
    assert second["_attempt_number"] == 2
    claimed = manager.get_task_by_id(task_id)
    assert claimed.meta["business_tag"] == "keep"
    assert not {"last_run_stop_kind", "last_run_ended_at", "last_run_note"}.intersection(claimed.meta)
    assert manager.store.get_input("test-anima", task_id)["description"] == payload["description"]

    stream = executor._anima.agent.run_cycle_streaming

    async def run(*args, **kwargs):
        if declare_done:
            manager.update_meta(task_id, {"completed_by": "agent_declaration", "result_note": "verified"})
            manager.update_status(task_id, "done")
        async for chunk in stream(*args, **kwargs):
            yield chunk

    executor._anima.agent.run_cycle_streaming = run
    with _execution_patches():
        await executor._execute_canonical_task(second)

    entry = manager.get_task_by_id(task_id)
    assert entry.status == expected_status
    assert entry.meta["last_run_stop_kind"] == stop_kind
    assert entry.meta["last_attempt_token"] == second["_attempt_token"]
    assert entry.meta["last_run_ended_at"] != "2026-09-08T20:25:28+09:00"
    assert entry.meta.get("last_run_note") != "old failure"
    assert manager.store.active_attempts("test-anima") == []
    assert manager.store.pending("test-anima") == []
    with manager.store.reader() as db:
        attempts = db.execute(
            "SELECT number,stop_kind FROM task_attempts WHERE task_id=? ORDER BY number", (task_id,)
        ).fetchall()
    assert [tuple(row) for row in attempts] == [(1, "crash"), (2, stop_kind)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_kind", ["crash", "interrupted"])
async def test_runner_termination_overrides_earlier_normal_cycle_metadata(tmp_path: Path, stop_kind: str) -> None:
    from core.taskboard.tasks import process_identity
    from core.tasks_dispatch import publish_tasks

    executor = _make_executor(tmp_path)
    task_id = "terminated-after-cycle"
    publish_tasks(executor._anima_dir, [_task(task_id)])
    manager = TaskQueueManager(executor._anima_dir)
    claim = manager.store.claim("test-anima", task_id, process_identity())
    assert claim is not None

    async def terminate_after_cycle(_payload):
        manager.update_status(task_id, "done")
        executor._record_run_ended(task_id, "normal")
        if stop_kind == "interrupted":
            raise asyncio.CancelledError
        raise RuntimeError("runner failed after recording its cycle outcome")

    executor.execute_pending_task = terminate_after_cycle
    if stop_kind == "interrupted":
        with pytest.raises(asyncio.CancelledError):
            await executor._execute_canonical_task(claim)
    else:
        await executor._execute_canonical_task(claim)

    entry = manager.get_task_by_id(task_id)
    assert entry.status == "done"  # A declared business result is not rolled back.
    assert entry.meta["last_run_stop_kind"] == stop_kind
    assert manager.store.pending("test-anima") == []
    with manager.store.reader() as db:
        attempt = db.execute("SELECT stop_kind FROM task_attempts WHERE token=?", (claim["_attempt_token"],)).fetchone()
    assert attempt["stop_kind"] == stop_kind


@pytest.mark.asyncio
async def test_mid_run_cancel_preserves_owners_business_reason(tmp_path: Path) -> None:
    from core.taskboard.tasks import process_identity
    from core.tasks_dispatch import publish_tasks

    executor = _make_executor(tmp_path)
    task_id = "cancelled-duplicate"
    publish_tasks(executor._anima_dir, [_task(task_id)])
    manager = TaskQueueManager(executor._anima_dir)
    claim = manager.store.claim("test-anima", task_id, process_identity())
    assert claim is not None
    reason = "Prior task already completed the work; cancel this duplicate"

    async def cancelled_during_stream(*_args, **_kwargs):
        yield {"type": "text_delta", "text": "Partial work"}
        manager.update_status(task_id, "cancelled", summary=reason)
        yield {"type": "cycle_done", "cycle_result": {"summary": "Partial work", "stop_kind": "interrupted"}}

    executor._anima.agent.run_cycle_streaming = cancelled_during_stream
    with _execution_patches():
        await executor._execute_canonical_task(claim)

    entry = manager.get_task_by_id(task_id)
    assert entry.status == "cancelled"
    assert entry.summary == reason
    assert entry.meta["last_run_stop_kind"] == "interrupted"
    assert manager.store.active_attempts("test-anima") == []
    assert manager.store.pending("test-anima") == []


def test_new_cancel_uses_generic_localized_summary(tmp_path: Path) -> None:
    from core.i18n import t
    from core.supervisor.pending_executor import _classify_task_result

    executor = _make_executor(tmp_path)
    manager = _queue_task(executor, "new-cancel")
    status, summary = _classify_task_result(_SENTINEL_CANCELLED)
    executor._sync_task_queue("new-cancel", status, summary=summary)
    entry = manager.get_task_by_id("new-cancel")
    assert entry.status == "cancelled"
    assert entry.summary == t("pending_executor.task_cancelled")


@pytest.mark.asyncio
async def test_interrupted_after_declaration_stays_done(tmp_path: Path) -> None:
    """A run that declared done before being interrupted keeps its declaration."""
    executor = _make_executor(tmp_path, "interrupted")
    manager = _queue_task(executor, "declared")

    async def declared_then_interrupted(*_args, **_kwargs):
        manager.update_status("declared", "done")
        manager.update_meta("declared", {"completed_by": "agent_declaration", "result_note": "verified"})
        yield {"type": "text_delta", "text": "output"}
        yield {
            "type": "cycle_done",
            "cycle_result": {
                "summary": "result",
                "action": "responded",
                "stop_kind": "interrupted",
                "tool_call_records": [],
            },
        }

    executor._anima.agent.run_cycle_streaming = declared_then_interrupted

    with _execution_patches():
        await executor._execute_llm_task(_task("declared"))

    entry = manager.get_task_by_id("declared")
    assert entry is not None
    assert entry.status == "done"
    assert not (executor._anima_dir / "state" / "pending" / "declared.json").exists()


@pytest.mark.asyncio
async def test_budget_skipped_keeps_queue_pending_and_records_activity(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path, "budget_skipped")
    manager = _queue_task(executor, "budget")

    with (
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger") as activity,
    ):
        await executor._execute_llm_task(_task("budget"))

    entry = manager.get_task_by_id("budget")
    assert entry is not None
    assert entry.status == "pending"
    assert activity.return_value.log.call_args_list[-1].kwargs["meta"]["status"] == "budget_skipped"


@pytest.mark.asyncio
async def test_cancelled_batch_result_does_not_start_dependent(tmp_path: Path) -> None:
    from core.taskboard.tasks import process_identity
    from core.tasks_dispatch import publish_tasks

    executor = _make_executor(tmp_path)
    publish_tasks(
        executor._anima_dir,
        [
            _task("parent", batch_id="batch", parallel=False),
            _task("child", batch_id="batch", parallel=False, depends_on=["parent"]),
        ],
    )
    manager = TaskQueueManager(executor._anima_dir)
    store = manager.store
    claim = store.claim("test-anima", "parent", process_identity())
    assert claim is not None
    assert store.claim("test-anima", "child", process_identity()) is None
    executor._run_llm_task = AsyncMock(return_value=_SENTINEL_CANCELLED)

    with _execution_patches():
        await executor._execute_canonical_task(claim)

    executor._run_llm_task.assert_awaited_once()
    assert store.get("test-anima", "parent").status == "cancelled"
    assert store.active_attempts("test-anima") == []
    assert store.claim("test-anima", "child", process_identity()) is None
    child = manager.get_task_by_id("child")
    assert child is not None
    assert child.status == "pending"
    assert child.summary == "Stop kind test"
    wakeups = [event for event in store.wakeups("test-anima") if event["task_id"] == "child"]
    assert len(wakeups) == 1
    assert wakeups[0]["reason"] == "dependency_cancelled"
    assert store.claim("test-anima", "child", process_identity()) is None


@pytest.mark.asyncio
async def test_normal_stop_with_declaration_completes(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    manager = _queue_task(executor, "normal")
    manager.update_meta("normal", {"completed_by": "agent_declaration", "result_note": "verified"})
    manager.update_status("normal", "done")

    with _execution_patches():
        await executor._execute_llm_task(_task("normal"))

    entry = manager.get_task_by_id("normal")
    assert entry is not None
    assert entry.status == "done"
    assert entry.summary == "verified"


@pytest.mark.asyncio
async def test_stream_error_is_not_suppressed_without_declaration(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    manager = _queue_task(executor, "stream-error")

    async def error_stream(*_args, **_kwargs):
        yield {"type": "error", "message": "connection lost"}
        yield {
            "type": "cycle_done",
            "cycle_result": {"summary": "partial", "action": "responded", "stop_kind": "normal"},
        }

    executor._anima.agent.run_cycle_streaming = error_stream
    with _execution_patches():
        await executor._execute_llm_task(_task("stream-error"))

    entry = manager.get_task_by_id("stream-error")
    assert entry is not None
    assert entry.status == "pending"
    assert "streaming error" in entry.meta["last_run_note"]
    assert entry.meta["last_run_stop_kind"] == "crash"
