from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.memory.task_queue import TaskQueueManager
from core.taskboard.tasks import TaskStore, attempt_scope, process_identity, task_database_path


def _claim(db: str, name: str, task_id: str):
    return TaskStore(Path(db)).claim(name, task_id, process_identity())


@pytest.fixture
def queue(tmp_path):
    directory = tmp_path / "animas" / "worker"
    directory.mkdir(parents=True)
    return TaskQueueManager(directory)


def _payload(task_id="task", **kwargs):
    return {"task_id": task_id, "task_type": "llm", "title": "Task", "description": "Do work", **kwargs}


def test_complete_input_is_preserved_and_no_descriptor_needed(queue):
    payload = _payload(
        description="long" * 7000,
        context="context" * 6000,
        model="openai/example",
        working_directory="/work",
        workspace="sandbox",
        acceptance_criteria=["tested"],
        constraints=["do not send"],
        file_paths=["src/a.py"],
        submitted_by="manager",
        reply_to="manager",
        relay_chain=["manager", "worker"],
    )
    entry = queue.submit(payload)
    assert entry.original_instruction == payload["description"]
    assert entry.relay_chain == ["manager", "worker"]
    assert queue.store.get_input("worker", "task") == payload
    assert not queue.queue_path.exists()
    assert not (queue.anima_dir / "state" / "pending").exists()


def test_processes_cannot_double_claim(queue):
    queue.submit(_payload())
    with ProcessPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(_claim, [str(queue.store.db_path)] * 12, ["worker"] * 12, ["task"] * 12))
    assert sum(claim is not None for claim in claims) == 1
    assert len(queue.store.active_attempts("worker")) == 1


def test_old_attempt_cannot_finish_or_declare_after_resume(queue):
    payload = _payload()
    queue.submit(payload)
    first = queue.store.claim("worker", "task", process_identity())
    assert queue.store.finish(first["_attempt_token"], status="pending", stop_kind="interrupted")
    assert len(queue.store.wakeups("worker")) == 1
    assert queue.store.pending("worker") == []
    queue.submit(payload, resume=True)
    second = queue.store.claim("worker", "task", process_identity())
    assert second["_attempt_number"] == 2
    assert queue.store.wakeups("worker") == []
    assert not queue.store.finish(first["_attempt_token"], status="done", stop_kind="late")
    with (
        attempt_scope({"anima": "worker", "task_id": "task", "token": first["_attempt_token"]}),
        pytest.raises(ValueError, match="Stale"),
    ):
        queue.update_status("task", "done")
    assert queue.get_task_by_id("task").status == "in_progress"


def test_cancel_sticks_and_terminal_delivery_is_idempotent(queue):
    payload = _payload()
    queue.submit(payload)
    claim = queue.store.claim("worker", "task", process_identity())
    queue.update_status("task", "cancelled")
    queue.store.finish(claim["_attempt_token"], status="done", stop_kind="late")
    queue.submit(payload)
    assert queue.get_task_by_id("task").status == "cancelled"
    assert queue.store.pending("worker") == []
    with pytest.raises(ValueError, match="terminal"):
        queue.submit(payload, resume=True)


def test_transaction_rolls_back_entire_batch(queue):
    with pytest.raises(ValueError), queue.store.transaction():
        queue.submit(_payload("a"))
        queue.submit({"task_id": "b"}, source="bad")
    assert queue.store.read("worker") == {}


def test_dependencies_and_serial_exclusion_are_transactional(queue):
    queue.submit(_payload("a", batch_id="batch", parallel=True))
    queue.submit(_payload("b", batch_id="batch", parallel=True, depends_on=["a"]))
    queue.submit(_payload("c", batch_id="batch", parallel=False))
    assert queue.store.claim("worker", "b", process_identity(), max_active=3) is None
    first = queue.store.claim("worker", "a", process_identity(), max_active=3)
    assert queue.store.claim("worker", "c", process_identity(), max_active=3) is None
    queue.store.finish(first["_attempt_token"], status="done", stop_kind="completed")
    assert queue.store.claim("worker", "b", process_identity(), max_active=3)


def test_alias_has_one_authoritative_record(queue):
    queue.submit(_payload())
    queue.store.alias("manager", "old-tracker", "worker", "task")
    assert queue.store.read("manager")["old-tracker"].status == "delegated"
    queue.update_status("task", "done")
    assert queue.store.read("manager")["old-tracker"].status == "done"
    with queue.store.reader() as db:
        assert db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_cancelled_active_attempt_cannot_reopen_itself(queue):
    queue.submit(_payload())
    claimed = queue.store.claim("worker", "task", process_identity())
    queue.update_status("task", "cancelled")
    with (
        attempt_scope({"anima": "worker", "task_id": "task", "token": claimed["_attempt_token"]}),
        pytest.raises(ValueError, match="active attempt"),
    ):
        queue.update_status("task", "pending")
    assert queue.get_task_by_id("task").status == "cancelled"


def test_backlog_add_cannot_replace_published_execution_input(queue):
    queue.submit(_payload())
    before = queue.store.get_input("worker", "task")
    with pytest.raises(ValueError, match="execution input"):
        queue.add_task(
            source="human", original_instruction="replacement", assignee="worker", summary="replacement", task_id="task"
        )
    assert queue.store.get_input("worker", "task") == before
    assert queue.get_task_by_id("task").original_instruction == before["description"]


def test_parallel_metadata_updates_do_not_lose_fields(queue):
    queue.submit(_payload())

    def update(index):
        TaskQueueManager(queue.anima_dir).update_meta("task", {str(index): index})

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(update, range(30)))
    meta = queue.get_task_by_id("task").meta
    assert all(meta[str(index)] == index for index in range(30))


@pytest.mark.parametrize("outcome", ["done", "cancelled"])
def test_dependency_on_delegation_alias_resolves_canonical_owner(queue, outcome):
    queue.submit(_payload())
    store = queue.store
    store.alias("manager", "tracker", "worker", "task")
    manager = TaskQueueManager(queue.anima_dir.parent / "manager")
    manager.submit(_payload("followup", depends_on=["tracker"]))
    claimed = store.claim("worker", "task", process_identity())
    queue.update_status("task", outcome)
    assert store.claim("manager", "followup", process_identity()) is None
    store.finish(claimed["_attempt_token"], status=outcome, stop_kind=outcome)
    followup = store.claim("manager", "followup", process_identity())
    if outcome == "done":
        assert followup is not None
    else:
        assert followup is None
        assert store.wakeups("manager")[0]["reason"] == "dependency_cancelled"


def test_quiesce_survives_new_store(queue):
    queue.submit(_payload())
    queue.store.pause_claims("worker")
    other = TaskStore(task_database_path(queue.anima_dir))
    assert other.claim("worker", "task", process_identity()) is None
    other.pause_claims("worker", paused=False)
    assert other.claim("worker", "task", process_identity())


@pytest.mark.asyncio
async def test_incomplete_attempt_creates_retryable_wakeup_not_retry(queue):
    from core.supervisor.pending_executor import PendingTaskExecutor

    queue.submit(_payload(reply_to="manager"))
    claim = queue.store.claim("worker", "task", process_identity())
    anima = SimpleNamespace(messenger=SimpleNamespace(send=Mock(side_effect=OSError("offline"))))
    executor = PendingTaskExecutor(anima, "worker", queue.anima_dir, asyncio.Event())
    executor.execute_pending_task = AsyncMock()
    executor._active_task_ids.add("task")
    await executor._execute_canonical_task(claim)
    assert queue.get_task_by_id("task").status == "pending"
    assert queue.store.pending("worker") == []
    executor._deliver_task_wakeups(queue.store)
    assert len(queue.store.wakeups("worker")) == 1
    anima.messenger.send.side_effect = None
    executor._deliver_task_wakeups(queue.store)
    assert queue.store.wakeups("worker") == []
    assert len(anima.messenger.send.call_args_list) == 3


def test_dead_attempt_is_not_automatically_reexecuted(queue):
    from core.supervisor.pending_executor import PendingTaskExecutor

    queue.submit(_payload())
    queue.store.claim("worker", "task", {"pid": 999999999, "process_start_time": 0})
    executor = PendingTaskExecutor(SimpleNamespace(), "worker", queue.anima_dir, asyncio.Event())
    executor._recover_task_attempts(queue.store)
    assert queue.get_task_by_id("task").status == "pending"
    assert queue.store.active_attempts("worker") == []
    assert queue.store.pending("worker") == []
    assert len(queue.store.wakeups("worker")) == 1


def test_stale_attempt_cannot_resume_itself(queue):
    payload = _payload()
    queue.submit(payload)
    claim = queue.store.claim("worker", "task", process_identity())
    queue.store.finish(claim["_attempt_token"], status="pending", stop_kind="interrupted")
    with (
        attempt_scope({"anima": "worker", "task_id": "task", "token": claim["_attempt_token"]}),
        pytest.raises(ValueError, match="Stale"),
    ):
        queue.submit(payload, resume=True)
    assert queue.store.pending("worker") == []


def test_dependency_waits_for_finished_attempt_and_cancel_notifies(queue):
    queue.submit(_payload("a", parallel=True))
    queue.submit(_payload("b", depends_on=["a"], parallel=True))
    first = queue.store.claim("worker", "a", process_identity(), max_active=2)
    queue.update_status("a", "done")
    assert queue.store.claim("worker", "b", process_identity(), max_active=2) is None
    queue.update_status("a", "cancelled")
    queue.store.finish(first["_attempt_token"], status="done", stop_kind="cancelled")
    assert queue.store.claim("worker", "b", process_identity(), max_active=2) is None
    assert queue.store.pending("worker") == []
    assert queue.store.wakeups("worker")[0]["reason"] == "dependency_cancelled"


@pytest.mark.asyncio
async def test_nonisolated_quiet_model_can_be_cancelled_without_heartbeat(queue, monkeypatch):
    from core.supervisor.pending_executor import PendingTaskExecutor

    monkeypatch.setattr("core.supervisor.pending_executor._CANCEL_POLL_SECONDS", 0.01)
    queue.submit(_payload())
    claim = queue.store.claim("worker", "task", process_identity())
    entered = asyncio.Event()

    async def quiet_provider(task_desc):
        entered.set()
        await asyncio.Event().wait()

    executor = PendingTaskExecutor(SimpleNamespace(), "worker", queue.anima_dir, asyncio.Event())
    executor._task_isolated = False
    executor.execute_pending_task = quiet_provider
    running = asyncio.create_task(executor._execute_canonical_task(claim))
    await asyncio.wait_for(entered.wait(), timeout=1)
    queue.update_status("task", "cancelled")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=2)
    assert queue.store.active_attempts("worker") == []
    assert queue.get_task_by_id("task").status == "cancelled"


def test_completion_notification_failure_does_not_undo_completion(queue):
    from core.supervisor.pending_executor import PendingTaskExecutor

    queue.submit(_payload(reply_to="manager"))
    claim = queue.store.claim("worker", "task", process_identity())
    queue.store.finish(claim["_attempt_token"], status="done", stop_kind="completed")
    anima = SimpleNamespace(messenger=SimpleNamespace(send=Mock(side_effect=OSError("offline"))))
    executor = PendingTaskExecutor(anima, "worker", queue.anima_dir, asyncio.Event())
    executor._deliver_task_wakeups(queue.store)
    assert queue.get_task_by_id("task").status == "done"
    assert queue.store.wakeups("worker")[0]["reason"] == "completion"
    anima.messenger.send.side_effect = None
    executor._deliver_task_wakeups(queue.store)
    assert queue.store.wakeups("worker") == []


def test_durable_inbox_delivery_deduplicates_after_archive(queue):
    from core.messenger import Messenger

    shared = queue.anima_dir.parent.parent / "shared"
    messenger = Messenger(shared, "worker")
    first = messenger.send("worker", "needs attention", msg_type="system_alert", delivery_id="test-outbox-token")
    messenger.archive_all()
    second = messenger.send("worker", "needs attention", msg_type="system_alert", delivery_id="test-outbox-token")
    assert first.id == second.id
    assert not list((shared / "inbox" / "worker").glob("*.json"))


def test_presentation_updates_run_only_after_task_transaction_commits(queue, monkeypatch):
    from core.taskboard.store import TaskBoardStore

    monkeypatch.setattr("core.taskboard.store.get_taskboard_db_path", lambda: queue.store.db_path)
    board = TaskBoardStore()
    queue.submit(_payload())
    board.upsert_metadata(anima_name="worker", task_id="task", actor="worker", visibility="active")
    with pytest.raises(ValueError), queue.store.transaction():
        queue.update_status("task", "done")
        raise ValueError("rollback")
    assert board.get_metadata("worker", "task").visibility.value == "active"
    assert queue.get_task_by_id("task").status == "pending"
    with queue.store.transaction():
        queue.update_status("task", "done")
    assert board.get_metadata("worker", "task").visibility.value == "archived"


def test_reader_keeps_one_consistent_snapshot_and_rejects_writes(queue):
    queue.submit(_payload())
    with queue.store.reader() as outer:
        with queue.store.reader() as inner:
            assert inner is outer
        with pytest.raises(RuntimeError, match="read-only"):
            queue.store.pause_claims("worker")
