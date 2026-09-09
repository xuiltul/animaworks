from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.anima import BackgroundWorkerSlot, DigitalAnima
from core.memory.task_queue import TaskQueueManager
from core.platform.processing_lease import processing_lease_path, write_processing_lease
from core.supervisor.pending_executor import PendingTaskExecutor
from core.taskboard.tasks import process_identity
from core.tasks_dispatch import publish_tasks


def _slot(slot_id: int) -> BackgroundWorkerSlot:
    return BackgroundWorkerSlot(
        slot_id=slot_id,
        agent=MagicMock(name=f"worker_agent_{slot_id}"),
        session_lock=asyncio.Lock(),
        interrupt_event=asyncio.Event(),
    )


def _bare_pool_anima(pool_size: int) -> DigitalAnima:
    anima = DigitalAnima.__new__(DigitalAnima)
    anima.name = "pool-test"
    anima._background_worker_pool_size = pool_size
    anima._background_worker_slots = [_slot(slot_id) for slot_id in range(pool_size)]
    anima._background_worker_queue = asyncio.Queue()
    for slot in anima._background_worker_slots:
        anima._background_worker_queue.put_nowait(slot)
    anima._active_background_workers = {}
    anima._background_worker_gate_lock = asyncio.Lock()
    anima._background_worker_gate_count = 0
    anima._background_lock = asyncio.Lock()
    anima._mark_busy_start = MagicMock()
    anima._mark_busy_progress = MagicMock()
    anima._clear_busy_status_sidecar_if_idle = MagicMock()
    anima._on_lock_released = None
    return anima


async def test_worker_lease_waits_at_capacity_and_release_unblocks() -> None:
    anima = _bare_pool_anima(pool_size=2)

    first = await anima._acquire_background_worker("task-one")
    second = await anima._acquire_background_worker("task-two")
    waiting = asyncio.create_task(anima._acquire_background_worker("task-three"))
    await asyncio.sleep(0)

    assert not waiting.done()
    assert anima._background_lock.locked()
    assert set(anima._active_background_workers.values()) == {"task-one", "task-two"}

    await anima._release_background_worker(first)
    third = await asyncio.wait_for(waiting, timeout=1)

    assert third is first
    assert anima._background_lock.locked()
    assert set(anima._active_background_workers.values()) == {"task-two", "task-three"}

    await anima._release_background_worker(second)
    assert anima._background_lock.locked()
    await anima._release_background_worker(third)

    assert not anima._background_lock.locked()
    assert anima._background_worker_queue.qsize() == 2
    anima._mark_busy_start.assert_called_once_with()


def test_pool_size_one_reuses_legacy_background_agent_and_session() -> None:
    anima = DigitalAnima.__new__(DigitalAnima)
    background_agent = MagicMock(name="legacy_background_agent")
    background_session_lock = asyncio.Lock()
    anima._background_worker_pool_size = 1
    anima._lane_agents = {"background": background_agent}
    anima._agent_session_locks = {"background": background_session_lock}
    anima._interrupt_events = {}
    anima._background_worker_slots = []
    anima._background_worker_queue = asyncio.Queue()

    anima._initialize_background_worker_pool()

    assert len(anima._background_worker_slots) == 1
    slot = anima._background_worker_slots[0]
    assert slot.slot_id == 0
    assert slot.agent is background_agent
    assert slot.session_lock is background_session_lock
    assert slot.interrupt_event is anima._get_interrupt_event("_background")
    assert anima._background_worker_queue.get_nowait() is slot


class _PoolStub:
    def __init__(self, pool_size: int) -> None:
        self._background_worker_pool_size = pool_size
        self.messenger = MagicMock()
        self._available: asyncio.Queue[BackgroundWorkerSlot] = asyncio.Queue()
        for slot_id in range(pool_size):
            self._available.put_nowait(_slot(slot_id))

    async def _acquire_background_worker(self, task_id: str) -> BackgroundWorkerSlot:
        return await self._available.get()

    async def _release_background_worker(self, slot: BackgroundWorkerSlot) -> None:
        self._available.put_nowait(slot)


def _executor(tmp_path: Path, *, pool_size: int = 2) -> PendingTaskExecutor:
    anima_dir = tmp_path / "pool-test"
    anima_dir.mkdir()
    return PendingTaskExecutor(
        anima=_PoolStub(pool_size),
        anima_name="pool-test",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _publish(executor: PendingTaskExecutor, *payloads: dict) -> None:
    publish_tasks(
        executor._anima_dir,
        [
            {"task_type": "llm", "title": payload["task_id"], "description": payload["task_id"], **payload}
            for payload in payloads
        ],
    )


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_duplicate_task_id_preserves_active_input_and_one_attempt(tmp_path: Path) -> None:
    executor = _executor(tmp_path, pool_size=2)
    queue = TaskQueueManager(executor._anima_dir)

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls: list[str] = []

    async def fake_execute(task_desc, *, worker_slot=None):
        calls.append(task_desc["task_id"])
        first_started.set()
        await release_first.wait()

    executor.execute_pending_task = fake_execute  # type: ignore[method-assign]
    first = {"task_type": "llm", "task_id": "duplicate-id", "description": "first"}
    _publish(executor, first)

    watcher = asyncio.create_task(executor.watcher_loop())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    duplicate = {"task_type": "llm", "task_id": "duplicate-id", "description": "second"}
    _publish(executor, duplicate)
    assert queue.store.get_input("pool-test", "duplicate-id")["description"] == "first"
    executor.wake()

    await asyncio.sleep(0)
    assert calls == ["duplicate-id"]
    assert len(queue.store.active_attempts("pool-test")) == 1
    assert queue.get_task_by_id("duplicate-id").status == "in_progress"

    release_first.set()
    await _wait_until(lambda: not executor._active_dispatch_tasks)
    executor._shutdown_event.set()
    executor.wake()
    await asyncio.wait_for(watcher, timeout=1)
    assert not executor._active_task_ids


async def test_duplicate_batch_task_id_is_rejected_before_atomic_publication(tmp_path: Path) -> None:
    executor = _executor(tmp_path, pool_size=2)
    dispatched: list[dict[str, object]] = []

    async def fake_dispatch(task_desc):
        dispatched.append(task_desc)
        executor._shutdown_event.set()
        executor.wake()

    executor.execute_pending_task = fake_dispatch  # type: ignore[method-assign]
    descriptor = {
        "task_type": "llm",
        "task_id": "duplicate-batch-id",
        "batch_id": "batch-one",
        "description": "batch task",
    }
    with pytest.raises(ValueError, match="Duplicate task_id"):
        _publish(executor, descriptor, descriptor)
    assert TaskQueueManager(executor._anima_dir).store.read("pool-test") == {}
    _publish(executor, descriptor)
    _publish(executor, descriptor)

    await asyncio.wait_for(executor.watcher_loop(), timeout=1)

    assert [task["task_id"] for task in dispatched] == ["duplicate-batch-id"]
    assert not TaskQueueManager(executor._anima_dir).store.active_attempts("pool-test")
    assert not executor._active_task_ids


async def test_duplicate_task_id_is_rejected_while_batch_is_running(tmp_path: Path) -> None:
    executor = _executor(tmp_path, pool_size=2)
    batch_started = asyncio.Event()
    release_batch = asyncio.Event()
    dispatched: list[list[dict[str, object]]] = []

    async def fake_dispatch(task_desc):
        dispatched.append([task_desc])
        batch_started.set()
        await release_batch.wait()

    executor.execute_pending_task = fake_dispatch  # type: ignore[method-assign]
    first = {
        "task_type": "llm",
        "task_id": "running-batch-id",
        "batch_id": "batch-running",
        "description": "first",
    }
    _publish(executor, first)

    watcher = asyncio.create_task(executor.watcher_loop())
    await asyncio.wait_for(batch_started.wait(), timeout=1)
    duplicate = {**first, "description": "duplicate"}
    _publish(executor, duplicate)
    assert (
        TaskQueueManager(executor._anima_dir).store.get_input("pool-test", "running-batch-id")["description"] == "first"
    )
    executor.wake()

    await asyncio.sleep(0)
    assert len(dispatched) == 1
    assert [task["task_id"] for task in dispatched[0]] == ["running-batch-id"]

    release_batch.set()
    await _wait_until(lambda: not executor._active_dispatch_tasks)
    executor._shutdown_event.set()
    executor.wake()
    await asyncio.wait_for(watcher, timeout=1)
    assert not executor._active_task_ids


async def test_distinct_task_ids_still_run_up_to_pool_size(tmp_path: Path) -> None:
    executor = _executor(tmp_path, pool_size=2)
    all_started = asyncio.Event()
    release = asyncio.Event()
    started: set[str] = set()

    async def fake_execute(task_desc, *, worker_slot=None):
        started.add(task_desc["task_id"])
        if len(started) == 2:
            all_started.set()
        await release.wait()

    executor.execute_pending_task = fake_execute  # type: ignore[method-assign]
    for task_id in ("distinct-one", "distinct-two", "distinct-three"):
        descriptor = {"task_type": "llm", "task_id": task_id, "description": task_id}
        _publish(executor, descriptor)

    watcher = asyncio.create_task(executor.watcher_loop())
    await asyncio.wait_for(all_started.wait(), timeout=1)
    assert started == {"distinct-one", "distinct-two"}
    assert executor._active_task_ids == started
    assert len(TaskQueueManager(executor._anima_dir).store.active_attempts("pool-test")) == 2

    release.set()
    await _wait_until(lambda: len(started) == 3)
    await _wait_until(lambda: not executor._active_dispatch_tasks)
    executor._shutdown_event.set()
    executor.wake()
    await asyncio.wait_for(watcher, timeout=1)
    assert not executor._active_task_ids


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_active_task_id_removed_after_claimed_dispatch_ends(tmp_path: Path, outcome: str) -> None:
    executor = _executor(tmp_path)
    _publish(executor, {"task_id": outcome})
    store = TaskQueueManager(executor._anima_dir).store
    claimed = store.claim("pool-test", outcome, process_identity())
    executor._active_task_ids.add(outcome)

    async def fake_execute(task_desc, *, worker_slot=None):
        if outcome == "failure":
            raise RuntimeError("boom")
        if outcome == "cancel":
            raise asyncio.CancelledError

    executor.execute_pending_task = fake_execute  # type: ignore[method-assign]
    run = executor._execute_canonical_task(claimed)
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run
    else:
        await run

    assert outcome not in executor._active_task_ids
    assert store.active_attempts("pool-test") == []
    assert store.pending("pool-test") == []
    assert store.read("pool-test")[outcome].status == "pending"
    assert len(store.wakeups("pool-test")) == 1


async def test_cancelled_claim_returns_pending_and_publishes_durable_wakeup(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    task_id = "cancelled-layer2"
    queue = TaskQueueManager(executor._anima_dir)
    _publish(executor, {"task_id": task_id, "description": "long running task"})
    claimed = queue.store.claim("pool-test", task_id, process_identity())

    async def cancel_execute(task_desc, *, worker_slot=None):
        raise asyncio.CancelledError

    executor.execute_pending_task = cancel_execute  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await executor._execute_canonical_task(claimed)

    entry = queue.get_task_by_id(task_id)
    assert entry.status == "pending"
    assert entry.meta["last_run_stop_kind"] == "interrupted"
    assert queue.store.wakeups("pool-test")[0]["reason"] == "interrupted"
    executor._deliver_task_wakeups(queue.store)
    assert executor._anima.messenger.send.call_args.kwargs["to"] == "pool-test"
    assert queue.store.wakeups("pool-test") == []


async def test_watcher_shutdown_waits_for_active_dispatch_to_finish(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    release = asyncio.Event()
    finished = asyncio.Event()

    async def active_dispatch() -> None:
        await release.wait()
        finished.set()

    dispatch = asyncio.create_task(active_dispatch())
    executor._track_dispatch_task(dispatch)
    executor._shutdown_event.set()
    release.set()
    config = SimpleNamespace(
        background_task=SimpleNamespace(shutdown_drain_seconds=1.0),
    )

    with patch("core.config.models.load_config", return_value=config):
        await executor.watcher_loop()

    assert finished.is_set()
    assert not dispatch.cancelled()
    assert not executor._active_dispatch_tasks


async def test_watcher_shutdown_cancels_dispatch_after_drain_timeout(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    cancelled = asyncio.Event()

    async def active_dispatch() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    dispatch = asyncio.create_task(active_dispatch())
    executor._track_dispatch_task(dispatch)
    executor._shutdown_event.set()
    config = SimpleNamespace(
        background_task=SimpleNamespace(shutdown_drain_seconds=0.0),
    )

    with patch("core.config.models.load_config", return_value=config):
        await executor.watcher_loop()

    assert cancelled.is_set()
    assert dispatch.cancelled()
    assert not executor._active_dispatch_tasks


async def test_command_claim_stays_active_until_background_task_finishes(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    processing_path = tmp_path / "processing" / "command.json"
    processing_path.parent.mkdir()
    processing_path.write_text('{"task_id":"command"}', encoding="utf-8")
    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    write_processing_lease(processing_path, anima="pool-test", task_id="command")
    executor._active_task_ids.add("command")
    release = asyncio.Event()

    async def wait_for_release() -> None:
        await release.wait()

    background_task = asyncio.create_task(wait_for_release())
    executor._track_command_claim(
        background_task,
        task_id="command",
        processing_path=processing_path,
    )
    assert "command" in executor._active_task_ids
    assert processing_path.exists()

    release.set()
    await background_task
    await asyncio.sleep(0)

    assert "command" not in executor._active_task_ids
    assert not processing_path.exists()
    assert not processing_lease_path(processing_path).exists()


async def _measure_concurrency(
    executor: PendingTaskExecutor,
    task_descriptions: list[dict[str, str]],
) -> int:
    active = 0
    maximum_active = 0

    async def fake_run_llm_task(*args, **kwargs) -> str:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        return "ok"

    executor._run_llm_task = fake_run_llm_task  # type: ignore[method-assign]
    await asyncio.gather(*(executor._run_task_in_worker(task) for task in task_descriptions))
    return maximum_active


async def test_same_working_directory_can_overlap(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    maximum_active = await _measure_concurrency(
        executor,
        [
            {"task_id": "same-one", "working_directory": str(workspace)},
            {"task_id": "same-two", "working_directory": str(workspace)},
        ],
    )

    assert maximum_active == 2


async def test_tasks_touching_the_same_pr_are_not_serialised(tmp_path: Path) -> None:
    """There is no exclusion any more: two tasks about the same PR run together."""
    executor = _executor(tmp_path)

    maximum_active = await _measure_concurrency(
        executor,
        [
            {"task_id": "pr-one", "summary": "review pr-3999", "exclusive_key": "pr-3999"},
            {"task_id": "pr-two", "summary": "merge pr-3999", "exclusive_key": "pr-3999"},
        ],
    )

    assert maximum_active == 2


async def test_same_pr_canonical_tasks_are_claimed_in_parallel(tmp_path: Path) -> None:
    """Two published tasks naming the same PR are both claimed by the pool."""
    executor = _executor(tmp_path, pool_size=2)
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: set[str] = set()

    async def fake_execute(task_desc, *, worker_slot=None):
        started.add(task_desc["task_id"])
        if len(started) == 2:
            both_started.set()
        await release.wait()

    executor.execute_pending_task = fake_execute  # type: ignore[method-assign]
    for task_id, summary in (("pr-review", "review pr-42"), ("pr-merge", "merge pr-42")):
        descriptor = {
            "task_type": "llm",
            "task_id": task_id,
            "description": summary,
            "summary": summary,
            "exclusive_key": "pr-42",
        }
        _publish(executor, descriptor)

    watcher = asyncio.create_task(executor.watcher_loop())
    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert started == {"pr-review", "pr-merge"}

    release.set()
    await _wait_until(lambda: not executor._active_dispatch_tasks)
    executor._shutdown_event.set()
    executor.wake()
    await asyncio.wait_for(watcher, timeout=1)
    assert not executor._active_task_ids


def test_active_worker_is_busy_and_drives_primary_status() -> None:
    anima = DigitalAnima.__new__(DigitalAnima)
    anima._conversation_locks = {}
    anima._background_lock = asyncio.Lock()
    anima._inbox_lock = asyncio.Lock()
    anima._active_background_workers = {1: "worker-task"}
    anima._status_slots = {"inbox": "idle", "background": "idle"}
    anima._task_slots = {"inbox": "", "background": ""}

    assert anima._has_active_busy_lock() is True
    assert anima.primary_status == "task_exec"
    assert anima.primary_task == "worker-task"

    anima._active_background_workers.clear()

    assert anima._has_active_busy_lock() is False
    assert anima.primary_status == "idle"
    assert anima.primary_task == ""


async def test_second_batch_dispatches_while_first_batch_is_still_running(tmp_path: Path) -> None:
    """Batches must not be serialized behind each other (2026-09-03).

    A heartbeat submits one batch per run; if batch N+1 waited for every task
    of batch N, a single CI-polling task would stall all later submissions.
    """
    executor = _executor(tmp_path, pool_size=2)
    started: list[str] = []
    release = asyncio.Event()

    async def fake_dispatch(task_desc):
        started.append(task_desc["batch_id"])
        await release.wait()

    executor.execute_pending_task = fake_dispatch  # type: ignore[method-assign]
    _publish(executor, {"task_id": "t-a", "batch_id": "batch-a", "description": "a"})
    watcher = asyncio.create_task(executor.watcher_loop())
    await _wait_until(lambda: started == ["batch-a"])

    _publish(executor, {"task_id": "t-b", "batch_id": "batch-b", "description": "b"})
    executor.wake()
    await _wait_until(lambda: started == ["batch-a", "batch-b"])

    release.set()
    await _wait_until(lambda: not executor._active_dispatch_tasks)
    executor._shutdown_event.set()
    executor.wake()
    await asyncio.wait_for(watcher, timeout=1)
