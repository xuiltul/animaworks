"""Canonical task scheduling replaces heuristic missing-descriptor recovery."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import PendingTaskExecutor


@pytest.mark.asyncio
async def test_watcher_executes_only_published_input_and_ignores_stray_descriptor(tmp_path):
    directory = tmp_path / "animas" / "worker"
    directory.mkdir(parents=True)
    queue = TaskQueueManager(directory)
    queue.submit({"task_id": "published", "task_type": "llm", "title": "work", "description": "exact input"})
    legacy = directory / "state" / "pending" / "not-published.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"task_id": "not-published", "task_type": "llm", "description": "must not execute"}))
    shutdown = asyncio.Event()
    anima = SimpleNamespace(messenger=SimpleNamespace(send=Mock()))
    executor = PendingTaskExecutor(anima, "worker", directory, shutdown)
    seen = []

    async def run(payload):
        seen.append(payload)
        queue.update_status(payload["task_id"], "done")
        shutdown.set()
        executor.wake()

    executor.execute_pending_task = run
    await asyncio.wait_for(executor.watcher_loop(), timeout=3)
    assert [item["task_id"] for item in seen] == ["published"]
    assert seen[0]["description"] == "exact input"
    assert seen[0]["_attempt_number"] == 1
    assert queue.get_task_by_id("published").status == "done"
    assert queue.store.active_attempts("worker") == []
    assert legacy.exists()  # retained evidence is neither runnable nor destroyed


def test_old_backlog_is_not_automatically_cancelled_or_republished(tmp_path):
    directory = tmp_path / "animas" / "worker"
    directory.mkdir(parents=True)
    queue = TaskQueueManager(directory)
    queue.add_task(
        source="human", original_instruction="important", assignee="worker", summary="backlog", task_id="old"
    )
    queue.update_meta("old", {"last_run_ended_at": "2000-01-01T00:00:00Z"})
    executor = PendingTaskExecutor(SimpleNamespace(), "worker", directory, asyncio.Event())
    executor._recover_task_attempts(queue.store)
    assert queue.get_task_by_id("old").status == "pending"
    assert queue.store.pending("worker") == []
    assert queue.store.wakeups("worker") == []
