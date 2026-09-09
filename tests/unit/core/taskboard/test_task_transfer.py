from __future__ import annotations

from pathlib import Path

import pytest

from core.lifecycle.anima_merge.task_refs import TaskReferenceRewriter
from core.memory.task_queue import TaskQueueManager


def _queues(tmp_path: Path):
    root = tmp_path / "runtime"
    source = root / "animas" / "source"
    target = root / "animas" / "target"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    return root, TaskQueueManager(source), TaskQueueManager(target)


def test_transfer_preserves_terminal_history_inputs_aliases_and_is_idempotent(tmp_path: Path):
    root, source, target = _queues(tmp_path)
    source.submit({"task_id": "same", "title": "source work", "description": "original input"})
    target.submit({"task_id": "same", "title": "target work", "description": "do not replace"})
    attempt = source.store.claim("source", "same", {"pid": 1, "process_start_time": 1})
    assert attempt
    token = attempt["_attempt_token"]
    result_path = source.anima_dir / "state" / "task_results" / "same" / f"{token}.md"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("actual result")
    assert source.store.finish(
        token, status="done", stop_kind="completed", result_ref=f"state/task_results/same/{token}.md"
    )
    with source.store.transaction() as database:
        database.execute("INSERT INTO task_aliases VALUES('manager','tracking','source','same')")
    rewriter = TaskReferenceRewriter(root, "source", "target")
    plan = rewriter.plan()
    assert plan.mapping == {"same": "same__from_source"}
    rewriter.apply(plan)
    rewriter.apply(plan)
    assert source.store.get_input("target", "same__from_source")["description"] == "original input"
    assert target.store.get_input("target", "same")["description"] == "do not replace"
    assert source.store.read("manager")["tracking"].status == "done"
    with source.store.reader() as database:
        history = database.execute("SELECT anima,task_id,token,result_ref FROM task_attempts").fetchall()
    assert [tuple(row) for row in history] == [
        ("target", "same__from_source", token, f"state/task_results/same__from_source/{token}.md")
    ]
    assert (target.anima_dir / "state/task_results/same__from_source" / f"{token}.md").read_text() == "actual result"
    assert not (target.anima_dir / "state" / "pending").exists()


def test_transfer_rejects_active_attempt_without_partial_moves(tmp_path: Path):
    root, source, target = _queues(tmp_path)
    for task_id in ("first", "active"):
        source.submit({"task_id": task_id, "title": task_id, "description": task_id})
    assert source.store.claim("source", "active", {"pid": 1, "process_start_time": 1})
    rewriter = TaskReferenceRewriter(root, "source", "target")
    with pytest.raises(ValueError, match="active attempt"):
        rewriter.apply(rewriter.plan())
    assert set(source.store.read("source")) == {"first", "active"}
    assert not target.store.read("target")


def test_canonical_merge_plan_is_read_only_without_legacy_projection(tmp_path: Path):
    root, source, target = _queues(tmp_path)
    source.submit({"task_id": "only-db", "description": "persisted"})
    before = source.store.db_path.read_bytes()
    plan = TaskReferenceRewriter(root, "source", "target").plan()
    assert plan.mapping == {"only-db": "only-db"}
    assert source.store.db_path.read_bytes() == before
    assert not source.queue_path.exists()


def test_transfer_preserves_unfinished_wakeup_without_auto_resubmission(tmp_path: Path):
    root, source, target = _queues(tmp_path)
    source.submit({"task_id": "paused", "description": "needs review"})
    attempt = source.store.claim("source", "paused", {"pid": 1, "process_start_time": 1})
    token = attempt["_attempt_token"]
    assert source.store.finish(token, status="pending", stop_kind="max_turns")
    rewriter = TaskReferenceRewriter(root, "source", "target")
    rewriter.apply(rewriter.plan())
    wakeups = target.store.wakeups("target")
    assert [(item["task_id"], item["attempt_token"], item["reason"]) for item in wakeups] == [
        ("paused", token, "max_turns")
    ]
    assert not target.store.pending("target")
    assert not source.store.wakeups("source")
