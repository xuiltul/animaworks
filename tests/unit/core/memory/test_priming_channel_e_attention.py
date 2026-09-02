from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine
from core.memory.task_queue import TaskQueueManager
from core.taskboard.store import TaskBoardStore
from core.time_utils import now_local


@pytest.fixture
def attention_env(tmp_path: Path) -> tuple[Path, TaskBoardStore]:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    for subdir in ["episodes", "knowledge", "skills", "state"]:
        (anima_dir / subdir).mkdir(parents=True, exist_ok=True)
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    return anima_dir, store


def _append_task_entry(
    anima_dir: Path,
    *,
    task_id: str,
    summary: str,
    updated_at: str,
    deadline: str | None = None,
    source: str = "human",
) -> None:
    queue_path = anima_dir / "state" / "task_queue.jsonl"
    entry = {
        "task_id": task_id,
        "ts": updated_at,
        "source": source,
        "original_instruction": summary,
        "assignee": anima_dir.name,
        "status": "pending",
        "summary": summary,
        "deadline": deadline,
        "relay_chain": [],
        "updated_at": updated_at,
        "meta": {},
    }
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@pytest.mark.asyncio
async def test_channel_e_filters_suppressed_tasks(attention_env: tuple[Path, TaskBoardStore]) -> None:
    anima_dir, store = attention_env
    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="human",
        original_instruction="visible work",
        assignee="sakura",
        summary="visible work",
        task_id="visible1234",
    )
    queue.add_task(
        source="human",
        original_instruction="archived work",
        assignee="sakura",
        summary="archived work",
        task_id="hidden1234",
    )
    store.upsert_metadata(anima_name="sakura", task_id="hidden1234", visibility="archived")

    result = await PrimingEngine(anima_dir)._channel_e_pending_tasks()

    assert "visible work" in result
    assert "archived work" not in result


@pytest.mark.asyncio
async def test_channel_e_filters_snoozed_tasks(attention_env: tuple[Path, TaskBoardStore]) -> None:
    anima_dir, store = attention_env
    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="human",
        original_instruction="snoozed work",
        assignee="sakura",
        summary="snoozed work",
        task_id="snoozed1234",
    )
    store.upsert_metadata(
        anima_name="sakura",
        task_id="snoozed1234",
        visibility="snoozed",
        snoozed_until=(now_local() + timedelta(hours=2)).isoformat(),
    )

    result = await PrimingEngine(anima_dir)._channel_e_pending_tasks()

    assert "snoozed work" not in result


@pytest.mark.asyncio
async def test_channel_e_filters_task_results(attention_env: tuple[Path, TaskBoardStore]) -> None:
    anima_dir, store = attention_env
    results_dir = anima_dir / "state" / "task_results"
    results_dir.mkdir(parents=True)
    (results_dir / "hidden1234.md").write_text("hidden result", encoding="utf-8")
    (results_dir / "recent1234.md").write_text("recent result", encoding="utf-8")
    old_file = results_dir / "orphan1234.md"
    old_file.write_text("old orphan result", encoding="utf-8")
    old = (now_local() - timedelta(hours=25)).timestamp()
    os.utime(old_file, (old, old))
    store.upsert_metadata(anima_name="sakura", task_id="hidden1234", visibility="tombstoned")

    result = await PrimingEngine(anima_dir)._channel_e_pending_tasks()

    assert "recent result" in result
    assert "hidden result" not in result
    assert "old orphan result" not in result


# ── "failed"-specific board behavior was retired along with the "failed"
# status itself (see A1 task-model teardown plan). What used to be
# test_channel_e_surfaces_recent_failed_tasks and
# test_channel_e_hides_explicitly_archived_failed_tasks tested a review
# workflow that no longer exists at the queue level.


@pytest.mark.asyncio
async def test_channel_e_preserves_legacy_prompt_signals(attention_env: tuple[Path, TaskBoardStore]) -> None:
    """STALE and auto-taskexec markers still surface (OVERDUE/deadline were retired)."""
    anima_dir, _store = attention_env
    now = now_local()
    _append_task_entry(
        anima_dir,
        task_id="stale1234",
        summary="stale board work",
        updated_at=(now - timedelta(minutes=45)).isoformat(),
    )
    TaskQueueManager(anima_dir).add_task(
        source="anima",
        original_instruction="auto taskexec work",
        assignee="sakura",
        summary="auto taskexec work",
        task_id="auto1234",
        meta={"executor": "taskexec"},
        status="in_progress",
    )

    result = await PrimingEngine(anima_dir)._channel_e_pending_tasks()

    assert "stale board work" in result
    assert "STALE" in result
    assert "OVERDUE" not in result
    assert "(auto: TaskExec)" in result


@pytest.mark.asyncio
async def test_channel_e_preserves_delegated_status_section(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    subordinate_dir = data_dir / "animas" / "hinata"
    for directory in [anima_dir, subordinate_dir]:
        for subdir in ["episodes", "knowledge", "skills", "state"]:
            (directory / subdir).mkdir(parents=True, exist_ok=True)
    TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")

    subordinate_task = TaskQueueManager(subordinate_dir).add_task(
        source="human",
        original_instruction="subordinate work",
        assignee="hinata",
        summary="subordinate work",
        task_id="child1234",
    )
    TaskQueueManager(anima_dir).add_delegated_task(
        original_instruction="delegated board work",
        assignee="sakura",
        summary="delegated board work",
        meta={"delegated_to": "hinata", "delegated_task_id": subordinate_task.task_id},
    )

    result = await PrimingEngine(anima_dir)._channel_e_pending_tasks()

    assert "delegated board work" in result
    assert "hinata: pending" in result
    assert "⏳" in result


@pytest.mark.asyncio
async def test_channel_e_falls_back_when_taskboard_db_is_corrupt(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    for subdir in ["episodes", "knowledge", "skills", "state"]:
        (anima_dir / subdir).mkdir(parents=True, exist_ok=True)
    shared_dir = data_dir / "shared"
    shared_dir.mkdir()
    (shared_dir / "taskboard.sqlite3").write_text("not sqlite", encoding="utf-8")

    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="human",
        original_instruction="fallback visible work",
        assignee="sakura",
        summary="fallback visible work",
        task_id="visible1234",
    )

    result = await PrimingEngine(anima_dir)._channel_e_pending_tasks()

    assert "fallback visible work" in result
