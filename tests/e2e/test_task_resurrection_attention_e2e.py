from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from core._anima_inbox import _rescue_regenerate_pending
from core.memory.task_queue import TaskQueueManager
from core.taskboard.models import AttentionVisibility
from core.taskboard.store import TaskBoardStore

pytestmark = pytest.mark.e2e


def test_taskboard_blocks_archived_delegation_rescue_resurrection(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)

    queue = TaskQueueManager(anima_dir)
    entry = queue.add_task(
        source="human",
        original_instruction="do not resurrect this task",
        assignee="sakura",
        summary="do not resurrect this task",
        task_id="archived1234",
        meta={"task_desc": {"title": "do not resurrect this task"}},
    )
    queue.update_meta(entry.task_id, {"last_run_stop_kind": "crash"})

    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        visibility=AttentionVisibility.ARCHIVED,
    )

    # Simulate a lost descriptor; rescue must respect the archived card.
    (anima_dir / "state" / "pending" / f"{entry.task_id}.json").unlink(missing_ok=True)
    _rescue_regenerate_pending(
        anima_dir,
        entry.task_id,
        SimpleNamespace(content="delegated work", from_person="manager"),
    )

    assert not (anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()
    assert not (anima_dir / "state" / "pending" / "deferred" / f"{entry.task_id}.json").exists()
    assert TaskQueueManager(anima_dir).get_task_by_id(entry.task_id).status == "cancelled"


def test_delegation_rescue_keeps_snoozed_task_in_executable_pending(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)

    queue = TaskQueueManager(anima_dir)
    entry = queue.add_task(
        source="human",
        original_instruction="wake later",
        assignee="sakura",
        summary="wake later",
        task_id="snoozed1234",
    )
    TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3").upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        visibility=AttentionVisibility.SNOOZED,
        snoozed_until=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )

    _rescue_regenerate_pending(
        anima_dir,
        entry.task_id,
        SimpleNamespace(content="delegated work", from_person="manager"),
    )

    assert (anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()
    assert not (anima_dir / "state" / "pending" / "deferred" / f"{entry.task_id}.json").exists()
