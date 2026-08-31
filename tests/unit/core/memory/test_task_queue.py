from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.memory.task_queue — TaskQueueManager and task lifecycle."""

from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.task_queue import (
    _ACTIVE_STATUSES,
    _TERMINAL_STATUSES,
    TaskPersistenceError,
    TaskQueueManager,
)

# ── Test 1: _append raises TaskPersistenceError on OSError ─────────────


def test_append_raises_task_persistence_error_on_oserror(tmp_path: Path) -> None:
    """Test that _append raises TaskPersistenceError when file write fails.

    Mock the file open to raise OSError, then verify TaskPersistenceError
    is raised. Test through add_task() which calls _append().
    """
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)

    tqm = TaskQueueManager(anima_dir)

    state_dir = anima_dir / "state"
    state_dir.chmod(0o444)
    try:
        with pytest.raises(TaskPersistenceError):
            tqm.add_task(
                source="human",
                original_instruction="test task",
                assignee="anima",
                summary="test",
                deadline="1h",
            )
    finally:
        state_dir.chmod(0o755)


# ── Test 2: "failed" status is valid ───────────────────────────────────


def test_update_status_failed_is_valid(tmp_path: Path) -> None:
    """Test that update_status("failed") works correctly.

    Create a task, update to "failed", verify it returns the entry
    with status="failed".
    """
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)

    tqm = TaskQueueManager(anima_dir)
    entry = tqm.add_task(
        source="human",
        original_instruction="test",
        assignee="anima",
        summary="test task",
        deadline="1h",
    )

    result = tqm.update_status(entry.task_id, "failed")
    assert result is not None
    assert result.status == "failed"
    assert result.task_id == entry.task_id


def test_update_meta_is_durable_and_preserves_status(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    tqm = TaskQueueManager(anima_dir)
    entry = tqm.add_task(
        source="human",
        original_instruction="retry task",
        assignee="anima",
        summary="retry task",
        meta={"task_desc": {"title": "retry"}},
    )

    updated = tqm.update_meta(entry.task_id, {"retry_count": 2})

    assert updated is not None
    assert updated.status == "pending"
    assert updated.meta["task_desc"] == {"title": "retry"}
    assert updated.meta["retry_count"] == 2

    reloaded = TaskQueueManager(anima_dir).get_task_by_id(entry.task_id)
    assert reloaded is not None
    assert reloaded.status == "pending"
    assert reloaded.meta["retry_count"] == 2


# ── Test 3: _TERMINAL_STATUSES and compact() ────────────────────────────


def test_compact_removes_terminal_statuses(tmp_path: Path) -> None:
    """Test that compact() removes "failed" tasks (along with "done" and "cancelled").

    Create tasks with various statuses, compact, verify only non-terminal remain.
    """
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)

    tqm = TaskQueueManager(anima_dir)

    e1 = tqm.add_task(
        source="human",
        original_instruction="task 1",
        assignee="anima",
        summary="pending",
        deadline="1h",
    )
    e2 = tqm.add_task(
        source="human",
        original_instruction="task 2",
        assignee="anima",
        summary="in progress",
        deadline="1h",
    )
    e3 = tqm.add_task(
        source="human",
        original_instruction="task 3",
        assignee="anima",
        summary="done",
        deadline="1h",
    )
    e4 = tqm.add_task(
        source="human",
        original_instruction="task 4",
        assignee="anima",
        summary="failed",
        deadline="1h",
    )
    e5 = tqm.add_task(
        source="human",
        original_instruction="task 5",
        assignee="anima",
        summary="cancelled",
        deadline="1h",
    )

    tqm.update_status(e3.task_id, "done")
    tqm.update_status(e4.task_id, "failed")
    tqm.update_status(e5.task_id, "cancelled")

    assert "failed" in _TERMINAL_STATUSES
    assert "done" in _TERMINAL_STATUSES
    assert "cancelled" in _TERMINAL_STATUSES

    removed = tqm.compact()

    assert removed == 3  # done, failed, cancelled
    remaining = tqm.list_tasks()
    remaining_ids = {t.task_id for t in remaining}
    assert e1.task_id in remaining_ids
    assert e2.task_id in remaining_ids
    assert e3.task_id not in remaining_ids
    assert e4.task_id not in remaining_ids
    assert e5.task_id not in remaining_ids


# ── Test 4: task_tracker filters "failed" correctly ─────────────────────


def test_task_tracker_completed_includes_failed(tmp_path: Path) -> None:
    """Test that task_tracker with status='completed' includes 'failed' tasks."""
    from unittest.mock import MagicMock

    from core.tooling.handler import ToolHandler

    animas_dir = tmp_path / "animas"
    sakura_dir = animas_dir / "sakura"
    hinata_dir = animas_dir / "hinata"
    sakura_dir.mkdir(parents=True)
    hinata_dir.mkdir(parents=True)
    (sakura_dir / "permissions.md").write_text("", encoding="utf-8")
    (sakura_dir / "state").mkdir(exist_ok=True)
    (hinata_dir / "state").mkdir(exist_ok=True)

    # Create delegated task in sakura's queue
    from core.memory.task_queue import TaskQueueManager

    sakura_tqm = TaskQueueManager(sakura_dir)
    hinata_tqm = TaskQueueManager(hinata_dir)

    hinata_task = hinata_tqm.add_task(
        source="human",
        original_instruction="subordinate task",
        assignee="hinata",
        summary="sub task",
        deadline="1h",
    )
    hinata_tqm.update_status(hinata_task.task_id, "failed")

    sakura_tqm.add_delegated_task(
        original_instruction="delegate to hinata",
        assignee="hinata",
        summary="delegated",
        deadline="1h",
        meta={"delegated_to": "hinata", "delegated_task_id": hinata_task.task_id},
    )

    memory = MagicMock()
    memory.read_permissions.return_value = ""

    from core.config.models import AnimaModelConfig

    mock_cfg = MagicMock()
    mock_cfg.animas = {
        "sakura": AnimaModelConfig(),
        "hinata": AnimaModelConfig(supervisor="sakura"),
    }

    with (
        patch("core.config.models.load_config", return_value=mock_cfg),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
    ):
        handler = ToolHandler(
            anima_dir=sakura_dir,
            memory=memory,
            tool_registry=[],
        )
        result = handler.handle("task_tracker", {"status": "completed"})

    import json

    parsed = json.loads(result)
    assert len(parsed) >= 1
    assert any(e["subordinate_status"] == "failed" for e in parsed)


def test_task_tracker_active_excludes_failed(tmp_path: Path) -> None:
    """Test that task_tracker with status='active' excludes 'failed' tasks."""
    from unittest.mock import MagicMock

    from core.tooling.handler import ToolHandler

    animas_dir = tmp_path / "animas"
    sakura_dir = animas_dir / "sakura"
    hinata_dir = animas_dir / "hinata"
    sakura_dir.mkdir(parents=True)
    hinata_dir.mkdir(parents=True)
    (sakura_dir / "permissions.md").write_text("", encoding="utf-8")
    (sakura_dir / "state").mkdir(exist_ok=True)
    (hinata_dir / "state").mkdir(exist_ok=True)

    from core.memory.task_queue import TaskQueueManager

    sakura_tqm = TaskQueueManager(sakura_dir)
    hinata_tqm = TaskQueueManager(hinata_dir)

    hinata_task = hinata_tqm.add_task(
        source="human",
        original_instruction="subordinate task",
        assignee="hinata",
        summary="sub task",
        deadline="1h",
    )
    hinata_tqm.update_status(hinata_task.task_id, "failed")

    sakura_tqm.add_delegated_task(
        original_instruction="delegate to hinata",
        assignee="hinata",
        summary="delegated",
        deadline="1h",
        meta={"delegated_to": "hinata", "delegated_task_id": hinata_task.task_id},
    )

    memory = MagicMock()
    memory.read_permissions.return_value = ""

    from core.config.models import AnimaModelConfig

    mock_cfg = MagicMock()
    mock_cfg.animas = {
        "sakura": AnimaModelConfig(),
        "hinata": AnimaModelConfig(supervisor="sakura"),
    }

    with (
        patch("core.config.models.load_config", return_value=mock_cfg),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
    ):
        handler = ToolHandler(
            anima_dir=sakura_dir,
            memory=memory,
            tool_registry=[],
        )
        result = handler.handle("task_tracker", {"status": "active"})

    # "active" should exclude failed — so we expect no matching delegated tasks
    # (or the "no matching" message)
    import json

    try:
        parsed = json.loads(result)
        # If we get a list, it should not contain failed tasks
        if isinstance(parsed, list):
            assert not any(e["subordinate_status"] == "failed" for e in parsed)
    except json.JSONDecodeError:
        # May return i18n message like "no matching delegated tasks"
        assert "委譲" in result or "delegated" in result.lower() or "matching" in result.lower()


# ── Test: compact() creates archive file ──────────────────────────────


def _make_tasks_with_statuses(tqm: TaskQueueManager) -> dict[str, str]:
    """Helper: create tasks and update to various statuses. Returns {task_id: status}."""
    ids: dict[str, str] = {}
    for status in ("pending", "in_progress", "done", "failed", "cancelled", "delegated", "blocked"):
        e = tqm.add_task(
            source="human",
            original_instruction=f"instruction for {status} task " + "x" * 300,
            assignee="anima",
            summary=f"summary-{status}",
            deadline="1h",
        )
        if status != "pending":
            tqm.update_status(e.task_id, status)
        ids[e.task_id] = status
    return ids


def test_compact_creates_archive_file(tmp_path: Path) -> None:
    """compact() should create task_queue_archive.jsonl with terminal tasks."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    _make_tasks_with_statuses(tqm)
    assert not tqm.archive_path.exists()

    removed = tqm.compact()

    assert removed == 3  # done, failed, cancelled
    assert tqm.archive_path.exists()


def test_compact_archive_contains_correct_tasks(tmp_path: Path) -> None:
    """Archive should contain exactly the terminal tasks with correct fields."""
    import json

    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    ids = _make_tasks_with_statuses(tqm)
    tqm.compact()

    archived = []
    for line in tqm.archive_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            archived.append(json.loads(line))

    archived_ids = {a["task_id"] for a in archived}
    for tid, status in ids.items():
        if status in _TERMINAL_STATUSES:
            assert tid in archived_ids, f"Terminal task {tid} ({status}) not in archive"
        else:
            assert tid not in archived_ids, f"Active task {tid} ({status}) should not be in archive"

    for a in archived:
        assert "original_instruction" in a
        assert "summary" in a
        assert a["status"] in _TERMINAL_STATUSES


def test_compact_queue_retains_only_active(tmp_path: Path) -> None:
    """After compact, queue should contain only non-terminal tasks."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    ids = _make_tasks_with_statuses(tqm)
    tqm.compact()

    all_tasks = tqm._load_all()
    for tid, entry in all_tasks.items():
        assert entry.status not in _TERMINAL_STATUSES, f"Terminal task {tid} still in queue"
    assert len(all_tasks) == sum(1 for s in ids.values() if s not in _TERMINAL_STATUSES)


def test_compact_appends_to_existing_archive(tmp_path: Path) -> None:
    """Multiple compact() calls should append to archive, not overwrite."""
    import json

    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    e1 = tqm.add_task(source="human", original_instruction="t1", assignee="a", summary="s1", deadline="1h")
    tqm.update_status(e1.task_id, "done")
    tqm.compact()

    e2 = tqm.add_task(source="human", original_instruction="t2", assignee="a", summary="s2", deadline="1h")
    tqm.update_status(e2.task_id, "cancelled")
    tqm.compact()

    lines = [ln for ln in tqm.archive_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    archived_ids = {json.loads(ln)["task_id"] for ln in lines}
    assert e1.task_id in archived_ids
    assert e2.task_id in archived_ids


def test_compact_no_terminal_returns_zero(tmp_path: Path) -> None:
    """compact() with no terminal tasks should return 0 and not create archive."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    tqm.add_task(source="human", original_instruction="t", assignee="a", summary="s", deadline="1h")
    assert tqm.compact() == 0
    assert not tqm.archive_path.exists()


# ── Test: list_tasks default returns only active ─────────────────────


def test_list_tasks_default_active_only(tmp_path: Path) -> None:
    """list_tasks() without filter should return only active statuses."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    ids = _make_tasks_with_statuses(tqm)
    result = tqm.list_tasks()

    result_statuses = {t.status for t in result}
    assert result_statuses <= _ACTIVE_STATUSES
    assert len(result) == sum(1 for s in ids.values() if s in _ACTIVE_STATUSES)


def test_list_tasks_status_filter_done(tmp_path: Path) -> None:
    """list_tasks(status='done') should return only done tasks."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    _make_tasks_with_statuses(tqm)
    result = tqm.list_tasks(status="done")

    assert all(t.status == "done" for t in result)
    assert len(result) == 1


def test_list_tasks_status_filter_delegated(tmp_path: Path) -> None:
    """list_tasks(status='delegated') returns delegated tasks."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    _make_tasks_with_statuses(tqm)
    result = tqm.list_tasks(status="delegated")

    assert all(t.status == "delegated" for t in result)
    assert len(result) == 1


# ── Test: _handle_list_tasks detail and truncation ───────────────────


def test_handle_list_tasks_truncates_instruction(tmp_path: Path) -> None:
    """_handle_list_tasks without detail truncates original_instruction."""
    import json as _json

    from core.tooling.handler_skills import _INSTRUCTION_TRUNCATE_LEN

    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    long_instruction = "A" * 500
    tqm.add_task(
        source="human",
        original_instruction=long_instruction,
        assignee="anima",
        summary="test",
        deadline="1h",
    )

    from core.tooling.handler_skills import SkillsToolsMixin

    mixin = SkillsToolsMixin.__new__(SkillsToolsMixin)
    mixin._anima_dir = anima_dir

    result = mixin._handle_list_tasks({})
    parsed = _json.loads(result)
    assert len(parsed) == 1
    instr = parsed[0]["original_instruction"]
    assert len(instr) == _INSTRUCTION_TRUNCATE_LEN + 3  # 200 + "..."
    assert instr.endswith("...")


def test_handle_list_tasks_detail_returns_full(tmp_path: Path) -> None:
    """_handle_list_tasks with detail=true returns full original_instruction."""
    import json as _json

    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    long_instruction = "B" * 500
    tqm.add_task(
        source="human",
        original_instruction=long_instruction,
        assignee="anima",
        summary="test",
        deadline="1h",
    )

    from core.tooling.handler_skills import SkillsToolsMixin

    mixin = SkillsToolsMixin.__new__(SkillsToolsMixin)
    mixin._anima_dir = anima_dir

    result = mixin._handle_list_tasks({"detail": True})
    parsed = _json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["original_instruction"] == long_instruction


def test_handle_list_tasks_no_indent(tmp_path: Path) -> None:
    """list_tasks output should not contain indentation (compact JSON)."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    tqm.add_task(
        source="human",
        original_instruction="short",
        assignee="anima",
        summary="test",
        deadline="1h",
    )

    from core.tooling.handler_skills import SkillsToolsMixin

    mixin = SkillsToolsMixin.__new__(SkillsToolsMixin)
    mixin._anima_dir = anima_dir

    result = mixin._handle_list_tasks({})
    assert "\n  " not in result


# ── Test: add_task task_id, meta, status (submit_tasks support) ────────────────


def test_add_task_with_task_id_uses_provided_id(tmp_path: Path) -> None:
    """add_task with task_id uses the provided ID instead of generating one."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    entry = tqm.add_task(
        source="anima",
        original_instruction="compile",
        assignee="self",
        summary="コンパイル",
        task_id="compile123",
    )
    assert entry.task_id == "compile123"


def test_add_task_with_meta_and_status_in_progress(tmp_path: Path) -> None:
    """add_task with meta and status=in_progress for TaskExec tracking."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    entry = tqm.add_task(
        source="anima",
        original_instruction="run build",
        assignee="self",
        summary="ビルド実行",
        task_id="build001",
        meta={"executor": "taskexec"},
        status="in_progress",
    )
    assert entry.status == "in_progress"
    assert entry.meta.get("executor") == "taskexec"


# ── Test: get_failed_taskexec and format_for_priming failed section ─────────


def test_get_failed_taskexec_returns_only_taskexec_failed(tmp_path: Path) -> None:
    """get_failed_taskexec returns only failed tasks with meta.executor=taskexec."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    e1 = tqm.add_task(
        source="human",
        original_instruction="t1",
        assignee="a",
        summary="s1",
        deadline="1h",
    )
    tqm.update_status(e1.task_id, "failed")

    e2 = tqm.add_task(
        source="anima",
        original_instruction="t2",
        assignee="self",
        summary="s2",
        task_id="taskexec2",
        meta={"executor": "taskexec"},
        status="in_progress",
    )
    tqm.update_status(e2.task_id, "failed", summary="API timeout")

    failed = tqm.get_failed_taskexec()
    assert len(failed) == 1
    assert failed[0].task_id == "taskexec2"
    assert failed[0].summary == "API timeout"


def test_format_for_priming_shows_auto_taskexec_for_in_progress(tmp_path: Path) -> None:
    """format_for_priming shows (auto: TaskExec) for in_progress tasks with meta.executor."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    tqm.add_task(
        source="anima",
        original_instruction="compile",
        assignee="self",
        summary="コンパイル",
        task_id="abc12345",
        meta={"executor": "taskexec"},
        status="in_progress",
    )

    result = tqm.format_for_priming(budget_tokens=500)
    assert "(auto: TaskExec)" in result
    assert "abc12345" in result or "abc1234" in result


def test_format_for_priming_includes_failed_section(tmp_path: Path) -> None:
    """format_for_priming includes failed TaskExec tasks when present."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "state").mkdir(parents=True)
    tqm = TaskQueueManager(anima_dir)

    tqm.add_task(
        source="anima",
        original_instruction="fetch data",
        assignee="self",
        summary="データ取得",
        task_id="xyz99999",
        meta={"executor": "taskexec"},
        status="in_progress",
    )
    tqm.update_status("xyz99999", "failed", summary="データ取得 — FAILED: API timeout")

    result = tqm.format_for_priming(budget_tokens=500)
    assert "❌ Failed" in result or "Failed (要対処)" in result or "Failed (action required)" in result
    assert "xyz99999" in result or "xyz9999" in result  # truncated to 8 chars
    assert "データ取得" in result


# ── TaskBoard metadata sync on terminal status ─────────────────────────


def _tqm_with_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "sakura"):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    anima_dir = data_dir / "animas" / name
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    return TaskQueueManager(anima_dir), data_dir


def test_update_status_terminal_archives_existing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal status closes existing TaskBoard metadata to archived/done."""
    from core.taskboard.models import AttentionVisibility, BoardColumn
    from core.taskboard.store import TaskBoardStore

    tqm, data_dir = _tqm_with_data_dir(tmp_path, monkeypatch)
    entry = tqm.add_task(
        source="human",
        original_instruction="delegate work",
        assignee="sakura",
        summary="delegate work",
        task_id="task-term-1",
    )
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        actor="delegator",
        visibility="active",
        column="waiting",
    )

    for terminal in ("done", "cancelled"):
        # Reset to active so each terminal path is exercised independently.
        tqm.update_status(entry.task_id, "pending")
        store.upsert_metadata(
            anima_name="sakura",
            task_id=entry.task_id,
            actor="test",
            visibility="active",
            column="waiting",
        )
        result = tqm.update_status(entry.task_id, terminal)
        assert result is not None
        assert result.status == terminal
        meta = store.get_metadata("sakura", entry.task_id)
        assert meta is not None
        assert meta.visibility == AttentionVisibility.ARCHIVED
        assert meta.column == BoardColumn.DONE
        assert meta.updated_by == "sakura"
        events = store.list_events(anima_name="sakura", task_id=entry.task_id)
        assert any(event["event_type"] == "archived" for event in events)


def test_update_status_failed_keeps_metadata_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """failed must NOT archive the card: it stays retriable (failed_review_window)."""
    from core.taskboard.models import AttentionVisibility
    from core.taskboard.store import TaskBoardStore

    tqm, data_dir = _tqm_with_data_dir(tmp_path, monkeypatch)
    entry = tqm.add_task(
        source="human",
        original_instruction="flaky work",
        assignee="sakura",
        summary="flaky work",
        task_id="task-fail-1",
    )
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        actor="delegator",
        visibility="active",
        column="todo",
    )

    result = tqm.update_status(entry.task_id, "failed")
    assert result is not None and result.status == "failed"
    meta = store.get_metadata("sakura", entry.task_id)
    assert meta is not None
    assert meta.visibility == AttentionVisibility.ACTIVE


def test_update_status_pending_reactivates_archived_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-queueing an archived task to pending revives the board card so the
    pending attention gate does not cancel it as "archived by TaskBoard"."""
    from core.taskboard.models import AttentionVisibility, BoardColumn
    from core.taskboard.store import TaskBoardStore

    tqm, data_dir = _tqm_with_data_dir(tmp_path, monkeypatch)
    entry = tqm.add_task(
        source="human",
        original_instruction="revivable work",
        assignee="sakura",
        summary="revivable work",
        task_id="task-revive-1",
    )
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        actor="delegator",
        visibility="active",
        column="todo",
    )
    tqm.update_status(entry.task_id, "cancelled")
    meta = store.get_metadata("sakura", entry.task_id)
    assert meta is not None and meta.visibility == AttentionVisibility.ARCHIVED

    result = tqm.update_status(entry.task_id, "pending")
    assert result is not None and result.status == "pending"
    meta = store.get_metadata("sakura", entry.task_id)
    assert meta is not None
    assert meta.visibility == AttentionVisibility.ACTIVE
    assert meta.column == BoardColumn.TODO

    # Tombstoned cards are deliberate suppressions and must stay suppressed.
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        actor="test",
        visibility="tombstoned",
    )
    tqm.update_status(entry.task_id, "pending")
    meta = store.get_metadata("sakura", entry.task_id)
    assert meta is not None
    assert meta.visibility == AttentionVisibility.TOMBSTONED


def test_update_status_terminal_does_not_create_metadata_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal status must not invent a TaskBoard metadata row when none exists."""
    from core.taskboard.store import TaskBoardStore

    tqm, data_dir = _tqm_with_data_dir(tmp_path, monkeypatch)
    entry = tqm.add_task(
        source="human",
        original_instruction="plain task",
        assignee="sakura",
        summary="plain task",
        task_id="task-no-meta",
    )

    result = tqm.update_status(entry.task_id, "done")
    assert result is not None
    assert result.status == "done"

    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    assert store.get_metadata("sakura", entry.task_id) is None
    assert store.list_metadata(anima_name="sakura") == []


@pytest.mark.parametrize("visibility", ["expired", "archived", "tombstoned"])
def test_update_status_terminal_preserves_suppressed_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visibility: str,
) -> None:
    """Terminal queue sync must not replace a more specific suppression reason."""
    from core.taskboard.models import AttentionVisibility
    from core.taskboard.store import TaskBoardStore

    tqm, data_dir = _tqm_with_data_dir(tmp_path, monkeypatch)
    entry = tqm.add_task(
        source="human",
        original_instruction="suppressed task",
        assignee="sakura",
        summary="suppressed task",
        task_id=f"task-{visibility}",
    )
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        actor="planner",
        visibility=visibility,
        column="suppressed",
    )

    tqm.update_status(entry.task_id, "cancelled")

    metadata = store.get_metadata("sakura", entry.task_id)
    assert metadata is not None
    assert metadata.visibility == AttentionVisibility(visibility)
    assert metadata.column.value == "suppressed"
    assert metadata.updated_by == "planner"


def test_update_status_succeeds_when_taskboard_store_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queue terminal update remains successful if TaskBoard store fails."""
    tqm, _data_dir = _tqm_with_data_dir(tmp_path, monkeypatch)
    entry = tqm.add_task(
        source="human",
        original_instruction="resilient task",
        assignee="sakura",
        summary="resilient task",
        task_id="task-resilient",
    )

    class _BoomStore:
        def get_metadata(self, *args, **kwargs):
            raise RuntimeError("store down")

        def upsert_metadata(self, *args, **kwargs):
            raise RuntimeError("store down")

    with patch("core.taskboard.store.TaskBoardStore", return_value=_BoomStore()):
        result = tqm.update_status(entry.task_id, "done")

    assert result is not None
    assert result.status == "done"
    reloaded = TaskQueueManager(tqm.anima_dir).get_task_by_id(entry.task_id)
    assert reloaded is not None
    assert reloaded.status == "done"


def test_update_status_non_terminal_does_not_touch_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-terminal transitions leave TaskBoard metadata unchanged."""
    from core.taskboard.models import AttentionVisibility, BoardColumn
    from core.taskboard.store import TaskBoardStore

    tqm, data_dir = _tqm_with_data_dir(tmp_path, monkeypatch)
    entry = tqm.add_task(
        source="human",
        original_instruction="active task",
        assignee="sakura",
        summary="active task",
        task_id="task-active",
    )
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        actor="planner",
        visibility="active",
        column="todo",
        position=3.0,
    )
    before = store.get_metadata("sakura", entry.task_id)
    assert before is not None
    before_updated_at = before.updated_at

    result = tqm.update_status(entry.task_id, "in_progress")
    assert result is not None
    assert result.status == "in_progress"

    after = store.get_metadata("sakura", entry.task_id)
    assert after is not None
    assert after.visibility == AttentionVisibility.ACTIVE
    assert after.column == BoardColumn.TODO
    assert after.position == 3.0
    assert after.updated_at == before_updated_at
    assert after.updated_by == "planner"
