"""Canonical publication boundaries: atomic batches, aliases and host fallback."""

from __future__ import annotations

import errno
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from core.memory.task_queue import TaskQueueManager
from core.tasks_dispatch import is_task_permission_error, publish_delegation, publish_tasks


@pytest.fixture
def anima_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    directory = tmp_path / "animas" / "worker"
    (directory / "state").mkdir(parents=True)
    return directory


def payload(task_id: str = "task-one", **fields) -> dict:
    return {"task_type": "llm", "task_id": task_id, "title": "Task", "description": "Complete this task", **fields}


def test_full_input_and_model_are_published_once(anima_dir):
    text = "大事な確定指示" * 5000
    task = payload(description=text, model="a:openai/model", acceptance_criteria=["preserve full input"])
    with patch("core.config.model_catalog.validate_model_override", return_value=None):
        first = publish_tasks(anima_dir, [task])[0]
        second = publish_tasks(anima_dir, [task])[0]
    manager = TaskQueueManager(anima_dir)
    assert first.task_id == second.task_id
    assert first.original_instruction == text
    assert manager.store.pending("worker") == [task]
    assert not list((anima_dir / "state" / "pending").glob("*.json"))


def test_all_workspace_validation_precedes_writes(anima_dir):
    with (
        patch("core.workspace.resolve_workspace", side_effect=ValueError("unknown workspace")),
        pytest.raises(ValueError, match="workspace"),
    ):
        publish_tasks(anima_dir, [payload(), payload("second", workspace="missing")])
    assert TaskQueueManager(anima_dir).list_tasks() == []


def test_second_task_failure_rolls_back_whole_batch(anima_dir):
    original = TaskQueueManager.submit

    def submit(manager, task, **kwargs):
        if task["task_id"] == "second":
            raise ValueError("publish interrupted")
        return original(manager, task, **kwargs)

    with patch.object(TaskQueueManager, "submit", submit), pytest.raises(ValueError, match="interrupted"):
        publish_tasks(anima_dir, [payload(), payload("second")])
    assert TaskQueueManager(anima_dir).list_tasks() == []
    assert TaskQueueManager(anima_dir).store.pending("worker") == []


def test_completed_redelivery_does_not_schedule_work_again(anima_dir):
    manager = TaskQueueManager(anima_dir)
    publish_tasks(anima_dir, [payload()])
    manager.update_status("task-one", "done")
    result = publish_tasks(anima_dir, [payload()])[0]
    assert result.status == "done"
    assert manager.store.pending("worker") == []


def test_delegation_is_one_task_with_requester_alias(anima_dir):
    publish_delegation(anima_dir, payload(), delegator="boss", tracking_task_id="tracking-one")
    store = TaskQueueManager(anima_dir).store
    assert store.read("boss")["tracking-one"].status == "delegated"
    with store.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    TaskQueueManager(anima_dir).update_status("task-one", "done")
    assert store.read("boss")["tracking-one"].status == "done"


@pytest.mark.parametrize("message", ["database is locked", "database disk image is malformed", "no such table: tasks"])
def test_database_errors_do_not_proxy_to_host(anima_dir, message):
    with (
        patch.object(
            TaskQueueManager, "store", new_callable=PropertyMock, side_effect=sqlite3.OperationalError(message)
        ),
        patch("httpx.post") as post,
        pytest.raises(sqlite3.OperationalError),
    ):
        publish_tasks(anima_dir, [payload()])
    post.assert_not_called()


def test_readonly_sandbox_publishes_through_host(anima_dir):
    response = MagicMock()
    response.json.return_value = {
        "tasks": [
            {
                "task_id": "task-one",
                "ts": "2026-09-08T00:00:00+00:00",
                "updated_at": "2026-09-08T00:00:00+00:00",
                "source": "anima",
                "original_instruction": "Complete this task",
                "assignee": "worker",
                "status": "pending",
                "summary": "Task",
            }
        ]
    }
    with (
        patch.object(
            TaskQueueManager, "store", new_callable=PropertyMock, side_effect=PermissionError(errno.EACCES, "denied")
        ),
        patch("httpx.post", return_value=response) as post,
    ):
        entries = publish_tasks(anima_dir, [payload()])
    assert entries[0].task_id == "task-one"
    assert post.call_args.kwargs["json"]["tasks"] == [payload()]
    response.raise_for_status.assert_called_once()


def test_permission_detection_does_not_mask_disk_failure():
    assert is_task_permission_error(PermissionError(errno.EACCES, "denied"))
    assert is_task_permission_error(sqlite3.OperationalError("attempt to write a readonly database"))
    assert not is_task_permission_error(OSError(errno.ENOSPC, "disk full"))
    assert not is_task_permission_error(sqlite3.OperationalError("unable to open database file"))


def test_readonly_single_lookup_passes_task_id_to_host(anima_dir):
    queue = TaskQueueManager(anima_dir)
    entry = publish_tasks(anima_dir, [payload()])[0]
    response = MagicMock()
    response.json.return_value = {"tasks": [entry.model_dump(mode="json")]}
    with (
        patch.object(
            TaskQueueManager, "store", new_callable=PropertyMock, side_effect=PermissionError(errno.EACCES, "denied")
        ),
        patch("httpx.get", return_value=response) as get,
    ):
        assert queue.get_task_by_id(entry.task_id) == entry
    assert get.call_args.kwargs["params"]["task_id"] == entry.task_id


def test_resume_uses_saved_input_and_preserves_source_meta(anima_dir):
    original = payload(
        constraints=["Never publish externally"], acceptance_criteria=["Tests pass"], context="Original context"
    )
    publish_tasks(anima_dir, [original], source="human", meta={"requester": "user"})
    manager = TaskQueueManager(anima_dir)
    attempt = manager.store.claim("worker", "task-one", {"pid": 1})
    assert attempt is not None
    manager.store.finish(attempt["_attempt_token"], status="pending", stop_kind="interrupted")
    # An ordinary duplicate cannot silently retry an ended attempt.
    publish_tasks(anima_dir, [payload(description="Dropped constraints")])
    assert manager.store.pending("worker") == []
    assert manager.store.get_input("worker", "task-one") == original
    entry = publish_tasks(anima_dir, [{"task_id": "task-one", "resume": True}])[0]
    assert manager.store.pending("worker") == [original]
    assert entry.source == "human"
    assert entry.meta["requester"] == "user"


@pytest.mark.parametrize(
    "task",
    [
        {"task_id": "unknown", "resume": True},
        {"task_id": "task-one", "resume": True, "description": "rewrite"},
        {"task_id": "../escape", "resume": True},
    ],
)
def test_invalid_resume_rolls_back_other_tasks(anima_dir, task):
    with pytest.raises(ValueError):
        publish_tasks(anima_dir, [payload("new-task"), task])
    assert TaskQueueManager(anima_dir).list_tasks() == []


def test_submit_tool_accepts_id_only_resume(anima_dir):
    import json

    from core.tooling.handler_skills import SkillsToolsMixin

    publish_tasks(anima_dir, [payload()])
    manager = TaskQueueManager(anima_dir)
    attempt = manager.store.claim("worker", "task-one", {"pid": 1})
    manager.store.finish(attempt["_attempt_token"], status="pending", stop_kind="interrupted")
    handler = object.__new__(SkillsToolsMixin)
    handler._anima_dir = anima_dir
    handler._anima_name = "worker"
    handler._pending_executor_wake = MagicMock()
    result = handler._handle_submit_tasks(
        {"batch_id": "resume-event", "tasks": [{"task_id": "task-one", "resume": True}]}
    )
    assert json.loads(result)["status"] == "submitted"
    assert manager.store.pending("worker") == [payload()]
    handler._pending_executor_wake.assert_called_once()


def test_dependency_can_reference_an_existing_canonical_task(anima_dir):
    publish_tasks(anima_dir, [payload("first")])
    publish_tasks(anima_dir, [payload("second", depends_on=["first"])])
    assert len(TaskQueueManager(anima_dir).list_tasks()) == 2


def test_command_tasks_are_not_accepted_at_llm_publication_boundary(anima_dir):
    with pytest.raises(ValueError, match="Only LLM tasks"):
        publish_tasks(anima_dir, [payload(task_type="command")])
    assert TaskQueueManager(anima_dir).list_tasks() == []


def test_host_update_identity_cannot_be_supplied_by_model(anima_dir):
    import json

    from core.taskboard.tasks import attempt_scope
    from core.tooling.handler_skills import SkillsToolsMixin

    entry = publish_tasks(anima_dir, [payload()])[0]
    response = MagicMock()
    response.json.return_value = {"ok": True, "task": entry.model_copy(update={"status": "done"}).model_dump()}
    handler = object.__new__(SkillsToolsMixin)
    handler._anima_dir = anima_dir
    handler._anima_name = "worker"
    handler._activity = MagicMock()
    identity = {"anima": "worker", "task_id": "task-one", "token": "actual-attempt"}
    with (
        attempt_scope(identity),
        patch.object(
            TaskQueueManager, "store", new_callable=PropertyMock, side_effect=PermissionError(errno.EACCES, "denied")
        ),
        patch("httpx.post", return_value=response) as post,
    ):
        result = handler._handle_update_task(
            {"task_id": "task-one", "status": "done", "attempt_identity": {"token": "model-invented"}}
        )
    assert json.loads(result)["status"] == "done"
    assert post.call_args.kwargs["json"]["attempt_identity"] == identity


def test_host_publication_propagates_execution_attempt_identity(anima_dir):
    from core.taskboard.tasks import attempt_scope

    entry = publish_tasks(anima_dir, [payload()])[0]
    response = MagicMock()
    response.json.return_value = {"ok": True, "tasks": [entry.model_dump()]}
    identity = {"anima": "worker", "task_id": "task-one", "token": "current-attempt"}
    with (
        attempt_scope(identity),
        patch.object(
            TaskQueueManager, "store", new_callable=PropertyMock, side_effect=PermissionError(errno.EACCES, "denied")
        ),
        patch("httpx.post", return_value=response) as post,
    ):
        publish_tasks(anima_dir, [{"task_id": "task-one", "resume": True}])
    assert post.call_args.kwargs["json"]["attempt_identity"] == identity
    assert post.call_args.kwargs["json"]["tasks"] == [{"task_id": "task-one", "resume": True}]
