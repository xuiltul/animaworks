"""Canonical task updates from CLI sandboxes use the same declaration contract as MCP."""

from __future__ import annotations

import errno
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cli.commands.task_cmd import _cmd_update
from core.memory.task_queue import TaskQueueManager
from core.taskboard.tasks import attempt_scope
from core.tasks_dispatch import publish_tasks


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    anima_dir = tmp_path / "animas" / "worker"
    anima_dir.mkdir(parents=True)
    publish_tasks(anima_dir, [{"task_id": "task-1", "title": "Work", "description": "Synthetic work"}])
    return TaskQueueManager(anima_dir)


def _args():
    return SimpleNamespace(task_id="task-1", status="done", summary="Verified once")


def test_cli_done_atomically_records_agent_declaration_and_result(task, capsys):
    _cmd_update(_args(), task)
    entry = task.get_task_by_id("task-1")
    assert entry.status == "done"
    assert entry.meta["completed_by"] == "agent_declaration"
    assert entry.meta["result_note"] == "Verified once"
    assert entry.meta["declared_at"]
    assert json.loads(capsys.readouterr().out)["status"] == "done"


@pytest.mark.parametrize(
    "failure",
    [
        PermissionError(errno.EACCES, "denied"),
        OSError(errno.EROFS, "read-only"),
        sqlite3.OperationalError("attempt to write a readonly database"),
    ],
)
def test_cli_storage_denial_proxies_same_identity_and_declaration(task, failure, capsys):
    entry = task.get_task_by_id("task-1")
    response = MagicMock()
    response.json.return_value = {
        "ok": True,
        "task": entry.model_copy(update={"status": "done"}).model_dump(mode="json"),
    }
    identity = {"anima": "worker", "task_id": "task-1", "token": "attempt-token"}
    with (
        attempt_scope(identity),
        patch.object(task, "update_meta", side_effect=failure),
        patch("httpx.post", return_value=response) as post,
    ):
        _cmd_update(_args(), task)
    post.assert_called_once()
    assert post.call_args.args[0].endswith("/api/internal/update-task")
    payload = post.call_args.kwargs["json"]
    assert payload["attempt_identity"] == identity
    assert payload["anima_name"] == "worker"
    assert payload["task_id"] == "task-1"
    assert payload["status"] == "done"
    assert payload["meta"]["completed_by"] == "agent_declaration"
    assert payload["meta"]["result_note"] == "Verified once"
    assert payload["meta"]["declared_at"]
    assert task.get_task_by_id("task-1").status == "pending"
    assert json.loads(capsys.readouterr().out)["status"] == "done"


@pytest.mark.parametrize("message", ["database is locked", "near UPDATE: syntax error"])
def test_cli_sql_failure_rolls_back_declaration_without_proxy(task, message):
    with (
        patch.object(task, "update_status", side_effect=sqlite3.OperationalError(message)),
        patch("httpx.post") as post,
        pytest.raises(SystemExit) as stopped,
    ):
        _cmd_update(_args(), task)
    assert stopped.value.code == 3
    post.assert_not_called()
    entry = task.get_task_by_id("task-1")
    assert entry.status == "pending"
    assert "completed_by" not in entry.meta
    assert "result_note" not in entry.meta


def test_cli_stale_attempt_is_rejected_without_proxy(task):
    first = task.store.claim("worker", "task-1", {"pid": 1})
    task.store.finish(first["_attempt_token"], status="pending", stop_kind="interrupted")
    publish_tasks(task.anima_dir, [{"task_id": "task-1", "resume": True}])
    task.store.claim("worker", "task-1", {"pid": 1})
    with (
        attempt_scope({"anima": "worker", "task_id": "task-1", "token": first["_attempt_token"]}),
        patch("httpx.post") as post,
        pytest.raises(SystemExit) as stopped,
    ):
        _cmd_update(_args(), task)
    assert stopped.value.code == 3
    post.assert_not_called()
    entry = task.get_task_by_id("task-1")
    assert entry.status == "in_progress"
    assert "completed_by" not in entry.meta


def test_cli_real_readonly_sqlite_falls_back_to_host_update_endpoint(task, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.routes.internal import create_internal_router

    monkeypatch.setattr("core.paths.get_animas_dir", lambda: task.anima_dir.parent)
    store = task.store
    attempt = store.claim("worker", "task-1", {"pid": 1})
    identity = {"anima": "worker", "task_id": "task-1", "token": attempt["_attempt_token"]}
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")
    readonly_errors = []

    @contextmanager
    def readonly_connection():
        db = sqlite3.connect(f"file:{store.db_path}?mode=ro", uri=True, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            yield db
        except Exception as exc:
            readonly_errors.append(exc)
            raise
        finally:
            db.close()

    with TestClient(app) as client:

        def post_to_host(url, *, json, timeout):
            return client.post("/api/internal/update-task", json=json)

        with (
            attempt_scope(identity),
            patch.object(store, "_connect", readonly_connection),
            patch("httpx.post", side_effect=post_to_host) as proxy,
        ):
            _cmd_update(_args(), task)

    assert len(readonly_errors) == 1
    # TaskQueueManager wraps SQLite failures in TaskPersistenceError.
    sqlite_error = readonly_errors[0]
    while not isinstance(sqlite_error, sqlite3.OperationalError):
        sqlite_error = sqlite_error.__cause__
        assert sqlite_error is not None
    assert sqlite_error.sqlite_errorcode & 0xFF == sqlite3.SQLITE_READONLY
    proxy.assert_called_once()
    assert proxy.call_args.kwargs["json"]["attempt_identity"] == identity
    entry = task.get_task_by_id("task-1")
    assert entry.status == "done"
    assert entry.meta["completed_by"] == "agent_declaration"
    assert entry.meta["result_note"] == "Verified once"
