from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct task dispatch for deterministic external events."""

import errno
import logging
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from core.exceptions import TaskPersistenceError
from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir
from core.schemas import TaskEntry

logger = logging.getLogger("animaworks.tasks_dispatch")


def _server_url() -> str:
    return os.environ.get("ANIMAWORKS_SERVER_URL", "http://localhost:18500").rstrip("/")


def _post_tasks(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Retry a lost response with the same IDs, never publish new work IDs."""
    import httpx

    from core.taskboard.tasks import current_attempt_identity

    payload = {**payload, "attempt_identity": current_attempt_identity()}
    for attempt in range(2):
        try:
            response = httpx.post(f"{_server_url()}/api/internal/{endpoint}", json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except httpx.TransportError:
            if attempt:
                raise
        except httpx.HTTPStatusError as exc:
            if attempt or exc.response.status_code < 500:
                raise
    raise RuntimeError("Unreachable task publication retry state")


def is_task_permission_error(exc: BaseException) -> bool:
    """Proxy denied storage access, never lock contention or malformed SQL."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
            return True
        if isinstance(current, sqlite3.OperationalError):
            code = getattr(current, "sqlite_errorcode", 0) & 0xFF
            denied = any(
                text in str(current).lower()
                for text in (
                    "readonly",
                    "read-only",
                    "permission denied",
                    "access denied",
                    "operation not permitted",
                )
            )
            if denied and code in {0, sqlite3.SQLITE_READONLY, sqlite3.SQLITE_PERM, sqlite3.SQLITE_CANTOPEN}:
                return True
        current = current.__cause__ or current.__context__
    return False


def update_task(
    manager: TaskQueueManager,
    task_id: str,
    status: str,
    *,
    summary: str | None = None,
    result: str | None = None,
) -> TaskEntry | None:
    """Atomically declare a task result, proxying only sandbox storage denial."""
    from core.time_utils import now_iso

    if status not in {"pending", "delegated", "done", "cancelled"}:
        raise ValueError(f"Invalid task update status: {status}")
    if result is not None:
        summary = result
    meta: dict[str, Any] = {}
    if status == "done":
        meta = {"completed_by": "agent_declaration", "declared_at": now_iso()}
        if summary is not None:
            meta["result_note"] = summary
    try:
        with manager.store.transaction():
            if meta and manager.update_meta(task_id, meta, summary=summary) is None:
                return None
            return manager.update_status(task_id, status, summary=summary)
    except Exception as exc:
        if not is_task_permission_error(exc):
            raise
        response = _post_tasks(
            "update-task",
            {
                "anima_name": manager.anima_dir.name,
                "task_id": task_id,
                "status": status,
                "summary": summary,
                "meta": meta,
            },
        )
        if response.get("ok") is not True or not isinstance(response.get("task"), dict):
            raise TaskPersistenceError("Unexpected response from task update host") from exc
        return TaskEntry.model_validate(response["task"])


def read_tasks_via_server(
    anima_name: str, *, include_archived: bool = False, task_id: str | None = None
) -> dict[str, TaskEntry]:
    """Read a host-owned snapshot when the worker cannot access the task DB."""
    import httpx

    params = {"anima_name": anima_name, "include_archived": include_archived}
    if task_id is not None:
        params["task_id"] = task_id
    response = httpx.get(
        f"{_server_url()}/api/internal/tasks",
        params=params,
        timeout=30.0,
    )
    response.raise_for_status()
    return {item["task_id"]: TaskEntry(**item) for item in response.json()["tasks"]}


def read_executable_ids_via_server(anima_name: str) -> set[str]:
    """Read execution-input availability without granting shared DB access."""
    import httpx

    response = httpx.get(f"{_server_url()}/api/internal/tasks", params={"anima_name": anima_name}, timeout=30.0)
    response.raise_for_status()
    return set(response.json()["input_ids"])


def validate_task_payloads(
    anima_name: str,
    payloads: list[dict[str, Any]],
    *,
    known_ids: set[str] | None = None,
    check_dependencies: bool = True,
) -> list[dict[str, Any]]:
    """Validate the whole batch before any task becomes visible to a worker."""
    from core.config.model_catalog import validate_model_override
    from core.workspace import resolve_workspace

    if not payloads:
        raise ValueError("At least one task is required")
    prepared = []
    for payload in payloads:
        if not isinstance(payload, dict):
            raise ValueError("Each task must be an object")
        fields = ("task_id",) if payload.get("resume") is True else ("task_id", "title", "description")
        for field in fields:
            if not isinstance(payload.get(field), str) or not payload[field].strip():
                raise ValueError(f"Task missing required field: {field}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:#-]{0,199}", payload["task_id"]):
            raise ValueError("Invalid task_id")
        if "resume" in payload and not isinstance(payload["resume"], bool):
            raise ValueError("resume must be a boolean")
        if payload.get("resume") is True:
            if set(payload) != {"task_id", "resume"}:
                raise ValueError("Resume accepts only task_id and resume; saved execution input is immutable")
            prepared.append(dict(payload))
            continue
        task = dict(payload)
        task.setdefault("task_type", "llm")
        if task["task_type"] != "llm":
            raise ValueError("Only LLM tasks can be published through this endpoint")
        model = task.get("model", "")
        if not isinstance(model, str):
            raise ValueError("Task model must be a string")
        if model and (error := validate_model_override(anima_name, model)):
            from core.i18n import t

            raise ValueError(f"Task {task['task_id']}: {t('tooling.model_list_hint', error=error)}")
        if "workspace" in task:
            task.setdefault("working_directory", "")
        workspace = task.get("working_directory") or task.pop("workspace", "")
        if workspace:
            task["working_directory"] = str(resolve_workspace(workspace))
        dependencies = task.get("depends_on", [])
        if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
            raise ValueError("depends_on must be a list of task IDs")
        prepared.append(task)
    _validate_task_dependencies(prepared, known_ids=known_ids, check_membership=check_dependencies)
    return prepared


def _validate_task_dependencies(
    prepared: list[dict[str, Any]],
    *,
    known_ids: set[str] | None = None,
    check_membership: bool = True,
) -> None:
    from core.supervisor.pending_executor import _topological_sort

    ids = {task["task_id"] for task in prepared}
    if len(ids) != len(prepared):
        raise ValueError("Duplicate task_id found in batch")
    for task in prepared:
        if check_membership and any(dep not in ids | (known_ids or set()) for dep in task.get("depends_on", [])):
            raise ValueError(f"Task '{task['task_id']}' depends on unknown task_id")
    _topological_sort(
        [{**task, "depends_on": [dep for dep in task.get("depends_on", []) if dep in ids]} for task in prepared]
    )


def publish_tasks(
    anima_dir: Path,
    payloads: list[dict[str, Any]],
    *,
    source: Literal["human", "anima"] = "anima",
    meta: dict[str, Any] | None = None,
    host_fallback: bool = True,
) -> list[TaskEntry]:
    """Atomically publish complete tasks; a sandbox uses the existing server."""
    prepared = validate_task_payloads(anima_dir.name, payloads, check_dependencies=False)
    if source not in {"human", "anima"}:
        raise ValueError("Invalid task source")
    try:
        manager = TaskQueueManager(anima_dir)
        store = manager.store
        with store.transaction():
            existing = store.read(anima_dir.name, archived=True)
            originals = {}
            for payload in prepared:
                if payload.get("resume") is True:
                    original = store.get_input(anima_dir.name, payload["task_id"])
                    previous = existing.get(payload["task_id"])
                    if previous is None or original is None:
                        raise ValueError(f"No saved execution input for task: {payload['task_id']}")
                    if previous.status != "pending":
                        raise ValueError("Only an ended pending task may be resumed")
                    validate_task_payloads(anima_dir.name, [original], check_dependencies=False)
                    originals[payload["task_id"]] = original
            _validate_task_dependencies(
                [originals.get(payload["task_id"], payload) for payload in prepared],
                known_ids=set(existing),
            )
            entries = []
            for payload in prepared:
                previous = existing.get(payload["task_id"])
                if payload.get("resume") is True:
                    original = originals[payload["task_id"]]
                    store.submit(anima_dir.name, previous, original, resume=True)
                    entries.append(store.read(anima_dir.name)[payload["task_id"]])
                    continue
                manager.submit(
                    payload,
                    source=source,
                    meta={
                        **(meta or {}),
                        **{
                            key: payload[key]
                            for key in ("model", "batch_id", "depends_on", "parallel")
                            if key in payload
                        },
                    },
                )
                entries.append(store.read(anima_dir.name, archived=True)[payload["task_id"]])
            return entries
    except (OSError, sqlite3.OperationalError, TaskPersistenceError) as exc:
        if not host_fallback or not is_task_permission_error(exc):
            raise
        result = _post_tasks(
            "submit-tasks",
            {
                "anima_name": anima_dir.name,
                "tasks": prepared,
                "source": source,
                "meta": meta or {},
            },
        )
        return [TaskEntry(**item) for item in result["tasks"]]


def publish_delegation(
    target_dir: Path,
    payload: dict[str, Any],
    *,
    delegator: str,
    tracking_task_id: str,
    host_fallback: bool = True,
) -> bool:
    """Publish task + alias atomically. Return whether the host proxy was used."""
    prepared = validate_task_payloads(target_dir.name, [payload])[0]
    prepared.setdefault("relay_chain", [delegator])
    try:
        manager = TaskQueueManager(target_dir)
        with manager.store.transaction():
            manager.submit(prepared, meta={"model": payload.get("model", "")})
            manager.store.alias(delegator, tracking_task_id, target_dir.name, payload["task_id"])
        return False
    except (OSError, sqlite3.OperationalError, TaskPersistenceError) as exc:
        if not host_fallback or not is_task_permission_error(exc):
            raise
        _post_tasks(
            "delegate-task",
            {
                "delegator": delegator,
                "target": target_dir.name,
                "instruction": prepared["description"],
                "summary": prepared["title"],
                "sub_task_id": prepared["task_id"],
                "tracking_task_id": tracking_task_id,
                "workspace": prepared.get("working_directory", ""),
                "acceptance_criteria": prepared.get("acceptance_criteria", []),
                "model": prepared.get("model", ""),
                "execution_input": prepared,
            },
        )
        return True


def dispatch_direct_task(
    *,
    target: str,
    task_id: str,
    summary: str,
    instruction: str,
    submitted_by: str = "github-event-dispatch",
    meta: dict | None = None,
    animas_dir: Path | None = None,
    model: str | None = None,
) -> bool:
    """Queue a deterministic task and publish it for TaskExec pickup.

    *model* is the optional ``"mode:model"`` (or plain ``"model"``) override
    applied by the pending executor at run time.  It is propagated into both
    the task record's ``meta`` and the published ``task_desc`` (the field the
    executor actually reads), so a per-task model override reaches execution
    without changing the anima's default configuration.
    """
    target_dir = (animas_dir or get_animas_dir()) / target
    if not target_dir.is_dir():
        raise ValueError(f"Anima directory not found: {target}")

    task_meta = {**(meta or {}), "origin": "github-event", "executor": "taskexec"}
    if model:
        task_meta["model"] = model
    task_desc = {
        "task_type": "llm",
        "task_id": task_id,
        "title": summary,
        "description": instruction,
        "context": "",
        "acceptance_criteria": [],
        "constraints": [],
        "file_paths": [],
        "submitted_by": submitted_by,
        "submitted_at": datetime.now(UTC).isoformat(),
        "reply_to": "",
        "source": "delegation",
        "working_directory": "",
    }
    if model:
        task_desc["model"] = model
    manager = TaskQueueManager(target_dir)
    prepared = validate_task_payloads(target, [task_desc])[0]
    with manager.store.transaction():
        if manager.get_task_by_id(task_id) is not None:
            return False
        manager.submit(prepared, meta=task_meta)
        return True
