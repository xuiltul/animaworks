from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct task dispatch for deterministic external events."""

import json
from datetime import UTC, datetime
from pathlib import Path

from core.memory._io import atomic_write_text
from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir

FAILING_CI_CONCLUSIONS = frozenset({"FAILURE", "CANCELLED", "TIMED_OUT", "STARTUP_FAILURE"})


def pr_exclusive_key(meta: dict | None) -> str:
    """Derive the ``pr-NNNN`` exclusion key so same-PR tasks run serially.

    Matches the manual ``delegate_task`` convention (e.g. ``pr-3999``), so an
    automated gh-ci/gh-review task and a hand-delegated task on the same PR
    share one lock instead of racing on the same worktree.
    """
    number = (meta or {}).get("number")
    return f"pr-{number}" if number else ""


def dispatch_direct_task(
    *,
    target: str,
    task_id: str,
    summary: str,
    instruction: str,
    deadline: str | None = "2h",
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
    exclusive_key = pr_exclusive_key(meta)
    if exclusive_key:
        task_meta["exclusive_key"] = exclusive_key
    if model:
        task_meta["model"] = model
    entry = TaskQueueManager(target_dir).add_task_if_absent(
        lambda task: task.task_id == task_id,
        source="anima",
        original_instruction=instruction,
        assignee=target,
        summary=summary,
        deadline=deadline,
        task_id=task_id,
        meta=task_meta,
    )
    if entry is None:
        return False

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
        "exclusive_key": exclusive_key,
    }
    if model:
        task_desc["model"] = model
    atomic_write_text(
        target_dir / "state" / "pending" / f"{task_id}.json",
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
    )
    return True
