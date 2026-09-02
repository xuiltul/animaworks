from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct task dispatch for deterministic external events."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from core.memory._io import atomic_write_text
from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir

logger = logging.getLogger("animaworks.tasks_dispatch")


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
    entry = TaskQueueManager(target_dir).add_task_if_absent(
        lambda task: task.task_id == task_id,
        source="anima",
        original_instruction=instruction,
        assignee=target,
        summary=summary,
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
    }
    if model:
        task_desc["model"] = model
    atomic_write_text(
        target_dir / "state" / "pending" / f"{task_id}.json",
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
    )
    return True
