from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct task dispatch for deterministic external events."""

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from core.execution.github_identity import resolve_github_token_env
from core.memory._io import atomic_write_text
from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir

logger = logging.getLogger("animaworks.tasks_dispatch")

FAILING_CI_CONCLUSIONS = frozenset({"FAILURE", "CANCELLED", "TIMED_OUT", "STARTUP_FAILURE"})


def pr_exclusive_key(meta: dict | None) -> str:
    """Derive the ``pr-NNNN`` exclusion key so same-PR tasks run serially.

    Matches the manual ``delegate_task`` convention (e.g. ``pr-3999``), so an
    automated gh-ci/gh-review task and a hand-delegated task on the same PR
    share one lock instead of racing on the same worktree.

    Multi-model review passes (``multipass is True``) are findings-only and
    write to per-model output files (``reviews/pr<N>-frc-...-<slug>.md``), so
    they do not race on a shared worktree and are left parallel (no key) to halve
    pipeline latency.  The ``multipass == "synth"`` integration pass and plain
    gh-ci/gh-cmd/gh-review tasks still serialize on ``pr-<N>``.
    """
    if (meta or {}).get("multipass") is True:
        return ""
    number = (meta or {}).get("number")
    return f"pr-{number}" if number else ""


def _post_pr_comment(target_dir: Path, repo: str, number: str, body: str) -> bool:
    """Post an issue/PR comment via ``gh api``. Returns True on success.

    Never raises; callers rely on the boolean to decide whether to record
    the queue ack.  Token resolution may return an empty dict, in which
    case ``gh`` falls back to the host's default identity.
    """
    env = os.environ.copy()
    env.update(resolve_github_token_env(target_dir))
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/issues/{number}/comments", "-f", f"body={body}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except Exception as exc:
        logger.warning("gh api comment failed for %s#%s: %s", repo, number, exc)
        return False
    if result.returncode != 0:
        logger.warning(
            "gh api comment exited %s for %s#%s: %s",
            result.returncode,
            repo,
            number,
            (result.stderr or "").strip(),
        )
        return False
    return True


def _find_holder_task(manager: TaskQueueManager, exclusive_key: str, task_id: str) -> str | None:
    """Return the oldest active task holding the same exclusive key (excluding *task_id*).

    ``blocked`` tasks are not holders: they wait on an external condition and may
    never complete, so announcing one as the predecessor promises a start that
    never comes.
    """
    candidates = [
        task
        for task in manager.list_tasks()
        if task.task_id != task_id
        and task.status in ("pending", "in_progress")
        and (task.meta or {}).get("exclusive_key") == exclusive_key
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda task: task.ts)
    return candidates[0].task_id


def _maybe_post_queue_ack(target_dir: Path, task_id: str, meta: dict) -> None:
    """Best-effort PR ack so the requester sees the task was received."""
    if not meta.get("exclusive_key"):
        return
    repo = meta.get("repo")
    number = meta.get("number")
    if not repo or not number:
        return
    if meta.get("queue_ack_posted"):
        return
    manager = TaskQueueManager(target_dir)
    holder_task_id = _find_holder_task(manager, meta["exclusive_key"], task_id)
    if holder_task_id is None:
        return
    # One ack per PR backlog: skip if another queued task on this key already acked.
    if any(
        task.task_id != task_id
        and task.status in ("pending", "in_progress", "blocked")
        and (task.meta or {}).get("exclusive_key") == meta["exclusive_key"]
        and (task.meta or {}).get("queue_ack_posted")
        for task in manager.list_tasks()
    ):
        return
    from core.i18n import t

    body = t("github_gateway.queue_ack", task_id=task_id, holder_task_id=holder_task_id)
    try:
        if _post_pr_comment(target_dir, repo, number, body):
            manager.update_meta(task_id, {"queue_ack_posted": True})
    except Exception as exc:
        logger.warning("queue ack post failed for task %s: %s", task_id, exc)


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

    _maybe_post_queue_ack(target_dir, task_id, task_meta)

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
    if exclusive_key:
        task_desc["exclusive_key"] = exclusive_key
    if model:
        task_desc["model"] = model
    atomic_write_text(
        target_dir / "state" / "pending" / f"{task_id}.json",
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
    )
    return True
