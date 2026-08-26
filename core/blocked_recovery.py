from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Heartbeat-driven recovery for blocked TaskExec tasks."""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from core.i18n import t
from core.memory._io import atomic_write_text
from core.memory.task_queue import TaskQueueManager
from core.schemas import TaskEntry
from core.time_utils import ensure_aware, now_iso, now_local

logger = logging.getLogger("animaworks.blocked_recovery")


def regenerate_pending_json(
    anima_dir: Path,
    anima_name: str,
    entry: TaskEntry,
    *,
    description_suffix: str = "",
) -> bool:
    """Publish a task queue entry for PendingTaskExecutor without retry accounting."""
    pending_dir = anima_dir / "state" / "pending"
    task_file = f"{entry.task_id}.json"
    if (pending_dir / task_file).exists() or (pending_dir / "processing" / task_file).exists():
        return True

    task_desc_meta = entry.meta.get("task_desc", {}) or {}
    description = task_desc_meta.get("description", entry.original_instruction)
    if description_suffix:
        description = f"{description}\n\n{description_suffix}"
    # Restore the per-task model override so blocked-recovery re-execution keeps it.
    # SSoT is the pending task_desc (task_desc_meta.model) with a fallback to the
    # queue entry meta (entry.meta.model) for direct-dispatch submissions.
    model = entry.meta.get("model") or task_desc_meta.get("model")
    task_desc = {
        "task_type": "llm",
        "task_id": entry.task_id,
        "batch_id": entry.meta.get("batch_id", ""),
        "title": task_desc_meta.get("title", entry.summary),
        "description": description,
        "parallel": False,
        "depends_on": [],
        "context": task_desc_meta.get("context", ""),
        "acceptance_criteria": task_desc_meta.get("acceptance_criteria", []),
        "constraints": task_desc_meta.get("constraints", []),
        "file_paths": task_desc_meta.get("file_paths", []),
        "submitted_by": anima_name,
        "submitted_at": now_iso(),
        "reply_to": task_desc_meta.get("reply_to", anima_name),
        "working_directory": task_desc_meta.get("working_directory", ""),
        "exclusive_key": task_desc_meta.get("exclusive_key") or entry.meta.get("exclusive_key", ""),
        "model": model if isinstance(model, str) else "",
    }
    atomic_write_text(
        pending_dir / task_file,
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
    )
    return True


def _age_hours(entry: TaskEntry) -> float | None:
    value = entry.meta.get("blocked_at") or entry.updated_at
    try:
        blocked_at = ensure_aware(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError):
        return None
    return max(0.0, (now_local() - blocked_at).total_seconds() / 3600)


def _blocked_since(entry: TaskEntry) -> datetime:
    """Return a stable timestamp for oldest-first recovery."""
    value = entry.meta.get("blocked_at") or entry.updated_at
    try:
        return ensure_aware(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError):
        return datetime.max.replace(tzinfo=now_local().tzinfo)


def _count(meta: dict, key: str) -> int:
    value = meta.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _record_check_failure(manager: TaskQueueManager, entry: TaskEntry) -> None:
    manager.update_meta(
        entry.task_id,
        {"unblock_check_failures": _count(entry.meta, "unblock_check_failures") + 1},
    )


def _alert_manual_intervention_required(
    manager: TaskQueueManager,
    anima_dir: Path,
    anima_name: str,
    entry: TaskEntry,
) -> None:
    if entry.meta.get("blocked_recovery_alerted"):
        return

    from core.config.models import read_anima_supervisor
    from core.delegation_recovery import _add_alert_task

    supervisor = read_anima_supervisor(anima_dir)
    if not supervisor:
        return
    supervisor_dir = anima_dir.parent / supervisor
    if not supervisor_dir.is_dir():
        return
    _add_alert_task(
        supervisor_dir,
        kind="blocked_task_manual_intervention_required",
        target_name=anima_name,
        delegated_task_id=entry.task_id,
        summary=f"Blocked task needs intervention: {anima_name}/{entry.task_id}",
        instruction=t(
            "blocked_recovery.manual_intervention_instruction",
            task_id=entry.task_id,
            anima_name=anima_name,
            original_instruction=entry.original_instruction,
        ),
        extra_meta={"blocked_task_id": entry.task_id},
    )
    manager.update_meta(entry.task_id, {"blocked_recovery_alerted": True})


def revalidate_blocked_tasks(anima_dir: Path, anima_name: str) -> list[str]:
    """Revalidate blocked tasks and return task IDs changed back to pending."""
    from core.config.models import load_config

    config = load_config().background_task
    if not config.blocked_recovery_enabled:
        return []

    manager = TaskQueueManager(anima_dir)
    unblocked: list[str] = []
    entries = sorted(
        manager.list_tasks(status="blocked"),
        key=lambda entry: (_blocked_since(entry), entry.task_id),
    )
    for entry in entries:
        if len(unblocked) >= config.blocked_reprobe_batch_limit:
            break
        try:
            try:
                from core.taskboard.attention_resolver import resolver_for_anima_dir

                decision = resolver_for_anima_dir(anima_dir).should_execute(
                    anima_name, entry.task_id, queue_status="pending"
                )
                if not decision.executable:
                    continue
            except Exception:
                logger.warning(
                    "TaskBoard recovery gate unavailable for task %s; failing open",
                    entry.task_id,
                    exc_info=True,
                )

            check = entry.meta.get("unblock_check")
            has_check = isinstance(check, str) and bool(check.strip())
            if has_check:
                env = {
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": os.environ.get("HOME", ""),
                    "ANIMAWORKS_ANIMA_DIR": str(anima_dir),
                }
                try:
                    result = subprocess.run(
                        [
                            "bwrap",
                            "--ro-bind",
                            "/",
                            "/",
                            "--dev",
                            "/dev",
                            "--proc",
                            "/proc",
                            "--tmpfs",
                            "/tmp",
                            "--unshare-net",
                            "--die-with-parent",
                            "--",
                            "/bin/sh",
                            "-c",
                            check,
                        ],
                        cwd=anima_dir,
                        env=env,
                        timeout=config.blocked_check_timeout_seconds,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except OSError:
                    logger.warning(
                        "unblock_check sandbox unavailable for task %s; failing closed",
                        entry.task_id,
                        exc_info=True,
                    )
                    _record_check_failure(manager, entry)
                    continue
                except subprocess.TimeoutExpired:
                    _record_check_failure(manager, entry)
                    continue
                if result.returncode != 0:
                    _record_check_failure(manager, entry)
                    continue
                suffix = ""
            else:
                # checkless: fail closed by default (no automatic pending requeue).
                age_hours = _age_hours(entry)
                if age_hours is None or age_hours < config.blocked_reprobe_after_hours:
                    continue
                if not config.blocked_checkless_reprobe_enabled:
                    _alert_manual_intervention_required(manager, anima_dir, anima_name, entry)
                    continue
                # Legacy time-based reprobe (opt-in via blocked_checkless_reprobe_enabled).
                reprobes = _count(entry.meta, "blocked_reprobe_count")
                if reprobes >= config.blocked_max_reprobes:
                    _alert_manual_intervention_required(manager, anima_dir, anima_name, entry)
                    continue
                manager.update_meta(entry.task_id, {"blocked_reprobe_count": reprobes + 1})
                suffix = t("blocked_recovery.reprobe_instruction")

            pending = manager.update_status(
                entry.task_id,
                "pending",
                summary=("auto-unblocked: check passed" if has_check else "auto-unblocked: scheduled reprobe"),
            )
            if pending is None:
                continue
            try:
                regenerate_pending_json(anima_dir, anima_name, pending, description_suffix=suffix)
            except Exception:
                manager.update_status(entry.task_id, "blocked", summary=entry.summary)
                raise
            unblocked.append(entry.task_id)
            from core.memory.activity import ActivityLogger

            ActivityLogger(anima_dir).log(
                "blocked_recovery",
                summary=("Unblock check passed" if has_check else "Scheduled blocked task reprobe"),
                meta={
                    "task_id": entry.task_id,
                    "method": "check" if has_check else "reprobe",
                },
                safe=True,
            )
        except Exception:
            logger.warning("Failed to revalidate blocked task %s", entry.task_id, exc_info=True)
    return unblocked
