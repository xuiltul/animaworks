# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Orphan task reaper.

Detects ledger rows (task_queue.jsonl) whose descriptor file is missing and
marks them ``cancelled`` so they stop blocking the owner's real work, then
notifies the owning anima once per sweep.

An orphan is a task that is only present in the JSONL ledger but has no
executable descriptor under ``state/pending/``.  The only thing that ever
runs a task is its descriptor, and the descriptor-regeneration path was
removed, so such a task will never execute.  This module sweeps those up.

Deliberate policy (2026-09-04): cancelled orphans are **not** re-queued to
``pending`` and their descriptor is **not** regenerated.  Re-running is a
deliberate decision the owner anima makes via ``submit_tasks``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from core.memory.task_queue import TaskQueueManager
from core.schemas import TaskEntry
from core.time_utils import ensure_aware, now_iso, now_local

logger = logging.getLogger(__name__)

# A task must be older than this before it can be reaped (default 30 min).
_ORPHAN_GRACE_SECONDS = 1800
# Run a sweep at most once per this interval (default 5 min).
_ORPHAN_SWEEP_INTERVAL_SECONDS = 300
# Upper bound of tasks processed in a single sweep.
_ORPHAN_REAP_MAX_PER_SWEEP = 50

# Env var that overrides the grace period (seconds). Read at call time.
_GRACE_ENV = "ANIMAWORKS_ORPHAN_GRACE_SECONDS"

# Target cap for the notification (entries listed in the message body).
_NOTIFY_MAX_ENTRIES = 20


def _effective_grace_seconds(grace_seconds: int | None) -> int:
    """Resolve the grace period, honoring the env override.

    Reads the environment at call time so tests can monkeypatch it without
    re-importing the module.
    """
    if grace_seconds is not None:
        return grace_seconds
    raw = os.environ.get(_GRACE_ENV)
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid %s=%r; using default %d",
                _GRACE_ENV,
                raw,
                _ORPHAN_GRACE_SECONDS,
            )
    return _ORPHAN_GRACE_SECONDS


def _task_timestamp(entry: TaskEntry) -> datetime | None:
    """Return the task's best timestamp for age checks, or None.

    Prefers ``updated_at``; falls back to ``ts`` when ``updated_at`` is
    missing or unparseable.  Returns None when neither is usable (the row is
    then skipped rather than reaped, errring on the safe side).
    """
    for field in ("updated_at", "ts"):
        raw = getattr(entry, field, None)
        if not raw:
            continue
        try:
            return ensure_aware(datetime.fromisoformat(raw))
        except (ValueError, TypeError):
            continue
    return None


def find_orphan_tasks(
    anima_dir: Path,
    *,
    now: datetime,
    active_task_ids: set[str],
    grace_seconds: int,
) -> list[TaskEntry]:
    """Return reaper candidates: pending/in_progress ledger rows with no descriptor.

    A row qualifies only when **all** of the following hold:

    - status is ``pending`` or ``in_progress`` (never ``delegated`` —
      delegated rows legitimately have no local descriptor)
    - no descriptor file exists for the task id (``descriptor_exists`` is False)
    - its last-updated time is older than ``now - grace_seconds``
    - its task id is not in ``active_task_ids`` (still being handled)

    Returns entries ordered by age (oldest first).  The caller caps the
    batch size.
    """
    manager = TaskQueueManager(anima_dir)
    cutoff = now - timedelta(seconds=grace_seconds)

    # Scan the descriptor tree exactly once, then use O(1) membership checks.
    # Avoids re-walking state/pending/ (rglob) once per candidate row.
    from core.memory import task_queue as tq

    descriptor_ids = tq._descriptor_ids(anima_dir)

    candidates: list[tuple[datetime, TaskEntry]] = []
    for entry in manager.list_tasks():
        if entry.status not in ("pending", "in_progress"):
            continue
        if entry.task_id in active_task_ids:
            continue
        if entry.task_id in descriptor_ids:
            continue
        ts = _task_timestamp(entry)
        if ts is None or ts >= cutoff:
            continue
        candidates.append((ts, entry))

    candidates.sort(key=lambda pair: pair[0])
    return [entry for _ts, entry in candidates]


def reap_orphan_tasks(
    anima_dir: Path,
    *,
    active_task_ids: set[str],
    grace_seconds: int | None = None,
    max_per_sweep: int = _ORPHAN_REAP_MAX_PER_SWEEP,
) -> list[TaskEntry]:
    """Mark up to ``max_per_sweep`` orphan tasks ``cancelled``.

    Applies, in order, to each candidate returned by ``find_orphan_tasks``:

    1. ``update_meta`` records ``orphan_reason`` and ``reaped_at``.
    2. ``update_status`` flips the row to ``cancelled`` **without touching
       its summary** — the title must be preserved (past bugs destroyed
       ledger readability by overwriting it).

    A failure on one task is logged and skipped; the rest of the sweep
    continues.  Returns the list of reaped ``TaskEntry`` (with the original
    summary preserved).
    """
    if grace_seconds is None:
        grace_seconds = _effective_grace_seconds(None)
    now = now_local()
    orphans = find_orphan_tasks(
        anima_dir,
        now=now,
        active_task_ids=active_task_ids,
        grace_seconds=grace_seconds,
    )

    manager = TaskQueueManager(anima_dir)
    reaped: list[TaskEntry] = []
    for entry in orphans[:max_per_sweep]:
        try:
            manager.update_meta(
                entry.task_id,
                {"orphan_reason": "descriptor missing", "reaped_at": now_iso()},
            )
            # Intentionally no summary argument: the task's title must live on.
            manager.update_status(entry.task_id, "cancelled")
            reaped.append(entry)
        except Exception:
            logger.warning(
                "Failed to reap orphan task %s; skipping (rest of sweep continues)",
                entry.task_id,
                exc_info=True,
            )
    return reaped


def _format_notify_body(reaped: list[TaskEntry]) -> str:
    """Build the notification message body for a single sweep."""
    lines: list[str] = [
        "次のタスクは実行ファイル（descriptor）が無く、実行されない状態のまま"
        "30 分以上経過したため cancelled にしました。",
        "",
    ]
    for entry in reaped[:_NOTIFY_MAX_ENTRIES]:
        summary = (entry.summary or "").strip()[:100]
        lines.append(f"- {entry.task_id}: {summary}")
    over = len(reaped) - _NOTIFY_MAX_ENTRIES
    if over > 0:
        lines.append(f"ほか {over} 件")
    lines.extend(
        [
            "",
            "まだ必要なタスクは、同じ task_id・元の指示・元の workspace で "
            "`submit_tasks` に投入し直してください。`update_task` は記録であって"
            "実行ではありません。",
        ]
    )
    return "\n".join(lines)


def notify_reaped(anima_name: str, reaped: list[TaskEntry]) -> None:
    """Send a single notification about reaped tasks to the owning anima.

    Only sends when ``reaped`` is non-empty, and only one message per sweep.
    ``source="system"`` is required: the default ``source="anima"`` would
    make the ``system-ops`` sender be quarantined because it is not a
    configured anima name (past messages were lost that way).
    """
    if not reaped:
        return
    from core.messenger import Messenger
    from core.paths import get_shared_dir

    Messenger(get_shared_dir(), "system-ops").send(
        to=anima_name,
        content=_format_notify_body(reaped),
        intent="report",
        source="system",
    )
