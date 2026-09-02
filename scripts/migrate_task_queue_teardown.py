#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Retire blocked/failed task_queue.jsonl entries to pending (A1 task-model teardown).

``blocked``/``failed`` were removed from the valid task status vocabulary
(see docs/plans/20260902_teardown-A1-task-model.md). ``TaskQueueManager``
already reads old blocked/failed rows as ``pending`` at load time, but this
script writes a durable append-only ``pending`` update event for each task
whose *latest* status is still blocked/failed in the raw jsonl, with a note
explaining the retirement so the owning anima can decide whether to proceed
or cancel.

Usage (never run against production without --dry-run first)::

    python scripts/migrate_task_queue_teardown.py --dry-run
    python scripts/migrate_task_queue_teardown.py
    python scripts/migrate_task_queue_teardown.py --animas-dir /path/to/animas
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir

logger = logging.getLogger("animaworks.migrate_task_queue_teardown")

_RETIRED_STATUSES = frozenset({"blocked", "failed"})
_RETIREMENT_NOTE = " [2026-09-02 blocked/failed 制度廃止。進めるか cancelled にするかは自分で判断]"


def _raw_latest_statuses(queue_path: Path) -> dict[str, str]:
    """Replay task_queue.jsonl and return each task_id's latest RAW status.

    Unlike ``TaskQueueManager._load_all``, this does NOT remap blocked/failed
    to pending — it is used to find the entries that need retiring.
    """
    statuses: dict[str, str] = {}
    if not queue_path.exists():
        return statuses
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = raw.get("task_id", "")
        if not task_id or "status" not in raw:
            continue
        statuses[task_id] = raw["status"]
    return statuses


@dataclass(frozen=True)
class AnimaResult:
    """Outcome of planning/migrating a single anima's task queue."""

    name: str
    retired_task_ids: tuple[str, ...]


def plan_anima(anima_dir: Path) -> list[str]:
    """Return task_ids under ``anima_dir`` whose latest raw status is blocked/failed."""
    queue_path = anima_dir / "state" / "task_queue.jsonl"
    statuses = _raw_latest_statuses(queue_path)
    return [tid for tid, status in statuses.items() if status in _RETIRED_STATUSES]


def migrate_anima(anima_dir: Path, *, dry_run: bool) -> AnimaResult:
    """Plan and, unless dry_run, append pending-retirement updates for one anima."""
    task_ids = plan_anima(anima_dir)
    if dry_run or not task_ids:
        return AnimaResult(name=anima_dir.name, retired_task_ids=tuple(task_ids))

    manager = TaskQueueManager(anima_dir)
    for task_id in task_ids:
        entry = manager.get_task_by_id(task_id)
        old_summary = entry.summary if entry is not None else ""
        manager.update_status(task_id, "pending", summary=f"{old_summary}{_RETIREMENT_NOTE}")
        logger.info("Retired blocked/failed task %s (anima=%s) to pending", task_id, anima_dir.name)

    return AnimaResult(name=anima_dir.name, retired_task_ids=tuple(task_ids))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retire blocked/failed task_queue.jsonl entries to pending (A1 task-model teardown).",
    )
    parser.add_argument(
        "--animas-dir",
        type=Path,
        default=None,
        help="Override animas/ directory (default: core.paths.get_animas_dir()).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing (recommended first).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    animas_dir = args.animas_dir if args.animas_dir is not None else get_animas_dir()
    if not animas_dir.is_dir():
        print(f"animas dir not found: {animas_dir}")
        return 1

    mode = "DRY-RUN" if args.dry_run else "EXECUTE"
    print(f"[{mode}] animas_dir={animas_dir}")

    total = 0
    for anima_dir in sorted(p for p in animas_dir.iterdir() if p.is_dir()):
        result = migrate_anima(anima_dir, dry_run=args.dry_run)
        if not result.retired_task_ids:
            continue
        total += len(result.retired_task_ids)
        verb = "Would retire" if args.dry_run else "Retired"
        print(f"{result.name}: {verb} {len(result.retired_task_ids)} task(s): {', '.join(result.retired_task_ids)}")

    print(f"Total: {total} task(s) {'planned' if args.dry_run else 'retired'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
