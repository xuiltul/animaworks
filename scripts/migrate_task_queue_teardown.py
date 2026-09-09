#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only legacy blocked/failed task diagnostics.

The mutation mode is retired: canonical migration requires an offline runtime,
a durable claim gate and an explicit backup via ``animaworks task-store migrate``.
This older script never rewrites legacy evidence or the canonical database.

Usage::

    python scripts/migrate_task_queue_teardown.py --dry-run
    python scripts/migrate_task_queue_teardown.py --dry-run --animas-dir /path/to/animas
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from core.i18n import t
from core.paths import get_animas_dir

logger = logging.getLogger("animaworks.migrate_task_queue_teardown")

_RETIRED_STATUSES = frozenset({"blocked", "failed"})


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
        if not isinstance(raw, dict):
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
    """Inspect old statuses; reject the retired write path before accessing state."""
    if not dry_run:
        raise RuntimeError(t("task_store.teardown_retired"))
    task_ids = plan_anima(anima_dir)
    return AnimaResult(name=anima_dir.name, retired_task_ids=tuple(task_ids))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=t("task_store.teardown_retired"),
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
    if not args.dry_run:
        print(t("task_store.teardown_retired"), file=sys.stderr)
        return 2

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
