#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Move anima-named directories out of shared/users/ into a dated backup.

Usage (never run against production from automated tests)::

    python scripts/migrate_shared_users_cleanup.py --dry-run
    python scripts/migrate_shared_users_cleanup.py --extra-names oldbot,legacy

Directories under shared/users/ whose names exactly match the anima roster
(active + on-disk tombstones + --extra-names) are **moved** (not deleted) to
``shared/users_backup_YYYYMMDD/``.
"""

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from core.anima_roster import load_anima_names
from core.paths import get_data_dir

logger = logging.getLogger("animaworks.migrate_shared_users_cleanup")


@dataclass(frozen=True)
class MovePlan:
    """One directory planned for relocation."""

    name: str
    source: Path
    destination: Path


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of a cleanup run."""

    dry_run: bool
    backup_dir: Path
    planned: tuple[MovePlan, ...]
    moved: tuple[MovePlan, ...]

    def as_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "backup_dir": str(self.backup_dir),
            "planned": [p.name for p in self.planned],
            "moved": [p.name for p in self.moved],
        }


def build_match_names(
    data_dir: Path,
    *,
    extra_names: list[str] | tuple[str, ...] | None = None,
) -> set[str]:
    """Build the set of directory names that should be moved out of shared/users."""
    names = load_anima_names(data_dir)
    if extra_names:
        for raw in extra_names:
            name = raw.strip()
            if name:
                names.add(name)
    return names


def plan_moves(
    data_dir: Path,
    *,
    match_names: set[str],
    backup_date: date | None = None,
) -> tuple[Path, list[MovePlan]]:
    """Plan moves of matching directories under shared/users/."""
    users_dir = data_dir / "shared" / "users"
    stamp = (backup_date or date.today()).strftime("%Y%m%d")
    backup_dir = data_dir / "shared" / f"users_backup_{stamp}"
    plans: list[MovePlan] = []

    if not users_dir.is_dir():
        return backup_dir, plans

    for entry in sorted(users_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        if entry.name not in match_names:
            continue
        dest = backup_dir / entry.name
        plans.append(MovePlan(name=entry.name, source=entry, destination=dest))
    return backup_dir, plans


def execute_cleanup(
    data_dir: Path,
    *,
    dry_run: bool = True,
    extra_names: list[str] | tuple[str, ...] | None = None,
    backup_date: date | None = None,
) -> CleanupResult:
    """Plan and optionally execute relocation of anima-named user dirs."""
    match_names = build_match_names(data_dir, extra_names=extra_names)
    backup_dir, plans = plan_moves(data_dir, match_names=match_names, backup_date=backup_date)
    moved: list[MovePlan] = []

    if not plans:
        return CleanupResult(
            dry_run=dry_run,
            backup_dir=backup_dir,
            planned=tuple(plans),
            moved=tuple(moved),
        )

    if dry_run:
        return CleanupResult(
            dry_run=True,
            backup_dir=backup_dir,
            planned=tuple(plans),
            moved=tuple(moved),
        )

    backup_dir.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        if plan.destination.exists():
            raise FileExistsError(f"Backup destination already exists: {plan.destination}")
        shutil.move(str(plan.source), str(plan.destination))
        moved.append(plan)
        logger.info("Moved %s -> %s", plan.source, plan.destination)

    return CleanupResult(
        dry_run=False,
        backup_dir=backup_dir,
        planned=tuple(plans),
        moved=tuple(moved),
    )


def _parse_extra_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            parts.append(name)
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move anima-named directories from shared/users/ to users_backup_YYYYMMDD/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned moves without modifying the filesystem (recommended first).",
    )
    parser.add_argument(
        "--extra-names",
        default="",
        help="Comma-separated retired anima names to treat as match targets.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory (default: ANIMAWORKS_DATA_DIR or ~/.animaworks).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = args.data_dir if args.data_dir is not None else get_data_dir()
    extra = _parse_extra_names(args.extra_names)
    result = execute_cleanup(
        data_dir,
        dry_run=bool(args.dry_run),
        extra_names=extra,
    )

    mode = "DRY-RUN" if result.dry_run else "EXECUTE"
    print(f"[{mode}] data_dir={data_dir}")
    print(f"backup_dir={result.backup_dir}")
    if not result.planned:
        print("No matching directories under shared/users/.")
        return 0

    print(f"Planned moves ({len(result.planned)}):")
    for plan in result.planned:
        print(f"  {plan.source} -> {plan.destination}")

    if result.dry_run:
        print("No files moved (dry-run).")
    else:
        print(f"Moved {len(result.moved)} directories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
