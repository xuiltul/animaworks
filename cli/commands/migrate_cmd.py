from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command: ``animaworks migrate``."""

import argparse
import sys
from typing import TYPE_CHECKING

from core.i18n import t

if TYPE_CHECKING:
    from core.migrations.registry import MigrationRunner


# ── Command implementation ──────────────────────────────────


def cmd_migrate(args: argparse.Namespace) -> None:
    """Run pending migrations on the runtime data directory."""
    from core.migrations.registry import MigrationRunner
    from core.migrations.steps import register_all_steps
    from core.paths import get_data_dir

    data_dir = get_data_dir()
    config_json = data_dir / "config.json"
    if not config_json.exists():
        print(t("migrate.no_runtime", data_dir=data_dir))
        sys.exit(1)

    dry_run: bool = getattr(args, "dry_run", False)
    verbose: bool = getattr(args, "verbose", False)
    force: bool = getattr(args, "force", False)
    list_mode: bool = getattr(args, "list", False)
    rollback: bool = getattr(args, "rollback_standard_files", False)
    anima_name: str | None = getattr(args, "anima", None)

    if rollback:
        _run_rollback(data_dir, anima_name=anima_name, dry_run=dry_run)
        return

    runner = MigrationRunner(data_dir)
    register_all_steps(runner)

    # --list: show step inventory and exit
    if list_mode:
        _print_list(runner)
        return

    # Warn if server is running
    server_pid = data_dir / "server.pid"
    if server_pid.exists():
        print(t("migrate.server_warning"))

    if dry_run:
        print(t("migrate.dry_run_header"))
        print()

    # Execute
    report = runner.run_all(dry_run=dry_run, verbose=verbose, force=force)

    # Print results
    for step, result in report.steps:
        line = t("migrate.step_result", name=step.name, changed=result.changed, skipped=result.skipped)
        if result.error:
            line += f"  ERROR: {result.error}"
        print(line)
        if verbose and result.details:
            for d in result.details:
                print(f"  {d}")

    print()
    print(
        t(
            "migrate.complete",
            changed=report.total_changed,
            skipped=report.total_skipped,
        )
    )
    if report.errors:
        print(t("migrate.error_summary", count=len(report.errors)))
        for e in report.errors:
            print(f"  - {e}")


def _run_rollback(data_dir, *, anima_name: str | None, dry_run: bool) -> None:
    """Roll back the standard workspace-file migration."""
    from core.migrations.standard_files_rollback import rollback_standard_files

    if dry_run:
        print(t("migrate.rollback_dry_run_header"))
        print()
    result = rollback_standard_files(data_dir, anima_name=anima_name, dry_run=dry_run)
    if not result.animas:
        print(t("migrate.rollback_no_archives", anima=anima_name or "all"))
        return

    if dry_run:
        for detail in result.details:
            print(f"  {detail}")
        print()
    print(
        t(
            "migrate.rollback_complete",
            animas=len(result.animas),
            restored=len(result.restored_files),
            moved=len(result.moved_new_files),
        )
    )
    if result.tracker_entry_removed:
        print(t("migrate.rollback_state_removed"))
    for warning in result.warnings:
        print(f"  WARNING: {warning}")
    print(t("migrate.rollback_old_version_note"))


def _print_list(runner: MigrationRunner) -> None:
    """Print migration step inventory."""
    print(t("migrate.list_header"))
    print()
    steps = runner.list_steps()
    current_cat = ""
    for s in steps:
        if s["category"] != current_cat:
            current_cat = s["category"]
            print(f"  [{current_cat}]")
        status = "✓" if s["applied"] else "·"
        at = f"  ({s['applied_at'][:19]})" if s["applied_at"] else ""
        print(f"    {status} {s['name']}{at}")
    print()
    applied = sum(1 for s in steps if s["applied"])
    print(f"  {applied}/{len(steps)} applied")


# ── Registration ────────────────────────────────────────────


def register_migrate_command(sub: argparse._SubParsersAction) -> None:
    """Register ``animaworks migrate`` subcommand."""
    p = sub.add_parser(
        "migrate",
        help=t("migrate.help", fallback="Run runtime data migrations"),
    )
    p.add_argument("--dry-run", action="store_true", help="Preview changes without modifying anything")
    p.add_argument("--verbose", action="store_true", help="Show detailed file-level changes")
    p.add_argument("--list", action="store_true", help="List all migration steps and their status")
    p.add_argument("--force", action="store_true", help="Re-apply all migrations regardless of state")
    p.add_argument(
        "--rollback-standard-files",
        action="store_true",
        help="Restore legacy workspace files from the latest standard-files archive",
    )
    p.add_argument("--anima", metavar="NAME", help="Limit rollback to one anima")
    p.set_defaults(func=cmd_migrate)
