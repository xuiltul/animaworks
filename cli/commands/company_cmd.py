from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for managing companies and their assets."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def register_company_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``animaworks company`` command group."""
    parser = subparsers.add_parser(
        "company",
        help="Create companies, manage memberships, and adopt assets",
        description="Manage company workspaces, anima memberships, and company-owned assets.",
    )
    company_sub = parser.add_subparsers(dest="company_command")

    create = company_sub.add_parser(
        "create",
        help="Create or complete a company workspace",
        description="Create a company workspace, or add any missing scaffold to an existing one.",
    )
    create.add_argument("name", help="Company name ([a-z0-9][a-z0-9_-]*)")
    create.add_argument(
        "--display-name",
        default=None,
        help="Human-readable company name (default: company name)",
    )
    create.set_defaults(func=cmd_company_create)

    list_command = company_sub.add_parser(
        "list",
        help="List companies and anima memberships",
        description="List all companies, their display names and member counts, plus unassigned animas.",
    )
    list_command.set_defaults(func=cmd_company_list)

    assign = company_sub.add_parser(
        "assign",
        help="Assign or unassign animas",
        description="Assign one or more animas to a company, or remove their company assignment.",
    )
    assign.add_argument("anima", nargs="+", help="Anima name(s)")
    assign_mode = assign.add_mutually_exclusive_group(required=True)
    assign_mode.add_argument("--to", metavar="NAME", help="Destination company")
    assign_mode.add_argument("--unassign", action="store_true", help="Remove the company assignment")
    assign.set_defaults(func=cmd_company_assign)

    adopt = company_sub.add_parser(
        "adopt",
        help="Move existing assets into a company workspace",
        description=(
            "Move data-directory assets under a company, first backing them up and normally leaving "
            "relative symbolic links at their old paths."
        ),
    )
    adopt.add_argument("path", nargs="+", type=Path, help="Data-directory-relative or absolute asset path(s)")
    adopt.add_argument("--to", required=True, metavar="NAME", help="Destination company")
    adopt.add_argument(
        "--dest",
        choices=["shared", "knowledge", "skills", "credentials", "."],
        default=None,
        metavar="SUBDIR",
        help="Destination subdirectory (default: infer from each source)",
    )
    adopt.add_argument(
        "--no-symlink",
        action="store_true",
        help="Do not leave a symbolic link at the old path",
    )
    adopt.set_defaults(func=cmd_company_adopt)

    split = company_sub.add_parser(
        "split",
        help="Apply a company split manifest",
        description="Plan or execute company creation, anima assignment, and asset adoption from a manifest.",
    )
    split.add_argument(
        "--manifest",
        required=True,
        type=Path,
        metavar="FILE",
        help="YAML or JSON split manifest",
    )
    split.add_argument("--execute", action="store_true", help="Execute the plan (default: dry-run only)")
    split.set_defaults(func=cmd_company_split)

    export = company_sub.add_parser(
        "export",
        help="Export a company for migration",
        description=(
            "Collect a company's members and assets into a portable migration bundle, "
            "with secrets redacted and remaining migration work documented."
        ),
    )
    export.add_argument("name", help="Company name")
    export.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory (must not already contain files)",
    )
    export.set_defaults(func=cmd_company_export)


def _run_company_action(action: Callable[[], Any]) -> Any:
    """Run a core action and render the shared company error contract."""
    from core.company import CompanyError

    try:
        return action()
    except CompanyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def cmd_company_create(args: argparse.Namespace) -> None:
    """Create or complete a company workspace."""
    from core.company import create_company
    from core.paths import get_data_dir

    created = _run_company_action(
        lambda: create_company(args.name, display_name=args.display_name, data_dir=get_data_dir())
    )
    action = "Created" if created else "Completed"
    print(f"{action} company '{args.name}'.")


def cmd_company_list(args: argparse.Namespace) -> None:
    """Print company membership summaries."""
    from core.company import list_companies
    from core.paths import get_data_dir

    companies, unassigned = _run_company_action(lambda: list_companies(data_dir=get_data_dir()))
    if companies:
        print("NAME\tDISPLAY NAME\tANIMAS")
        for company in companies:
            print(f"{company.name}\t{company.display_name}\t{company.member_count}")
    else:
        print("No companies found.")

    print("Unassigned animas:")
    if unassigned:
        for anima_name in unassigned:
            print(f"  - {anima_name}")
    else:
        print("  (none)")


def cmd_company_assign(args: argparse.Namespace) -> None:
    """Assign or unassign one or more animas."""
    from core.company import assign_animas
    from core.paths import get_data_dir

    lines = _run_company_action(
        lambda: assign_animas(
            args.anima,
            company_name=args.to,
            unassign=bool(args.unassign),
            data_dir=get_data_dir(),
        )
    )
    _print_lines(lines)


def cmd_company_adopt(args: argparse.Namespace) -> None:
    """Adopt assets into a company workspace."""
    from core.company import adopt_assets
    from core.paths import get_data_dir

    results = _run_company_action(
        lambda: adopt_assets(
            args.path,
            company_name=args.to,
            dest=args.dest,
            leave_symlinks=not args.no_symlink,
            data_dir=get_data_dir(),
        )
    )
    for result in results:
        print(f"Moved: {result.source} -> {result.destination}")
        print(f"  Backup: {result.backup_path}")
        if result.symlink_created:
            print(f"  Symlink: {result.source} -> {result.destination}")


def cmd_company_split(args: argparse.Namespace) -> None:
    """Plan or execute a company split manifest."""
    from core.company import CompanyError, SplitExecutionError, split_companies
    from core.paths import get_data_dir

    try:
        lines = split_companies(args.manifest, execute=bool(args.execute), data_dir=get_data_dir())
    except SplitExecutionError as exc:
        _print_lines(exc.completed_lines)
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except CompanyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    _print_lines(lines)


def cmd_company_export(args: argparse.Namespace) -> None:
    """Export one company into a portable migration bundle."""
    from core.company import export_company
    from core.paths import get_data_dir

    result = _run_company_action(lambda: export_company(args.name, args.out, data_dir=get_data_dir()))
    print(f"Exported company '{args.name}' to {result.output_dir}.")
    print(f"Members: {len(result.members)}")
    print(f"Skipped symlinks: {len(result.skipped_symlinks)}")
    print(f"Scan hits: {result.scan_hit_count}")


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)
