from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command: ``animaworks anima export``."""

import argparse
import sys
from pathlib import Path

from core.i18n import t


def cmd_anima_export(args: argparse.Namespace) -> None:
    """Export one Anima workspace for another agent runtime."""
    from core.anima.export import export_anima
    from core.paths import get_data_dir

    try:
        result = export_anima(
            get_data_dir(),
            args.anima,
            args.format,
            Path(args.out).expanduser().resolve(),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(t("cli.export_failed", error=exc))
        sys.exit(1)
    print(
        t(
            "cli.export_complete",
            anima=result.anima,
            format=result.format,
            out=result.out_dir,
            count=len(result.files),
        )
    )
    for warning in result.warnings:
        print(f"  WARNING: {warning}")


def register_anima_export_command(anima_sub: argparse._SubParsersAction) -> None:
    """Register the ``anima export`` subcommand."""
    p = anima_sub.add_parser("export", help=t("cli.export_help"))
    p.add_argument("anima", help="Anima name to export")
    p.add_argument(
        "--format",
        required=True,
        choices=["openclaw", "hermes"],
        help="Target workspace format",
    )
    p.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory",
    )
    p.set_defaults(func=_lazy_anima_export)


def _lazy_anima_export(args: argparse.Namespace) -> None:
    cmd_anima_export(args)


__all__ = ["cmd_anima_export", "register_anima_export_command"]
