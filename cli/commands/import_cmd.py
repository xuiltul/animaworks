from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command: ``animaworks import hermes/openclaw``."""

import argparse
import json
import sys
from pathlib import Path

from core.paths import get_data_dir
from core.skills.migration import HermesImportOptions, OpenClawImportOptions, import_hermes, import_openclaw


def register_import_command(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("import", help="Import Hermes or OpenClaw data into AnimaWorks")
    sub = parser.add_subparsers(dest="import_source")

    hermes = sub.add_parser("hermes", help="Import Hermes Agent data")
    _common_args(hermes)
    hermes.add_argument("--common-skills", action="store_true", help="Import skills into common_skills/community")
    hermes.add_argument(
        "--target-anima", default=None, help="Target anima for personal skills, usage, tasks, and drafts"
    )
    hermes.set_defaults(func=cmd_import_hermes)

    openclaw = sub.add_parser("openclaw", help="Import OpenClaw data")
    _common_args(openclaw)
    openclaw.add_argument("--target-anima", required=True, help="Target anima for generated drafts")
    openclaw.set_defaults(func=cmd_import_openclaw)


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", required=True, help="Source directory, e.g. ~/.hermes or ~/.openclaw")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview without changing the runtime filesystem")
    mode.add_argument("--apply", action="store_true", help="Apply importable items and write migration report")
    parser.add_argument(
        "--replace", action="store_true", help="Replace existing generated targets after backup manifest"
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output JSON instead of Markdown")


def cmd_import_hermes(args: argparse.Namespace) -> None:
    try:
        if not args.common_skills and not args.target_anima:
            raise ValueError("Hermes import requires --target-anima unless --common-skills is set")
        report = import_hermes(
            HermesImportOptions(
                source_path=Path(args.path),
                data_dir=get_data_dir(),
                target_anima=args.target_anima,
                common_skills=bool(args.common_skills),
                apply=bool(args.apply),
                replace=bool(args.replace),
            )
        )
        _print_report(report, json_output=bool(args.json_output))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc


def cmd_import_openclaw(args: argparse.Namespace) -> None:
    try:
        report = import_openclaw(
            OpenClawImportOptions(
                source_path=Path(args.path),
                data_dir=get_data_dir(),
                target_anima=args.target_anima,
                apply=bool(args.apply),
                replace=bool(args.replace),
            )
        )
        _print_report(report, json_output=bool(args.json_output))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc


def _print_report(report, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(report.to_markdown())
