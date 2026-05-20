from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""RAG repair command."""

import argparse
import os
import sys


def setup_repair_rag_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the top-level repair-rag command."""
    parser = subparsers.add_parser(
        "repair-rag",
        help="Quarantine and rebuild RAG vectordb data",
        description="Stop target animas before running this command in production.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--anima", help="Anima name to repair")
    target.add_argument("--all", action="store_true", help="Repair all enabled animas")
    target.add_argument("--suspect-only", action="store_true", help="Repair animas with recent RAG corruption evidence")
    target.add_argument("--list-suspects", action="store_true", help="List suspected corrupt RAG DBs without repairing")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Required confirmation for destructive quarantine and full rebuild",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Reindex shared common_knowledge and common_skills into this anima DB",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=None,
        help="Lookback window for --suspect-only/--list-suspects (default: repair config window)",
    )
    parser.add_argument(
        "--reason",
        default="manual_repair_rag_cli",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(func=repair_rag_command)


def repair_rag_command(args: argparse.Namespace) -> None:
    """Run synchronous RAG repair."""
    list_suspects = bool(getattr(args, "list_suspects", False))
    if not list_suspects and not args.full:
        print("repair-rag requires --full for destructive quarantine and rebuild", file=sys.stderr)
        raise SystemExit(2)

    from core.memory.rag.repair import get_repair_service

    service = get_repair_service()
    suspect_only = bool(getattr(args, "suspect_only", False))
    all_animas = bool(getattr(args, "all", False))
    anima = getattr(args, "anima", None)
    window_minutes = getattr(args, "window_minutes", None)

    if list_suspects:
        suspects = service.discover_suspect_animas(window_minutes=window_minutes)
        if suspects:
            print("RAG repair suspects:")
            for name in suspects:
                print(f"  {name}")
        else:
            print("No RAG repair suspects found.")
        return

    if not anima and not all_animas and not suspect_only:
        print("repair-rag requires one of --anima, --all, --suspect-only, or --list-suspects", file=sys.stderr)
        raise SystemExit(2)

    temp_worker = None
    if not os.environ.get("ANIMAWORKS_VECTOR_URL"):
        from core.memory.rag.vector_worker_client import start_temporary_vector_worker

        temp_worker = start_temporary_vector_worker()

    reason = str(getattr(args, "reason", "manual_repair_rag_cli"))
    try:
        if anima:
            result = service.repair_anima_if_allowed(
                anima,
                reason=reason,
                collection=None,
                source="cli",
                include_shared=bool(args.shared),
            )
            _print_single_result(result)
            return

        if all_animas:
            targets = service.list_repairable_animas()
        else:
            targets = service.discover_suspect_animas(window_minutes=window_minutes)

        if not targets:
            print("No RAG repair targets found.")
            return

        results = service.repair_animas_if_allowed(
            targets,
            reason=reason,
            source="cli",
            include_shared=bool(args.shared),
        )
        failed = False
        for result in results.values():
            failed = failed or not result.ok
            _print_result_line(result)
        if failed:
            raise SystemExit(1)
    finally:
        if temp_worker is not None:
            temp_worker.stop()


def _print_single_result(result) -> None:
    if result.ok:
        print(
            "RAG repair succeeded: "
            f"anima={result.anima_name} chunks={result.chunks_indexed} quarantine={result.quarantine_path}"
        )
        return

    print(
        "RAG repair failed: "
        f"anima={result.anima_name} status={result.status} stage={result.stage} error={result.error}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _print_result_line(result) -> None:
    if result.ok:
        print(
            "RAG repair succeeded: "
            f"anima={result.anima_name} chunks={result.chunks_indexed} quarantine={result.quarantine_path}"
        )
    else:
        print(
            "RAG repair failed: "
            f"anima={result.anima_name} status={result.status} stage={result.stage} error={result.error}",
            file=sys.stderr,
        )
