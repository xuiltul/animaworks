#!/usr/bin/env python3
"""CLI entry point for the unattended CI auto-fix loop v0."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from collections.abc import Sequence
from pathlib import Path

from swe.ci_autofix_loop import CIAutofixLoop
from swe.ci_autofix_types import (
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_MAX_ITER,
    DEFAULT_QUALITY_COMMANDS,
    AutofixConfig,
    AutofixOutcome,
    CIRun,
    CommandResult,
    CommandRunner,
    GateResult,
    GitHubActionsClient,
    LoopStatus,
    QualityGate,
    SubprocessCommandRunner,
    parse_latest_run,
)

__all__ = [
    "AutofixConfig",
    "AutofixOutcome",
    "CIAutofixLoop",
    "CIRun",
    "CommandResult",
    "CommandRunner",
    "GateResult",
    "GitHubActionsClient",
    "LoopStatus",
    "QualityGate",
    "SubprocessCommandRunner",
    "parse_latest_run",
]


def _split_command(value: str) -> tuple[str, ...]:
    command = tuple(shlex.split(value))
    if not command:
        raise argparse.ArgumentTypeError("command must not be empty")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AnimaWorks CI auto-fix loop v0.")
    parser.add_argument("--mode", choices=["ci", "local"], default="ci")
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--branch", default=None)
    parser.add_argument("--repo", default=None, help="GitHub repo in owner/name form for gh -R.")
    parser.add_argument("--workflow", default=None, help="Optional GitHub Actions workflow filter.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--fix-command", type=_split_command, default=None)
    parser.add_argument("--agent", dest="agent_name", default="swe-architect")
    parser.add_argument("--review-command", type=_split_command, default=None)
    parser.add_argument("--review-agent", dest="review_agent_name", default="swe-reviewer")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument(
        "--gate-command",
        action="append",
        type=_split_command,
        default=None,
        help="Override quality gate command. May be repeated.",
    )
    parser.add_argument("--push", action="store_true", help="Push the successful repair commit.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow starting from a dirty worktree.")
    parser.add_argument("--allow-stale-run", action="store_false", dest="match_head_sha")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument(
        "--escalation-command",
        type=_split_command,
        default=("animaworks-tool", "call_human"),
        help="Command prefix used for escalation. Subject/body/priority are appended.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_COMMAND_TIMEOUT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable outcome JSON.")
    return parser


def config_from_args(args: argparse.Namespace) -> AutofixConfig:
    quality_commands = tuple(args.gate_command) if args.gate_command else DEFAULT_QUALITY_COMMANDS
    return AutofixConfig(
        repo_dir=args.repo_dir,
        mode=args.mode,
        branch=args.branch,
        repo=args.repo,
        workflow=args.workflow,
        max_iter=args.max_iter,
        quality_commands=quality_commands,
        fix_command=args.fix_command,
        agent_name=args.agent_name,
        review_command=args.review_command,
        review_agent_name=args.review_agent_name,
        skip_review=args.skip_review,
        push=args.push,
        allow_dirty=args.allow_dirty,
        match_head_sha=args.match_head_sha,
        result_dir=args.result_dir,
        escalation_command=args.escalation_command,
        command_timeout=args.timeout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    parser = build_parser()
    args = parser.parse_args(argv)
    outcome = CIAutofixLoop(config_from_args(args)).run()

    if args.json:
        print(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"status: {outcome.status.value}")
        if outcome.message:
            print(f"message: {outcome.message}")
        if outcome.commit_sha:
            print(f"commit: {outcome.commit_sha}")
        if outcome.escalation_path:
            print(f"escalation: {outcome.escalation_path}")

    if outcome.status in {LoopStatus.SUCCESS, LoopStatus.NO_FAILURE, LoopStatus.PENDING}:
        return 0
    if outcome.status == LoopStatus.DIRTY_WORKTREE:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
