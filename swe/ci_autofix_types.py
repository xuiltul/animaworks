"""Shared types and command adapters for the CI auto-fix loop."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITER = 3
DEFAULT_COMMAND_TIMEOUT = 900
DEFAULT_LOG_LIMIT = 12000
DEFAULT_QUALITY_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check", "core/", "cli/", "server/"),
    ("ruff", "format", "--check", "core/", "cli/", "server/"),
    ("pytest", "tests/unit", "-m", "not live and not azure and not ollama"),
)


class LoopStatus(StrEnum):
    """Final status values for one auto-fix run."""

    SUCCESS = "success"
    NO_FAILURE = "no_failure"
    PENDING = "pending"
    EXHAUSTED = "exhausted"
    DIRTY_WORKTREE = "dirty_worktree"
    FAILED = "failed"


@dataclass
class AutofixConfig:
    """Configuration for one CI auto-fix run."""

    repo_dir: Path
    mode: str = "ci"
    branch: str | None = None
    repo: str | None = None
    workflow: str | None = None
    max_iter: int = DEFAULT_MAX_ITER
    quality_commands: tuple[tuple[str, ...], ...] = DEFAULT_QUALITY_COMMANDS
    fix_command: tuple[str, ...] | None = None
    agent_name: str = "swe-architect"
    review_command: tuple[str, ...] | None = None
    review_agent_name: str = "swe-reviewer"
    skip_review: bool = False
    push: bool = False
    allow_dirty: bool = False
    match_head_sha: bool = True
    result_dir: Path | None = None
    escalation_command: tuple[str, ...] = ("animaworks-tool", "call_human")
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    log_limit: int = DEFAULT_LOG_LIMIT

    def __post_init__(self) -> None:
        self.repo_dir = self.repo_dir.resolve()
        if self.result_dir is None:
            self.result_dir = self.repo_dir / "swe" / "results"
        else:
            self.result_dir = self.result_dir.resolve()


@dataclass(frozen=True)
class AutofixOutcome:
    """Result returned by ``CIAutofixLoop.run``."""

    status: LoopStatus
    attempts: int = 0
    branch: str = ""
    run_id: str = ""
    message: str = ""
    commit_sha: str = ""
    escalation_path: str = ""
    gate_summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "attempts": self.attempts,
            "branch": self.branch,
            "run_id": self.run_id,
            "message": self.message,
            "commit_sha": self.commit_sha,
            "escalation_path": self.escalation_path,
            "gate_summary": self.gate_summary,
        }


@dataclass(frozen=True)
class CommandResult:
    """Result of an external command."""

    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def command_text(self) -> str:
        return shlex.join(self.args)

    @property
    def combined_output(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)

    def tail(self, limit: int = DEFAULT_LOG_LIMIT) -> str:
        output = self.combined_output
        return output[-limit:] if len(output) > limit else output


class CommandRunner(Protocol):
    """Runs external commands from the auto-fix loop."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run a command and return a normalized result."""


class SubprocessCommandRunner:
    """Command runner backed by ``subprocess.run``."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        try:
            result = subprocess.run(
                list(args),
                cwd=str(cwd),
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=merged_env,
            )
            return CommandResult(tuple(args), result.returncode, result.stdout, result.stderr)
        except FileNotFoundError as exc:
            return CommandResult(tuple(args), 127, "", str(exc))
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return CommandResult(tuple(args), 124, stdout, stderr or f"Timed out after {timeout}s")


@dataclass(frozen=True)
class CIRun:
    """Small subset of GitHub Actions run metadata used by the loop."""

    database_id: str
    status: str
    conclusion: str
    head_branch: str = ""
    head_sha: str = ""
    workflow_name: str = ""
    display_title: str = ""

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failure(self) -> bool:
        return self.is_completed and self.conclusion == "failure"


def parse_latest_run(stdout: str, head_sha: str | None = None) -> CIRun | None:
    """Parse ``gh run list --json`` output and return the latest matching run."""

    try:
        rows = json.loads(stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid gh run list JSON: {exc}") from exc
    if not rows:
        return None

    if head_sha:
        rows = [row for row in rows if str(row.get("headSha") or "") == head_sha]
        if not rows:
            return None

    row = rows[0]
    database_id = row.get("databaseId") or row.get("id")
    if database_id is None:
        raise ValueError("gh run list output did not contain databaseId")

    return CIRun(
        database_id=str(database_id),
        status=str(row.get("status") or ""),
        conclusion=str(row.get("conclusion") or ""),
        head_branch=str(row.get("headBranch") or ""),
        head_sha=str(row.get("headSha") or ""),
        workflow_name=str(row.get("workflowName") or ""),
        display_title=str(row.get("displayTitle") or ""),
    )


class GitHubActionsClient:
    """GitHub Actions access through the GitHub CLI."""

    def __init__(self, runner: CommandRunner, repo_dir: Path, repo: str | None = None) -> None:
        self.runner = runner
        self.repo_dir = repo_dir
        self.repo = repo

    def latest_run(self, branch: str, workflow: str | None = None, head_sha: str | None = None) -> CIRun | None:
        args = [
            "gh",
            "run",
            "list",
            "--branch",
            branch,
            "--limit",
            "20",
            "--json",
            "databaseId,conclusion,status,headBranch,headSha,workflowName,displayTitle",
        ]
        if workflow:
            args.extend(["--workflow", workflow])
        if self.repo:
            args.extend(["-R", self.repo])

        result = self.runner.run(args, cwd=self.repo_dir, timeout=60)
        if not result.ok:
            raise RuntimeError(f"gh run list failed: {result.tail()}")
        return parse_latest_run(result.stdout, head_sha=head_sha)

    def failed_logs(self, run_id: str) -> str:
        args = ["gh", "run", "view", run_id, "--log-failed"]
        if self.repo:
            args.extend(["-R", self.repo])

        result = self.runner.run(args, cwd=self.repo_dir, timeout=180)
        if not result.ok:
            raise RuntimeError(f"gh run view --log-failed failed: {result.tail()}")
        return result.stdout


@dataclass(frozen=True)
class GateResult:
    """Quality gate command results."""

    command_results: tuple[CommandResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.command_results)

    def summary(self, limit: int = DEFAULT_LOG_LIMIT) -> str:
        if self.ok:
            return "All quality gates passed."

        parts = []
        for result in self.command_results:
            if result.ok:
                continue
            parts.append(
                "\n".join(
                    [
                        f"$ {result.command_text}",
                        f"exit={result.returncode}",
                        result.tail(limit),
                    ]
                ).strip()
            )
        return "\n\n".join(parts)[-limit:]


class QualityGate:
    """Runs the configured v0 quality checks."""

    def __init__(
        self,
        runner: CommandRunner,
        repo_dir: Path,
        commands: Sequence[Sequence[str]],
        timeout: int = DEFAULT_COMMAND_TIMEOUT,
    ) -> None:
        self.runner = runner
        self.repo_dir = repo_dir
        self.commands = tuple(tuple(command) for command in commands)
        self.timeout = timeout

    def run(self) -> GateResult:
        results: list[CommandResult] = []
        for command in self.commands:
            logger.info("Quality gate: %s", shlex.join(command))
            result = self.runner.run(command, cwd=self.repo_dir, timeout=self.timeout)
            results.append(result)
            if not result.ok:
                break
        return GateResult(tuple(results))
