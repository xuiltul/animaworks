"""State machine for the CI auto-fix loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from swe.ci_autofix_types import (
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_LOG_LIMIT,
    DEFAULT_MAX_ITER,
    DEFAULT_QUALITY_COMMANDS,
    CommandResult,
    CommandRunner,
    GitHubActionsClient,
    LoopStatus,
    QualityGate,
    SubprocessCommandRunner,
)

logger = logging.getLogger(__name__)


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


class CIAutofixLoop:
    """State machine for CI/local auto-repair."""

    def __init__(self, config: AutofixConfig, runner: CommandRunner | None = None) -> None:
        self.config = config
        self.runner = runner or SubprocessCommandRunner()
        self.github = GitHubActionsClient(self.runner, config.repo_dir, config.repo)
        self.gates = QualityGate(
            self.runner,
            config.repo_dir,
            config.quality_commands,
            timeout=config.command_timeout,
        )

    def run(self) -> AutofixOutcome:
        if self.config.mode not in {"ci", "local"}:
            return AutofixOutcome(LoopStatus.FAILED, message=f"Unsupported mode: {self.config.mode}")

        dirty = self._git_status()
        if dirty and not self.config.allow_dirty:
            return AutofixOutcome(
                LoopStatus.DIRTY_WORKTREE,
                message="Working tree has uncommitted changes. Commit/stash them or pass --allow-dirty.",
                gate_summary=dirty,
            )

        branch = self.config.branch or self._current_branch()
        run_id = ""

        try:
            if self.config.mode == "ci":
                head_sha = self._current_commit() if self.config.match_head_sha else None
                ci_run = self.github.latest_run(branch, self.config.workflow, head_sha=head_sha)
                if ci_run is None:
                    return AutofixOutcome(
                        LoopStatus.NO_FAILURE,
                        branch=branch,
                        message="No CI runs found for the branch/current HEAD.",
                    )
                run_id = ci_run.database_id
                if not ci_run.is_completed:
                    return AutofixOutcome(
                        LoopStatus.PENDING,
                        branch=branch,
                        run_id=run_id,
                        message=f"Latest CI run is {ci_run.status}.",
                    )
                if not ci_run.is_failure:
                    return AutofixOutcome(
                        LoopStatus.NO_FAILURE,
                        branch=branch,
                        run_id=run_id,
                        message=f"Latest CI conclusion is {ci_run.conclusion or 'unknown'}.",
                    )
                failure_context = self.github.failed_logs(ci_run.database_id)
            else:
                initial_gate = self.gates.run()
                if initial_gate.ok:
                    return AutofixOutcome(
                        LoopStatus.NO_FAILURE,
                        branch=branch,
                        message="Local quality gates already pass.",
                    )
                failure_context = initial_gate.summary(self.config.log_limit)

            return self._repair_loop(branch=branch, run_id=run_id, failure_context=failure_context)
        except Exception as exc:
            logger.exception("CI auto-fix failed")
            return AutofixOutcome(
                LoopStatus.FAILED,
                branch=branch,
                run_id=run_id,
                message=str(exc),
            )

    def _repair_loop(self, *, branch: str, run_id: str, failure_context: str) -> AutofixOutcome:
        last_summary = failure_context[-self.config.log_limit :]

        for attempt in range(1, self.config.max_iter + 1):
            logger.info("Auto-fix attempt %d/%d", attempt, self.config.max_iter)
            prompt = self._build_repair_prompt(attempt, branch, run_id, last_summary)
            fixer_result = self._run_fixer(prompt)
            if not fixer_result.ok:
                last_summary = self._attempt_summary("Fixer command failed", fixer_result)
                continue

            if not self._has_worktree_changes():
                last_summary = self._attempt_summary("Fixer completed but produced no git diff", fixer_result)
                continue

            gate_result = self.gates.run()
            if not gate_result.ok:
                last_summary = gate_result.summary(self.config.log_limit)
                continue

            diff_check = self.runner.run(["git", "diff", "--check"], cwd=self.config.repo_dir, timeout=60)
            if not diff_check.ok:
                last_summary = self._attempt_summary("git diff --check failed", diff_check)
                continue

            review = self._run_review(gate_result.summary(self.config.log_limit))
            if not review.ok:
                last_summary = self._attempt_summary("Reviewer gate failed", review)
                continue

            commit = self._commit_successful_fix(branch=branch, run_id=run_id, attempt=attempt)
            if not commit.ok:
                last_summary = self._attempt_summary("git commit failed", commit)
                continue

            if self.config.push:
                push = self.runner.run(["git", "push"], cwd=self.config.repo_dir, timeout=self.config.command_timeout)
                if not push.ok:
                    last_summary = self._attempt_summary("git push failed", push)
                    continue

            commit_sha = self._current_commit()
            return AutofixOutcome(
                LoopStatus.SUCCESS,
                attempts=attempt,
                branch=branch,
                run_id=run_id,
                message="Auto-fix loop completed and committed a green local repair.",
                commit_sha=commit_sha,
                gate_summary=gate_result.summary(self.config.log_limit),
            )

        escalation_path = self._escalate(branch=branch, run_id=run_id, summary=last_summary)
        return AutofixOutcome(
            LoopStatus.EXHAUSTED,
            attempts=self.config.max_iter,
            branch=branch,
            run_id=run_id,
            message="Auto-fix loop exhausted max attempts.",
            escalation_path=escalation_path,
            gate_summary=last_summary,
        )

    def _build_repair_prompt(self, attempt: int, branch: str, run_id: str, failure_context: str) -> str:
        run_line = f"GitHub Actions run id: {run_id}" if run_id else "Local quality gate failure"
        return f"""You are the CI auto-fix Architect for AnimaWorks.

Repository: {self.config.repo_dir}
Branch: {branch}
Attempt: {attempt}/{self.config.max_iter}
{run_line}

Your task:
1. Inspect the failure logs below.
2. Edit the repository directly with the minimal fix.
3. Do not refactor unrelated code.
4. Preserve public APIs unless the log proves a change is required.
5. Leave the working tree with a focused git diff.

Failure logs:

{failure_context[-self.config.log_limit :]}
"""

    def _run_fixer(self, prompt: str) -> CommandResult:
        if self.config.fix_command:
            return self.runner.run(
                self.config.fix_command,
                cwd=self.config.repo_dir,
                input_text=prompt,
                timeout=self.config.command_timeout,
            )

        command = ["uv", "run", "animaworks", "chat", self.config.agent_name, prompt]
        return self.runner.run(command, cwd=self.config.repo_dir, timeout=self.config.command_timeout)

    def _run_review(self, gate_summary: str) -> CommandResult:
        if self.config.skip_review:
            return CommandResult(("internal-review",), 0, "APPROVE: review skipped by configuration", "")

        diff = self.runner.run(["git", "diff", "HEAD"], cwd=self.config.repo_dir, timeout=60)
        if not diff.ok:
            return diff

        prompt = f"""You are the CI auto-fix Reviewer for AnimaWorks.

Review this patch for obvious breakage, regressions, and unsafe broad changes.
Reply with exactly one verdict line starting with APPROVE or NEEDS_CHANGES,
then concise reasons.

Quality gate summary:
{gate_summary[-self.config.log_limit :]}

Patch:
```diff
{diff.stdout[-self.config.log_limit :]}
```
"""
        if self.config.review_command:
            result = self.runner.run(
                self.config.review_command,
                cwd=self.config.repo_dir,
                input_text=prompt,
                timeout=self.config.command_timeout,
            )
        else:
            command = ["uv", "run", "animaworks", "chat", self.config.review_agent_name, prompt]
            result = self.runner.run(command, cwd=self.config.repo_dir, timeout=self.config.command_timeout)

        if not result.ok:
            return result

        verdict = result.combined_output.upper()
        if "NEEDS_CHANGES" in verdict or "REQUEST_CHANGES" in verdict:
            return CommandResult(result.args, 1, result.stdout, result.stderr)
        if "APPROVE" in verdict:
            return result
        return CommandResult(
            result.args,
            1,
            result.stdout,
            result.stderr + "\nReviewer did not return APPROVE or NEEDS_CHANGES.",
        )

    def _attempt_summary(self, label: str, result: CommandResult) -> str:
        return "\n".join(
            [
                label,
                f"$ {result.command_text}",
                f"exit={result.returncode}",
                result.tail(self.config.log_limit),
            ]
        )[-self.config.log_limit :]

    def _git_status(self) -> str:
        result = self.runner.run(["git", "status", "--porcelain"], cwd=self.config.repo_dir, timeout=60)
        if not result.ok:
            raise RuntimeError(f"git status failed: {result.tail()}")
        return result.stdout.strip()

    def _current_branch(self) -> str:
        result = self.runner.run(["git", "branch", "--show-current"], cwd=self.config.repo_dir, timeout=60)
        if not result.ok:
            raise RuntimeError(f"git branch --show-current failed: {result.tail()}")
        branch = result.stdout.strip()
        if not branch:
            raise RuntimeError("Cannot infer branch from detached HEAD; pass --branch.")
        return branch

    def _has_worktree_changes(self) -> bool:
        return bool(self._git_status())

    def _commit_successful_fix(self, *, branch: str, run_id: str, attempt: int) -> CommandResult:
        add = self.runner.run(["git", "add", "-A"], cwd=self.config.repo_dir, timeout=60)
        if not add.ok:
            return add

        subject = "fix: auto-repair CI failure"
        details = [f"branch: {branch}", f"attempt: {attempt}"]
        if run_id:
            details.insert(0, f"run: {run_id}")
        message = subject + "\n\n" + "\n".join(details)
        return self.runner.run(["git", "commit", "-m", message], cwd=self.config.repo_dir, timeout=120)

    def _current_commit(self) -> str:
        result = self.runner.run(["git", "rev-parse", "HEAD"], cwd=self.config.repo_dir, timeout=60)
        if not result.ok:
            return ""
        return result.stdout.strip()

    def _escalate(self, *, branch: str, run_id: str, summary: str) -> str:
        subject = f"CI auto-fix exhausted for {branch}"
        body = "\n\n".join(
            [
                f"Repository: {self.config.repo_dir}",
                f"Branch: {branch}",
                f"Run ID: {run_id or 'local'}",
                f"Attempts: {self.config.max_iter}",
                "Last failure summary:",
                summary[-self.config.log_limit :],
            ]
        )
        command = [*self.config.escalation_command, subject, body, "--priority", "high"]
        result = self.runner.run(command, cwd=self.config.repo_dir, timeout=120)
        if result.ok:
            return ""
        return self._write_escalation_artifact(subject, body, result)

    def _write_escalation_artifact(self, subject: str, body: str, result: CommandResult) -> str:
        assert self.config.result_dir is not None
        self.config.result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.config.result_dir / f"ci_autofix_escalation_{timestamp}.md"
        path.write_text(
            "\n".join(
                [
                    f"# {subject}",
                    "",
                    body,
                    "",
                    "## call_human fallback reason",
                    "",
                    f"$ {result.command_text}",
                    f"exit={result.returncode}",
                    result.tail(self.config.log_limit),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return str(path)
