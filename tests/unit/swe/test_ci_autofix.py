"""Tests for swe/ci_autofix.py."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import swe.ci_autofix as ci_cli
from swe.ci_autofix import (
    AutofixConfig,
    AutofixOutcome,
    CIAutofixLoop,
    CommandResult,
    GitHubActionsClient,
    LoopStatus,
    SubprocessCommandRunner,
    parse_latest_run,
)


def _run(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, check=True)


def _init_repo(path: Path) -> None:
    _run(["git", "init", "-b", "main"], path)
    _run(["git", "config", "user.email", "ci-autofix@test.local"], path)
    _run(["git", "config", "user.name", "CI Autofix"], path)
    (path / "README.md").write_text("initial\n", encoding="utf-8")
    _run(["git", "add", "."], path)
    _run(["git", "commit", "-m", "initial"], path)


def test_parse_latest_run_matches_head_sha() -> None:
    stdout = json.dumps(
        [
            {
                "databaseId": 101,
                "status": "completed",
                "conclusion": "failure",
                "headBranch": "main",
                "headSha": "old",
                "workflowName": "CI",
                "displayTitle": "old run",
            },
            {
                "databaseId": 102,
                "status": "completed",
                "conclusion": "success",
                "headBranch": "main",
                "headSha": "abc123",
                "workflowName": "CI",
                "displayTitle": "current run",
            },
        ]
    )

    run = parse_latest_run(stdout, head_sha="abc123")

    assert run is not None
    assert run.database_id == "102"
    assert run.conclusion == "success"
    assert run.head_sha == "abc123"
    assert parse_latest_run(stdout, head_sha="missing") is None


def test_parse_latest_run_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="Invalid gh run list JSON"):
        parse_latest_run("{not json")


class QueueRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        self.calls.append(tuple(args))
        return self.results.pop(0)


def test_github_actions_client_uses_run_list_and_failed_logs(tmp_path: Path) -> None:
    runner = QueueRunner(
        [
            CommandResult(
                ("gh",),
                0,
                json.dumps(
                    [
                        {
                            "databaseId": 200,
                            "status": "completed",
                            "conclusion": "failure",
                            "headBranch": "feature",
                            "headSha": "abc",
                        }
                    ]
                ),
                "",
            ),
            CommandResult(("gh",), 0, "pytest failed\n", ""),
        ]
    )
    client = GitHubActionsClient(runner, tmp_path, repo="owner/repo")

    run = client.latest_run("feature", workflow="CI", head_sha="abc")
    logs = client.failed_logs("200")

    assert run is not None
    assert run.database_id == "200"
    assert logs == "pytest failed\n"
    assert runner.calls[0][:4] == ("gh", "run", "list", "--branch")
    assert any("headSha" in part for part in runner.calls[0])
    assert runner.calls[0][-2:] == ("-R", "owner/repo")
    assert runner.calls[1][:4] == ("gh", "run", "view", "200")


def test_ci_mode_pending_run_returns_pending(tmp_path: Path) -> None:
    runner = QueueRunner(
        [
            CommandResult(("git", "status"), 0, "", ""),
            CommandResult(("git", "branch"), 0, "main\n", ""),
            CommandResult(("git", "rev-parse"), 0, "abc\n", ""),
            CommandResult(
                ("gh",),
                0,
                json.dumps(
                    [
                        {
                            "databaseId": 300,
                            "status": "in_progress",
                            "conclusion": "",
                            "headBranch": "main",
                            "headSha": "abc",
                        }
                    ]
                ),
                "",
            ),
        ]
    )
    config = AutofixConfig(repo_dir=tmp_path, mode="ci")

    outcome = CIAutofixLoop(config, runner=runner).run()

    assert outcome.status == LoopStatus.PENDING
    assert outcome.run_id == "300"


def test_local_loop_fixes_gates_reviews_and_commits(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    gate = (
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; sys.exit(0 if Path('fixed.txt').exists() else 1)",
    )
    fix = (
        sys.executable,
        "-c",
        "from pathlib import Path; Path('fixed.txt').write_text('ok\\n', encoding='utf-8')",
    )
    review = (sys.executable, "-c", "import sys; sys.stdin.read(); print('APPROVE')")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        max_iter=1,
        quality_commands=(gate,),
        fix_command=fix,
        review_command=review,
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.SUCCESS
    assert outcome.attempts == 1
    assert (tmp_path / "fixed.txt").read_text(encoding="utf-8") == "ok\n"
    assert _run(["git", "status", "--porcelain"], tmp_path).stdout == ""
    assert "fix: auto-repair CI failure" in _run(["git", "log", "-1", "--pretty=%B"], tmp_path).stdout


def test_local_loop_no_failure_when_gates_already_pass(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    gate = (sys.executable, "-c", "import sys; sys.exit(0)")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        quality_commands=(gate,),
        fix_command=(sys.executable, "-c", "raise SystemExit('unused')"),
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.NO_FAILURE
    assert outcome.message == "Local quality gates already pass."


def test_reviewer_needs_changes_exhausts_without_commit(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    gate = (
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; sys.exit(0 if Path('fixed.txt').exists() else 1)",
    )
    fix = (
        sys.executable,
        "-c",
        "from pathlib import Path; Path('fixed.txt').write_text('ok\\n', encoding='utf-8')",
    )
    review = (sys.executable, "-c", "import sys; sys.stdin.read(); print('NEEDS_CHANGES')")
    escalation = (sys.executable, "-c", "import sys; sys.exit(1)")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        max_iter=1,
        quality_commands=(gate,),
        fix_command=fix,
        review_command=review,
        escalation_command=escalation,
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.EXHAUSTED
    assert "Reviewer gate failed" in outcome.gate_summary
    assert _run(["git", "log", "-1", "--pretty=%s"], tmp_path).stdout.strip() == "initial"


def test_local_loop_exhausts_and_writes_escalation_artifact(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    gate = (sys.executable, "-c", "import sys; print('still broken'); sys.exit(1)")
    fix = (sys.executable, "-c", "import sys; sys.stdin.read(); print('no changes')")
    escalation = (sys.executable, "-c", "import sys; sys.exit(1)")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        max_iter=3,
        quality_commands=(gate,),
        fix_command=fix,
        skip_review=True,
        escalation_command=escalation,
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.EXHAUSTED
    assert outcome.attempts == 3
    assert outcome.escalation_path
    artifact = Path(outcome.escalation_path)
    assert artifact.exists()
    text = artifact.read_text(encoding="utf-8")
    assert "CI auto-fix exhausted" in text
    assert "Fixer completed but produced no git diff" in text


def test_dirty_worktree_is_refused_by_default(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("dirty\n", encoding="utf-8")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        quality_commands=((sys.executable, "-c", "print('unused')"),),
        fix_command=(sys.executable, "-c", "print('unused')"),
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.DIRTY_WORKTREE
    assert "README.md" in outcome.gate_summary


def test_cli_config_from_args_parses_overrides(tmp_path: Path) -> None:
    parser = ci_cli.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "local",
            "--repo-dir",
            str(tmp_path),
            "--fix-command",
            "python -c 'print(1)'",
            "--review-command",
            "python -c 'print(\"APPROVE\")'",
            "--gate-command",
            "python -m pytest test_file.py",
            "--allow-dirty",
            "--allow-stale-run",
            "--push",
            "--result-dir",
            str(tmp_path / "results"),
            "--escalation-command",
            "notify-human",
            "--timeout",
            "12",
        ]
    )

    config = ci_cli.config_from_args(args)

    assert config.mode == "local"
    assert config.repo_dir == tmp_path.resolve()
    assert config.fix_command == ("python", "-c", "print(1)")
    assert config.review_command == ("python", "-c", 'print("APPROVE")')
    assert config.quality_commands == (("python", "-m", "pytest", "test_file.py"),)
    assert config.allow_dirty is True
    assert config.match_head_sha is False
    assert config.push is True
    assert config.escalation_command == ("notify-human",)
    assert config.command_timeout == 12


def test_cli_main_json_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeLoop:
        def __init__(self, config: AutofixConfig) -> None:
            self.config = config

        def run(self) -> AutofixOutcome:
            return AutofixOutcome(LoopStatus.NO_FAILURE, message="ok")

    monkeypatch.setattr(ci_cli, "CIAutofixLoop", FakeLoop)

    code = ci_cli.main(["--mode", "local", "--json"])

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "no_failure"
    assert data["message"] == "ok"


def test_cli_main_dirty_exit_code(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeLoop:
        def __init__(self, config: AutofixConfig) -> None:
            self.config = config

        def run(self) -> AutofixOutcome:
            return AutofixOutcome(LoopStatus.DIRTY_WORKTREE, message="dirty")

    monkeypatch.setattr(ci_cli, "CIAutofixLoop", FakeLoop)

    code = ci_cli.main(["--mode", "local"])

    assert code == 2
    assert "status: dirty_worktree" in capsys.readouterr().out
