"""E2E coverage for the local CI auto-fix loop."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from swe.ci_autofix import AutofixConfig, CIAutofixLoop, LoopStatus, SubprocessCommandRunner


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, check=True)


def _init_broken_repo(repo: Path) -> None:
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "ci-autofix-e2e@test.local"], repo)
    _run(["git", "config", "user.name", "CI Autofix E2E"], repo)
    (repo / "mathlib.py").write_text(
        textwrap.dedent(
            """\
            def divide(a, b):
                return a / b
            """
        ),
        encoding="utf-8",
    )
    (repo / "test_mathlib.py").write_text(
        textwrap.dedent(
            """\
            import pytest
            from mathlib import divide


            def test_divide_by_zero_raises_value_error():
                with pytest.raises(ValueError, match="Cannot divide by zero"):
                    divide(10, 0)
            """
        ),
        encoding="utf-8",
    )
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "initial broken mathlib"], repo)


@pytest.mark.e2e
def test_local_loop_repairs_failing_test_and_commits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_broken_repo(repo)

    fixer = tmp_path / "fixer.py"
    fixer.write_text(
        textwrap.dedent(
            """\
            from pathlib import Path
            import sys

            sys.stdin.read()
            Path("mathlib.py").write_text(
                "def divide(a, b):\\n"
                "    if b == 0:\\n"
                "        raise ValueError('Cannot divide by zero')\\n"
                "    return a / b\\n",
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.py"
    review.write_text("import sys\nsys.stdin.read()\nprint('APPROVE')\n", encoding="utf-8")

    config = AutofixConfig(
        repo_dir=repo,
        mode="local",
        max_iter=1,
        quality_commands=((sys.executable, "-m", "pytest", "test_mathlib.py", "-q"),),
        fix_command=(sys.executable, str(fixer)),
        review_command=(sys.executable, str(review)),
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.SUCCESS
    assert "Cannot divide by zero" in (repo / "mathlib.py").read_text(encoding="utf-8")
    assert _run(["git", "status", "--porcelain"], repo).stdout == ""
    assert "fix: auto-repair CI failure" in _run(["git", "log", "-1", "--pretty=%s"], repo).stdout
    _run([sys.executable, "-m", "pytest", "test_mathlib.py", "-q"], repo)
