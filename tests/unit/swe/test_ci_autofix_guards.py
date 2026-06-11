"""Guard-specific tests for the CI auto-fix loop."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from swe.ci_autofix import AutofixConfig, CIAutofixLoop, LoopStatus, SubprocessCommandRunner


def _run(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, check=True)


def _init_repo(path: Path) -> None:
    _run(["git", "init", "-b", "main"], path)
    _run(["git", "config", "user.email", "ci-autofix@test.local"], path)
    _run(["git", "config", "user.name", "CI Autofix"], path)
    (path / "README.md").write_text("initial\n", encoding="utf-8")
    _run(["git", "add", "."], path)
    _run(["git", "commit", "-m", "initial"], path)


def test_allow_dirty_does_not_commit_baseline_untracked_directory(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    baseline_dir = tmp_path / "baseline_dir"
    baseline_dir.mkdir()
    (baseline_dir / "secret.txt").write_text("do not commit\n", encoding="utf-8")
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
    escalation = (sys.executable, "-c", "import sys; sys.exit(1)")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        max_iter=1,
        quality_commands=(gate,),
        fix_command=fix,
        review_command=review,
        allow_dirty=True,
        escalation_command=escalation,
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.FAILED
    assert "baseline_dir/secret.txt" in outcome.gate_summary
    assert _run(["git", "log", "-1", "--pretty=%s"], tmp_path).stdout.strip() == "initial"


def test_reviewer_side_effects_are_rejected(tmp_path: Path) -> None:
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
    review = (
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; sys.stdin.read(); "
        "Path('review_artifact.txt').write_text('artifact\\n', encoding='utf-8'); print('APPROVE')",
    )
    escalation = (sys.executable, "-c", "import sys; sys.exit(1)")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        max_iter=2,
        quality_commands=(gate,),
        fix_command=fix,
        review_command=review,
        escalation_command=escalation,
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.FAILED
    assert "Reviewer command modified the worktree" in outcome.gate_summary
    assert _run(["git", "log", "-1", "--pretty=%s"], tmp_path).stdout.strip() == "initial"
    assert "review_artifact.txt" not in _run(["git", "ls-tree", "-r", "--name-only", "HEAD"], tmp_path).stdout


def test_failing_reviewer_side_effects_stop_before_retry(tmp_path: Path) -> None:
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
    review = (
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; sys.stdin.read(); "
        "Path('review_artifact.txt').write_text('artifact\\n', encoding='utf-8'); sys.exit(1)",
    )
    escalation = (sys.executable, "-c", "import sys; sys.exit(1)")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        max_iter=2,
        quality_commands=(gate,),
        fix_command=fix,
        review_command=review,
        escalation_command=escalation,
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.FAILED
    assert "Reviewer command modified the worktree" in outcome.gate_summary
    assert _run(["git", "log", "-1", "--pretty=%s"], tmp_path).stdout.strip() == "initial"


def test_allow_dirty_restores_baseline_file_modified_by_fixer(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("user change\n", encoding="utf-8")
    gate = (
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; sys.exit(0 if Path('fixed.txt').exists() else 1)",
    )
    fix = (
        sys.executable,
        "-c",
        "from pathlib import Path; "
        "Path('README.md').write_text('initial\\n', encoding='utf-8'); "
        "Path('fixed.txt').write_text('ok\\n', encoding='utf-8')",
    )
    review = (sys.executable, "-c", "import sys; sys.stdin.read(); print('APPROVE')")
    escalation = (sys.executable, "-c", "import sys; sys.exit(1)")
    config = AutofixConfig(
        repo_dir=tmp_path,
        mode="local",
        max_iter=2,
        quality_commands=(gate,),
        fix_command=fix,
        review_command=review,
        allow_dirty=True,
        escalation_command=escalation,
        result_dir=tmp_path / "results",
    )

    outcome = CIAutofixLoop(config, runner=SubprocessCommandRunner()).run()

    assert outcome.status == LoopStatus.FAILED
    assert "README.md" in outcome.gate_summary
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "user change\n"
    assert _run(["git", "log", "-1", "--pretty=%s"], tmp_path).stdout.strip() == "initial"


def test_reviewer_verdict_requires_leading_approve(tmp_path: Path) -> None:
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
    review = (sys.executable, "-c", "import sys; sys.stdin.read(); print('I cannot approve this patch')")
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
    assert "Reviewer did not return APPROVE or NEEDS_CHANGES" in outcome.gate_summary
    assert _run(["git", "log", "-1", "--pretty=%s"], tmp_path).stdout.strip() == "initial"
