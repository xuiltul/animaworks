"""Tests for swe/runner.py — unit tests for helpers and orchestration logic."""
from __future__ import annotations

import subprocess
from pathlib import Path


from swe.runner import (
    create_test_instance,
    git_diff,
)


class TestCreateTestInstance:
    def test_creates_repo(self, tmp_path):
        instance = create_test_instance(tmp_path)
        repo_dir = Path(instance["_local_repo_dir"])

        assert repo_dir.exists()
        assert (repo_dir / "calculator.py").exists()
        assert (repo_dir / "test_calculator.py").exists()

    def test_has_required_fields(self, tmp_path):
        instance = create_test_instance(tmp_path)

        assert "instance_id" in instance
        assert "problem_statement" in instance
        assert "base_commit" in instance
        assert instance["repo"] == "_local_"

    def test_repo_is_git(self, tmp_path):
        instance = create_test_instance(tmp_path)
        repo_dir = Path(instance["_local_repo_dir"])

        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "true"

    def test_calculator_has_bugs(self, tmp_path):
        instance = create_test_instance(tmp_path)
        repo_dir = Path(instance["_local_repo_dir"])

        result = subprocess.run(
            ["python3", "-m", "pytest", "test_calculator.py", "-v", "--tb=no"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "FAILED" in result.stdout


class TestGitDiff:
    def test_empty_diff(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "t"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "f.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )

        assert git_diff(tmp_path) == ""

    def test_detects_changes(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "t"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "f.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )

        (tmp_path / "f.txt").write_text("world")
        diff = git_diff(tmp_path)
        assert "hello" in diff
        assert "world" in diff
