"""Tests for core/tools/github.py — GitHub integration via gh CLI."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from core.tools.github import GitHubClient, get_tool_schemas


# ── GitHubClient ──────────────────────────────────────────────────


class TestGitHubClient:
    @pytest.fixture(autouse=True)
    def _mock_gh_auth(self):
        """Mock gh auth status so GitHubClient.__init__ succeeds."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["gh", "auth", "status"], returncode=0,
                stdout="", stderr="",
            )
            yield mock_run

    def test_init_no_repo(self):
        client = GitHubClient()
        assert client.repo is None

    def test_init_with_repo(self):
        client = GitHubClient(repo="owner/repo")
        assert client.repo == "owner/repo"

    def test_check_gh_not_installed(self, _mock_gh_auth):
        _mock_gh_auth.side_effect = FileNotFoundError()
        with pytest.raises(RuntimeError, match="not installed"):
            GitHubClient()

    def test_check_gh_not_authenticated(self, _mock_gh_auth):
        _mock_gh_auth.side_effect = subprocess.CalledProcessError(1, "gh")
        with pytest.raises(RuntimeError, match="not authenticated"):
            GitHubClient()


class TestGitHubClientRun:
    @pytest.fixture(autouse=True)
    def _mock_gh(self):
        with patch("subprocess.run") as mock_run:
            # First call = auth check
            auth_result = subprocess.CompletedProcess(
                args=["gh", "auth", "status"], returncode=0,
                stdout="", stderr="",
            )
            mock_run.return_value = auth_result
            self._mock_run = mock_run
            yield

    def _set_run_output(self, stdout: str, returncode: int = 0):
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # auth
            subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="error msg"),
        ]
        self._mock_run.side_effect = results

    def test_run_appends_repo_flag(self):
        self._set_run_output('[]')
        client = GitHubClient(repo="owner/repo")
        client._run(["issue", "list"])
        call_args = self._mock_run.call_args_list[-1]
        cmd = call_args.args[0] if call_args.args else call_args.kwargs.get("args", [])
        assert "-R" in cmd
        assert "owner/repo" in cmd

    def test_run_no_repo_flag(self):
        self._set_run_output('[]')
        client = GitHubClient()
        client._run(["issue", "list"])
        call_args = self._mock_run.call_args_list[-1]
        cmd = call_args.args[0] if call_args.args else call_args.kwargs.get("args", [])
        assert "-R" not in cmd

    def test_run_error(self):
        self._set_run_output("", returncode=1)
        client = GitHubClient()
        with pytest.raises(RuntimeError, match="gh command failed"):
            client._run(["issue", "list"])


class TestListIssues:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("subprocess.run") as mock_run:
            auth_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            mock_run.return_value = auth_result
            self._mock_run = mock_run
            yield

    def test_list_issues(self):
        issues = [{"number": 1, "title": "Bug"}]
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(issues), stderr=""),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        result = client.list_issues()
        assert result == issues

    def test_list_issues_with_labels(self):
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr=""),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        client.list_issues(labels=["bug", "urgent"])
        cmd = self._mock_run.call_args_list[-1].args[0]
        assert "--label" in cmd
        assert "bug,urgent" in cmd


class TestGetIssue:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("subprocess.run") as mock_run:
            auth_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            mock_run.return_value = auth_result
            self._mock_run = mock_run
            yield

    def test_get_issue(self):
        issue = {"number": 42, "title": "Feature request", "body": "..."}
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(issue), stderr=""),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        result = client.get_issue(42)
        assert result["number"] == 42


class TestCreateIssue:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("subprocess.run") as mock_run:
            auth_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            mock_run.return_value = auth_result
            self._mock_run = mock_run
            yield

    def test_create_issue(self):
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout="https://github.com/owner/repo/issues/99", stderr="",
            ),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        result = client.create_issue(title="New Issue", body="Description")
        assert result["number"] == 99
        assert "issues/99" in result["url"]


class TestListPrs:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("subprocess.run") as mock_run:
            auth_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            mock_run.return_value = auth_result
            self._mock_run = mock_run
            yield

    def test_list_prs(self):
        prs = [{"number": 10, "title": "Fix thing"}]
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(prs), stderr=""),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        result = client.list_prs()
        assert result == prs


class TestCreatePr:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("subprocess.run") as mock_run:
            auth_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            mock_run.return_value = auth_result
            self._mock_run = mock_run
            yield

    def test_create_pr(self):
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout="https://github.com/owner/repo/pull/50", stderr="",
            ),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        result = client.create_pr(
            title="New PR", body="PR body", head="feature-branch",
        )
        assert result["number"] == 50

    def test_create_pr_draft(self):
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout="https://github.com/owner/repo/pull/51", stderr="",
            ),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        client.create_pr(title="T", body="B", head="h", draft=True)
        cmd = self._mock_run.call_args_list[-1].args[0]
        assert "--draft" in cmd


class TestPrChecks:
    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch("subprocess.run") as mock_run:
            auth_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            mock_run.return_value = auth_result
            self._mock_run = mock_run
            yield

    def test_pr_checks_success(self):
        checks = [{"name": "CI", "state": "SUCCESS", "conclusion": "success"}]
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(checks), stderr=""),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        result = client.pr_checks(10)
        assert result == checks

    def test_pr_checks_error(self):
        results = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error"),
        ]
        self._mock_run.side_effect = results
        client = GitHubClient()
        result = client.pr_checks(10)
        assert result == []


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 4
        names = {s["name"] for s in schemas}
        assert names == {
            "github_list_issues", "github_create_issue",
            "github_list_prs", "github_create_pr",
        }

    def test_create_issue_requires_title_and_body(self):
        schemas = get_tool_schemas()
        ci = [s for s in schemas if s["name"] == "github_create_issue"][0]
        assert set(ci["input_schema"]["required"]) == {"title", "body"}

    def test_create_pr_requires_title_body_head(self):
        schemas = get_tool_schemas()
        cp = [s for s in schemas if s["name"] == "github_create_pr"][0]
        assert set(cp["input_schema"]["required"]) == {"title", "body", "head"}
