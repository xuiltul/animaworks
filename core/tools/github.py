# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks GitHub tool — gh CLI wrapper.

Provides Issue and PR operations via the GitHub CLI.
Requires ``gh`` to be installed and authenticated (``gh auth login``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from core.tools._base import logger

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "issues":       {"expected_seconds": 15, "background_eligible": False},
    "issue":        {"expected_seconds": 10, "background_eligible": False},
    "create-issue": {"expected_seconds": 15, "background_eligible": False},
    "prs":          {"expected_seconds": 15, "background_eligible": False},
    "create-pr":    {"expected_seconds": 15, "background_eligible": False},
}


# ──────────────────────────────────────────────────────────────────────────────
# GitHubClient
# ──────────────────────────────────────────────────────────────────────────────

class GitHubClient:
    """GitHub operations via the ``gh`` CLI."""

    def __init__(self, repo: str | None = None) -> None:
        """Initialise the client.

        Args:
            repo: Repository in ``"owner/repo"`` format.
                  If *None*, ``gh`` infers the repo from the current directory.
        """
        self.repo = repo
        self._check_gh()

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _check_gh() -> None:
        """Verify ``gh`` is installed and authenticated."""
        try:
            subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "GitHub CLI 'gh' is not installed. "
                "See https://cli.github.com/ for installation instructions."
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "GitHub CLI 'gh' is not authenticated. Run: gh auth login"
            )

    def _run(self, args: list[str], input_text: str | None = None) -> str:
        """Execute a ``gh`` sub-command and return *stdout*.

        The ``-R owner/repo`` flag is appended automatically when
        ``self.repo`` is set.
        """
        cmd = ["gh"] + args
        if self.repo:
            cmd.extend(["-R", self.repo])
        logger.debug("gh command: %s", cmd)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=input_text,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
        return result.stdout.strip()

    # ── Issues ─────────────────────────────────────────────────────────────

    def list_issues(
        self,
        state: str = "open",
        labels: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List issues.

        Args:
            state:  ``"open"``, ``"closed"``, or ``"all"``.
            labels: Filter by label names.
            limit:  Maximum number of issues to return.

        Returns:
            List of issue dicts.
        """
        args = [
            "issue", "list",
            "--state", state,
            "--limit", str(limit),
            "--json", "number,title,state,labels,createdAt,author,assignees,body",
        ]
        if labels:
            args.extend(["--label", ",".join(labels)])
        return json.loads(self._run(args))

    def get_issue(self, number: int) -> dict[str, Any]:
        """Get a single issue by number.

        Args:
            number: Issue number.

        Returns:
            Issue dict with body, comments, labels, etc.
        """
        args = [
            "issue", "view", str(number),
            "--json",
            "number,title,state,body,labels,comments,createdAt,author,assignees",
        ]
        return json.loads(self._run(args))

    def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new issue.

        Args:
            title:  Issue title.
            body:   Issue body (Markdown).
            labels: Labels to apply.

        Returns:
            Dict with ``number``, ``url``, and ``title``.
        """
        args = ["issue", "create", "--title", title, "--body", body]
        if labels:
            for label in labels:
                args.extend(["--label", label])
        output = self._run(args)
        # gh issue create returns the URL of the created issue
        number = output.strip().rstrip("/").split("/")[-1]
        return {"number": int(number), "url": output.strip(), "title": title}

    # ── Pull Requests ──────────────────────────────────────────────────────

    def list_prs(
        self,
        state: str = "open",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List pull requests.

        Args:
            state:  ``"open"``, ``"closed"``, ``"merged"``, or ``"all"``.
            limit:  Maximum number of PRs to return.

        Returns:
            List of PR dicts.
        """
        args = [
            "pr", "list",
            "--state", state,
            "--limit", str(limit),
            "--json",
            "number,title,state,headRefName,baseRefName,createdAt,author,isDraft",
        ]
        return json.loads(self._run(args))

    def create_pr(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        draft: bool = False,
    ) -> dict[str, Any]:
        """Create a pull request.

        Args:
            title: PR title.
            body:  PR body (Markdown).
            head:  Source branch.
            base:  Target branch (default ``"main"``).
            draft: Create as draft PR.

        Returns:
            Dict with ``number``, ``url``, and ``title``.
        """
        args = [
            "pr", "create",
            "--title", title,
            "--body", body,
            "--head", head,
            "--base", base,
        ]
        if draft:
            args.append("--draft")
        output = self._run(args)
        number = output.strip().rstrip("/").split("/")[-1]
        return {"number": int(number), "url": output.strip(), "title": title}

    def pr_checks(self, number: int) -> list[dict[str, Any]]:
        """Get CI check status for a pull request.

        Args:
            number: PR number.

        Returns:
            List of check dicts with ``name``, ``state``, ``conclusion``.
            Returns an empty list if checks cannot be retrieved.
        """
        args = [
            "pr", "checks", str(number),
            "--json", "name,state,conclusion",
        ]
        try:
            return json.loads(self._run(args))
        except (json.JSONDecodeError, RuntimeError):
            return []


# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas (Anthropic tool_use format)
# ──────────────────────────────────────────────────────────────────────────────

def get_tool_schemas() -> list[dict[str, Any]]:
    """Return Anthropic-compatible tool schemas for GitHub operations."""
    return [
        {
            "name": "github_list_issues",
            "description": (
                "List GitHub issues for the repository, optionally filtered by "
                "state and labels."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in 'owner/repo' format. Optional.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Issue state filter. Default: open.",
                        "default": "open",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by label names.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of issues. Default: 20.",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "github_create_issue",
            "description": "Create a new GitHub issue.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in 'owner/repo' format. Optional.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Issue title.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue body (Markdown).",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to apply.",
                    },
                },
                "required": ["title", "body"],
            },
        },
        {
            "name": "github_list_prs",
            "description": "List pull requests for the repository.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in 'owner/repo' format. Optional.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "merged", "all"],
                        "description": "PR state filter. Default: open.",
                        "default": "open",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of PRs. Default: 20.",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "github_create_pr",
            "description": "Create a new pull request.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in 'owner/repo' format. Optional.",
                    },
                    "title": {
                        "type": "string",
                        "description": "PR title.",
                    },
                    "body": {
                        "type": "string",
                        "description": "PR body (Markdown).",
                    },
                    "head": {
                        "type": "string",
                        "description": "Source branch name.",
                    },
                    "base": {
                        "type": "string",
                        "description": "Target branch name. Default: main.",
                        "default": "main",
                    },
                    "draft": {
                        "type": "boolean",
                        "description": "Create as draft PR.",
                        "default": False,
                    },
                },
                "required": ["title", "body", "head"],
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def get_cli_guide() -> str:
    """Return CLI usage guide for GitHub tools."""
    return """\
### GitHub
```bash
animaworks-tool github issues -j
animaworks-tool github issues --repo owner/repo -j
animaworks-tool github issue 123 -j
animaworks-tool github create-issue --title "タイトル" --body "本文"
animaworks-tool github prs -j
animaworks-tool github create-pr --title "タイトル" --body "本文" --head ブランチ名
```"""


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI for GitHub operations.

    Sub-commands::

        issues                              List open issues
        issue <number>                      Show a single issue
        create-issue --title T --body B     Create an issue
        prs                                 List open PRs
        create-pr --title T --body B --head H  Create a PR
    """
    parser = argparse.ArgumentParser(
        prog="animaworks-github",
        description="AnimaWorks GitHub CLI",
    )
    parser.add_argument(
        "--repo", default=None, help="Repository in 'owner/repo' format"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # issues
    p_issues = sub.add_parser("issues", help="List issues")
    p_issues.add_argument("--state", default="open", choices=["open", "closed", "all"])
    p_issues.add_argument("--labels", nargs="*", default=None)
    p_issues.add_argument("--limit", type=int, default=20)

    # issue <number>
    p_issue = sub.add_parser("issue", help="Show a single issue")
    p_issue.add_argument("number", type=int)

    # create-issue
    p_ci = sub.add_parser("create-issue", help="Create an issue")
    p_ci.add_argument("--title", required=True)
    p_ci.add_argument("--body", required=True)
    p_ci.add_argument("--labels", nargs="*", default=None)

    # prs
    p_prs = sub.add_parser("prs", help="List pull requests")
    p_prs.add_argument("--state", default="open", choices=["open", "closed", "merged", "all"])
    p_prs.add_argument("--limit", type=int, default=20)

    # create-pr
    p_pr = sub.add_parser("create-pr", help="Create a pull request")
    p_pr.add_argument("--title", required=True)
    p_pr.add_argument("--body", required=True)
    p_pr.add_argument("--head", required=True)
    p_pr.add_argument("--base", default="main")
    p_pr.add_argument("--draft", action="store_true")

    args = parser.parse_args(argv)
    client = GitHubClient(repo=args.repo)

    if args.command == "issues":
        result = client.list_issues(
            state=args.state, labels=args.labels, limit=args.limit
        )
    elif args.command == "issue":
        result = client.get_issue(args.number)
    elif args.command == "create-issue":
        result = client.create_issue(
            title=args.title, body=args.body, labels=args.labels
        )
    elif args.command == "prs":
        result = client.list_prs(state=args.state, limit=args.limit)
    elif args.command == "create-pr":
        result = client.create_pr(
            title=args.title,
            body=args.body,
            head=args.head,
            base=args.base,
            draft=args.draft,
        )
    else:
        parser.print_help()
        sys.exit(1)

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, default=str)
    print()  # trailing newline


# ── Dispatch ──────────────────────────────────────────


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    if tool_name == "github_list_issues":
        client = GitHubClient(repo=args.get("repo"))
        return client.list_issues(
            state=args.get("state", "open"),
            labels=args.get("labels"),
            limit=args.get("limit", 20),
        )
    if tool_name == "github_create_issue":
        client = GitHubClient(repo=args.get("repo"))
        return client.create_issue(
            title=args["title"],
            body=args.get("body", ""),
            labels=args.get("labels"),
        )
    if tool_name == "github_list_prs":
        client = GitHubClient(repo=args.get("repo"))
        return client.list_prs(
            state=args.get("state", "open"),
            limit=args.get("limit", 20),
        )
    if tool_name == "github_create_pr":
        client = GitHubClient(repo=args.get("repo"))
        return client.create_pr(
            title=args["title"],
            body=args.get("body", ""),
            head=args["head"],
            base=args.get("base", "main"),
            draft=args.get("draft", False),
        )
    raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    cli_main()