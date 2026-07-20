# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""GitHub external tasks collector via ``gh`` CLI."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from core.external_tasks.models import ExternalTask

_PR_PRIORITY = 90
_ISSUE_PRIORITY = 75
_SEARCH_LIMIT = 30
_JSON_FIELDS = "number,title,url,createdAt,updatedAt,repository"


def collect_github() -> list[ExternalTask]:
    """Collect open PRs requesting review and open issues assigned to me.

    Task ids:
      - ``github-pr-{repo}-{number}``
      - ``github-issue-{repo}-{number}``
    """
    _ensure_gh()
    tasks: list[ExternalTask] = []
    tasks.extend(_collect_review_prs())
    tasks.extend(_collect_assigned_issues())
    return tasks


def _ensure_gh() -> None:
    """Verify ``gh`` is installed and authenticated.

    Mirrors :meth:`core.tools.github.GitHubClient._check_gh` but raises
    :class:`CredentialNotFoundError` so the collector can mark the source
    unavailable without treating missing tooling as a hard failure.
    """
    # Local import avoids circular import with collector → sources.
    from core.external_tasks.collector import CredentialNotFoundError

    try:
        subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise CredentialNotFoundError(
            "GitHub CLI 'gh' is not installed. See https://cli.github.com/"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise CredentialNotFoundError(
            "GitHub CLI 'gh' is not authenticated. Run: gh auth login"
        ) from exc


def _run_gh_json(args: list[str]) -> list[dict[str, Any]]:
    """Run a ``gh`` command that returns a JSON array."""
    cmd = ["gh", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    stdout = result.stdout.strip()
    if not stdout:
        return []
    data = json.loads(stdout)
    if not isinstance(data, list):
        raise RuntimeError(f"gh command returned non-list JSON: {type(data).__name__}")
    return data


def _repo_name(item: dict[str, Any]) -> str:
    repo = item.get("repository")
    if isinstance(repo, dict):
        name = repo.get("name") or repo.get("nameWithOwner")
        if name:
            return str(name)
        name_with_owner = repo.get("nameWithOwner")
        if name_with_owner:
            return str(name_with_owner).rsplit("/", 1)[-1]
    if isinstance(repo, str) and repo:
        return repo.rsplit("/", 1)[-1]
    return "unknown"


def _repo_id_part(item: dict[str, Any]) -> str:
    """Stable repo segment for task ids (prefer nameWithOwner, ``/`` → ``-``)."""
    repo = item.get("repository")
    if isinstance(repo, dict):
        nwo = repo.get("nameWithOwner") or repo.get("name")
        if nwo:
            return str(nwo).replace("/", "-")
    if isinstance(repo, str) and repo:
        return repo.replace("/", "-")
    return "unknown"


def _collect_review_prs() -> list[ExternalTask]:
    items = _run_gh_json(
        [
            "search",
            "prs",
            "--review-requested=@me",
            "--state=open",
            "--json",
            _JSON_FIELDS,
            "--limit",
            str(_SEARCH_LIMIT),
        ]
    )
    tasks: list[ExternalTask] = []
    for item in items:
        number = item.get("number")
        if number is None:
            continue
        repo = _repo_name(item)
        repo_id = _repo_id_part(item)
        title_text = item.get("title") or ""
        tasks.append(
            ExternalTask(
                id=f"github-pr-{repo_id}-{number}",
                title=f"{repo} #{number}: {title_text}",
                status="open",
                source_type="github",
                source_icon="github",
                source_url=item.get("url"),
                created_at=item.get("createdAt") or item.get("updatedAt") or "",
                last_updated_at=item.get("updatedAt") or item.get("createdAt") or "",
                priority=_PR_PRIORITY,
            )
        )
    return tasks


def _collect_assigned_issues() -> list[ExternalTask]:
    items = _run_gh_json(
        [
            "search",
            "issues",
            "--assignee=@me",
            "--state=open",
            "--json",
            _JSON_FIELDS,
            "--limit",
            str(_SEARCH_LIMIT),
        ]
    )
    tasks: list[ExternalTask] = []
    for item in items:
        number = item.get("number")
        if number is None:
            continue
        # gh search issues may include PRs; skip if clearly a PR via url
        url = item.get("url") or ""
        if "/pull/" in url:
            continue
        repo = _repo_name(item)
        repo_id = _repo_id_part(item)
        title_text = item.get("title") or ""
        tasks.append(
            ExternalTask(
                id=f"github-issue-{repo_id}-{number}",
                title=f"{repo} #{number}: {title_text}",
                status="open",
                source_type="github",
                source_icon="github",
                source_url=item.get("url"),
                created_at=item.get("createdAt") or item.get("updatedAt") or "",
                last_updated_at=item.get("updatedAt") or item.get("createdAt") or "",
                priority=_ISSUE_PRIORITY,
            )
        )
    return tasks
