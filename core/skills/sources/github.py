from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub source adapter for Skill Hub imports."""

import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError

from core.skills.sources.local import stage_local_source
from core.skills.sources.url import stage_url_source


def stage_github_source(source: str, staging_root: Path) -> tuple[Path, str | None]:
    """Stage ``github:owner/repo/path`` and return ``(skill_dir, commit)``."""
    owner, repo, repo_path = _parse_github_source(source)
    if shutil.which("gh"):
        checkout = staging_root / "repo"
        subprocess.run(
            [
                "gh",
                "repo",
                "clone",
                f"{owner}/{repo}",
                str(checkout),
                "--",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "sparse-checkout", "set", "--no-cone", repo_path],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = _git_head(checkout)
        staged = stage_local_source(str(checkout / repo_path), staging_root / "from-gh")
        return staged, commit

    candidates = _raw_url_candidates(owner, repo, repo_path)
    errors: list[str] = []
    for url in candidates:
        try:
            return stage_url_source(url, staging_root / "from-raw"), None
        except (HTTPError, URLError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("GitHub source could not be fetched without gh CLI: " + "; ".join(errors))


def _parse_github_source(source: str) -> tuple[str, str, str]:
    raw = source.removeprefix("github:").strip("/")
    parts = raw.split("/", 2)
    if len(parts) < 3 or not all(parts):
        raise ValueError("GitHub sources must use github:owner/repo/path/to/skill")
    owner, repo, repo_path = parts
    if ".." in Path(repo_path).parts or Path(repo_path).is_absolute():
        raise ValueError("GitHub source path must be relative and must not contain '..'")
    return owner, repo, repo_path


def _raw_url_candidates(owner: str, repo: str, repo_path: str) -> list[str]:
    suffix = repo_path if repo_path.endswith("SKILL.md") else f"{repo_path.rstrip('/')}/SKILL.md"
    return [
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/{suffix}",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/{suffix}",
    ]


def _git_head(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None
