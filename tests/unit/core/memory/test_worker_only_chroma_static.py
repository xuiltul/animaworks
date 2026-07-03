from __future__ import annotations

import subprocess
from pathlib import Path


def test_production_chroma_construction_has_no_unapproved_call_sites() -> None:
    """Prevent production code from reintroducing direct ChromaDB call sites."""
    repo = Path(__file__).resolve().parents[4]
    allowed = {
        repo / "core/memory/rag/store.py",
        repo / "core/memory/rag/direct_access.py",
    }
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "docs",
        "tests",
        "tmp",
    }
    needles = ("ChromaVectorStore(", "chromadb.PersistentClient(")
    offenders: list[str] = []

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
            check=True,
            cwd=repo,
            capture_output=True,
            text=True,
        )
        paths = [repo / line for line in result.stdout.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        paths = list(repo.rglob("*.py"))

    for path in paths:
        if any(part in ignored_parts for part in path.relative_to(repo).parts):
            continue
        if path in allowed:
            continue
        # git ls-files --cached can list a file that is still staged in the
        # index but has been removed from the working tree (e.g. deleted in a
        # worktree before the removal is committed). Skip such phantom paths.
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            offenders.append(str(path.relative_to(repo)))

    assert offenders == []
