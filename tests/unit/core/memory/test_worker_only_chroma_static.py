from __future__ import annotations

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
        "__pycache__",
        "docs",
        "tests",
    }
    needles = ("ChromaVectorStore(", "chromadb.PersistentClient(")
    offenders: list[str] = []

    for path in repo.rglob("*.py"):
        if any(part in ignored_parts for part in path.relative_to(repo).parts):
            continue
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            offenders.append(str(path.relative_to(repo)))

    assert offenders == []
