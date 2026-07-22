from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared canonical-path policy helpers for per-Anima file read denies."""

import os
from collections.abc import Iterable
from pathlib import Path

_ANIMA_MEMORY_ROOTS = frozenset(
    {
        "activity_log",
        "archive",
        "episodes",
        "facts",
        "knowledge",
        "procedures",
        "skills",
        "state",
    }
)


def resolve_denied_roots(roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """Canonicalize configured deny roots once for repeated comparisons."""
    return tuple(Path(root).resolve() for root in roots)


def company_shared_write_root(anima_dir: Path) -> Path | None:
    """Return the assigned company's canonical writable ``shared`` root.

    Membership is resolved through :func:`core.company.get_company`, matching
    the company deny policy.  Both the company directory and ``shared`` itself
    must remain direct children of their expected roots so a malformed
    membership or symlink cannot turn this narrow grant into an arbitrary
    filesystem write.
    """
    from core.company import get_company

    anima_dir = Path(anima_dir)
    company = get_company(anima_dir.name, animas_dir=anima_dir.parent)
    if company is None:
        return None

    data_dir = anima_dir.resolve().parent.parent
    companies_dir = (data_dir / "companies").resolve()
    company_root = companies_dir / company
    if company_root.parent != companies_dir or company_root.is_symlink():
        return None
    company_root = company_root.resolve()
    if company_root.parent != companies_dir:
        return None

    shared_root = company_root / "shared"
    if shared_root.is_symlink():
        return None
    shared_root = shared_root.resolve()
    if shared_root.parent != company_root:
        return None
    return shared_root


def company_denied_roots(anima_dir: Path) -> tuple[Path, ...]:
    """Return canonical roots for every company other than the Anima's own.

    Membership is read from ``status.json`` on every call so assignment
    changes take effect without a server restart.  Unassigned Animas retain
    the legacy unrestricted behavior.
    """
    from core.company import get_company

    company = get_company(anima_dir.name, animas_dir=anima_dir.parent)
    if company is None:
        return ()

    companies_dir = anima_dir.parent.parent / "companies"
    try:
        candidates = tuple(companies_dir.iterdir())
    except OSError:
        return ()

    denied: list[Path] = []
    for candidate in sorted(candidates, key=lambda path: path.name):
        if candidate.name == company:
            continue
        try:
            if candidate.is_dir():
                denied.append(candidate.resolve())
        except (OSError, RuntimeError):
            continue
    return tuple(denied)


def resolve_effective_denied_roots(
    anima_dir: Path,
    configured_roots: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Merge configured and company-derived denies into canonical roots.

    Roots nested inside another deny root are dropped: the outer root already
    covers them, and sandbox backends that materialize one mount per deny root
    (bwrap) fail to create a mountpoint inside an already-denied read-only
    subtree, aborting the whole sandbox before the command starts.
    """
    merged = tuple(dict.fromkeys((*resolve_denied_roots(configured_roots), *company_denied_roots(anima_dir))))
    return tuple(root for root in merged if not any(other != root and root.is_relative_to(other) for other in merged))


def load_denied_roots(anima_dir: Path) -> tuple[Path, ...]:
    """Load the Anima's configured and company-derived file deny roots."""
    from core.config.models import load_permissions

    return resolve_effective_denied_roots(anima_dir, load_permissions(anima_dir).file_roots_denied)


def find_denied_root(path: str | Path, denied_roots: tuple[Path, ...]) -> Path | None:
    """Return the canonical deny root containing *path*, following symlinks."""
    resolved = Path(path).resolve()
    return next((root for root in denied_roots if resolved.is_relative_to(root)), None)


def _is_vectordb_variant(name: str) -> bool:
    """Return whether a top-level Anima path is a live/staged vector cache."""
    return name == "vectordb" or name.startswith(("vectordb-", "vectordb.", "vectordb_"))


def find_internal_cache_root(path: str | Path, anima_dir: Path) -> Path | None:
    """Return the protected internal cache containing *path*, if any.

    Both the symlink-resolved target and lexical absolute path are checked:
    an allowed symlink into a cache and a cache symlink pointing outward must
    both remain inaccessible to model-facing file tools.
    """
    anima_root = anima_dir.resolve()
    requested = Path(path)
    candidates = (requested.resolve(), Path(os.path.abspath(requested)))

    for candidate in candidates:
        if not candidate.is_relative_to(anima_root):
            continue
        relative = candidate.relative_to(anima_root)
        if not relative.parts:
            continue

        top = relative.parts[0]
        if top in {".codex_home", "archive"}:
            return anima_root / top
        if _is_vectordb_variant(top):
            return anima_root / top
        if top == "state" and len(relative.parts) >= 2:
            name = relative.parts[1]
            if name.startswith("bm25_longterm_index.") or name.startswith(".bm25_longterm_index."):
                return anima_root / "state" / name
    return None


def shell_internal_deny_paths(anima_dir: Path) -> tuple[Path, ...]:
    """Return runtime paths hidden from a deny-enabled model shell.

    The whole state/archive trees are brokered through host/MCP services.
    Existing vector cache variants are included alongside the stable live
    path; archive quarantines are covered by the archive root.
    """
    anima_root = anima_dir.resolve()
    paths = {
        (anima_root / "state").resolve(),
        (anima_root / "archive").resolve(),
        (anima_root / ".codex_home").resolve(),
        (anima_root / "vectordb").resolve(),
        anima_root / "vectordb-*",
        anima_root / "vectordb.*",
        anima_root / "vectordb_*",
    }
    try:
        paths.update(child.resolve() for child in anima_root.iterdir() if _is_vectordb_variant(child.name))
    except OSError:
        pass
    return tuple(sorted(paths, key=str))


def resolve_memory_source_path(anima_dir: Path, source: str) -> Path | None:
    """Resolve a trusted memory ``source_file``/derived doc path to a real path.

    Relative values must start with a known memory namespace.  Opaque vector
    IDs are deliberately not guessed: when deny is active their callers can
    fail closed instead of accidentally releasing cached content.
    """
    source = str(source or "").strip().split("#", 1)[0]
    if not source or source == "unknown":
        return None
    path = Path(source)
    if path.is_absolute():
        return path.resolve()
    if not path.parts or ".." in path.parts:
        return None

    from core.paths import get_common_knowledge_dir, get_common_skills_dir, get_data_dir, get_reference_dir

    shared_roots = {
        "common_knowledge": get_common_knowledge_dir(),
        "common_skills": get_common_skills_dir(),
        "reference": get_reference_dir(),
    }
    namespace = path.parts[0]
    shared_root = shared_roots.get(namespace)
    if shared_root is not None:
        return shared_root.joinpath(*path.parts[1:]).resolve()
    if namespace == "shared":
        return (get_data_dir() / path).resolve()
    if namespace == "companies":
        from core.company_resources import get_company_resources, infer_data_dir

        resources = get_company_resources(anima_dir)
        if resources is None or path.parts[:2] != ("companies", resources.name):
            return None
        candidate = (infer_data_dir(anima_dir) / path).resolve()
        try:
            candidate.relative_to(resources.root)
        except ValueError:
            return None
        return candidate
    if namespace in _ANIMA_MEMORY_ROOTS:
        return (anima_dir / path).resolve()
    return None


def memory_source_is_allowed(
    anima_dir: Path,
    source: str,
    denied_roots: tuple[Path, ...],
) -> bool:
    """Check a cached memory source, failing closed on ambiguity when deny is active."""
    if not denied_roots:
        return True
    source_path = resolve_memory_source_path(anima_dir, source)
    if source_path is None:
        return False
    return find_denied_root(source_path, denied_roots) is None
