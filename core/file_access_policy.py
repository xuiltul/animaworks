from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared canonical-path policy helpers for per-Anima file read denies."""

import os
from collections.abc import Iterable
from pathlib import Path

from core.i18n import t

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


class FileRootsConfigError(ValueError):
    """Raised when ``file_roots`` violates the write-access charter."""


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


def effective_write_roots(
    anima_dir: Path,
    file_roots: list[str],
    task_cwd: Path | None = None,
) -> tuple[Path, ...]:
    """Return canonical writable roots from the write-access charter.

    See ``docs/specs/write-access-charter.ja.md``.
    """
    anima_dir = Path(anima_dir)
    data_dir = anima_dir.resolve().parent.parent
    company_shared = company_shared_write_root(anima_dir)
    roots: list[Path] = []
    if company_shared is not None:
        company_shared.mkdir(parents=True, exist_ok=True)
        roots.append(company_shared)

    for root in file_roots:
        resolved = Path(root).expanduser().resolve()
        if resolved.is_relative_to(data_dir) and not (
            company_shared is not None and resolved.is_relative_to(company_shared)
        ):
            raise FileRootsConfigError(t("config.file_roots_inside_data_dir", path=resolved))
        roots.append(resolved)

    if task_cwd is not None:
        roots.append(Path(task_cwd).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


def shared_tool_cache_write_root(anima_dir: Path) -> Path | None:
    """Return the shared external-tool cache root that must stay writable.

    Chatwork/Slack style tools keep their identity map and SQLite message
    caches under ``<data>/cache``.  A sandbox that grants writes only inside
    the Anima directory turns even a read of the Anima's own inbox into
    ``EROFS``, because those tools write the cache before serving the read.
    """
    data_dir = Path(anima_dir).resolve().parent.parent
    cache_root = data_dir / "cache"
    if cache_root.is_symlink():
        return None
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    resolved = cache_root.resolve()
    if resolved.parent != data_dir:
        return None
    return resolved


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


def find_internal_cache_root(path: str | Path, anima_dir: Path) -> Path | None:
    """Return the protected credential/runtime-control root containing *path*.

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

        if relative.parts[0] == ".codex_home":
            return anima_root / ".codex_home"
    return None


def foreign_owned_ssh_config_dirs(ssh_config_d: Path = Path("/etc/ssh/ssh_config.d")) -> list[str]:
    """Return ssh drop-in dirs that would fail ssh's owner check inside bwrap.

    Inside the sandbox's user namespace only the current uid keeps its
    identity; everything else (root included) shows up as ``nobody``.  ssh
    requires included config files to be owned by root or the caller, so any
    drop-in not owned by us makes *every* ``ssh`` exit 255.
    """
    try:
        if any(p.stat().st_uid != os.getuid() for p in ssh_config_d.glob("*.conf")):
            return [str(ssh_config_d)]
    except OSError:
        pass
    return []


def shell_internal_deny_paths(anima_dir: Path) -> tuple[Path, ...]:
    """Return credential/runtime-control paths hidden from the model shell."""
    anima_root = anima_dir.resolve()
    return ((anima_root / ".codex_home").resolve(),)


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
