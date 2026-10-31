from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared canonical-path policy helpers for per-Anima file access."""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.anima.workspace_files import (
    AGENTS_FILE,
    BOOTSTRAP_FILE,
    HEARTBEAT_FILE,
    IDENTITY_FILE,
    LEGACY_IDENTITY,
    LEGACY_RENAMES,
    SOUL_FILE,
)
from core.i18n import t

PROTECTED_FILE_PATHS = frozenset(
    {
        "permissions.md",
        "permissions.json",
        IDENTITY_FILE,
        SOUL_FILE,
        AGENTS_FILE,
        BOOTSTRAP_FILE,
        "status.json",
        "state/bm25_longterm_index.json",
        "state/bm25_longterm_index.dirty",
    }
) | (
    # Legacy names are protected too (APFS case folding).  heartbeat.md is
    # excluded because it only differs from the writable HEARTBEAT.md by case.
    frozenset({LEGACY_IDENTITY, *LEGACY_RENAMES}) - frozenset({HEARTBEAT_FILE.casefold()})
)

PROTECTED_DIRECTORY_MARKERS = frozenset({"activity_log"})


@dataclass(frozen=True)
class FileAccessContext:
    """Resolved inputs for one file-access decision.

    Callers load configuration and filesystem-backed allowlists before creating
    this value.  ``evaluate_file_access`` only reads this context and resolves
    paths; it never loads config, checks directory existence, or logs.
    """

    anima_dir: Path
    data_dir: Path
    denied_roots: tuple[Path, ...] = ()
    file_roots: tuple[str, ...] = ("/",)
    file_roots_readonly: tuple[str, ...] = ()
    write_roots: tuple[Path, ...] = ()
    restrict_reads_to_roots: bool = True
    additional_read_dirs: tuple[Path, ...] = ()
    subordinate_activity_dirs: tuple[Path, ...] = ()
    descendant_activity_dirs: tuple[Path, ...] = ()
    peer_activity_dirs: tuple[Path, ...] = ()
    subordinate_management_files: tuple[Path, ...] = ()
    subordinate_root_dirs: tuple[Path, ...] = ()
    descendant_read_files: tuple[Path, ...] = ()
    descendant_read_dirs: tuple[Path, ...] = ()
    protected_files: frozenset[str] = PROTECTED_FILE_PATHS
    protected_directory_markers: frozenset[str] = PROTECTED_DIRECTORY_MARKERS
    superuser: bool = False
    trusted_internal_cache_write: bool = False


@dataclass(frozen=True)
class FileAccessDecision:
    """Pure file-access result plus details needed by caller-specific errors."""

    allowed: bool
    reason: str = "allowed"
    denied_root: Path | None = None
    internal_cache_root: Path | None = None
    protected_path: str | None = None
    readonly_root: Path | None = None
    allowed_dirs: tuple[Path, ...] = ()


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

    Membership is resolved through :func:`core.org.company.get_company`, matching
    the company deny policy.  Both the company directory and ``shared`` itself
    must remain direct children of their expected roots so a malformed
    membership or symlink cannot turn this narrow grant into an arbitrary
    filesystem write.
    """
    from core.org.company import get_company

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


def check_file_write_roots(
    path: str | Path,
    *,
    file_roots: list[str],
    file_roots_readonly: list[str],
    write_roots: tuple[Path, ...],
) -> str | None:
    """Check a write target against already-resolved writable/read-only roots.

    Returns ``"readonly_dir"`` or ``"outside_allowed_dirs"`` when denied,
    otherwise ``None``. Callers handle higher-priority grants and explicit
    deny rules before using this common root-boundary check.
    """
    resolved = Path(path).resolve()
    if any(resolved.is_relative_to(root) for root in write_roots):
        return None
    if not file_roots:
        return "outside_allowed_dirs"

    readonly_roots = tuple(Path(root).resolve() for root in file_roots_readonly if Path(root).is_absolute())
    if any(resolved.is_relative_to(root) for root in readonly_roots):
        return "readonly_dir"
    return "outside_allowed_dirs"


def company_denied_roots(anima_dir: Path) -> tuple[Path, ...]:
    """Return canonical roots for every company other than the Anima's own.

    Membership is read from ``status.json`` on every call so assignment
    changes take effect without a server restart.  Unassigned Animas retain
    the legacy unrestricted behavior.
    """
    from core.org.company import get_company

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


def load_denied_roots(anima_dir: Path, *, read_only: bool = False) -> tuple[Path, ...]:
    """Load the Anima's configured and company-derived file deny roots."""
    from core.config.models import load_permissions

    return resolve_effective_denied_roots(
        anima_dir,
        load_permissions(anima_dir, read_only=read_only).file_roots_denied,
    )


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


def _is_protected_relative(relative: str, protected_files: frozenset[str]) -> bool:
    """Compare protected file names case-insensitively for APFS compatibility."""
    return relative.casefold() in {item.casefold() for item in protected_files}


def evaluate_protected_write(
    anima_dir: Path,
    path: str | Path,
    *,
    protected_files: frozenset[str] = PROTECTED_FILE_PATHS,
    protected_directory_markers: frozenset[str] = PROTECTED_DIRECTORY_MARKERS,
) -> FileAccessDecision | None:
    """Return the shared protected-target decision for a write, if any."""
    resolved = Path(path).resolve()
    anima_root = Path(anima_dir).resolve()
    if not resolved.is_relative_to(anima_root):
        return FileAccessDecision(False, "outside_anima")

    relative_path = resolved.relative_to(anima_root)
    relative = str(relative_path)
    if _is_protected_relative(relative, protected_files):
        return FileAccessDecision(False, "protected_file", protected_path=relative)
    for marker in sorted(protected_directory_markers):
        if marker in relative_path.parts:
            return FileAccessDecision(False, "protected_directory", protected_path=marker)
    return None


def _path_is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    """Check containment against paths already supplied by the context."""
    return any(path.is_relative_to(Path(root).resolve()) for root in roots)


def _path_equals_any(path: Path, candidates: tuple[Path, ...]) -> bool:
    """Check exact path membership against paths supplied by the context."""
    return any(path == Path(candidate).resolve() for candidate in candidates)


def _absolute_roots(roots: tuple[str, ...]) -> tuple[Path, ...]:
    """Resolve configured absolute roots, ignoring relative entries for reads."""
    return tuple(Path(root).resolve() for root in roots if Path(root).is_absolute())


def evaluate_file_access(
    path: str | Path,
    ctx: FileAccessContext,
    *,
    write: bool,
) -> FileAccessDecision:
    """Evaluate one file access against the shared per-Anima policy.

    The context builder owns config reads and filesystem-backed allowlist
    discovery. This function is deterministic for a given context and path,
    apart from canonical path resolution needed to enforce symlink boundaries.
    """
    if not str(path):
        return FileAccessDecision(True)

    resolved = Path(path).resolve()
    anima_root = Path(ctx.anima_dir).resolve()
    data_root = Path(ctx.data_dir).resolve()
    root_config_path = data_root / "config.json"
    lexical_path = Path(os.path.abspath(path))
    if write and (resolved == root_config_path or lexical_path == root_config_path):
        return FileAccessDecision(False, "protected_file", protected_path="config.json")
    if write:
        animas_root = data_root / "animas"
        for candidate in (resolved, lexical_path):
            if not candidate.is_relative_to(animas_root):
                continue
            relative = candidate.relative_to(animas_root)
            if len(relative.parts) < 2:
                continue
            anima_relative = Path(*relative.parts[1:])
            if _is_protected_relative(str(anima_relative), ctx.protected_files):
                return FileAccessDecision(False, "protected_file", protected_path=str(anima_relative))
            for marker in sorted(ctx.protected_directory_markers):
                if marker in anima_relative.parts:
                    return FileAccessDecision(False, "protected_directory", protected_path=marker)
    if write and resolved.is_relative_to(anima_root):
        protected = evaluate_protected_write(
            anima_root,
            resolved,
            protected_files=ctx.protected_files,
            protected_directory_markers=ctx.protected_directory_markers,
        )
        if protected is not None:
            return protected
    if ctx.superuser:
        return FileAccessDecision(True)

    # Deny configured credential caches whenever explicit denies are active.
    # Trusted internal writes are a narrow ToolHandler-only maintenance path;
    # explicit file_roots_denied still applies below.
    if ctx.denied_roots and not (write and ctx.trusted_internal_cache_write):
        # Pass the original path so find_internal_cache_root can inspect both
        # the lexical path and its symlink-resolved target.
        internal_cache_root = find_internal_cache_root(path, anima_root)
        if internal_cache_root is not None:
            return FileAccessDecision(
                False,
                "internal_cache",
                internal_cache_root=internal_cache_root,
            )

    denied_root = find_denied_root(resolved, ctx.denied_roots)
    if denied_root is not None:
        return FileAccessDecision(False, "denied_root", denied_root=denied_root)

    # Protect the root configuration and system permissions from model-facing
    # file tools even when a workspace has unusually broad write roots.
    if write and resolved.name == "permissions.global.json" and resolved.is_relative_to(data_root):
        return FileAccessDecision(False, "global_permissions")

    if resolved.is_relative_to(anima_root):
        if write:
            protected = evaluate_protected_write(
                anima_root,
                resolved,
                protected_files=ctx.protected_files,
                protected_directory_markers=ctx.protected_directory_markers,
            )
            if protected is not None:
                return protected
        return FileAccessDecision(True)

    # Explicit supervisor/peer exceptions are evaluated before the general
    # inter-Anima boundary. Management files are read/write; all other grants
    # in this block are read-only.
    if not write:
        if _path_is_under_any(resolved, ctx.subordinate_activity_dirs):
            return FileAccessDecision(True)
        if _path_is_under_any(resolved, ctx.descendant_activity_dirs):
            return FileAccessDecision(True)
        if _path_is_under_any(resolved, ctx.peer_activity_dirs):
            return FileAccessDecision(True)

    if _path_equals_any(resolved, ctx.subordinate_management_files):
        return FileAccessDecision(True)

    if not write:
        if _path_equals_any(resolved, ctx.descendant_read_files):
            return FileAccessDecision(True)
        if _path_is_under_any(resolved, ctx.descendant_read_dirs):
            return FileAccessDecision(True)
        if _path_equals_any(resolved, ctx.subordinate_root_dirs):
            return FileAccessDecision(True)
        if _path_is_under_any(resolved, ctx.additional_read_dirs):
            return FileAccessDecision(True)

    # Other Anima directories stay private except for the grants above. This
    # boundary takes precedence over general file_roots grants.
    animas_root = anima_root.parent
    if resolved.is_relative_to(animas_root) and not resolved.is_relative_to(anima_root):
        return FileAccessDecision(False, "other_anima")

    # Mode S historically allows reads outside other Anima directories without
    # consulting file_roots. ToolHandler applies the configured read roots.
    if not write and not ctx.restrict_reads_to_roots:
        return FileAccessDecision(True)

    if write:
        write_denial = check_file_write_roots(
            resolved,
            file_roots=list(ctx.file_roots),
            file_roots_readonly=list(ctx.file_roots_readonly),
            write_roots=ctx.write_roots,
        )
        readonly_roots = _absolute_roots(ctx.file_roots_readonly)
        if write_denial == "readonly_dir":
            readonly_root = next(root for root in readonly_roots if resolved.is_relative_to(root))
            return FileAccessDecision(
                False,
                "readonly_dir",
                readonly_root=readonly_root,
            )
        if write_denial is not None:
            allowed_dirs = (*ctx.write_roots, *readonly_roots) if ctx.file_roots else ()
            return FileAccessDecision(
                False,
                "outside_allowed_dirs",
                allowed_dirs=tuple(Path(root).resolve() for root in allowed_dirs),
            )
        return FileAccessDecision(True)

    allowed_roots = _absolute_roots(ctx.file_roots)
    if _path_is_under_any(resolved, allowed_roots):
        return FileAccessDecision(True)
    if not ctx.file_roots:
        return FileAccessDecision(False, "outside_allowed_dirs")

    readonly_roots = _absolute_roots(ctx.file_roots_readonly)
    if _path_is_under_any(resolved, readonly_roots):
        return FileAccessDecision(True)
    return FileAccessDecision(
        False,
        "outside_allowed_dirs",
        allowed_dirs=(*allowed_roots, *readonly_roots),
    )


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
        from core.org.company_resources import get_company_resources, infer_data_dir

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
