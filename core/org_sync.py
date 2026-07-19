from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Periodic organizational structure synchronization.

Scans anima directories, extracts supervisor relationships from status.json
(or identity.md), and syncs them into config.json entries.
status.json is the single source of truth; config.json is kept in sync.
"""

import json
import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from core.config.models import (
    AnimaModelConfig,
    load_config,
    read_anima_company,
    read_anima_supervisor,
    save_config,
)
from core.exceptions import AnimaWorksError  # noqa: F401

logger = logging.getLogger(__name__)


# ── Circular reference detection ─────────────────────────────────


def _detect_circular_references(
    relationships: dict[str, str | None],
) -> list[tuple[str, ...]]:
    """Detect circular supervisor references.

    Args:
        relationships: Mapping of anima name to supervisor name.

    Returns:
        List of cycles found, each represented as a tuple of names.
    """
    cycles: list[tuple[str, ...]] = []
    visited: set[str] = set()

    for start in relationships:
        if start in visited:
            continue

        path: list[str] = []
        path_set: set[str] = set()
        current: str | None = start

        while current is not None and current not in visited:
            if current in path_set:
                # Found a cycle — extract the cycle portion
                cycle_start = path.index(current)
                cycle = tuple(path[cycle_start:])
                cycles.append(cycle)
                break
            path.append(current)
            path_set.add(current)
            current = relationships.get(current)

        visited.update(path_set)

    return cycles


# ── Main sync function ───────────────────────────────────────────


def sync_org_structure(
    animas_dir: Path,
    config_path: Path | None = None,
) -> dict[str, str | None]:
    """Sync organizational structure from anima files to config.json.

    ``status.json`` (or ``identity.md``) is the single source of truth for
    supervisor relationships.  This function syncs those values *into*
    ``config.json`` so that code reading ``config.animas`` directly stays
    up-to-date.

    For each anima directory:

    1. Extract supervisor from status.json / identity.md via
       :func:`~core.config.models.read_anima_supervisor`.
    2. If config.json has no entry for this anima, create one.
    3. If the supervisor value differs, update config.json to match disk.

    Args:
        animas_dir: Path to the animas directory
            (e.g. ``~/.animaworks/animas``).
        config_path: Optional explicit path to config.json.  When ``None``,
            derived from ``animas_dir.parent / "config.json"`` so that
            config always lives under the same data root as *animas_dir*.

    Returns:
        Dict of ``{anima_name: supervisor_value}`` for all discovered entries.
    """
    if config_path is None:
        config_path = animas_dir.parent / "config.json"
    if not animas_dir.is_dir():
        logger.debug("Animas directory does not exist: %s", animas_dir)
        return {}

    # ── Phase 1: discover supervisor relationships from disk ──────

    discovered: dict[str, str | None] = {}
    discovered_companies: dict[str, str | None] = {}

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        if not (anima_dir / "identity.md").exists():
            continue

        name = anima_dir.name
        discovered[name] = read_anima_supervisor(anima_dir)
        discovered_companies[name] = read_anima_company(anima_dir)

    if not discovered:
        logger.debug("No animas discovered in %s", animas_dir)
        return {}

    logger.info("Org sync: discovered %d animas from disk", len(discovered))

    # ── Phase 2: detect circular references ──────────────────────

    cycles = _detect_circular_references(discovered)
    circular_animas: set[str] = set()
    for cycle in cycles:
        circular_animas.update(cycle)
        logger.warning(
            "Org sync: circular supervisor reference detected: %s",
            " -> ".join(cycle) + " -> " + cycle[0],
        )

    # ── Phase 3: reconcile with config.json ──────────────────────

    config = load_config(config_path)
    changed = False

    for name, disk_supervisor in discovered.items():
        # Skip animas involved in circular references
        if name in circular_animas:
            logger.warning(
                "Org sync: skipping %s due to circular reference",
                name,
            )
            continue

        if name not in config.animas:
            # Anima not yet in config — add with discovered supervisor
            config.animas[name] = AnimaModelConfig(
                supervisor=disk_supervisor,
                company=discovered_companies[name],
            )
            changed = True
            logger.info(
                "Org sync: added anima '%s' with supervisor=%s",
                name,
                disk_supervisor,
            )
            continue

        existing = config.animas[name]

        if existing.supervisor != disk_supervisor:
            # status.json / identity.md is the SSoT — sync to config.json
            logger.info(
                "Org sync: updating supervisor for '%s': '%s' -> '%s'",
                name,
                existing.supervisor,
                disk_supervisor,
            )
            existing.supervisor = disk_supervisor
            changed = True

        disk_company = discovered_companies[name]
        if existing.company != disk_company:
            logger.info(
                "Org sync: updating company for '%s': '%s' -> '%s'",
                name,
                existing.company,
                disk_company,
            )
            existing.company = disk_company
            changed = True

    # ── Phase 4: prune config entries with no directory on disk ──

    for cname in list(config.animas.keys()):
        if cname in discovered:
            continue
        candidate = animas_dir / cname
        if not candidate.is_dir():
            del config.animas[cname]
            changed = True
            logger.info(
                "Org sync: pruned config entry '%s' (no directory on disk)",
                cname,
            )

    # ── Phase 5: persist if anything changed ─────────────────────

    if changed:
        save_config(config, config_path)
        logger.info("Org sync: config.json updated")
    else:
        logger.debug("Org sync: no changes needed")

    return discovered


# ── Orphan anima detection ───────────────────────────────────────


def _find_orphan_supervisor(
    orphan_dir: Path,
    animas_dir: Path,
) -> str | None:
    """Determine which supervisor should be notified about an orphan anima.

    Resolution order:
        1. ``status.json`` in *orphan_dir* — ``supervisor`` field.
        2. ``config.json`` (global) — ``animas.<name>.supervisor``.
        3. Fallback: first anima in *animas_dir* whose ``config.json``
           entry has ``supervisor=None`` and whose ``identity.md`` exists
           (i.e. a top-level anima).

    Args:
        orphan_dir: Directory of the orphan anima.
        animas_dir: Root animas directory.

    Returns:
        Supervisor anima name, or ``None`` if no candidate found.
    """
    name = orphan_dir.name

    # 1) status.json in the orphan directory itself
    status_path = orphan_dir / "status.json"
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            sup = data.get("supervisor")
            if sup:
                return str(sup)
        except (json.JSONDecodeError, OSError):
            pass

    # 2) config.json global entry / 3) fallback to top-level anima
    try:
        config_path = animas_dir.parent / "config.json"
        config = load_config(config_path)
        anima_cfg = config.animas.get(name)
        if anima_cfg and anima_cfg.supervisor:
            return anima_cfg.supervisor
        # Fallback: first top-level anima (supervisor=None, identity.md exists)
        for pname, pcfg in config.animas.items():
            if pcfg.supervisor is None:
                candidate_dir = animas_dir / pname
                if candidate_dir.is_dir() and (candidate_dir / "identity.md").exists():
                    return pname
    except Exception:
        logger.debug("Failed to resolve orphan supervisor for '%s'", name, exc_info=True)

    return None


_TRIVIAL_ENTRIES = frozenset(
    {
        ".orphan_notified",
        "vectordb",
        "status.json",
        "index_meta.json",
        "identity.md",
    }
)


def _is_trivial_orphan(entry: Path) -> bool:
    """Return True if the orphan directory contains only trivial leftovers.

    Trivial entries are vectordb dirs, marker files, etc. that carry no
    user-meaningful state and can be safely auto-removed.
    """
    try:
        children = {c.name for c in entry.iterdir()}
    except OSError:
        return False
    return children.issubset(_TRIVIAL_ENTRIES)


def _auto_cleanup_orphan(entry: Path) -> bool:
    """Remove a trivial orphan directory tree and its config entry.

    Returns True on success.
    """
    try:
        shutil.rmtree(entry)
        logger.info("Orphan detection: auto-removed trivial orphan '%s'", entry.name)
    except OSError:
        logger.warning(
            "Orphan detection: failed to auto-remove '%s'",
            entry.name,
            exc_info=True,
        )
        return False

    # Also remove from config.json if present
    try:
        from core.config.models import unregister_anima_from_config

        data_dir = entry.parent.parent  # animas_dir -> data_dir
        if unregister_anima_from_config(data_dir, entry.name):
            logger.info(
                "Orphan detection: unregistered '%s' from config.json",
                entry.name,
            )
    except Exception:
        logger.debug(
            "Orphan detection: config cleanup skipped for '%s'",
            entry.name,
            exc_info=True,
        )

    return True


_ARCHIVE_MAX_AGE_DAYS = 30


def _archive_and_remove_orphan(entry: Path) -> bool:
    """Archive a non-trivial orphan directory, then remove the original.

    The directory is copied to ``~/.animaworks/archive/orphans/{name}_{timestamp}/``
    before deletion so that data can be recovered if needed.

    Returns True on success.
    """
    data_dir = entry.parent.parent  # animas/{name} -> animas -> data_dir
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    archive_dir = data_dir / "archive" / "orphans" / f"{entry.name}_{ts}"

    try:
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(entry, archive_dir)
    except OSError:
        logger.warning(
            "Orphan detection: failed to archive '%s' to '%s'",
            entry.name,
            archive_dir,
            exc_info=True,
        )
        return False

    try:
        shutil.rmtree(entry)
        logger.info(
            "Orphan detection: archived and removed non-trivial orphan '%s'",
            entry.name,
        )
    except OSError:
        logger.warning(
            "Orphan detection: archived '%s' but failed to remove original",
            entry.name,
            exc_info=True,
        )
        return False

    try:
        from core.config.models import unregister_anima_from_config

        if unregister_anima_from_config(data_dir, entry.name):
            logger.info(
                "Orphan detection: unregistered '%s' from config.json",
                entry.name,
            )
    except Exception:
        logger.debug(
            "Orphan detection: config cleanup skipped for '%s'",
            entry.name,
            exc_info=True,
        )

    return True


def cleanup_orphan_archives(
    data_dir: Path,
    max_age_days: int = _ARCHIVE_MAX_AGE_DAYS,
) -> int:
    """Remove orphan archive directories older than *max_age_days*.

    Args:
        data_dir: Runtime data directory (e.g. ``~/.animaworks/``).
        max_age_days: Maximum age in days before an archive is deleted.

    Returns:
        Number of archive directories removed.
    """
    archive_root = data_dir / "archive" / "orphans"
    if not archive_root.is_dir():
        return 0

    now = time.time()
    max_age_s = max_age_days * 86400
    removed = 0

    for entry in sorted(archive_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
            if (now - mtime) > max_age_s:
                shutil.rmtree(entry)
                logger.info(
                    "Orphan archive cleanup: removed expired archive '%s'",
                    entry.name,
                )
                removed += 1
        except OSError:
            logger.debug(
                "Orphan archive cleanup: failed to remove '%s'",
                entry.name,
                exc_info=True,
            )

    if removed:
        logger.info("Orphan archive cleanup: removed %d expired archive(s)", removed)

    return removed


def detect_orphan_animas(
    animas_dir: Path,
    shared_dir: Path,
    age_threshold_s: float = 300,
) -> list[dict[str, str]]:
    """Detect and auto-clean orphan anima directories.

    A directory is considered *orphan* when:

    - ``identity.md`` does not exist, **or**
    - ``identity.md`` is empty or contains only ``"未定義"``.

    Directories that start with ``_`` or ``.``, and directories younger than
    *age_threshold_s* seconds (possibly still being created) are skipped.

    **Trivial orphans** (containing only ``vectordb/``, ``.orphan_notified``,
    ``status.json``, or ``index_meta.json``) are automatically deleted.
    **Non-trivial orphans** (containing episodes, knowledge, state, etc.)
    are archived to ``archive/orphans/{name}_{timestamp}/`` and then deleted.

    Also runs :func:`cleanup_orphan_archives` to purge archives older than
    30 days.

    Args:
        animas_dir: Runtime animas directory (e.g. ``~/.animaworks/animas/``).
        shared_dir: Shared directory (kept for backward compat, no longer
            used for messaging).
        age_threshold_s: Minimum directory age in seconds before it is
            considered orphan.  Defaults to 300 (5 minutes).

    Returns:
        List of dicts with keys ``name`` and ``action``
        (``"auto_removed"``, ``"archived"``, or ``"skipped"``).
    """
    if not animas_dir.is_dir():
        return []

    now = time.time()
    results: list[dict[str, str]] = []

    for entry in sorted(animas_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Skip hidden / internal directories
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue

        # Check identity.md validity
        identity_path = entry / "identity.md"
        is_orphan = False
        if not identity_path.exists():
            is_orphan = True
        else:
            try:
                content = identity_path.read_text(encoding="utf-8").strip()
                if not content or content == "未定義":
                    is_orphan = True
            except OSError:
                is_orphan = True

        if not is_orphan:
            continue

        # Skip directories younger than age_threshold_s (possibly still being created)
        try:
            dir_mtime = entry.stat().st_mtime
            if (now - dir_mtime) < age_threshold_s:
                continue
        except OSError:
            continue

        # Trivial orphans → auto-remove
        if _is_trivial_orphan(entry):
            removed = _auto_cleanup_orphan(entry)
            results.append(
                {
                    "name": entry.name,
                    "action": "auto_removed" if removed else "skipped",
                }
            )
            continue

        # Non-trivial orphans → archive then remove
        logger.info(
            "Orphan detection: non-trivial orphan '%s' — archiving (contents: %s)",
            entry.name,
            ", ".join(sorted(c.name for c in entry.iterdir())),
        )
        archived = _archive_and_remove_orphan(entry)
        results.append(
            {
                "name": entry.name,
                "action": "archived" if archived else "skipped",
            }
        )

    if results:
        logger.info("Orphan detection: processed %d orphan(s)", len(results))
    else:
        logger.debug("Orphan detection: no orphans found")

    # Purge expired archives (30 days)
    data_dir = animas_dir.parent
    cleanup_orphan_archives(data_dir)

    return results
