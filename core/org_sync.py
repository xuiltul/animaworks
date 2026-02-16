from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.

"""Periodic organizational structure synchronization.

Scans person directories, extracts supervisor relationships from identity.md
tables and status.json, and reconciles them with config.json entries.
Manual config overrides are never clobbered; mismatches are logged as warnings.
"""

import json
import logging
import time
from pathlib import Path

from core.config.models import (
    PersonModelConfig,
    load_config,
    read_person_supervisor,
    save_config,
)
from core.messenger import Messenger

logger = logging.getLogger(__name__)


# ── Circular reference detection ─────────────────────────────────


def _detect_circular_references(
    relationships: dict[str, str | None],
) -> list[tuple[str, ...]]:
    """Detect circular supervisor references.

    Args:
        relationships: Mapping of person name to supervisor name.

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
    persons_dir: Path,
    config_path: Path | None = None,
) -> dict[str, str | None]:
    """Sync organizational structure from person files to config.json.

    For each person directory:

    1. Extract supervisor from identity.md / status.json via
       :func:`~core.config.models.read_person_supervisor`.
    2. If config.json has no entry for this person, create one.
    3. If config.json has ``supervisor=None`` but a value was found, update it.
    4. If config.json already has a supervisor set, don't overwrite
       (respect manual config).
    5. Log warnings for mismatches.

    Args:
        persons_dir: Path to the persons directory
            (e.g. ``~/.animaworks/persons``).
        config_path: Optional explicit path to config.json.  When ``None``,
            the default location is resolved automatically.

    Returns:
        Dict of ``{person_name: supervisor_value}`` for all discovered entries.
    """
    if not persons_dir.is_dir():
        logger.debug("Persons directory does not exist: %s", persons_dir)
        return {}

    # ── Phase 1: discover supervisor relationships from disk ──────

    discovered: dict[str, str | None] = {}

    for person_dir in sorted(persons_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        if not (person_dir / "identity.md").exists():
            continue

        name = person_dir.name
        discovered[name] = read_person_supervisor(person_dir)

    if not discovered:
        logger.debug("No persons discovered in %s", persons_dir)
        return {}

    logger.info("Org sync: discovered %d persons from disk", len(discovered))

    # ── Phase 2: detect circular references ──────────────────────

    cycles = _detect_circular_references(discovered)
    circular_persons: set[str] = set()
    for cycle in cycles:
        circular_persons.update(cycle)
        logger.warning(
            "Org sync: circular supervisor reference detected: %s",
            " -> ".join(cycle) + " -> " + cycle[0],
        )

    # ── Phase 3: reconcile with config.json ──────────────────────

    config = load_config(config_path)
    changed = False

    for name, disk_supervisor in discovered.items():
        # Skip persons involved in circular references
        if name in circular_persons:
            logger.warning(
                "Org sync: skipping %s due to circular reference", name,
            )
            continue

        if name not in config.persons:
            # Person not yet in config — add with discovered supervisor
            config.persons[name] = PersonModelConfig(supervisor=disk_supervisor)
            changed = True
            logger.info(
                "Org sync: added person '%s' with supervisor=%s",
                name,
                disk_supervisor,
            )
            continue

        existing = config.persons[name]

        if existing.supervisor is None and disk_supervisor is not None:
            # Config has no supervisor but disk has one — fill it in
            existing.supervisor = disk_supervisor
            changed = True
            logger.info(
                "Org sync: set supervisor for '%s' to '%s' (was None)",
                name,
                disk_supervisor,
            )
        elif (
            existing.supervisor is not None
            and disk_supervisor is not None
            and existing.supervisor != disk_supervisor
        ):
            # Config already has a different supervisor — warn but don't overwrite
            logger.warning(
                "Org sync: supervisor mismatch for '%s': "
                "config='%s', identity.md='%s' (keeping config value)",
                name,
                existing.supervisor,
                disk_supervisor,
            )

    # ── Phase 4: persist if anything changed ─────────────────────

    if changed:
        save_config(config, config_path)
        logger.info("Org sync: config.json updated")
    else:
        logger.debug("Org sync: no changes needed")

    return discovered


# ── Orphan person detection ───────────────────────────────────────


def _find_orphan_supervisor(
    orphan_dir: Path,
    persons_dir: Path,
) -> str | None:
    """Determine which supervisor should be notified about an orphan person.

    Resolution order:
        1. ``status.json`` in *orphan_dir* — ``supervisor`` field.
        2. ``config.json`` (global) — ``persons.<name>.supervisor``.
        3. Fallback: first person in *persons_dir* whose ``config.json``
           entry has ``supervisor=None`` and whose ``identity.md`` exists
           (i.e. a top-level person).

    Args:
        orphan_dir: Directory of the orphan person.
        persons_dir: Root persons directory.

    Returns:
        Supervisor person name, or ``None`` if no candidate found.
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

    # 2) config.json global entry / 3) fallback to top-level person
    try:
        config = load_config()
        person_cfg = config.persons.get(name)
        if person_cfg and person_cfg.supervisor:
            return person_cfg.supervisor
        # Fallback: first top-level person (supervisor=None, identity.md exists)
        for pname, pcfg in config.persons.items():
            if pcfg.supervisor is None:
                candidate_dir = persons_dir / pname
                if candidate_dir.is_dir() and (candidate_dir / "identity.md").exists():
                    return pname
    except Exception:
        pass

    return None


def detect_orphan_persons(
    persons_dir: Path,
    shared_dir: Path,
    age_threshold_s: float = 300,
) -> list[dict[str, str]]:
    """Detect person directories missing a valid identity.md and notify supervisors.

    A directory is considered *orphan* when:

    - ``identity.md`` does not exist, **or**
    - ``identity.md`` is empty or contains only ``"未定義"``.

    Directories that start with ``_`` or ``.``, directories younger than
    *age_threshold_s* seconds (possibly still being created), and directories
    already bearing a ``.orphan_notified`` marker are skipped.

    For each orphan the function:

    1. Resolves a supervisor via :func:`_find_orphan_supervisor`.
    2. Sends a ``system_alert`` message through :class:`~core.messenger.Messenger`.
    3. Writes a ``.orphan_notified`` marker so repeated runs don't re-notify.

    Args:
        persons_dir: Runtime persons directory (e.g. ``~/.animaworks/persons/``).
        shared_dir: Shared directory used by Messenger.
        age_threshold_s: Minimum directory age in seconds before it is
            considered orphan.  Defaults to 300 (5 minutes).

    Returns:
        List of dicts with keys ``name``, ``supervisor``, and ``notified``
        (``"yes"`` or ``"no"``).
    """
    if not persons_dir.is_dir():
        return []

    now = time.time()
    results: list[dict[str, str]] = []

    for entry in sorted(persons_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Skip hidden / internal directories
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue

        # Skip if already notified
        marker = entry / ".orphan_notified"
        if marker.exists():
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

        # Resolve supervisor and send notification
        supervisor = _find_orphan_supervisor(entry, persons_dir)
        notified = "no"

        if supervisor:
            try:
                messenger = Messenger(shared_dir, "system")
                messenger.send(
                    to=supervisor,
                    content=(
                        f"[Orphan Detection] Person directory '{entry.name}' "
                        f"has no valid identity.md.  Please review and either "
                        f"complete setup or remove the directory."
                    ),
                    msg_type="system_alert",
                )
                notified = "yes"
                logger.info(
                    "Orphan detection: notified '%s' about orphan '%s'",
                    supervisor,
                    entry.name,
                )
            except Exception:
                logger.exception(
                    "Orphan detection: failed to notify '%s' about '%s'",
                    supervisor,
                    entry.name,
                )

        # Write marker regardless of notification success so we don't retry
        try:
            marker.write_text(
                f"notified={notified} supervisor={supervisor or 'none'}\n",
                encoding="utf-8",
            )
        except OSError:
            logger.warning(
                "Orphan detection: could not write marker for '%s'",
                entry.name,
            )

        results.append({
            "name": entry.name,
            "supervisor": supervisor or "",
            "notified": notified,
        })

    if results:
        logger.info("Orphan detection: found %d orphan(s)", len(results))
    else:
        logger.debug("Orphan detection: no orphans found")

    return results
