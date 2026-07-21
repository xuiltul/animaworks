from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic checks for memory files that need semantic cleanup."""

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.time_utils import today_local

logger = logging.getLogger("animaworks.memory.hygiene")

_REPORT_CATEGORIES = (
    "merged_leftovers",
    "inherited_dirs",
    "mdc_files",
    "oversized_knowledge",
    "noncanonical_archive_dirs",
    "noncanonical_episodes",
)
_NONCANONICAL_ARCHIVE_NAMES = ("archived", "_archived", ".archive")
_OVERSIZED_KNOWLEDGE_BYTES = 32 * 1024
# Canonical episode names: YYYY-MM-DD.md or YYYY-MM-DD_<suffix>.md
_CANONICAL_EPISODE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}(_[A-Za-z0-9._-]+)?\.md$")


def scan_memory_hygiene(anima_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan one Anima's active knowledge tree and persist its hygiene report.

    The scan never changes knowledge files.  The only write is the report at
    ``state/memory_hygiene.json``.  Existing ``first_seen`` dates are retained
    for paths that remain in the same category.
    """
    anima_dir = Path(anima_dir)
    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"
    report_path = anima_dir / "state" / "memory_hygiene.json"
    today_date = today_local()
    today = today_date.isoformat()
    previous = _load_previous_report(report_path, today_date)

    report: dict[str, list[dict[str, Any]]] = {category: [] for category in _REPORT_CATEGORIES}
    if knowledge_dir.is_dir():
        active_files = sorted(
            path for path in knowledge_dir.rglob("*") if path.is_file() and not _is_in_archive(path, knowledge_dir)
        )

        for path in active_files:
            relative = path.relative_to(anima_dir).as_posix()
            if path.name.startswith("_merged_"):
                report["merged_leftovers"].append(_entry(relative, previous["merged_leftovers"], today))
            if path.suffix == ".mdc":
                report["mdc_files"].append(_entry(relative, previous["mdc_files"], today))
            if path.suffix == ".md" and path.stat().st_size > _OVERSIZED_KNOWLEDGE_BYTES:
                report["oversized_knowledge"].append(
                    _entry(
                        relative,
                        previous["oversized_knowledge"],
                        today,
                        size_bytes=path.stat().st_size,
                    )
                )

        for path in sorted(knowledge_dir.glob("inherited-*")):
            if path.is_dir():
                relative = path.relative_to(anima_dir).as_posix()
                report["inherited_dirs"].append(_entry(relative, previous["inherited_dirs"], today))

        for name in _NONCANONICAL_ARCHIVE_NAMES:
            path = knowledge_dir / name
            if path.is_dir():
                relative = path.relative_to(anima_dir).as_posix()
                report["noncanonical_archive_dirs"].append(
                    _entry(relative, previous["noncanonical_archive_dirs"], today)
                )

    if episodes_dir.is_dir():
        for path in sorted(episodes_dir.iterdir()):
            if not path.is_file():
                continue
            if _CANONICAL_EPISODE_NAME.match(path.name):
                continue
            relative = path.relative_to(anima_dir).as_posix()
            report["noncanonical_episodes"].append(
                _entry(relative, previous["noncanonical_episodes"], today)
            )

    atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _is_in_archive(path: Path, knowledge_dir: Path) -> bool:
    """Return whether *path* is below a canonical ``archive/`` directory."""
    return "archive" in path.relative_to(knowledge_dir).parts[:-1]


def _entry(
    path: str,
    previous: dict[str, str],
    today: str,
    **extra: Any,
) -> dict[str, Any]:
    return {"path": path, "first_seen": previous.get(path, today), **extra}


def _load_previous_report(report_path: Path, today: date) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {category: {} for category in _REPORT_CATEGORIES}
    if not report_path.is_file():
        return indexed
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring invalid memory hygiene report: %s", report_path, exc_info=True)
        return indexed
    if not isinstance(payload, dict):
        return indexed
    for category in _REPORT_CATEGORIES:
        entries = payload.get(category, [])
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            first_seen = item.get("first_seen")
            if isinstance(path, str) and isinstance(first_seen, str):
                indexed[category][path] = _validated_first_seen(first_seen, today)
    return indexed


def _validated_first_seen(first_seen: str, today: date) -> str:
    """Return a valid, non-future ISO date or reset it to today."""
    try:
        parsed = date.fromisoformat(first_seen)
    except ValueError:
        return today.isoformat()
    if parsed > today:
        return today.isoformat()
    return parsed.isoformat()


__all__ = ["scan_memory_hygiene"]
