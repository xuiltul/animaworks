from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cached roster of known Anima names (active + tombstoned / on-disk).

Used to prevent anima names from being treated as human conversation
log directories under ``shared/users/``.
"""

import json
import logging
from pathlib import Path

from core.paths import get_animas_dir, get_data_dir

logger = logging.getLogger("animaworks.anima_roster")

_anima_name_cache: frozenset[str] | None = None


def load_anima_names(data_dir: Path | None = None) -> set[str]:
    """Load anima names from disk (animas/ dirs + config.json keys).

    Directory entries include tombstoned (disabled) animas whose directories
    are retained. Completely deleted names are not included.
    """
    root = data_dir if data_dir is not None else get_data_dir()
    names: set[str] = set()

    animas_dir = (root / "animas") if data_dir is not None else get_animas_dir()

    if animas_dir.is_dir():
        for entry in animas_dir.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                names.add(entry.name)

    config_path = root / "config.json"
    if config_path.is_file():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("Failed to read config.json for anima roster", exc_info=True)
        else:
            animas_section = raw.get("animas")
            if isinstance(animas_section, dict):
                for key in animas_section:
                    if isinstance(key, str) and key:
                        names.add(key)

    return names


def get_anima_roster() -> frozenset[str]:
    """Return the cached anima name roster, loading on first use."""
    global _anima_name_cache
    if _anima_name_cache is None:
        _anima_name_cache = frozenset(load_anima_names())
        logger.debug("Loaded anima roster cache (%d names)", len(_anima_name_cache))
    return _anima_name_cache


def refresh_anima_roster() -> frozenset[str]:
    """Reload anima names from disk and update the cache."""
    global _anima_name_cache
    _anima_name_cache = frozenset(load_anima_names())
    logger.debug("Refreshed anima roster cache (%d names)", len(_anima_name_cache))
    return _anima_name_cache


def invalidate_anima_roster() -> None:
    """Clear the cached roster so the next access reloads from disk."""
    global _anima_name_cache
    _anima_name_cache = None


def is_anima_name(name: str) -> bool:
    """Return True if *name* is an exact match for a known anima name."""
    if not name:
        return False
    return name in get_anima_roster()
