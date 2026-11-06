from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared read and locked-update access for per-Anima ``status.json`` files."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.platform.atomic_io import update_json
from core.platform.process_role import assert_settings_write_allowed

logger = logging.getLogger(__name__)


def read_status_checked(anima_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Read an Anima status object and report whether it was readable."""
    status_path = Path(anima_dir) / "status.json"
    try:
        with status_path.open(encoding="utf-8") as status_file:
            status = json.load(status_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("Unable to read %s: %s", status_path, exc)
        return False, {}
    if not isinstance(status, dict):
        logger.debug("Ignoring non-object status file %s", status_path)
        return False, {}
    return True, status


def read_status(anima_dir: Path) -> dict[str, Any]:
    """Read an Anima status object, falling back to an empty object on errors.

    This matches the tolerant behavior used by the primary status readers for
    missing, unreadable, or malformed files. Writers must use
    :func:`update_status`, which rejects malformed existing JSON instead of
    replacing it with an empty object.
    """
    _, status = read_status_checked(anima_dir)
    return status


def update_status(anima_dir: Path, fn: Callable[[dict[str, Any]], dict[str, Any] | None]) -> dict[str, Any]:
    """Atomically update one Anima's status while holding an inter-process lock.

    The callback runs under the ``status.json.lock`` exclusive flock. A missing
    status file starts as an empty object; invalid or non-object JSON raises
    without writing anything.
    """
    anima_dir = Path(anima_dir)
    assert_settings_write_allowed(anima_dir / "status.json")
    return update_json(anima_dir / "status.json", fn)


__all__ = ["read_status", "read_status_checked", "update_status"]
