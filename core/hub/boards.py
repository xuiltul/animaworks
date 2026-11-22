from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Board names that belong to the hub."""

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def _is_valid_board_name(name: str) -> bool:
    """Return whether *name* follows the existing Board name rules."""
    from core.messaging.messenger import _validate_name

    try:
        _validate_name(name, "board name")
    except Exception:
        return False
    return True


def _extra_board_candidates(value: object) -> Iterable[object]:
    if isinstance(value, (list, tuple)):
        return value
    if value is None:
        return ()
    logger.warning("Ignoring hub.extra_boards because it is not a list: %r", value)
    return ()


def hub_boards(config: Any) -> list[str]:
    """Return the hub Board names in priority order.

    The configured ``board`` is always first (falling back to ``hub``), followed
    by the valid, non-empty, unique names from ``extra_boards``.  Invalid extra
    names are ignored with a warning instead of breaking the hub surface.
    """
    primary = str(getattr(config, "board", "hub") or "hub")
    result = [primary]
    seen = {primary}

    for raw in _extra_board_candidates(getattr(config, "extra_boards", None)):
        if not isinstance(raw, str):
            logger.warning("Ignoring non-string hub.extra_boards entry: %r", raw)
            continue
        name = raw.strip()
        if not name:
            continue
        if name in seen:
            continue
        if not _is_valid_board_name(name):
            logger.warning("Ignoring invalid hub.extra_boards entry: %r", raw)
            continue
        seen.add(name)
        result.append(name)
    return result
