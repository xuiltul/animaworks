from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for identifying message senders across CLI and core workflows."""

import logging

from core.anima.workspace_files import IDENTITY_FILE

logger = logging.getLogger("animaworks")


def resolve_sender_source(name: str) -> str:
    """Return ``anima`` for a registered/local Anima, otherwise ``human``."""
    from core.paths import get_animas_dir

    known: set[str] = set()
    try:
        from core.config.models import load_config

        known = set(load_config().animas.keys())
    except Exception:
        logger.debug("Could not load configured anima names", exc_info=True)
    if name in known:
        return "anima"
    try:
        if (get_animas_dir() / name / IDENTITY_FILE).exists():
            return "anima"
    except Exception:
        logger.debug("Could not inspect identity file for %s", name, exc_info=True)
    return "human"
