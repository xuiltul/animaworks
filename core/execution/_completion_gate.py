from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Shared completion-gate helpers used by multiple execution modes.

Mode S injects the checklist directly via the Stop hook's ``reason``
parameter — no tool call is needed.  Mode A requires the agent to call
the ``completion_gate`` tool; the marker file is the IPC signal between
the tool handler and the loop.  Helpers here (marker I/O, trigger
filtering) are used by both paths.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def completion_gate_marker_path(anima_dir: Path) -> Path:
    """Path to the IPC marker written when ``completion_gate`` is invoked."""
    return anima_dir / "run" / "completion_gate_called"


def gate_marker_exists(anima_dir: Path) -> bool:
    """Return True if the completion gate marker file exists."""
    return completion_gate_marker_path(anima_dir).is_file()


def cleanup_gate_marker(anima_dir: Path) -> None:
    """Remove the completion gate marker if present.  Ignores missing file."""
    p = completion_gate_marker_path(anima_dir)
    try:
        if p.exists():
            p.unlink()
    except OSError:
        logger.debug("Failed to cleanup completion gate marker", exc_info=True)


def completion_gate_applies_to_trigger(trigger: str | None) -> bool:
    """Return True when pre-completion verification applies.

    Gating is skipped for heartbeat and inbox triggers.
    """
    if trigger is None:
        return True
    if trigger == "heartbeat":
        return False
    return not trigger.startswith("inbox")
