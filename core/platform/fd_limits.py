from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-level file descriptor limit helpers."""

import logging

_NOFILE_INFINITY_FALLBACK = 1_048_576

_logger = logging.getLogger(__name__)


def raise_fd_soft_limit(
    *,
    logger: logging.Logger | None = None,
    process_label: str = "process",
) -> tuple[int, int]:
    """Raise the process soft RLIMIT_NOFILE to the hard limit when possible."""
    active_logger = logger or _logger
    try:
        import resource
    except ImportError:
        active_logger.debug("resource module unavailable; skipping %s RLIMIT_NOFILE raise", process_label)
        return (0, 0)

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        active_logger.warning("Failed to inspect %s RLIMIT_NOFILE", process_label, exc_info=True)
        return (0, 0)

    infinity = getattr(resource, "RLIM_INFINITY", -1)
    target = _NOFILE_INFINITY_FALLBACK if hard == infinity else hard
    if soft == infinity or soft >= target:
        return soft, hard

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        active_logger.warning(
            "Failed to raise %s RLIMIT_NOFILE soft limit: %s -> %s (hard=%s)",
            process_label,
            soft,
            target,
            hard,
            exc_info=True,
        )
        return soft, hard

    active_logger.info(
        "Raised %s RLIMIT_NOFILE soft limit: %s -> %s (hard=%s)",
        process_label,
        soft,
        target,
        hard,
    )
    return target, hard
