from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared observability helpers for fact extraction and related indexes."""

import logging
import time
from types import TracebackType

_DEFAULT_RATE_LIMIT_SECONDS = 3600.0
_LAST_WARNING_AT: dict[str, float] = {}

ExcInfo = bool | tuple[type[BaseException], BaseException, TracebackType | None]


def warn_rate_limited(
    logger: logging.Logger,
    key: str,
    message: str,
    *args: object,
    exc_info: ExcInfo = False,
    rate_limit_seconds: float = _DEFAULT_RATE_LIMIT_SECONDS,
) -> bool:
    """Emit a warning at most once per rate-limit window for *key*."""

    now = time.monotonic()
    last = _LAST_WARNING_AT.get(key)
    if last is not None and now - last < rate_limit_seconds:
        return False
    _LAST_WARNING_AT[key] = now
    logger.warning(message, *args, exc_info=exc_info)
    return True


def reset_warning_rate_limits() -> None:
    """Clear warning rate-limit state for tests."""

    _LAST_WARNING_AT.clear()
