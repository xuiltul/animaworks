# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Shared retry/backoff utility for AnimaWorks tools."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Constants ──────────────────────────────────────────────

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0


# ── Public API ─────────────────────────────────────────────


def retry_with_backoff(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _DEFAULT_BASE_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> T:
    """Execute *fn* with exponential backoff retry.

    Args:
        fn: The callable to execute.
        *args: Positional arguments forwarded to *fn*.
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Cap on the computed backoff delay.
        retry_on: Tuple of exception types that trigger a retry.
        on_retry: Optional callback ``(exc, attempt, wait)`` invoked
            before each retry sleep.  Useful for custom logging or
            extracting ``Retry-After`` headers.
        sleep_fn: Sleep function (default ``time.sleep``).  Allows
            callers to swap in an async-compatible sleep or a no-op
            for testing.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last caught exception when all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return fn(*args, **kwargs)
        except retry_on as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            wait = min(base_delay * (2 ** attempt), max_delay)
            if on_retry is not None:
                on_retry(exc, attempt + 1, wait)
            else:
                logger.warning(
                    "Retry %d/%d after %.1fs – %s: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    type(exc).__name__,
                    exc,
                )
            sleep_fn(wait)
    raise last_exc  # type: ignore[misc]


def retry_on_rate_limit(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 5,
    default_wait: float = 30.0,
    get_retry_after: Callable[[Exception], float | None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> T:
    """Execute *fn* with rate-limit-aware retry.

    Unlike :func:`retry_with_backoff` this function honours a
    ``Retry-After`` header (or equivalent) extracted via
    *get_retry_after* instead of computing exponential backoff.

    Args:
        fn: The callable to execute.
        *args: Positional arguments forwarded to *fn*.
        max_retries: Maximum number of retry attempts.
        default_wait: Fallback wait time when ``Retry-After`` is
            unavailable.
        get_retry_after: Callable that receives the caught exception
            and returns a wait time in seconds, or ``None`` to use
            *default_wait*.
        sleep_fn: Sleep function (default ``time.sleep``).
        retry_on: Tuple of exception types that trigger a retry.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last caught exception when all retries are exhausted, or
        ``RuntimeError`` when retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return fn(*args, **kwargs)
        except retry_on as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            wait = default_wait
            if get_retry_after is not None:
                extracted = get_retry_after(exc)
                if extracted is not None:
                    wait = extracted
            logger.warning(
                "Rate limited – retry %d/%d after %.0fs: %s",
                attempt + 1,
                max_retries,
                wait,
                exc,
            )
            sleep_fn(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Rate limit retry exhausted")  # pragma: no cover
