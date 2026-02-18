# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Async compatibility helpers for tools with synchronous HTTP clients."""
from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Public API ─────────────────────────────────────────────


async def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in a thread-pool executor.

    This prevents blocking the asyncio event loop when calling
    synchronous HTTP libraries (``requests``, ``slack_sdk``, etc.).

    Args:
        fn: The synchronous callable.
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))
