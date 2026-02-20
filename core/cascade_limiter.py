from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Conversation depth limiter for preventing bilateral message cascades.

Tracks message exchange depth between Anima pairs within a sliding time
window.  When the depth exceeds the configured threshold, further sends
are blocked until the window expires.
"""

import logging
import time

from core.config.models import load_config

logger = logging.getLogger("animaworks.cascade_limiter")

# Kept for backward compatibility; authoritative values now live in HeartbeatConfig.
_DEPTH_WINDOW_S = 600      # 10 minutes
_MAX_DEPTH_DEFAULT = 6     # 6 turns = 3 round-trips


class ConversationDepthLimiter:
    """Track message exchange depth between Anima pairs within a time window.

    Module-level singleton: all Anima instances share the same state.
    """

    def __init__(
        self,
        window_s: float | None = None,
        max_depth: int | None = None,
    ) -> None:
        cfg = load_config()
        self._window_s = window_s if window_s is not None else cfg.heartbeat.depth_window_s
        self._max_depth = max_depth if max_depth is not None else cfg.heartbeat.max_depth
        self._exchanges: dict[tuple[str, str], list[float]] = {}

    def _pair_key(self, a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b)))  # type: ignore[return-value]

    def _evict_expired(self, key: tuple[str, str]) -> list[float]:
        now = time.monotonic()
        times = self._exchanges.get(key, [])
        times = [t for t in times if now - t < self._window_s]
        if times:
            self._exchanges[key] = times
        elif key in self._exchanges:
            del self._exchanges[key]
        return times

    def check_and_record(self, sender: str, receiver: str) -> bool:
        """Check if the exchange is allowed, and record it if so.

        Returns True if allowed, False if depth exceeded.
        """
        key = self._pair_key(sender, receiver)
        times = self._evict_expired(key)
        if len(times) >= self._max_depth:
            logger.warning(
                "DEPTH LIMIT: %s -> %s blocked (%d exchanges in %ds window)",
                sender, receiver, len(times), self._window_s,
            )
            return False
        times.append(time.monotonic())
        self._exchanges[key] = times
        return True

    def current_depth(self, a: str, b: str) -> int:
        """Return current exchange count for a pair within the active window."""
        key = self._pair_key(a, b)
        return len(self._evict_expired(key))


# Module-level singleton
depth_limiter = ConversationDepthLimiter()
