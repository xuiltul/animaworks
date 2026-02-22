from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""System-reminder injection queue for execution loops.

Each executor holds a SystemReminderQueue. Event producers (context tracker,
tool handler, etc.) push reminder content; the executor drains the queue
each iteration and injects ``<system-reminder>`` blocks into the conversation.
"""

import asyncio
import logging
from collections import deque

logger = logging.getLogger(__name__)

# Maximum queued reminders before oldest are dropped
_MAX_QUEUE_SIZE = 10

# ── Standard reminder messages ──────────────────────────────
MSG_CONTEXT_THRESHOLD = "コンテキスト使用量: {ratio:.0%}。出力を簡潔にし、重要な状態をセッション状態に保存せよ。"
MSG_OUTPUT_TRUNCATED = "出力がmax_tokensで途切れた。残りの内容を小さく分割して続行せよ。"


class SystemReminderQueue:
    """Thread-safe queue for system-reminder injection into execution loops.

    Event sources push short text snippets via :meth:`push`.  The executor
    calls :meth:`drain` each iteration; if any content is queued, it is
    concatenated into a single ``<system-reminder>`` block.
    """

    def __init__(self, max_size: int = _MAX_QUEUE_SIZE) -> None:
        self._items: deque[str] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def push(self, content: str) -> None:
        """Enqueue a reminder.  Oldest entry is dropped if queue is full."""
        async with self._lock:
            if len(self._items) >= self._items.maxlen:
                dropped = self._items.popleft()
                logger.debug("SystemReminderQueue overflow; dropped: %s", dropped[:80])
            self._items.append(content)

    def push_sync(self, content: str) -> None:
        """Synchronous push (for use from sync code paths)."""
        if len(self._items) >= self._items.maxlen:
            dropped = self._items.popleft()
            logger.debug("SystemReminderQueue overflow; dropped: %s", dropped[:80])
        self._items.append(content)

    async def drain(self) -> str | None:
        """Drain all queued items, returning combined content or None."""
        async with self._lock:
            if not self._items:
                return None
            items = list(self._items)
            self._items.clear()
        return "\n\n".join(items)

    def drain_sync(self) -> str | None:
        """Synchronous drain (for use from sync code paths)."""
        if not self._items:
            return None
        items = list(self._items)
        self._items.clear()
        return "\n\n".join(items)

    @staticmethod
    def format_reminder(content: str) -> str:
        """Wrap content in <system-reminder> tags."""
        return f"<system-reminder>\n{content}\n</system-reminder>"

    def drain_formatted(self) -> str | None:
        """Drain and format as <system-reminder> block. Sync."""
        raw = self.drain_sync()
        if raw is None:
            return None
        return self.format_reminder(raw)

    async def drain_formatted_async(self) -> str | None:
        """Drain and format as <system-reminder> block. Async."""
        raw = await self.drain()
        if raw is None:
            return None
        return self.format_reminder(raw)
