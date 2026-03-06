# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Per-Anima × per-thread_id idle compaction timer management.

Manages compaction timers using asyncio.Handle (via loop.call_later).
When a session goes idle for ``idle_minutes``, mode-specific compaction
runs (SDK compact, conversation compress, shortterm save, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.anima import DigitalAnima

logger = logging.getLogger("animaworks.session_compactor")

# LRU limit for _timers (same as conversation_locks).
_MAX_TIMERS = 20


# ── SessionCompactor ──────────────────────────────────────────


class SessionCompactor:
    """Anima別 × thread_id別のアイドルコンパクションタイマー管理。"""

    def __init__(self, idle_minutes: float = 10.0) -> None:
        """Initialize compactor with idle threshold in minutes.

        Args:
            idle_minutes: Minutes of inactivity before compaction runs.
        """
        self._idle_minutes = idle_minutes
        # key: (anima_name, thread_id) → asyncio.Handle
        self._timers: dict[tuple[str, str], asyncio.Handle] = {}

    def schedule(
        self,
        anima_name: str,
        thread_id: str,
        callback: Callable[[], None],
    ) -> None:
        """Schedule a compaction timer (cancel existing if any, then reschedule).

        Args:
            anima_name: Anima name.
            thread_id: Thread/conversation ID.
            callback: Sync callable invoked when timer fires (e.g. schedules
                asyncio.create_task(run_idle_compaction(...))).
        """
        key = (anima_name, thread_id)
        if key in self._timers:
            self._timers[key].cancel()
            del self._timers[key]

        while len(self._timers) >= _MAX_TIMERS:
            oldest = next(iter(self._timers))
            h = self._timers.pop(oldest)
            h.cancel()
            logger.debug("SessionCompactor LRU evict: %s", oldest)

        loop = asyncio.get_running_loop()
        handle = loop.call_later(
            self._idle_minutes * 60,
            self._fire,
            key,
            callback,
        )
        self._timers[key] = handle

    def _fire(self, key: tuple[str, str], callback: Callable[[], None]) -> None:
        """Invoked when timer fires; removes handle and invokes callback."""
        self._timers.pop(key, None)
        try:
            callback()
        except Exception:
            logger.exception("SessionCompactor callback failed for %s", key)

    def cancel(self, anima_name: str, thread_id: str) -> None:
        """Cancel the timer for the given anima and thread."""
        key = (anima_name, thread_id)
        if key in self._timers:
            self._timers[key].cancel()
            del self._timers[key]

    def cancel_all_for_anima(self, anima_name: str) -> None:
        """Cancel all timers for an anima (e.g. on anima stop)."""
        to_remove = [k for k in self._timers if k[0] == anima_name]
        for key in to_remove:
            self._timers[key].cancel()
            del self._timers[key]

    def shutdown(self) -> None:
        """Cancel all timers (e.g. on server shutdown)."""
        for handle in self._timers.values():
            handle.cancel()
        self._timers.clear()


# ── Mode-specific compaction ──────────────────────────────────


async def _compact_mode_s(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode S: SDK compact_session (resume + /compact)."""
    executor = anima.agent._executor
    if not hasattr(executor, "compact_session"):
        return False
    result = await executor.compact_session(
        anima_dir=anima.anima_dir,
        session_type="chat",
        thread_id=thread_id,
    )
    return result


async def _compact_mode_a(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode A: conversation compress_if_needed + finalize_if_session_ended."""
    from core.memory.conversation import ConversationMemory

    conv = ConversationMemory(anima.anima_dir, anima.agent.model_config, thread_id=thread_id)
    compressed = await conv.compress_if_needed()
    await conv.finalize_if_session_ended()
    return compressed


async def _compact_mode_b(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode B: same as Mode A."""
    return await _compact_mode_a(anima, thread_id)


async def _compact_mode_c(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode C: conversation compress + shortterm save + codex thread discard."""
    from core.execution.codex_sdk import _clear_thread_id
    from core.memory.conversation import ConversationMemory
    from core.memory.shortterm import SessionState, ShortTermMemory
    from core.time_utils import now_jst

    conv = ConversationMemory(anima.anima_dir, anima.agent.model_config, thread_id=thread_id)
    await conv.compress_if_needed()

    state = conv.load()
    summary_parts: list[str] = []
    if state.compressed_summary:
        summary_parts.append(state.compressed_summary)
    if state.turns:
        for turn in state.turns[-3:]:
            summary_parts.append(f"{turn.role}: {turn.content[:200]}")

    shortterm = ShortTermMemory(anima.anima_dir, session_type="chat", thread_id=thread_id)
    shortterm.save(
        SessionState(
            accumulated_response="\n".join(summary_parts)[:4000],
            timestamp=now_jst().isoformat(),
            trigger="idle_compaction",
            notes="Auto-saved before Codex thread discard",
        )
    )

    _clear_thread_id(anima.anima_dir, "chat", thread_id)
    await conv.finalize_if_session_ended()
    return True


# ── Public API ────────────────────────────────────────────────


async def run_idle_compaction(anima: DigitalAnima, thread_id: str) -> None:
    """Run mode-specific idle compaction for the given anima and thread.

    Acquires the thread lock with a 30-second timeout. If the lock cannot
    be acquired, compaction is skipped. Logs an "idle_compaction" activity
    event on success.
    """
    mode = anima.agent.execution_mode
    lock = anima._get_thread_lock(thread_id)

    try:
        await asyncio.wait_for(lock.acquire(), timeout=30)
    except TimeoutError:
        logger.warning(
            "Idle compaction skipped for %s/%s: lock timeout",
            anima.name,
            thread_id,
        )
        return

    try:
        if mode == "s":
            ok = await _compact_mode_s(anima, thread_id)
            if not ok:
                await _compact_mode_a(anima, thread_id)
        elif mode == "a":
            await _compact_mode_a(anima, thread_id)
        elif mode == "c":
            await _compact_mode_c(anima, thread_id)
        elif mode == "b":
            await _compact_mode_b(anima, thread_id)
        else:
            await _compact_mode_a(anima, thread_id)
    except Exception:
        logger.exception("Idle compaction failed for %s/%s", anima.name, thread_id)
        return
    finally:
        lock.release()

    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima.anima_dir)
        activity.log(
            "idle_compaction",
            summary=f"Idle compaction completed (mode={mode}, thread={thread_id})",
            meta={"mode": mode, "thread_id": thread_id},
        )
    except Exception:
        logger.debug("Failed to log idle_compaction activity", exc_info=True)
