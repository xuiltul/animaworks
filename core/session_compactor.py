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
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.anima import DigitalAnima

logger = logging.getLogger("animaworks.session_compactor")

# ── Activity-log extraction constants ─────────────────────
_MAX_CONVERSATION_ROUNDS = 3
_MAX_TOOL_ENTRIES = 10
_CHAT_ENTRY_TYPES = frozenset(
    {
        "message_received",
        "response_sent",
        "tool_use",
        "tool_result",
    }
)
_TOOL_INPUT_TRUNCATE = 500
_TOOL_RESULT_TRUNCATE = 500

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
            logger.debug("SessionCompactor rescheduled: %s (%.1f min)", key, self._idle_minutes)
        else:
            logger.debug("SessionCompactor scheduled: %s (%.1f min)", key, self._idle_minutes)

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
        logger.info("SessionCompactor timer fired: %s", key)
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
            logger.debug("SessionCompactor cancelled: %s", key)

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


# ── Activity-log based context extraction ─────────────────────


def _extract_recent_chat_context(anima_dir: Path) -> dict[str, Any]:
    """Extract recent chat context from today's activity_log.

    Scans the current day's log file in reverse to collect the most
    recent chat session entries:

    - Up to ``_MAX_CONVERSATION_ROUNDS`` user/assistant exchange rounds
    - Up to ``_MAX_TOOL_ENTRIES`` tool_use + tool_result pairs

    Returns a dict with keys matching ``SessionState`` fields:
    ``accumulated_response``, ``tool_uses``, ``original_prompt``,
    ``timestamp``, ``trigger``, ``notes``.
    """
    from core.time_utils import now_local

    log_dir = anima_dir / "activity_log"
    today_file = log_dir / f"{now_local().date().isoformat()}.jsonl"

    if not today_file.exists():
        return {}

    raw_entries: list[dict[str, Any]] = []
    try:
        with today_file.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = entry.get("type") or entry.get("event", "")
                if etype in _CHAT_ENTRY_TYPES:
                    if etype == "message_received":
                        from_type = (entry.get("meta") or {}).get("from_type", "")
                        if from_type != "human":
                            continue
                    raw_entries.append(entry)
    except OSError:
        logger.warning("Failed to read activity log for context extraction", exc_info=True)
        return {}

    if not raw_entries:
        return {}

    user_messages: list[dict[str, str]] = []
    assistant_messages: list[dict[str, str]] = []
    tool_entries: list[dict[str, Any]] = []

    for entry in reversed(raw_entries):
        etype = entry.get("type") or entry.get("event", "")
        ts = entry.get("ts") or entry.get("timestamp", "")

        if etype == "message_received" and len(user_messages) < _MAX_CONVERSATION_ROUNDS:
            content = entry.get("content", "")
            user_messages.append({"ts": ts, "content": content})

        elif etype == "response_sent" and len(assistant_messages) < _MAX_CONVERSATION_ROUNDS:
            content = entry.get("content", "") or entry.get("summary", "")
            assistant_messages.append({"ts": ts, "content": content})

        elif etype == "tool_use" and len(tool_entries) < _MAX_TOOL_ENTRIES:
            meta = entry.get("meta") or {}
            tool_name = entry.get("tool", "")
            args = meta.get("args", {})
            tool_use_id = meta.get("tool_use_id", "")
            tool_entries.append(
                {
                    "type": "tool_use",
                    "tool_use_id": tool_use_id,
                    "name": tool_name or entry.get("content", "")[:100],
                    "input": str(args)[:_TOOL_INPUT_TRUNCATE]
                    if args
                    else entry.get("content", "")[:_TOOL_INPUT_TRUNCATE],
                }
            )

        elif etype == "tool_result" and len(tool_entries) < _MAX_TOOL_ENTRIES:
            meta = entry.get("meta") or {}
            tool_use_id = meta.get("tool_use_id", "")
            tool_entries.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "result": entry.get("content", "")[:_TOOL_RESULT_TRUNCATE],
                    "is_error": meta.get("is_error", False),
                }
            )

    user_messages.reverse()
    assistant_messages.reverse()
    tool_entries.reverse()

    conversation_parts: list[str] = []
    for u, a in zip(user_messages, assistant_messages, strict=False):
        conversation_parts.append(f"user: {u['content']}")
        conversation_parts.append(f"assistant: {a['content']}")
    if len(user_messages) > len(assistant_messages):
        for u in user_messages[len(assistant_messages) :]:
            conversation_parts.append(f"user: {u['content']}")
    elif len(assistant_messages) > len(user_messages):
        for a in assistant_messages[len(user_messages) :]:
            conversation_parts.append(f"assistant: {a['content']}")

    tool_uses: list[dict[str, Any]] = []
    pending_use: dict[str, Any] | None = None
    for te in tool_entries:
        if te["type"] == "tool_use":
            if pending_use:
                tool_uses.append(pending_use)
            pending_use = {"name": te["name"], "input": te["input"]}
        elif te["type"] == "tool_result":
            if pending_use and te.get("tool_use_id") == pending_use.get("_tool_use_id", ""):
                pending_use["result"] = te["result"]
                tool_uses.append(pending_use)
                pending_use = None
            elif pending_use:
                tool_uses.append(pending_use)
                pending_use = None
                tool_uses.append({"name": "tool_result", "input": "", "result": te["result"]})
            else:
                tool_uses.append({"name": "tool_result", "input": "", "result": te["result"]})
    if pending_use:
        tool_uses.append(pending_use)

    original_prompt = user_messages[0]["content"] if user_messages else ""

    return {
        "accumulated_response": "\n".join(conversation_parts)[:8000],
        "tool_uses": tool_uses[-_MAX_TOOL_ENTRIES:],
        "original_prompt": original_prompt[:2000],
        "timestamp": now_local().isoformat(),
        "trigger": "idle_compaction",
        "notes": "Auto-extracted from activity_log (session discarded)",
    }


# ── Mode-specific compaction ──────────────────────────────────


async def _compact_mode_s(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode S: activity_log extraction → shortterm save → session_id clear.

    Extracts recent chat context from the activity_log, saves it to
    ShortTermMemory, and clears the SDK session_id. The next chat
    starts as a fresh session with shortterm injected into the system
    prompt via ``inject_shortterm``.

    Does NOT send ``/compact`` to the SDK — the session is discarded,
    not compacted, ensuring predictable post-compaction context size.
    """
    from core.execution._sdk_session import SESSION_TYPE_CHAT, _clear_session_id
    from core.memory.shortterm import SessionState, ShortTermMemory

    logger.debug("_compact_mode_s: entry (anima=%s, thread=%s)", anima.name, thread_id)

    ctx = _extract_recent_chat_context(anima.anima_dir)

    if ctx.get("accumulated_response") or ctx.get("tool_uses"):
        shortterm = ShortTermMemory(anima.anima_dir, session_type="chat", thread_id=thread_id)
        shortterm.save(
            SessionState(
                accumulated_response=ctx.get("accumulated_response", ""),
                tool_uses=ctx.get("tool_uses", []),
                original_prompt=ctx.get("original_prompt", ""),
                timestamp=ctx.get("timestamp", ""),
                trigger=ctx.get("trigger", "idle_compaction"),
                notes=ctx.get("notes", ""),
            )
        )
        logger.info("_compact_mode_s: shortterm saved from activity_log")

    _clear_session_id(anima.anima_dir, SESSION_TYPE_CHAT, thread_id)
    logger.info("_compact_mode_s: session_id cleared (anima=%s, thread=%s)", anima.name, thread_id)
    return True


async def _compact_mode_a(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode A: conversation compress + shortterm save + finalize."""
    from core.memory.conversation import ConversationMemory
    from core.memory.shortterm import SessionState, ShortTermMemory
    from core.time_utils import now_local

    logger.debug("_compact_mode_a: entry (anima=%s, thread=%s)", anima.name, thread_id)
    conv = ConversationMemory(anima.anima_dir, anima.agent.model_config, thread_id=thread_id)
    compressed = await conv.compress_if_needed()

    state = conv.load()
    summary_parts: list[str] = []
    if state.compressed_summary:
        summary_parts.append(state.compressed_summary)
    if state.turns:
        for turn in state.turns[-3:]:
            summary_parts.append(f"{turn.role}: {turn.content[:200]}")

    if summary_parts:
        shortterm = ShortTermMemory(anima.anima_dir, session_type="chat", thread_id=thread_id)
        shortterm.save(
            SessionState(
                accumulated_response="\n".join(summary_parts)[:4000],
                timestamp=now_local().isoformat(),
                trigger="idle_compaction",
                notes="Auto-saved during idle compaction",
            )
        )

    await conv.finalize_if_session_ended()
    logger.debug("_compact_mode_a: exit (compressed=%s)", compressed)
    return compressed


async def _compact_mode_b(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode B: same as Mode A."""
    return await _compact_mode_a(anima, thread_id)


async def _compact_mode_c(anima: DigitalAnima, thread_id: str) -> bool:
    """Mode C: conversation compress + shortterm save + codex thread discard."""
    from core.execution.codex_sdk import _clear_thread_id
    from core.memory.conversation import ConversationMemory
    from core.memory.shortterm import SessionState, ShortTermMemory
    from core.time_utils import now_local

    logger.debug("_compact_mode_c: entry (anima=%s, thread=%s)", anima.name, thread_id)
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
            timestamp=now_local().isoformat(),
            trigger="idle_compaction",
            notes="Auto-saved before Codex thread discard",
        )
    )

    _clear_thread_id(anima.anima_dir, "chat", thread_id)
    await conv.finalize_if_session_ended()
    logger.debug("_compact_mode_c: exit (success)")
    return True


# ── Public API ────────────────────────────────────────────────


async def run_idle_compaction(anima: DigitalAnima, thread_id: str) -> None:
    """Run mode-specific idle compaction for the given anima and thread.

    Acquires the thread lock with a 30-second timeout. If the lock cannot
    be acquired, compaction is skipped. Logs an "idle_compaction" activity
    event on success.
    """
    mode = anima.agent.execution_mode
    logger.info(
        "run_idle_compaction: start (anima=%s, mode=%s, thread=%s)",
        anima.name,
        mode,
        thread_id,
    )
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
                logger.info(
                    "Mode S compaction returned False for %s/%s; falling back to Mode A",
                    anima.name,
                    thread_id,
                )
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

    logger.info(
        "run_idle_compaction: success (anima=%s, mode=%s, thread=%s)",
        anima.name,
        mode,
        thread_id,
    )
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima.anima_dir)
        activity.log(
            "idle_compaction",
            summary=f"Idle compaction completed (mode={mode}, thread={thread_id})",
            meta={"mode": mode, "thread_id": thread_id},
        )
    except Exception:
        logger.warning("Failed to log idle_compaction activity", exc_info=True)
