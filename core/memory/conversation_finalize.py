# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Session finalization for conversation memory."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.i18n import t
from core.memory.conversation_compression import (
    _call_compression_llm,
    _call_llm,
    _format_turns_for_compression,
)
from core.memory.conversation_models import (
    SESSION_GAP_MINUTES,
    ConversationState,
    ConversationTurn,
    ParsedSessionSummary,
)
from core.paths import load_prompt
from core.time_utils import ensure_aware, now_local, today_local

logger = logging.getLogger("animaworks.conversation_memory")


def _gather_activity_context(anima_dir: Path, turns: list[ConversationTurn]) -> str:
    """Gather non-conversation activities from activity log for episode enrichment."""
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)

        if not turns:
            return ""
        first_ts = turns[0].timestamp
        last_ts = turns[-1].timestamp

        entries = activity.recent(
            days=1,
            limit=30,
            types=[
                "message_sent",
                "message_received",
                "channel_post",
                "channel_read",
                "tool_use",
                "human_notify",
                "cron_executed",
            ],
        )

        session_entries = [e for e in entries if first_ts <= e.ts <= last_ts]

        if not session_entries:
            return ""

        lines = [t("conversation.activity_context_header")]
        for e in session_entries:
            text = e.summary or e.content[:100]
            lines.append(f"- [{e.type}] {text}")

        return "\n".join(lines)
    except Exception:
        logger.debug("Failed to gather activity context", exc_info=True)
        return ""


async def _summarize_session_with_state(
    turns: list[ConversationTurn],
    activity_context: str = "",
) -> str:
    """Summarize a conversation session with state change extraction."""
    conversation_text = _format_turns_for_compression(turns)

    system = load_prompt("memory/session_summary")

    user_content = conversation_text
    if activity_context:
        user_content += f"\n\n{activity_context}"

    return await _call_llm(system, user_content)


def _parse_session_summary(raw: str) -> ParsedSessionSummary:
    """Parse Markdown-formatted LLM output into structured data."""
    episode_match = re.search(
        r"##\s*(?:エピソード要約|Episode Summary)\s*\n(.+?)(?=##\s*(?:ステート変更|State Changes)|\Z)",
        raw,
        re.DOTALL,
    )
    episode_body = episode_match.group(1).strip() if episode_match else raw.strip()

    lines = episode_body.splitlines()
    title = lines[0][:50] if lines else t("conversation.title_fallback")
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else episode_body

    state_match = re.search(
        r"##\s*(?:ステート変更|State Changes)\s*\n(.+)",
        raw,
        re.DOTALL,
    )

    resolved_items: list[str] = []
    new_tasks: list[str] = []
    current_status = ""

    if state_match:
        state_text = state_match.group(1)

        resolved_match = re.search(
            r"###\s*(?:解決済み|Resolved)\s*\n(.+?)(?=###|\Z)",
            state_text,
            re.DOTALL,
        )
        if resolved_match:
            for line in resolved_match.group(1).strip().splitlines():
                item = line.strip().lstrip("- ").strip()
                if item and item not in ("なし", "None", "none"):
                    resolved_items.append(item)

        tasks_match = re.search(
            r"###\s*(?:新規タスク|New Tasks)\s*\n(.+?)(?=###|\Z)",
            state_text,
            re.DOTALL,
        )
        if tasks_match:
            for line in tasks_match.group(1).strip().splitlines():
                item = line.strip().lstrip("- ").strip()
                if item and item not in ("なし", "None", "none"):
                    new_tasks.append(item)

        status_match = re.search(
            r"###\s*(?:現在の状態|Current State)\s*\n(.+?)(?=###|\Z)",
            state_text,
            re.DOTALL,
        )
        if status_match:
            current_status = status_match.group(1).strip()

    return ParsedSessionSummary(
        title=title,
        episode_body=body,
        resolved_items=resolved_items,
        new_tasks=new_tasks,
        current_status=current_status,
        has_state_changes=bool(resolved_items or new_tasks or current_status),
    )


async def finalize_session(
    anima_dir: Path,
    state: ConversationState,
    model_config: Any,
    save_fn: Callable[[], None],
    min_turns: int = 3,
    injected_procedures: list[Path] | None = None,
    session_id: str = "",
) -> bool:
    """Finalize the current conversation session (differential).

    Only summarizes turns since last_finalized_turn_index, preventing
    duplicate episode entries. Also extracts state changes and resolution
    information from the conversation.

    Returns:
        True if session was finalized and written to episodes/, False if skipped.
    """
    new_turns = state.turns[state.last_finalized_turn_index :]
    if len(new_turns) < min_turns:
        logger.debug(
            "Session finalization skipped: only %d new turns (min %d)",
            len(new_turns),
            min_turns,
        )
        return False

    activity_context = _gather_activity_context(anima_dir, new_turns)

    try:
        raw_summary = await _summarize_session_with_state(new_turns, activity_context)
    except Exception:
        logger.exception("Failed to summarize session; skipping episode write")
        return False

    parsed = _parse_session_summary(raw_summary)

    from core.memory.manager import MemoryManager

    memory_mgr = MemoryManager(anima_dir)
    timestamp = now_local()
    time_str = timestamp.strftime("%H:%M")
    episode_entry = f"## {time_str} — {parsed.title}\n\n{parsed.episode_body}\n"
    memory_mgr.append_episode(episode_entry)

    from core.memory.conversation_state_update import (
        _auto_track_procedure_outcomes,
        _record_resolutions,
        _update_state_from_summary,
    )

    if parsed.has_state_changes:
        _update_state_from_summary(anima_dir, memory_mgr, parsed)

    if parsed.resolved_items:
        _record_resolutions(anima_dir, memory_mgr, parsed.resolved_items)

    if injected_procedures:
        _auto_track_procedure_outcomes(
            anima_dir,
            memory_mgr,
            new_turns,
            injected_procedures=injected_procedures,
            session_id=session_id,
        )

    turn_text = _format_turns_for_compression(new_turns)
    old_summary = state.compressed_summary
    try:
        compressed = await _call_compression_llm(old_summary, turn_text)
        state.compressed_summary = compressed
    except Exception:
        logger.warning("Compression failed during finalization; keeping raw turns")

    state.last_finalized_turn_index = len(state.turns)
    state.compressed_turn_count += len(new_turns)
    save_fn()

    new_status = parsed.current_status.strip() if parsed.current_status else "status: idle"
    memory_mgr.archive_and_reset_state(new_status or "status: idle")

    logger.info(
        "Session finalized: %d new turns summarized and written to episodes/%s.md",
        len(new_turns),
        today_local().isoformat(),
    )

    return True


async def finalize_if_session_ended(
    lock: Any,
    load_fn: Callable[[], ConversationState],
    save_fn: Callable[[], None],
    needs_compression_fn: Callable[[], bool],
    compress_fn: Callable[[], Awaitable[None]],
    finalize_session_fn: Callable[..., Awaitable[bool]],
    load_pending_fn: Callable[[], tuple[list[Path], str]],
    anima_name: str = "",
) -> bool:
    """Finalize if session has ended (10-minute idle gap).

    Returns True if finalization was performed.
    """
    async with lock:
        state = load_fn()
        if not state.turns:
            return False

        if state.last_finalized_turn_index > len(state.turns):
            logger.info(
                "Clamping stale last_finalized_turn_index %d -> %d",
                state.last_finalized_turn_index,
                len(state.turns),
            )
            state.last_finalized_turn_index = len(state.turns)
            save_fn()

        last_turn_ts = datetime.fromisoformat(state.turns[-1].timestamp)
        idle_seconds = (now_local() - ensure_aware(last_turn_ts)).total_seconds()
        is_idle = idle_seconds >= SESSION_GAP_MINUTES * 60

        if is_idle and needs_compression_fn():
            logger.info(
                "Pre-compressing idle conversation for %s (idle %.0fs, %d turns)",
                anima_name,
                idle_seconds,
                len(state.turns),
            )
            await compress_fn()
            state = load_fn()

        new_turns = state.turns[state.last_finalized_turn_index :]
        if not new_turns:
            return False
        if not is_idle:
            return False

        procedures, session_id = load_pending_fn()
        return await finalize_session_fn(procedures or None, session_id)
