# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Session finalization for conversation memory."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.i18n import t
from core.memory.conversation_compression import (
    _call_llm,
    _format_turns_for_compression,
    _generate_compression_summary,
)
from core.memory.conversation_models import (
    SESSION_GAP_MINUTES,
    ConversationState,
    ConversationTurn,
    ParsedSessionSummary,
)
from core.memory.fact_observability import warn_rate_limited
from core.paths import load_prompt
from core.time_utils import ensure_aware, now_local, today_local

logger = logging.getLogger("animaworks.conversation_memory")
_FACT_EXTRACTION_TASKS: set[asyncio.Task[tuple[int, int]]] = set()


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


async def _extract_session_facts_nonfatal(
    anima_dir: Path,
    text: str,
    *,
    source_episode: str,
    source_session_id: str,
    reference_time: str | None,
) -> tuple[int, int]:
    try:
        from core.memory.fact_extraction import extract_and_store_facts_with_outcome

        outcome = await extract_and_store_facts_with_outcome(
            anima_dir,
            text,
            source_episode=source_episode,
            source_session_id=source_session_id,
            reference_time=reference_time,
            origin="conversation",
        )
        logger.info(
            ("Session atomic fact extraction complete: facts_extracted=%d facts_failed=%d source_session_id=%s"),
            outcome.facts_extracted,
            outcome.facts_failed,
            source_session_id,
        )
        return outcome.facts_extracted, outcome.facts_failed
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.session",
            "Session atomic fact extraction failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        logger.info(
            "Session atomic fact extraction complete: facts_extracted=0 facts_failed=1 source_session_id=%s",
            source_session_id,
        )
        return 0, 1


def _finish_fact_task(task: asyncio.Task[tuple[int, int]]) -> None:
    _FACT_EXTRACTION_TASKS.discard(task)
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


def _schedule_session_fact_extraction(
    anima_dir: Path,
    turns: list[ConversationTurn],
    *,
    session_id: str,
) -> str:
    try:
        from core.memory.fact_extraction import _facts_extraction_enabled, format_turns_for_fact_extraction

        if not _facts_extraction_enabled():
            return "disabled"
        text = format_turns_for_fact_extraction(turns)
        if not text.strip():
            return "skipped_empty"
        reference_time = turns[-1].timestamp if turns else None
        task = asyncio.create_task(
            _extract_session_facts_nonfatal(
                anima_dir,
                text,
                source_episode=f"episodes/{today_local().isoformat()}.md",
                source_session_id=session_id,
                reference_time=reference_time,
            )
        )
        _FACT_EXTRACTION_TASKS.add(task)
        task.add_done_callback(_finish_fact_task)
        return "scheduled"
    except RuntimeError as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.session_schedule",
            "No running loop for session atomic fact extraction",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.session_schedule",
            "Failed to schedule session atomic fact extraction",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    return "failed"


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
    fact_extraction_status = _schedule_session_fact_extraction(anima_dir, new_turns, session_id=session_id)

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

    # If a previous finalization wrote an episode but compression failed, the
    # already-finalized raw turns remain before last_finalized_turn_index.
    # Include them in the next successful compression before clearing turns.
    turns_to_compress = state.turns if state.last_finalized_turn_index > 0 else new_turns
    turn_text = _format_turns_for_compression(turns_to_compress)
    old_summary = state.compressed_summary
    try:
        compressed, status, fallback_used, error = await _generate_compression_summary(
            old_summary,
            turn_text,
            turns_to_compress,
            model_config,
        )
        state.compressed_summary = compressed
        # Finalized turns are now in compressed_summary; clear to prevent
        # double-counting in total_token_estimate and avoid stale turns
        # triggering needs_compression() at the start of the next session.
        state.turns = []
        state.last_finalized_turn_index = 0
        state.compressed_turn_count += len(turns_to_compress)
        logger.info(
            "Finalized conversation compressed: %d turns -> summary (%d chars), status=%s fallback=%s",
            len(turns_to_compress),
            len(compressed),
            status,
            fallback_used or "none",
        )
        if error:
            logger.warning("Finalization compression fallback details: %s", error)
    except Exception:
        logger.warning("Compression failed during finalization; keeping raw turns")
        # The episode/state update has been written for these turns, so advance
        # the finalization cursor to avoid duplicate episode entries. Do not
        # increment compressed_turn_count: the raw turns are still present and
        # will be folded into compressed_summary on the next successful run.
        state.last_finalized_turn_index = len(state.turns)

    save_fn()

    new_status = parsed.current_status.strip() if parsed.current_status else "status: idle"
    memory_mgr.archive_and_reset_state(new_status or "status: idle")

    logger.info(
        (
            "Session finalized: %d new turns summarized and written to episodes/%s.md "
            "fact_extraction=%s facts_extracted=0 facts_failed=%d"
        ),
        len(new_turns),
        today_local().isoformat(),
        fact_extraction_status,
        int(fact_extraction_status == "failed"),
    )

    return True


async def finalize_if_session_ended(
    lock: Any,
    load_fn: Callable[[], ConversationState],
    save_fn: Callable[[], None],
    needs_compression_fn: Callable[[], bool],
    compress_fn: Callable[[], Awaitable[Any]],
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
