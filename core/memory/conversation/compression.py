# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Compression logic for conversation memory."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.i18n import t
from core.memory.conversation.models import (
    _MAX_DISPLAY_TURNS,
    _MAX_TURNS_BEFORE_COMPRESS,
    ConversationState,
    ConversationTurn,
)
from core.paths import load_prompt

logger = logging.getLogger("animaworks.conversation_memory")

_DETERMINISTIC_SUMMARY_MAX_CHARS = 12_000
_DETERMINISTIC_TURN_EXCERPT_CHARS = 500


@dataclass(frozen=True)
class CompressionResult:
    """Detailed outcome of a conversation compression attempt."""

    status: str
    performed: bool
    fallback_used: str = ""
    raw_turns_before: int = 0
    raw_turns_after: int = 0
    compressed_turns: int = 0
    summary_chars: int = 0
    error: str = ""


async def _call_llm(
    system: str,
    user_content: str,
    max_tokens: int = 1000,
    *,
    helper_role: str = "episode_summary",
    anima_dir: Path | None = None,
) -> str:
    """Call a registry-selected helper model or raise when all candidates fail."""
    from core.llm.helper_completion import one_shot_helper_completion

    result = await one_shot_helper_completion(
        user_content,
        role=helper_role,
        anima_dir=anima_dir,
        system_prompt=system,
        max_tokens=max_tokens,
    )
    if result is None:
        raise RuntimeError("All helper-model backends failed for conversation LLM call")
    return result


def _format_turns_for_compression(turns: list[ConversationTurn]) -> str:
    """Format turns into readable text for the compression prompt."""
    lines: list[str] = []
    for turn in turns:
        role = t("conversation.role_you") if turn.role == "assistant" else turn.role
        text = f"[{turn.timestamp}] {role}: {turn.content}"
        if turn.tool_records:
            tools = ", ".join(tr.tool_name for tr in turn.tool_records)
            text += "\n  " + t("conversation.tools_used", tools=tools)
        lines.append(text)
    return "\n\n".join(lines)


async def _call_compression_llm(
    old_summary: str,
    new_turns: str,
    *,
    anima_dir: Path | None = None,
) -> str:
    """Call the LLM to produce a compressed conversation summary."""
    system = load_prompt("memory/conversation_compression")

    user_content = ""
    if old_summary:
        user_content += f"{t('conversation.existing_summary_header')}\n\n{old_summary}\n\n---\n\n"
    user_content += f"{t('conversation.new_turns_header')}\n\n{new_turns}\n\n"
    user_content += t("conversation.integrate_instruction")

    return await _call_llm(
        system,
        user_content,
        max_tokens=2000,
        helper_role="conversation_compression",
        anima_dir=anima_dir,
    )


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ...[truncated]"


def _build_deterministic_summary(
    old_summary: str,
    turns: list[ConversationTurn],
) -> str:
    """Build a bounded extractive summary when all LLM compression paths fail."""
    lines: list[str] = []
    if old_summary.strip():
        lines.extend(
            [
                "## Previous summary",
                _truncate_text(old_summary.strip(), _DETERMINISTIC_SUMMARY_MAX_CHARS // 2),
                "",
            ]
        )

    lines.extend(
        [
            "## Deterministic compression fallback",
            f"Compressed {len(turns)} older turns because LLM compression was unavailable.",
            "",
        ]
    )

    for turn in turns:
        role = turn.role or "unknown"
        content = " ".join((turn.content or "").split())
        excerpt = _truncate_text(content, _DETERMINISTIC_TURN_EXCERPT_CHARS)
        lines.append(f"- [{turn.timestamp}] {role}: {excerpt}")
        if turn.tool_records:
            tools = ", ".join(tr.tool_name for tr in turn.tool_records[:5])
            lines.append(f"  Tools: {tools}")

    summary = "\n".join(lines).strip()
    return _truncate_text(summary, _DETERMINISTIC_SUMMARY_MAX_CHARS)


async def _generate_compression_summary(
    old_summary: str,
    turn_text: str,
    turns: list[ConversationTurn],
    model_config: Any,
    *,
    anima_dir: Path | None = None,
) -> tuple[str, str, str, str]:
    """Use the conversation-compression role, then deterministic compression only."""
    # The helper registry picks the model; it uses the Anima's own model only
    # when no conversation_compression helper is configured.
    del model_config
    try:
        summary = await _call_compression_llm(old_summary, turn_text, anima_dir=anima_dir)
        if summary and summary.strip():
            return summary, "llm_primary", "", ""
        error = "helper model returned empty summary"
    except Exception as exc:
        error = f"helper model failed: {type(exc).__name__}: {exc}"

    return (
        _build_deterministic_summary(old_summary, turns),
        "deterministic_fallback",
        "deterministic",
        error,
    )


def needs_compression(
    state: ConversationState,
    model_config: Any,
    load_context_window_overrides_fn: Callable[[], dict[str, int] | None],
) -> bool:
    """Check whether conversation history exceeds the compression threshold."""
    if len(state.turns) < 4:
        return False

    # Turn-count trigger: force compression regardless of token estimate
    if len(state.turns) > _MAX_TURNS_BEFORE_COMPRESS:
        return True

    from core.prompt.context import resolve_context_window

    window = resolve_context_window(model_config.model, load_context_window_overrides_fn())

    # Auto-scale threshold for small context models.
    configured = model_config.conversation_history_threshold
    if window < 64_000:
        auto_threshold = max(0.10, window / 64_000 * 0.30)
        effective_threshold = min(configured, auto_threshold)
    else:
        effective_threshold = configured

    threshold_tokens = int(window * effective_threshold)
    return state.total_token_estimate > threshold_tokens


async def compress_if_needed(
    state: ConversationState,
    model_config: Any,
    load_context_window_overrides_fn: Callable[[], dict[str, int] | None],
    save_fn: Callable[[], Any],
    anima_name: str = "",
    anima_dir: Path | None = None,
) -> bool:
    """Compress older conversation turns if the threshold is exceeded.

    Returns True if compression was performed.
    """
    result = await compress_if_needed_detailed(
        state,
        model_config,
        load_context_window_overrides_fn,
        save_fn,
        anima_name,
        anima_dir,
    )
    return result.performed


async def compress_if_needed_detailed(
    state: ConversationState,
    model_config: Any,
    load_context_window_overrides_fn: Callable[[], dict[str, int] | None],
    save_fn: Callable[[], Any],
    anima_name: str = "",
    anima_dir: Path | None = None,
) -> CompressionResult:
    """Compress older turns if needed and return a detailed outcome."""
    if not needs_compression(state, model_config, load_context_window_overrides_fn):
        raw_turns = len(state.turns)
        return CompressionResult(
            status="skipped_no_compression_needed",
            performed=False,
            raw_turns_before=raw_turns,
            raw_turns_after=raw_turns,
        )
    return await _compress(state, model_config, save_fn, anima_name, anima_dir)


async def _compress(
    state: ConversationState,
    model_config: Any,
    save_fn: Callable[[], Any],
    anima_name: str = "",
    anima_dir: Path | None = None,
) -> CompressionResult:
    """Perform LLM-based compression of older conversation turns."""
    raw_turns_before = len(state.turns)
    if len(state.turns) < 4:
        return CompressionResult(
            status="skipped_too_few_turns",
            performed=False,
            raw_turns_before=raw_turns_before,
            raw_turns_after=raw_turns_before,
        )

    # Keep a fixed number of recent turns (matches _MAX_DISPLAY_TURNS)
    keep_count = min(_MAX_DISPLAY_TURNS, len(state.turns) - 1)
    to_compress = state.turns[:-keep_count]
    to_keep = state.turns[-keep_count:]

    old_summary = state.compressed_summary
    turn_text = _format_turns_for_compression(to_compress)

    summary, status, fallback_used, error = await _generate_compression_summary(
        old_summary,
        turn_text,
        to_compress,
        model_config,
        anima_dir=anima_dir,
    )

    removed_count = len(to_compress)
    state.turns = to_keep
    state.compressed_summary = summary
    state.compressed_turn_count += removed_count

    # Shift finalization index to match the shortened turns array.
    if state.last_finalized_turn_index > 0:
        state.last_finalized_turn_index = max(
            0,
            state.last_finalized_turn_index - removed_count,
        )

    saved = save_fn()
    if inspect.isawaitable(saved):
        await saved

    logger.info(
        (
            "Conversation compressed for %s: %d turns -> summary (%d chars), "
            "keeping %d recent turns, status=%s fallback=%s"
        ),
        anima_name,
        len(to_compress),
        len(summary),
        len(to_keep),
        status,
        fallback_used or "none",
    )

    if error:
        logger.warning("Conversation compression fallback details for %s: %s", anima_name, error)

    return CompressionResult(
        status=status,
        performed=True,
        fallback_used=fallback_used,
        raw_turns_before=raw_turns_before,
        raw_turns_after=len(state.turns),
        compressed_turns=removed_count,
        summary_chars=len(summary),
        error=error,
    )
