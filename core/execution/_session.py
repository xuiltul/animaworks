from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Shared session-chaining helper for inline executors (A2 / Fallback).

Both ``LiteLLMExecutor`` and ``AnthropicFallbackExecutor`` monitor context
usage mid-conversation and restart the session with short-term memory when
the configured threshold is crossed.  This module extracts that shared
logic so it lives in exactly one place.
"""

import logging
from collections.abc import Callable

from core.memory import MemoryManager
from core.time_utils import now_iso
from core.memory.shortterm import SessionState, ShortTermMemory
from core.paths import load_prompt
from core.prompt.builder import BuildResult, build_system_prompt, inject_shortterm
from core.prompt.context import ContextTracker

logger = logging.getLogger("animaworks.execution.session")


# ── Public helper ─────────────────────────────────────────

async def handle_session_chaining(
    tracker: ContextTracker,
    shortterm: ShortTermMemory | None,
    memory: MemoryManager,
    current_text: str,
    system_prompt_builder: Callable[[], BuildResult | str],
    max_chains: int,
    chain_count: int,
    *,
    session_id: str = "",
    trigger: str = "",
    original_prompt: str = "",
    accumulated_response: str = "",
    turn_count: int = 0,
) -> tuple[str | None, int]:
    """Handle context-threshold session chaining for inline executors.

    When the context tracker indicates the threshold has been crossed and
    chaining is still allowed, this function:

    1. Saves the current session state to short-term memory.
    2. Resets the tracker for a fresh session.
    3. Builds a new system prompt with the short-term memory injected.
    4. Clears the short-term memory file (state is now in the prompt).

    Args:
        tracker: Context usage tracker (must already reflect the latest
            API response).
        shortterm: Short-term memory handle.  If ``None``, chaining is
            skipped.
        memory: ``MemoryManager`` for rebuilding the system prompt.
        current_text: The text produced so far in the current iteration
            (used for logging / accumulated_response).
        system_prompt_builder: Zero-arg callable that returns the base
            system prompt (before short-term injection).  Typically a
            partial of ``build_system_prompt(memory, ...)``.
        max_chains: Maximum number of allowed chain restarts.
        chain_count: How many chains have occurred so far.
        session_id: Identifier for the session origin (e.g.
            ``"litellm-a2"`` or ``"anthropic-fallback"``).
        trigger: Trigger label stored in the ``SessionState``.
        original_prompt: The original user prompt (stored in state).
        accumulated_response: All response text accumulated before this
            call (the ``current_text`` is appended automatically).
        turn_count: Number of LLM turns executed so far.

    Returns:
        A 2-tuple ``(new_system_prompt, new_chain_count)``:
        - ``new_system_prompt`` is the rebuilt prompt if chaining was
          performed, or ``None`` if no chaining occurred.
        - ``new_chain_count`` is the updated chain counter.
    """
    if shortterm is None:
        return None, chain_count

    if not tracker.threshold_exceeded:
        return None, chain_count

    if chain_count >= max_chains:
        return None, chain_count

    chain_count += 1
    logger.info(
        "Session chaining %d/%d: context at %.1f%%",
        chain_count,
        max_chains,
        tracker.usage_ratio * 100,
    )

    # Combine accumulated text with the latest response fragment
    full_accumulated = accumulated_response
    if current_text:
        full_accumulated = (
            f"{accumulated_response}\n{current_text}"
            if accumulated_response
            else current_text
        )

    shortterm.save(
        SessionState(
            session_id=session_id,
            timestamp=now_iso(),
            trigger=trigger,
            original_prompt=original_prompt,
            accumulated_response=full_accumulated,
            context_usage_ratio=tracker.usage_ratio,
            turn_count=turn_count,
        )
    )

    tracker.reset()
    built = system_prompt_builder()
    base_prompt = built.system_prompt if isinstance(built, BuildResult) else built
    new_system_prompt = inject_shortterm(
        base_prompt,
        shortterm,
    )
    shortterm.clear()

    return new_system_prompt, chain_count


def build_continuation_prompt() -> str:
    """Load the session continuation prompt template.

    Convenience wrapper so callers need not import ``core.paths`` directly.
    """
    return load_prompt("session_continuation")


def build_stream_retry_prompt(checkpoint: "StreamCheckpoint") -> str:
    """Build a continuation prompt from a stream checkpoint.

    Summarises what was completed before the disconnect and instructs the
    LLM to continue from where it left off.

    Args:
        checkpoint: The checkpoint recorded during the interrupted stream.

    Returns:
        A prompt string for the retry session.
    """
    from core.memory.shortterm import StreamCheckpoint  # avoid circular at module level

    completed_lines: list[str] = []
    for i, tool in enumerate(checkpoint.completed_tools, 1):
        name = tool.get("tool_name", "unknown")
        summary = tool.get("summary", "")
        completed_lines.append(f"{i}. ✅ {name}: {summary}")

    completed_section = "\n".join(completed_lines) if completed_lines else "(なし)"

    # Truncate accumulated text to avoid oversized prompt
    acc_text = checkpoint.accumulated_text
    if len(acc_text) > 2000:
        acc_text = "...(前半省略)...\n" + acc_text[-2000:]

    return (
        "あなたは以下のタスクを実行中でしたが、通信エラーで中断されました。\n"
        "続きから実行してください。\n"
        "\n"
        "## 元の指示\n"
        f"{checkpoint.original_prompt}\n"
        "\n"
        "## 完了済みステップ\n"
        f"{completed_section}\n"
        "\n"
        "## これまでの出力\n"
        f"{acc_text}\n"
        "\n"
        "## 注意\n"
        "- 完了済みステップを繰り返さないでください\n"
        "- ファイルが既に存在する場合はスキップまたは更新してください\n"
        "- 中断前の作業の続きを実行してください\n"
    )
