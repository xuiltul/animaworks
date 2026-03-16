# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Prompt building functions for conversation memory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from core.i18n import t
from core.memory.conversation_models import (
    _MAX_DISPLAY_TURNS,
    _MAX_HUMAN_CHARS_IN_HISTORY,
    _MAX_RENDERED_TOOL_RECORDS,
    _MAX_RESPONSE_CHARS_IN_HISTORY,
    ConversationState,
    _sanitize_tool_id,
)
from core.paths import load_prompt

if TYPE_CHECKING:
    from core.schemas import ModelConfig


def build_chat_prompt(
    state: ConversationState,
    content: str,
    from_person: str = "human",
    max_history_chars: int | None = None,
    model_config: ModelConfig | None = None,
) -> str:
    """Build the user prompt with conversation history injected.

    Args:
        state: The conversation state to render.
        content: The current user message.
        from_person: Sender identifier.
        max_history_chars: If set, truncate the rendered history text
            to at most this many characters (tail-preserving).
        model_config: Optional model config (unused, for API compatibility).
    """
    history_block = _format_history(state, max_chars=max_history_chars)

    if history_block:
        return load_prompt(
            "chat_message_with_history",
            conversation_history=history_block,
            from_person=from_person,
            content=content,
        )
    else:
        return load_prompt(
            "chat_message",
            from_person=from_person,
            content=content,
        )


def build_structured_messages(
    state: ConversationState,
    content: str,
    fmt: str = "openai",
    model_config: ModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Build structured message history for Mode A/Fallback.

    Preserves tool_use/tool_result structure to prevent the LLM
    from learning to describe tool calls in text instead of actually
    calling them.

    Args:
        state: The conversation state to render.
        content: The current user message content.
        fmt: Message format — ``"openai"`` for LiteLLM/OpenAI-style
            or ``"anthropic"`` for native Anthropic API format.
        model_config: Optional model config (unused, for API compatibility).

    Returns:
        List of message dicts ready to pass to the LLM API.
    """
    messages: list[dict[str, Any]] = []

    # Compressed summary as context
    if state.compressed_summary:
        messages.append(
            {
                "role": "user",
                "content": (
                    t("conversation.summary_label", count=state.compressed_turn_count)
                    + f"\n\n{state.compressed_summary}"
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": t("conversation.summary_ack"),
            }
        )

    # Track total tool records rendered
    rendered_tool_count = 0
    display_turns = state.turns[-_MAX_DISPLAY_TURNS:]

    # Track pending tool_result blocks for Anthropic format merging
    pending_tool_results: list[dict[str, Any]] = []

    for turn in display_turns:
        if turn.role == "human":
            display = turn.content
            if len(display) > _MAX_HUMAN_CHARS_IN_HISTORY:
                display = display[:_MAX_HUMAN_CHARS_IN_HISTORY] + "..."
            if fmt == "anthropic" and pending_tool_results:
                # Merge tool_result blocks with this user message
                merged_content = pending_tool_results + [{"type": "text", "text": display}]
                messages.append({"role": "user", "content": merged_content})
                pending_tool_results = []
            else:
                messages.append({"role": "user", "content": display})

        elif turn.role == "assistant":
            display = turn.content
            if len(display) > _MAX_RESPONSE_CHARS_IN_HISTORY:
                display = display[:_MAX_RESPONSE_CHARS_IN_HISTORY] + "..."

            has_tools = turn.tool_records and rendered_tool_count < _MAX_RENDERED_TOOL_RECORDS

            if has_tools and fmt == "openai":
                # OpenAI/LiteLLM format
                tool_calls: list[dict[str, Any]] = []
                for tr in turn.tool_records:
                    if rendered_tool_count >= _MAX_RENDERED_TOOL_RECORDS:
                        break
                    tool_calls.append(
                        {
                            "id": _sanitize_tool_id(tr.tool_id or f"hist_{rendered_tool_count}"),
                            "type": "function",
                            "function": {
                                "name": tr.tool_name,
                                "arguments": json.dumps(
                                    {"_summary": tr.input_summary},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                    rendered_tool_count += 1
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                )
                # Tool results — match by index
                for i, tc in enumerate(tool_calls):
                    result_text = turn.tool_records[i].result_summary if i < len(turn.tool_records) else ""
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_text or "(completed)",
                        }
                    )

            elif has_tools and fmt == "anthropic":
                # Anthropic format
                content_blocks: list[dict[str, Any]] = []
                if display:
                    content_blocks.append({"type": "text", "text": display})
                for tr in turn.tool_records:
                    if rendered_tool_count >= _MAX_RENDERED_TOOL_RECORDS:
                        break
                    tid = _sanitize_tool_id(tr.tool_id or f"hist_{rendered_tool_count}")
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tid,
                            "name": tr.tool_name,
                            "input": {"_summary": tr.input_summary},
                        }
                    )
                    pending_tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": tr.result_summary or "(completed)",
                        }
                    )
                    rendered_tool_count += 1
                messages.append({"role": "assistant", "content": content_blocks})

            else:
                # No tool records or budget exceeded — plain text
                messages.append({"role": "assistant", "content": display})

    # Current user message (merge with any pending tool_results)
    if pending_tool_results:
        merged_content = pending_tool_results + [{"type": "text", "text": content}]
        messages.append({"role": "user", "content": merged_content})
    elif fmt == "anthropic" and messages and messages[-1]["role"] == "user":
        # Merge to avoid consecutive user roles (Anthropic API requirement)
        existing = messages[-1]["content"]
        if isinstance(existing, list):
            existing.append({"type": "text", "text": content})
        else:
            messages[-1]["content"] = [
                {"type": "text", "text": existing},
                {"type": "text", "text": content},
            ]
    else:
        messages.append({"role": "user", "content": content})

    return messages


def _format_history(
    state: ConversationState,
    max_chars: int | None = None,
) -> str:
    """Format conversation history for prompt injection.

    Args:
        state: The conversation state to render.
        max_chars: If set, truncate the rendered text to at most this
            many characters, keeping the tail (most recent turns) and
            prepending an ellipsis marker.
    """
    parts: list[str] = []

    if state.compressed_summary:
        parts.append(
            t("conversation.history_summary_header", count=state.compressed_turn_count)
            + f"\n\n{state.compressed_summary}"
        )

    if state.turns:
        display_turns = state.turns[-_MAX_DISPLAY_TURNS:]
        turn_lines: list[str] = []
        for turn in display_turns:
            # Extract time portion for compact display
            ts = turn.timestamp[11:16] if len(turn.timestamp) >= 16 else turn.timestamp
            role_label = t("conversation.role_you") if turn.role == "assistant" else turn.role
            display = turn.content
            if turn.role == "assistant" and len(display) > _MAX_RESPONSE_CHARS_IN_HISTORY:
                display = display[:_MAX_RESPONSE_CHARS_IN_HISTORY] + "..."
            elif turn.role != "assistant" and len(display) > _MAX_HUMAN_CHARS_IN_HISTORY:
                display = display[:_MAX_HUMAN_CHARS_IN_HISTORY] + "..."
            if turn.role == "assistant" and turn.tool_records:
                tool_names = ", ".join(tr.tool_name for tr in turn.tool_records)
                display += "\n" + t("conversation.tools_executed", tool_names=tool_names)
            turn_lines.append(f"**[{ts}] {role_label}:**\n{display}")

        if parts:
            parts.append(t("conversation.recent_conversation_header") + "\n\n" + "\n\n".join(turn_lines))
        else:
            parts.append("\n\n".join(turn_lines))

    result = "\n\n---\n\n".join(parts) if parts else ""

    if max_chars and len(result) > max_chars:
        result = result[-max_chars:]
        result = t("conversation.ellipsis_omitted") + "\n" + result

    return result
