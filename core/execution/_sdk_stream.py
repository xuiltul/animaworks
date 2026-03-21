from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode S tool call/result logging, sanitization, and stream block processing.

Depends on ``core.execution.base`` for ``ToolCallRecord`` and truncation
helpers, and ``core.prompt.context`` for ``resolve_context_window``.

Also provides ``StreamingContext`` / ``StreamingState`` and the
``process_stream_messages`` async generator used by
``AgentSDKExecutor.execute_streaming``.
"""

import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.execution.base import (
    ToolCallRecord,
    _truncate_for_record,
    tool_input_save_budget,
    tool_result_save_budget,
)
from core.prompt.context import resolve_context_window

logger = logging.getLogger("animaworks.execution.agent_sdk")


# ── Tool logging helpers ─────────────────────────────────────


def _summarise_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Return a concise one-line summary of tool_input for activity log content."""
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return cmd[:300] if cmd else "(empty)"
    if tool_name in ("Read", "Write", "Edit"):
        return tool_input.get("file_path", "(no path)")
    if tool_name == "Grep":
        return tool_input.get("pattern", "(no pattern)")
    if tool_name == "Glob":
        return tool_input.get("pattern", "(no pattern)")
    return str(tool_input)[:300]


def _sanitise_tool_args(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Strip large payload fields from tool_input before logging."""
    if tool_name == "Write":
        sanitised = {k: v for k, v in tool_input.items() if k != "content"}
        if "content" in tool_input:
            sanitised["content_length"] = len(tool_input["content"])
        return sanitised
    if tool_name == "Edit":
        sanitised = {}
        for k, v in tool_input.items():
            if k in ("old_string", "new_string"):
                sanitised[k] = v[:200] if isinstance(v, str) else v
            else:
                sanitised[k] = v
        return sanitised
    return tool_input


def _log_tool_use(
    anima_dir: Path,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    tool_use_id: str | None = None,
    blocked: bool = False,
    block_reason: str = "",
) -> None:
    """Record a tool call to the activity log (best-effort, never raises)."""
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        meta: dict[str, Any] = {"args": _sanitise_tool_args(tool_name, tool_input)}
        if tool_use_id:
            meta["tool_use_id"] = tool_use_id
        if blocked:
            meta["blocked"] = True
            meta["reason"] = block_reason
        activity.log(
            "tool_use",
            tool=tool_name,
            content=_summarise_tool_input(tool_name, tool_input),
            meta=meta,
        )
    except Exception:
        # Never let logging failures disrupt tool execution.
        logger.debug("Failed to log tool_use for %s", tool_name, exc_info=True)


def _log_tool_result(
    anima_dir: Path,
    tool_name: str,
    tool_use_id: str,
    result_content: str,
    *,
    is_error: bool = False,
) -> None:
    """Record a tool result to the activity log (best-effort, never raises)."""
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        activity.log(
            "tool_result",
            tool=tool_name,
            content=result_content[:20_000] if len(result_content) > 20_000 else result_content,
            meta={"tool_use_id": tool_use_id, "is_error": is_error},
        )
    except Exception:
        logger.debug("Failed to log tool_result for %s", tool_name, exc_info=True)


# ── Stream block handlers ────────────────────────────────────


def _handle_tool_use_block(
    block: Any,
    pending_records: dict[str, ToolCallRecord],
    journal: Any | None,
    model: str,
    *,
    cw_overrides: dict[str, int] | None = None,
) -> ToolCallRecord:
    """Process a ToolUseBlock from AssistantMessage.

    Registers the block in ``pending_records`` and writes a WAL entry
    via the streaming journal (if provided).
    """
    context_window = resolve_context_window(model, cw_overrides)
    record = ToolCallRecord(
        tool_name=block.name,
        tool_id=block.id,
        input_summary=_truncate_for_record(
            str(getattr(block, "input", "")),
            tool_input_save_budget(context_window),
        ),
        result_summary="",
        is_error=False,
    )
    pending_records[block.id] = record
    if journal:
        journal.write_tool_start(block.name, record.input_summary, tool_id=block.id)
    logger.debug("ToolUseBlock registered: tool=%s id=%s", block.name, block.id)
    return record


def _tool_result_content_len(block: Any) -> int:
    """Return the character length of a ToolResultBlock's textual content.

    Used by session_stats tracking to estimate context consumption
    without duplicating the full content extraction in the outer loop.
    """
    content = block.content
    if isinstance(content, list):
        return sum(len(str(c.get("text", ""))) for c in content if isinstance(c, dict))
    return len(str(content)) if content else 0


def _handle_tool_result_block(
    block: Any,
    pending_records: dict[str, ToolCallRecord],
    journal: Any | None,
    model: str,
    *,
    anima_dir: Path | None = None,
    cw_overrides: dict[str, int] | None = None,
) -> None:
    """Process a ToolResultBlock from UserMessage.

    Updates the matching entry in ``pending_records`` and writes a WAL
    entry via the streaming journal (if provided).  When *anima_dir* is
    given, the full result is also recorded in the activity log.
    """
    content = block.content
    if isinstance(content, list):
        content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    content_str = str(content) if content else ""
    is_error = block.is_error if block.is_error is not None else False

    record = pending_records.get(block.tool_use_id)
    if record:
        context_window = resolve_context_window(model, cw_overrides)
        record.result_summary = _truncate_for_record(
            content_str,
            tool_result_save_budget(record.tool_name, context_window),
        )
        record.is_error = is_error
        logger.info(
            "ToolResult captured: tool=%s id=%s result_len=%d is_error=%s",
            record.tool_name,
            block.tool_use_id,
            len(content_str),
            is_error,
        )
    else:
        logger.warning("ToolResultBlock for unknown tool_use_id=%s", block.tool_use_id)

    if journal:
        tool_name = record.tool_name if record else "unknown"
        journal.write_tool_end(tool_name, content_str[:500], tool_id=block.tool_use_id)

    # Record full tool result in activity log
    if anima_dir is not None:
        tool_name = record.tool_name if record else "unknown"
        _log_tool_result(
            anima_dir,
            tool_name,
            block.tool_use_id,
            content_str,
            is_error=is_error,
        )


def _finalize_pending_records(
    pending_records: dict[str, ToolCallRecord],
) -> list[ToolCallRecord]:
    """Collect all tool records after the message loop ends.

    Marks any record that never received a result as an error.
    """
    records: list[ToolCallRecord] = []
    for tool_id, record in pending_records.items():
        if not record.result_summary:
            record.is_error = True
            logger.warning(
                "ToolCallRecord without result: tool=%s id=%s",
                record.tool_name,
                tool_id,
            )
        records.append(record)
    return records


# ── Streaming message processing ─────────────────────────────


@dataclass
class StreamingContext:
    """Read-only parameters for the streaming message loop."""

    prompt: str
    images: list[Any] | None
    session_stats: dict[str, Any]
    tracker: Any
    session_type: str
    model: str
    anima_dir: Path
    cw_overrides: dict[str, int] | None
    check_interrupted: Callable[[], bool]
    thread_id: str = "default"


@dataclass
class StreamingState:
    """Mutable state shared across the streaming message loop and caller."""

    response_text: list[str] = field(default_factory=list)
    pending_records: dict[str, ToolCallRecord] = field(default_factory=dict)
    active_tool_ids: set[str] = field(default_factory=set)
    result_message: Any = None
    message_count: int = 0
    usage_acc: Any = None


async def process_stream_messages(
    client: Any,
    ctx: StreamingContext,
    state: StreamingState,
) -> AsyncGenerator[dict[str, Any], None]:
    """Process streaming messages from an SDK client and yield UI events.

    Extracted from ``AgentSDKExecutor.execute_streaming`` inner
    ``_stream_messages`` to keep the executor class thin.
    """
    from core.execution._sdk_interrupt import _graceful_interrupt_stream
    from core.execution._sdk_session import _RESUMABLE_SESSION_TYPES, _build_sdk_query_input, _save_session_id
    from core.execution._tool_summary import make_tool_detail_chunk

    got_stream_event = False
    _in_thinking_block = False
    _captured_session_id: str | None = None

    await client.query(_build_sdk_query_input(ctx.prompt, ctx.images))
    async for message in client.receive_messages():
        if ctx.check_interrupted():
            logger.info("Agent SDK streaming interrupted — sending graceful interrupt")
            yield {"type": "text_delta", "text": "[Session interrupted by user]"}
            await _graceful_interrupt_stream(
                client,
                ctx.anima_dir,
                ctx.session_type,
                _captured_session_id,
                thread_id=ctx.thread_id,
            )
            return

        # Lazy imports are cached by Python — negligible cost after first call.
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
        from claude_agent_sdk.types import StreamEvent

        if isinstance(message, StreamEvent):
            _captured_session_id = message.session_id
            got_stream_event = True
            event = message.event
            event_type = event.get("type", "")

            if event_type == "message_start":
                usage = event.get("message", {}).get("usage", {})
                if usage:
                    ctx.tracker.update_from_message_start(usage)
                    state.usage_acc.cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
                    state.usage_acc.cache_write_tokens += usage.get("cache_creation_input_tokens", 0) or 0
                    yield {
                        "type": "context_update",
                        "context_usage_ratio": ctx.tracker.usage_ratio,
                        "input_tokens": ctx.tracker._input_tokens,
                        "context_window": ctx.tracker.context_window,
                        "threshold": ctx.tracker.threshold,
                    }

            elif event_type == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    state.active_tool_ids.add(tool_id)
                    yield {"type": "tool_start", "tool_name": tool_name, "tool_id": tool_id}
                elif block.get("type") == "thinking":
                    _in_thinking_block = True
                    yield {"type": "thinking_start"}

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield {"type": "text_delta", "text": text}
                elif delta.get("type") == "thinking_delta":
                    thinking_text = delta.get("thinking", "")
                    if thinking_text:
                        yield {"type": "thinking_delta", "text": thinking_text}

            elif event_type == "content_block_stop":
                if _in_thinking_block:
                    _in_thinking_block = False
                    yield {"type": "thinking_end"}

        elif isinstance(message, AssistantMessage):
            if not got_stream_event:
                continue
            state.message_count += 1
            for block in message.content:
                if isinstance(block, TextBlock):
                    state.response_text.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    _handle_tool_use_block(
                        block,
                        state.pending_records,
                        None,
                        ctx.model,
                        cw_overrides=ctx.cw_overrides,
                    )
                    detail_chunk = make_tool_detail_chunk(block.name, block.id, block.input or {})
                    if detail_chunk:
                        yield detail_chunk
                    if block.id in state.active_tool_ids:
                        state.active_tool_ids.discard(block.id)
                        yield {"type": "tool_end", "tool_id": block.id, "tool_name": block.name}

        elif isinstance(message, UserMessage):
            if not got_stream_event:
                continue
            if isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        ctx.session_stats["total_result_bytes"] += _tool_result_content_len(block)
                        _handle_tool_result_block(
                            block,
                            state.pending_records,
                            None,
                            ctx.model,
                            anima_dir=ctx.anima_dir,
                            cw_overrides=ctx.cw_overrides,
                        )

        elif isinstance(message, ResultMessage):
            state.result_message = message
            if message.session_id and ctx.session_type in _RESUMABLE_SESSION_TYPES:
                _save_session_id(ctx.anima_dir, message.session_id, ctx.session_type, thread_id=ctx.thread_id)
            if message.usage:
                u = message.usage
                state.usage_acc.input_tokens = u.get("input_tokens", 0) or 0
                state.usage_acc.output_tokens = u.get("output_tokens", 0) or 0
            break

        elif isinstance(message, SystemMessage):
            if message.subtype == "init" and message.data:
                mcp_servers = message.data.get("mcp_servers", [])
                for srv in mcp_servers:
                    name = srv.get("name", "unknown")
                    status = srv.get("status", "unknown")
                    if status != "connected":
                        logger.error("MCP server '%s' failed to connect: status=%s", name, status)
                    else:
                        logger.info("MCP server '%s' connected successfully", name)
