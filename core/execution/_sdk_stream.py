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
"""

import logging
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
            content=result_content,
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
) -> ToolCallRecord:
    """Process a ToolUseBlock from AssistantMessage.

    Registers the block in ``pending_records`` and writes a WAL entry
    via the streaming journal (if provided).
    """
    context_window = resolve_context_window(model)
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
        return sum(
            len(str(c.get("text", "")))
            for c in content
            if isinstance(c, dict)
        )
    return len(str(content)) if content else 0


def _handle_tool_result_block(
    block: Any,
    pending_records: dict[str, ToolCallRecord],
    journal: Any | None,
    model: str,
    *,
    anima_dir: Path | None = None,
) -> None:
    """Process a ToolResultBlock from UserMessage.

    Updates the matching entry in ``pending_records`` and writes a WAL
    entry via the streaming journal (if provided).  When *anima_dir* is
    given, the full result is also recorded in the activity log.
    """
    content = block.content
    if isinstance(content, list):
        content = " ".join(
            str(c.get("text", "")) for c in content if isinstance(c, dict)
        )
    content_str = str(content) if content else ""
    is_error = block.is_error if block.is_error is not None else False

    record = pending_records.get(block.tool_use_id)
    if record:
        context_window = resolve_context_window(model)
        record.result_summary = _truncate_for_record(
            content_str,
            tool_result_save_budget(record.tool_name, context_window),
        )
        record.is_error = is_error
        logger.info(
            "ToolResult captured: tool=%s id=%s result_len=%d is_error=%s",
            record.tool_name, block.tool_use_id, len(content_str), is_error,
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
            anima_dir, tool_name, block.tool_use_id,
            content_str, is_error=is_error,
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
                record.tool_name, tool_id,
            )
        records.append(record)
    return records
