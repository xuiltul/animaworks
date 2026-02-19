from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Shared streaming helpers for A2 / A1-Fallback / B executors.

Provides chunk accumulation utilities for LiteLLM streaming responses
and a context-manager helper for converting API errors into
``StreamDisconnectedError``.
"""

import json as _json
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from core.execution.base import StreamDisconnectedError

logger = logging.getLogger("animaworks.execution._streaming")


# ── LiteLLM tool-call chunk accumulation ─────────────────────


def accumulate_tool_call_chunks(
    tool_calls_acc: dict[int, dict[str, Any]],
    delta_tool_calls: list[Any],
) -> list[str]:
    """Accumulate LiteLLM streaming ``delta.tool_calls`` fragments.

    Each streaming chunk from ``litellm.acompletion(stream=True)``
    contains partial tool_call data.  The first chunk for a given
    index carries ``id`` and ``function.name``; subsequent chunks
    carry ``function.arguments`` fragments.

    Args:
        tool_calls_acc: Mutable accumulator dict keyed by tool_call
            index.  Mutated in place.
        delta_tool_calls: ``delta.tool_calls`` list from a single
            streaming chunk.

    Returns:
        List of newly discovered tool names (i.e. tool names seen
        for the first time in this call).  Callers use this to emit
        ``tool_start`` events.
    """
    new_tools: list[str] = []
    for tc in delta_tool_calls:
        idx = tc.index
        if idx not in tool_calls_acc:
            tool_calls_acc[idx] = {
                "id": tc.id,
                "name": getattr(tc.function, "name", None) or "",
                "arguments": "",
            }
            tool_name = tool_calls_acc[idx]["name"]
            if tool_name:
                new_tools.append(tool_name)
        arg_frag = getattr(tc.function, "arguments", "") or ""
        if arg_frag:
            tool_calls_acc[idx]["arguments"] += arg_frag
    return new_tools


def parse_accumulated_tool_calls(
    tool_calls_acc: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse accumulated tool-call argument JSON strings.

    Returns a list of dicts with ``id``, ``name``, and ``arguments``
    (parsed from JSON).  Entries with unparseable arguments are
    included with the raw string under ``raw_arguments``.
    """
    result: list[dict[str, Any]] = []
    for _idx in sorted(tool_calls_acc):
        entry = tool_calls_acc[_idx]
        try:
            args = _json.loads(entry["arguments"])
        except (_json.JSONDecodeError, TypeError):
            args = None
        result.append({
            "id": entry["id"],
            "name": entry["name"],
            "arguments": args,
            "raw_arguments": entry["arguments"] if args is None else None,
        })
    return result


# ── Error wrapping ───────────────────────────────────────────


@asynccontextmanager
async def stream_error_boundary(
    partial_text_parts: list[str],
    *,
    executor_name: str = "unknown",
) -> AsyncGenerator[None, None]:
    """Context manager that converts API errors to ``StreamDisconnectedError``.

    Usage::

        text_parts: list[str] = []
        async with stream_error_boundary(text_parts, executor_name="A2"):
            async for chunk in llm_stream:
                ...

    On any exception the accumulated ``partial_text_parts`` are joined
    and attached to the raised ``StreamDisconnectedError.partial_text``.
    """
    try:
        yield
    except StreamDisconnectedError:
        raise  # already wrapped
    except Exception as e:
        partial = "\n".join(partial_text_parts)
        logger.exception("%s streaming error", executor_name)
        raise StreamDisconnectedError(
            f"{executor_name} stream error: {e}",
            partial_text=partial,
        ) from e
