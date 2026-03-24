from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Shared streaming helpers for A / S-Fallback / B executors.

Provides chunk accumulation utilities for LiteLLM streaming responses
and a context-manager helper for converting API errors into
``StreamDisconnectedError``.
"""

import json as _json
import logging
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
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


def _repair_json_arguments(raw: str) -> dict[str, Any] | None:
    """Attempt to extract valid JSON from a malformed arguments string.

    Some models (e.g. GLM-4.7 on vLLM with thinking enabled) produce
    duplicate JSON objects concatenated without a separator::

        {"command":"docker ps -a"{"command": "docker ps -a"}

    This helper scans for ``{`` characters and tries ``json.loads``
    starting from each one, returning the first valid parse result.
    """
    if not raw or not raw.strip():
        return None
    # Try each '{' as a potential start of a valid JSON object
    for i in range(len(raw)):
        if raw[i] != "{":
            continue
        candidate = raw[i:]
        # Try parsing progressively shorter suffixes to find valid JSON
        depth = 0
        for j, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = _json.loads(candidate[: j + 1])
                        if isinstance(parsed, dict):
                            logger.info(
                                "Repaired malformed tool-call JSON (offset=%d, len=%d)",
                                i,
                                j + 1,
                            )
                            return parsed
                    except (_json.JSONDecodeError, TypeError):
                        pass
                    break
    return None


def _extract_json_object_candidates(text: str) -> list[str]:
    """Extract balanced JSON-object substrings from *text*.

    This is used for models that prepend a natural-language preamble before
    emitting a JSON tool call as plain text.
    """
    candidates: list[str] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            cur = text[end]
            if in_string:
                if escape:
                    escape = False
                elif cur == "\\":
                    escape = True
                elif cur == '"':
                    in_string = False
                continue
            if cur == '"':
                in_string = True
            elif cur == "{":
                depth += 1
            elif cur == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : end + 1].strip())
                    break
    return candidates


def _parse_text_tool_call_object(
    obj: dict[str, Any],
    tool_names: set[str],
) -> tuple[str, str] | None:
    """Convert a parsed JSON object into ``(tool_name, arguments_json)``."""
    func_info = obj.get("function") or {}
    func_name = func_info.get("name") or obj.get("name")
    if not func_name or func_name not in tool_names:
        return None
    raw_args = func_info.get("arguments") or obj.get("arguments") or {}
    if isinstance(raw_args, str):
        try:
            raw_args = _json.loads(raw_args)
        except Exception:
            raw_args = _repair_json_arguments(raw_args) or {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    return func_name, _json.dumps(raw_args, ensure_ascii=False)


def try_parse_text_tool_call(
    text: str,
    tools: list[dict[str, Any]],
) -> tuple[str, str] | None:
    """Try to parse a tool call embedded in plain text.

    Supported forms:
    1. Structured OpenAI-style JSON
    2. Top-level ``{"name": "...", "arguments": {...}}``
    3. Natural-language preamble followed by one of the JSON objects above
    4. Python-style ``tool_name(key="value")`` lines
    """
    if not tools or not text:
        return None

    tool_names: set[str] = set()
    for tool in tools:
        if isinstance(tool, dict) and "function" in tool:
            name = tool["function"].get("name")
            if name:
                tool_names.add(name)
    if not tool_names:
        return None

    stripped = text.strip()
    candidates: list[str] = [stripped]
    for match in re.finditer(r"```(?:json|tool_call)?\s*\n?(.*?)```", stripped, re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if block:
            candidates.append(block)
    candidates.extend(_extract_json_object_candidates(stripped))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            obj = _json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            parsed = _parse_text_tool_call_object(obj, tool_names)
            if parsed:
                return parsed

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r"(\w+)\(([^)]*)\)", line)
        if not match:
            continue
        func_name = match.group(1)
        if func_name not in tool_names:
            continue
        args: dict[str, Any] = {}
        for arg_match in re.finditer(r'(\w+)=(?:"([^"]*?)"|\'([^\']*?)\')', match.group(2)):
            key = arg_match.group(1)
            val = arg_match.group(2) if arg_match.group(2) is not None else arg_match.group(3)
            args[key] = val
        return func_name, _json.dumps(args, ensure_ascii=False)

    return None


def parse_accumulated_tool_calls(
    tool_calls_acc: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse accumulated tool-call argument JSON strings.

    Returns a list of dicts with ``id``, ``name``, and ``arguments``
    (parsed from JSON).  Entries with unparseable arguments are
    included with the raw string under ``raw_arguments``.

    When ``json.loads`` fails, :func:`_repair_json_arguments` attempts
    to extract a valid JSON object from the malformed string (handles
    duplicate-object concatenation produced by some models).
    """
    result: list[dict[str, Any]] = []
    for _idx in sorted(tool_calls_acc):
        entry = tool_calls_acc[_idx]
        try:
            args = _json.loads(entry["arguments"])
        except (_json.JSONDecodeError, TypeError):
            args = _repair_json_arguments(entry["arguments"])
            if args is None:
                logger.warning(
                    "Unrepairable tool-call arguments for %s: %.200s",
                    entry.get("name", "?"),
                    entry["arguments"],
                )
        result.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "arguments": args,
                "raw_arguments": entry["arguments"] if args is None else None,
            }
        )
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
        async with stream_error_boundary(text_parts, executor_name="A"):
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
