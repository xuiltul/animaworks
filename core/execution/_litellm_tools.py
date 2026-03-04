from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tool processing mixin for LiteLLMExecutor.

Handles tool discovery, activation, execution, and parallel/serial
partitioning of tool calls.
"""

import asyncio
import json as _json
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from core.execution._sanitize import TOOL_TRUST_LEVELS, wrap_tool_result
from core.execution._tool_summary import make_tool_detail_chunk
from core.execution.base import ToolCallRecord, _truncate_for_record, tool_input_save_budget, tool_result_save_budget
from core.exceptions import ToolExecutionError
from core.tooling.schemas import (
    build_tool_list,
    load_external_schemas,
    to_litellm_format,
)

logger = logging.getLogger("animaworks.execution.litellm_loop")

_WRITE_TOOLS = frozenset({"write_file", "edit_file", "write_memory_file"})

_tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-quick")

_bg_tool_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool-bg")


# ── Module-level dataclass replacing dynamic _FakeTC ─────


@dataclass
class _ToolCallShim:
    """Lightweight shim adapting parsed tool-call dicts into the
    ``.id`` / ``.function.name`` / ``.function.arguments`` interface
    expected by partitioning and execution.
    """

    @dataclass
    class _Function:
        name: str
        arguments: str

    id: str
    function: _Function


def _partition_tool_calls(
    tool_calls: list,
) -> tuple[list, list[list]]:
    """Split tool_calls into parallel-safe and serial batches.

    Returns:
        (parallel_batch, serial_batches)
        - parallel_batch: tool_calls safe to run concurrently
        - serial_batches: groups of same-path writes that must run sequentially
    """
    parallel: list = []
    serial_by_path: dict[str, list] = {}

    for tc in tool_calls:
        fn_name = tc.function.name
        if fn_name in _WRITE_TOOLS:
            try:
                args = _json.loads(tc.function.arguments)
            except _json.JSONDecodeError:
                parallel.append(tc)
                continue
            path = args.get("path", "")
            if path in serial_by_path:
                serial_by_path[path].append(tc)
            else:
                serial_by_path[path] = []
                parallel.append(tc)  # first write to each path is parallel-safe
        else:
            parallel.append(tc)

    serial_batches = [calls for calls in serial_by_path.values() if calls]
    return parallel, serial_batches


def _convert_litellm_tool_calls(
    tool_calls: list,
) -> list[dict[str, Any]]:
    """Convert LiteLLM iteration-level tool_call objects to parsed dict format.

    Transforms objects with ``.function.name``, ``.function.arguments``,
    ``.id`` attributes into the same dict format produced by
    ``parse_accumulated_tool_calls()`` so that both token-level and
    iteration-level paths can share ``_process_streaming_tool_calls()``.
    """
    result: list[dict[str, Any]] = []
    for tc in tool_calls:
        try:
            args = _json.loads(tc.function.arguments)
        except (_json.JSONDecodeError, TypeError):
            args = None
        result.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": args,
            "raw_arguments": tc.function.arguments if args is None else None,
        })
    return result


class ToolProcessingMixin:
    """Mixin providing tool discovery, activation, and execution for LiteLLMExecutor."""

    _BG_POOL_TOOLS = frozenset({
        "generate_character_assets",
        "generate_fullbody", "generate_bustup", "generate_chibi",
        "generate_3d_model", "generate_rigged_model", "generate_animations",
        "local_llm", "run_command",
    })

    def _build_base_tools(self) -> list[dict[str, Any]]:
        """Build the base LiteLLM-format tool list (no external tools)."""
        canonical = build_tool_list(
            include_file_tools=True,
            include_search_tools=True,
            include_discovery_tools=True,
            include_notification_tools=self._tool_handler._human_notifier is not None,
            include_admin_tools=(self._anima_dir / "skills" / "newstaff.md").exists(),
            include_supervisor_tools=self._has_subordinates(),
            include_tool_management=True,
            include_task_tools=True,
            include_plan_tasks=True,
            include_background_task_tools=getattr(self._tool_handler, "_background_manager", None) is not None,
            include_skill_tools=True,
            skill_metas=self._memory.list_skill_metas(),
            common_skill_metas=self._memory.list_common_skill_metas(),
            procedure_metas=self._memory.list_procedure_metas(),
        )
        return to_litellm_format(canonical)

    def _list_tool_categories(self) -> str:
        """Return a summary of available external tool categories."""
        if not self._tool_registry:
            return "No external tool categories available."
        from core.tools import TOOL_MODULES
        lines = ["Available tool categories:"]
        for cat in sorted(self._tool_registry):
            if cat in TOOL_MODULES:
                lines.append(f"- {cat}")
        if self._personal_tools:
            for cat in sorted(self._personal_tools):
                lines.append(f"- {cat} (personal)")
        return "\n".join(lines)

    def _activate_category(self, category: str) -> list[dict[str, Any]]:
        """Load and return canonical schemas for a tool category."""
        if category in self._personal_tools:
            from core.tooling.schemas import load_personal_tool_schemas
            return load_personal_tool_schemas({category: self._personal_tools[category]})
        return load_external_schemas([category])

    def _refresh_tools_inline(self, tools: list[dict[str, Any]]) -> str:
        """Re-discover personal/common tools and update the tools list in-place."""
        from core.tools import discover_common_tools, discover_personal_tools
        from core.tooling.schemas import load_personal_tool_schemas

        personal = discover_personal_tools(self._anima_dir)
        common = discover_common_tools()
        merged = {**common, **personal}

        if not merged:
            return "No personal or common tools found."

        self._personal_tools = merged
        self._tool_handler._external.update_personal_tools(merged)

        new_schemas = load_personal_tool_schemas(merged)
        new_litellm = to_litellm_format(new_schemas)

        dynamic_names = {
            s["name"] for s in new_schemas
        }
        tools[:] = [
            t for t in tools
            if t.get("function", {}).get("name") not in dynamic_names
        ] + new_litellm

        names = ", ".join(sorted(merged.keys()))
        return f"Refreshed tools ({len(merged)} discovered): {names}"

    _TRUST_ORDER: dict[str, int] = {"trusted": 2, "medium": 1, "untrusted": 0}

    async def _execute_tool_call(self, tc, fn_args: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call, offloading sync work to a thread.

        Long-running tools (image gen, local LLM, etc.) use a dedicated
        background thread pool to avoid starving quick tool calls.
        """
        loop = asyncio.get_running_loop()
        executor = (
            _bg_tool_executor
            if tc.function.name in self._BG_POOL_TOOLS
            else _tool_executor
        )
        result = await loop.run_in_executor(
            executor,
            self._tool_handler.handle,
            tc.function.name,
            fn_args,
            tc.id,
        )

        trust = TOOL_TRUST_LEVELS.get(tc.function.name, "untrusted")
        trust_rank = self._TRUST_ORDER.get(trust, 0)
        self._tool_handler._min_trust_seen = min(
            self._tool_handler._min_trust_seen, trust_rank,
        )

        return {"role": "tool", "tool_call_id": tc.id, "content": wrap_tool_result(tc.function.name, result)}

    async def _process_streaming_tool_calls(
        self,
        parsed_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        active_categories: set[str],
        context_window: int = 128_000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process parsed tool calls: discover_tools, refresh_tools, and execute.

        Appends tool result messages to ``messages`` in place.  Yields
        ``tool_end`` events after each individual tool completes so the
        caller can forward them to the client in real-time.
        """
        pending_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for tc in parsed_calls:
            fn_name = tc["name"]
            fn_args = tc["arguments"]
            tc_id = tc["id"]

            detail_chunk = make_tool_detail_chunk(fn_name, tc_id, fn_args or {})
            if detail_chunk:
                yield detail_chunk

            # Handle unparseable arguments
            if fn_args is None:
                error_content = _json.dumps({
                    "status": "error",
                    "error_type": "InvalidArguments",
                    "message": "Failed to parse tool arguments",
                    "context": {"raw_arguments": (tc.get("raw_arguments") or "")[:500]},
                    "suggestion": "Ensure arguments are valid JSON",
                }, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": wrap_tool_result(fn_name, error_content),
                })
                yield {
                    "type": "tool_end", "tool_id": tc_id, "tool_name": fn_name,
                    "record": ToolCallRecord(
                        tool_name=fn_name, tool_id=tc_id,
                        input_summary="(invalid arguments)",
                        result_summary=_truncate_for_record(error_content, tool_result_save_budget(fn_name, context_window)),
                    ),
                }
                continue

            # Handle discover_tools inline
            if fn_name == "discover_tools":
                category = fn_args.get("category")
                if category is None:
                    result = self._list_tool_categories()
                elif category not in active_categories:
                    new_schemas = self._activate_category(category)
                    if new_schemas:
                        tools.extend(to_litellm_format(new_schemas))
                        active_categories.add(category)
                        result = f"Activated {len(new_schemas)} tools for '{category}'"
                    else:
                        result = f"No tools found for category '{category}'"
                else:
                    result = f"Category '{category}' is already active"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": wrap_tool_result(fn_name, result),
                })
                yield {
                    "type": "tool_end", "tool_id": tc_id, "tool_name": fn_name,
                    "record": ToolCallRecord(
                        tool_name=fn_name, tool_id=tc_id,
                        input_summary=_truncate_for_record(str(fn_args), tool_input_save_budget(context_window)),
                        result_summary=_truncate_for_record(result, tool_result_save_budget(fn_name, context_window)),
                    ),
                }
                continue

            # Handle refresh_tools inline
            if fn_name == "refresh_tools":
                result = self._refresh_tools_inline(tools)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": wrap_tool_result(fn_name, result),
                })
                yield {
                    "type": "tool_end", "tool_id": tc_id, "tool_name": fn_name,
                    "record": ToolCallRecord(
                        tool_name=fn_name, tool_id=tc_id,
                        input_summary=_truncate_for_record(str(fn_args), tool_input_save_budget(context_window)),
                        result_summary=_truncate_for_record(result, tool_result_save_budget(fn_name, context_window)),
                    ),
                }
                continue

            pending_calls.append((tc, fn_args))

        # Execute remaining tool calls with parallelism
        if not pending_calls:
            return

        shims = [
            _ToolCallShim(
                id=tc["id"],
                function=_ToolCallShim._Function(
                    name=tc["name"],
                    arguments=(
                        _json.dumps(tc["arguments"], ensure_ascii=False)
                        if tc["arguments"] is not None
                        else ""
                    ),
                ),
            )
            for tc, _ in pending_calls
        ]
        args_map = {tc["id"]: fn_args for tc, fn_args in pending_calls}

        parallel, serial_batches = _partition_tool_calls(shims)

        if parallel:
            coros = [
                self._execute_tool_call(shim, args_map[shim.id])
                for shim in parallel
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for i, r in enumerate(results):
                shim = parallel[i]
                if isinstance(r, BaseException):
                    logger.warning("Parallel tool execution error: %s", r)
                    error_content = _json.dumps({
                        "status": "error",
                        "error_type": "ExecutionError",
                        "message": str(r),
                    }, ensure_ascii=False)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": shim.id,
                        "content": wrap_tool_result(shim.function.name, error_content),
                    })
                    result_summary = _truncate_for_record(error_content, tool_result_save_budget(shim.function.name, context_window))
                elif isinstance(r, dict):
                    messages.append(r)
                    result_summary = _truncate_for_record(r.get("content", ""), tool_result_save_budget(shim.function.name, context_window))
                else:
                    messages.append({"role": "tool", "tool_call_id": shim.id, "content": str(r)})
                    result_summary = _truncate_for_record(str(r), tool_result_save_budget(shim.function.name, context_window))
                yield {
                    "type": "tool_end",
                    "tool_id": shim.id,
                    "tool_name": shim.function.name,
                    "record": ToolCallRecord(
                        tool_name=shim.function.name,
                        tool_id=shim.id,
                        input_summary=_truncate_for_record(str(args_map[shim.id]), tool_input_save_budget(context_window)),
                        result_summary=result_summary,
                    ),
                }

        for batch in serial_batches:
            for shim in batch:
                try:
                    r = await self._execute_tool_call(shim, args_map[shim.id])
                    messages.append(r)
                    result_summary = _truncate_for_record(r.get("content", ""), tool_result_save_budget(shim.function.name, context_window))
                except ToolExecutionError as e:
                    logger.warning("Serial tool execution error: %s", e)
                    error_content = _json.dumps({
                        "status": "error",
                        "error_type": "ToolExecutionError",
                        "message": str(e),
                    }, ensure_ascii=False)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": shim.id,
                        "content": wrap_tool_result(shim.function.name, error_content),
                    })
                    result_summary = _truncate_for_record(error_content, tool_result_save_budget(shim.function.name, context_window))
                except Exception as e:
                    logger.warning("Serial tool execution error (unexpected): %s", e)
                    error_content = _json.dumps({
                        "status": "error",
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }, ensure_ascii=False)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": shim.id,
                        "content": wrap_tool_result(shim.function.name, error_content),
                    })
                    result_summary = _truncate_for_record(error_content, tool_result_save_budget(shim.function.name, context_window))
                yield {
                    "type": "tool_end",
                    "tool_id": shim.id,
                    "tool_name": shim.function.name,
                    "record": ToolCallRecord(
                        tool_name=shim.function.name,
                        tool_id=shim.id,
                        input_summary=_truncate_for_record(str(args_map[shim.id]), tool_input_save_budget(context_window)),
                        result_summary=result_summary,
                    ),
                }
