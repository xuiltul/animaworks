from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Mode A2 executor: LiteLLM + tool_use loop.

Runs any tool_use-capable model (GPT-4o, Gemini Pro, etc.) in a loop where
the LLM autonomously calls tools until it produces a final text response
or hits the iteration limit.  Session chaining is handled inline when the
context threshold is crossed mid-conversation.
"""

import asyncio
import json as _json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from core.prompt.context import ContextTracker
from core.execution._session import build_continuation_prompt, handle_session_chaining
from core.execution.base import BaseExecutor, ExecutionResult
from core.memory import MemoryManager
from core.prompt.builder import build_system_prompt
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from core.tooling.handler import ToolHandler
from core.tooling.schemas import (
    build_tool_list,
    load_external_schemas,
    to_litellm_format,
)

logger = logging.getLogger("animaworks.execution.litellm_loop")

# Tools that perform writes — same-path writes must be serialised.
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "write_memory_file"})

# Thread-pool for running synchronous tool handlers concurrently.
_tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-quick")

# Dedicated thread-pool for long-running background tools (image gen, local LLM, etc.)
_bg_tool_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool-bg")


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


class LiteLLMExecutor(BaseExecutor):
    """Execute via LiteLLM with a tool_use loop (Mode A2).

    The LLM calls tools autonomously (memory, files, commands, delegation)
    until it produces a final text response or hits ``max_turns``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        tool_handler: ToolHandler,
        tool_registry: list[str],
        memory: MemoryManager,
        personal_tools: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model_config, anima_dir)
        self._tool_handler = tool_handler
        self._tool_registry = tool_registry
        self._memory = memory
        self._personal_tools = personal_tools or {}

    def _build_base_tools(self) -> list[dict[str, Any]]:
        """Build the base LiteLLM-format tool list (no external tools)."""
        canonical = build_tool_list(
            include_file_tools=True,
            include_search_tools=True,
            include_discovery_tools=True,
            include_notification_tools=self._tool_handler._human_notifier is not None,
            include_admin_tools=(self._anima_dir / "skills" / "newstaff.md").exists(),
            include_tool_management=True,
        )
        return to_litellm_format(canonical)

    def _build_llm_kwargs(self) -> dict[str, Any]:
        """Credential + model kwargs for ``litellm.acompletion``."""
        kwargs: dict[str, Any] = {
            "model": self._model_config.model,
            "max_tokens": self._model_config.max_tokens,
        }
        api_key = self._resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        if self._model_config.api_base_url:
            kwargs["api_base"] = self._model_config.api_base_url
        # Ollama thinking control: default to off for ollama/ models
        if self._model_config.thinking is not None:
            kwargs["think"] = self._model_config.thinking
        elif self._model_config.model.startswith("ollama/"):
            kwargs["think"] = False
        return kwargs

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
        # Check personal tools first
        if category in self._personal_tools:
            from core.tooling.schemas import load_personal_tool_schemas
            return load_personal_tool_schemas({category: self._personal_tools[category]})
        # Then core tools
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

        # Update internal state
        self._personal_tools = merged
        self._tool_handler._external.update_personal_tools(merged)

        # Load new schemas and inject into tools list
        new_schemas = load_personal_tool_schemas(merged)
        new_litellm = to_litellm_format(new_schemas)

        # Replace old dynamic tool schemas with new ones
        dynamic_names = {
            s["name"] for s in new_schemas
        }
        tools[:] = [
            t for t in tools
            if t.get("function", {}).get("name") not in dynamic_names
        ] + new_litellm

        names = ", ".join(sorted(merged.keys()))
        return f"Refreshed tools ({len(merged)} discovered): {names}"

    # Tools that use the dedicated background thread pool.
    _BG_POOL_TOOLS = frozenset({
        "generate_character_assets",
        "generate_fullbody", "generate_bustup", "generate_chibi",
        "generate_3d_model", "generate_rigged_model", "generate_animations",
        "local_llm", "run_command",
    })

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
        )
        return {"role": "tool", "tool_call_id": tc.id, "content": result}

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Run the LiteLLM tool-use loop.

        Returns ``ExecutionResult`` with the accumulated response text.
        """
        import litellm

        tools = self._build_base_tools()
        active_categories: set[str] = set()

        # Build initial messages with optional image content (OpenAI vision format)
        if images:
            content_parts: list[dict[str, Any]] = []
            for img in images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['media_type']};base64,{img['data']}",
                    },
                })
            content_parts.append({"type": "text", "text": prompt})
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ]
        else:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        all_response_text: list[str] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = self._model_config.max_turns
        chain_count = 0

        for iteration in range(max_iterations):
            logger.debug(
                "A2 tool loop iteration=%d messages=%d",
                iteration, len(messages),
            )
            try:
                response = await litellm.acompletion(
                    messages=messages,
                    tools=tools,
                    **llm_kwargs,
                )
            except Exception as e:
                logger.exception("LiteLLM API error")
                return ExecutionResult(text=f"[LLM API Error: {e}]")

            choice = response.choices[0]
            message = choice.message

            # ── Context tracking + session chaining ───────────
            if tracker and hasattr(response, "usage") and response.usage:
                usage_dict = {
                    "input_tokens": response.usage.prompt_tokens or 0,
                    "output_tokens": response.usage.completion_tokens or 0,
                }
                tracker.update_from_usage(usage_dict)

                current_text = message.content or ""
                new_sys, chain_count = await handle_session_chaining(
                    tracker=tracker,
                    shortterm=shortterm,
                    memory=self._memory,
                    current_text=current_text,
                    system_prompt_builder=partial(
                        build_system_prompt,
                        self._memory,
                        tool_registry=self._tool_registry,
                        personal_tools=self._personal_tools,
                        execution_mode="a2",
                        message=prompt,
                    ),
                    max_chains=self._model_config.max_chains,
                    chain_count=chain_count,
                    session_id="litellm-a2",
                    trigger="a2_tool_loop",
                    original_prompt=prompt,
                    accumulated_response="\n".join(all_response_text),
                    turn_count=iteration,
                )
                if new_sys is not None:
                    if current_text:
                        all_response_text.append(current_text)
                    messages = [
                        {"role": "system", "content": new_sys},
                        {"role": "user", "content": build_continuation_prompt()},
                    ]
                    continue

            # ── Check for tool calls ──────────────────────────
            tool_calls = message.tool_calls
            if not tool_calls:
                final_text = message.content or ""
                all_response_text.append(final_text)
                logger.debug("A2 final response at iteration=%d", iteration)
                return ExecutionResult(text="\n".join(all_response_text))

            # ── Process tool calls ────────────────────────────
            logger.info(
                "A2 tool calls at iteration=%d: %s",
                iteration,
                ", ".join(tc.function.name for tc in tool_calls),
            )
            messages.append(message.model_dump())

            # Phase 1: Parse arguments and handle discover_tools / JSON errors
            pending_calls: list[tuple] = []  # (tc, fn_args)
            for tc in tool_calls:
                fn_name = tc.function.name

                # JSON parse with structured error on failure
                try:
                    fn_args = _json.loads(tc.function.arguments)
                except _json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _json.dumps({
                            "status": "error",
                            "error_type": "InvalidArguments",
                            "message": f"Failed to parse tool arguments: {e}",
                            "context": {"raw_arguments": tc.function.arguments[:500]},
                            "suggestion": "Ensure arguments are valid JSON",
                        }, ensure_ascii=False),
                    })
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
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    continue

                # Handle refresh_tools inline — hot-reload personal/common tools
                if fn_name == "refresh_tools":
                    result = self._refresh_tools_inline(tools)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    continue

                pending_calls.append((tc, fn_args))

            # Phase 2: Execute remaining tool calls with parallelism
            if pending_calls:
                parallel, serial_batches = _partition_tool_calls(
                    [tc for tc, _ in pending_calls],
                )
                # Build args lookup by tool_call_id
                args_map = {tc.id: fn_args for tc, fn_args in pending_calls}

                # Parallel batch
                if parallel:
                    coros = [
                        self._execute_tool_call(tc, args_map[tc.id])
                        for tc in parallel
                    ]
                    results = await asyncio.gather(*coros, return_exceptions=True)
                    for i, r in enumerate(results):
                        if isinstance(r, Exception):
                            logger.warning("Parallel tool execution error: %s", r)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": parallel[i].id,
                                "content": _json.dumps({
                                    "status": "error",
                                    "error_type": "ExecutionError",
                                    "message": str(r),
                                }, ensure_ascii=False),
                            })
                        else:
                            messages.append(r)

                # Serial batches (same-path writes)
                for batch in serial_batches:
                    for tc in batch:
                        try:
                            r = await self._execute_tool_call(tc, args_map[tc.id])
                            messages.append(r)
                        except Exception as e:
                            logger.warning("Serial tool execution error: %s", e)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": _json.dumps({
                                    "status": "error",
                                    "error_type": "ExecutionError",
                                    "message": str(e),
                                }, ensure_ascii=False),
                            })

        logger.warning("A2 max iterations (%d) reached", max_iterations)
        return ExecutionResult(
            text="\n".join(all_response_text) or "(max iterations reached)",
        )
