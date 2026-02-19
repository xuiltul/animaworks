from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode A2 executor: LiteLLM + tool_use loop.

Runs any tool_use-capable model (GPT-4o, Gemini Pro, etc.) in a loop where
the LLM autonomously calls tools until it produces a final text response
or hits the iteration limit.  Session chaining is handled inline when the
context threshold is crossed mid-conversation.
"""

import asyncio
import json as _json
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from core.prompt.context import ContextTracker, _resolve_context_window
from core.execution._session import build_continuation_prompt, handle_session_chaining
from core.execution._streaming import (
    accumulate_tool_call_chunks,
    parse_accumulated_tool_calls,
    stream_error_boundary,
)
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError
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


# ── M1: Module-level dataclass replacing dynamic _FakeTC ─────


@dataclass
class _ToolCallShim:
    """Lightweight shim to satisfy ``_partition_tool_calls`` / ``_execute_tool_call``.

    Adapts parsed tool-call dicts (from ``parse_accumulated_tool_calls`` or
    iteration-level LiteLLM responses) into the ``.id`` / ``.function.name``
    / ``.function.arguments`` interface expected by partitioning and execution.
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

    @property
    def _is_ollama_model(self) -> bool:
        """Return True if the configured model is served via Ollama."""
        model = self._model_config.model
        return model.startswith("ollama/") or model.startswith("ollama_chat/")

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

    # ── C3b: execute() uses _build_initial_messages / _preflight_clamp ──

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

        messages = self._build_initial_messages(system_prompt, prompt, images)
        all_response_text: list[str] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = self._model_config.max_turns
        chain_count = 0

        for iteration in range(max_iterations):
            logger.debug(
                "A2 tool loop iteration=%d messages=%d",
                iteration, len(messages),
            )

            # ── Pre-flight: clamp max_tokens to fit context window ──
            iter_kwargs = self._preflight_clamp(
                llm_kwargs, messages, tools, litellm,
            )
            if iter_kwargs is None:
                return ExecutionResult(
                    text=f"[Error: prompt too large for "
                    f"{self._model_config.model}]",
                )

            try:
                response = await litellm.acompletion(
                    messages=messages,
                    tools=tools,
                    **iter_kwargs,
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

            # Convert and delegate to shared tool-call processor
            parsed_calls = _convert_litellm_tool_calls(tool_calls)
            async for _event in self._process_streaming_tool_calls(
                parsed_calls, messages, tools, active_categories,
            ):
                pass  # tool_end events not needed in non-streaming execute()

        logger.warning("A2 max iterations (%d) reached", max_iterations)
        return ExecutionResult(
            text="\n".join(all_response_text) or "(max iterations reached)",
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream execution events from the LiteLLM tool-use loop.

        Dispatches to token-level streaming (non-Ollama) or iteration-level
        streaming (Ollama) based on the model type.

        Yields:
            Event dicts: ``text_delta``, ``tool_start``, ``tool_end``, ``done``.
        """
        if self._is_ollama_model:
            async for event in self._stream_iteration_level(
                system_prompt, prompt, tracker, images,
            ):
                yield event
        else:
            async for event in self._stream_token_level(
                system_prompt, prompt, tracker, images,
            ):
                yield event

    # ── Token-level streaming (GPT-4o, Gemini, etc.) ─────────

    async def _stream_token_level(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Token-level streaming via ``litellm.acompletion(stream=True)``.

        Note: Session chaining is handled by AgentCore.run_cycle_streaming(),
        not within this method.
        """
        import litellm

        tools = self._build_base_tools()
        active_categories: set[str] = set()

        messages = self._build_initial_messages(system_prompt, prompt, images)
        all_response_text: list[str] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = self._model_config.max_turns

        async with stream_error_boundary(
            all_response_text, executor_name="A2-stream",
        ):
            for iteration in range(max_iterations):
                logger.debug(
                    "A2 stream iteration=%d messages=%d",
                    iteration, len(messages),
                )

                # ── Pre-flight: clamp max_tokens to fit context window ──
                iter_kwargs = self._preflight_clamp(
                    llm_kwargs, messages, tools, litellm,
                )
                if iter_kwargs is None:
                    # Prompt too large -- emit error and stop
                    error_msg = (
                        f"[Error: prompt too large for "
                        f"{self._model_config.model}]"
                    )
                    yield {"type": "text_delta", "text": error_msg}
                    yield {
                        "type": "done",
                        "full_text": error_msg,
                        "result_message": None,
                    }
                    return

                # Stream the LLM response
                response = await litellm.acompletion(
                    messages=messages,
                    tools=tools,
                    stream=True,
                    stream_options={"include_usage": True},
                    **iter_kwargs,
                )

                # Accumulate streamed chunks
                iter_text_parts: list[str] = []
                tool_calls_acc: dict[int, dict[str, Any]] = {}
                finish_reason: str | None = None
                usage_data: dict[str, int] | None = None

                async for chunk in response:
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        # Usage-only chunk (last chunk with stream_options)
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_data = {
                                "input_tokens": chunk.usage.prompt_tokens or 0,
                                "output_tokens": chunk.usage.completion_tokens or 0,
                            }
                        continue

                    # H4: null check for choice.delta
                    delta = choice.delta
                    if not delta:
                        continue

                    # Text content
                    if delta.content:
                        iter_text_parts.append(delta.content)
                        yield {"type": "text_delta", "text": delta.content}

                    # H1: Use accumulate_tool_call_chunks return value
                    if delta.tool_calls:
                        new_tool_names = accumulate_tool_call_chunks(
                            tool_calls_acc, delta.tool_calls,
                        )
                        for tool_name in new_tool_names:
                            for idx, entry in tool_calls_acc.items():
                                if entry["name"] == tool_name:
                                    yield {
                                        "type": "tool_start",
                                        "tool_name": tool_name,
                                        "tool_id": entry["id"],
                                    }
                                    break

                    # Check finish reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # Usage from choice-bearing chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = {
                            "input_tokens": chunk.usage.prompt_tokens or 0,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                        }

                # Update context tracker
                if tracker and usage_data:
                    tracker.update_from_usage(usage_data)

                iter_text = "".join(iter_text_parts)
                if iter_text:
                    all_response_text.append(iter_text)

                # ── No tool calls: final response ──
                if not tool_calls_acc:
                    full_text = "\n".join(all_response_text)
                    logger.debug(
                        "A2 stream final response at iteration=%d", iteration,
                    )
                    yield {
                        "type": "done",
                        "full_text": full_text,
                        "result_message": None,
                    }
                    return

                # ── Process tool calls ──
                parsed_calls = parse_accumulated_tool_calls(tool_calls_acc)
                logger.info(
                    "A2 stream tool calls at iteration=%d: %s",
                    iteration,
                    ", ".join(tc["name"] for tc in parsed_calls),
                )

                # Reconstruct assistant message for conversation history
                assistant_tool_calls = []
                for tc in parsed_calls:
                    assistant_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                _json.dumps(tc["arguments"], ensure_ascii=False)
                                if tc["arguments"] is not None
                                else tc.get("raw_arguments", "")
                            ),
                        },
                    })
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": iter_text or None,
                    "tool_calls": assistant_tool_calls,
                }
                messages.append(assistant_msg)

                # H2: Execute tool calls and yield tool_end per-tool
                async for event in self._process_streaming_tool_calls(
                    parsed_calls, messages, tools, active_categories,
                ):
                    yield event

        # If we exit the loop without returning, max iterations reached
        full_text = "\n".join(all_response_text) or "(max iterations reached)"
        logger.warning("A2 stream max iterations (%d) reached", max_iterations)
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": None,
        }

    # ── Iteration-level streaming (Ollama) ───────────────────

    async def _stream_iteration_level(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Iteration-level streaming for Ollama models.

        Each LLM call is blocking (no token streaming). After each
        iteration, the full text and tool events are yielded.

        Note: Session chaining is handled by AgentCore.run_cycle_streaming(),
        not within this method.
        """
        import litellm

        tools = self._build_base_tools()
        active_categories: set[str] = set()

        messages = self._build_initial_messages(system_prompt, prompt, images)
        all_response_text: list[str] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = self._model_config.max_turns

        async with stream_error_boundary(
            all_response_text, executor_name="A2-ollama-stream",
        ):
            for iteration in range(max_iterations):
                logger.debug(
                    "A2 ollama stream iteration=%d messages=%d",
                    iteration, len(messages),
                )

                # ── Pre-flight: clamp max_tokens to fit context window ──
                iter_kwargs = self._preflight_clamp(
                    llm_kwargs, messages, tools, litellm,
                )
                if iter_kwargs is None:
                    error_msg = (
                        f"[Error: prompt too large for "
                        f"{self._model_config.model}]"
                    )
                    yield {"type": "text_delta", "text": error_msg}
                    yield {
                        "type": "done",
                        "full_text": error_msg,
                        "result_message": None,
                    }
                    return

                response = await litellm.acompletion(
                    messages=messages,
                    tools=tools,
                    **iter_kwargs,
                )

                choice = response.choices[0]
                message = choice.message

                # ── Context tracking ──
                if tracker and hasattr(response, "usage") and response.usage:
                    usage_dict = {
                        "input_tokens": response.usage.prompt_tokens or 0,
                        "output_tokens": response.usage.completion_tokens or 0,
                    }
                    tracker.update_from_usage(usage_dict)

                # ── Yield iteration text ──
                iter_text = message.content or ""
                if iter_text:
                    all_response_text.append(iter_text)
                    yield {"type": "text_delta", "text": iter_text}

                # ── Check for tool calls ──
                tool_calls = message.tool_calls
                if not tool_calls:
                    full_text = "\n".join(all_response_text)
                    logger.debug(
                        "A2 ollama stream final response at iteration=%d",
                        iteration,
                    )
                    yield {
                        "type": "done",
                        "full_text": full_text,
                        "result_message": None,
                    }
                    return

                # ── C1: Patch Ollama tool_call IDs BEFORE model_dump ──
                for i, tc in enumerate(tool_calls):
                    if not tc.id:
                        tc.id = f"ollama_{iteration}_{i}"

                logger.info(
                    "A2 ollama stream tool calls at iteration=%d: %s",
                    iteration,
                    ", ".join(tc.function.name for tc in tool_calls),
                )
                messages.append(message.model_dump())

                # Yield tool_start events
                for tc in tool_calls:
                    yield {
                        "type": "tool_start",
                        "tool_name": tc.function.name,
                        "tool_id": tc.id,
                    }

                # C3a: Convert iteration-level tool_calls to parsed dict
                # format and delegate to _process_streaming_tool_calls
                parsed_calls = _convert_litellm_tool_calls(tool_calls)

                # H2: Execute and yield tool_end per-tool in real-time
                async for event in self._process_streaming_tool_calls(
                    parsed_calls, messages, tools, active_categories,
                ):
                    yield event

        # Max iterations reached
        full_text = "\n".join(all_response_text) or "(max iterations reached)"
        logger.warning(
            "A2 ollama stream max iterations (%d) reached", max_iterations,
        )
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": None,
        }

    # ── Shared helpers for streaming ─────────────────────────

    def _build_initial_messages(
        self,
        system_prompt: str,
        prompt: str,
        images: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Build the initial messages list for an LLM call."""
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
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _preflight_clamp(
        self,
        llm_kwargs: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        litellm: Any,
    ) -> dict[str, Any] | None:
        """Pre-flight context window check, clamping max_tokens if needed.

        Returns the (possibly adjusted) kwargs dict, or ``None`` if the
        prompt is too large to fit in the context window at all.
        """
        try:
            from core.config import load_config
            _cw_overrides = load_config().model_context_windows
        except Exception:
            _cw_overrides = None
        ctx_window = _resolve_context_window(
            self._model_config.model, _cw_overrides,
        )
        try:
            est_input = litellm.token_counter(
                model=self._model_config.model,
                messages=messages,
                tools=tools,
            )
        except Exception:
            msg_chars = sum(len(str(m.get("content", ""))) for m in messages)
            tool_chars = len(_json.dumps(tools)) if tools else 0
            est_input = (msg_chars + tool_chars) // 2
        available = ctx_window - est_input
        configured_max = llm_kwargs.get("max_tokens", 4096)

        if available < configured_max:
            if available - 128 < 256:
                logger.error(
                    "Prompt too large for context window: "
                    "~%d tokens input, %d window",
                    est_input, ctx_window,
                )
                return None
            clamped = available - 128
            logger.info(
                "Clamping max_tokens %d -> %d "
                "(est_input ~%d, window %d)",
                configured_max, clamped, est_input, ctx_window,
            )
            return {**llm_kwargs, "max_tokens": clamped}

        return llm_kwargs

    # H2: _process_streaming_tool_calls is now an async generator
    # that yields tool_end events after each individual tool completes.

    async def _process_streaming_tool_calls(
        self,
        parsed_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        active_categories: set[str],
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

            # Handle unparseable arguments
            if fn_args is None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps({
                        "status": "error",
                        "error_type": "InvalidArguments",
                        "message": "Failed to parse tool arguments",
                        "context": {"raw_arguments": (tc.get("raw_arguments") or "")[:500]},
                        "suggestion": "Ensure arguments are valid JSON",
                    }, ensure_ascii=False),
                })
                yield {"type": "tool_end", "tool_id": tc_id, "tool_name": fn_name}
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
                    "content": result,
                })
                yield {"type": "tool_end", "tool_id": tc_id, "tool_name": fn_name}
                continue

            # Handle refresh_tools inline
            if fn_name == "refresh_tools":
                result = self._refresh_tools_inline(tools)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })
                yield {"type": "tool_end", "tool_id": tc_id, "tool_name": fn_name}
                continue

            pending_calls.append((tc, fn_args))

        # Execute remaining tool calls with parallelism
        if not pending_calls:
            return

        # M1: Use _ToolCallShim dataclass instead of dynamic type()
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
                yield {
                    "type": "tool_end",
                    "tool_id": parallel[i].id,
                    "tool_name": parallel[i].function.name,
                }

        for batch in serial_batches:
            for shim in batch:
                try:
                    r = await self._execute_tool_call(shim, args_map[shim.id])
                    messages.append(r)
                except Exception as e:
                    logger.warning("Serial tool execution error: %s", e)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": shim.id,
                        "content": _json.dumps({
                            "status": "error",
                            "error_type": "ExecutionError",
                            "message": str(e),
                        }, ensure_ascii=False),
                    })
                yield {
                    "type": "tool_end",
                    "tool_id": shim.id,
                    "tool_name": shim.function.name,
                }


# ── Module-level helper ──────────────────────────────────────


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
