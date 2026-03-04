from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode A executor: LiteLLM + tool_use loop.

Runs any tool_use-capable model (GPT-4o, Gemini Pro, etc.) in a loop where
the LLM autonomously calls tools until it produces a final text response
or hits the iteration limit.  Session chaining is handled inline when the
context threshold is crossed mid-conversation.

Implementation is split across Mixin modules:
  - ``_litellm_tools``     — tool discovery, execution, partitioning
  - ``_litellm_context``   — LLM kwargs, message building, context clamping
  - ``_litellm_streaming`` — token-level & iteration-level streaming
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import Any

from core.exceptions import LLMAPIError, ToolExecutionError, ConfigError  # noqa: F401
from core.prompt.context import ContextTracker, resolve_context_window
from core.execution._session import build_continuation_prompt, handle_session_chaining
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, TokenUsage, ToolCallRecord, strip_thinking_tags
from core.execution.reminder import MSG_CONTEXT_THRESHOLD, MSG_FINAL_ITERATION, MSG_OUTPUT_TRUNCATED, SystemReminderQueue
from core.memory import MemoryManager
from core.prompt.builder import build_system_prompt
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from core.tooling.handler import ToolHandler

# ── Mixin imports ──────────────────────────────────────────
from core.execution._litellm_tools import ToolProcessingMixin
from core.execution._litellm_context import ContextMixin, _extract_tool_uses_from_messages
from core.execution._litellm_streaming import StreamingMixin

# ── Backward-compatible re-exports ────────────────────────
from core.execution._litellm_tools import (  # noqa: F401
    _WRITE_TOOLS,
    _ToolCallShim,
    _bg_tool_executor,
    _convert_litellm_tool_calls,
    _partition_tool_calls,
    _tool_executor,
)

logger = logging.getLogger("animaworks.execution.litellm_loop")


class LiteLLMExecutor(
    ToolProcessingMixin,
    ContextMixin,
    StreamingMixin,
    BaseExecutor,
):
    """Execute via LiteLLM with a tool_use loop (Mode A).

    The LLM calls tools autonomously (memory, files, commands, delegation)
    until it produces a final text response or hits ``max_turns``.

    Composed from three Mixins:
      - ``ToolProcessingMixin``  — tool discovery, execution, partitioning
      - ``ContextMixin``         — LLM kwargs, message building, context clamping
      - ``StreamingMixin``       — token-level & iteration-level streaming
    """

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        tool_handler: ToolHandler,
        tool_registry: list[str],
        memory: MemoryManager,
        personal_tools: dict[str, str] | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(model_config, anima_dir, interrupt_event=interrupt_event)
        self._tool_handler = tool_handler
        self._tool_registry = tool_registry
        self._memory = memory
        self._personal_tools = personal_tools or {}

    @property
    def _is_ollama_model(self) -> bool:
        """Return True if the configured model is served via Ollama."""
        model = self._model_config.model
        return model.startswith("ollama/") or model.startswith("ollama_chat/")

    # ── Non-streaming execution ──────────────────────────────

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
    ) -> ExecutionResult:
        """Run the LiteLLM tool-use loop.

        Returns ``ExecutionResult`` with the accumulated response text.
        """
        import litellm

        tools = self._build_base_tools()
        active_categories: set[str] = set()
        context_window = resolve_context_window(self._model_config.model)

        messages = self._build_initial_messages(
            system_prompt, prompt, images, prior_messages=prior_messages,
        )
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = max_turns_override or self._model_config.max_turns
        chain_count = 0
        usage_acc = TokenUsage()

        for iteration in range(max_iterations):
            if self._check_interrupted():
                logger.info("LiteLLM execute interrupted at iteration=%d", iteration)
                return ExecutionResult(text="[Session interrupted by user]")

            is_final_iteration = (
                max_iterations > 1 and iteration == max_iterations - 1
            )
            iter_tools = [] if is_final_iteration else tools

            if is_final_iteration:
                messages.append({
                    "role": "user",
                    "content": SystemReminderQueue.format_reminder(
                        MSG_FINAL_ITERATION,
                    ),
                })
                logger.info(
                    "A final iteration=%d: tools removed, requesting final answer",
                    iteration,
                )

            logger.debug(
                "A tool loop iteration=%d messages=%d",
                iteration, len(messages),
            )

            # ── Pre-flight: clamp max_tokens to fit context window ──
            iter_kwargs = self._preflight_clamp(
                llm_kwargs, messages, iter_tools, litellm,
            )
            if iter_kwargs is None:
                return ExecutionResult(
                    text=f"[Error: prompt too large for "
                    f"{self._model_config.model}]",
                    tool_call_records=all_tool_records,
                )

            call_kwargs: dict[str, Any] = {
                "messages": messages,
                **iter_kwargs,
            }
            if not is_final_iteration:
                call_kwargs["tools"] = tools

            try:
                response = await litellm.acompletion(**call_kwargs)
            except LLMAPIError:
                raise
            except Exception as e:
                logger.exception("LiteLLM API error")
                raise LLMAPIError(f"LiteLLM API error: {e}") from e

            choice = response.choices[0]
            message = choice.message

            # ── Context tracking + session chaining ───────────
            if hasattr(response, "usage") and response.usage:
                _inp = response.usage.prompt_tokens or 0
                _out = response.usage.completion_tokens or 0
                usage_acc.input_tokens += _inp
                usage_acc.output_tokens += _out
                usage_dict = {"input_tokens": _inp, "output_tokens": _out}
                if tracker:
                    tracker.update_from_usage(usage_dict)

                if tracker and tracker.threshold_exceeded:
                    try:
                        ratio = float(tracker.usage_ratio)
                    except (TypeError, ValueError):
                        ratio = 0.0
                    self.reminder_queue.push_sync(
                        MSG_CONTEXT_THRESHOLD.format(ratio=ratio)
                    )

                current_text = message.content or ""
                _, current_text = strip_thinking_tags(current_text)
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
                        execution_mode="a",
                        message=prompt,
                    ),
                    max_chains=self._model_config.max_chains,
                    chain_count=chain_count,
                    session_id="litellm-a",
                    trigger="a_tool_loop",
                    original_prompt=prompt,
                    accumulated_response="\n".join(all_response_text),
                    turn_count=iteration,
                    tool_uses=_extract_tool_uses_from_messages(messages),
                )
                if new_sys is not None:
                    if current_text:
                        all_response_text.append(current_text)
                    messages = [
                        {"role": "system", "content": new_sys},
                        {"role": "user", "content": build_continuation_prompt()},
                    ]
                    continue

            # ── P1-2: output truncation reminder ─────────────────
            if choice.finish_reason == "length":
                self.reminder_queue.push_sync(MSG_OUTPUT_TRUNCATED)

            # ── Check for tool calls ──────────────────────────
            tool_calls = message.tool_calls
            if not tool_calls:
                final_text = message.content or ""
                _, final_text = strip_thinking_tags(final_text)
                all_response_text.append(final_text)
                logger.debug("A final response at iteration=%d", iteration)
                final_reminder = self.reminder_queue.drain_formatted()
                if final_reminder:
                    all_response_text.append(final_reminder)
                return ExecutionResult(
                    text="\n".join(all_response_text),
                    tool_call_records=all_tool_records,
                    usage=usage_acc,
                )

            # ── Process tool calls ────────────────────────────
            logger.info(
                "A tool calls at iteration=%d: %s",
                iteration,
                ", ".join(tc.function.name or "unknown" for tc in tool_calls),
            )
            messages.append(message.model_dump())

            parsed_calls = _convert_litellm_tool_calls(tool_calls)
            async for _event in self._process_streaming_tool_calls(
                parsed_calls, messages, tools, active_categories,
                context_window=context_window,
            ):
                if "record" in _event:
                    all_tool_records.append(_event["record"])

            # ── Drain reminder queue after tool results ────────
            reminder = self.reminder_queue.drain_sync()
            if reminder:
                messages.append({
                    "role": "user",
                    "content": SystemReminderQueue.format_reminder(reminder),
                })

        logger.warning("A max iterations (%d) reached", max_iterations)
        return ExecutionResult(
            text="\n".join(all_response_text) or "(max iterations reached)",
            tool_call_records=all_tool_records,
            usage=usage_acc,
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream execution events from the LiteLLM tool-use loop.

        Dispatches to token-level streaming (non-Ollama) or iteration-level
        streaming (Ollama) based on the model type.

        Note: Session chaining is handled by AgentCore.run_cycle_streaming(),
        not within this method.

        Yields:
            Event dicts: ``text_delta``, ``tool_start``, ``tool_end``, ``done``.
        """
        if self._is_ollama_model:
            async for event in self._stream_iteration_level(
                system_prompt, prompt, tracker, images,
                prior_messages=prior_messages,
                max_turns_override=max_turns_override,
            ):
                yield event
        else:
            async for event in self._stream_token_level(
                system_prompt, prompt, tracker, images,
                prior_messages=prior_messages,
                max_turns_override=max_turns_override,
            ):
                yield event
