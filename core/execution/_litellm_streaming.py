from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Streaming mixin for LiteLLMExecutor.

Provides token-level streaming (GPT-4o, Gemini, Claude via LiteLLM, etc.)
and iteration-level streaming (Ollama) execution paths.
"""

import json as _json
import logging
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any, cast

from core.execution._streaming import (
    accumulate_tool_call_chunks,
    parse_accumulated_tool_calls,
    stream_error_boundary,
)
from core.execution.base import ExecutionResult, StreamingThinkFilter, TokenUsage, ToolCallRecord, strip_thinking_tags
from core.execution.reminder import MSG_CONTEXT_THRESHOLD, MSG_FINAL_ITERATION, MSG_OUTPUT_TRUNCATED, SystemReminderQueue
from core.prompt.context import ContextTracker, resolve_context_window
from core.execution._litellm_tools import _convert_litellm_tool_calls

logger = logging.getLogger("animaworks.execution.litellm_loop")


class StreamingMixin:
    """Mixin providing token-level and iteration-level streaming for LiteLLMExecutor."""

    # ── Token-level streaming (GPT-4o, Gemini, etc.) ─────────

    async def _stream_token_level(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Token-level streaming via ``litellm.acompletion(stream=True)``.

        Note: Session chaining is handled by AgentCore.run_cycle_streaming(),
        not within this method.
        """
        import litellm
        # When thinking_blocks are missing from assistant messages on tool-call
        # turns, Anthropic/Bedrock returns 400.  This flag tells LiteLLM to
        # silently drop the thinking param for that turn instead of crashing.
        litellm.modify_params = True

        tools = self._build_base_tools()
        active_categories: set[str] = set()
        context_window = resolve_context_window(self._model_config.model)

        messages = self._build_initial_messages(system_prompt, prompt, images, prior_messages=prior_messages)
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = max_turns_override or self._model_config.max_turns
        _usage_acc = TokenUsage()

        # Inject synthetic thinking_blocks into prior assistant messages
        # that have tool_calls but no thinking_blocks.  Without this,
        # LiteLLM drops the thinking param for the entire session because
        # the Anthropic API requires thinking_blocks on every assistant
        # turn with tool_use when extended thinking is enabled.
        _thinking_enabled = (
            llm_kwargs.get("thinking") or llm_kwargs.get("reasoning_effort")
        )
        if _thinking_enabled:
            _patched = 0
            for msg in messages:
                if (
                    msg.get("role") == "assistant"
                    and msg.get("tool_calls")
                    and not msg.get("thinking_blocks")
                ):
                    msg["thinking_blocks"] = [
                        {"type": "thinking", "thinking": "(resumed session)"},
                    ]
                    _patched += 1
            if _patched:
                logger.info(
                    "A stream: injected synthetic thinking_blocks into "
                    "%d prior assistant message(s)",
                    _patched,
                )

        async with stream_error_boundary(
            all_response_text, executor_name="A-stream",
        ):
            for iteration in range(max_iterations):
                if self._check_interrupted():
                    logger.info("LiteLLM streaming interrupted at iteration=%d", iteration)
                    yield {"type": "text_delta", "text": "[Session interrupted by user]"}
                    yield {"type": "done", "full_text": "[Session interrupted by user]", "result_message": None}
                    return

                is_final_iteration = (
                    max_iterations > 1 and iteration == max_iterations - 1
                )

                if is_final_iteration:
                    messages.append({
                        "role": "user",
                        "content": SystemReminderQueue.format_reminder(
                            MSG_FINAL_ITERATION,
                        ),
                    })
                    logger.info(
                        "A stream final iteration=%d: tools removed",
                        iteration,
                    )

                logger.debug(
                    "A stream iteration=%d messages=%d",
                    iteration, len(messages),
                )

                iter_tools = [] if is_final_iteration else tools

                # ── Pre-flight: clamp max_tokens to fit context window ──
                iter_kwargs = self._preflight_clamp(
                    llm_kwargs, messages, iter_tools, litellm,
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

                # Stream the LLM response
                call_kwargs: dict[str, Any] = {
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **iter_kwargs,
                }
                if not is_final_iteration:
                    call_kwargs["tools"] = tools

                response = cast(Any, await litellm.acompletion(**call_kwargs))

                # Accumulate streamed chunks
                iter_text_parts: list[str] = []
                tool_calls_acc: dict[int, dict[str, Any]] = {}
                finish_reason: str | None = None
                usage_data: dict[str, int] | None = None
                _chunk_count = 0
                _reasoning_seen = False
                _reasoning_parts: list[str] = []
                _think_filter = StreamingThinkFilter()

                async for chunk in response:
                    _chunk_count += 1
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_data = {
                                "input_tokens": chunk.usage.prompt_tokens or 0,
                                "output_tokens": chunk.usage.completion_tokens or 0,
                            }
                        continue

                    delta = choice.delta
                    if not delta:
                        continue

                    # Text content (with <think> tag filtering)
                    if delta.content:
                        thinking, response_text = _think_filter.feed(delta.content)
                        if thinking:
                            if not _reasoning_seen:
                                _reasoning_seen = True
                                yield {"type": "thinking_start"}
                            yield {"type": "thinking_delta", "text": thinking}
                        if response_text:
                            iter_text_parts.append(response_text)
                            yield {"type": "text_delta", "text": response_text}

                    # Reasoning content → thinking_delta events
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        _reasoning_parts.append(reasoning)
                        if not _reasoning_seen:
                            _reasoning_seen = True
                            logger.info(
                                "A stream: reasoning_content detected "
                                "(model may be in thinking mode)",
                            )
                            yield {"type": "thinking_start"}
                        yield {"type": "thinking_delta", "text": reasoning}

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

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = {
                            "input_tokens": chunk.usage.prompt_tokens or 0,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                        }

                if _reasoning_seen:
                    yield {"type": "thinking_end"}

                # Try to extract thinking_blocks from streamed response
                _iter_thinking_blocks: list[dict[str, Any]] | None = None
                if _reasoning_seen:
                    _resp_msg = getattr(response, "response_uptil_now", None)
                    if _resp_msg:
                        _choices = getattr(_resp_msg, "choices", None)
                        if _choices:
                            _msg_obj = getattr(_choices[0], "message", None)
                            if _msg_obj:
                                _iter_thinking_blocks = getattr(
                                    _msg_obj, "thinking_blocks", None,
                                )
                    # Fallback: build from reasoning_parts when response_uptil_now
                    # doesn't provide thinking_blocks (e.g. Bedrock streaming).
                    # Prevents LiteLLM from dropping thinking param on next turn.
                    if _iter_thinking_blocks is None and _reasoning_parts:
                        _iter_thinking_blocks = [
                            {"type": "thinking", "thinking": "".join(_reasoning_parts)},
                        ]
                        logger.debug(
                            "A stream: built thinking_blocks from reasoning_parts (%d chars)",
                            sum(len(p) for p in _reasoning_parts),
                        )

                # Post-stream diagnostics
                if not iter_text_parts and not tool_calls_acc:
                    logger.warning(
                        "A stream: empty response at iteration=%d "
                        "chunks=%d finish_reason=%s reasoning_seen=%s "
                        "usage=%s model=%s",
                        iteration, _chunk_count, finish_reason,
                        _reasoning_seen, usage_data,
                        self._model_config.model,
                    )

                # Update context tracker + accumulate usage
                if usage_data:
                    _usage_acc.input_tokens += usage_data.get("input_tokens", 0) or usage_data.get("prompt_tokens", 0) or 0
                    _usage_acc.output_tokens += usage_data.get("output_tokens", 0) or usage_data.get("completion_tokens", 0) or 0
                if tracker and usage_data:
                    tracker.update_from_usage(usage_data)

                    if tracker.threshold_exceeded:
                        try:
                            ratio = float(tracker.usage_ratio)
                        except (TypeError, ValueError):
                            ratio = 0.0
                        self.reminder_queue.push_sync(
                            MSG_CONTEXT_THRESHOLD.format(ratio=ratio)
                        )

                if finish_reason == "length":
                    self.reminder_queue.push_sync(MSG_OUTPUT_TRUNCATED)

                flushed = _think_filter.flush()
                if flushed:
                    iter_text_parts.append(flushed)
                    yield {"type": "text_delta", "text": flushed}

                iter_text = "".join(iter_text_parts)
                if iter_text:
                    all_response_text.append(iter_text)

                # ── No tool calls: final response ──
                if not tool_calls_acc:
                    full_text = "\n".join(all_response_text)
                    logger.debug(
                        "A stream final response at iteration=%d", iteration,
                    )
                    yield {
                        "type": "done",
                        "full_text": full_text,
                        "result_message": None,
                        "tool_call_records": [asdict(r) for r in all_tool_records],
                        "usage": _usage_acc.to_dict(),
                    }
                    return

                # ── Process tool calls ──
                parsed_calls = parse_accumulated_tool_calls(tool_calls_acc)
                logger.info(
                    "A stream tool calls at iteration=%d: %s",
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
                if _iter_thinking_blocks:
                    assistant_msg["thinking_blocks"] = _iter_thinking_blocks
                    logger.debug(
                        "A stream: preserved %d thinking_blocks for tool-call turn",
                        len(_iter_thinking_blocks),
                    )
                messages.append(assistant_msg)

                async for event in self._process_streaming_tool_calls(
                    parsed_calls, messages, tools, active_categories,
                    context_window=context_window,
                ):
                    if "record" in event:
                        all_tool_records.append(event["record"])
                    yield event

                # ── Drain reminder queue after tool results ────
                reminder = self.reminder_queue.drain_sync()
                if reminder:
                    messages.append({
                        "role": "user",
                        "content": SystemReminderQueue.format_reminder(reminder),
                    })

        # If we exit the loop without returning, max iterations reached
        full_text = "\n".join(all_response_text) or "(max iterations reached)"
        logger.warning("A stream max iterations (%d) reached", max_iterations)
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": None,
            "tool_call_records": [asdict(r) for r in all_tool_records],
            "usage": _usage_acc.to_dict(),
        }

    # ── Iteration-level streaming (Ollama) ───────────────────

    async def _stream_iteration_level(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
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
        context_window = resolve_context_window(self._model_config.model)

        messages = self._build_initial_messages(system_prompt, prompt, images, prior_messages=prior_messages)
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = max_turns_override or self._model_config.max_turns
        _usage_acc_ol = TokenUsage()

        async with stream_error_boundary(
            all_response_text, executor_name="A-ollama-stream",
        ):
            for iteration in range(max_iterations):
                if self._check_interrupted():
                    logger.info("LiteLLM streaming interrupted at iteration=%d", iteration)
                    yield {"type": "text_delta", "text": "[Session interrupted by user]"}
                    yield {"type": "done", "full_text": "[Session interrupted by user]", "result_message": None, "usage": _usage_acc_ol.to_dict()}
                    return

                is_final_iteration = (
                    max_iterations > 1 and iteration == max_iterations - 1
                )

                if is_final_iteration:
                    messages.append({
                        "role": "user",
                        "content": SystemReminderQueue.format_reminder(
                            MSG_FINAL_ITERATION,
                        ),
                    })
                    logger.info(
                        "A ollama stream final iteration=%d: tools removed",
                        iteration,
                    )

                logger.debug(
                    "A ollama stream iteration=%d messages=%d",
                    iteration, len(messages),
                )

                iter_tools = [] if is_final_iteration else tools

                # ── Pre-flight: clamp max_tokens to fit context window ──
                iter_kwargs = self._preflight_clamp(
                    llm_kwargs, messages, iter_tools, litellm,
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

                call_kwargs: dict[str, Any] = {
                    "messages": messages,
                    **iter_kwargs,
                }
                if not is_final_iteration:
                    call_kwargs["tools"] = tools

                response = cast(Any, await litellm.acompletion(**call_kwargs))

                choice = response.choices[0]
                message = choice.message

                # ── Context tracking ──
                if hasattr(response, "usage") and response.usage:
                    _inp_ol = response.usage.prompt_tokens or 0
                    _out_ol = response.usage.completion_tokens or 0
                    _usage_acc_ol.input_tokens += _inp_ol
                    _usage_acc_ol.output_tokens += _out_ol
                    usage_dict = {"input_tokens": _inp_ol, "output_tokens": _out_ol}
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

                if choice.finish_reason == "length":
                    self.reminder_queue.push_sync(MSG_OUTPUT_TRUNCATED)

                # ── Yield iteration text ──
                iter_text = message.content or ""
                thinking, iter_text = strip_thinking_tags(iter_text)
                if thinking:
                    yield {"type": "thinking_start"}
                    yield {"type": "thinking_delta", "text": thinking}
                    yield {"type": "thinking_end"}
                if iter_text:
                    all_response_text.append(iter_text)
                    yield {"type": "text_delta", "text": iter_text}

                # ── Check for tool calls ──
                tool_calls = message.tool_calls
                if not tool_calls:
                    full_text = "\n".join(all_response_text)
                    logger.debug(
                        "A ollama stream final response at iteration=%d",
                        iteration,
                    )
                    yield {
                        "type": "done",
                        "full_text": full_text,
                        "result_message": None,
                        "tool_call_records": [asdict(r) for r in all_tool_records],
                        "usage": _usage_acc_ol.to_dict(),
                    }
                    return

                # ── Patch Ollama tool_call IDs BEFORE model_dump ──
                for i, tc in enumerate(tool_calls):
                    if not tc.id:
                        tc.id = f"ollama_{iteration}_{i}"

                logger.info(
                    "A ollama stream tool calls at iteration=%d: %s",
                    iteration,
                    ", ".join(tc.function.name or "unknown" for tc in tool_calls),
                )
                messages.append(message.model_dump())

                # Yield tool_start events
                for tc in tool_calls:
                    yield {
                        "type": "tool_start",
                        "tool_name": tc.function.name,
                        "tool_id": tc.id,
                    }

                parsed_calls = _convert_litellm_tool_calls(tool_calls)

                async for event in self._process_streaming_tool_calls(
                    parsed_calls, messages, tools, active_categories,
                    context_window=context_window,
                ):
                    if "record" in event:
                        all_tool_records.append(event["record"])
                    yield event

                # ── Drain reminder queue after tool results ────
                reminder = self.reminder_queue.drain_sync()
                if reminder:
                    messages.append({
                        "role": "user",
                        "content": SystemReminderQueue.format_reminder(reminder),
                    })

        # Max iterations reached
        full_text = "\n".join(all_response_text) or "(max iterations reached)"
        logger.warning(
            "A ollama stream max iterations (%d) reached", max_iterations,
        )
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": None,
            "tool_call_records": [asdict(r) for r in all_tool_records],
            "usage": _usage_acc_ol.to_dict(),
        }
