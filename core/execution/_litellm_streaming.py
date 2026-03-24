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

import asyncio
import json as _json
import logging
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any, cast

from core.execution._litellm_tools import _convert_litellm_tool_calls
from core.execution._streaming import (
    accumulate_tool_call_chunks,
    parse_accumulated_tool_calls,
    stream_error_boundary,
    try_parse_text_tool_call,
)
from core.execution.base import (
    RepetitionDetector,
    StreamingThinkFilter,
    TokenUsage,
    ToolCallRecord,
    strip_thinking_tags,
    strip_untagged_thinking,
    supports_streaming_tool_use,
)
from core.execution.reminder import (
    SystemReminderQueue,
    msg_context_threshold,
    msg_final_iteration,
    msg_output_truncated,
)
from core.prompt.context import ContextTracker
from core.schemas import ImageData

logger = logging.getLogger("animaworks.execution.litellm_loop")


async def _empty_aiter() -> AsyncGenerator[Any, None]:
    """Yield nothing — used to skip streaming chunk loop for non-streaming responses."""
    return
    yield  # pragma: no cover – makes this an async generator


class StreamingMixin:
    """Mixin providing token-level and iteration-level streaming for LiteLLMExecutor."""

    # ── Token-level streaming (GPT-4o, Gemini, etc.) ─────────

    async def _stream_token_level(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
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

        tools = self._build_base_tools(trigger=trigger)
        _active_categories: set[str] = set()
        context_window = self._resolve_cw()

        messages = self._build_initial_messages(system_prompt, prompt, images, prior_messages=prior_messages)
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = max_turns_override or self._model_config.max_turns
        _usage_acc = TokenUsage()
        _repetition_detector = RepetitionDetector()
        _repetition_detected = False

        # Inject synthetic thinking_blocks into prior assistant messages
        # that have tool_calls but no thinking_blocks.  Without this,
        # LiteLLM drops the thinking param for the entire session because
        # the Anthropic API requires thinking_blocks on every assistant
        # turn with tool_use when extended thinking is enabled.
        # NOTE: This is an Anthropic-specific requirement.  Other providers
        # (Qwen, OpenAI, etc.) do not need this and may leak the dummy
        # content into model-visible context.
        _thinking_enabled = llm_kwargs.get("thinking") or llm_kwargs.get("reasoning_effort")
        if _thinking_enabled:
            _patched = 0
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls") and not msg.get("thinking_blocks"):
                    msg["thinking_blocks"] = [
                        {"type": "thinking", "thinking": "(resumed session)"},
                    ]
                    _patched += 1
            if _patched:
                logger.info(
                    "A stream: injected synthetic thinking_blocks into %d prior assistant message(s)",
                    _patched,
                )

        async with stream_error_boundary(
            all_response_text,
            executor_name="A-stream",
        ):
            for iteration in range(max_iterations):
                if self._check_interrupted():
                    logger.info("LiteLLM streaming interrupted at iteration=%d", iteration)
                    yield {"type": "text_delta", "text": "[Session interrupted by user]"}
                    yield {"type": "done", "full_text": "[Session interrupted by user]", "result_message": None}
                    return

                is_final_iteration = max_iterations > 1 and iteration == max_iterations - 1

                if is_final_iteration:
                    messages.append(
                        {
                            "role": "user",
                            "content": SystemReminderQueue.format_reminder(
                                msg_final_iteration(),
                            ),
                        }
                    )
                    logger.info(
                        "A stream final iteration=%d: tools removed",
                        iteration,
                    )

                logger.debug(
                    "A stream iteration=%d messages=%d",
                    iteration,
                    len(messages),
                )

                iter_tools = [] if is_final_iteration else tools

                # ── Pre-flight: clamp max_tokens to fit context window ──
                iter_kwargs = await self._preflight_clamp_with_compaction(
                    llm_kwargs,
                    messages,
                    iter_tools,
                    litellm,
                )
                if iter_kwargs is None:
                    error_msg = f"[Error: prompt too large for {self._model_config.model}]"
                    yield {"type": "text_delta", "text": error_msg}
                    yield {
                        "type": "done",
                        "full_text": error_msg,
                        "result_message": None,
                    }
                    return

                # Stream the LLM response
                # Bedrock requires toolConfig in every request that has toolUse/toolResult
                # in the conversation history — omitting tools causes ValidationException.
                _has_tool_history_s = any(
                    msg.get("role") == "tool" or (msg.get("role") == "assistant" and msg.get("tool_calls"))
                    for msg in messages
                )
                _bedrock_needs_tools_s = (
                    is_final_iteration and _has_tool_history_s and self._model_config.model.startswith("bedrock/")
                )
                _has_tools = (not is_final_iteration or _bedrock_needs_tools_s) and bool(tools)
                _use_stream = True
                if _has_tools and not supports_streaming_tool_use(self._model_config.model):
                    _use_stream = False
                    logger.info(
                        "A stream: model %s does not support streaming tool use, "
                        "falling back to non-streaming for this iteration",
                        self._model_config.model,
                    )

                call_kwargs: dict[str, Any] = {
                    "messages": messages,
                    "stream": _use_stream,
                    **iter_kwargs,
                }
                if _use_stream:
                    call_kwargs["stream_options"] = {"include_usage": True}
                if _has_tools:
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

                # ── Non-streaming fallback: convert response to streaming events ──
                if not _use_stream:
                    msg_obj = response.choices[0].message
                    _rc = getattr(msg_obj, "reasoning_content", None) or getattr(msg_obj, "reasoning", None) or ""
                    if _rc and not _reasoning_seen:
                        _reasoning_seen = True
                        _reasoning_parts.append(_rc)
                        yield {"type": "thinking_start"}
                        yield {"type": "thinking_delta", "text": _rc}
                        yield {"type": "thinking_end"}
                    if msg_obj.content:
                        thinking_text, response_text = strip_thinking_tags(msg_obj.content)
                        if thinking_text and not _reasoning_seen:
                            yield {"type": "thinking_start"}
                            yield {"type": "thinking_delta", "text": thinking_text}
                            yield {"type": "thinking_end"}
                            _reasoning_seen = True
                            _reasoning_parts.append(thinking_text)
                        if not _reasoning_seen and not thinking_text:
                            _ut, response_text = strip_untagged_thinking(
                                response_text if not thinking_text else response_text
                            )
                            if _ut:
                                _reasoning_seen = True
                                yield {"type": "thinking_start"}
                                yield {"type": "thinking_delta", "text": _ut}
                                yield {"type": "thinking_end"}
                        if response_text:
                            # Detect text-format tool calls.
                            # Llama 4 Maverick sometimes returns function calls as plain
                            # text (e.g. `read_memory_file(path="...")`) instead of using
                            # the structured tool_calls field.  Parse and treat as a real
                            # tool call so the execution loop can actually run the tool.
                            _text_tc: tuple[str, str] | None = None
                            if not msg_obj.tool_calls and iter_tools:
                                _text_tc = try_parse_text_tool_call(response_text, iter_tools)
                            if _text_tc:
                                _tc_name, _tc_args_json = _text_tc
                                _tc_id = f"text_call_{iteration}_{id(_tc_name) % 0xFFFF:04x}"
                                tool_calls_acc[0] = {
                                    "id": _tc_id,
                                    "name": _tc_name,
                                    "arguments": _tc_args_json,
                                }
                                yield {"type": "tool_start", "tool_name": _tc_name, "tool_id": _tc_id}
                                logger.info(
                                    "A stream: text-format tool call parsed: %s args=%s",
                                    _tc_name,
                                    _tc_args_json,
                                )
                            else:
                                iter_text_parts.append(response_text)
                                yield {"type": "text_delta", "text": response_text}
                    if msg_obj.tool_calls:
                        for idx, tc in enumerate(msg_obj.tool_calls):
                            tc_id = tc.id or f"call_{idx}"
                            tc_name = tc.function.name
                            tc_args = tc.function.arguments or "{}"
                            tool_calls_acc[idx] = {
                                "id": tc_id,
                                "name": tc_name,
                                "arguments": tc_args,
                            }
                            yield {"type": "tool_start", "tool_name": tc_name, "tool_id": tc_id}
                    finish_reason = response.choices[0].finish_reason
                    if hasattr(response, "usage") and response.usage:
                        _u = response.usage
                        _cr_ns = getattr(_u, "cache_read_input_tokens", 0) or 0
                        _cw_ns = getattr(_u, "cache_creation_input_tokens", 0) or 0
                        if not _cr_ns:
                            _ptd_ns = getattr(_u, "prompt_tokens_details", None)
                            if _ptd_ns:
                                _cr_ns = getattr(_ptd_ns, "cached_tokens", 0) or 0
                        usage_data = {
                            "input_tokens": _u.prompt_tokens or 0,
                            "output_tokens": _u.completion_tokens or 0,
                            "cache_read_tokens": _cr_ns,
                            "cache_write_tokens": _cw_ns,
                        }
                    _chunk_count = 0

                async for chunk in response if _use_stream else _empty_aiter():
                    _chunk_count += 1
                    # DEBUG: raw chunk inspection (first 5 chunks)
                    if _chunk_count <= 5:
                        logger.debug(
                            "A stream raw chunk #%d: type=%s choices=%s",
                            _chunk_count,
                            type(chunk).__name__,
                            repr(chunk.choices[:1]) if chunk.choices else "[]",
                        )
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        if hasattr(chunk, "usage") and chunk.usage:
                            _u_ch = chunk.usage
                            _cr_ch = getattr(_u_ch, "cache_read_input_tokens", 0) or 0
                            _cw_ch = getattr(_u_ch, "cache_creation_input_tokens", 0) or 0
                            if not _cr_ch:
                                _ptd_ch = getattr(_u_ch, "prompt_tokens_details", None)
                                if _ptd_ch:
                                    _cr_ch = getattr(_ptd_ch, "cached_tokens", 0) or 0
                            usage_data = {
                                "input_tokens": _u_ch.prompt_tokens or 0,
                                "output_tokens": _u_ch.completion_tokens or 0,
                                "cache_read_tokens": _cr_ch,
                                "cache_write_tokens": _cw_ch,
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
                            if _repetition_detector.feed(response_text):
                                logger.warning(
                                    "A stream: degenerate repetition detected at iteration=%d, "
                                    "tokens=%d — truncating response",
                                    iteration,
                                    len(_repetition_detector._tokens),
                                )
                                _truncation_msg = "\n\n[Response truncated: repetition detected]"
                                iter_text_parts.append(_truncation_msg)
                                yield {"type": "text_delta", "text": _truncation_msg}
                                _repetition_detected = True
                                break

                    # Reasoning content → thinking_delta events
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        _reasoning_parts.append(reasoning)
                        if not _reasoning_seen:
                            _reasoning_seen = True
                            logger.info(
                                "A stream: reasoning_content detected (model may be in thinking mode)",
                            )
                            yield {"type": "thinking_start"}
                        yield {"type": "thinking_delta", "text": reasoning}

                    if delta.tool_calls:
                        new_tool_names = accumulate_tool_call_chunks(
                            tool_calls_acc,
                            delta.tool_calls,
                        )
                        for tool_name in new_tool_names:
                            for _, entry in tool_calls_acc.items():
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
                        _u_ch2 = chunk.usage
                        _cr_ch2 = getattr(_u_ch2, "cache_read_input_tokens", 0) or 0
                        _cw_ch2 = getattr(_u_ch2, "cache_creation_input_tokens", 0) or 0
                        if not _cr_ch2:
                            _ptd_ch2 = getattr(_u_ch2, "prompt_tokens_details", None)
                            if _ptd_ch2:
                                _cr_ch2 = getattr(_ptd_ch2, "cached_tokens", 0) or 0
                        usage_data = {
                            "input_tokens": _u_ch2.prompt_tokens or 0,
                            "output_tokens": _u_ch2.completion_tokens or 0,
                            "cache_read_tokens": _cr_ch2,
                            "cache_write_tokens": _cw_ch2,
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
                                    _msg_obj,
                                    "thinking_blocks",
                                    None,
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
                        iteration,
                        _chunk_count,
                        finish_reason,
                        _reasoning_seen,
                        usage_data,
                        self._model_config.model,
                    )

                # Update context tracker + accumulate usage
                if usage_data:
                    _usage_acc.input_tokens += (
                        usage_data.get("input_tokens", 0) or usage_data.get("prompt_tokens", 0) or 0
                    )
                    _usage_acc.output_tokens += (
                        usage_data.get("output_tokens", 0) or usage_data.get("completion_tokens", 0) or 0
                    )
                    _usage_acc.cache_read_tokens += usage_data.get("cache_read_tokens", 0) or 0
                    _usage_acc.cache_write_tokens += usage_data.get("cache_write_tokens", 0) or 0
                if tracker and usage_data:
                    tracker.update_from_usage(usage_data)
                    yield {
                        "type": "context_update",
                        "context_usage_ratio": tracker.usage_ratio,
                        "input_tokens": tracker._input_tokens,
                        "context_window": tracker.context_window,
                        "threshold": tracker.threshold,
                    }

                    if tracker.threshold_exceeded:
                        try:
                            ratio = float(tracker.usage_ratio)
                        except (TypeError, ValueError):
                            ratio = 0.0
                        self.reminder_queue.push_sync(msg_context_threshold(ratio=ratio))

                if finish_reason == "length":
                    self.reminder_queue.push_sync(msg_output_truncated())

                flushed = _think_filter.flush()
                if flushed:
                    if _reasoning_seen:
                        yield {"type": "thinking_delta", "text": flushed}
                    else:
                        iter_text_parts.append(flushed)
                        yield {"type": "text_delta", "text": flushed}

                iter_text = "".join(iter_text_parts)
                # DEBUG: log thinking vs content for diagnosis
                if _reasoning_seen or _reasoning_parts:
                    logger.debug(
                        "A stream thinking debug: reasoning_chars=%d, content_chars=%d, "
                        "chunks=%d, content_preview=%.100r",
                        sum(len(p) for p in _reasoning_parts),
                        len(iter_text),
                        _chunk_count,
                        iter_text[:100],
                    )
                elif iter_text and len(iter_text) < 30:
                    logger.debug(
                        "A stream short response: chars=%d, chunks=%d, content=%.100r, think_filter_state=%s",
                        len(iter_text),
                        _chunk_count,
                        iter_text,
                        _think_filter._state if hasattr(_think_filter, "_state") else "N/A",
                    )
                if iter_text:
                    all_response_text.append(iter_text)

                # ── Detect text-format tool calls in streaming path ──
                # Some models (e.g. qwen3.5:9b via Ollama) emit tool calls as
                # JSON text in the content stream rather than using the
                # structured tool_calls field.  Detect and convert here so the
                # execution loop can run the tool normally.
                if not tool_calls_acc and not _repetition_detected and iter_tools:
                    _stream_text_tc = try_parse_text_tool_call(iter_text, iter_tools)
                    if _stream_text_tc:
                        _stc_name, _stc_args_json = _stream_text_tc
                        _stc_id = f"text_call_{iteration}_{id(_stc_name) % 0xFFFF:04x}"
                        tool_calls_acc[0] = {
                            "id": _stc_id,
                            "name": _stc_name,
                            "arguments": _stc_args_json,
                        }
                        # Clear the text parts so the JSON blob isn't shown as
                        # a chat message — the tool result will follow instead.
                        iter_text_parts.clear()
                        if iter_text in all_response_text:
                            all_response_text.remove(iter_text)
                        yield {"type": "tool_start", "tool_name": _stc_name, "tool_id": _stc_id}
                        logger.info(
                            "A stream: text-format tool call parsed (streaming): %s args=%s",
                            _stc_name,
                            _stc_args_json,
                        )

                # ── No tool calls (or repetition detected): final response ──
                if not tool_calls_acc or _repetition_detected:
                    full_text = "\n".join(all_response_text)
                    # Safety net: strip any residual <think> tags that the
                    # streaming filter missed (e.g. vLLM returning thinking
                    # in content without proper <think> opening tag).
                    _leaked_think, _clean = strip_thinking_tags(full_text)
                    if _leaked_think:
                        logger.info(
                            "A stream: post-stream strip_thinking_tags caught leaked thinking (%d chars)",
                            len(_leaked_think),
                        )
                        full_text = _clean
                        if not _reasoning_seen:
                            _reasoning_seen = True
                            yield {"type": "thinking_start"}
                        yield {"type": "thinking_delta", "text": _leaked_think}
                        yield {"type": "thinking_end"}

                    if not _reasoning_seen and not _leaked_think:
                        _untagged_think, _clean2 = strip_untagged_thinking(full_text)
                        if _untagged_think:
                            logger.info(
                                "A stream: strip_untagged_thinking detected thinking (%d chars)",
                                len(_untagged_think),
                            )
                            full_text = _clean2
                            _reasoning_seen = True
                            yield {"type": "thinking_start"}
                            yield {"type": "thinking_delta", "text": _untagged_think}
                            yield {"type": "thinking_end"}

                    if _reasoning_parts and not _leaked_think and not _reasoning_seen:
                        _rc_text = "".join(_reasoning_parts)
                        if _rc_text:
                            _reasoning_seen = True
                            yield {"type": "thinking_start"}
                            yield {"type": "thinking_delta", "text": _rc_text}
                            yield {"type": "thinking_end"}
                    logger.debug(
                        "A stream final response at iteration=%d",
                        iteration,
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

                # Reconstruct assistant message for conversation history.
                # When arguments failed to parse, use "{}" instead of the
                # raw malformed string — sending invalid JSON in the
                # conversation history causes some backends (e.g. vLLM) to
                # reject the entire request on the next iteration.
                assistant_tool_calls = []
                for tc in parsed_calls:
                    if tc["arguments"] is not None:
                        args_str = _json.dumps(tc["arguments"], ensure_ascii=False)
                    else:
                        args_str = "{}"
                    assistant_tool_calls.append(
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": args_str,
                            },
                        }
                    )
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
                    parsed_calls,
                    messages,
                    tools,
                    _active_categories,
                    context_window=context_window,
                ):
                    if "record" in event:
                        all_tool_records.append(event["record"])
                    yield event

                # ── Drain reminder queue after tool results ────
                reminder = self.reminder_queue.drain_sync()
                if reminder:
                    messages.append(
                        {
                            "role": "user",
                            "content": SystemReminderQueue.format_reminder(reminder),
                        }
                    )

        # If we exit the loop without returning, max iterations reached
        full_text = "\n".join(all_response_text) or "(max iterations reached)"
        # Safety net: strip any residual <think> tags (same as final-response path)
        _leaked, _clean = strip_thinking_tags(full_text)
        if _leaked:
            logger.info(
                "A stream max-iter: post-stream strip_thinking_tags caught leaked thinking (%d chars)",
                len(_leaked),
            )
            full_text = _clean
        elif not _reasoning_seen:
            _ut, _clean2 = strip_untagged_thinking(full_text)
            if _ut:
                logger.info(
                    "A stream max-iter: strip_untagged_thinking detected thinking (%d chars)",
                    len(_ut),
                )
                full_text = _clean2
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
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Iteration-level streaming for Ollama models.

        Each LLM call is blocking (no token streaming). After each
        iteration, the full text and tool events are yielded.

        Note: Session chaining is handled by AgentCore.run_cycle_streaming(),
        not within this method.
        """
        import litellm

        tools = self._build_base_tools(trigger=trigger)
        _active_categories: set[str] = set()
        context_window = self._resolve_cw()

        messages = self._build_initial_messages(system_prompt, prompt, images, prior_messages=prior_messages)
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        llm_kwargs = self._build_llm_kwargs()
        max_iterations = max_turns_override or self._model_config.max_turns
        _usage_acc_ol = TokenUsage()
        _repetition_detector = RepetitionDetector()

        async with stream_error_boundary(
            all_response_text,
            executor_name="A-ollama-stream",
        ):
            for iteration in range(max_iterations):
                if self._check_interrupted():
                    logger.info("LiteLLM streaming interrupted at iteration=%d", iteration)
                    yield {"type": "text_delta", "text": "[Session interrupted by user]"}
                    yield {
                        "type": "done",
                        "full_text": "[Session interrupted by user]",
                        "result_message": None,
                        "usage": _usage_acc_ol.to_dict(),
                    }
                    return

                is_final_iteration = max_iterations > 1 and iteration == max_iterations - 1

                if is_final_iteration:
                    messages.append(
                        {
                            "role": "user",
                            "content": SystemReminderQueue.format_reminder(
                                msg_final_iteration(),
                            ),
                        }
                    )
                    logger.info(
                        "A ollama stream final iteration=%d: tools removed",
                        iteration,
                    )

                logger.debug(
                    "A ollama stream iteration=%d messages=%d",
                    iteration,
                    len(messages),
                )

                iter_tools = [] if is_final_iteration else tools

                # ── Pre-flight: clamp max_tokens to fit context window ──
                iter_kwargs = await self._preflight_clamp_with_compaction(
                    llm_kwargs,
                    messages,
                    iter_tools,
                    litellm,
                )
                if iter_kwargs is None:
                    error_msg = f"[Error: prompt too large for {self._model_config.model}]"
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
                # Bedrock requires toolConfig in every request that has toolUse/toolResult
                # in the conversation history — omitting tools causes ValidationException.
                _has_tool_history_ol = any(
                    msg.get("role") == "tool" or (msg.get("role") == "assistant" and msg.get("tool_calls"))
                    for msg in messages
                )
                _bedrock_needs_tools_ol = (
                    is_final_iteration and _has_tool_history_ol and self._model_config.model.startswith("bedrock/")
                )
                if not is_final_iteration or _bedrock_needs_tools_ol:
                    call_kwargs["tools"] = tools

                # Apply a hard total-request timeout so that a frozen Ollama
                # instance (e.g. OOM/GPU stall with no tokens generated) does
                # not cause this coroutine to await indefinitely.
                # LiteLLM's ``timeout`` kwarg maps to httpx's *read* timeout
                # (per-read-operation), which does NOT fire when the model is
                # very slow but still sending the occasional token; we need an
                # explicit asyncio timeout on the whole acompletion call.
                # Use ``ollama_total_timeout`` (default 7200s = 2h) — generous
                # enough for large CPU-offloaded models while still catching
                # truly stuck requests (e.g. OOM / GPU stall).
                try:
                    from core.config import load_config as _lc_ot

                    _total_timeout_s = float(_lc_ot().server.ollama_total_timeout)
                except Exception:
                    _total_timeout_s = 7200.0
                try:
                    response = cast(
                        Any,
                        await asyncio.wait_for(
                            litellm.acompletion(**call_kwargs),
                            timeout=_total_timeout_s,
                        ),
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Ollama LLM call hard-timeout after %.0fs (trigger=%s model=%s)",
                        _total_timeout_s,
                        trigger,
                        self._model_config.model,
                    )
                    from core.exceptions import LLMAPIError

                    raise LLMAPIError(
                        f"Ollama request timed out after {_total_timeout_s:.0f}s "
                        f"(model={self._model_config.model})"
                    )

                choice = response.choices[0]
                message = choice.message

                # ── Context tracking ──
                if hasattr(response, "usage") and response.usage:
                    _inp_ol = response.usage.prompt_tokens or 0
                    _out_ol = response.usage.completion_tokens or 0
                    _usage_acc_ol.input_tokens += _inp_ol
                    _usage_acc_ol.output_tokens += _out_ol
                    _cr_ol = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                    _cw_ol = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                    if not _cr_ol:
                        _ptd_ol = getattr(response.usage, "prompt_tokens_details", None)
                        if _ptd_ol:
                            _cr_ol = getattr(_ptd_ol, "cached_tokens", 0) or 0
                    _usage_acc_ol.cache_read_tokens += _cr_ol
                    _usage_acc_ol.cache_write_tokens += _cw_ol
                    usage_dict = {"input_tokens": _inp_ol, "output_tokens": _out_ol}
                    if tracker:
                        tracker.update_from_usage(usage_dict)
                        yield {
                            "type": "context_update",
                            "context_usage_ratio": tracker.usage_ratio,
                            "input_tokens": tracker._input_tokens,
                            "context_window": tracker.context_window,
                            "threshold": tracker.threshold,
                        }

                    if tracker and tracker.threshold_exceeded:
                        try:
                            ratio = float(tracker.usage_ratio)
                        except (TypeError, ValueError):
                            ratio = 0.0
                        self.reminder_queue.push_sync(msg_context_threshold(ratio=ratio))

                if choice.finish_reason == "length":
                    self.reminder_queue.push_sync(msg_output_truncated())

                # ── Yield iteration text ──
                iter_text = message.content or ""
                thinking, iter_text = strip_thinking_tags(iter_text)
                if thinking:
                    yield {"type": "thinking_start"}
                    yield {"type": "thinking_delta", "text": thinking}
                    yield {"type": "thinking_end"}
                if iter_text and _repetition_detector.check_full_text(iter_text):
                    logger.warning(
                        "A ollama stream: degenerate repetition detected at iteration=%d — truncating",
                        iteration,
                    )
                    _truncation_msg = "\n\n[Response truncated: repetition detected]"
                    iter_text += _truncation_msg
                    all_response_text.append(iter_text)
                    yield {"type": "text_delta", "text": iter_text}
                    full_text = "\n".join(all_response_text)
                    yield {
                        "type": "done",
                        "full_text": full_text,
                        "result_message": None,
                        "tool_call_records": [asdict(r) for r in all_tool_records],
                        "usage": _usage_acc_ol.to_dict(),
                    }
                    return
                # ── Check for text-format tool calls BEFORE yielding as text ──
                # qwen3.5:9b and similar models sometimes output tool calls as
                # JSON text instead of using the structured tool_calls field.
                # Detect and convert so the execution loop runs the tool.
                tool_calls = message.tool_calls
                _ol_text_tc: tuple[str, str] | None = None
                if not tool_calls and iter_text and iter_tools:
                    _ol_text_tc = try_parse_text_tool_call(iter_text, iter_tools)

                if iter_text and not _ol_text_tc:
                    all_response_text.append(iter_text)
                    yield {"type": "text_delta", "text": iter_text}

                if _ol_text_tc:
                    _ol_tc_name, _ol_tc_args_json = _ol_text_tc
                    _ol_tc_id = f"text_call_{iteration}_{id(_ol_tc_name) % 0xFFFF:04x}"
                    logger.info(
                        "A ollama stream: text-format tool call parsed: %s args=%s",
                        _ol_tc_name,
                        _ol_tc_args_json,
                    )
                    yield {"type": "tool_start", "tool_name": _ol_tc_name, "tool_id": _ol_tc_id}
                    # Build a synthetic tool_calls list so the existing
                    # processing path below handles execution normally.
                    from types import SimpleNamespace
                    _syn_fn = SimpleNamespace(name=_ol_tc_name, arguments=_ol_tc_args_json)
                    _syn_tc = SimpleNamespace(id=_ol_tc_id, function=_syn_fn)
                    tool_calls = [_syn_tc]
                    # Reconstruct message.model_dump with synthetic tool_calls
                    _msg_dict = message.model_dump() if hasattr(message, "model_dump") else {}
                    _msg_dict["content"] = None
                    _msg_dict["tool_calls"] = [
                        {
                            "id": _ol_tc_id,
                            "type": "function",
                            "function": {"name": _ol_tc_name, "arguments": _ol_tc_args_json},
                        }
                    ]
                    messages.append(_msg_dict)

                # ── Check for tool calls ──
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
                # Skip for synthetic text-format tool calls (already appended above).
                if not _ol_text_tc:
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
                else:
                    logger.info(
                        "A ollama stream tool calls at iteration=%d (text-format): %s",
                        iteration,
                        ", ".join(tc.function.name or "unknown" for tc in tool_calls),
                    )

                parsed_calls = _convert_litellm_tool_calls(tool_calls)

                async for event in self._process_streaming_tool_calls(
                    parsed_calls,
                    messages,
                    tools,
                    _active_categories,
                    context_window=context_window,
                ):
                    if "record" in event:
                        all_tool_records.append(event["record"])
                    yield event

                # ── Drain reminder queue after tool results ────
                reminder = self.reminder_queue.drain_sync()
                if reminder:
                    messages.append(
                        {
                            "role": "user",
                            "content": SystemReminderQueue.format_reminder(reminder),
                        }
                    )

        # Max iterations reached
        full_text = "\n".join(all_response_text) or "(max iterations reached)"
        # Safety net: strip any residual <think> tags
        _leaked_ol, _clean_ol = strip_thinking_tags(full_text)
        if _leaked_ol:
            logger.info(
                "A ollama max-iter: strip_thinking_tags caught leaked thinking (%d chars)",
                len(_leaked_ol),
            )
            full_text = _clean_ol
        logger.warning(
            "A ollama stream max iterations (%d) reached",
            max_iterations,
        )
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": None,
            "tool_call_records": [asdict(r) for r in all_tool_records],
            "usage": _usage_acc_ol.to_dict(),
        }
