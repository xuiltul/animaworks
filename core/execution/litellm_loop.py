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
or the runaway guard halts pathological repetition. Session chaining is handled inline when the
context threshold is crossed mid-conversation.

Implementation is split across Mixin modules:
  - ``_litellm_tools``     — tool discovery, execution, partitioning
  - ``_litellm_context``   — LLM kwargs, message building, context clamping
  - ``_litellm_streaming`` — token-level & iteration-level streaming
"""

import asyncio
import json as _json
import logging
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.exceptions import ConfigError, LLMAPIError, ToolExecutionError  # noqa: F401
from core.execution._litellm_context import ContextMixin, _extract_tool_uses_from_messages
from core.execution._litellm_streaming import StreamingMixin

# ── Mixin imports ──────────────────────────────────────────
# ── Backward-compatible re-exports ────────────────────────
from core.execution._litellm_tools import (  # noqa: F401
    _WRITE_TOOLS,
    ToolProcessingMixin,
    _bg_tool_executor,
    _convert_litellm_tool_calls,
    _partition_tool_calls,
    _tool_executor,
    _ToolCallShim,
)
from core.execution._session import build_continuation_prompt, handle_session_chaining
from core.execution._streaming import try_parse_text_tool_call
from core.execution.backoff import decorrelated_jitter
from core.execution.base import (
    BaseExecutor,
    ExecutionResult,
    TokenUsage,
    ToolCallRecord,
    join_answer_parts,
    strip_thinking_tags,
)
from core.execution.error_classifier import (
    FailoverReason,
    classify_llm_error,
    guard_key,
    litellm_realm_of,
    provider_family_of,
)
from core.execution.loop_guards import (
    FINAL_RESPONSE_ERROR_TEXT,
    EmptyResponseTracker,
    LlmCallInterrupted,
    RunawayGuard,
    call_llm_with_retry,
    record_finalization_failure,
    record_runaway_event,
    strip_tool_protocol_messages,
    tool_call_signature,
)
from core.execution.rate_guard import get_rate_guard
from core.execution.reminder import (
    SystemReminderQueue,
    msg_context_threshold,
    msg_empty_response,
    msg_final_iteration,
    msg_output_truncated,
    msg_tool_loop_halt,
    msg_tool_loop_warning,
)
from core.memory import MemoryManager
from core.memory.shortterm import ShortTermMemory
from core.prompt.builder import build_system_prompt
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig

if TYPE_CHECKING:
    from core.tooling.handler import ToolHandler

logger = logging.getLogger("animaworks.execution.litellm_loop")


class LiteLLMExecutor(
    ToolProcessingMixin,
    ContextMixin,
    StreamingMixin,
    BaseExecutor,
):
    """Execute via LiteLLM with a tool_use loop (Mode A).

    The LLM calls tools autonomously (memory, files, commands, delegation)
    until it produces a final text response or the runaway guard halts it.

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
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run the LiteLLM tool-use loop.

        Returns ``ExecutionResult`` with the accumulated response text.
        """
        import litellm

        litellm.modify_params = True

        tools = self._build_base_tools(trigger=trigger)
        _active_categories: set[str] = set()
        context_window = self._resolve_cw()

        messages = self._build_initial_messages(
            system_prompt,
            prompt,
            images,
            prior_messages=prior_messages,
        )
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        llm_kwargs = self._build_llm_kwargs()
        # In-loop retry (call_llm_with_retry) is the single retry authority on
        # this path — LiteLLM's internal retries would multiply the effective
        # attempt count (worst case ~16) past the specified maximum of 3.
        llm_kwargs["num_retries"] = 0
        # Pre-flight rate-guard query.  This is observability-only: the session
        # continues even when the realm is guarded and relies on LiteLLM's own
        # retries — the guard's job is fleet-wide suppression at the one-shot
        # layer, not deferring Mode A cycles.  The realm is derived from the
        # model prefix (bedrock/vertex authenticate against distinct creds).
        _guard_family = provider_family_of(self._model_config.model)
        _guard_key = guard_key(_guard_family, litellm_realm_of(self._model_config.model))
        _guard = get_rate_guard()
        _guard_blocked = _guard.blocked_remaining(_guard_key)
        if _guard_blocked > 0:
            logger.info(
                "A session start: %s rate-guarded for %.0fs (continuing; retries apply)",
                _guard_key,
                _guard_blocked,
            )

        chain_count = 0
        usage_acc = TokenUsage()
        _empty_tracker = EmptyResponseTracker()

        def _salvage_text() -> str:
            """Prefer answer text already produced over the generic error."""
            salvaged = join_answer_parts(all_response_text)
            if not salvaged:
                return FINAL_RESPONSE_ERROR_TEXT
            logger.warning("A: no final answer; salvaging %d chars of prior text", len(salvaged))
            return salvaged

        _runaway_guard = RunawayGuard()
        _force_final = False
        _final_reminder_sent = False
        _classify = partial(classify_llm_error, provider_family=_guard_family)

        def _on_llm_error(reason: Any, hint: Any, exc: Exception, attempt: int) -> None:
            if reason in (FailoverReason.RATE_LIMIT, FailoverReason.OVERLOADED):
                _guard.report_block(
                    _guard_key,
                    hint.backoff_s or _guard.config.default_block_seconds,
                    reason.value,
                )
            elif reason in (FailoverReason.AUTH, FailoverReason.BILLING):
                logger.error(
                    "A LiteLLM %s — human attention required: %s",
                    reason.value,
                    exc,
                )

        iteration = -1
        while True:
            iteration += 1
            if (iteration + 1) % 10 == 0:
                logger.info("A tool loop progress: %d iterations", iteration + 1)
            if self._check_interrupted():
                logger.info("LiteLLM execute interrupted at iteration=%d", iteration)
                return ExecutionResult(text="[Session interrupted by user]", truncated=True)

            is_final_iteration = _force_final
            iter_tools = [] if is_final_iteration else tools

            if is_final_iteration and not _final_reminder_sent:
                _final_reminder_sent = True
                messages.append(
                    {
                        "role": "user",
                        "content": SystemReminderQueue.format_reminder(
                            msg_final_iteration(),
                        ),
                    }
                )
                logger.info(
                    "A final iteration=%d: tools removed, requesting final answer",
                    iteration,
                )

            logger.debug(
                "A tool loop iteration=%d messages=%d",
                iteration,
                len(messages),
            )

            iteration_messages = strip_tool_protocol_messages(messages) if is_final_iteration else messages

            # ── Pre-flight: clamp max_tokens to fit context window ──
            iter_kwargs = await self._preflight_clamp_with_compaction(
                llm_kwargs,
                iteration_messages,
                iter_tools,
                litellm,
            )
            if iter_kwargs is None:
                return ExecutionResult(
                    text=f"[Error: prompt too large for {self._model_config.model}]",
                    tool_call_records=all_tool_records,
                )

            call_kwargs: dict[str, Any] = {
                "messages": iteration_messages,
                **iter_kwargs,
            }
            if not is_final_iteration:
                call_kwargs["tools"] = tools

            try:
                response = await call_llm_with_retry(
                    partial(litellm.acompletion, **call_kwargs),
                    classify=_classify,
                    next_backoff=decorrelated_jitter,
                    interrupt_check=self._check_interrupted,
                    on_classified_error=_on_llm_error,
                )
            except LlmCallInterrupted:
                logger.info(
                    "LiteLLM execute interrupted during retry backoff at iteration=%d",
                    iteration,
                )
                return ExecutionResult(text="[Session interrupted by user]", truncated=True)
            except LLMAPIError:
                raise
            except Exception as e:
                # Classification + guard reporting happen per-attempt inside
                # call_llm_with_retry via _on_llm_error.
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
                _cr = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                _cw = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                if not _cr:
                    _ptd = getattr(response.usage, "prompt_tokens_details", None)
                    if _ptd:
                        _cr = getattr(_ptd, "cached_tokens", 0) or 0
                usage_acc.cache_read_tokens += _cr
                usage_acc.cache_write_tokens += _cw
                usage_dict = {"input_tokens": _inp, "output_tokens": _out}
                if tracker:
                    tracker.update_from_usage(usage_dict)

                if tracker and tracker.threshold_exceeded:
                    try:
                        ratio = float(tracker.usage_ratio)
                    except (TypeError, ValueError):
                        ratio = 0.0
                    self.reminder_queue.push_sync(msg_context_threshold(ratio=ratio))

                current_text = message.content or ""
                _, current_text = strip_thinking_tags(current_text)
                new_sys = None
                if not is_final_iteration:
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
                self.reminder_queue.push_sync(msg_output_truncated())

            # ── Check for tool calls ──────────────────────────
            tool_calls = message.tool_calls
            _content = message.content or ""
            if not tool_calls and iter_tools:
                _text_tc = try_parse_text_tool_call(_content, iter_tools)
                if _text_tc:
                    from types import SimpleNamespace

                    _tc_name, _tc_args_json = _text_tc
                    _tc_id = f"text_call_{iteration}_{id(_tc_name) % 0xFFFF:04x}"
                    logger.info(
                        "A text-format tool call parsed: %s args=%s",
                        _tc_name,
                        _tc_args_json,
                    )
                    _syn_fn = SimpleNamespace(name=_tc_name, arguments=_tc_args_json)
                    _syn_tc = SimpleNamespace(id=_tc_id, function=_syn_fn)
                    tool_calls = [_syn_tc]
                    _content = ""
            if not tool_calls:
                final_text = message.content or ""
                _, final_text = strip_thinking_tags(final_text)

                # ── Empty-response recovery ──
                if EmptyResponseTracker.is_empty(final_text, has_tool_calls=False):
                    if is_final_iteration:
                        record_finalization_failure(
                            self._anima_dir,
                            mode="A",
                            reason="empty_grace_response",
                        )
                        logger.error("A grace turn returned no answer at iteration=%d", iteration)
                        return ExecutionResult(
                            text=_salvage_text(),
                            tool_call_records=all_tool_records,
                            usage=usage_acc,
                            truncated=True,
                        )
                    if _empty_tracker.should_reprompt():
                        messages.append({"role": "assistant", "content": message.content or ""})
                        messages.append(
                            {
                                "role": "user",
                                "content": SystemReminderQueue.format_reminder(
                                    msg_empty_response(),
                                ),
                            }
                        )
                        logger.info(
                            "A empty response at iteration=%d; reprompting (%d used)",
                            iteration,
                            _empty_tracker.reprompts_used,
                        )
                        continue
                    logger.warning(
                        "A empty response persisted after reprompts at iteration=%d",
                        iteration,
                    )
                    record_finalization_failure(
                        self._anima_dir,
                        mode="A",
                        reason="empty_response_after_reprompts",
                    )
                    return ExecutionResult(
                        text=_salvage_text(),
                        tool_call_records=all_tool_records,
                        usage=usage_acc,
                        truncated=True,
                    )

                all_response_text.append(final_text)
                logger.debug("A final response at iteration=%d", iteration)
                # Drain undelivered reminders; do not append to user-facing text.
                final_reminder = self.reminder_queue.drain_formatted()
                if final_reminder:
                    logger.debug("Undelivered reminders at loop end: %s", final_reminder[:200])
                return ExecutionResult(
                    text=join_answer_parts(all_response_text),
                    tool_call_records=all_tool_records,
                    usage=usage_acc,
                    truncated=is_final_iteration,
                )

            # ── Process tool calls ────────────────────────────
            logger.info(
                "A tool calls at iteration=%d: %s",
                iteration,
                ", ".join(tc.function.name or "unknown" for tc in tool_calls),
            )

            try:
                parsed_calls = _convert_litellm_tool_calls(tool_calls)
            except (ValueError, KeyError, AttributeError) as exc:
                logger.warning(
                    "Tool call conversion failed (model=%s, iteration=%d): %s — treating as text response",
                    self._model_config.model,
                    iteration,
                    exc,
                )
                if _content:
                    _, _content = strip_thinking_tags(_content)
                    all_response_text.append(_content)
                continue

            if not parsed_calls:
                logger.info(
                    "All tool calls filtered (empty names) at iteration=%d — treating as text response",
                    iteration,
                )
                if _content:
                    _, _content = strip_thinking_tags(_content)
                    all_response_text.append(_content)
                continue

            # ── Runaway halt follow-up: the model kept calling tools even
            # after finalization was forced (Bedrock keeps toolConfig for API
            # compliance) — finalize with what has been gathered instead of
            # burning the remaining iterations.
            if is_final_iteration:
                logger.error(
                    "A tool call during finalization at iteration=%d — returning an explicit error",
                    iteration,
                )
                record_finalization_failure(
                    self._anima_dir,
                    mode="A",
                    reason="tool_call_during_grace_turn",
                )
                return ExecutionResult(
                    text=_salvage_text(),
                    tool_call_records=all_tool_records,
                    usage=usage_acc,
                    truncated=True,
                )

            # ── Runaway guard: consecutive identical tool-call turns ──
            _turn_sig = tuple(tool_call_signature(tc["name"], tc["arguments"]) for tc in parsed_calls)
            _guard_decision = _runaway_guard.observe(_turn_sig)
            if _guard_decision == RunawayGuard.HALT:
                _count = _runaway_guard.repetition_count
                logger.error(
                    "A runaway tool loop halted at iteration=%d (count=%d): %s",
                    iteration,
                    _count,
                    ", ".join(tc["name"] for tc in parsed_calls),
                )
                record_runaway_event(
                    self._anima_dir,
                    decision=_guard_decision,
                    mode="A",
                    iteration=iteration,
                    count=_count,
                    tool_names=sorted({tc["name"] for tc in parsed_calls}),
                )
                _force_final = True
                _final_reminder_sent = True
                messages.append(
                    {
                        "role": "user",
                        "content": SystemReminderQueue.format_reminder(
                            msg_tool_loop_halt(count=_count),
                        ),
                    }
                )
                continue
            if _guard_decision == RunawayGuard.WARN:
                _count = _runaway_guard.repetition_count
                self.reminder_queue.push_sync(
                    msg_tool_loop_warning(
                        tool_names=", ".join(sorted({tc["name"] for tc in parsed_calls})),
                        count=_count,
                    )
                )
                logger.warning(
                    "A runaway tool loop warning at iteration=%d (count=%d)",
                    iteration,
                    _count,
                )
                record_runaway_event(
                    self._anima_dir,
                    decision=_guard_decision,
                    mode="A",
                    iteration=iteration,
                    count=_count,
                    tool_names=sorted({tc["name"] for tc in parsed_calls}),
                )

            # Reconstruct assistant message with repaired arguments.
            # model_dump() would preserve malformed JSON that some models
            # (e.g. GLM-4.7) produce, causing 400 errors on the next
            # API call.  Re-serialize through json.dumps instead.
            _assistant_tc = []
            for tc in parsed_calls:
                if tc["arguments"] is not None:
                    _args_str = _json.dumps(tc["arguments"], ensure_ascii=False)
                else:
                    _args_str = "{}"
                _assistant_tc.append(
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": _args_str,
                        },
                    }
                )
            if _content:
                _, _content = strip_thinking_tags(_content)
                # Text written alongside the tool call belongs to the reply.
                if _content.strip():
                    all_response_text.append(_content)
            messages.append(
                {
                    "role": "assistant",
                    "content": _content or None,
                    "tool_calls": _assistant_tc,
                }
            )
            async for _event in self._process_streaming_tool_calls(
                parsed_calls,
                messages,
                tools,
                _active_categories,
                context_window=context_window,
            ):
                if "record" in _event:
                    all_tool_records.append(_event["record"])

            # ── Drain reminder queue after tool results ────────
            reminder = self.reminder_queue.drain_sync()
            if reminder:
                messages.append(
                    {
                        "role": "user",
                        "content": SystemReminderQueue.format_reminder(reminder),
                    }
                )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        trigger: str = "",
        thread_id: str = "default",
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
                system_prompt,
                prompt,
                tracker,
                images,
                prior_messages=prior_messages,
                trigger=trigger,
            ):
                yield event
        else:
            async for event in self._stream_token_level(
                system_prompt,
                prompt,
                tracker,
                images,
                prior_messages=prior_messages,
                trigger=trigger,
            ):
                yield event
