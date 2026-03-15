from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode B executor: text-based pseudo-tool-call loop.

Instead of using the ``tools`` API parameter (which some models like
GLM-flash don't support), this executor injects tool specifications as
plain text into the system prompt and parses JSON code blocks from the
LLM's response to detect tool calls.

Loop:
  1. LiteLLM acompletion (no ``tools`` parameter)
  2. Extract tool-call JSON from response text
  3. If tool call found → execute via ToolHandler → inject result → goto 1
  4. If no tool call → return final response
  5. If max_turns reached → return accumulated text
"""

import ast
import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.exceptions import (
    AnimaWorksError,
    ExecutionError,
    LLMAPIError,
    ToolExecutionError,
)  # noqa: F401
from core.execution._sanitize import wrap_tool_result
from core.execution._streaming import stream_error_boundary
from core.execution._tool_summary import make_tool_detail_chunk
from core.execution.base import (
    BaseExecutor,
    ExecutionResult,
    TokenUsage,
    ToolCallRecord,
    _truncate_for_record,
    strip_thinking_tags,
    tool_input_save_budget,
    tool_result_save_budget,
)
from core.execution.reminder import SystemReminderQueue, msg_output_truncated
from core.i18n import t
from core.memory import MemoryManager
from core.memory.shortterm import ShortTermMemory
from core.messenger import Messenger
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig
from core.tooling.handler import ToolHandler
from core.tooling.schemas import build_unified_tool_list, to_text_format

logger = logging.getLogger("animaworks.execution.assisted")

_MAX_TOOL_OUTPUT_BYTES = 4096
_MAX_INTENT_REPROMPTS = 2

_TOOL_INTENT_PATTERNS_JA = re.compile(
    r"(?:調べ|確認し|実行し|検索し|取得し|チェックし|見てみ|探し|読み込|読んで)"
    r"(?:ます|ましょう|てみます|ますね|ましょうか|ていきます)",
)
_TOOL_INTENT_PATTERNS_EN = re.compile(
    r"(?:I(?:'ll| will) (?:check|look|search|run|execute|fetch|retrieve|find|read))"
    r"|(?:Let me (?:check|look|search|run|execute|fetch|retrieve|find|read))",
    re.IGNORECASE,
)

_INTENT_REPROMPT_JA = t("assisted.intent_reprompt", locale="ja")
_INTENT_REPROMPT_EN = t("assisted.intent_reprompt", locale="en")


def _looks_like_tool_intent(text: str) -> bool:
    """Detect whether response text implies the model intended to call a tool.

    Returns True if the text contains phrases like "調べます" or "I'll check"
    without an actual tool-call JSON block.
    """
    if not text or len(text) > 2000:
        return False
    return bool(_TOOL_INTENT_PATTERNS_JA.search(text) or _TOOL_INTENT_PATTERNS_EN.search(text))


# ── JSON extraction ─────────────────────────────────────────


def extract_tool_call(text: str) -> dict | None:
    """Extract a tool-call JSON object from LLM response text.

    Uses a multi-stage fallback strategy to handle malformed JSON:
      1. ```json code block extraction
      2. Bare ``{"tool": ...}`` object extraction
      3. Standard ``json.loads``
      4. ``json_repair`` library (broken JSON repair)
      5. ``ast.literal_eval`` (Python dict literal fallback)

    Returns:
        Parsed dict with at least a ``"tool"`` key, or ``None`` if no
        tool call is detected.
    """
    json_str: str | None = None

    # Step 1: ```json ... ``` code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Step 2: Bare {"tool": ...} block (non-greedy, no nesting)
        match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
        if not match:
            # Step 2b: Allow nested braces
            match = re.search(r'(\{.*"tool".*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1) if match.lastindex else match.group(0)

    if json_str is None:
        return None

    # Step 3: Standard json.loads
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Step 4: json_repair (broken JSON repair)
    try:
        from json_repair import repair_json

        repaired = repair_json(json_str, return_objects=True, ensure_ascii=False)
        if isinstance(repaired, dict) and "tool" in repaired:
            return repaired
    except Exception:
        logger.debug("json_repair fallback failed", exc_info=True)

    # Step 5: ast.literal_eval (Python dict literal)
    try:
        parsed = ast.literal_eval(json_str)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed
    except (ValueError, SyntaxError):
        pass

    return None


def _strip_tool_call_block(text: str) -> str:
    """Remove the tool-call JSON code block from response text.

    Returns the surrounding text (thinking/explanation) without the
    tool invocation itself, so it can be accumulated as narrative.
    """
    # Remove ```json...``` blocks
    stripped = re.sub(r"```(?:json)?\s*\n?.*?```", "", text, flags=re.DOTALL)
    return stripped.strip()


def _truncate_tool_output(result: str, max_bytes: int = _MAX_TOOL_OUTPUT_BYTES) -> str:
    """Truncate tool output to prevent context overflow in Mode B."""
    encoded = result.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return result
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{truncated}\n" + t("assisted.output_truncated", size=len(encoded))


# ── Executor ────────────────────────────────────────────────


class AssistedExecutor(BaseExecutor):
    """Execute in text-based tool-call loop mode (Mode B).

    Flow:
      1. Build tool specification text from canonical schemas
      2. Inject tool spec into system prompt (provided by AgentCore)
      3. Loop: call LLM → parse tool call → execute → inject result
      4. Return accumulated response when no more tool calls or max_turns
    """

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        tool_handler: ToolHandler,
        memory: MemoryManager,
        messenger: Messenger | None = None,
        tool_registry: list[str] | None = None,
        personal_tools: dict[str, str] | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(model_config, anima_dir, interrupt_event=interrupt_event)
        self._tool_handler = tool_handler
        self._memory = memory
        self._messenger = messenger
        self._tool_registry = tool_registry or []
        self._personal_tools = personal_tools or {}
        self._known_tools = self._build_known_tools()

    def _build_known_tools(self) -> set[str]:
        """Build a whitelist of known tool names."""
        schemas = self._build_tool_schemas()
        return {s["name"] for s in schemas}

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """Build canonical tool schemas for text format generation (unified 18-tool schema)."""
        canonical = build_unified_tool_list(
            include_notification_tools=self._tool_handler._human_notifier is not None,
            include_supervisor_tools=self._has_subordinates(),
            skill_metas=self._memory.list_skill_metas(),
            common_skill_metas=self._memory.list_common_skill_metas(),
            procedure_metas=self._memory.list_procedure_metas(),
        )
        return canonical

    def _build_tool_spec_text(self) -> str:
        """Build the tool specification text for system prompt injection."""
        schemas = self._build_tool_schemas()
        return to_text_format(schemas)

    def _preflight_check(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Pre-flight context window check for Mode B.

        Returns a dict with ``max_tokens`` (possibly clamped), or ``None``
        if the prompt is too large to fit in the context window at all.
        """
        import litellm as _litellm

        ctx_window = self._resolve_cw()

        try:
            est_input = _litellm.token_counter(
                model=self._model_config.model,
                messages=messages,
            )
        except Exception:
            logger.debug("Token counter fallback to char estimate", exc_info=True)
            msg_chars = sum(len(str(m.get("content", ""))) for m in messages)
            est_input = msg_chars // 2

        available = ctx_window - est_input
        configured_max = self._model_config.max_tokens

        if available - 128 < 256:
            # Last resort: truncate system message to fit
            if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
                sys_content = messages[0]["content"]
                excess_tokens = est_input - ctx_window + 512
                excess_chars = excess_tokens * 4
                _min_sys_chars = 2000
                available_for_trim = len(sys_content) - _min_sys_chars
                if available_for_trim > 0:
                    trim_amount = min(excess_chars, available_for_trim)
                    messages[0]["content"] = sys_content[: len(sys_content) - trim_amount]
                    logger.warning(
                        "Mode B preflight: hard-truncated system prompt by %d chars to fit context window %d",
                        trim_amount,
                        ctx_window,
                    )
                    try:
                        est_input = _litellm.token_counter(
                            model=self._model_config.model,
                            messages=messages,
                        )
                    except Exception:
                        msg_chars = sum(len(str(m.get("content", ""))) for m in messages)
                        est_input = msg_chars // 2
                    available = ctx_window - est_input
                    if available - 128 >= 256:
                        clamped = min(available - 128, configured_max)
                        return {"max_tokens": clamped}

            logger.error(
                "Mode B preflight: prompt too large (~%d tokens input, %d window)",
                est_input,
                ctx_window,
            )
            return None

        if available < configured_max:
            clamped = available - 128
            logger.info(
                "Mode B preflight: clamping max_tokens %d -> %d (est_input ~%d, window %d)",
                configured_max,
                clamped,
                est_input,
                ctx_window,
            )
            return {"max_tokens": clamped}

        return {"max_tokens": configured_max}

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        max_tokens_override: int | None = None,
    ) -> Any:
        """Call LiteLLM ``acompletion`` without tools parameter."""
        import litellm

        litellm.modify_params = True

        from core.config.models import resolve_max_tokens
        from core.execution.base import (
            is_adaptive_model,
            is_anthropic_claude,
            is_bedrock_glm,
            is_bedrock_kimi,
            is_bedrock_qwen,
            resolve_thinking_effort,
        )

        _eff_max = (
            max_tokens_override
            if max_tokens_override is not None
            else resolve_max_tokens(
                self._model_config.model,
                self._model_config.max_tokens,
                self._model_config.thinking,
            )
        )
        kwargs: dict[str, Any] = {
            "model": self._model_config.model,
            "messages": messages,
            "max_tokens": _eff_max,
            "timeout": self._resolve_llm_timeout(),
            "num_retries": self._resolve_num_retries(),
        }

        api_key = self._resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        if self._model_config.api_base_url:
            kwargs["api_base"] = self._model_config.api_base_url
        self._apply_provider_kwargs(kwargs)

        # Extended thinking / reasoning control
        if self._model_config.thinking is not None:
            model = self._model_config.model
            if is_bedrock_kimi(model):
                if self._model_config.thinking:
                    kwargs["reasoning_config"] = resolve_thinking_effort(
                        model,
                        self._model_config.thinking_effort,
                    )
            elif is_bedrock_qwen(model) or is_bedrock_glm(model):
                if self._model_config.thinking:
                    kwargs["enable_thinking"] = True
            elif model.startswith("bedrock/"):
                if self._model_config.thinking:
                    kwargs["reasoning_effort"] = resolve_thinking_effort(
                        model,
                        self._model_config.thinking_effort,
                    )
            elif is_anthropic_claude(model):
                if self._model_config.thinking:
                    if is_adaptive_model(model):
                        kwargs["thinking"] = {"type": "adaptive"}
                        kwargs["reasoning_effort"] = resolve_thinking_effort(
                            model,
                            self._model_config.thinking_effort,
                        )
                    else:
                        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                    kwargs["temperature"] = 1
            elif model.startswith("openai/"):
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"]["enable_thinking"] = self._model_config.thinking
            else:
                kwargs["think"] = self._model_config.thinking
        elif self._model_config.model.startswith("ollama/"):
            kwargs["think"] = False

        # Ollama num_ctx: explicitly set context window to prevent silent truncation
        if self._model_config.model.startswith("ollama/"):
            kwargs["num_ctx"] = self._resolve_cw()

        return await litellm.acompletion(**kwargs)

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run the text-based tool-call loop.

        Returns ``ExecutionResult`` with the accumulated response text.
        """
        if images:
            logger.warning("Mode B (text-loop) does not support image input; images will be ignored")
        logger.info(
            "Mode B text-loop START prompt_len=%d trigger=%s",
            len(prompt),
            trigger,
        )

        # ── 1. Build tool spec and augment system prompt ─────
        tool_spec = self._build_tool_spec_text()
        full_system = system_prompt + "\n\n" + tool_spec if system_prompt else tool_spec

        context_window = self._resolve_cw()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        max_iterations = max_turns_override or self._model_config.max_turns
        intent_reprompt_count = 0
        usage_acc = TokenUsage()

        # ── 2. Tool-call loop ────────────────────────────────
        for iteration in range(max_iterations):
            logger.debug(
                "Mode B iteration=%d messages=%d",
                iteration,
                len(messages),
            )

            # ── Preflight: check context window ───────────
            preflight = self._preflight_check(messages)
            if preflight is None:
                all_response_text.append(f"[Error: prompt too large for {self._model_config.model}]")
                break

            try:
                response = await self._call_llm(
                    messages,
                    max_tokens_override=preflight.get("max_tokens"),
                )
            except LLMAPIError:
                raise
            except AnimaWorksError:
                raise
            except Exception as e:
                logger.exception("LiteLLM API error in Mode B")
                raise ExecutionError(str(e)) from e

            choice = response.choices[0]
            content = choice.message.content or ""
            if hasattr(response, "usage") and response.usage:
                usage_acc.input_tokens += response.usage.prompt_tokens or 0
                usage_acc.output_tokens += response.usage.completion_tokens or 0

            # P1-2: output truncation reminder
            if choice.finish_reason == "length":
                self.reminder_queue.push_sync(msg_output_truncated())

            # ── 3. Extract tool call ─────────────────────────
            tool_call = extract_tool_call(content)
            if tool_call is None:
                # Check for intent-without-action: model says "I'll check"
                # but doesn't actually call a tool.  Re-prompt up to
                # _MAX_INTENT_REPROMPTS times to coax out the JSON block.
                if intent_reprompt_count < _MAX_INTENT_REPROMPTS and _looks_like_tool_intent(content):
                    intent_reprompt_count += 1
                    logger.info(
                        "Mode B intent detected without tool call (reprompt %d/%d): %.80s",
                        intent_reprompt_count,
                        _MAX_INTENT_REPROMPTS,
                        content,
                    )
                    from core.tooling.prompt_db import _get_locale

                    reprompt = _INTENT_REPROMPT_JA if _get_locale() == "ja" else _INTENT_REPROMPT_EN
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": reprompt})
                    continue

                # No tool call → final response
                all_response_text.append(content)
                logger.info(
                    "Mode B final response at iteration=%d len=%d",
                    iteration,
                    len(content),
                )
                break

            # ── 4. Validate tool name ────────────────────────
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            if tool_name not in self._known_tools:
                logger.warning(
                    "Mode B unknown tool: %s (known: %s)",
                    tool_name,
                    sorted(self._known_tools)[:10],
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": t(
                            "assisted.unknown_tool",
                            tool_name=tool_name,
                            available=sorted(self._known_tools),
                        ),
                    }
                )
                continue

            # ── 5. Execute tool ───────────────────────────────
            logger.info(
                "Mode B tool call: %s args=%s",
                tool_name,
                list(tool_args.keys()),
            )

            tool_id = f"assisted_{iteration}_{tool_name}"
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._tool_handler.handle,
                    tool_name,
                    tool_args,
                    tool_id,
                )
            except ToolExecutionError as e:
                logger.warning("Mode B tool execution error: %s – %s", tool_name, e)
                result = t("assisted.tool_exec_error", error=e)
            except AnimaWorksError:
                raise
            except Exception as e:
                logger.exception("Mode B unexpected tool error: %s", tool_name)
                result = t("assisted.tool_exec_error", error=e)

            result = _truncate_tool_output(str(result))
            all_tool_records.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    input_summary=_truncate_for_record(str(tool_args), tool_input_save_budget(context_window)),
                    result_summary=_truncate_for_record(result, tool_result_save_budget(tool_name, context_window)),
                )
            )

            # ── 6. Inject result and continue ─────────────────
            narrative = _strip_tool_call_block(content)
            if narrative:
                all_response_text.append(narrative)

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": t("assisted.tool_result_header") + "\n" + wrap_tool_result(tool_name, result),
                }
            )

            # ── Drain reminder queue into tool result message ──
            reminder = self.reminder_queue.drain_sync()
            if reminder:
                messages[-1]["content"] += "\n\n" + SystemReminderQueue.format_reminder(reminder)
        else:
            # max_turns reached
            logger.warning(
                "Mode B max iterations (%d) reached",
                max_iterations,
            )

        # ── Final drain: deliver any undelivered reminders ──
        final_reminder = self.reminder_queue.drain_formatted()
        if final_reminder:
            all_response_text.append(final_reminder)
        final_text = "\n".join(filter(None, all_response_text))
        _, final_text = strip_thinking_tags(final_text)
        logger.info("Mode B text-loop END total_len=%d", len(final_text))
        return ExecutionResult(
            text=final_text or "(max iterations reached)",
            tool_call_records=all_tool_records,
            usage=usage_acc,
        )

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream execution events from the text-based tool-call loop.

        Mode B is iteration-level streaming: each loop iteration calls
        ``_call_llm()`` (blocking), then yields the full response as
        events.  This is not token-level streaming but gives the UI
        incremental progress as each tool call completes.

        Yields:
            ``{"type": "text_delta", "text": "..."}`` — narrative text
            ``{"type": "tool_start", "tool_name": "...", "tool_id": "..."}``
            ``{"type": "tool_end", "tool_id": "...", "tool_name": "..."}``
            ``{"type": "done", "full_text": "...", "result_message": None}``
        """
        if images:
            logger.warning("Mode B (text-loop) streaming does not support image input; images will be ignored")
        logger.info(
            "Mode B streaming START prompt_len=%d",
            len(prompt),
        )

        # ── 1. Build tool spec and augment system prompt ─────
        tool_spec = self._build_tool_spec_text()
        full_system = system_prompt + "\n\n" + tool_spec if system_prompt else tool_spec

        context_window = self._resolve_cw()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        max_iterations = max_turns_override or self._model_config.max_turns
        intent_reprompt_count = 0
        _usage_acc_bs = TokenUsage()

        # ── 2. Tool-call loop ────────────────────────────────
        async with stream_error_boundary(
            all_response_text,
            executor_name="Mode B",
        ):
            for iteration in range(max_iterations):
                logger.debug(
                    "Mode B streaming iteration=%d messages=%d",
                    iteration,
                    len(messages),
                )

                # ── Preflight: check context window ───────────
                preflight = self._preflight_check(messages)
                if preflight is None:
                    error_msg = f"[Error: prompt too large for {self._model_config.model}]"
                    yield {"type": "text_delta", "text": error_msg}
                    break

                response = await self._call_llm(
                    messages,
                    max_tokens_override=preflight.get("max_tokens"),
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                thinking, content = strip_thinking_tags(content)
                if thinking:
                    yield {"type": "thinking_start"}
                    yield {"type": "thinking_delta", "text": thinking}
                    yield {"type": "thinking_end"}
                if hasattr(response, "usage") and response.usage:
                    _usage_acc_bs.input_tokens += response.usage.prompt_tokens or 0
                    _usage_acc_bs.output_tokens += response.usage.completion_tokens or 0

                # P1-2: output truncation reminder
                if choice.finish_reason == "length":
                    self.reminder_queue.push_sync(msg_output_truncated())

                # ── 3. Extract tool call ─────────────────────
                tool_call = extract_tool_call(content)

                if tool_call is None:
                    # Check for intent-without-action (same as non-streaming)
                    if intent_reprompt_count < _MAX_INTENT_REPROMPTS and _looks_like_tool_intent(content):
                        intent_reprompt_count += 1
                        logger.info(
                            "Mode B streaming intent detected without tool call (reprompt %d/%d): %.80s",
                            intent_reprompt_count,
                            _MAX_INTENT_REPROMPTS,
                            content,
                        )
                        from core.tooling.prompt_db import _get_locale

                        reprompt = _INTENT_REPROMPT_JA if _get_locale() == "ja" else _INTENT_REPROMPT_EN
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": reprompt})
                        continue

                    # No tool call → final response
                    all_response_text.append(content)
                    if content:
                        yield {"type": "text_delta", "text": content}
                    logger.info(
                        "Mode B streaming final response at iteration=%d len=%d",
                        iteration,
                        len(content),
                    )
                    break

                # ── 4. Validate tool name ────────────────────
                tool_name = tool_call.get("tool", "")
                tool_args = tool_call.get("arguments", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                if tool_name not in self._known_tools:
                    logger.warning(
                        "Mode B streaming unknown tool: %s (known: %s)",
                        tool_name,
                        sorted(self._known_tools)[:10],
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": t(
                                "assisted.unknown_tool",
                                tool_name=tool_name,
                                available=sorted(self._known_tools),
                            ),
                        }
                    )
                    continue

                # ── 5. Yield narrative text (tool JSON stripped) ──
                narrative = _strip_tool_call_block(content)
                if narrative:
                    all_response_text.append(narrative)
                    yield {"type": "text_delta", "text": narrative}

                # ── 6. Yield tool_start + tool_detail ────────
                tool_id = f"assisted_{iteration}_{tool_name}"
                yield {
                    "type": "tool_start",
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                }
                detail_chunk = make_tool_detail_chunk(tool_name, tool_id, tool_args)
                if detail_chunk:
                    yield detail_chunk

                # ── 7. Execute tool ──────────────────────────
                logger.info(
                    "Mode B streaming tool call: %s args=%s",
                    tool_name,
                    list(tool_args.keys()),
                )

                try:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        self._tool_handler.handle,
                        tool_name,
                        tool_args,
                        tool_id,
                    )
                except ToolExecutionError as e:
                    logger.warning(
                        "Mode B streaming tool error: %s – %s",
                        tool_name,
                        e,
                    )
                    result = t("assisted.tool_exec_error", error=e)
                except AnimaWorksError:
                    raise
                except Exception as e:
                    logger.exception(
                        "Mode B streaming unexpected tool error: %s",
                        tool_name,
                    )
                    result = t("assisted.tool_exec_error", error=e)

                result = _truncate_tool_output(str(result))
                all_tool_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        tool_id=tool_id,
                        input_summary=_truncate_for_record(str(tool_args), tool_input_save_budget(context_window)),
                        result_summary=_truncate_for_record(result, tool_result_save_budget(tool_name, context_window)),
                    )
                )

                # ── 8. Yield tool_end ────────────────────────
                yield {
                    "type": "tool_end",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                }

                # ── 9. Inject result and continue ────────────
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": t("assisted.tool_result_header") + "\n" + wrap_tool_result(tool_name, result),
                    }
                )

                # ── Drain reminder queue into tool result message ──
                reminder = self.reminder_queue.drain_sync()
                if reminder:
                    messages[-1]["content"] += "\n\n" + SystemReminderQueue.format_reminder(reminder)
            else:
                # max_turns reached
                logger.warning(
                    "Mode B streaming max iterations (%d) reached",
                    max_iterations,
                )

        final_text = "\n".join(filter(None, all_response_text))
        logger.info("Mode B streaming END total_len=%d", len(final_text))
        yield {
            "type": "done",
            "full_text": final_text or "(max iterations reached)",
            "result_message": None,
            "tool_call_records": [asdict(r) for r in all_tool_records],
            "usage": _usage_acc_bs.to_dict(),
        }
