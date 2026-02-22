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

from core.exceptions import LLMAPIError, ToolExecutionError, ConfigError  # noqa: F401
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, ToolCallRecord, _truncate_for_record, tool_input_save_budget, tool_result_save_budget
from core.execution.reminder import MSG_OUTPUT_TRUNCATED, SystemReminderQueue
from core.execution._streaming import stream_error_boundary
from core.memory import MemoryManager
from core.messenger import Messenger
from core.prompt.context import ContextTracker, resolve_context_window
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from core.tooling.handler import ToolHandler
from core.tooling.schemas import build_tool_list, to_text_format

logger = logging.getLogger("animaworks.execution.assisted")

_MAX_TOOL_OUTPUT_BYTES = 4096


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
    return (
        f"{truncated}\n"
        f"... [出力切り捨て: 元のサイズ {len(encoded)}バイト]"
    )


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
    ) -> None:
        super().__init__(model_config, anima_dir)
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
        """Build canonical tool schemas for text format generation."""
        from core.tooling.schemas import (
            load_external_schemas,
            load_personal_tool_schemas,
        )

        canonical = build_tool_list(
            include_file_tools=True,
            include_search_tools=True,
            include_discovery_tools=False,  # Not needed in text mode
            include_notification_tools=self._tool_handler._human_notifier is not None,
            include_tool_management=False,  # Not needed in text mode
            include_skill_tools=True,
            skill_metas=self._memory.list_skill_metas(),
            common_skill_metas=self._memory.list_common_skill_metas(),
            procedure_metas=self._memory.list_procedure_metas(),
        )

        # Load external tool schemas
        external = load_external_schemas(self._tool_registry)
        if external:
            canonical.extend(external)

        # Load personal tool schemas
        if self._personal_tools:
            personal = load_personal_tool_schemas(self._personal_tools)
            canonical.extend(personal)

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
        from core.prompt.context import resolve_context_window
        from core.config import load_config

        try:
            _cw_overrides = load_config().model_context_windows
        except Exception:
            logger.debug("Failed to load model context windows", exc_info=True)
            _cw_overrides = None

        ctx_window = resolve_context_window(
            self._model_config.model, _cw_overrides,
        )

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
            logger.error(
                "Mode B preflight: prompt too large "
                "(~%d tokens input, %d window)",
                est_input, ctx_window,
            )
            return None

        if available < configured_max:
            clamped = available - 128
            logger.info(
                "Mode B preflight: clamping max_tokens %d -> %d "
                "(est_input ~%d, window %d)",
                configured_max, clamped, est_input, ctx_window,
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

        kwargs: dict[str, Any] = {
            "model": self._model_config.model,
            "messages": messages,
            "max_tokens": max_tokens_override if max_tokens_override is not None else self._model_config.max_tokens,
            "timeout": self._resolve_llm_timeout(),
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

        # Ollama num_ctx: explicitly set context window to prevent silent truncation
        if self._model_config.model.startswith("ollama/"):
            from core.prompt.context import resolve_context_window
            from core.config import load_config
            try:
                _cw_overrides = load_config().model_context_windows
            except Exception:
                logger.debug("Failed to load model context windows", exc_info=True)
                _cw_overrides = None
            kwargs["num_ctx"] = resolve_context_window(
                self._model_config.model, _cw_overrides,
            )

        return await litellm.acompletion(**kwargs)

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
        """Run the text-based tool-call loop.

        Returns ``ExecutionResult`` with the accumulated response text.
        """
        if images:
            logger.warning(
                "Mode B (text-loop) does not support image input; "
                "images will be ignored"
            )
        logger.info(
            "Mode B text-loop START prompt_len=%d trigger=%s",
            len(prompt), trigger,
        )

        # ── 1. Build tool spec and augment system prompt ─────
        tool_spec = self._build_tool_spec_text()
        full_system = system_prompt + "\n\n" + tool_spec if system_prompt else tool_spec

        context_window = resolve_context_window(self._model_config.model)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        max_iterations = max_turns_override or self._model_config.max_turns

        # ── 2. Tool-call loop ────────────────────────────────
        for iteration in range(max_iterations):
            logger.debug(
                "Mode B iteration=%d messages=%d",
                iteration, len(messages),
            )

            # ── Preflight: check context window ───────────
            preflight = self._preflight_check(messages)
            if preflight is None:
                all_response_text.append(
                    f"[Error: prompt too large for "
                    f"{self._model_config.model}]"
                )
                break

            try:
                response = await self._call_llm(
                    messages,
                    max_tokens_override=preflight.get("max_tokens"),
                )
            except Exception as e:
                logger.exception("LiteLLM API error in Mode B")
                return ExecutionResult(text=f"[LLM API Error: {e}]")

            choice = response.choices[0]
            content = choice.message.content or ""

            # P1-2: output truncation reminder
            if choice.finish_reason == "length":
                self.reminder_queue.push_sync(MSG_OUTPUT_TRUNCATED)

            # ── 3. Extract tool call ─────────────────────────
            tool_call = extract_tool_call(content)
            if tool_call is None:
                # No tool call → final response
                all_response_text.append(content)
                logger.info(
                    "Mode B final response at iteration=%d len=%d",
                    iteration, len(content),
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
                    tool_name, sorted(self._known_tools)[:10],
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"エラー: 不明なツール '{tool_name}' です。"
                        f"利用可能なツール: {sorted(self._known_tools)}"
                    ),
                })
                continue

            # ── 5. Execute tool ───────────────────────────────
            logger.info(
                "Mode B tool call: %s args=%s",
                tool_name, list(tool_args.keys()),
            )

            tool_id = f"assisted_{iteration}_{tool_name}"
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._tool_handler.handle,
                    tool_name,
                    tool_args,
                )
            except Exception as e:
                logger.exception("Mode B tool execution error: %s", tool_name)
                result = f"ツール実行エラー: {e}"

            result = _truncate_tool_output(str(result))
            all_tool_records.append(ToolCallRecord(
                tool_name=tool_name,
                tool_id=tool_id,
                input_summary=_truncate_for_record(str(tool_args), tool_input_save_budget(context_window)),
                result_summary=_truncate_for_record(result, tool_result_save_budget(tool_name, context_window)),
            ))

            # ── 6. Inject result and continue ─────────────────
            narrative = _strip_tool_call_block(content)
            if narrative:
                all_response_text.append(narrative)

            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"ツール実行結果:\n{result}",
            })

            # ── Drain reminder queue into tool result message ──
            reminder = self.reminder_queue.drain_sync()
            if reminder:
                messages[-1]["content"] += "\n\n" + SystemReminderQueue.format_reminder(reminder)
        else:
            # max_turns reached
            logger.warning(
                "Mode B max iterations (%d) reached", max_iterations,
            )

        # ── Final drain: deliver any undelivered reminders ──
        final_reminder = self.reminder_queue.drain_formatted()
        if final_reminder:
            all_response_text.append(final_reminder)
        final_text = "\n".join(filter(None, all_response_text))
        logger.info("Mode B text-loop END total_len=%d", len(final_text))
        return ExecutionResult(
            text=final_text or "(max iterations reached)",
            tool_call_records=all_tool_records,
        )

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
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
            logger.warning(
                "Mode B (text-loop) streaming does not support image input; "
                "images will be ignored"
            )
        logger.info(
            "Mode B streaming START prompt_len=%d", len(prompt),
        )

        # ── 1. Build tool spec and augment system prompt ─────
        tool_spec = self._build_tool_spec_text()
        full_system = system_prompt + "\n\n" + tool_spec if system_prompt else tool_spec

        context_window = resolve_context_window(self._model_config.model)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        max_iterations = max_turns_override or self._model_config.max_turns

        # ── 2. Tool-call loop ────────────────────────────────
        async with stream_error_boundary(
            all_response_text, executor_name="Mode B",
        ):
            for iteration in range(max_iterations):
                logger.debug(
                    "Mode B streaming iteration=%d messages=%d",
                    iteration, len(messages),
                )

                # ── Preflight: check context window ───────────
                preflight = self._preflight_check(messages)
                if preflight is None:
                    error_msg = (
                        f"[Error: prompt too large for "
                        f"{self._model_config.model}]"
                    )
                    yield {"type": "text_delta", "text": error_msg}
                    break

                response = await self._call_llm(
                    messages,
                    max_tokens_override=preflight.get("max_tokens"),
                )
                choice = response.choices[0]
                content = choice.message.content or ""

                # P1-2: output truncation reminder
                if choice.finish_reason == "length":
                    self.reminder_queue.push_sync(MSG_OUTPUT_TRUNCATED)

                # ── 3. Extract tool call ─────────────────────
                tool_call = extract_tool_call(content)

                if tool_call is None:
                    # No tool call → final response
                    all_response_text.append(content)
                    if content:
                        yield {"type": "text_delta", "text": content}
                    logger.info(
                        "Mode B streaming final response at iteration=%d len=%d",
                        iteration, len(content),
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
                        tool_name, sorted(self._known_tools)[:10],
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"エラー: 不明なツール '{tool_name}' です。"
                            f"利用可能なツール: {sorted(self._known_tools)}"
                        ),
                    })
                    continue

                # ── 5. Yield narrative text (tool JSON stripped) ──
                narrative = _strip_tool_call_block(content)
                if narrative:
                    all_response_text.append(narrative)
                    yield {"type": "text_delta", "text": narrative}

                # ── 6. Yield tool_start ──────────────────────
                tool_id = f"assisted_{iteration}_{tool_name}"
                yield {
                    "type": "tool_start",
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                }

                # ── 7. Execute tool ──────────────────────────
                logger.info(
                    "Mode B streaming tool call: %s args=%s",
                    tool_name, list(tool_args.keys()),
                )

                try:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        self._tool_handler.handle,
                        tool_name,
                        tool_args,
                    )
                except Exception as e:
                    logger.exception(
                        "Mode B streaming tool execution error: %s", tool_name,
                    )
                    result = f"ツール実行エラー: {e}"

                result = _truncate_tool_output(str(result))
                all_tool_records.append(ToolCallRecord(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    input_summary=_truncate_for_record(str(tool_args), tool_input_save_budget(context_window)),
                    result_summary=_truncate_for_record(result, tool_result_save_budget(tool_name, context_window)),
                ))

                # ── 8. Yield tool_end ────────────────────────
                yield {
                    "type": "tool_end",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                }

                # ── 9. Inject result and continue ────────────
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"ツール実行結果:\n{result}",
                })

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
        }
