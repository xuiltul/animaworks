from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Anthropic SDK fallback executor.

Used when the Claude Agent SDK is not available but the model is Claude.
Calls the Anthropic messages API directly with tool_use for memory operations.
Handles mid-conversation context monitoring and session chaining.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from core.prompt.context import ContextTracker, resolve_context_window
from core.execution._session import build_continuation_prompt, handle_session_chaining
from core.execution._streaming import stream_error_boundary
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, ToolCallRecord, _truncate_for_record, tool_input_save_budget, tool_result_save_budget
from core.execution.reminder import MSG_CONTEXT_THRESHOLD, MSG_OUTPUT_TRUNCATED, SystemReminderQueue
from core.memory import MemoryManager
from core.prompt.builder import build_system_prompt
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from core.tooling.handler import ToolHandler
from core.tooling.guide import load_tool_schemas
from core.tooling.schemas import (
    build_tool_list,
    to_anthropic_format,
)
import httpx

logger = logging.getLogger("animaworks.execution.anthropic_fallback")


def _extract_tool_uses_from_messages(messages: list[dict]) -> list[dict]:
    """Extract tool_use info from Anthropic-format messages."""
    tool_uses: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_uses.append({
                            "name": block.get("name", ""),
                            "input": str(block.get("input", ""))[:500],
                        })
                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_uses.append({
                            "name": getattr(block, "name", ""),
                            "input": str(getattr(block, "input", ""))[:500],
                        })
        elif msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result" and tool_uses:
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                b.get("text", "") for b in result_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        tool_uses[-1]["result"] = str(result_content)[:500]
    return tool_uses[-20:]  # Keep last 20 entries


class AnthropicFallbackExecutor(BaseExecutor):
    """Fallback: use Anthropic SDK with tool_use for memory ops.

    Mid-conversation context monitoring: if the threshold is crossed,
    state is externalized and the conversation is restarted with
    restored short-term memory.
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

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build the Anthropic-format tool list."""
        external = load_tool_schemas(self._tool_registry, self._personal_tools)
        canonical = build_tool_list(
            include_file_tools=False,
            include_notification_tools=self._tool_handler._human_notifier is not None,
            include_admin_tools=(self._anima_dir / "skills" / "newstaff.md").exists(),
            include_supervisor_tools=self._has_subordinates(),
            include_tool_management=True,
            include_task_tools=True,
            include_skill_tools=True,
            skill_metas=self._memory.list_skill_metas(),
            common_skill_metas=self._memory.list_common_skill_metas(),
            procedure_metas=self._memory.list_procedure_metas(),
            external_schemas=external,
        )
        return to_anthropic_format(canonical)

    def _build_client(self):
        """Create an AsyncAnthropic client with resolved credentials."""
        import anthropic

        client_kwargs: dict[str, str] = {}
        api_key = self._resolve_api_key()
        if api_key:
            client_kwargs["api_key"] = api_key
        if self._model_config.api_base_url:
            client_kwargs["base_url"] = self._model_config.api_base_url
        return anthropic.AsyncAnthropic(**client_kwargs)

    def _build_initial_messages(
        self,
        prompt: str,
        images: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the initial user message with optional image content blocks."""
        if images:
            content_blocks: list[dict[str, Any]] = []
            for img in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                })
            content_blocks.append({"type": "text", "text": prompt})
            return [{"role": "user", "content": content_blocks}]
        return [{"role": "user", "content": prompt}]

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
        """Run Anthropic SDK with tool_use loop."""
        client = self._build_client()
        tools = self._build_tools()
        context_window = resolve_context_window(self._model_config.model)
        if prior_messages:
            messages = prior_messages  # Structured history including current msg
        else:
            messages = self._build_initial_messages(prompt, images)
        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        chain_count = 0
        max_iterations = max_turns_override or self._model_config.max_turns

        for iteration in range(max_iterations):
            logger.debug(
                "API call iteration=%d messages_count=%d",
                iteration, len(messages),
            )
            try:
                response = await client.messages.create(
                    model=self._model_config.model,
                    max_tokens=self._model_config.max_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    timeout=httpx.Timeout(self._resolve_llm_timeout()),
                )
            except Exception as e:
                logger.exception("Anthropic API error")
                return ExecutionResult(text=f"[LLM API Error: {e}]")

            # ── Context tracking + session chaining ───────────
            if tracker:
                usage_dict = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                tracker.update_from_usage(usage_dict)

                # P1-1: context threshold reminder
                if tracker.threshold_exceeded:
                    try:
                        ratio = float(tracker.usage_ratio)
                    except (TypeError, ValueError):
                        ratio = 0.0
                    self.reminder_queue.push_sync(
                        MSG_CONTEXT_THRESHOLD.format(ratio=ratio)
                    )

                current_text = "\n".join(
                    b.text for b in response.content if b.type == "text"
                )
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
                        message=prompt,
                    ),
                    max_chains=self._model_config.max_chains,
                    chain_count=chain_count,
                    session_id="anthropic-fallback",
                    trigger="anthropic_sdk",
                    original_prompt=prompt,
                    accumulated_response="\n".join(all_response_text),
                    turn_count=iteration,
                    tool_uses=_extract_tool_uses_from_messages(messages),
                )
                if new_sys is not None:
                    all_response_text.append(current_text)
                    system_prompt = new_sys
                    messages = [
                        {"role": "user", "content": build_continuation_prompt()}
                    ]
                    continue

            # ── P1-2: output truncation reminder ─────────────────
            if response.stop_reason == "max_tokens":
                self.reminder_queue.push_sync(MSG_OUTPUT_TRUNCATED)

            # ── Check for tool use ────────────────────────────
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                logger.debug("Final response received at iteration=%d", iteration)
                final_text = "\n".join(
                    b.text for b in response.content if b.type == "text"
                )
                all_response_text.append(final_text)
                # ── Final drain: deliver any undelivered reminders ──
                final_reminder = self.reminder_queue.drain_formatted()
                if final_reminder:
                    all_response_text.append(final_reminder)
                return ExecutionResult(
                    text="\n".join(all_response_text),
                    tool_call_records=all_tool_records,
                )

            # ── Process tool calls ────────────────────────────
            logger.info(
                "Tool calls at iteration=%d: %s",
                iteration, ", ".join(tu.name for tu in tool_uses),
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                result = self._tool_handler.handle(tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })
                all_tool_records.append(ToolCallRecord(
                    tool_name=tu.name,
                    tool_id=tu.id,
                    input_summary=_truncate_for_record(str(tu.input), tool_input_save_budget(context_window)),
                    result_summary=_truncate_for_record(str(result), tool_result_save_budget(tu.name, context_window)),
                ))
            messages.append({"role": "user", "content": tool_results})

            # ── Drain reminder queue into tool result message ──
            reminder = self.reminder_queue.drain_sync()
            if reminder:
                formatted = SystemReminderQueue.format_reminder(reminder)
                last_content = messages[-1]["content"]
                if isinstance(last_content, list):
                    last_content.append({
                        "type": "text",
                        "text": formatted,
                    })
                elif isinstance(last_content, str):
                    messages[-1]["content"] += "\n\n" + formatted

        logger.warning("Max iterations (%d) reached, returning fallback response", max_iterations)
        return ExecutionResult(
            text="\n".join(all_response_text) or "(max iterations reached)",
            tool_call_records=all_tool_records,
        )

    # ── Streaming execution ───────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream execution events using the Anthropic SDK messages.stream().

        Yields event dicts matching the A1 streaming protocol:
            - ``{"type": "text_delta", "text": "..."}``
            - ``{"type": "tool_start", "tool_name": "...", "tool_id": "..."}``
            - ``{"type": "tool_end", "tool_id": "...", "tool_name": "..."}``
            - ``{"type": "done", "full_text": "...", "result_message": None}``

        Session chaining is NOT handled here — AgentCore manages that
        externally for streaming paths.
        """
        client = self._build_client()
        tools = self._build_tools()
        context_window = resolve_context_window(self._model_config.model)
        if prior_messages:
            messages = prior_messages
        else:
            messages = self._build_initial_messages(prompt, images)

        all_response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        _MAX_ITERATIONS = max_turns_override or self._model_config.max_turns

        async with stream_error_boundary(
            all_response_text, executor_name="AnthropicFallback",
        ):
            for iteration in range(_MAX_ITERATIONS):
                logger.debug(
                    "Streaming API call iteration=%d messages_count=%d",
                    iteration, len(messages),
                )

                # ── Stream one API round ──────────────────────
                iteration_text_parts: list[str] = []
                final_message = None

                async with client.messages.stream(
                    model=self._model_config.model,
                    max_tokens=self._model_config.max_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    timeout=httpx.Timeout(self._resolve_llm_timeout()),
                ) as stream:
                    async for event in stream:
                        # Text deltas — forward immediately
                        if event.type == "text":
                            iteration_text_parts.append(event.text)
                            yield {"type": "text_delta", "text": event.text}

                        # Content block start — detect tool_use blocks
                        elif event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "tool_use":
                                yield {
                                    "type": "tool_start",
                                    "tool_name": block.name,
                                    "tool_id": block.id,
                                }

                    # After consuming the full stream, get the final message
                    final_message = await stream.get_final_message()

                # ── Context tracking ──────────────────────────
                if tracker and final_message:
                    usage_dict = {
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    }
                    tracker.update_from_usage(usage_dict)

                    # P1-1: context threshold reminder
                    if tracker.threshold_exceeded:
                        try:
                            ratio = float(tracker.usage_ratio)
                        except (TypeError, ValueError):
                            ratio = 0.0
                        self.reminder_queue.push_sync(
                            MSG_CONTEXT_THRESHOLD.format(ratio=ratio)
                        )

                # P1-2: output truncation reminder
                if final_message and getattr(final_message, "stop_reason", None) == "max_tokens":
                    self.reminder_queue.push_sync(MSG_OUTPUT_TRUNCATED)

                # ── Check for tool use ────────────────────────
                tool_uses = [
                    b for b in final_message.content if b.type == "tool_use"
                ]
                if not tool_uses:
                    # No tools — this is the final response
                    iteration_text = "".join(iteration_text_parts)
                    if iteration_text:
                        all_response_text.append(iteration_text)
                    logger.debug(
                        "Streaming final response at iteration=%d", iteration,
                    )
                    break

                # ── Execute tool calls ────────────────────────
                iteration_text = "".join(iteration_text_parts)
                if iteration_text:
                    all_response_text.append(iteration_text)

                logger.info(
                    "Streaming tool calls at iteration=%d: %s",
                    iteration, ", ".join(tu.name for tu in tool_uses),
                )
                messages.append({
                    "role": "assistant",
                    "content": final_message.content,
                })

                loop = asyncio.get_running_loop()
                tool_results = []
                for tu in tool_uses:
                    try:
                        result = await loop.run_in_executor(
                            None,
                            self._tool_handler.handle,
                            tu.name,
                            tu.input,
                        )
                    except Exception as tool_err:
                        logger.exception("Tool execution error: %s", tu.name)
                        result = f"ツール実行エラー: {tool_err}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    })
                    all_tool_records.append(ToolCallRecord(
                        tool_name=tu.name,
                        tool_id=tu.id,
                        input_summary=_truncate_for_record(str(tu.input), tool_input_save_budget(context_window)),
                        result_summary=_truncate_for_record(str(result), tool_result_save_budget(tu.name, context_window)),
                    ))
                    yield {
                        "type": "tool_end",
                        "tool_id": tu.id,
                        "tool_name": tu.name,
                    }
                messages.append({"role": "user", "content": tool_results})

                # ── Drain reminder queue into tool result message ──
                reminder = self.reminder_queue.drain_sync()
                if reminder:
                    formatted = SystemReminderQueue.format_reminder(reminder)
                    last_content = messages[-1]["content"]
                    if isinstance(last_content, list):
                        last_content.append({
                            "type": "text",
                            "text": formatted,
                        })
                    elif isinstance(last_content, str):
                        messages[-1]["content"] += "\n\n" + formatted
            else:
                # for-else: max iterations reached without break
                logger.warning(
                    "Streaming max iterations (%d) reached", _MAX_ITERATIONS,
                )

        full_text = "\n".join(all_response_text) or "(no response)"
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": None,
            "tool_call_records": [asdict(r) for r in all_tool_records],
        }
