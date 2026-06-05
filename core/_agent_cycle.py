from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CycleMixin -- blocking and streaming execution cycles.

Extracted from ``core.agent.AgentCore`` as a Mixin.  All ``self`` references
are resolved at runtime via MRO when mixed into ``AgentCore``.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from core.execution.base import ExecutionResult

from core._agent_prompt_log import _save_prompt_log, _save_prompt_log_end
from core.execution.session_context import RuntimeSessionContext, runtime_session_scope
from core.execution.session_types import is_clean_start_session, resolve_runtime_session_type, trigger_uses_chat_session
from core.i18n import t
from core.memory.shortterm import SessionState, ShortTermMemory
from core.prompt.builder import build_system_prompt, inject_shortterm
from core.prompt.context import CHARS_PER_TOKEN, ContextTracker
from core.schemas import CycleResult, ImageData, ModelConfig
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.agent")


_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def _update_tracker_from_prompt_estimate(
    tracker: ContextTracker,
    system_prompt: str,
    prompt: str,
) -> None:
    estimated_tokens = (len(system_prompt) + len(prompt)) // CHARS_PER_TOKEN
    tracker.update({"input_tokens": estimated_tokens}, include_output_in_ratio=False)


def _merge_stream_usage(acc: dict[str, int], chunk_usage: dict[str, int] | None) -> None:
    """Accumulate chunk usage into the streaming accumulator dict."""
    if not chunk_usage:
        return
    for k in _USAGE_KEYS:
        acc[k] = acc.get(k, 0) + (chunk_usage.get(k, 0) or 0)


def _log_session_token_usage(
    anima_dir: Path,
    *,
    model: str,
    mode: str,
    trigger: str,
    usage: dict[str, int] | None,
    duration_ms: int = 0,
    turns: int = 0,
    chains: int = 0,
) -> None:
    """Fire-and-forget token usage log entry."""
    if not usage or not any(usage.values()):
        return
    try:
        from core.memory.token_usage import TokenUsageLogger

        tul = TokenUsageLogger(anima_dir)
        tul.log(
            model=model,
            trigger=trigger,
            mode=mode,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_tokens", 0),
            cache_write_tokens=usage.get("cache_write_tokens", 0),
            turns=turns,
            chains=chains,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.debug("Failed to log token usage", exc_info=True)


class CycleMixin:
    """Mixin: blocking and streaming execution cycles + session chaining."""

    def _prepare_clean_start_session(
        self,
        *,
        trigger: str,
        session_type: str,
        thread_id: str,
        shortterm: ShortTermMemory,
    ) -> None:
        """Clear stale runtime state for non-chat sessions before execution."""
        if not is_clean_start_session(trigger):
            return

        try:
            shortterm.clear_for_clean_start()
        except Exception:
            logger.debug("Failed to clear non-chat shortterm state", exc_info=True)

        try:
            from core.execution._sdk_session import clear_session_id_for_type

            clear_session_id_for_type(self.anima_dir, session_type, thread_id)
        except Exception:
            logger.debug("Failed to clear non-chat SDK session ID", exc_info=True)

        try:
            from core.execution.codex_sdk import clear_codex_thread_id

            clear_codex_thread_id(self.anima_dir, session_type, thread_id)
        except Exception:
            logger.debug("Failed to clear non-chat Codex thread ID", exc_info=True)

    # ── Public API ─────────────────────────────────────────

    async def run_cycle(
        self,
        prompt: str,
        trigger: str = "manual",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
        max_turns_override: int | None = None,
        thread_id: str = "default",
        model_config_override: ModelConfig | None = None,
    ) -> CycleResult:
        """Run one agent cycle with autonomous memory search.

        Routing:
          - Mode B (basic):      ``AssistedExecutor``  -- text-based tool loop
          - Mode A (autonomous): ``LiteLLMExecutor`` -- LiteLLM + tool_use
          - Mode C (codex):      ``CodexSDKExecutor`` -- Codex CLI wrapper
          - Mode S (SDK):        ``AgentSDKExecutor`` -- Claude Agent SDK

        If the context threshold is crossed (A mode only), the session is
        externalized to short-term memory and automatically continued.
        S and C modes rely on the SDK's built-in context management.
        """
        async with self._get_agent_lock(thread_id):
            return await self._run_cycle_inner(
                prompt,
                trigger,
                images=images,
                prior_messages=prior_messages,
                message_intent=message_intent,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
                model_config_override=model_config_override,
            )

    async def _run_cycle_inner(
        self,
        prompt: str,
        trigger: str,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
        max_turns_override: int | None = None,
        thread_id: str = "default",
        model_config_override: ModelConfig | None = None,
    ) -> CycleResult:
        session_type = resolve_runtime_session_type(trigger)
        ctx = RuntimeSessionContext.create(
            session_type=session_type,
            thread_id=thread_id,
            trigger=trigger,
        )
        with runtime_session_scope(ctx):
            self._tool_handler.bind_runtime_session(ctx)
            token = self._tool_handler.set_active_session_type(session_type)
            try:
                result = await self._run_cycle_inner_scoped(
                    prompt,
                    trigger,
                    images=images,
                    prior_messages=prior_messages,
                    message_intent=message_intent,
                    max_turns_override=max_turns_override,
                    thread_id=thread_id,
                    model_config_override=model_config_override,
                )
                result.session_type = ctx.session_type
                result.thread_id = ctx.thread_id
                result.request_id = ctx.request_id
                result.tool_session_id = ctx.tool_session_id
                return result
            finally:
                from core.tooling.handler import active_session_type

                try:
                    active_session_type.reset(token)
                except (TypeError, ValueError):
                    pass

    async def _run_cycle_inner_scoped(
        self,
        prompt: str,
        trigger: str,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
        max_turns_override: int | None = None,
        thread_id: str = "default",
        model_config_override: ModelConfig | None = None,
    ) -> CycleResult:
        start = time.monotonic()
        active_model_config = model_config_override or self.model_config
        active_executor = (
            self._create_executor(active_model_config) if model_config_override is not None else self._executor
        )
        mode = self._resolve_execution_mode(active_model_config)
        logger.info(
            "run_cycle START trigger=%s prompt_len=%d mode=%s",
            trigger,
            len(prompt),
            mode,
        )

        # ── Resolve context window and prompt tier ────────────
        from core.prompt.builder import resolve_prompt_tier
        from core.prompt.context import resolve_context_window

        _ctx_window = resolve_context_window(
            active_model_config.model,
            overrides=self._load_context_window_overrides(),
        )
        _prompt_tier = resolve_prompt_tier(_ctx_window)

        # ── Priming: Automatic memory retrieval ────────────────
        priming_section, pending_human_notifications = await self._run_priming(
            prompt,
            trigger,
            message_intent=message_intent,
            prompt_tier=_prompt_tier,
            model_config=active_model_config,
        )

        session_type = resolve_runtime_session_type(trigger)
        uses_chat_session = trigger_uses_chat_session(trigger)
        shortterm = ShortTermMemory(self.anima_dir, session_type=session_type, thread_id=thread_id)
        self._prepare_clean_start_session(
            trigger=trigger,
            session_type=session_type,
            thread_id=thread_id,
            shortterm=shortterm,
        )
        tracker = ContextTracker(
            model=active_model_config.model,
            threshold=active_model_config.context_threshold,
            context_window_overrides=self._load_context_window_overrides(),
        )

        build_result = build_system_prompt(
            self.memory,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            priming_section=priming_section,
            execution_mode=mode,
            message=prompt,
            retriever=self._get_retriever(),
            trigger=trigger,
            context_window=_ctx_window,
            pending_human_notifications=pending_human_notifications,
            thread_id=thread_id,
        )
        system_prompt = build_result.system_prompt
        logger.debug("System prompt assembled, length=%d tier=%s", len(system_prompt), _prompt_tier)

        # ── Context-window-aware tier downgrade ────────────
        system_prompt = self._fit_prompt_to_context_window(
            system_prompt,
            prompt,
            _ctx_window,
            priming_section=priming_section,
            mode=mode,
            trigger=trigger,
            pending_human_notifications=pending_human_notifications,
            thread_id=thread_id,
        )

        if uses_chat_session and shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)
            logger.info("Injected short-term memory into system prompt")

        # ── Prompt log: save full payload for debugging ───
        from core.tooling.schemas import load_all_tool_schemas

        _tool_schemas = load_all_tool_schemas(
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
        )
        _save_prompt_log(
            self.anima_dir,
            trigger=trigger,
            sender=self._extract_sender(prompt, trigger),
            model=active_model_config.model,
            mode=mode,
            system_prompt=system_prompt,
            user_message=prompt,
            tools=self._tool_registry,
            session_id=self._tool_handler.session_id,
            context_window=_ctx_window,
            prior_messages=prior_messages,
            tool_schemas=_tool_schemas,
        )

        # ── Helper: convert ExecutionResult tool records to dicts ──
        def _tool_records_to_dicts(result: ExecutionResult) -> list[dict]:
            from dataclasses import asdict as _asdict

            return [_asdict(r) for r in result.tool_call_records]

        # ── Mode B: text-based tool-call loop ─────────────
        if mode == "b":
            result = await active_executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                trigger=trigger,
                images=images,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
            )
            _save_prompt_log_end(
                self.anima_dir,
                session_id=self._tool_handler.session_id,
                tool_call_count=len(result.tool_call_records),
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (mode-b) trigger=%s duration_ms=%d response_len=%d",
                trigger,
                duration_ms,
                len(result.text),
            )
            _b_usage = result.usage.to_dict() if result.usage else None
            _log_session_token_usage(
                self.anima_dir,
                model=active_model_config.model,
                mode="b",
                trigger=trigger,
                usage=_b_usage,
                duration_ms=duration_ms,
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
                context_window=tracker.context_window,
                context_threshold=tracker.threshold,
                tool_call_records=_tool_records_to_dicts(result),
                usage=_b_usage,
            )

        # ── Mode C: Codex SDK ─────────────────────────────
        if mode == "c":
            _update_tracker_from_prompt_estimate(tracker, system_prompt, prompt)
            result = await active_executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                trigger=trigger,
                images=images,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
            )
            if result.replied_to_from_transcript:
                self._tool_handler.merge_replied_to(result.replied_to_from_transcript)
            _save_prompt_log_end(
                self.anima_dir,
                session_id=self._tool_handler.session_id,
                tool_call_count=len(result.tool_call_records),
            )
            if tracker.threshold_exceeded and uses_chat_session:
                shortterm.clear()
                shortterm.save(
                    SessionState(
                        session_id=result.result_message.session_id if result.result_message else "",
                        timestamp=now_iso(),
                        trigger=trigger,
                        original_prompt=prompt,
                        accumulated_response=result.text,
                        context_usage_ratio=tracker.usage_ratio,
                        turn_count=result.result_message.num_turns if result.result_message else 0,
                    )
                )
                try:
                    from core.execution.codex_sdk import clear_codex_thread_ids

                    clear_codex_thread_ids(self.anima_dir, thread_id)
                except Exception:
                    logger.debug("Failed to clear Codex thread ID after Mode C threshold", exc_info=True)
            elif uses_chat_session:
                shortterm.clear()
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (c) trigger=%s duration_ms=%d response_len=%d",
                trigger,
                duration_ms,
                len(result.text),
            )
            _c_usage = result.usage.to_dict() if result.usage else None
            _log_session_token_usage(
                self.anima_dir,
                model=active_model_config.model,
                mode="c",
                trigger=trigger,
                usage=_c_usage,
                duration_ms=duration_ms,
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
                context_window=tracker.context_window,
                context_threshold=tracker.threshold,
                tool_call_records=_tool_records_to_dicts(result),
                usage=_c_usage,
            )

        # ── Mode D: Cursor Agent CLI ─────────────────────
        if mode == "d":
            result = await active_executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                trigger=trigger,
                images=images,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
            )
            if result.replied_to_from_transcript:
                self._tool_handler.merge_replied_to(result.replied_to_from_transcript)
            _save_prompt_log_end(
                self.anima_dir,
                session_id=self._tool_handler.session_id,
                tool_call_count=len(result.tool_call_records),
            )
            if result.session_rotation_pending and uses_chat_session:
                from dataclasses import asdict as _d_asdict

                shortterm.save(
                    SessionState(
                        timestamp=now_iso(),
                        trigger=trigger,
                        original_prompt=prompt,
                        accumulated_response=result.text[-2000:] if result.text else "",
                        tool_uses=[_d_asdict(r) for r in result.tool_call_records],
                        turn_count=0,
                    )
                )
                logger.info("Mode D rotation pending — saved shortterm for next turn")
            elif uses_chat_session:
                shortterm.clear()
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (d) trigger=%s duration_ms=%d response_len=%d",
                trigger,
                duration_ms,
                len(result.text),
            )
            _d_usage = result.usage.to_dict() if result.usage else None
            _log_session_token_usage(
                self.anima_dir,
                model=active_model_config.model,
                mode="d",
                trigger=trigger,
                usage=_d_usage,
                duration_ms=duration_ms,
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
                context_window=tracker.context_window,
                context_threshold=tracker.threshold,
                tool_call_records=_tool_records_to_dicts(result),
                usage=_d_usage,
            )

        # ── Mode G: Gemini CLI ─────────────────────────────
        if mode == "g":
            result = await active_executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                trigger=trigger,
                images=images,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
            )
            if result.replied_to_from_transcript:
                self._tool_handler.merge_replied_to(result.replied_to_from_transcript)
            _save_prompt_log_end(
                self.anima_dir,
                session_id=self._tool_handler.session_id,
                tool_call_count=len(result.tool_call_records),
            )
            if uses_chat_session:
                shortterm.clear()
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (g) trigger=%s duration_ms=%d response_len=%d",
                trigger,
                duration_ms,
                len(result.text),
            )
            _g_usage = result.usage.to_dict() if result.usage else None
            _log_session_token_usage(
                self.anima_dir,
                model=active_model_config.model,
                mode="g",
                trigger=trigger,
                usage=_g_usage,
                duration_ms=duration_ms,
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
                context_window=tracker.context_window,
                context_threshold=tracker.threshold,
                tool_call_records=_tool_records_to_dicts(result),
                usage=_g_usage,
            )

        # ── Mode A: LiteLLM tool_use loop ─────────────────
        if mode == "a":
            result = await active_executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                shortterm=shortterm if uses_chat_session else None,
                images=images,
                prior_messages=prior_messages,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
                trigger=trigger,
            )
            _save_prompt_log_end(
                self.anima_dir,
                session_id=self._tool_handler.session_id,
                tool_call_count=len(result.tool_call_records),
            )
            if uses_chat_session and not tracker.threshold_exceeded:
                shortterm.clear()
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (a) trigger=%s duration_ms=%d response_len=%d",
                trigger,
                duration_ms,
                len(result.text),
            )
            _a_usage = result.usage.to_dict() if result.usage else None
            _log_session_token_usage(
                self.anima_dir,
                model=active_model_config.model,
                mode="a",
                trigger=trigger,
                usage=_a_usage,
                duration_ms=duration_ms,
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
                context_window=tracker.context_window,
                context_threshold=tracker.threshold,
                tool_call_records=_tool_records_to_dicts(result),
                usage=_a_usage,
            )

        # ── Mode S: Claude Agent SDK ──────────────────────
        # Pre-flight: check prompt size to prevent Agent SDK buffer overflow
        conv_memory = None
        if uses_chat_session:
            from core.memory.conversation import ConversationMemory

            conv_memory = ConversationMemory(self.anima_dir, active_model_config, thread_id=thread_id)
        system_prompt, prompt, use_fallback = await self._preflight_size_check(
            system_prompt,
            prompt,
            conv_memory,
            priming_section=priming_section,
            mode=mode,
            message=prompt,
            trigger=trigger,
            context_window=_ctx_window,
            pending_human_notifications=pending_human_notifications,
        )
        if use_fallback:
            executor = self._create_fallback_executor(active_model_config)
            result = await executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                trigger=trigger,
                images=images,
                prior_messages=prior_messages,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
            )
        else:
            result = await active_executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                trigger=trigger,
                images=images,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
            )
        # Merge transcript-parsed replied_to for S mode
        if result.replied_to_from_transcript:
            self._tool_handler.merge_replied_to(result.replied_to_from_transcript)
            logger.info("Merged transcript replied_to: %s", result.replied_to_from_transcript)
        result_msg = result.result_message
        accumulated_tool_records = _tool_records_to_dicts(result)

        # Session chaining: if threshold was crossed, continue in a new session.
        # force_chain is set by S mode mid-session context auto-compact (PreToolUse
        # hook returned continue_=False).  In that case ResultMessage.usage may
        # not have updated the tracker, so we force the threshold flag.
        if result.force_chain and not tracker.threshold_exceeded:
            tracker.force_threshold()
            logger.info(
                "Context auto-compact: forcing threshold_exceeded for session "
                "chaining (S mode mid-session context budget exceeded)"
            )

        session_chained = False
        total_turns = result_msg.num_turns if result_msg else 0
        chain_count = 0
        accumulated_text = result.text

        if tracker.threshold_exceeded and uses_chat_session:
            # Save shortterm for the next message to pick up via inject_shortterm.
            # Do NOT chain here — chaining mid-response causes the LLM to produce
            # unnatural "session handoff" messages.
            logger.info(
                "Session context at %.1f%% — saving shortterm, will resume on next message",
                tracker.usage_ratio * 100,
            )
            shortterm.clear()
            shortterm.save(
                SessionState(
                    session_id=result_msg.session_id if result_msg else "",
                    timestamp=now_iso(),
                    trigger=trigger,
                    original_prompt=prompt,
                    accumulated_response=accumulated_text,
                    context_usage_ratio=tracker.usage_ratio,
                    turn_count=result_msg.num_turns if result_msg else 0,
                )
            )
            # Clear SDK session ID so the next session starts fresh
            if mode == "s":
                try:
                    from core.execution._sdk_session import (
                        _RESUMABLE_SESSION_TYPES,
                        _clear_session_id,
                        _resolve_session_type,
                    )

                    _st = _resolve_session_type(trigger)
                    if _st in _RESUMABLE_SESSION_TYPES:
                        _clear_session_id(self.anima_dir, _st, thread_id)
                except Exception:
                    logger.debug("Failed to clear session ID for deferred chain", exc_info=True)
        elif uses_chat_session:
            shortterm.clear()

        _save_prompt_log_end(
            self.anima_dir,
            session_id=self._tool_handler.session_id,
            tool_call_count=len(accumulated_tool_records),
        )

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "run_cycle END trigger=%s duration_ms=%d response_len=%d chained=%s",
            trigger,
            duration_ms,
            len(accumulated_text),
            session_chained,
        )
        _cycle_usage = result.usage.to_dict() if result.usage else None
        _log_session_token_usage(
            self.anima_dir,
            model=active_model_config.model,
            mode="s",
            trigger=trigger,
            usage=_cycle_usage,
            duration_ms=duration_ms,
            turns=total_turns,
            chains=chain_count if session_chained else 0,
        )
        return CycleResult(
            trigger=trigger,
            action="responded",
            summary=accumulated_text,
            duration_ms=duration_ms,
            context_usage_ratio=tracker.usage_ratio,
            context_window=tracker.context_window,
            context_threshold=tracker.threshold,
            session_chained=session_chained,
            total_turns=total_turns,
            tool_call_records=accumulated_tool_records,
            usage=_cycle_usage,
        )

    # ── Streaming ──────────────────────────────────────────

    async def run_cycle_streaming(
        self,
        prompt: str,
        trigger: str = "manual",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
        max_turns_override: int | None = None,
        thread_id: str = "default",
        model_config_override: ModelConfig | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Streaming version of run_cycle.

        Yields stream chunks. Session chaining is handled seamlessly.
        Final event is ``{"type": "cycle_done", "cycle_result": {...}}``.
        """
        session_type = resolve_runtime_session_type(trigger)
        ctx = RuntimeSessionContext.create(
            session_type=session_type,
            thread_id=thread_id,
            trigger=trigger,
        )
        async with self._get_agent_lock(thread_id):
            with runtime_session_scope(ctx):
                self._tool_handler.bind_runtime_session(ctx)
                token = self._tool_handler.set_active_session_type(session_type)
                try:
                    async for chunk in self._run_cycle_streaming_inner(
                        prompt,
                        trigger,
                        images=images,
                        prior_messages=prior_messages,
                        message_intent=message_intent,
                        max_turns_override=max_turns_override,
                        thread_id=thread_id,
                        model_config_override=model_config_override,
                    ):
                        if chunk.get("type") == "cycle_done":
                            cycle_result = chunk.get("cycle_result")
                            if isinstance(cycle_result, dict):
                                cycle_result["session_type"] = cycle_result.get("session_type") or ctx.session_type
                                cycle_result["thread_id"] = cycle_result.get("thread_id") or ctx.thread_id
                                cycle_result["request_id"] = cycle_result.get("request_id") or ctx.request_id
                                cycle_result["tool_session_id"] = (
                                    cycle_result.get("tool_session_id") or ctx.tool_session_id
                                )
                        yield chunk
                finally:
                    from core.tooling.handler import active_session_type

                    try:
                        active_session_type.reset(token)
                    except (TypeError, ValueError):
                        pass

    async def _run_cycle_streaming_inner(
        self,
        prompt: str,
        trigger: str = "manual",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
        max_turns_override: int | None = None,
        thread_id: str = "default",
        model_config_override: ModelConfig | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Streaming implementation scoped by ``run_cycle_streaming``."""
        start = time.monotonic()
        active_model_config = model_config_override or self.model_config
        active_executor = (
            self._create_executor(active_model_config) if model_config_override is not None else self._executor
        )
        mode = self._resolve_execution_mode(active_model_config)
        logger.info(
            "run_cycle_streaming START trigger=%s prompt_len=%d mode=%s",
            trigger,
            len(prompt),
            mode,
        )

        # Non-streaming executors: fall back to blocking execution
        if not active_executor.supports_streaming:
            cycle = await self._run_cycle_inner_scoped(
                prompt,
                trigger,
                images=images,
                prior_messages=prior_messages,
                message_intent=message_intent,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
                model_config_override=model_config_override,
            )
            yield {"type": "text_delta", "text": cycle.summary}
            yield {
                "type": "cycle_done",
                "cycle_result": cycle.model_dump(mode="json"),
            }
            return

        # ── Resolve context window and prompt tier ────────────
        from core.prompt.builder import resolve_prompt_tier as _rpt
        from core.prompt.context import resolve_context_window as _rcw

        _ctx_window_s = _rcw(
            active_model_config.model,
            overrides=self._load_context_window_overrides(),
        )
        _prompt_tier_s = _rpt(_ctx_window_s)

        # ── Streaming executor (S / A / all modes) ───────────────
        priming_section, pending_human_notifications = await self._run_priming(
            prompt,
            trigger,
            message_intent=message_intent,
            prompt_tier=_prompt_tier_s,
            model_config=active_model_config,
        )

        session_type = resolve_runtime_session_type(trigger)
        uses_chat_session = trigger_uses_chat_session(trigger)
        shortterm = ShortTermMemory(self.anima_dir, session_type=session_type, thread_id=thread_id)
        self._prepare_clean_start_session(
            trigger=trigger,
            session_type=session_type,
            thread_id=thread_id,
            shortterm=shortterm,
        )
        tracker = ContextTracker(
            model=active_model_config.model,
            threshold=active_model_config.context_threshold,
            context_window_overrides=self._load_context_window_overrides(),
        )

        build_result = build_system_prompt(
            self.memory,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            priming_section=priming_section,
            execution_mode=mode,
            message=prompt,
            retriever=self._get_retriever(),
            trigger=trigger,
            context_window=_ctx_window_s,
            pending_human_notifications=pending_human_notifications,
            thread_id=thread_id,
        )
        system_prompt = build_result.system_prompt

        # ── Context-window-aware tier downgrade ────────────
        system_prompt = self._fit_prompt_to_context_window(
            system_prompt,
            prompt,
            _ctx_window_s,
            priming_section=priming_section,
            mode=mode,
            trigger=trigger,
            pending_human_notifications=pending_human_notifications,
        )

        if uses_chat_session and shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)

        # Pre-flight size check for streaming path
        conv_memory = None
        if uses_chat_session:
            from core.memory.conversation import ConversationMemory

            conv_memory = ConversationMemory(self.anima_dir, active_model_config, thread_id=thread_id)
        system_prompt, prompt, use_fallback = await self._preflight_size_check(
            system_prompt,
            prompt,
            conv_memory,
            priming_section=priming_section,
            mode=mode,
            message=prompt,
            trigger=trigger,
            context_window=_ctx_window_s,
            pending_human_notifications=pending_human_notifications,
            thread_id=thread_id,
        )
        if use_fallback:
            logger.warning("Streaming fallback: using blocking S Fallback for oversized prompt")
            cycle = await self._run_cycle_inner_scoped(
                prompt,
                trigger,
                message_intent=message_intent,
                images=images,
                max_turns_override=max_turns_override,
                thread_id=thread_id,
                model_config_override=model_config_override,
            )
            yield {"type": "text_delta", "text": cycle.summary}
            yield {
                "type": "cycle_done",
                "cycle_result": cycle.model_dump(mode="json"),
            }
            return

        if mode == "c":
            _update_tracker_from_prompt_estimate(tracker, system_prompt, prompt)

        # ── Prompt log: save full payload for debugging ───
        from core.tooling.schemas import load_all_tool_schemas as _lats

        _tool_schemas_s = _lats(
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
        )
        _save_prompt_log(
            self.anima_dir,
            trigger=trigger,
            sender=self._extract_sender(prompt, trigger),
            model=active_model_config.model,
            mode=mode,
            system_prompt=system_prompt,
            user_message=prompt,
            tools=self._tool_registry,
            session_id=self._tool_handler.session_id,
            context_window=_ctx_window_s,
            prior_messages=prior_messages,
            tool_schemas=_tool_schemas_s,
        )

        # ── Stream retry configuration ────────────────────
        retry_cfg = self._load_stream_retry_config()
        checkpoint_enabled = retry_cfg["checkpoint_enabled"]
        max_retries = retry_cfg["retry_max"]
        retry_delay = retry_cfg["retry_delay_s"]

        # Primary session with checkpoint + retry support
        full_text_parts: list[str] = []
        thinking_text_parts: list[str] = []
        all_tool_call_records: list[dict] = []
        result_message: Any = None
        _stream_force_chain = False
        _stream_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        terminal_error_message = ""
        current_prompt = prompt
        current_system_prompt = system_prompt
        retry_count = 0

        while True:
            completed_tools: list[dict[str, Any]] = []
            text_parts_this_attempt: list[str] = []
            stream_succeeded = False

            try:
                async for chunk in active_executor.execute_streaming(
                    current_system_prompt,
                    current_prompt,
                    tracker,
                    images=images,
                    prior_messages=prior_messages,
                    max_turns_override=max_turns_override,
                    trigger=trigger,
                    thread_id=thread_id,
                ):
                    if chunk["type"] == "done":
                        full_text_parts.append(chunk["full_text"])
                        text_parts_this_attempt.append(chunk["full_text"])
                        result_message = chunk["result_message"]
                        # Accumulate tool call records from executor
                        all_tool_call_records.extend(chunk.get("tool_call_records", []))
                        _merge_stream_usage(_stream_usage, chunk.get("usage"))
                        # Merge transcript replied_to
                        transcript_replied = chunk.get("replied_to_from_transcript", set())
                        if transcript_replied:
                            self._tool_handler.merge_replied_to(transcript_replied)
                        # Capture force_chain from S mode auto-compact
                        if chunk.get("force_chain", False):
                            _stream_force_chain = True
                        stream_succeeded = True
                    elif chunk["type"] == "tool_end" and checkpoint_enabled:
                        record = chunk.get("record")
                        summary = (getattr(record, "result_summary", "") if record else "") or chunk.get(
                            "tool_name", "unknown"
                        )
                        completed_tools.append(
                            {
                                "tool_name": chunk.get("tool_name", ""),
                                "tool_id": chunk.get("tool_id", ""),
                                "summary": summary,
                            }
                        )
                        # Save checkpoint after each tool completion
                        from core.memory.shortterm import StreamCheckpoint

                        shortterm.save_checkpoint(
                            StreamCheckpoint(
                                timestamp=now_iso(),
                                trigger=trigger,
                                original_prompt=prompt,
                                completed_tools=completed_tools,
                                accumulated_text="\n".join(full_text_parts),
                                retry_count=retry_count,
                            )
                        )
                        if self._progress_callback:
                            self._progress_callback()
                        yield chunk
                    else:
                        if chunk["type"] == "text_delta":
                            text_parts_this_attempt.append(chunk.get("text", ""))
                            if self._progress_callback:
                                self._progress_callback()
                        elif chunk["type"] == "thinking_delta":
                            thinking_text_parts.append(chunk.get("text", ""))
                        yield chunk

            except Exception as e:
                from core.execution.base import StreamDisconnectedError

                is_stream_error = isinstance(e, StreamDisconnectedError)
                if not is_stream_error:
                    # Non-stream errors: log and break
                    logger.exception("Agent SDK streaming error (non-retryable)")
                    terminal_error_message = f"[Agent SDK Error: {e}]"
                    yield {"type": "error", "message": terminal_error_message}
                    break

                # ── Stream disconnect: attempt retry ──────────
                partial_text = getattr(e, "partial_text", "") or ""
                if partial_text:
                    full_text_parts.append(partial_text)

                if retry_count >= max_retries:
                    terminal_error_message = t("agent.stream_retry_exhausted", retry_count=retry_count)
                    logger.error(
                        "Stream retry exhausted (%d/%d)",
                        retry_count,
                        max_retries,
                    )
                    yield {
                        "type": "error",
                        "message": terminal_error_message,
                    }
                    break

                retry_count += 1
                skip_delay = getattr(e, "immediate_retry", False)
                actual_delay = 0.5 if skip_delay else retry_delay
                logger.warning(
                    "Stream disconnected, retrying %d/%d after %.1fs%s",
                    retry_count,
                    max_retries,
                    actual_delay,
                    " (immediate: buffer overflow)" if skip_delay else "",
                )
                # リトライ1回目は必ずfresh session（壊れたセッションIDを持ち越さない）
                if retry_count == 1:
                    try:
                        if mode == "c" and uses_chat_session:
                            from core.execution.codex_sdk import clear_codex_thread_ids

                            clear_codex_thread_ids(self.anima_dir, thread_id)
                        elif mode != "c":
                            from core.execution._sdk_session import (
                                _RESUMABLE_SESSION_TYPES,
                                _clear_session_id,
                                _resolve_session_type,
                            )

                            _st_retry = _resolve_session_type(trigger)
                            if _st_retry in _RESUMABLE_SESSION_TYPES:
                                _clear_session_id(self.anima_dir, _st_retry, thread_id)
                        logger.info("Session IDs cleared for retry 1 (fresh session forced)")
                    except Exception as e:
                        logger.warning("Failed to clear session IDs for retry: %s", e)
                yield {
                    "type": "retry_start",
                    "retry": retry_count,
                    "max_retries": max_retries,
                }

                # Load checkpoint and build retry prompt
                from core.execution._session import build_stream_retry_prompt
                from core.memory.shortterm import StreamCheckpoint

                checkpoint = shortterm.load_checkpoint()
                if checkpoint is None:
                    checkpoint = StreamCheckpoint(
                        timestamp=now_iso(),
                        trigger=trigger,
                        original_prompt=prompt,
                        completed_tools=completed_tools,
                        accumulated_text="\n".join(full_text_parts),
                        retry_count=retry_count,
                    )

                checkpoint.retry_count = retry_count
                current_prompt = build_stream_retry_prompt(checkpoint)

                # Reset tracker for fresh session
                tracker.reset()
                current_system_prompt = build_system_prompt(
                    self.memory,
                    tool_registry=self._tool_registry,
                    personal_tools=self._personal_tools,
                    priming_section=priming_section,
                    execution_mode=mode,
                    message=prompt,
                    retriever=self._get_retriever(),
                    trigger=trigger,
                    context_window=_ctx_window_s,
                    pending_human_notifications=pending_human_notifications,
                    thread_id=thread_id,
                ).system_prompt

                await asyncio.sleep(actual_delay)
                continue

            if stream_succeeded:
                # Clear checkpoint on success
                shortterm.clear_checkpoint()
                break

        if not uses_chat_session:
            shortterm.clear_checkpoint()

        session_chained = False
        total_turns = result_message.num_turns if result_message else 0
        chain_count = 0

        # Session chaining — force_chain from mid-session auto-compact.
        if _stream_force_chain and not tracker.threshold_exceeded:
            tracker.force_threshold()
            logger.info("Context auto-compact (stream): forcing threshold_exceeded")

        if tracker.threshold_exceeded and uses_chat_session:
            # Save shortterm for the next message to pick up via inject_shortterm.
            # Do NOT chain here — chaining mid-response causes the LLM to produce
            # unnatural "session handoff" messages.
            logger.info(
                "Session context at %.1f%% — saving shortterm, will resume on next message (stream)",
                tracker.usage_ratio * 100,
            )
            shortterm.clear()
            shortterm.save(
                SessionState(
                    session_id=result_message.session_id if result_message else "",
                    timestamp=now_iso(),
                    trigger=trigger,
                    original_prompt=prompt,
                    accumulated_response="\n".join(full_text_parts),
                    context_usage_ratio=tracker.usage_ratio,
                    turn_count=result_message.num_turns if result_message else 0,
                )
            )
            # Clear SDK session ID so the next session starts fresh
            if mode == "s":
                try:
                    from core.execution._sdk_session import (
                        _RESUMABLE_SESSION_TYPES,
                        _clear_session_id,
                        _resolve_session_type,
                    )

                    _st = _resolve_session_type(trigger)
                    if _st in _RESUMABLE_SESSION_TYPES:
                        _clear_session_id(self.anima_dir, _st, thread_id)
                except Exception:
                    logger.debug("Failed to clear session ID for deferred chain", exc_info=True)
            elif mode == "c":
                try:
                    from core.execution.codex_sdk import clear_codex_thread_ids

                    clear_codex_thread_ids(self.anima_dir, thread_id)
                except Exception:
                    logger.debug("Failed to clear Codex thread ID for deferred chain", exc_info=True)
        elif uses_chat_session:
            shortterm.clear()

        _save_prompt_log_end(
            self.anima_dir,
            session_id=self._tool_handler.session_id,
            tool_call_count=len(all_tool_call_records),
        )

        full_text = "\n".join(full_text_parts)
        thinking_text = "".join(thinking_text_parts)
        final_action = "error" if terminal_error_message else "responded"
        final_summary = full_text or terminal_error_message
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "run_cycle_streaming END trigger=%s duration_ms=%d response_len=%d chained=%s retries=%d",
            trigger,
            duration_ms,
            len(full_text),
            session_chained,
            retry_count,
        )

        _final_usage = _stream_usage if any(_stream_usage.values()) else None
        _log_session_token_usage(
            self.anima_dir,
            model=active_model_config.model,
            mode=mode,
            trigger=trigger,
            usage=_final_usage,
            duration_ms=duration_ms,
            turns=total_turns,
            chains=chain_count if session_chained else 0,
        )
        yield {
            "type": "cycle_done",
            "cycle_result": CycleResult(
                trigger=trigger,
                action=final_action,
                summary=final_summary,
                thinking_text=thinking_text[:10000],
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
                context_window=tracker.context_window,
                context_threshold=tracker.threshold,
                session_chained=session_chained,
                total_turns=total_turns,
                tool_call_records=all_tool_call_records,
                usage=_final_usage,
            ).model_dump(mode="json"),
        }
