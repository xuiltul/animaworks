from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""PrimingMixin -- priming, context-window control, prompt size checking.

Extracted from ``core.agent.AgentCore`` as a Mixin.  All ``self`` references
are resolved at runtime via MRO when mixed into ``AgentCore``.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.conversation import ConversationMemory

from core._agent_prompt_log import _PROMPT_HARD_LIMIT_BYTES, _PROMPT_SOFT_LIMIT_BYTES
from core.i18n import t
from core.prompt.builder import build_system_prompt

logger = logging.getLogger("animaworks.agent")


class PrimingMixin:
    """Mixin: priming (auto-recall), context fitting, pre-flight size check."""

    def _compute_overflow_files(self) -> list[str] | None:
        """Always return None to enable full Channel C search.

        DK injection still runs in builder.py, but Channel C now searches
        all knowledge/procedures regardless of DK coverage.
        Phase 1 of DK removal: docs/issues/20260304_dk-removal-phase1-*.md
        """
        return None

    async def _run_priming(
        self,
        prompt: str,
        trigger: str,
        *,
        message_intent: str = "",
        overflow_files: list[str] | None = None,
        prompt_tier: str = "full",
    ) -> str:
        """Run priming layer to automatically retrieve relevant memories.

        Args:
            prompt: The user message (may include conversation history)
            trigger: Trigger type (e.g., "message:yamada")
            overflow_files: File stems that didn't fit in knowledge injection.
                None = legacy (full Channel C), [] = all injected (skip C),
                [...] = overflow files only for Channel C.
            prompt_tier: Prompt tier for budget control
                ("full"/"standard"/"light"/"minimal").

        Returns:
            Formatted priming section for system prompt injection, or empty string.
        """
        from core.memory.priming import PrimingEngine, format_priming_section
        from core.prompt.builder import TIER_LIGHT, TIER_MINIMAL, TIER_STANDARD

        if prompt_tier == TIER_MINIMAL:
            logger.debug("Priming: skipped (tier=minimal)")
            return ""

        sender_name = "human"
        if trigger.startswith("message:"):
            sender_name = trigger.split(":", 1)[1]

        message = self._extract_message_from_prompt(prompt)

        try:
            if not hasattr(self, "_priming_engine"):
                from core.paths import get_shared_dir
                from core.prompt.context import resolve_context_window as _rcw_priming

                ctx_window = _rcw_priming(
                    self.model_config.model,
                    overrides=self._load_context_window_overrides(),
                )
                self._priming_engine = PrimingEngine(
                    self.anima_dir,
                    get_shared_dir(),
                    context_window=ctx_window,
                )
                # Inject callback for active parallel tasks (DAG scheduler)
                if hasattr(self, "_active_parallel_tasks_getter"):
                    self._priming_engine._get_active_parallel_tasks = self._active_parallel_tasks_getter
            channel = "heartbeat" if trigger == "heartbeat" else "cron" if trigger.startswith("cron") else "chat"

            result = await self._priming_engine.prime_memories(
                message,
                sender_name,
                channel=channel,
                intent=message_intent,
                overflow_files=overflow_files,
                enable_dynamic_budget=True,
            )

            if result.is_empty():
                logger.debug("Priming: No memories found")
                return ""

            # T3 Light: sender_profile only
            if prompt_tier == TIER_LIGHT:
                if result.sender_profile:
                    logger.info("Priming: tier=light, returning sender_profile only")
                    return t("agent.priming_tier_light_header", sender_name=sender_name) + result.sender_profile
                return ""

            formatted = format_priming_section(result, sender_name)
            logger.info(
                "Priming: Retrieved %d tokens of memories (tier=%s)",
                result.estimated_tokens(),
                prompt_tier,
            )

            # T2 Standard: truncate to ~1000 tokens (≈4000 chars)
            if prompt_tier == TIER_STANDARD and len(formatted) > 4000:
                formatted = formatted[:4000] + t("agent.omitted_rest")

            return formatted

        except Exception:
            logger.exception("Priming failed; continuing without primed memories")
            return ""

    def _extract_message_from_prompt(self, prompt: str) -> str:
        """Extract the latest message content from a chat prompt.

        The prompt from ConversationMemory.build_chat_prompt() has format:
        - If no history: just the message content
        - If history: conversation history + separator + latest message

        We want to extract just the latest message for keyword extraction.
        """
        # Look for the pattern "**[HH:MM] from_person:**" which marks conversation history
        # The actual message is typically after the last history entry
        lines = prompt.strip().splitlines()

        # If there's no history marker, the whole prompt is the message
        if not any("**[" in line and "]" in line and ":**" in line for line in lines):
            return prompt

        # Find the last content block after history
        # Heuristic: take last paragraph that doesn't look like a history entry
        content_lines = []
        for line in reversed(lines):
            if line.startswith("**[") and "]" in line and ":**" in line:
                # Hit a history entry, stop
                break
            if line.strip():
                content_lines.insert(0, line)

        return "\n".join(content_lines) if content_lines else prompt

    # ── Context window overrides ─────────────────────────────

    def _load_context_window_overrides(self) -> dict[str, int]:
        """Load model_context_windows from config.json."""
        try:
            from core.config import load_config

            config = load_config()
            return config.model_context_windows
        except Exception:
            logger.debug("Failed to load context window overrides; using defaults")
            return {}

    # ── Stream retry config ─────────────────────────────────

    def _load_stream_retry_config(self) -> dict[str, Any]:
        """Load stream retry settings from config.json server section."""
        try:
            from core.config import load_config

            config = load_config()
            return {
                "checkpoint_enabled": config.server.stream_checkpoint_enabled,
                "retry_max": config.server.stream_retry_max,
                "retry_delay_s": config.server.stream_retry_delay_s,
            }
        except Exception:
            logger.debug("Failed to load stream retry config; using defaults")
            return {
                "checkpoint_enabled": True,
                "retry_max": 3,
                "retry_delay_s": 5.0,
            }

    # ── Context-window-aware tier downgrade ─────────────────

    _BYTES_PER_TOKEN_ESTIMATE = 4
    _TOKENS_PER_MCP_SCHEMA = 200
    _TOKENS_PER_TOOL_SCHEMA = 150
    _MIN_TOOL_OVERHEAD = 5000
    _MAX_TOOL_OVERHEAD = 20000

    def _estimate_tool_overhead(self) -> int:
        """Estimate tool definition overhead in tokens based on schema count."""
        registry = getattr(self, "_tool_registry", None) or []
        mode = getattr(self, "_execution_mode", "a")
        per_schema = self._TOKENS_PER_MCP_SCHEMA if mode in ("s", "c") else self._TOKENS_PER_TOOL_SCHEMA
        return min(max(len(registry) * per_schema, self._MIN_TOOL_OVERHEAD), self._MAX_TOOL_OVERHEAD)

    def _fit_prompt_to_context_window(
        self,
        system_prompt: str,
        prompt: str,
        context_window: int,
        *,
        priming_section: str,
        mode: str,
        trigger: str,
    ) -> str:
        """Ensure system prompt fits context window, shrinking budget if needed.

        Estimates total token consumption and rebuilds the system prompt
        with progressively smaller system_budget until it fits within 80%
        of the context window.

        Returns the (possibly rebuilt) system prompt.
        """
        from core.prompt.builder import _compute_system_budget

        tool_overhead = self._estimate_tool_overhead()
        sys_bytes = len(system_prompt.encode("utf-8"))
        prompt_bytes = len(prompt.encode("utf-8"))
        estimated_tokens = (sys_bytes + prompt_bytes) // self._BYTES_PER_TOKEN_ESTIMATE + tool_overhead
        max_input_tokens = int(context_window * 0.80)

        if estimated_tokens <= max_input_tokens:
            return system_prompt

        original_budget = _compute_system_budget(context_window)
        logger.warning(
            "Estimated prompt %d tokens exceeds context limit %d "
            "(budget=%d, context_window=%d); attempting budget shrink",
            estimated_tokens,
            max_input_tokens,
            original_budget,
            context_window,
        )

        best_prompt = system_prompt
        for shrink in (0.75, 0.50, 0.25):
            reduced_budget = int(original_budget * shrink)
            build_result = build_system_prompt(
                self.memory,
                tool_registry=self._tool_registry,
                personal_tools=self._personal_tools,
                priming_section="" if shrink <= 0.25 else priming_section,
                execution_mode=mode,
                message=prompt,
                retriever=self._get_retriever(),
                trigger=trigger,
                context_window=context_window,
                system_budget=reduced_budget,
            )
            best_prompt = build_result.system_prompt
            new_sys_bytes = len(best_prompt.encode("utf-8"))
            new_estimated = (
                new_sys_bytes + prompt_bytes
            ) // self._BYTES_PER_TOKEN_ESTIMATE + tool_overhead
            if new_estimated <= max_input_tokens:
                logger.warning(
                    "Prompt budget shrunk: %d -> %d chars (estimated %d -> %d tokens, limit %d)",
                    original_budget,
                    reduced_budget,
                    estimated_tokens,
                    new_estimated,
                    max_input_tokens,
                )
                return best_prompt

        max_sys_bytes = max(
            (max_input_tokens - tool_overhead) * self._BYTES_PER_TOKEN_ESTIMATE - prompt_bytes,
            2000,
        )
        if len(best_prompt.encode("utf-8")) > max_sys_bytes:
            logger.error(
                "Hard-truncating system prompt from %d to %d bytes to fit context window %d",
                len(best_prompt.encode("utf-8")),
                max_sys_bytes,
                context_window,
            )
            best_prompt = best_prompt.encode("utf-8")[:max_sys_bytes].decode(
                "utf-8",
                errors="ignore",
            )

        return best_prompt

    # ── Pre-flight prompt size check ─────────────────────────

    async def _preflight_size_check(
        self,
        system_prompt: str,
        prompt: str,
        conv_memory: ConversationMemory | None,
        *,
        priming_section: str,
        mode: str,
        message: str,
        trigger: str = "",
        context_window: int = 200_000,
    ) -> tuple[str, str, bool]:
        """Check combined prompt size and shrink if necessary.

        Returns (system_prompt, prompt, fell_back_to_fallback).
        """
        total = len(system_prompt.encode("utf-8")) + len(prompt.encode("utf-8"))
        logger.info(
            "Pre-flight prompt size: %d bytes (system=%d, user=%d)",
            total,
            len(system_prompt.encode("utf-8")),
            len(prompt.encode("utf-8")),
        )

        if total <= _PROMPT_SOFT_LIMIT_BYTES:
            return system_prompt, prompt, False

        # ── Stage 1: Force conversation compression ──────────
        logger.warning(
            "Prompt size %d exceeds soft limit %d; forcing conversation compression",
            total,
            _PROMPT_SOFT_LIMIT_BYTES,
        )
        if conv_memory is not None:
            try:
                await conv_memory._compress()
                prompt = conv_memory.build_chat_prompt(message, "human")
                system_prompt = build_system_prompt(
                    self.memory,
                    tool_registry=self._tool_registry,
                    personal_tools=self._personal_tools,
                    priming_section=priming_section,
                    execution_mode=mode,
                    message=prompt,
                    retriever=self._get_retriever(),
                    trigger=trigger,
                    context_window=context_window,
                ).system_prompt
            except Exception:
                logger.exception("Forced compression failed")

        total = len(system_prompt.encode("utf-8")) + len(prompt.encode("utf-8"))
        logger.info("Post-compression prompt size: %d bytes", total)

        if total <= _PROMPT_HARD_LIMIT_BYTES:
            return system_prompt, prompt, False

        # ── Stage 2: Fall back to Anthropic SDK (no JSON buffer limit) ──
        logger.warning(
            "Prompt size %d still exceeds hard limit %d; switching to S Fallback",
            total,
            _PROMPT_HARD_LIMIT_BYTES,
        )
        return system_prompt, prompt, True
