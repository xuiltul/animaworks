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
from core.prompt.builder import build_system_prompt
from core.prompt.tokens import estimate_tokens

logger = logging.getLogger("animaworks.agent")


class PrimingMixin:
    """Mixin: priming (auto-recall), context fitting, pre-flight size check."""

    async def _run_priming(
        self,
        prompt: str,
        trigger: str,
        *,
        message_intent: str = "",
        prompt_tier: str = "full",
        model_config=None,
    ) -> tuple[str, str]:
        """Run priming layer to automatically retrieve relevant memories.

        Args:
            prompt: The user message (may include conversation history)
            trigger: Trigger type (e.g., "message:yamada")
            prompt_tier: Prompt tier for budget control
                ("full"/"standard"/"light"/"minimal"/"micro").

        Returns:
            Tuple of (priming_section, pending_human_notifications).
        """
        from core.execution.session_types import (
            SESSION_TYPE_CHAT,
            SESSION_TYPE_CRON,
            SESSION_TYPE_HEARTBEAT,
            SESSION_TYPE_INBOX,
            SESSION_TYPE_TASK,
            resolve_runtime_session_type,
        )
        from core.memory.priming import PrimingEngine, format_priming_section
        from core.prompt.builder import TIER_LIGHT, TIER_MICRO, TIER_MINIMAL, TIER_STANDARD

        session_type = resolve_runtime_session_type(trigger)
        if session_type == SESSION_TYPE_HEARTBEAT:
            channel = "heartbeat"
        elif session_type == SESSION_TYPE_CRON:
            channel = "cron"
        elif session_type == SESSION_TYPE_INBOX:
            channel = "inbox"
        elif session_type == SESSION_TYPE_TASK:
            channel = "task"
        elif session_type == SESSION_TYPE_CHAT:
            channel = "chat"
        else:
            channel = session_type

        if channel == "heartbeat":
            message = self._get_recent_reflections_text()
        else:
            message = self._extract_message_from_prompt(prompt)

        sender_name = "human"
        if trigger.startswith("message:"):
            sender_name = trigger.split(":", 1)[1]
        elif trigger.startswith("inbox:"):
            senders = trigger.split(":", 1)[1]
            sender_name = senders.split(",")[0].strip() or "human"

        active_model_config = model_config or self.model_config
        recent_human_messages = self._get_recent_human_messages(trigger, model_config=active_model_config)

        try:
            from core.memory.priming.policy import resolve_priming_policy

            policy = resolve_priming_policy(self.anima_dir)
            if not hasattr(self, "_priming_engine"):
                from core.paths import get_shared_dir
                from core.prompt.context import resolve_context_window as _rcw_priming

                ctx_window = _rcw_priming(
                    active_model_config.model,
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

            result = await self._priming_engine.prime_memories(
                message,
                sender_name,
                channel=channel,
                intent=message_intent,
                enable_dynamic_budget=policy.dynamic_budget,
                recent_human_messages=recent_human_messages,
                profile="compact" if prompt_tier in (TIER_MINIMAL, TIER_MICRO, TIER_LIGHT) else policy.profile,
                max_tokens=min(policy.max_tokens, 1000) if prompt_tier == TIER_STANDARD else policy.max_tokens,
                include_related=prompt_tier not in (TIER_MINIMAL, TIER_MICRO, TIER_LIGHT),
            )

            pending_notifications = result.pending_human_notifications

            if result.is_empty():
                logger.debug("Priming: No memories found")
                return ("", pending_notifications)

            formatted = format_priming_section(result, sender_name)
            logger.info(
                "Priming: Retrieved %d tokens of memories (tier=%s)",
                result.estimated_tokens(),
                prompt_tier,
            )

            return (formatted, pending_notifications)

        except Exception:
            logger.exception("Priming failed; continuing without primed memories")
            return ("", "")

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

    def _get_recent_human_messages(self, trigger: str, *, model_config=None) -> list[str]:
        """Get recent human messages from the trigger's own activity source.

        Returns newest-first list of human message contents.
        Chat behavior remains conversation-backed; inbox reads only the activity
        log so unrelated chat history cannot leak into a background run.
        """
        from core.execution.session_types import trigger_uses_chat_session

        if trigger.startswith("inbox:"):
            try:
                from core.memory.activity import ActivityLogger

                animas_dir = self.anima_dir.parent
                anima_names = {path.name.casefold() for path in animas_dir.iterdir() if path.is_dir()}
                entries = ActivityLogger(self.anima_dir).recent(
                    days=2,
                    types=["message_received"],
                    limit=100,
                )
                messages: list[str] = []
                for entry in reversed(entries):
                    sender = str(entry.from_person or "").strip()
                    if not sender or sender.casefold() == "system" or sender.casefold() in anima_names:
                        continue
                    content = str(entry.content or entry.summary or "").strip()
                    if content:
                        messages.append(content[:200])
                    if len(messages) == 3:
                        break
                return messages
            except Exception:
                logger.debug("Failed to load recent inbox human messages for priming", exc_info=True)
                return []

        if not trigger.startswith("message:") or not trigger_uses_chat_session(trigger):
            return []
        try:
            from core.memory.conversation import ConversationMemory

            conv = ConversationMemory(self.anima_dir, model_config or self.model_config)
            state = conv.load()
            human_turns = [t for t in state.turns if t.role == "human"]
            recent = human_turns[-5:]
            recent.reverse()
            return [t.content for t in recent]
        except Exception:
            logger.debug("Failed to load recent human messages for priming")
            return []

    def _get_recent_reflections_text(self) -> str:
        """Get recent heartbeat REFLECTION text for priming query.

        Retrieves last 3 heartbeat_reflection entries from activity log,
        providing high-density situation/insight text instead of the
        full heartbeat prompt template.
        """
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(self.anima_dir)
            entries = activity.recent(
                days=3,
                types=["heartbeat_reflection"],
                limit=3,
            )
            if not entries:
                return ""
            parts = []
            for e in entries:
                content = e.content or e.summary
                if content:
                    parts.append(content[:500])
            return "\n".join(parts)
        except Exception:
            logger.debug("Failed to load reflections for heartbeat priming")
            return ""

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

    _TOKENS_PER_MCP_SCHEMA = 200
    _TOKENS_PER_TOOL_SCHEMA = 150
    _MIN_TOOL_OVERHEAD = 5000
    _MAX_TOOL_OVERHEAD = 20000

    def _estimate_tool_overhead(self, mode: str | None = None) -> int:
        """Estimate tool definition overhead in tokens based on schema count."""
        registry = getattr(self, "_tool_registry", None) or []
        mode = mode or getattr(self, "_execution_mode", "a")
        per_schema = self._TOKENS_PER_MCP_SCHEMA if mode in ("s", "c", "x") else self._TOKENS_PER_TOOL_SCHEMA
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
        pending_human_notifications: str = "",
        thread_id: str = "default",
        shortterm_text: str = "",
    ) -> str:
        """Ensure system prompt fits context window, shrinking budget if needed.

        Estimates total token consumption and rebuilds the system prompt
        with progressively smaller system_budget until it fits within 80%
        of the context window.

        Returns the (possibly rebuilt) system prompt. If mandatory context
        still cannot fit, raise ExecutionError instead of stripping authority
        boundaries, human decisions, or durable task context.
        """
        from core.prompt.builder import _compute_system_budget

        tool_overhead = self._estimate_tool_overhead(mode)
        prompt_tokens = estimate_tokens(prompt)
        estimated_tokens = estimate_tokens(system_prompt) + prompt_tokens + tool_overhead
        max_input_tokens = int(context_window * 0.80)

        if estimated_tokens <= max_input_tokens:
            return system_prompt

        original_budget = _compute_system_budget(context_window)
        logger.warning(
            "Estimated prompt %d tokens exceeds context limit %d "
            "(target=%d, ceiling=%d, context_window=%d); attempting budget shrink",
            estimated_tokens,
            max_input_tokens,
            original_budget.target,
            original_budget.ceiling,
            context_window,
        )

        best_prompt = system_prompt
        for shrink in (0.75, 0.50, 0.25):
            reduced_budget = int(original_budget.target * shrink)
            build_result = build_system_prompt(
                self.memory,
                tool_registry=self._tool_registry,
                personal_tools=self._personal_tools,
                priming_section=priming_section,
                execution_mode=mode,
                message=prompt,
                retriever=self._get_retriever(),
                trigger=trigger,
                context_window=context_window,
                system_budget=reduced_budget,
                pending_human_notifications=pending_human_notifications,
                thread_id=thread_id,
                shortterm_text=shortterm_text,
            )
            best_prompt = build_result.system_prompt
            new_estimated = estimate_tokens(best_prompt) + prompt_tokens + tool_overhead
            if new_estimated <= max_input_tokens:
                logger.warning(
                    "Prompt budget shrunk: %d -> %d tokens (estimated %d -> %d tokens, limit %d)",
                    original_budget.target,
                    reduced_budget,
                    estimated_tokens,
                    new_estimated,
                    max_input_tokens,
                )
                return best_prompt

        from core.exceptions import ExecutionError
        from core.i18n import t

        logger.error(
            "Mandatory prompt context cannot safely fit: estimated=%d limit=%d context_window=%d",
            new_estimated,
            max_input_tokens,
            context_window,
        )
        raise ExecutionError(
            t(
                "agent.context_cannot_fit_safely",
                estimated=new_estimated,
                limit=max_input_tokens,
            )
        )

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
        pending_human_notifications: str = "",
        thread_id: str = "default",
        shortterm_text: str = "",
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

        # ── Stage 1: Force conversation compression for chat only ──────────
        if conv_memory is not None:
            logger.warning(
                "Prompt size %d exceeds soft limit %d; forcing conversation compression",
                total,
                _PROMPT_SOFT_LIMIT_BYTES,
            )
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
                    pending_human_notifications=pending_human_notifications,
                    thread_id=thread_id,
                    shortterm_text=shortterm_text,
                ).system_prompt
            except Exception:
                logger.exception("Forced compression failed")
        else:
            logger.warning(
                "Prompt size %d exceeds soft limit %d; skipping conversation compression for non-chat trigger=%s",
                total,
                _PROMPT_SOFT_LIMIT_BYTES,
                trigger,
            )

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
