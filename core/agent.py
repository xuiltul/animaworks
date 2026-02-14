from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""AgentCore -- orchestrator for Digital Person execution cycles.

This module is intentionally slim: it owns only the execution mode routing,
session chaining orchestration, and lifecycle coordination.  Actual LLM
execution is delegated to engines in ``core.execution``, tool dispatch to
``core.tool_handler``, and schema management to ``core.tool_schemas``.
"""

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.prompt.context import ContextTracker
from core.memory import MemoryManager
from core.messenger import Messenger
from core.paths import load_prompt
from core.prompt.builder import build_system_prompt, inject_shortterm
from core.schemas import CycleResult, ModelConfig
from core.memory.shortterm import SessionState, ShortTermMemory
from core.tooling.handler import ToolHandler

logger = logging.getLogger("animaworks.agent")


class AgentCore:
    """Orchestrates execution of a Digital Person's thinking/acting cycle.

    Delegates actual LLM execution to an appropriate Executor:
      - A1 (autonomous, Claude):      ``AgentSDKExecutor``
      - A1 fallback (no Agent SDK):   ``AnthropicFallbackExecutor``
      - A2 (autonomous, non-Claude):   ``LiteLLMExecutor``
      - B  (assisted):                 ``AssistedExecutor``
    """

    def __init__(
        self,
        person_dir: Path,
        memory: MemoryManager,
        model_config: ModelConfig | None = None,
        messenger: Messenger | None = None,
    ) -> None:
        self.person_dir = person_dir
        self.memory = memory
        self.model_config = model_config or ModelConfig()
        self.messenger = messenger
        self._tool_registry = self._init_tool_registry()
        self._personal_tools = self._discover_personal_tools()
        self._sdk_available = self._check_sdk()
        self._agent_lock = asyncio.Lock()

        # Composable subsystems
        self._tool_handler = ToolHandler(
            person_dir=person_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
        )
        self._executor = self._create_executor()

        mode = self._resolve_execution_mode()
        logger.info(
            "AgentCore: model=%s, mode=%s, api_key=%s, base_url=%s",
            self.model_config.model,
            mode,
            "direct" if self.model_config.api_key else f"env:{self.model_config.api_key_env}",
            self.model_config.api_base_url or "(default)",
        )

    def set_on_message_sent(self, fn: Callable[[str, str, str], None]) -> None:
        """Inject a callback invoked after a send_message tool call."""
        self._tool_handler.on_message_sent = fn

    def reset_reply_tracking(self) -> None:
        """Clear reply tracking (call at start of each heartbeat cycle)."""
        self._tool_handler.reset_replied_to()

    @property
    def replied_to(self) -> set[str]:
        """Person names this agent has sent messages to in the current cycle."""
        return self._tool_handler.replied_to

    # ── Model / mode helpers ───────────────────────────────

    def _is_claude_model(self) -> bool:
        """True if the configured model is a Claude model usable with Agent SDK."""
        m = self.model_config.model
        return m.startswith("claude-") or m.startswith("anthropic/")

    def _resolve_execution_mode(self) -> str:
        """Determine the effective execution mode: ``a1``, ``a2``, or ``b``.

        Uses ``resolved_mode`` from config when available.
        Falls back to auto-detection for legacy config.md paths.
        """
        rm = self.model_config.resolved_mode
        if rm:
            mode = rm.lower()  # "A1" → "a1"
            if mode == "a1" and not self._sdk_available:
                return "a2"  # SDK unavailable fallback
            return mode

        # Fallback (resolved_mode absent = legacy config.md path)
        if self._is_claude_model() and self._sdk_available:
            return "a1"
        return "a2"

    @staticmethod
    def _check_sdk() -> bool:
        try:
            from claude_agent_sdk import query  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "claude-agent-sdk not available, falling back to anthropic SDK"
            )
            return False

    # Matches permission lines like "- image_gen: yes", "* web_search: OK"
    _PERMISSION_RE = re.compile(
        r"[-*]?\s*(\w+)\s*:\s*(OK|yes|enabled|true)\s*$",
        re.IGNORECASE,
    )

    def _init_tool_registry(self) -> list[str]:
        """Initialize tool registry with tools allowed in permissions.md.

        Parses the external tools section and accepts common affirmative
        values: ``OK``, ``yes``, ``enabled``, ``true`` (case-insensitive).
        """
        try:
            from core.tools import TOOL_MODULES
            permissions = self.memory.read_permissions() if self.memory else ""
            if "\u5916\u90e8\u30c4\u30fc\u30eb" not in permissions:
                return []
            allowed = []
            for line in permissions.splitlines():
                m = self._PERMISSION_RE.match(line.strip())
                if m and m.group(1) in TOOL_MODULES:
                    allowed.append(m.group(1))
            return allowed
        except Exception:
            logger.debug("Tool registry initialization skipped")
            return []

    def _discover_personal_tools(self) -> dict[str, str]:
        """Discover personal tool modules in ``{person_dir}/tools/``."""
        try:
            from core.tools import discover_personal_tools
            return discover_personal_tools(self.person_dir)
        except Exception:
            logger.debug("Personal tools discovery skipped", exc_info=True)
            return {}

    def _create_executor(self):
        """Factory: create the appropriate executor for the resolved mode.

        For mode A1 (Agent SDK), falls back gracefully:
          1. Try ``AgentSDKExecutor`` (requires ``claude_agent_sdk``)
          2. If ImportError, try ``AnthropicFallbackExecutor`` (requires ``anthropic``)
          3. If that also fails, fall back to ``LiteLLMExecutor`` with the
             ``anthropic/`` provider prefix
        """
        from core.execution import (
            AnthropicFallbackExecutor,
            AssistedExecutor,
            LiteLLMExecutor,
        )

        mode = self._resolve_execution_mode()

        if mode == "a1":
            # ── Try Agent SDK first ──────────────────────────
            try:
                from core.execution.agent_sdk import AgentSDKExecutor
                return AgentSDKExecutor(
                    model_config=self.model_config,
                    person_dir=self.person_dir,
                    tool_registry=self._tool_registry,
                    personal_tools=self._personal_tools,
                )
            except ImportError:
                logger.warning(
                    "AgentSDKExecutor unavailable (claude_agent_sdk not installed), "
                    "trying AnthropicFallbackExecutor"
                )

            # ── Try Anthropic SDK fallback ────────────────────
            try:
                import anthropic  # noqa: F401
                logger.info("Using AnthropicFallbackExecutor for Claude model")
                return AnthropicFallbackExecutor(
                    model_config=self.model_config,
                    person_dir=self.person_dir,
                    tool_handler=self._tool_handler,
                    tool_registry=self._tool_registry,
                    memory=self.memory,
                    personal_tools=self._personal_tools,
                )
            except ImportError:
                logger.warning(
                    "AnthropicFallbackExecutor also unavailable (anthropic not installed), "
                    "falling back to LiteLLM with anthropic provider"
                )

            # ── Last resort: LiteLLM with anthropic provider ─
            return LiteLLMExecutor(
                model_config=self.model_config,
                person_dir=self.person_dir,
                tool_handler=self._tool_handler,
                tool_registry=self._tool_registry,
                memory=self.memory,
                personal_tools=self._personal_tools,
            )

        if mode == "a2":
            return LiteLLMExecutor(
                model_config=self.model_config,
                person_dir=self.person_dir,
                tool_handler=self._tool_handler,
                tool_registry=self._tool_registry,
                memory=self.memory,
                personal_tools=self._personal_tools,
            )

        # mode == "b"
        return AssistedExecutor(
            model_config=self.model_config,
            person_dir=self.person_dir,
            memory=self.memory,
            messenger=self.messenger,
        )

    def _resolve_api_key(self) -> str | None:
        """Resolve the actual API key (direct value from config.json, then env var)."""
        import os
        if self.model_config.api_key:
            return self.model_config.api_key
        return os.environ.get(self.model_config.api_key_env)

    async def _run_priming(self, prompt: str, trigger: str) -> str:
        """Run priming layer to automatically retrieve relevant memories.

        Args:
            prompt: The user message (may include conversation history)
            trigger: Trigger type (e.g., "message:山田")

        Returns:
            Formatted priming section for system prompt injection, or empty string.
        """
        from core.memory.priming import PrimingEngine, format_priming_section

        # Extract sender name from trigger (format: "message:sender_name")
        sender_name = "human"
        if trigger.startswith("message:"):
            sender_name = trigger.split(":", 1)[1]

        # Extract the actual message content from prompt
        # The prompt may contain conversation history, so we need to extract the latest message
        message = self._extract_message_from_prompt(prompt)

        try:
            engine = PrimingEngine(self.person_dir)
            result = await engine.prime_memories(message, sender_name)

            if result.is_empty():
                logger.debug("Priming: No memories found")
                return ""

            formatted = format_priming_section(result, sender_name)
            logger.info(
                "Priming: Retrieved %d tokens of memories",
                result.estimated_tokens(),
            )
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
        in_content = False
        content_lines = []
        for line in reversed(lines):
            if line.startswith("**[") and "]" in line and ":**" in line:
                # Hit a history entry, stop
                break
            if line.strip():
                content_lines.insert(0, line)

        return "\n".join(content_lines) if content_lines else prompt

    # ── Public API ─────────────────────────────────────────

    async def run_cycle(
        self, prompt: str, trigger: str = "manual"
    ) -> CycleResult:
        """Run one agent cycle with autonomous memory search.

        Routing:
          - Mode B  (assisted):  ``AssistedExecutor``  -- 1-shot, no tools
          - Mode A2 (autonomous): ``LiteLLMExecutor`` -- LiteLLM + tool_use
          - Mode A1 (autonomous): ``AgentSDKExecutor`` -- Claude Agent SDK

        If the context threshold is crossed (A1 only), the session is
        externalized to short-term memory and automatically continued.
        """
        async with self._agent_lock:
            return await self._run_cycle_inner(prompt, trigger)

    async def _run_cycle_inner(
        self, prompt: str, trigger: str
    ) -> CycleResult:
        start = time.monotonic()
        mode = self._resolve_execution_mode()
        logger.info(
            "run_cycle START trigger=%s prompt_len=%d mode=%s",
            trigger, len(prompt), mode,
        )

        # ── Priming: Automatic memory retrieval ────────────────
        priming_section = await self._run_priming(prompt, trigger)

        # ── Mode B: assisted (1-shot, no tools) ──────────
        if mode == "b":
            result = await self._executor.execute(prompt=prompt, trigger=trigger)
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (assisted) trigger=%s duration_ms=%d",
                trigger, duration_ms,
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
            )

        shortterm = ShortTermMemory(self.person_dir)
        tracker = ContextTracker(
            model=self.model_config.model,
            threshold=self.model_config.context_threshold,
        )

        # Build system prompt with priming; inject short-term memory from prior session
        system_prompt = build_system_prompt(
            self.memory,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            priming_section=priming_section,
        )
        logger.debug("System prompt assembled, length=%d", len(system_prompt))
        if shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)
            logger.info("Injected short-term memory into system prompt")

        # ── Mode A2: LiteLLM tool_use loop ────────────────
        if mode == "a2":
            result = await self._executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                shortterm=shortterm,
            )
            shortterm.clear()
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (a2) trigger=%s duration_ms=%d response_len=%d",
                trigger, duration_ms, len(result.text),
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
            )

        # ── Mode A1: Claude Agent SDK ─────────────────────
        result = await self._executor.execute(
            prompt=prompt,
            system_prompt=system_prompt,
            tracker=tracker,
        )
        # Merge transcript-parsed replied_to for A1 mode
        if result.replied_to_from_transcript:
            self._tool_handler.merge_replied_to(result.replied_to_from_transcript)
            logger.info("Merged transcript replied_to: %s", result.replied_to_from_transcript)
        result_msg = result.result_message

        # Session chaining: if threshold was crossed, continue in a new session
        session_chained = False
        total_turns = result_msg.num_turns if result_msg else 0
        chain_count = 0
        accumulated_text = result.text

        while (
            tracker.threshold_exceeded
            and chain_count < self.model_config.max_chains
        ):
            session_chained = True
            chain_count += 1
            logger.info(
                "Session chain %d/%d: context at %.1f%%",
                chain_count,
                self.model_config.max_chains,
                tracker.usage_ratio * 100,
            )

            shortterm.clear()
            shortterm.save(
                SessionState(
                    session_id=result_msg.session_id if result_msg else "",
                    timestamp=datetime.now().isoformat(),
                    trigger=trigger,
                    original_prompt=prompt,
                    accumulated_response=accumulated_text,
                    context_usage_ratio=tracker.usage_ratio,
                    turn_count=result_msg.num_turns if result_msg else 0,
                )
            )

            tracker.reset()
            system_prompt_2 = inject_shortterm(
                build_system_prompt(
                    self.memory,
                    tool_registry=self._tool_registry,
                    personal_tools=self._personal_tools,
                    priming_section=priming_section,  # Reuse priming from initial session
                ),
                shortterm,
            )
            continuation_prompt = load_prompt("session_continuation")
            try:
                result_2 = await self._executor.execute(
                    prompt=continuation_prompt,
                    system_prompt=system_prompt_2,
                    tracker=tracker,
                )
                # Merge from chained session too
                if result_2.replied_to_from_transcript:
                    self._tool_handler.merge_replied_to(result_2.replied_to_from_transcript)
            except Exception:
                logger.exception(
                    "Chained session %d failed; preserving short-term memory",
                    chain_count,
                )
                break
            accumulated_text = accumulated_text + "\n" + result_2.text
            result_msg = result_2.result_message
            if result_msg:
                total_turns += result_msg.num_turns

        shortterm.clear()

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "run_cycle END trigger=%s duration_ms=%d response_len=%d chained=%s",
            trigger, duration_ms, len(accumulated_text), session_chained,
        )
        return CycleResult(
            trigger=trigger,
            action="responded",
            summary=accumulated_text,
            duration_ms=duration_ms,
            context_usage_ratio=tracker.usage_ratio,
            session_chained=session_chained,
            total_turns=total_turns,
        )

    # ── Streaming ──────────────────────────────────────────

    async def run_cycle_streaming(
        self, prompt: str, trigger: str = "manual"
    ) -> AsyncGenerator[dict, None]:
        """Streaming version of run_cycle.

        Yields stream chunks. Session chaining is handled seamlessly.
        Final event is ``{"type": "cycle_done", "cycle_result": {...}}``.
        """
        start = time.monotonic()
        mode = self._resolve_execution_mode()
        logger.info(
            "run_cycle_streaming START trigger=%s prompt_len=%d mode=%s",
            trigger, len(prompt), mode,
        )

        # Non-streaming executors: fall back to blocking execution
        if not self._executor.supports_streaming:
            async with self._agent_lock:
                cycle = await self._run_cycle_inner(prompt, trigger)
            yield {"type": "text_delta", "text": cycle.summary}
            yield {
                "type": "cycle_done",
                "cycle_result": cycle.model_dump(),
            }
            return

        # ── Streaming executor (A1 Agent SDK) ─────────────
        # Priming: Automatic memory retrieval
        priming_section = await self._run_priming(prompt, trigger)

        shortterm = ShortTermMemory(self.person_dir)
        tracker = ContextTracker(
            model=self.model_config.model,
            threshold=self.model_config.context_threshold,
        )

        system_prompt = build_system_prompt(
            self.memory,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            priming_section=priming_section,
        )
        if shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)

        # Primary session
        full_text_parts: list[str] = []
        result_message: Any = None

        async for chunk in self._executor.execute_streaming(
            system_prompt, prompt, tracker
        ):
            if chunk["type"] == "done":
                full_text_parts.append(chunk["full_text"])
                result_message = chunk["result_message"]
                # Merge transcript replied_to
                transcript_replied = chunk.get("replied_to_from_transcript", set())
                if transcript_replied:
                    self._tool_handler.merge_replied_to(transcript_replied)
            else:
                yield chunk

        # Session chaining
        session_chained = False
        total_turns = result_message.num_turns if result_message else 0
        chain_count = 0

        while (
            tracker.threshold_exceeded
            and chain_count < self.model_config.max_chains
        ):
            session_chained = True
            chain_count += 1
            logger.info(
                "Session chain (stream) %d/%d: context at %.1f%%",
                chain_count,
                self.model_config.max_chains,
                tracker.usage_ratio * 100,
            )

            yield {"type": "chain_start", "chain": chain_count}

            shortterm.clear()
            shortterm.save(
                SessionState(
                    session_id=result_message.session_id if result_message else "",
                    timestamp=datetime.now().isoformat(),
                    trigger=trigger,
                    original_prompt=prompt,
                    accumulated_response="\n".join(full_text_parts),
                    context_usage_ratio=tracker.usage_ratio,
                    turn_count=result_message.num_turns if result_message else 0,
                )
            )

            tracker.reset()
            system_prompt_2 = inject_shortterm(
                build_system_prompt(
                    self.memory,
                    tool_registry=self._tool_registry,
                    personal_tools=self._personal_tools,
                    priming_section=priming_section,  # Reuse priming from initial session
                ),
                shortterm,
            )
            continuation_prompt = load_prompt("session_continuation")

            try:
                async for chunk in self._executor.execute_streaming(
                    system_prompt_2, continuation_prompt, tracker
                ):
                    if chunk["type"] == "done":
                        full_text_parts.append(chunk["full_text"])
                        result_message = chunk["result_message"]
                        if result_message:
                            total_turns += result_message.num_turns
                        # Merge transcript replied_to
                        transcript_replied = chunk.get("replied_to_from_transcript", set())
                        if transcript_replied:
                            self._tool_handler.merge_replied_to(transcript_replied)
                    else:
                        yield chunk
            except Exception:
                logger.exception(
                    "Chained session (stream) %d failed", chain_count,
                )
                yield {"type": "error", "message": f"Session chain {chain_count} failed"}
                break

        shortterm.clear()

        full_text = "\n".join(full_text_parts)
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "run_cycle_streaming END trigger=%s duration_ms=%d response_len=%d chained=%s",
            trigger, duration_ms, len(full_text), session_chained,
        )

        yield {
            "type": "cycle_done",
            "cycle_result": CycleResult(
                trigger=trigger,
                action="responded",
                summary=full_text,
                duration_ms=duration_ms,
                context_usage_ratio=tracker.usage_ratio,
                session_chained=session_chained,
                total_turns=total_turns,
            ).model_dump(),
        }
