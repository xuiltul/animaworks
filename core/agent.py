from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""AgentCore -- orchestrator for Digital Anima execution cycles.

This module is intentionally slim: it owns only the execution mode routing,
session chaining orchestration, and lifecycle coordination.  Actual LLM
execution is delegated to engines in ``core.execution``, tool dispatch to
``core.tool_handler``, and schema management to ``core.tool_schemas``.
"""

import asyncio
import json as _json
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from core.time_utils import now_iso

from core.background import BackgroundTaskManager
from core.prompt.context import ContextTracker
from core.memory import MemoryManager
from core.messenger import Messenger
from core.paths import load_prompt
from core.prompt.builder import BuildResult, build_system_prompt, inject_shortterm
from core.exceptions import AnimaWorksError  # noqa: F401
from core.schemas import CycleResult, ModelConfig
from core.memory.shortterm import SessionState, ShortTermMemory
from core.tooling.handler import ToolHandler

logger = logging.getLogger("animaworks.agent")

# ── Prompt size guards ──────────────────────────────────────────
# Agent SDK uses JSON-RPC with a default 1 MB buffer (now raised to 4 MB via
# max_buffer_size).  These thresholds trigger defensive actions well before
# the hard limit is hit.  JSON framing + tool schemas add ~30-50% overhead
# on top of the raw text, so we use conservative byte limits.
_PROMPT_SOFT_LIMIT_BYTES = 600_000   # Force compression
_PROMPT_HARD_LIMIT_BYTES = 1_200_000  # Fall back to A1 Fallback


def _save_prompt_log(
    anima_dir: Path,
    *,
    trigger: str,
    sender: str,
    model: str,
    mode: str,
    system_prompt: str,
    user_message: str,
    tools: list[str],
    session_id: str,
) -> None:
    """Persist the full prompt payload to a JSONL log for post-hoc debugging.

    Writes to ``{anima_dir}/prompt_logs/{date}.jsonl``.
    Failures are silently logged -- prompt logging must never break execution.
    """
    try:
        log_dir = anima_dir / "prompt_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = now_iso()[:10]  # YYYY-MM-DD
        entry = {
            "ts": now_iso(),
            "trigger": trigger,
            "from": sender,
            "model": model,
            "mode": mode,
            "system_prompt_length": len(system_prompt),
            "system_prompt": system_prompt,
            "user_message": user_message,
            "tools": tools,
            "session_id": session_id,
        }
        log_file = log_dir / f"{today}.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("Prompt log saved: %s (%d bytes)", log_file, len(system_prompt))
    except Exception:
        logger.warning("Failed to save prompt log", exc_info=True)


class AgentCore:
    """Orchestrates execution of a Digital Anima's thinking/acting cycle.

    Delegates actual LLM execution to an appropriate Executor:
      - A1 (autonomous, Claude):      ``AgentSDKExecutor``
      - A1 fallback (no Agent SDK):   ``AnthropicFallbackExecutor``
      - A2 (autonomous, non-Claude):   ``LiteLLMExecutor``
      - B  (assisted):                 ``AssistedExecutor``
    """

    def __init__(
        self,
        anima_dir: Path,
        memory: MemoryManager,
        model_config: ModelConfig | None = None,
        messenger: Messenger | None = None,
    ) -> None:
        self.anima_dir = anima_dir
        self.memory = memory
        self.model_config = model_config or ModelConfig()
        self.messenger = messenger
        self._tool_registry = self._init_tool_registry()
        self._personal_tools = self._discover_personal_tools()
        self._sdk_available = self._check_sdk()
        self._agent_lock = asyncio.Lock()

        # Build human notifier for top-level animas
        human_notifier = self._build_human_notifier()

        # Background task manager
        self._background_manager = self._build_background_manager()

        # Composable subsystems
        self._tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            human_notifier=human_notifier,
            background_manager=self._background_manager,
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

    def set_on_schedule_changed(self, fn: Callable[[str], Any] | None) -> None:
        """Inject a callback invoked when heartbeat.md or cron.md is modified."""
        self._tool_handler.on_schedule_changed = fn

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notification events from ToolHandler."""
        return self._tool_handler.drain_notifications()

    def reset_reply_tracking(self) -> None:
        """Clear reply tracking (call at start of each heartbeat cycle)."""
        self._tool_handler.reset_replied_to()

    @property
    def replied_to(self) -> set[str]:
        """Anima names this agent has sent messages to in the current cycle."""
        return self._tool_handler.replied_to

    @staticmethod
    def _extract_sender(prompt: str, trigger: str) -> str:
        """Extract sender name from trigger string."""
        if trigger.startswith("message:"):
            return trigger.split(":", 1)[1]
        return trigger  # heartbeat, cron, manual, etc.

    # ── Human notification ─────────────────────────────────

    def _build_human_notifier(self) -> "HumanNotifier | None":
        """Build HumanNotifier if this anima is top-level and notification is enabled."""
        try:
            from core.config import load_config
            from core.notification.notifier import HumanNotifier

            config = load_config()
            if not config.human_notification.enabled:
                return None

            # Only top-level animas (no supervisor) get the notifier
            anima_cfg = config.animas.get(self.anima_dir.name)
            if anima_cfg is not None and anima_cfg.supervisor is not None:
                return None

            notifier = HumanNotifier.from_config(config.human_notification)
            if notifier.channel_count == 0:
                return None

            logger.info(
                "HumanNotifier enabled for %s with %d channel(s)",
                self.anima_dir.name,
                notifier.channel_count,
            )
            return notifier
        except Exception:
            logger.debug("Failed to build HumanNotifier", exc_info=True)
            return None

    @property
    def has_human_notifier(self) -> bool:
        """True if this agent has a configured human notifier."""
        return self._tool_handler._human_notifier is not None

    @property
    def human_notifier(self):
        """Return the human notifier instance (or None)."""
        return self._tool_handler._human_notifier

    # ── Background task management ────────────────────────────

    def _build_background_manager(self) -> BackgroundTaskManager | None:
        """Build BackgroundTaskManager if enabled in config."""
        try:
            from core.config import load_config
            config = load_config()
            if not config.background_task.enabled:
                return None

            from core.tools import TOOL_MODULES
            from core.tools._base import load_execution_profiles

            profiles = load_execution_profiles(TOOL_MODULES)
            config_eligible = {
                name: tc.threshold_s
                for name, tc in config.background_task.eligible_tools.items()
            }

            return BackgroundTaskManager.from_profiles(
                anima_dir=self.anima_dir,
                anima_name=self.anima_dir.name,
                profiles=profiles,
                config_eligible=config_eligible or None,
            )
        except Exception:
            logger.debug("BackgroundTaskManager init skipped", exc_info=True)
            return None

    @property
    def background_manager(self) -> BackgroundTaskManager | None:
        """Return the background task manager (if enabled)."""
        return self._background_manager

    # ── Model / mode helpers ───────────────────────────────

    def _is_claude_model(self) -> bool:
        """True if the configured model is a Claude model usable with Agent SDK."""
        m = self.model_config.model
        return m.startswith("claude-") or m.startswith("anthropic/")

    @property
    def execution_mode(self) -> str:
        """Public access to the resolved execution mode."""
        return self._resolve_execution_mode()

    def _resolve_execution_mode(self) -> str:
        """Determine the effective execution mode: ``a1``, ``a2``, or ``b``.

        Uses ``resolved_mode`` from config when available.
        Falls back to auto-detection for legacy config.md paths.
        """
        rm = self.model_config.resolved_mode
        if rm:
            mode = rm.lower()  # "A1" → "a1"
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
    # Matches deny entries like "- chatwork: no", "* gmail: disabled"
    _DENY_RE = re.compile(
        r"[-*]?\s*(\w+)\s*:\s*(no|deny|disabled|false)\s*$",
        re.IGNORECASE,
    )

    def _init_tool_registry(self) -> list[str]:
        """Initialize tool registry from permissions.md (default-all).

        Strategy:
          1. No ``外部ツール`` section present -> ALL tools (default-all)
          2. ``- all: yes`` found -> ALL tools minus any ``- tool: no`` deny entries
          3. Individual ``- tool: yes`` entries -> whitelist mode (backward compat)
          4. Section present but no allow/deny entries match -> ALL tools
        """
        try:
            from core.tools import TOOL_MODULES
            all_tools = sorted(TOOL_MODULES.keys())
            permissions = self.memory.read_permissions() if self.memory else ""

            # No 外部ツール section → default-all
            if "外部ツール" not in permissions:
                return all_tools

            # Parse allow and deny entries
            has_all_yes = False
            allowed: list[str] = []
            denied: list[str] = []
            for line in permissions.splitlines():
                stripped = line.strip()
                # Check allow entries
                m_allow = self._PERMISSION_RE.match(stripped)
                if m_allow:
                    name = m_allow.group(1)
                    if name == "all":
                        has_all_yes = True
                    elif name in TOOL_MODULES:
                        allowed.append(name)
                    continue
                # Check deny entries
                m_deny = self._DENY_RE.match(stripped)
                if m_deny:
                    name = m_deny.group(1)
                    if name in TOOL_MODULES:
                        denied.append(name)

            # "- all: yes" → all tools minus denied
            if has_all_yes:
                return [t for t in all_tools if t not in denied]

            # Individual allow entries → whitelist (backward compat)
            if allowed:
                return allowed

            # Section present but no matching entries → default-all
            return all_tools
        except Exception:
            logger.debug("Tool registry initialization skipped")
            return []

    def _discover_personal_tools(self) -> dict[str, str]:
        """Discover common and personal tool modules."""
        try:
            from core.tools import discover_personal_tools, discover_common_tools
            common = discover_common_tools()
            personal = discover_personal_tools(self.anima_dir)
            # Personal overrides common (higher priority)
            return {**common, **personal}
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
                    anima_dir=self.anima_dir,
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
                    anima_dir=self.anima_dir,
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
                anima_dir=self.anima_dir,
                tool_handler=self._tool_handler,
                tool_registry=self._tool_registry,
                memory=self.memory,
                personal_tools=self._personal_tools,
            )

        if mode == "a2":
            return LiteLLMExecutor(
                model_config=self.model_config,
                anima_dir=self.anima_dir,
                tool_handler=self._tool_handler,
                tool_registry=self._tool_registry,
                memory=self.memory,
                personal_tools=self._personal_tools,
            )

        # mode == "b"
        return AssistedExecutor(
            model_config=self.model_config,
            anima_dir=self.anima_dir,
            tool_handler=self._tool_handler,
            memory=self.memory,
            messenger=self.messenger,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
        )

    def _resolve_api_key(self) -> str | None:
        """Resolve the actual API key (direct value from config.json, then env var)."""
        import os
        if self.model_config.api_key:
            return self.model_config.api_key
        return os.environ.get(self.model_config.api_key_env)

    def _get_retriever(self) -> object | None:
        """Return the RAG retriever from priming engine, if available.

        Used by build_system_prompt for Tier 3 vector-based skill matching.
        Returns None if priming engine has not been initialized yet.
        """
        engine = getattr(self, "_priming_engine", None)
        if engine is None:
            return None
        return getattr(engine, "_retriever", None)

    async def _run_priming(self, prompt: str, trigger: str, *, message_intent: str = "") -> str:
        """Run priming layer to automatically retrieve relevant memories.

        Args:
            prompt: The user message (may include conversation history)
            trigger: Trigger type (e.g., "message:yamada")

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
            if not hasattr(self, "_priming_engine"):
                from core.paths import get_shared_dir
                self._priming_engine = PrimingEngine(self.anima_dir, get_shared_dir())
            channel = (
                "heartbeat" if trigger == "heartbeat"
                else "cron" if trigger.startswith("cron")
                else "chat"
            )
            result = await self._priming_engine.prime_memories(
                message,
                sender_name,
                channel=channel,
                intent=message_intent,
            )

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

    # ── Pre-flight prompt size check ─────────────────────────

    async def _preflight_size_check(
        self,
        system_prompt: str,
        prompt: str,
        conv_memory: "ConversationMemory | None",
        *,
        priming_section: str,
        mode: str,
        message: str,
        trigger: str = "",
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
            total, _PROMPT_SOFT_LIMIT_BYTES,
        )
        if conv_memory is not None:
            try:
                await conv_memory._compress()
                # Rebuild prompt with compressed history
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
                ).system_prompt
            except Exception:
                logger.exception("Forced compression failed")

        total = len(system_prompt.encode("utf-8")) + len(prompt.encode("utf-8"))
        logger.info("Post-compression prompt size: %d bytes", total)

        if total <= _PROMPT_HARD_LIMIT_BYTES:
            return system_prompt, prompt, False

        # ── Stage 2: Fall back to Anthropic SDK (no JSON buffer limit) ──
        logger.warning(
            "Prompt size %d still exceeds hard limit %d; switching to A1 Fallback",
            total, _PROMPT_HARD_LIMIT_BYTES,
        )
        return system_prompt, prompt, True

    def _create_fallback_executor(self):
        """Create an AnthropicFallbackExecutor for when A1 SDK can't handle the prompt."""
        from core.execution import AnthropicFallbackExecutor
        logger.info("Creating AnthropicFallbackExecutor for oversized prompt")
        return AnthropicFallbackExecutor(
            model_config=self.model_config,
            anima_dir=self.anima_dir,
            tool_handler=self._tool_handler,
            tool_registry=self._tool_registry,
            memory=self.memory,
            personal_tools=self._personal_tools,
        )

    # ── Public API ─────────────────────────────────────────

    async def run_cycle(
        self,
        prompt: str,
        trigger: str = "manual",
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
    ) -> CycleResult:
        """Run one agent cycle with autonomous memory search.

        Routing:
          - Mode B  (assisted):  ``AssistedExecutor``  -- text-based tool loop
          - Mode A2 (autonomous): ``LiteLLMExecutor`` -- LiteLLM + tool_use
          - Mode A1 (autonomous): ``AgentSDKExecutor`` -- Claude Agent SDK

        If the context threshold is crossed (A1 only), the session is
        externalized to short-term memory and automatically continued.
        """
        async with self._agent_lock:
            return await self._run_cycle_inner(
                prompt,
                trigger,
                images=images,
                prior_messages=prior_messages,
                message_intent=message_intent,
            )

    async def _run_cycle_inner(
        self,
        prompt: str,
        trigger: str,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
    ) -> CycleResult:
        start = time.monotonic()
        mode = self._resolve_execution_mode()
        logger.info(
            "run_cycle START trigger=%s prompt_len=%d mode=%s",
            trigger, len(prompt), mode,
        )

        # ── Priming: Automatic memory retrieval ────────────────
        priming_section = await self._run_priming(
            prompt,
            trigger,
            message_intent=message_intent,
        )

        shortterm = ShortTermMemory(self.anima_dir)
        tracker = ContextTracker(
            model=self.model_config.model,
            threshold=self.model_config.context_threshold,
            context_window_overrides=self._load_context_window_overrides(),
        )

        # Build system prompt with priming; inject short-term memory from prior session
        build_result = build_system_prompt(
            self.memory,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            priming_section=priming_section,
            execution_mode=mode,
            message=prompt,
            retriever=self._get_retriever(),
            trigger=trigger,
        )
        system_prompt = build_result.system_prompt
        injected_procedures = build_result.injected_procedures
        logger.debug("System prompt assembled, length=%d", len(system_prompt))

        # Persist injected procedures for heartbeat-triggered finalization
        if injected_procedures:
            from core.memory.conversation import ConversationMemory as _CM
            _cm = _CM(self.anima_dir, self.model_config)
            _cm.store_injected_procedures(
                injected_procedures,
                session_id=self._tool_handler.session_id,
            )
        if shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)
            logger.info("Injected short-term memory into system prompt")

        # ── Prompt log: save full payload for debugging ───
        _save_prompt_log(
            self.anima_dir,
            trigger=trigger,
            sender=self._extract_sender(prompt, trigger),
            model=self.model_config.model,
            mode=mode,
            system_prompt=system_prompt,
            user_message=prompt,
            tools=self._tool_registry,
            session_id=self._tool_handler.session_id,
        )

        # ── Helper: convert ExecutionResult tool records to dicts ──
        def _tool_records_to_dicts(result: "ExecutionResult") -> list[dict]:
            from dataclasses import asdict as _asdict
            return [_asdict(r) for r in result.tool_call_records]

        # ── Mode B: text-based tool-call loop ─────────────
        if mode == "b":
            result = await self._executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                trigger=trigger,
                images=images,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "run_cycle END (mode-b) trigger=%s duration_ms=%d response_len=%d",
                trigger, duration_ms, len(result.text),
            )
            return CycleResult(
                trigger=trigger,
                action="responded",
                summary=result.text,
                duration_ms=duration_ms,
                tool_call_records=_tool_records_to_dicts(result),
            )

        # ── Mode A2: LiteLLM tool_use loop ────────────────
        if mode == "a2":
            result = await self._executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                shortterm=shortterm,
                images=images,
                prior_messages=prior_messages,
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
                tool_call_records=_tool_records_to_dicts(result),
            )

        # ── Mode A1: Claude Agent SDK ─────────────────────
        # Pre-flight: check prompt size to prevent Agent SDK buffer overflow
        from core.memory.conversation import ConversationMemory
        conv_memory = ConversationMemory(self.anima_dir, self.model_config)
        system_prompt, prompt, use_fallback = await self._preflight_size_check(
            system_prompt, prompt, conv_memory,
            priming_section=priming_section,
            mode=mode,
            message=prompt,
            trigger=trigger,
        )
        if use_fallback:
            executor = self._create_fallback_executor()
            result = await executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                images=images,
                prior_messages=prior_messages,
            )
        else:
            result = await self._executor.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                images=images,
            )
        # Merge transcript-parsed replied_to for A1 mode
        if result.replied_to_from_transcript:
            self._tool_handler.merge_replied_to(result.replied_to_from_transcript)
            logger.info("Merged transcript replied_to: %s", result.replied_to_from_transcript)
        result_msg = result.result_message
        accumulated_tool_records = _tool_records_to_dicts(result)

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
                    timestamp=now_iso(),
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
                    execution_mode=mode,
                    message=prompt,
                    retriever=self._get_retriever(),
                    trigger=trigger,
                ).system_prompt,
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
            accumulated_tool_records.extend(_tool_records_to_dicts(result_2))
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
            tool_call_records=accumulated_tool_records,
        )

    # ── Streaming ──────────────────────────────────────────

    async def run_cycle_streaming(
        self,
        prompt: str,
        trigger: str = "manual",
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        message_intent: str = "",
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
                cycle = await self._run_cycle_inner(
                    prompt,
                    trigger,
                    images=images,
                    prior_messages=prior_messages,
                    message_intent=message_intent,
                )
            yield {"type": "text_delta", "text": cycle.summary}
            yield {
                "type": "cycle_done",
                "cycle_result": cycle.model_dump(mode="json"),
            }
            return

        # ── Streaming executor (A1 / A2 / all modes) ─────────────
        # Priming: Automatic memory retrieval
        priming_section = await self._run_priming(
            prompt,
            trigger,
            message_intent=message_intent,
        )

        shortterm = ShortTermMemory(self.anima_dir)
        tracker = ContextTracker(
            model=self.model_config.model,
            threshold=self.model_config.context_threshold,
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
        )
        system_prompt = build_result.system_prompt
        # Persist injected procedures for heartbeat-triggered finalization
        if build_result.injected_procedures:
            from core.memory.conversation import ConversationMemory as _CM
            _cm = _CM(self.anima_dir, self.model_config)
            _cm.store_injected_procedures(
                build_result.injected_procedures,
                session_id=self._tool_handler.session_id,
            )
        if shortterm.has_pending():
            system_prompt = inject_shortterm(system_prompt, shortterm)

        # Pre-flight size check for streaming path
        from core.memory.conversation import ConversationMemory
        conv_memory = ConversationMemory(self.anima_dir, self.model_config)
        system_prompt, prompt, use_fallback = await self._preflight_size_check(
            system_prompt, prompt, conv_memory,
            priming_section=priming_section,
            mode=mode,
            message=prompt,
            trigger=trigger,
        )
        if use_fallback:
            # Fall back to non-streaming execution
            logger.warning("Streaming fallback: using blocking A1 Fallback for oversized prompt")
            async with self._agent_lock:
                cycle = await self._run_cycle_inner(
                    prompt,
                    trigger,
                    message_intent=message_intent,
                    images=images,
                )
            yield {"type": "text_delta", "text": cycle.summary}
            yield {
                "type": "cycle_done",
                "cycle_result": cycle.model_dump(mode="json"),
            }
            return

        # ── Prompt log: save full payload for debugging ───
        _save_prompt_log(
            self.anima_dir,
            trigger=trigger,
            sender=self._extract_sender(prompt, trigger),
            model=self.model_config.model,
            mode=mode,
            system_prompt=system_prompt,
            user_message=prompt,
            tools=self._tool_registry,
            session_id=self._tool_handler.session_id,
        )

        # ── Stream retry configuration ────────────────────
        retry_cfg = self._load_stream_retry_config()
        checkpoint_enabled = retry_cfg["checkpoint_enabled"]
        max_retries = retry_cfg["retry_max"]
        retry_delay = retry_cfg["retry_delay_s"]

        # Primary session with checkpoint + retry support
        full_text_parts: list[str] = []
        all_tool_call_records: list[dict] = []
        result_message: Any = None
        current_prompt = prompt
        current_system_prompt = system_prompt
        retry_count = 0

        while True:
            completed_tools: list[dict[str, Any]] = []
            text_parts_this_attempt: list[str] = []
            stream_succeeded = False

            try:
                async for chunk in self._executor.execute_streaming(
                    current_system_prompt, current_prompt, tracker,
                    images=images,
                    prior_messages=prior_messages,
                ):
                    if chunk["type"] == "done":
                        full_text_parts.append(chunk["full_text"])
                        text_parts_this_attempt.append(chunk["full_text"])
                        result_message = chunk["result_message"]
                        # Accumulate tool call records from executor
                        all_tool_call_records.extend(
                            chunk.get("tool_call_records", [])
                        )
                        # Merge transcript replied_to
                        transcript_replied = chunk.get("replied_to_from_transcript", set())
                        if transcript_replied:
                            self._tool_handler.merge_replied_to(transcript_replied)
                        stream_succeeded = True
                    elif chunk["type"] == "tool_end" and checkpoint_enabled:
                        completed_tools.append({
                            "tool_name": chunk.get("tool_name", ""),
                            "tool_id": chunk.get("tool_id", ""),
                            "summary": chunk.get("tool_name", ""),
                        })
                        # Save checkpoint after each tool completion
                        from core.memory.shortterm import StreamCheckpoint
                        shortterm.save_checkpoint(StreamCheckpoint(
                            timestamp=now_iso(),
                            trigger=trigger,
                            original_prompt=prompt,
                            completed_tools=completed_tools,
                            accumulated_text="\n".join(full_text_parts),
                            retry_count=retry_count,
                        ))
                        yield chunk
                    else:
                        if chunk["type"] == "text_delta":
                            text_parts_this_attempt.append(chunk.get("text", ""))
                        yield chunk

            except Exception as e:
                from core.execution.base import StreamDisconnectedError

                is_stream_error = isinstance(e, StreamDisconnectedError)
                if not is_stream_error:
                    # Non-stream errors: log and break
                    logger.exception("Agent SDK streaming error (non-retryable)")
                    yield {"type": "error", "message": f"[Agent SDK Error: {e}]"}
                    break

                # ── Stream disconnect: attempt retry ──────────
                partial_text = e.partial_text if is_stream_error else ""
                if partial_text:
                    full_text_parts.append(partial_text)

                if retry_count >= max_retries:
                    logger.error(
                        "Stream retry exhausted (%d/%d)",
                        retry_count, max_retries,
                    )
                    yield {
                        "type": "error",
                        "message": f"ストリームが{retry_count}回切断されました。最大リトライ回数に達しました。",
                    }
                    break

                retry_count += 1
                logger.warning(
                    "Stream disconnected, retrying %d/%d after %.1fs",
                    retry_count, max_retries, retry_delay,
                )
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
                ).system_prompt

                await asyncio.sleep(retry_delay)
                continue

            if stream_succeeded:
                # Clear checkpoint on success
                shortterm.clear_checkpoint()
                break

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
                    timestamp=now_iso(),
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
                    execution_mode=mode,
                    message=prompt,
                    retriever=self._get_retriever(),
                    trigger=trigger,
                ).system_prompt,
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
                        all_tool_call_records.extend(
                            chunk.get("tool_call_records", [])
                        )
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
            "run_cycle_streaming END trigger=%s duration_ms=%d response_len=%d chained=%s retries=%d",
            trigger, duration_ms, len(full_text), session_chained, retry_count,
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
                tool_call_records=all_tool_call_records,
            ).model_dump(mode="json"),
        }
