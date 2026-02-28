from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""ExecutorFactoryMixin -- tool registry, executor factory, API key helpers.

Extracted from ``core.agent.AgentCore`` as a Mixin.  All ``self`` references
are resolved at runtime via MRO when mixed into ``AgentCore``.
"""

import logging
import re

logger = logging.getLogger("animaworks.agent")


class ExecutorFactoryMixin:
    """Mixin: tool registry initialisation, executor creation, API key resolution."""

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

        For mode S (Agent SDK), falls back gracefully:
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

        if mode == "s":
            # ── Try Agent SDK first ──────────────────────────
            try:
                from core.execution.agent_sdk import AgentSDKExecutor
                return AgentSDKExecutor(
                    model_config=self.model_config,
                    anima_dir=self.anima_dir,
                    tool_registry=self._tool_registry,
                    personal_tools=self._personal_tools,
                    interrupt_event=self._interrupt_event,
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
                    interrupt_event=self._interrupt_event,
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
                interrupt_event=self._interrupt_event,
            )

        if mode == "c":
            try:
                from core.execution.codex_sdk import (
                    CodexSDKExecutor,
                    is_codex_sdk_available,
                )
                if not is_codex_sdk_available():
                    raise ImportError("openai_codex_sdk not installed")
                return CodexSDKExecutor(
                    model_config=self.model_config,
                    anima_dir=self.anima_dir,
                    tool_registry=self._tool_registry,
                    personal_tools=self._personal_tools,
                    interrupt_event=self._interrupt_event,
                )
            except ImportError:
                logger.warning(
                    "CodexSDKExecutor unavailable (openai-codex-sdk not installed), "
                    "falling back to LiteLLM (Mode A)"
                )
                fallback_model_config = self.model_config.model_copy(deep=True)
                fallback_model: str | None = fallback_model_config.fallback_model

                # If fallback_model is not explicitly configured, pick a safe
                # provider/model based on available credentials.
                if not fallback_model:
                    model_name = fallback_model_config.model
                    uses_anthropic_key = (
                        bool(fallback_model_config.api_key and fallback_model_config.api_key.startswith("sk-ant-"))
                        or fallback_model_config.api_key_env.upper().startswith("ANTHROPIC")
                    )
                    if model_name.startswith("codex/"):
                        if uses_anthropic_key:
                            fallback_model = "anthropic/claude-sonnet-4-6"
                        else:
                            bare = model_name.split("/", 1)[1]
                            fallback_model = f"openai/{bare}"

                if fallback_model:
                    fallback_model_config.model = fallback_model
                    logger.warning(
                        "Mode C fallback remapped model: %s -> %s",
                        self.model_config.model,
                        fallback_model_config.model,
                    )
                return LiteLLMExecutor(
                    model_config=fallback_model_config,
                    anima_dir=self.anima_dir,
                    tool_handler=self._tool_handler,
                    tool_registry=self._tool_registry,
                    memory=self.memory,
                    personal_tools=self._personal_tools,
                    interrupt_event=self._interrupt_event,
                )

        if mode == "a":
            return LiteLLMExecutor(
                model_config=self.model_config,
                anima_dir=self.anima_dir,
                tool_handler=self._tool_handler,
                tool_registry=self._tool_registry,
                memory=self.memory,
                personal_tools=self._personal_tools,
                interrupt_event=self._interrupt_event,
            )

        # mode == "b" (basic)
        return AssistedExecutor(
            model_config=self.model_config,
            anima_dir=self.anima_dir,
            tool_handler=self._tool_handler,
            memory=self.memory,
            messenger=self.messenger,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            interrupt_event=self._interrupt_event,
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

    def _create_fallback_executor(self):
        """Create an AnthropicFallbackExecutor for when S mode SDK can't handle the prompt."""
        from core.execution import AnthropicFallbackExecutor
        logger.info("Creating AnthropicFallbackExecutor for oversized prompt")
        return AnthropicFallbackExecutor(
            model_config=self.model_config,
            anima_dir=self.anima_dir,
            tool_handler=self._tool_handler,
            tool_registry=self._tool_registry,
            memory=self.memory,
            personal_tools=self._personal_tools,
            interrupt_event=self._interrupt_event,
        )
