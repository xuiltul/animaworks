from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""ExecutorFactoryMixin -- tool registry, executor factory, API key helpers.

Extracted from ``core.agent.AgentCore`` as a Mixin.  All ``self`` references
are resolved at runtime via MRO when mixed into ``AgentCore``.
"""

import logging

logger = logging.getLogger("animaworks.agent")


class ExecutorFactoryMixin:
    """Mixin: tool registry initialisation, executor creation, API key resolution."""

    def _init_tool_registry(self) -> list[str]:
        """Initialize tool registry from permissions config (default-all).

        Loads :class:`PermissionsConfig` and delegates to
        :func:`core.tooling.permissions.get_permitted_tools`.
        """
        try:
            from core.config.models import PermissionsConfig, load_permissions
            from core.tooling.permissions import get_permitted_tools

            if self.memory and hasattr(self.memory, "anima_dir"):
                config = load_permissions(self.memory.anima_dir)
            else:
                config = PermissionsConfig()
            return sorted(get_permitted_tools(config))
        except Exception:
            logger.debug("Tool registry initialization skipped")
            return []

    def _discover_personal_tools(self) -> dict[str, str]:
        """Discover common and personal tool modules."""
        try:
            from core.tools import discover_common_tools, discover_personal_tools

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
                    "AgentSDKExecutor unavailable (claude_agent_sdk not installed), trying AnthropicFallbackExecutor"
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
                    "CodexSDKExecutor unavailable (openai-codex-sdk not installed), falling back to LiteLLM (Mode A)"
                )
                fallback_model_config = self.model_config.model_copy(deep=True)
                fallback_model: str | None = fallback_model_config.fallback_model

                # If fallback_model is not explicitly configured, pick a safe
                # provider/model based on available credentials.
                if not fallback_model:
                    model_name = fallback_model_config.model
                    uses_anthropic_key = bool(
                        fallback_model_config.api_key and fallback_model_config.api_key.startswith("sk-ant-")
                    ) or fallback_model_config.api_key_env.upper().startswith("ANTHROPIC")
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
