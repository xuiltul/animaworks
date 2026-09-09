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

    def _create_executor(self, model_config=None, *, _unavailable_modes=frozenset()):
        """Construct the selected adapter; only configured alternatives may replace it."""
        from importlib import import_module

        from core.config.model_config import resolve_unavailable_model_config
        from core.exceptions import ExecutorUnavailableError
        from core.execution import AssistedExecutor, LiteLLMExecutor
        from core.i18n import t

        active_config = model_config or self.model_config
        mode = self._resolve_execution_mode(active_config)
        common = {
            "model_config": active_config,
            "anima_dir": self.anima_dir,
            "tool_registry": self._tool_registry,
            "personal_tools": self._personal_tools,
            "interrupt_event": self._interrupt_event,
        }
        adapters = {
            "s": ("agent_sdk", "AgentSDKExecutor", None),
            "c": ("codex_sdk", "CodexSDKExecutor", "is_codex_sdk_available"),
            "d": ("cursor_agent", "CursorAgentExecutor", "is_cursor_agent_available"),
            "g": ("gemini_cli", "GeminiCLIExecutor", "is_gemini_cli_available"),
            "x": ("grok_cli", "GrokCLIExecutor", "is_grok_cli_available"),
        }
        if mode in adapters:
            module_name, class_name, availability = adapters[mode]
            try:
                if mode == "s" and not self._sdk_available:
                    raise ImportError("claude_agent_sdk unavailable")
                module = import_module(f"core.execution.{module_name}")
                if availability and not getattr(module, availability)():
                    raise ImportError(f"{class_name} unavailable")
                executor_class = getattr(module, class_name)
            except ImportError as exc:
                unavailable = _unavailable_modes | {mode.upper()}
                fallback = resolve_unavailable_model_config(active_config, unavailable_modes=unavailable)
                if fallback is None:
                    raise ExecutorUnavailableError(
                        t("executor.unavailable_no_configured_fallback", mode=mode.upper(), model=active_config.model)
                    ) from exc
                from core.execution.fallback_activity import log_model_fallback
                from core.memory.activity import ActivityLogger

                log_model_fallback(
                    ActivityLogger(self.anima_dir),
                    active_config,
                    fallback,
                    channel="executor",
                    phase="unavailable",
                )
                return self._create_executor(fallback, _unavailable_modes=unavailable)
            if mode == "c":
                common["codex_home"] = self._codex_home
            return executor_class(**common)

        common.update(tool_handler=self._tool_handler, memory=self.memory)
        if mode == "a":
            return LiteLLMExecutor(**common)
        return AssistedExecutor(**common, messenger=self.messenger)

    def _resolve_api_key(self) -> str | None:
        """Resolve the actual API key (direct value from config.json, then env var)."""
        import os

        if self.model_config.api_key:
            return self.model_config.api_key
        if not self.model_config.api_key_env:
            return None
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

    def _create_fallback_executor(self, model_config=None):
        """Create an AnthropicFallbackExecutor for when S mode SDK can't handle the prompt."""
        from core.execution import AnthropicFallbackExecutor

        active_config = model_config or self.model_config
        logger.info("Creating AnthropicFallbackExecutor for oversized prompt")
        return AnthropicFallbackExecutor(
            model_config=active_config,
            anima_dir=self.anima_dir,
            tool_handler=self._tool_handler,
            tool_registry=self._tool_registry,
            memory=self.memory,
            personal_tools=self._personal_tools,
            interrupt_event=self._interrupt_event,
        )
