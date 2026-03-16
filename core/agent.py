from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0


"""AgentCore -- orchestrator for Digital Anima execution cycles.

This module is the public façade.  Implementation is split into Mixin
sub-modules for manageability:

- ``_agent_prompt_log``  -- prompt-log constants & helpers
- ``_agent_executor``    -- ExecutorFactoryMixin
- ``_agent_priming``     -- PrimingMixin
- ``_agent_cycle``       -- CycleMixin (Streaming executor (S / A / all modes))
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.notification.notifier import HumanNotifier

from core._agent_cycle import CycleMixin

# ── Mixin imports ───────────────────────────────────────────────
from core._agent_executor import ExecutorFactoryMixin
from core._agent_priming import PrimingMixin

# ── Re-exports for backward compatibility ───────────────────────
# Tests and other modules import these symbols from ``core.agent``.
from core._agent_prompt_log import (  # noqa: F401
    _PROMPT_HARD_LIMIT_BYTES,
    _PROMPT_LOG_RETENTION_DAYS,
    _PROMPT_SOFT_LIMIT_BYTES,
    _rotate_prompt_logs,
    _save_prompt_log,
    _save_prompt_log_end,
)
from core.background import BackgroundTaskManager
from core.exceptions import AnimaWorksError  # noqa: F401
from core.memory import MemoryManager
from core.messenger import Messenger
from core.schemas import ModelConfig
from core.tooling.handler import ToolHandler

logger = logging.getLogger("animaworks.agent")


class AgentCore(
    CycleMixin,
    PrimingMixin,
    ExecutorFactoryMixin,
):
    """Orchestrates execution of a Digital Anima's thinking/acting cycle.

    Delegates actual LLM execution to an appropriate Executor:
      - S  (SDK, Claude):             ``AgentSDKExecutor``
      - S  fallback (no Agent SDK):   ``AnthropicFallbackExecutor``
      - A  (autonomous, non-Claude):   ``LiteLLMExecutor``
      - B  (basic):                    ``AssistedExecutor``
    """

    _MAX_AGENT_LOCKS = 20

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
        self._interrupt_event: asyncio.Event | None = None
        self._tool_registry = self._init_tool_registry()
        self._personal_tools = self._discover_personal_tools()
        self._sdk_available = self._check_sdk()
        self._agent_locks: dict[str, asyncio.Lock] = {}
        self._progress_callback: Callable[[], None] | None = None

        # Build human notifier for top-level animas
        human_notifier = self._build_human_notifier()

        # Background task manager
        self._background_manager = self._build_background_manager()

        # Composable subsystems
        from core.config.models import resolve_context_window

        cw = resolve_context_window(self.model_config.model) or 32_000
        self._tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=self._tool_registry,
            personal_tools=self._personal_tools,
            human_notifier=human_notifier,
            background_manager=self._background_manager,
            context_window=cw,
            superuser=self._is_debug_superuser(),
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

    def set_interrupt_event(self, event: asyncio.Event) -> None:
        """Set interrupt event for execution cancellation.

        Sets both the per-task ContextVar (thread-safe for parallel streams)
        and the legacy instance attribute (backward compat for single-thread).
        """
        from core.execution.base import _active_interrupt_event

        _active_interrupt_event.set(event)
        self._interrupt_event = event
        if hasattr(self._executor, "_interrupt_event"):
            self._executor._interrupt_event = event

    def _get_agent_lock(self, thread_id: str = "default") -> asyncio.Lock:
        """Get or create a per-thread agent lock with LRU eviction."""
        if thread_id not in self._agent_locks:
            if len(self._agent_locks) >= self._MAX_AGENT_LOCKS:
                for k in list(self._agent_locks):
                    if not self._agent_locks[k].locked():
                        del self._agent_locks[k]
                        break
            self._agent_locks[thread_id] = asyncio.Lock()
        return self._agent_locks[thread_id]

    def set_on_message_sent(self, fn: Callable[[str, str, str], None]) -> None:
        """Inject a callback invoked after a send_message tool call."""
        self._tool_handler.on_message_sent = fn

    def set_on_schedule_changed(self, fn: Callable[[str], None] | None) -> None:
        """Inject a callback invoked when heartbeat.md or cron.md is modified."""
        self._tool_handler.on_schedule_changed = fn

    def set_task_cwd(self, cwd: Path | None) -> None:
        """Set the working directory for the next TaskExec session.

        When set, execution engines use this instead of anima_dir as cwd.
        Call with None to reset after task completion.
        """
        if hasattr(self._executor, "set_task_cwd"):
            self._executor.set_task_cwd(cwd)
        if hasattr(self._tool_handler, "set_task_cwd"):
            self._tool_handler.set_task_cwd(cwd)

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notification events from ToolHandler."""
        return self._tool_handler.drain_notifications()

    def update_model_config(self, new_config: ModelConfig) -> None:
        """Update model config, rebuild executor and refresh context window."""
        self.model_config = new_config
        from core.config.models import resolve_context_window

        cw = resolve_context_window(new_config.model) or 32_000
        self._tool_handler._context_window = cw
        self._executor = self._create_executor()
        logger.info(
            "AgentCore: config reloaded, model=%s, context_window=%d",
            new_config.model,
            cw,
        )

    def reset_reply_tracking(self, session_type: str | None = None) -> None:
        """Clear reply tracking (call at start of each heartbeat cycle)."""
        self._tool_handler.reset_replied_to(session_type=session_type)

    def reset_posted_channels(self, session_type: str | None = None) -> None:
        """Clear posted-channels tracking (call at start of each heartbeat cycle)."""
        self._tool_handler.reset_posted_channels(session_type=session_type)

    def reset_read_paths(self) -> None:
        """Clear read-path tracking (call at start of each cycle)."""
        self._tool_handler.reset_read_paths()

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

    def _build_human_notifier(self) -> HumanNotifier | None:
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
            config_eligible = {name: tc.threshold_s for name, tc in config.background_task.eligible_tools.items()}

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
        """Determine the effective execution mode: ``s``, ``c``, ``a``, or ``b``.

        Uses ``resolved_mode`` from config when available.
        Falls back to auto-detection for legacy config.md paths.
        """
        rm = self.model_config.resolved_mode
        if rm:
            mode = rm.lower()  # "S" → "s"
            return mode

        # Fallback (resolved_mode absent = legacy config.md path)
        if self._is_claude_model() and self._sdk_available:
            return "s"
        return "a"

    @staticmethod
    def _check_sdk() -> bool:
        try:
            from claude_agent_sdk import query  # noqa: F401

            return True
        except ImportError:
            logger.warning("claude-agent-sdk not available, falling back to anthropic SDK")
            return False

    def _is_debug_superuser(self) -> bool:
        """Check if this anima has debug_superuser flag in status.json."""
        status_path = self.anima_dir / "status.json"
        if not status_path.is_file():
            return False
        try:
            import json as _json_mod

            data = _json_mod.loads(status_path.read_text(encoding="utf-8"))
            return bool(data.get("debug_superuser"))
        except (ValueError, OSError):
            return False
