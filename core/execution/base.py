from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Base class and result type for execution engines."""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.prompt.context import ContextTracker
from core.execution.reminder import SystemReminderQueue
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory


# ── Streaming error ──────────────────────────────────────────

from core.exceptions import StreamDisconnectedError  # noqa: F401 – re-export



# ── Dynamic tool-record budget ───────────────────────────────

# Per-tool base budgets (character count) calibrated for a 128K context model.
_TOOL_RESULT_BASE_BUDGET: dict[str, int] = {
    "Read": 4000, "Grep": 4000, "Glob": 4000,
    "Bash": 2000,
    "web_search": 1500, "x_search": 1500,
    "write_file": 500, "edit_file": 500,
    "search_memory": 1500, "read_file": 4000,
}
_TOOL_RESULT_DEFAULT_BUDGET = 1000
_TOOL_INPUT_BASE_BUDGET = 500
_REFERENCE_CONTEXT_WINDOW = 128_000
_BUDGET_SCALE_MAX = 2.0
_BUDGET_SCALE_MIN = 0.25
_BUDGET_FLOOR = 300


def tool_result_save_budget(tool_name: str, context_window: int) -> int:
    """Return a context-window-proportional budget for tool result storage.

    Larger context windows get proportionally larger budgets so that
    tool results are not over-truncated.  The scale factor is clamped
    between ``_BUDGET_SCALE_MIN`` and ``_BUDGET_SCALE_MAX`` and a floor
    of ``_BUDGET_FLOOR`` characters is enforced.
    """
    base = _TOOL_RESULT_BASE_BUDGET.get(tool_name, _TOOL_RESULT_DEFAULT_BUDGET)
    scale = context_window / _REFERENCE_CONTEXT_WINDOW
    scale = min(scale, _BUDGET_SCALE_MAX)
    scale = max(scale, _BUDGET_SCALE_MIN)
    return max(_BUDGET_FLOOR, int(base * scale))


def tool_input_save_budget(context_window: int) -> int:
    """Return a context-window-proportional budget for tool input storage."""
    scale = context_window / _REFERENCE_CONTEXT_WINDOW
    scale = min(scale, _BUDGET_SCALE_MAX)
    scale = max(scale, _BUDGET_SCALE_MIN)
    return max(200, int(_TOOL_INPUT_BASE_BUDGET * scale))


# ── Result ───────────────────────────────────────────────────


def _truncate_for_record(text: str, max_len: int) -> str:
    """Truncate text for tool call record storage."""
    s = str(text)
    return s[:max_len] + "..." if len(s) > max_len else s


@dataclass
class ToolCallRecord:
    """Lightweight tool call record for propagation through the execution chain."""

    tool_name: str
    tool_id: str = ""
    input_summary: str = ""
    result_summary: str = ""
    is_error: bool = False


@dataclass
class ExecutionResult:
    """Result of a single execution session.

    Attributes:
        text: The textual response from the LLM.
        result_message: Provider-specific metadata (e.g. ResultMessage
            from Claude Agent SDK).  Used for session chaining in A1 mode.
        tool_call_records: Tool calls made during this execution session.
        force_chain: When True, AgentCore should force session chaining
            regardless of the ContextTracker state.  Set by the A1 executor
            when mid-session context auto-compact triggers via PreToolUse
            ``continue_=False``.
    """

    text: str
    result_message: Any = field(default=None, repr=False)
    replied_to_from_transcript: set[str] = field(default_factory=set)
    unconfirmed_sends: list[dict] = field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = field(default_factory=list)
    force_chain: bool = False


class BaseExecutor(ABC):
    """Abstract base for execution engines.

    Each subclass implements one execution mode (A1, A2, B, or fallback).
    Common credential resolution lives here.

    Parameter usage by mode:

    +------------------+--------+--------------+---------+----------+
    | Parameter        | A1     | A2           | B       | Fallback |
    |                  | (SDK)  | (LiteLLM)    | (Asst.) | (Anthr.) |
    +==================+========+==============+=========+==========+
    | prompt           | YES    | YES          | YES     | YES      |
    | system_prompt    | YES    | YES          | (own)   | YES      |
    | tracker          | YES    | YES          | no      | YES      |
    | shortterm        | no*    | YES          | no      | YES      |
    +------------------+--------+--------------+---------+----------+

    * A1 session chaining is managed externally by AgentCore.
      A2 and Fallback handle session chaining inline via
      handle_session_chaining().
    """

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
    ) -> None:
        self._model_config = model_config
        self._anima_dir = anima_dir
        self.reminder_queue: SystemReminderQueue = SystemReminderQueue()

    # -- Properties ----------------------------------------

    @property
    def supports_streaming(self) -> bool:
        """Whether this executor supports streaming execution.

        Returns True by default.  All executors now implement
        ``execute_streaming()`` — either token-level (A1, A1 Fallback,
        A2 non-Ollama) or iteration-level (A2 Ollama, B).
        """
        return True

    # -- Subordinate detection ----------------------------

    def _has_subordinates(self) -> bool:
        """Check if this anima has any subordinates (is a supervisor)."""
        try:
            from core.config.models import load_config
            config = load_config()
            my_name = self._anima_dir.name
            return any(
                cfg.supervisor == my_name
                for cfg in config.animas.values()
            )
        except Exception:
            return False

    # -- Credential helpers --------------------------------

    def _resolve_api_key(self) -> str | None:
        """Resolve the actual API key (direct value from config, then env var)."""
        if self._model_config.api_key:
            return self._model_config.api_key
        return os.environ.get(self._model_config.api_key_env)

    def _read_replied_to_file(self) -> set[str]:
        """Read replied_to entries from persistent file."""
        import json as _json
        import logging

        replied_to_path = self._anima_dir / "run" / "replied_to.jsonl"
        if not replied_to_path.exists():
            return set()
        names: set[str] = set()
        _logger = logging.getLogger("animaworks.execution.base")
        try:
            for line in replied_to_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                    if entry.get("success"):
                        names.add(entry["to"])
                except (KeyError, _json.JSONDecodeError):
                    continue
        except Exception as e:
            _logger.warning("Failed to read replied_to file: %s", e)
        return names

    def _resolve_llm_timeout(self) -> int:
        """Resolve LLM API call timeout in seconds.

        Priority:
          1. Explicit ``llm_timeout`` in ModelConfig (per-anima setting)
          2. Automatic: 300s for ``ollama/`` models, 600s for API models
        """
        if self._model_config.llm_timeout is not None:
            return self._model_config.llm_timeout
        if self._model_config.model.startswith("ollama/"):
            return 300
        return 600

    # -- Execution -----------------------------------------

    @abstractmethod
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
        """Run the execution engine and return the response.

        Args:
            prompt: The user/trigger prompt.
            system_prompt: Assembled system prompt (not used by Mode B,
                which builds its own from memory).
            tracker: Context usage tracker for monitoring window consumption.
                Not used by Mode B.
            shortterm: Short-term memory for inline session chaining
                (A2 / Fallback). A1 chaining is managed by AgentCore.
            trigger: Trigger identifier (e.g. "message:sakura", "heartbeat").
                Used by Mode B for post-call send judgement. Other modes
                ignore this parameter.
            images: Optional list of image dicts with ``data`` (base64) and
                ``media_type`` keys. Supported by A1 Fallback and A2 modes.
            max_turns_override: If provided, overrides ``max_turns`` from
                ModelConfig for this single execution.

        Returns:
            ExecutionResult with the response text and optional metadata.
        """
        ...

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream execution events from the engine.

        Default implementation falls back to blocking execute() and
        yields the full response as a single text_delta followed by
        a done event.

        Subclasses with native streaming support (e.g. AgentSDKExecutor)
        should override this method.

        Args:
            system_prompt: Assembled system prompt.
            prompt: The user/trigger prompt.
            tracker: Context usage tracker.
            images: Optional list of image dicts for multimodal input.
            prior_messages: Optional structured conversation history.
            max_turns_override: If provided, overrides ``max_turns`` from
                ModelConfig for this single execution.

        Yields:
            Dicts with at least a type key. Common types:
                - {"type": "text_delta", "text": "..."}
                - {"type": "done", "full_text": "...", "result_message": ...}
        """
        from dataclasses import asdict

        result = await self.execute(
            prompt=prompt,
            system_prompt=system_prompt,
            tracker=tracker,
            images=images,
            prior_messages=prior_messages,
            max_turns_override=max_turns_override,
        )
        yield {"type": "text_delta", "text": result.text}
        yield {
            "type": "done",
            "full_text": result.text,
            "result_message": result.result_message,
            "tool_call_records": [asdict(r) for r in result.tool_call_records],
        }
