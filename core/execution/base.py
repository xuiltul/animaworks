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
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory


# ── Streaming error ──────────────────────────────────────────


class StreamDisconnectedError(Exception):
    """Raised when a streaming session disconnects unexpectedly.

    Carries partial response text accumulated before the disconnect
    so AgentCore can build a checkpoint-based retry prompt.
    """

    def __init__(
        self,
        message: str = "Stream disconnected",
        *,
        partial_text: str = "",
    ) -> None:
        super().__init__(message)
        self.partial_text = partial_text


# ── Result ───────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Result of a single execution session.

    Attributes:
        text: The textual response from the LLM.
        result_message: Provider-specific metadata (e.g. ResultMessage
            from Claude Agent SDK).  Used for session chaining in A1 mode.
    """

    text: str
    result_message: Any = field(default=None, repr=False)
    replied_to_from_transcript: set[str] = field(default_factory=set)


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

    # -- Properties ----------------------------------------

    @property
    def supports_streaming(self) -> bool:
        """Whether this executor supports streaming execution.

        Returns True by default.  All executors now implement
        ``execute_streaming()`` — either token-level (A1, A1 Fallback,
        A2 non-Ollama) or iteration-level (A2 Ollama, B).
        """
        return True

    # -- Credential helpers --------------------------------

    def _resolve_api_key(self) -> str | None:
        """Resolve the actual API key (direct value from config, then env var)."""
        if self._model_config.api_key:
            return self._model_config.api_key
        return os.environ.get(self._model_config.api_key_env)

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

        Yields:
            Dicts with at least a type key. Common types:
                - {"type": "text_delta", "text": "..."}
                - {"type": "done", "full_text": "...", "result_message": ...}
        """
        result = await self.execute(
            prompt=prompt,
            system_prompt=system_prompt,
            tracker=tracker,
            images=images,
        )
        yield {"type": "text_delta", "text": result.text}
        yield {
            "type": "done",
            "full_text": result.text,
            "result_message": result.result_message,
        }
