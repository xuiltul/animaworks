from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Base class and result type for execution engines."""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ── Streaming error ──────────────────────────────────────────
from core.exceptions import StreamDisconnectedError  # noqa: F401 – re-export
from core.execution.reminder import SystemReminderQueue
from core.memory.shortterm import ShortTermMemory
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig

# ── Per-task interrupt event ─────────────────────────────────
# Each asyncio task (i.e. each concurrent HTTP request) gets its own
# interrupt event via this ContextVar, avoiding race conditions when
# multiple chat threads share a single executor instance.

_active_interrupt_event: ContextVar[asyncio.Event | None] = ContextVar(
    "_active_interrupt_event",
    default=None,
)


# ── Adaptive Thinking helpers ─────────────────────────────────

_ADAPTIVE_MODELS = frozenset({"claude-opus-4-6", "claude-sonnet-4-6"})

_PROVIDER_PREFIX_RE = re.compile(
    r"^(?:anthropic|bedrock|vertex_ai)/"
    r"(?:[a-z]{2}\.anthropic\.)?"
)


def _bare_model_name(model: str) -> str:
    """Strip provider and region prefixes to get the canonical model name.

    ``bedrock/jp.anthropic.claude-sonnet-4-6`` → ``claude-sonnet-4-6``
    ``anthropic/claude-opus-4-6``              → ``claude-opus-4-6``
    ``claude-sonnet-4-6``                      → ``claude-sonnet-4-6``
    ``openai/gpt-4o``                          → ``gpt-4o``
    """
    stripped = _PROVIDER_PREFIX_RE.sub("", model)
    if "/" in stripped:
        return stripped.split("/")[-1]
    return stripped


def is_adaptive_model(model: str) -> bool:
    """Return True if *model* supports Anthropic adaptive thinking (4.6 series)."""
    return _bare_model_name(model) in _ADAPTIVE_MODELS


def is_anthropic_claude(model: str) -> bool:
    """Return True if *model* is an Anthropic Claude model."""
    return _bare_model_name(model).startswith("claude-")


def is_bedrock_qwen(model: str) -> bool:
    """Return True if *model* is a Qwen model on AWS Bedrock."""
    return model.startswith("bedrock/") and "qwen" in model.lower()


def is_bedrock_glm(model: str) -> bool:
    """Return True if *model* is a ZhipuAI GLM model on AWS Bedrock."""
    return model.startswith("bedrock/") and "glm" in model.lower()


def is_bedrock_kimi(model: str) -> bool:
    """Return True if *model* is a Moonshot Kimi model on AWS Bedrock."""
    return model.startswith("bedrock/") and ("kimi" in model.lower() or "moonshot" in model.lower())


def supports_streaming_tool_use(model: str) -> bool:
    """Return True if *model* supports tool use in streaming mode.

    Some Bedrock models (Llama 4, etc.) support tool use only via non-streaming
    Converse API.  When streaming is requested with tools, Bedrock returns:
    ``"This model doesn't support tool use in streaming mode."``

    For these models, callers should fall back to ``stream=False`` when tools
    are present in the request.
    """
    if not model.startswith("bedrock/"):
        return True
    bare = _bare_model_name(model).lower()
    # Models known NOT to support streaming tool use on Bedrock:
    _no_streaming_tool_use = (
        "llama4",  # Meta Llama 4 Scout / Maverick
    )
    return not any(tag in bare for tag in _no_streaming_tool_use)


def resolve_thinking_effort(model: str, effort: str | None) -> str:
    """Resolve thinking effort, clamping ``"max"`` to ``"high"`` for non-Opus-4.6."""
    resolved = effort or "high"
    if resolved == "max":
        if _bare_model_name(model) != "claude-opus-4-6":
            return "high"
    return resolved


# ── Think-tag strip filter ────────────────────────────────────

_THINK_TAG_RE = re.compile(r"^(.*?)</think>\s*", re.DOTALL)
_MAX_THINK_BUFFER = 50_000


def strip_thinking_tags(text: str) -> tuple[str, str]:
    """Strip ``<think>...</think>`` block from *text* (non-streaming).

    Some models (e.g. Qwen3.5) emit reasoning inside ``content`` wrapped
    in ``<think>`` tags rather than using a dedicated ``reasoning_content``
    field.  This helper splits the text into ``(thinking, response)``.

    Also handles the case where ``<think>`` is absent but ``</think>``
    is present (e.g. vLLM reasoning parser strips the opening tag).

    Returns ``("", text)`` when no ``</think>`` closing tag is found.
    """
    m = _THINK_TAG_RE.match(text)
    if m:
        raw = m.group(1)
        stripped = raw.lstrip()
        if stripped.startswith("<think>"):
            raw = stripped[len("<think>") :]
        # Strip remaining orphan </think> tags (e.g. Qwen emitting multiple think blocks)
        response = re.sub(r"</think>\s*", "", text[m.end() :])
        return raw, response
    return "", text


class StreamingThinkFilter:
    """Streaming filter that routes ``<think>`` content to thinking deltas.

    Feed each streamed chunk via :meth:`feed`; it returns a
    ``(thinking_delta, text_delta)`` tuple.  Content is only buffered
    when the stream begins with ``<think>``; otherwise chunks pass
    through immediately as *text*.

    The ``</think>`` check runs **before** the early-exit so that a
    single chunk like ``"thinking</think>response"`` (vLLM reasoning
    parser may strip the opening tag) is still split correctly.

    A safety valve flushes the buffer as plain text if it exceeds
    ``_MAX_THINK_BUFFER`` characters without encountering ``</think>``.

    For models that emit thinking without proper ``<think>`` tags, the
    post-stream ``strip_thinking_tags`` safety net in the streaming
    executor catches any residual leaks in the final ``full_text``.
    """

    _THINK_OPEN = "<think>"

    __slots__ = ("_buffer", "_done")

    def __init__(self) -> None:
        self._buffer = ""
        self._done = False

    def feed(self, delta: str) -> tuple[str, str]:
        """Process *delta* and return ``(thinking_text, response_text)``."""
        if self._done:
            return ("", delta)
        self._buffer += delta

        # Check for </think> FIRST — handles both normal <think>...</think>
        # and the edge case where <think> is absent (vLLM reasoning parser).
        if "</think>" in self._buffer:
            self._done = True
            parts = self._buffer.split("</think>", 1)
            response = parts[1].lstrip("\n")
            self._buffer = ""
            thinking_raw = parts[0]
            stripped_t = thinking_raw.lstrip()
            if stripped_t.startswith("<think>"):
                thinking_raw = stripped_t[len("<think>") :]
            return (thinking_raw, response)

        # Early exit: if accumulated text clearly doesn't start with <think>,
        # pass through immediately so non-think streams aren't buffered.
        stripped = self._buffer.lstrip()
        if stripped and not stripped.startswith(self._THINK_OPEN) and not self._THINK_OPEN.startswith(stripped):
            self._done = True
            text = self._buffer
            self._buffer = ""
            return ("", text)

        if len(self._buffer) > _MAX_THINK_BUFFER:
            text = self._buffer
            self._buffer = ""
            self._done = True
            return ("", text)
        return ("", "")

    def flush(self) -> str:
        """Return any remaining buffered text at end-of-stream."""
        if self._buffer:
            text = self._buffer
            self._buffer = ""
            return text
        return ""


# ── Repetition detection ────────────────────────────────────


class RepetitionDetector:
    """Detect degenerate repetition in streaming token output.

    Uses n-gram frequency counting on word-level tokens. When any single
    n-gram appears *threshold* times after *min_tokens* words have been
    accumulated, the detector fires.
    """

    def __init__(
        self,
        n: int = 10,
        threshold: int = 10,
        min_tokens: int = 100,
    ) -> None:
        self._n = n
        self._threshold = threshold
        self._min_tokens = min_tokens
        self._tokens: list[str] = []
        self._ngram_counts: dict[tuple[str, ...], int] = {}

    def feed(self, text: str) -> bool:
        """Feed a text chunk and check for repetition.

        Returns ``True`` if degenerate repetition is detected.
        """
        words = text.split()
        if not words:
            return False
        self._tokens.extend(words)
        if len(self._tokens) < self._min_tokens:
            return False
        for i in range(len(self._tokens) - len(words), len(self._tokens)):
            start = i - self._n + 1
            if start < 0:
                continue
            ngram = tuple(self._tokens[start : i + 1])
            self._ngram_counts[ngram] = self._ngram_counts.get(ngram, 0) + 1
            if self._ngram_counts[ngram] >= self._threshold:
                return True
        return False

    def check_full_text(self, text: str) -> bool:
        """Check complete response text for repetition (post-hoc).

        Useful for iteration-level streaming where tokens aren't
        available incrementally.
        """
        words = text.split()
        if len(words) < self._min_tokens:
            return False
        counts: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - self._n + 1):
            ngram = tuple(words[i : i + self._n])
            counts[ngram] = counts.get(ngram, 0) + 1
            if counts[ngram] >= self._threshold:
                return True
        return False


# ── Dynamic tool-record budget ───────────────────────────────

# Per-tool base budgets (character count) calibrated for a 128K context model.
_TOOL_RESULT_BASE_BUDGET: dict[str, int] = {
    "Read": 4000,
    "Grep": 4000,
    "Glob": 4000,
    "Bash": 2000,
    "web_search": 1500,
    "x_search": 1500,
    "write_file": 500,
    "edit_file": 500,
    "search_memory": 1500,
    "read_file": 4000,
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


# ── Session result protocol ───────────────────────────────────


@runtime_checkable
class SessionResultLike(Protocol):
    """Minimal interface required for session chaining.

    S mode's ``ResultMessage`` satisfies this structurally.
    A/B modes pass ``None``.
    """

    @property
    def num_turns(self) -> int: ...

    @property
    def session_id(self) -> str: ...


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
class TokenUsage:
    """Token usage for a single execution session (all modes)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def merge(self, other: TokenUsage) -> None:
        """Accumulate usage from another TokenUsage (for chaining)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ExecutionResult:
    """Result of a single execution session.

    Attributes:
        text: The textual response from the LLM.
        result_message: Provider-specific metadata (e.g. ResultMessage
            from Claude Agent SDK).  Used for session chaining in S mode.
        tool_call_records: Tool calls made during this execution session.
        force_chain: When True, AgentCore should force session chaining
            regardless of the ContextTracker state.  Set by the S executor
            when mid-session context auto-compact triggers via PreToolUse
            ``continue_=False``.
        usage: Token usage for this session.  Populated by each executor.
    """

    text: str
    result_message: SessionResultLike | None = field(default=None, repr=False)
    replied_to_from_transcript: set[str] = field(default_factory=set)
    unconfirmed_sends: list[dict] = field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = field(default_factory=list)
    force_chain: bool = False
    usage: TokenUsage | None = None


class BaseExecutor(ABC):
    """Abstract base for execution engines.

    Each subclass implements one execution mode (S, C, A, B, or fallback).
    Common credential resolution lives here.

    Parameter usage by mode:

    +------------------+--------+--------+--------------+---------+----------+
    | Parameter        | S      | C      | A            | B       | Fallback |
    |                  | (SDK)  | (Codex)| (LiteLLM)    | (Asst.) | (Anthr.) |
    +==================+========+========+==============+=========+==========+
    | prompt           | YES    | YES    | YES          | YES     | YES      |
    | system_prompt    | YES    | YES    | YES          | (own)   | YES      |
    | tracker          | YES    | YES    | YES          | no      | YES      |
    | shortterm        | no*    | no*    | YES          | no      | YES      |
    +------------------+--------+--------+--------------+---------+----------+

    * S and C session chaining is managed externally by AgentCore.
      A and Fallback handle session chaining inline via
      handle_session_chaining().
    """

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        self._model_config = model_config
        self._anima_dir = anima_dir
        self._interrupt_event = interrupt_event
        self.reminder_queue: SystemReminderQueue = SystemReminderQueue()
        self._hb_soft_timeout_s: int = self._load_hb_soft_timeout()

    def _load_hb_soft_timeout(self) -> int:
        """Load heartbeat soft_timeout_seconds from config (cached at init)."""
        try:
            from core.config.models import load_config
            return load_config().heartbeat.soft_timeout_seconds
        except Exception:
            return 300

    # -- Properties ----------------------------------------

    @property
    def supports_streaming(self) -> bool:
        """Whether this executor supports streaming execution.

        Returns True by default.  All executors now implement
        ``execute_streaming()`` — either token-level (S, S Fallback,
        A non-Ollama) or iteration-level (A Ollama, B).
        """
        return True

    # -- Subordinate detection ----------------------------

    def _has_subordinates(self) -> bool:
        """Check if this anima has any subordinates (is a supervisor)."""
        try:
            from core.config.models import load_config

            config = load_config()
            my_name = self._anima_dir.name
            return any(cfg.supervisor == my_name for cfg in config.animas.values())
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

    def _apply_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Apply provider-specific LiteLLM kwargs from ``extra_keys``."""
        model = self._model_config.model
        extra = self._model_config.extra_keys

        if model.startswith("azure/"):
            api_version = extra.get("api_version") or os.environ.get("AZURE_API_VERSION")
            if api_version:
                kwargs["api_version"] = api_version

        elif model.startswith("vertex_ai/"):
            for key in ("vertex_project", "vertex_location", "vertex_credentials"):
                val = extra.get(key) or os.environ.get(key.upper())
                if val:
                    kwargs[key] = val

        elif model.startswith("bedrock/"):
            for key in ("aws_access_key_id", "aws_secret_access_key", "aws_region_name"):
                val = extra.get(key) or os.environ.get(key.upper())
                if val:
                    kwargs[key] = val

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

    def _resolve_num_retries(self) -> int:
        """Resolve LLM API retry count from ``config.server.llm_num_retries``."""
        try:
            from core.config import load_config

            return load_config().server.llm_num_retries
        except Exception:
            return 3

    def _resolve_cw(self) -> int:
        """Resolve context window with config overrides."""
        from core.config import load_config
        from core.exceptions import ConfigError
        from core.prompt.context import resolve_context_window

        try:
            overrides = load_config().model_context_windows
        except (ConfigError, OSError):
            overrides = None
        return resolve_context_window(self._model_config.model, overrides)

    def _resolve_cw_overrides(self) -> dict[str, int] | None:
        """Return config model_context_windows or None."""
        from core.config import load_config
        from core.exceptions import ConfigError

        try:
            return load_config().model_context_windows
        except (ConfigError, OSError):
            return None

    def _check_interrupted(self) -> bool:
        """Return True if the interrupt event has been set.

        Prefers the per-task ContextVar (thread-safe for parallel streams)
        over the instance-level fallback.
        """
        evt = _active_interrupt_event.get(None)
        if evt is None:
            evt = self._interrupt_event
        return evt is not None and evt.is_set()

    # -- Execution -----------------------------------------

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run the execution engine and return the response.

        Args:
            prompt: The user/trigger prompt.
            system_prompt: Assembled system prompt (not used by Mode B,
                which builds its own from memory).
            tracker: Context usage tracker for monitoring window consumption.
                Not used by Mode B.
            shortterm: Short-term memory for inline session chaining
                (A / Fallback). S chaining is managed by AgentCore.
            trigger: Trigger identifier (e.g. "message:sakura", "heartbeat").
                Used by Mode B for post-call send judgement. Other modes
                ignore this parameter.
            images: Optional list of image dicts with ``data`` (base64) and
                ``media_type`` keys. Supported by S Fallback and A modes.
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
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
        thread_id: str = "default",
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
            thread_id=thread_id,
        )
        yield {"type": "text_delta", "text": result.text}
        yield {
            "type": "done",
            "full_text": result.text,
            "result_message": result.result_message,
            "tool_call_records": [asdict(r) for r in result.tool_call_records],
        }
