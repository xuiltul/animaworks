from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Base class and result type for execution engines."""

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

# ── Streaming error ──────────────────────────────────────────
from core.exceptions import StreamDisconnectedError  # noqa: F401 – re-export
from core.execution.events import done_event, stream_events, text_delta_event
from core.execution.reminder import SystemReminderQueue
from core.execution.session.session_store import SessionEngine
from core.execution.session.session_types import is_resumable_trigger, resolve_runtime_session_type
from core.memory.conversation.shortterm import ShortTermMemory
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger(__name__)

# ── Per-task interrupt event ─────────────────────────────────
# Each asyncio task (i.e. each concurrent HTTP request) gets its own
# interrupt event via this ContextVar, avoiding race conditions when
# multiple chat threads share a single executor instance.

_active_interrupt_event: ContextVar[asyncio.Event | None] = ContextVar(
    "_active_interrupt_event",
    default=None,
)


# ── Adaptive Thinking helpers ─────────────────────────────────

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


# Anthropic model ids: claude-<family>-<major>[-<minor>][-<date>]
_CLAUDE_VERSION_RE = re.compile(r"^claude-(opus|sonnet|fable|mythos|haiku)-(\d+)(?:-(\d{1,2}))?(?!\d)")


def is_adaptive_model(model: str) -> bool:
    """Return True if *model* uses Anthropic adaptive thinking + ``effort``.

    Opus/Sonnet 4.6 and every Opus/Sonnet/Fable/Mythos release from 4.7 on.
    These reject ``budget_tokens`` (4.7+) and are tuned through ``effort``.
    """
    m = _CLAUDE_VERSION_RE.match(_bare_model_name(model))
    if m is None:
        return False
    family, major, minor = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    if family == "haiku":
        return False
    if (major, minor) >= (4, 7):
        return True
    return (major, minor) == (4, 6) and family in ("opus", "sonnet")


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


# OpenAI's own model families (gpt-4.1, gpt-5.4, o3, o4-mini, chatgpt-4o-latest).
# gpt-oss is open-weight and usually served by vLLM / llama.cpp.
_OPENAI_HOSTED_RE = re.compile(r"^(?:gpt-(?!oss)|o\d|chatgpt-)", re.IGNORECASE)


def is_openai_hosted_model(model: str) -> bool:
    """Return True if *model* is an ``openai/`` model that OpenAI itself serves.

    The ``openai/`` prefix also covers self-hosted OpenAI-compatible servers
    (vLLM, llama.cpp). Those take chat-template fields such as
    ``enable_thinking``; the OpenAI API rejects them with 400.
    """
    if not model.startswith("openai/"):
        return False
    name = model[len("openai/") :]
    # Hugging Face style ids (org/model) are self-hosted.
    return "/" not in name and _OPENAI_HOSTED_RE.match(name) is not None


def supports_streaming_tool_use(model: str) -> bool:
    """Return True if *model* supports tool use in streaming mode.

    Checks two sources:
    1. ``models.json`` ``tool_calling.stream`` — if explicitly ``false``,
       streaming tool use is disabled for that model.
    2. Hardcoded list — some Bedrock models (Llama 4, etc.) support tool use
       only via non-streaming Converse API.

    For these models, callers should fall back to ``stream=False`` when tools
    are present in the request.
    """
    try:
        from core.config.model_mode import _match_models_json

        entry = _match_models_json(model)
        if entry:
            tc_cfg = entry.get("tool_calling")
            if isinstance(tc_cfg, dict) and tc_cfg.get("stream") is False:
                return False
    except Exception:
        logger.debug(
            "Failed to load models.json for streaming tool-use check: model=%s",
            model,
            exc_info=True,
        )

    if not model.startswith("bedrock/"):
        return True
    bare = _bare_model_name(model).lower()
    _no_streaming_tool_use = (
        "llama4",  # Meta Llama 4 Scout / Maverick
    )
    return not any(tag in bare for tag in _no_streaming_tool_use)


def supports_max_effort(model: str) -> bool:
    """Return True if *model* accepts ``effort="max"``.

    Among Claude models every adaptive model except Sonnet 4.6 does; Kimi on
    Bedrock only takes ``"high"``.  Any other model receives the configured
    value unchanged: an operator who wrote ``"max"`` for e.g. ``gpt-6-luna``
    or DeepSeek meant it.
    """
    if is_bedrock_kimi(model):
        return False
    bare = _bare_model_name(model)
    if not bare.startswith("claude-"):
        return True
    return is_adaptive_model(model) and not bare.startswith("claude-sonnet-4-6")


def resolve_thinking_effort(model: str, effort: str | None) -> str:
    """Resolve thinking effort, clamping ``"max"`` to ``"high"`` where unsupported."""
    resolved = effort or "high"
    if resolved == "max" and not supports_max_effort(model):
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


_THINK_CLOSE_ONLY_RE = re.compile(r"</think>\s*")


def resolve_streamed_leaked_thinking(full_text: str) -> tuple[str, str]:
    """Post-stream disambiguation of a ``</think>`` found in *streamed* content.

    The streaming ``StreamingThinkFilter`` only discards content preceding a
    ``</think>`` when the stream *starts* with ``<think>``; otherwise it streams
    the head to the client as real answer text.  The post-stream safety net,
    however, used :func:`strip_thinking_tags` directly, which discards
    everything before the first ``</think>`` even when there was **no opening
    tag** — silently deleting an answer that was already shown to the user when
    a model emits a spurious/orphan ``</think>`` *after* real content.

    This resolver keeps the genuine cases handled while refusing to delete
    already-streamed answer text:

    * With an opening ``<think>`` present → behave exactly like
      :func:`strip_thinking_tags` (real inline reasoning block).
    * With an orphan ``</think>`` and **no** opening tag → treat the tag as
      spurious: preserve the preceding answer and strip only the stray tag(s).

    Returns ``(thinking, clean_text)``.
    """
    leaked, clean = strip_thinking_tags(full_text)
    if leaked and "<think>" not in full_text:
        # Orphan </think> with no opening tag: do not discard the answer that
        # the streaming filter already emitted as text. Drop only the stray tag.
        return "", _THINK_CLOSE_ONLY_RE.sub("", full_text)
    return leaked, clean


_UNTAGGED_THINKING_PREFIX_RE = re.compile(
    r"^(?:Thinking Process|思考プロセス|Let me think|考え|Let me analyze|Reasoning)",
    re.IGNORECASE,
)


def strip_untagged_thinking(text: str) -> tuple[str, str]:
    """Detect and strip an untagged thinking block from *text*.

    Some vLLM-hosted models output reasoning without ``<think>``/``</think>``
    tags, starting instead with a natural-language prefix like
    ``"Thinking Process:"`` or ``"Let me think"``.

    This helper splits the text at a paragraph boundary (triple newline
    preferred; double newline as fallback) into ``(thinking, response)``.

    Double-newline fallback is skipped when the trailing segment is more
    than 40 % of the total length (i.e. the boundary is likely mid-content,
    not a thinking/response separator).

    Returns ``("", text)`` when no thinking pattern is detected.
    """
    stripped = text.lstrip()
    if not _UNTAGGED_THINKING_PREFIX_RE.match(stripped):
        return ("", text)

    # Triple newline takes priority — unambiguous separator
    if "\n\n\n" in text:
        idx = text.index("\n\n\n")
        thinking = text[:idx]
        response = text[idx + 3 :]
        if thinking and response:
            return (thinking, response)

    # Fall back to double newline using the last occurrence
    if "\n\n" in text:
        idx = text.rfind("\n\n")
        thinking = text[:idx]
        response = text[idx + 2 :]
        # Skip when trailing segment dominates — it's probably continuation text
        if len(response) > len(text) * 0.4:
            return ("", text)
        if thinking and response:
            return (thinking, response)

    return ("", text)


_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
# Paragraphs shorter than this are deduplicated only against their immediate
# neighbour — a short line ("はい。", "---") may legitimately recur later on.
_DEDUP_MIN_PARAGRAPH_LEN = 8
# Markdown horizontal rule: meaningful between sections, noise at the end.
_RULE_ONLY_RE = re.compile(r"^(?:[-*_] ?){3,}$")


def join_answer_parts(parts: Iterable[str]) -> str:
    """Join answer fragments produced across tool-calling iterations.

    A single reply is often written across several assistant turns: text
    emitted alongside a tool call, then the final turn's text.  When a forced
    retry makes the model restate its answer, the same paragraphs arrive
    twice, so paragraphs already present are skipped.  Turns that end with a
    section rule would otherwise pile those rules up at the end of the merged
    answer, so trailing rules are dropped.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for part in parts:
        for paragraph in _PARAGRAPH_SPLIT_RE.split(part or ""):
            text = paragraph.strip()
            if not text:
                continue
            if len(text) >= _DEDUP_MIN_PARAGRAPH_LEN:
                if text in seen:
                    continue
                seen.add(text)
            elif merged and merged[-1] == text:
                continue
            merged.append(text)
    while merged and _RULE_ONLY_RE.match(merged[-1]):
        merged.pop()
    return "\n\n".join(merged)


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

    __slots__ = ("_buffer", "_done", "_saw_think_close")

    def __init__(self) -> None:
        self._buffer = ""
        self._done = False
        self._saw_think_close = False  # True only when </think> was processed in buffer

    _THINK_CLOSE_RE = re.compile(r"</think>\s*")

    def feed(self, delta: str) -> tuple[str, str]:
        """Process *delta* and return ``(thinking_text, response_text)``."""
        if self._done:
            if "</think>" in delta:
                if self._saw_think_close:
                    # A proper think block was already handled; strip orphan </think>
                    # (Qwen3.5 multi-block pattern).
                    delta = self._THINK_CLOSE_RE.sub("", delta)
                # else: early-exit path — leave </think> intact so the post-stream
                # strip_thinking_tags safety net can detect the leaked thinking.
            return ("", delta)
        self._buffer += delta

        # Check for </think> FIRST — handles both normal <think>...</think>
        # and the edge case where <think> is absent (vLLM reasoning parser).
        if "</think>" in self._buffer:
            self._done = True
            self._saw_think_close = True
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
    "Write": 500,
    "Edit": 500,
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
            when PreCompact blocks SDK auto-compact and the subsequent
            PreToolUse returns ``continue_=False``.
        task_compact_requested: When True, a task hit its configured context
            limit and should be compacted then resumed in the same SDK session.
        usage: Token usage for this session.  Populated by each executor.
        truncated: True when execution ended through runaway-guard
            finalization, external interruption, or another abnormal path
            rather than a normal final response.
    """

    text: str
    result_message: SessionResultLike | None = field(default=None, repr=False)
    replied_to_from_transcript: set[str] = field(default_factory=set)
    tool_call_records: list[ToolCallRecord] = field(default_factory=list)
    force_chain: bool = False
    task_compact_requested: bool = False
    usage: TokenUsage | None = None
    session_rotation_pending: bool = False
    truncated: bool = False
    error: bool = False
    reason: str = ""


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

    * S and C session state is managed externally by AgentCore.
      A saves threshold-triggered state for resumption on the next message.
    """

    session_engine: ClassVar[SessionEngine | None] = None
    tracks_sdk_session_state: ClassVar[bool] = False
    """Whether this engine owns an SDK session state tracked by ContextTracker."""
    saves_threshold_shortterm: ClassVar[bool] = False
    """Whether execute() saves threshold handoff state itself."""
    wants_structured_history: ClassVar[bool] = False
    """Whether blocking execution consumes structured prior conversation history."""

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        self._model_config = model_config
        self._anima_dir = anima_dir
        self._interrupt_event = interrupt_event
        self._task_cwd: Path | None = None
        self.reminder_queue: SystemReminderQueue = SystemReminderQueue()
        self._hb_soft_timeout_s: int = self._load_hb_soft_timeout()

    def set_task_cwd(self, cwd: Path | None) -> None:
        """Set override cwd for TaskExec sessions."""
        self._task_cwd = cwd

    async def aclear_session(self, trigger: str, thread_id: str = "default") -> None:
        """Asynchronously clear this executor's persisted session."""
        if self.session_engine is None:
            return
        from core.execution.session.engine_session import aclear_engine_session

        await aclear_engine_session(
            self._anima_dir,
            self.session_engine,
            resolve_runtime_session_type(trigger),
            thread_id,
        )

    def clear_session(self, trigger: str, thread_id: str = "default") -> None:
        """Synchronous compatibility adapter for local callers and tests."""
        if self.session_engine is None:
            return
        from core.execution.session.engine_session import clear_engine_session

        clear_engine_session(
            self._anima_dir,
            self.session_engine,
            resolve_runtime_session_type(trigger),
            thread_id,
        )

    def is_resumable(self, trigger: str) -> bool:
        """Return whether this engine can resume the trigger's session."""
        return self.session_engine is not None and is_resumable_trigger(trigger)

    def prepare_tracker(self, tracker: ContextTracker, system_prompt: str, prompt: str) -> None:
        """Prepare context accounting for engine-specific prompt/session state."""
        return None

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
        ``execute_streaming()`` — either token-level (S, A non-Ollama) or
        iteration-level (A Ollama, B).
        """
        return True

    @property
    def supports_message_injection(self) -> bool:
        """Whether an active streaming turn accepts additional user input."""
        return False

    async def inject_message(self, message: str) -> bool:
        """Inject a user message into the active stream when supported."""
        return False

    # -- Subordinate detection ----------------------------

    def _has_subordinates(self) -> bool:
        """Check if this anima has any subordinates (is a supervisor)."""
        try:
            from core.config import load_config

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
        if not self._model_config.api_key_env:
            return None
        return os.environ.get(self._model_config.api_key_env)

    def _read_replied_to_file(self) -> set[str]:
        """Read replied_to entries from persistent file."""
        import json as _json
        import logging

        from core.execution.session.session_context import current_runtime_session

        ctx = current_runtime_session()
        if ctx is not None:
            replied_to_path = (
                self._anima_dir / "run" / "replied_to" / ctx.session_type / f"{ctx.thread_id or 'default'}.jsonl"
            )
        else:
            replied_to_path = self._anima_dir / "run" / "replied_to" / "unknown" / "default.jsonl"
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

    def _resolve_num_retries(self) -> int:
        """Resolve LLM API retry count from ``config.server.llm_num_retries``."""
        try:
            from core.config import load_config

            return load_config().server.llm_num_retries
        except Exception:
            return 3

    def _resolve_cw(self) -> int:
        """Resolve context window (models.json SSoT → hardcoded fallback)."""
        from core.prompt.context import resolve_context_window

        return resolve_context_window(self._model_config.model)

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
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run the execution engine and return the response.

        Args:
            prompt: The user/trigger prompt.
            system_prompt: Assembled system prompt.
            tracker: Context usage tracker for monitoring window consumption.
            shortterm: Short-term memory for inline session chaining
                (A). S chaining is managed by AgentCore.
            trigger: Trigger identifier (e.g. "message:sakura", "heartbeat").
            images: Optional list of image dicts with ``data`` (base64) and
                ``media_type`` keys. Supported by S and A modes.

        Returns:
            ExecutionResult with the response text and optional metadata.
        """
        ...

    @stream_events
    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
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
            thread_id=thread_id,
        )
        yield text_delta_event(result.text)
        yield done_event(
            result.text,
            result_message=result.result_message,
            tool_call_records=[asdict(r) for r in result.tool_call_records],
            truncated=result.truncated,
        )
