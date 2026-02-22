# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Context window usage tracker.

Monitors token consumption and detects when the 50% threshold is crossed.
Uses transcript file size as a heuristic for the Agent SDK path,
and direct usage data from the Anthropic SDK fallback path.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("animaworks.context_tracker")

# Approximate characters per token (tunable constant).
# JSON transcripts are verbose, so 4 chars/token is a reasonable estimate.
CHARS_PER_TOKEN = 4

# Context window sizes per model family (input tokens).
# Keys are matched as prefixes against the model name (after stripping provider/).
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-sonnet-4": 200_000,
    "claude-sonnet-3.5": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-3.5": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "o1": 200_000,
    "o3": 200_000,
    # Google
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    # GLM (THUDM)
    "glm-4": 131_072,
    # Ollama / local (conservative defaults)
    "gemma3": 128_000,
    "llama3": 128_000,
    "qwen2.5": 128_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000


def resolve_context_window(
    model: str,
    overrides: dict[str, int] | None = None,
) -> int:
    """Return the context window size for the given model name.

    Resolution priority:
      1. ``overrides`` (config-driven, fnmatch wildcard)
      2. ``MODEL_CONTEXT_WINDOWS`` (hardcoded, prefix match)
      3. ``_DEFAULT_CONTEXT_WINDOW`` (128K fallback)

    Strips the ``provider/`` prefix (e.g. ``openai/gpt-4o`` -> ``gpt-4o``)
    before matching.
    """
    bare = model.split("/", 1)[-1] if "/" in model else model
    # Phase 1: config overrides (fnmatch wildcard)
    if overrides:
        for pattern, size in overrides.items():
            if fnmatch.fnmatch(model, pattern) or fnmatch.fnmatch(bare, pattern):
                return size
    # Phase 2: hardcoded defaults (prefix match)
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if bare.startswith(prefix):
            return size
    return _DEFAULT_CONTEXT_WINDOW

# Backward-compatible alias (deprecated)
_resolve_context_window = resolve_context_window


@dataclass
class ContextTracker:
    """Tracks context window usage across an agent session.

    Two estimation modes:
      1. Transcript-based (Agent SDK): file size of transcript_path / CHARS_PER_TOKEN
      2. Usage-based (Anthropic SDK / ResultMessage): direct input_tokens from API
    """

    model: str = ""
    threshold: float = 0.50
    context_window_overrides: dict[str, int] = field(default_factory=dict)

    # Internal state
    _last_ratio: float = field(default=0.0, init=False, repr=False)
    _threshold_hit: bool = field(default=False, init=False, repr=False)
    _input_tokens: int = field(default=0, init=False, repr=False)
    _output_tokens: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model:
            from core.config.models import AnimaDefaults
            self.model = AnimaDefaults().model

    @property
    def context_window(self) -> int:
        return resolve_context_window(
            self.model, self.context_window_overrides or None
        )

    @property
    def usage_ratio(self) -> float:
        return self._last_ratio

    @property
    def threshold_exceeded(self) -> bool:
        return self._threshold_hit

    def force_threshold(self) -> None:
        """Force the threshold flag for external triggers (e.g. A1 auto-compact)."""
        if not self._threshold_hit:
            self._threshold_hit = True

    # ── Transcript-based estimation (Agent SDK) ────────────

    def estimate_from_transcript(self, transcript_path: str) -> float:
        """Estimate context usage ratio from transcript file size.

        Returns the estimated ratio (0.0-1.0+).
        """
        if not transcript_path:
            return self._last_ratio
        try:
            file_size = os.path.getsize(transcript_path)
        except OSError:
            return self._last_ratio

        estimated_tokens = file_size // CHARS_PER_TOKEN
        ratio = estimated_tokens / self.context_window
        self._last_ratio = ratio

        if not self._threshold_hit and ratio >= self.threshold:
            self._threshold_hit = True
            logger.warning(
                "Context threshold %.0f%% exceeded (transcript estimate): "
                "~%d tokens / %d window (%.1f%%)",
                self.threshold * 100,
                estimated_tokens,
                self.context_window,
                ratio * 100,
            )

        return ratio

    # ── Unified usage update ────────────────────────────────

    def update(
        self,
        usage: dict | None,
        *,
        include_output_in_ratio: bool = False,
    ) -> bool:
        """Unified context-usage update for any provider's usage dict.

        Args:
            usage: Dict with ``input_tokens`` and ``output_tokens`` keys.
                Accepts usage dicts from the Anthropic API, LiteLLM
                (``prompt_tokens`` / ``completion_tokens``), or Agent SDK
                ``ResultMessage.usage``.  ``None`` is a safe no-op.
            include_output_in_ratio: If True, the ratio is computed as
                ``(input + output) / window``.  This is appropriate for
                Agent SDK ``ResultMessage.usage`` where the snapshot
                represents the full session cost.  When False (default),
                only ``input_tokens`` is used -- correct for per-request
                usage where output tokens from prior turns are already
                folded into the next request's ``input_tokens``.

        Returns:
            True if the threshold was *newly* crossed by this update.
        """
        if not usage:
            return False

        # Normalise keys: LiteLLM uses prompt_tokens / completion_tokens
        self._input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        self._output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )

        numerator = (
            (self._input_tokens + self._output_tokens)
            if include_output_in_ratio
            else self._input_tokens
        )
        self._last_ratio = (
            numerator / self.context_window if self.context_window else 0.0
        )

        newly_crossed = False
        if not self._threshold_hit and self._last_ratio >= self.threshold:
            self._threshold_hit = True
            newly_crossed = True
            logger.warning(
                "Context threshold %.0f%% exceeded: "
                "%d tokens / %d window (%.1f%%)",
                self.threshold * 100,
                numerator,
                self.context_window,
                self._last_ratio * 100,
            )
        return newly_crossed

    # ── Legacy convenience methods (delegate to update()) ─

    def update_from_usage(self, usage: dict) -> bool:
        """Update from per-request API usage (A2 / Fallback).

        Uses ``input_tokens`` alone as the fullness measure because output
        tokens from prior turns are already included in the next request's
        ``input_tokens``.

        Returns True if the threshold was newly crossed.
        """
        return self.update(usage, include_output_in_ratio=False)

    def update_from_result_message(self, usage: dict | None) -> None:
        """Update from Agent SDK ``ResultMessage.usage`` (post-session snapshot).

        Uses ``input_tokens + output_tokens`` because the snapshot
        represents the total session cost.
        """
        self.update(usage, include_output_in_ratio=True)

    def reset(self) -> None:
        """Reset tracker for a new session."""
        self._last_ratio = 0.0
        self._threshold_hit = False
        self._input_tokens = 0
        self._output_tokens = 0
