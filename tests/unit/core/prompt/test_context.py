"""Unit tests for core/prompt/context.py — ContextTracker and model resolution."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.prompt.context import (
    CHARS_PER_TOKEN,
    MODEL_CONTEXT_WINDOWS,
    ContextTracker,
    _DEFAULT_CONTEXT_WINDOW,
    _resolve_context_window,
)


# ── _resolve_context_window ───────────────────────────────


class TestResolveContextWindow:
    def test_claude_sonnet(self):
        assert _resolve_context_window("claude-sonnet-4-20250514") == 200_000

    def test_claude_opus(self):
        assert _resolve_context_window("claude-opus-4-20250514") == 200_000

    def test_gpt4o(self):
        assert _resolve_context_window("gpt-4o") == 128_000

    def test_gemini_flash(self):
        assert _resolve_context_window("gemini-2.0-flash") == 1_048_576

    def test_unknown_model(self):
        assert _resolve_context_window("unknown-model") == _DEFAULT_CONTEXT_WINDOW

    def test_provider_prefix_stripped(self):
        assert _resolve_context_window("openai/gpt-4o") == 128_000
        assert _resolve_context_window("anthropic/claude-sonnet-4") == 200_000
        assert _resolve_context_window("google/gemini-2.5-pro") == 1_048_576

    def test_all_known_models(self):
        for prefix, expected in MODEL_CONTEXT_WINDOWS.items():
            assert _resolve_context_window(prefix) == expected


# ── _resolve_context_window with overrides ──────────────


class TestResolveContextWindowOverrides:
    """Config-driven overrides take priority over hardcoded defaults."""

    def test_override_takes_priority(self):
        """Config override should win over hardcoded MODEL_CONTEXT_WINDOWS."""
        overrides = {"claude-sonnet-4*": 100_000}
        assert _resolve_context_window("claude-sonnet-4-20250514", overrides) == 100_000

    def test_override_with_provider_prefix(self):
        """Override should match both full model name and bare name."""
        overrides = {"openai/glm-4*": 16_384}
        assert _resolve_context_window("openai/glm-4.7-flash", overrides) == 16_384

    def test_override_bare_pattern_matches_provider_model(self):
        """A bare pattern (no provider/) should match after prefix stripping."""
        overrides = {"glm-4*": 16_384}
        assert _resolve_context_window("openai/glm-4.7-flash", overrides) == 16_384

    def test_fallback_to_hardcoded_when_no_override_match(self):
        """When overrides don't match, fall back to hardcoded defaults."""
        overrides = {"some-custom-model*": 4096}
        assert _resolve_context_window("claude-sonnet-4-20250514", overrides) == 200_000

    def test_fallback_to_default_when_nothing_matches(self):
        """When neither overrides nor hardcoded match, use _DEFAULT_CONTEXT_WINDOW."""
        overrides = {"some-custom-model*": 4096}
        assert _resolve_context_window("totally-unknown-model", overrides) == _DEFAULT_CONTEXT_WINDOW

    def test_empty_overrides_dict(self):
        """Empty overrides dict should behave like no overrides."""
        assert _resolve_context_window("claude-sonnet-4-20250514", {}) == 200_000

    def test_none_overrides(self):
        """None overrides should behave like no overrides."""
        assert _resolve_context_window("claude-sonnet-4-20250514", None) == 200_000

    def test_exact_match_override(self):
        """Exact model name in overrides should match."""
        overrides = {"gpt-4o": 256_000}
        assert _resolve_context_window("gpt-4o", overrides) == 256_000

    def test_wildcard_star(self):
        """Wildcard * should match multiple characters."""
        overrides = {"ollama/*": 8192}
        assert _resolve_context_window("ollama/my-custom-model", overrides) == 8192

    def test_first_matching_pattern_wins(self):
        """When multiple overrides match, the first one (dict insertion order) wins."""
        overrides = {"glm-4*": 16_384, "glm-*": 32_000}
        assert _resolve_context_window("glm-4.7-flash", overrides) == 16_384


# ── ContextTracker init ───────────────────────────────────


class TestContextTrackerInit:
    def test_defaults(self):
        ct = ContextTracker()
        assert ct.model == "claude-sonnet-4-20250514"
        assert ct.threshold == 0.50
        assert ct.context_window_overrides == {}

    def test_custom(self):
        ct = ContextTracker(model="gpt-4o", threshold=0.70)
        assert ct.model == "gpt-4o"
        assert ct.threshold == 0.70

    def test_custom_with_overrides(self):
        overrides = {"glm-4*": 16_384}
        ct = ContextTracker(
            model="openai/glm-4.7-flash",
            context_window_overrides=overrides,
        )
        assert ct.context_window_overrides == overrides
        assert ct.context_window == 16_384


class TestContextTrackerProperties:
    def test_context_window(self):
        ct = ContextTracker(model="claude-sonnet-4-20250514")
        assert ct.context_window == 200_000

    def test_context_window_with_overrides(self):
        ct = ContextTracker(
            model="claude-sonnet-4-20250514",
            context_window_overrides={"claude-sonnet-4*": 100_000},
        )
        assert ct.context_window == 100_000

    def test_context_window_override_empty_dict(self):
        """Empty overrides should fall through to hardcoded defaults."""
        ct = ContextTracker(model="gpt-4o", context_window_overrides={})
        assert ct.context_window == 128_000

    def test_usage_ratio_initial(self):
        ct = ContextTracker()
        assert ct.usage_ratio == 0.0

    def test_threshold_exceeded_initial(self):
        ct = ContextTracker()
        assert ct.threshold_exceeded is False


# ── estimate_from_transcript ──────────────────────────────


class TestEstimateFromTranscript:
    def test_empty_path(self):
        ct = ContextTracker()
        ratio = ct.estimate_from_transcript("")
        assert ratio == 0.0

    def test_nonexistent_file(self):
        ct = ContextTracker()
        ratio = ct.estimate_from_transcript("/nonexistent/path.txt")
        assert ratio == 0.0

    def test_real_file(self, tmp_path):
        # Create a file of known size
        f = tmp_path / "transcript.json"
        content = "x" * 40_000  # 40000 chars / 4 = 10000 tokens
        f.write_text(content)

        ct = ContextTracker(model="claude-sonnet-4-20250514")  # 200k window
        ratio = ct.estimate_from_transcript(str(f))
        expected = 10_000 / 200_000  # 0.05
        assert abs(ratio - expected) < 0.01

    def test_threshold_detection(self, tmp_path):
        # Create a large file to exceed threshold
        f = tmp_path / "big.json"
        # 200k window * 0.50 threshold = 100k tokens needed
        # 100k tokens * 4 chars = 400k chars
        f.write_text("x" * 500_000)

        ct = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.50)
        ratio = ct.estimate_from_transcript(str(f))
        assert ratio >= 0.50
        assert ct.threshold_exceeded is True

    def test_threshold_only_triggers_once(self, tmp_path):
        f = tmp_path / "big.json"
        f.write_text("x" * 500_000)

        ct = ContextTracker(threshold=0.50)
        ct.estimate_from_transcript(str(f))
        assert ct.threshold_exceeded is True
        # Second call shouldn't change the flag
        ct.estimate_from_transcript(str(f))
        assert ct.threshold_exceeded is True


# ── update_from_usage ─────────────────────────────────────


class TestUpdateFromUsage:
    def test_basic_usage(self):
        ct = ContextTracker(model="claude-sonnet-4-20250514")
        result = ct.update_from_usage({"input_tokens": 50_000, "output_tokens": 5_000})
        assert ct.usage_ratio == pytest.approx(50_000 / 200_000, abs=0.001)
        assert result is False  # didn't cross threshold

    def test_threshold_crossed(self):
        ct = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.50)
        result = ct.update_from_usage({"input_tokens": 110_000, "output_tokens": 10_000})
        assert result is True
        assert ct.threshold_exceeded is True

    def test_threshold_not_crossed_twice(self):
        ct = ContextTracker(threshold=0.50)
        ct.update_from_usage({"input_tokens": 110_000, "output_tokens": 10_000})
        assert ct.threshold_exceeded is True
        result = ct.update_from_usage({"input_tokens": 120_000, "output_tokens": 10_000})
        assert result is False  # already triggered

    def test_empty_usage(self):
        ct = ContextTracker()
        ct.update_from_usage({})
        assert ct.usage_ratio == 0.0


# ── update_from_result_message ────────────────────────────


class TestUpdateFromResultMessage:
    def test_with_usage(self):
        ct = ContextTracker(model="claude-sonnet-4-20250514")
        ct.update_from_result_message({"input_tokens": 80_000, "output_tokens": 20_000})
        expected_ratio = (80_000 + 20_000) / 200_000
        assert ct.usage_ratio == pytest.approx(expected_ratio, abs=0.001)

    def test_threshold_detection(self):
        ct = ContextTracker(threshold=0.40)
        ct.update_from_result_message({"input_tokens": 80_000, "output_tokens": 20_000})
        assert ct.threshold_exceeded is True

    def test_none_usage(self):
        ct = ContextTracker()
        ct.update_from_result_message(None)
        assert ct.usage_ratio == 0.0

    def test_empty_dict(self):
        ct = ContextTracker()
        ct.update_from_result_message({})
        assert ct.usage_ratio == 0.0


# ── reset ─────────────────────────────────────────────────


class TestReset:
    def test_resets_all(self):
        ct = ContextTracker()
        ct.update_from_usage({"input_tokens": 150_000, "output_tokens": 10_000})
        assert ct.threshold_exceeded is True
        ct.reset()
        assert ct.usage_ratio == 0.0
        assert ct.threshold_exceeded is False
        assert ct._input_tokens == 0
        assert ct._output_tokens == 0
