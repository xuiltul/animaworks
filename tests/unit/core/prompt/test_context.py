"""Unit tests for core/prompt/context.py — ContextTracker and model resolution."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from core.prompt.context import (
    _DEFAULT_CONTEXT_WINDOW,
    _THRESHOLD_CEILING,
    CHARS_PER_TOKEN,
    MODEL_CONTEXT_WINDOWS,
    ContextTracker,
    _resolve_context_window,
    resolve_context_threshold,
    resolve_context_window,
)

_PATCH_TARGET = "core.config.model_mode._match_models_json"


# ── resolve_context_window SSoT (models.json) ────────────


class TestResolveContextWindowSSoT:
    """models.json is the SSoT and takes highest priority."""

    def test_models_json_claude_sonnet(self):
        entry = {"mode": "S", "context_window": 200_000}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("claude-sonnet-4-6") == 200_000

    def test_models_json_claude_opus(self):
        entry = {"mode": "S", "context_window": 200_000}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("claude-opus-4-6") == 200_000

    def test_models_json_codex_gpt54(self):
        entry = {"mode": "C", "context_window": 272_000}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("codex/gpt-5.4") == 272_000

    def test_models_json_grok_45(self):
        from core.paths import TEMPLATES_DIR

        models_path = TEMPLATES_DIR / "_shared" / "config_defaults" / "models.json"
        entry = json.loads(models_path.read_text(encoding="utf-8"))["grok/grok-4.5"]
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("grok/grok-4.5") == 500_000

    def test_models_json_mistral(self):
        entry = {"mode": "A", "context_window": 256_000}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("mistral/mistral-large") == 256_000

    def test_models_json_priority_over_overrides(self):
        """models.json wins even if overrides (config.json) would match."""
        entry = {"mode": "S", "context_window": 200_000}
        overrides = {"claude-*": 100_000}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("claude-sonnet-4-6", overrides) == 200_000

    def test_models_json_priority_over_hardcoded(self):
        """models.json wins over hardcoded MODEL_CONTEXT_WINDOWS."""
        entry = {"mode": "S", "context_window": 200_000}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("claude-opus-4-6") == 200_000

    def test_models_json_bad_context_window_falls_through(self):
        """Non-numeric context_window in models.json falls through to overrides."""
        entry = {"mode": "S", "context_window": "invalid"}
        overrides = {"claude-*": 100_000}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("claude-sonnet-4-6", overrides) == 100_000

    def test_models_json_missing_context_window_falls_through(self):
        """models.json entry without context_window falls through."""
        entry = {"mode": "S"}
        with patch(_PATCH_TARGET, return_value=entry):
            assert resolve_context_window("claude-opus-4-6") == MODEL_CONTEXT_WINDOWS.get(
                "claude-opus-4-6", _DEFAULT_CONTEXT_WINDOW
            )


# ── resolve_context_window fallback (no models.json) ─────


class TestResolveContextWindowFallback:
    """When models.json has no match, fall through to overrides → hardcoded."""

    def test_claude_sonnet_hardcoded(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("claude-sonnet-4-20250514") == 200_000

    def test_claude_opus_hardcoded(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("claude-opus-4-6") == 128_000

    def test_gpt4o_hardcoded(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("gpt-4o") == 128_000

    def test_gemini_flash_hardcoded(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("gemini-2.0-flash") == 1_048_576

    def test_unknown_model(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("unknown-model") == _DEFAULT_CONTEXT_WINDOW

    def test_provider_prefix_stripped(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("openai/gpt-4o") == 128_000
            assert _resolve_context_window("anthropic/claude-sonnet-4") == 200_000
            assert _resolve_context_window("google/gemini-2.5-pro") == 1_048_576

    def test_bedrock_cross_region_prefix_stripped(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("bedrock/jp.anthropic.claude-sonnet-4-6") == 128_000
            assert _resolve_context_window("bedrock/us.anthropic.claude-sonnet-4-6") == 128_000
            assert _resolve_context_window("bedrock/eu.anthropic.claude-opus-4-6") == 128_000
            assert _resolve_context_window("bedrock/claude-sonnet-4-6") == 128_000

    def test_all_known_models(self):
        with patch(_PATCH_TARGET, return_value=None):
            for prefix, expected in MODEL_CONTEXT_WINDOWS.items():
                assert _resolve_context_window(prefix) == expected


# ── resolve_context_window with overrides (deprecated) ───


class TestResolveContextWindowOverrides:
    """Config-driven overrides take priority over hardcoded defaults (when models.json has no match)."""

    def test_override_takes_priority(self):
        overrides = {"claude-sonnet-4*": 100_000}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("claude-sonnet-4-20250514", overrides) == 100_000

    def test_override_with_provider_prefix(self):
        overrides = {"openai/glm-4*": 16_384}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("openai/glm-4.7-flash", overrides) == 16_384

    def test_override_bare_pattern_matches_provider_model(self):
        overrides = {"glm-4*": 16_384}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("openai/glm-4.7-flash", overrides) == 16_384

    def test_fallback_to_hardcoded_when_no_override_match(self):
        overrides = {"some-custom-model*": 4096}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("claude-sonnet-4-20250514", overrides) == 200_000

    def test_fallback_to_default_when_nothing_matches(self):
        overrides = {"some-custom-model*": 4096}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("totally-unknown-model", overrides) == _DEFAULT_CONTEXT_WINDOW

    def test_empty_overrides_dict(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("claude-sonnet-4-20250514", {}) == 200_000

    def test_none_overrides(self):
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("claude-sonnet-4-20250514", None) == 200_000

    def test_exact_match_override(self):
        overrides = {"gpt-4o": 256_000}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("gpt-4o", overrides) == 256_000

    def test_wildcard_star(self):
        overrides = {"ollama/*": 8192}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("ollama/my-custom-model", overrides) == 8192

    def test_first_matching_pattern_wins(self):
        overrides = {"glm-4*": 16_384, "glm-*": 32_000}
        with patch(_PATCH_TARGET, return_value=None):
            assert _resolve_context_window("glm-4.7-flash", overrides) == 16_384


# ── resolve_context_threshold ─────────────────────────────


class TestResolveContextThreshold:
    """Auto-scaling of compaction threshold based on context window size."""

    def test_large_model_unchanged(self):
        assert resolve_context_threshold(0.50, 200_000) == 0.50
        assert resolve_context_threshold(0.50, 1_000_000) == 0.50
        assert resolve_context_threshold(0.50, 2_000_000) == 0.50

    def test_small_model_scales_up(self):
        result = resolve_context_threshold(0.50, 128_000)
        assert result > 0.50
        assert result < _THRESHOLD_CEILING

    def test_very_small_model_near_ceiling(self):
        result = resolve_context_threshold(0.50, 30_000)
        assert result >= 0.85

    def test_ceiling_not_exceeded(self):
        result = resolve_context_threshold(0.10, 1_000)
        assert result <= _THRESHOLD_CEILING

    def test_sliding_scale_monotonic(self):
        windows = [30_000, 128_000, 200_000, 500_000, 1_000_000]
        thresholds = [resolve_context_threshold(0.50, w) for w in windows]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1]

    def test_expected_values(self):
        assert resolve_context_threshold(0.50, 1_000_000) == 0.50
        assert resolve_context_threshold(0.50, 200_000) == 0.50
        assert resolve_context_threshold(0.50, 128_000) == pytest.approx(0.673, abs=0.01)
        assert resolve_context_threshold(0.50, 30_000) == pytest.approx(0.908, abs=0.01)


# ── ContextTracker init ───────────────────────────────────


class TestContextTrackerInit:
    def test_defaults(self):
        entry = {"mode": "S", "context_window": 200_000}
        with patch(_PATCH_TARGET, return_value=entry):
            ct = ContextTracker()
            assert ct.model == "claude-sonnet-4-6"
            assert ct.threshold == 0.50
            assert ct.context_window_overrides == {}

    def test_custom(self):
        entry = {"mode": "S", "context_window": 200_000}
        with patch(_PATCH_TARGET, return_value=entry):
            ct = ContextTracker(model="claude-sonnet-4-6", threshold=0.70)
            assert ct.model == "claude-sonnet-4-6"
            assert ct.threshold == 0.70

    def test_custom_with_overrides(self):
        with patch(_PATCH_TARGET, return_value=None):
            overrides = {"glm-4*": 16_384}
            ct = ContextTracker(
                model="openai/glm-4.7-flash",
                context_window_overrides=overrides,
            )
            assert ct.context_window_overrides == overrides
            assert ct.context_window == 16_384

    def test_threshold_auto_scaled_for_small_model(self):
        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(
                model="openai/glm-4.7-flash",
                threshold=0.50,
                context_window_overrides={"glm-4*": 30_000},
            )
            assert ct.threshold > 0.90

    def test_threshold_auto_scaled_for_128k_model(self):
        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(model="claude-sonnet-4-6", threshold=0.50)
            assert ct.threshold > 0.50
            assert ct.threshold < 0.80

    def test_threshold_unchanged_for_200k_model(self):
        """200K model from models.json keeps configured threshold."""
        entry = {"mode": "S", "context_window": 200_000}
        with patch(_PATCH_TARGET, return_value=entry):
            ct = ContextTracker(model="claude-sonnet-4-6", threshold=0.50)
            assert ct.threshold == 0.50


class TestContextTrackerProperties:
    def test_context_window(self):
        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(model="claude-sonnet-4-20250514")
            assert ct.context_window == 200_000

    def test_context_window_from_models_json(self):
        entry = {"mode": "S", "context_window": 200_000}
        with patch(_PATCH_TARGET, return_value=entry):
            ct = ContextTracker(model="claude-opus-4-6")
            assert ct.context_window == 200_000

    def test_context_window_with_overrides(self):
        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(
                model="claude-sonnet-4-20250514",
                context_window_overrides={"claude-sonnet-4*": 100_000},
            )
            assert ct.context_window == 100_000

    def test_context_window_override_empty_dict(self):
        with patch(_PATCH_TARGET, return_value=None):
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
        f = tmp_path / "transcript.json"
        content = "x" * 40_000  # 40000 chars / 4 = 10000 tokens
        f.write_text(content)

        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(model="claude-sonnet-4-20250514")  # 200k hardcoded
            ratio = ct.estimate_from_transcript(str(f))
            expected = 10_000 / 200_000
            assert abs(ratio - expected) < 0.01

    def test_threshold_detection(self, tmp_path):
        f = tmp_path / "big.json"
        f.write_text("x" * 400_000)

        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(model="claude-sonnet-4-6", threshold=0.50)
            ratio = ct.estimate_from_transcript(str(f))
            assert ratio >= ct.threshold
            assert ct.threshold_exceeded is True

    def test_threshold_only_triggers_once(self, tmp_path):
        f = tmp_path / "big.json"
        ct = ContextTracker(threshold=0.50)
        needed_tokens = int(ct.context_window * (ct.threshold + 0.05))
        f.write_text("x" * (needed_tokens * CHARS_PER_TOKEN))

        ct.estimate_from_transcript(str(f))
        assert ct.threshold_exceeded is True
        ct.estimate_from_transcript(str(f))
        assert ct.threshold_exceeded is True


# ── update_from_usage ─────────────────────────────────────


class TestUpdateFromUsage:
    def test_basic_usage(self):
        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(model="claude-sonnet-4-20250514")
            result = ct.update_from_usage({"input_tokens": 50_000, "output_tokens": 5_000})
            assert ct.usage_ratio == pytest.approx(50_000 / 200_000, abs=0.001)
            assert result is False

    def test_threshold_crossed(self):
        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(model="claude-sonnet-4-6", threshold=0.50)
            result = ct.update_from_usage({"input_tokens": 90_000, "output_tokens": 10_000})
            assert result is True
            assert ct.threshold_exceeded is True

    def test_threshold_not_crossed_twice(self):
        ct = ContextTracker(threshold=0.50)
        over_threshold = int(ct.context_window * (ct.threshold + 0.05))
        ct.update_from_usage({"input_tokens": over_threshold, "output_tokens": 10_000})
        assert ct.threshold_exceeded is True
        result = ct.update_from_usage({"input_tokens": over_threshold + 10_000, "output_tokens": 10_000})
        assert result is False

    def test_empty_usage(self):
        ct = ContextTracker()
        ct.update_from_usage({})
        assert ct.usage_ratio == 0.0


# ── update_from_result_message ────────────────────────────


class TestUpdateFromResultMessage:
    def test_with_usage(self):
        with patch(_PATCH_TARGET, return_value=None):
            ct = ContextTracker(model="claude-sonnet-4-20250514")
            ct.update_from_result_message({"input_tokens": 80_000, "output_tokens": 20_000})
            expected_ratio = (80_000 + 20_000) / 200_000
            assert ct.usage_ratio == pytest.approx(expected_ratio, abs=0.001)

    def test_threshold_detection(self):
        ct = ContextTracker(threshold=0.40)
        over_threshold = int(ct.context_window * (ct.threshold + 0.05))
        ct.update_from_result_message({"input_tokens": over_threshold, "output_tokens": 10_000})
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
        over_threshold = int(ct.context_window * (ct.threshold + 0.05))
        ct.update_from_usage({"input_tokens": over_threshold, "output_tokens": 10_000})
        assert ct.threshold_exceeded is True
        ct.reset()
        assert ct.usage_ratio == 0.0
        assert ct.threshold_exceeded is False
        assert ct._input_tokens == 0
        assert ct._output_tokens == 0
