from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adaptive thinking helpers, resolve_max_tokens, and schema extensions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.execution.base import is_adaptive_model, is_anthropic_claude, resolve_thinking_effort
from core.config.models import (
    DEFAULT_MAX_TOKENS,
    AnimaDefaults,
    AnimaWorksConfig,
    resolve_max_tokens,
)
from core.schemas import ModelConfig


# ── is_adaptive_model ─────────────────────────────────────────


class TestIsAdaptiveModel:
    def test_opus_46(self):
        assert is_adaptive_model("claude-opus-4-6") is True

    def test_sonnet_46(self):
        assert is_adaptive_model("claude-sonnet-4-6") is True

    def test_with_provider_prefix(self):
        assert is_adaptive_model("anthropic/claude-opus-4-6") is True

    def test_old_sonnet(self):
        assert is_adaptive_model("claude-sonnet-4-5-20250929") is False

    def test_non_claude(self):
        assert is_adaptive_model("openai/gpt-4o") is False

    def test_ollama(self):
        assert is_adaptive_model("ollama/qwen3:14b") is False


# ── is_anthropic_claude ───────────────────────────────────────


class TestIsAnthropicClaude:
    def test_claude_model(self):
        assert is_anthropic_claude("claude-sonnet-4-6") is True

    def test_with_prefix(self):
        assert is_anthropic_claude("anthropic/claude-opus-4-6") is True

    def test_non_claude(self):
        assert is_anthropic_claude("openai/gpt-4o") is False

    def test_ollama(self):
        assert is_anthropic_claude("ollama/qwen3:14b") is False


# ── resolve_thinking_effort ───────────────────────────────────


class TestResolveThinkingEffort:
    def test_default_high(self):
        assert resolve_thinking_effort("claude-opus-4-6", None) == "high"

    def test_explicit_medium(self):
        assert resolve_thinking_effort("claude-opus-4-6", "medium") == "medium"

    def test_max_on_opus(self):
        assert resolve_thinking_effort("claude-opus-4-6", "max") == "max"

    def test_max_clamped_on_non_opus(self):
        assert resolve_thinking_effort("claude-sonnet-4-6", "max") == "high"

    def test_low(self):
        assert resolve_thinking_effort("claude-sonnet-4-6", "low") == "low"


# ── resolve_max_tokens ────────────────────────────────────────


class TestResolveMaxTokens:
    def test_explicit_high_value_preserved(self):
        result = resolve_max_tokens("claude-sonnet-4-6", 32000, True)
        assert result == 32000

    def test_default_when_no_thinking(self):
        result = resolve_max_tokens("openai/gpt-4o", None, None)
        assert result == DEFAULT_MAX_TOKENS

    def test_thinking_raises_minimum(self):
        result = resolve_max_tokens("claude-sonnet-4-6", None, True)
        assert result == 16384

    def test_pattern_match_from_config(self):
        config = AnimaWorksConfig(model_max_tokens={"claude-opus-*": 65536})
        result = resolve_max_tokens("claude-opus-4-6", None, None, config)
        assert result == 65536

    def test_pattern_match_overrides_thinking_default(self):
        config = AnimaWorksConfig(model_max_tokens={"claude-*": 32000})
        result = resolve_max_tokens("claude-sonnet-4-6", None, True, config)
        assert result == 32000

    def test_explicit_8192_treated_as_default(self):
        """When explicit value equals DEFAULT_MAX_TOKENS, thinking may override."""
        result = resolve_max_tokens("claude-sonnet-4-6", 8192, True)
        assert result == 16384

    def test_thinking_raises_low_explicit(self):
        """Thinking minimum floor raises values below 16384."""
        result = resolve_max_tokens("claude-sonnet-4-6", 1024, True)
        assert result == 16384


# ── ModelConfig schema ────────────────────────────────────────


class TestModelConfigSchema:
    def test_thinking_effort_field_exists(self):
        cfg = ModelConfig(thinking_effort="medium")
        assert cfg.thinking_effort == "medium"

    def test_thinking_effort_defaults_none(self):
        cfg = ModelConfig()
        assert cfg.thinking_effort is None


# ── AnimaDefaults schema ──────────────────────────────────────


class TestAnimaDefaultsSchema:
    def test_thinking_effort_field_exists(self):
        defaults = AnimaDefaults(thinking_effort="low")
        assert defaults.thinking_effort == "low"

    def test_thinking_effort_defaults_none(self):
        defaults = AnimaDefaults()
        assert defaults.thinking_effort is None


# ── AnimaWorksConfig.model_max_tokens ─────────────────────────


class TestModelMaxTokensConfig:
    def test_model_max_tokens_default_empty(self):
        config = AnimaWorksConfig()
        assert config.model_max_tokens == {}

    def test_model_max_tokens_roundtrip(self):
        config = AnimaWorksConfig(model_max_tokens={"claude-*": 32000})
        data = config.model_dump()
        restored = AnimaWorksConfig.model_validate(data)
        assert restored.model_max_tokens == {"claude-*": 32000}


# ── status.json loading ──────────────────────────────────────


class TestStatusJsonThinkingEffort:
    def test_thinking_effort_loaded_from_status(self, tmp_path):
        """Verify _load_status_json includes thinking_effort."""
        import json
        from core.config.models import _load_status_json

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        status = {"model": "claude-opus-4-6", "thinking": True, "thinking_effort": "medium"}
        (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

        result = _load_status_json(anima_dir)
        assert result.get("thinking_effort") == "medium"

    def test_thinking_effort_absent_is_omitted(self, tmp_path):
        """When thinking_effort is absent from status.json, key is absent from result."""
        import json
        from core.config.models import _load_status_json

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        status = {"model": "claude-opus-4-6", "thinking": True}
        (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

        result = _load_status_json(anima_dir)
        assert "thinking_effort" not in result


# ── LiteLLM adaptive thinking kwargs ─────────────────────────


class TestLiteLLMAdaptiveThinking:
    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test"
        d.mkdir(parents=True)
        (d / "permissions.md").write_text("", encoding="utf-8")
        (d / "identity.md").write_text("# Test", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm"]:
            (d / sub).mkdir(exist_ok=True)
        return d

    @pytest.fixture
    def memory(self, anima_dir: Path) -> MagicMock:
        from core.memory import MemoryManager
        m = MagicMock(spec=MemoryManager)
        m.read_permissions.return_value = ""
        m.search_memory_text.return_value = []
        m.anima_dir = anima_dir
        return m

    @pytest.fixture
    def tool_handler(self) -> MagicMock:
        from core.tooling.handler import ToolHandler
        return MagicMock(spec=ToolHandler)

    def test_claude_46_gets_adaptive_thinking(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor
        cfg = ModelConfig(
            model="claude-sonnet-4-6", thinking=True,
            thinking_effort="medium", api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg, anima_dir=anima_dir,
            tool_handler=tool_handler, tool_registry=[], memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["temperature"] == 1

    def test_old_claude_gets_manual_thinking(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor
        cfg = ModelConfig(
            model="claude-sonnet-4-5-20250929", thinking=True, api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg, anima_dir=anima_dir,
            tool_handler=tool_handler, tool_registry=[], memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert kwargs["temperature"] == 1

    def test_non_claude_gets_think_param(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor
        cfg = ModelConfig(
            model="openai/gpt-4o", thinking=True, api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg, anima_dir=anima_dir,
            tool_handler=tool_handler, tool_registry=[], memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs.get("think") is True
        assert "thinking" not in kwargs

    def test_bedrock_gets_reasoning_effort(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor
        cfg = ModelConfig(
            model="bedrock/claude-opus-4-6", thinking=True,
            thinking_effort="medium", api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg, anima_dir=anima_dir,
            tool_handler=tool_handler, tool_registry=[], memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["reasoning_effort"] == "medium"
        assert "thinking" not in kwargs
