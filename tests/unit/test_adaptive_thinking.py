from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adaptive thinking helpers, resolve_max_tokens, and schema extensions."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config.models import (
    DEFAULT_MAX_TOKENS,
    AnimaDefaults,
    AnimaWorksConfig,
    resolve_max_tokens,
)
from core.execution.base import (
    is_adaptive_model,
    is_anthropic_claude,
    is_bedrock_glm,
    is_bedrock_kimi,
    is_bedrock_qwen,
    resolve_thinking_effort,
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

    def test_bedrock_region_prefix(self):
        assert is_adaptive_model("bedrock/jp.anthropic.claude-sonnet-4-6") is True

    def test_bedrock_region_prefix_opus(self):
        assert is_adaptive_model("bedrock/us.anthropic.claude-opus-4-6") is True

    def test_bedrock_without_region(self):
        assert is_adaptive_model("bedrock/claude-sonnet-4-6") is True

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

    def test_bedrock_region_prefix(self):
        assert is_anthropic_claude("bedrock/jp.anthropic.claude-sonnet-4-6") is True

    def test_bedrock_region_prefix_eu(self):
        assert is_anthropic_claude("bedrock/eu.anthropic.claude-opus-4-6") is True

    def test_non_claude(self):
        assert is_anthropic_claude("openai/gpt-4o") is False

    def test_ollama(self):
        assert is_anthropic_claude("ollama/qwen3:14b") is False


# ── is_bedrock_qwen ───────────────────────────────────────────


class TestIsBedrockQwen:
    def test_bedrock_qwen_model(self):
        assert is_bedrock_qwen("bedrock/qwen.qwen3-next-80b-a3b") is True

    def test_bedrock_qwen_32b(self):
        assert is_bedrock_qwen("bedrock/qwen.qwen3-32b-instruct") is True

    def test_bedrock_qwen_vl(self):
        assert is_bedrock_qwen("bedrock/qwen.qwen3-vl-235b-a22b") is True

    def test_bedrock_claude_not_qwen(self):
        assert is_bedrock_qwen("bedrock/claude-opus-4-6") is False

    def test_bedrock_region_claude(self):
        assert is_bedrock_qwen("bedrock/us.anthropic.claude-sonnet-4-6") is False

    def test_non_bedrock_qwen(self):
        assert is_bedrock_qwen("ollama/qwen3:14b") is False

    def test_openai_not_bedrock(self):
        assert is_bedrock_qwen("openai/gpt-4o") is False


# ── is_bedrock_glm ────────────────────────────────────────────


class TestIsBedrockGlm:
    def test_bedrock_glm_47(self):
        assert is_bedrock_glm("bedrock/zhipuai.glm-4.7-250414") is True

    def test_bedrock_glm_generic(self):
        assert is_bedrock_glm("bedrock/glm-4.7") is True

    def test_bedrock_glm_upper(self):
        assert is_bedrock_glm("bedrock/zhipuai.GLM-4.7-Chat") is True

    def test_bedrock_claude_not_glm(self):
        assert is_bedrock_glm("bedrock/claude-opus-4-6") is False

    def test_bedrock_qwen_not_glm(self):
        assert is_bedrock_glm("bedrock/qwen.qwen3-next-80b-a3b") is False

    def test_non_bedrock_glm(self):
        assert is_bedrock_glm("ollama/glm-4.7") is False

    def test_openai_glm_not_bedrock(self):
        assert is_bedrock_glm("openai/glm-4.7-flash") is False


# ── is_bedrock_kimi ───────────────────────────────────────────


class TestIsBedrockKimi:
    def test_bedrock_kimi_k25(self):
        assert is_bedrock_kimi("bedrock/moonshotai.kimi-k2.5") is True

    def test_bedrock_kimi_k2_thinking(self):
        assert is_bedrock_kimi("bedrock/moonshot.kimi-k2-thinking") is True

    def test_bedrock_moonshot_prefix(self):
        assert is_bedrock_kimi("bedrock/moonshotai.Kimi-K2-Instruct") is True

    def test_bedrock_claude_not_kimi(self):
        assert is_bedrock_kimi("bedrock/claude-opus-4-6") is False

    def test_bedrock_qwen_not_kimi(self):
        assert is_bedrock_kimi("bedrock/qwen.qwen3-next-80b-a3b") is False

    def test_non_bedrock_kimi(self):
        assert is_bedrock_kimi("ollama/kimi-k2:latest") is False


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

    def test_max_on_bedrock_opus(self):
        assert resolve_thinking_effort("bedrock/jp.anthropic.claude-opus-4-6", "max") == "max"

    def test_max_clamped_on_bedrock_sonnet(self):
        assert resolve_thinking_effort("bedrock/jp.anthropic.claude-sonnet-4-6", "max") == "high"

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
            model="claude-sonnet-4-6",
            thinking=True,
            thinking_effort="medium",
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["temperature"] == 1

    def test_old_claude_gets_manual_thinking(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="claude-sonnet-4-5-20250929",
            thinking=True,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert kwargs["temperature"] == 1

    def test_ollama_gets_think_param(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="ollama/qwen3:14b",
            thinking=True,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs.get("think") is True
        assert "thinking" not in kwargs
        assert "extra_body" not in kwargs

    def test_bedrock_gets_reasoning_effort(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="bedrock/claude-opus-4-6",
            thinking=True,
            thinking_effort="medium",
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["reasoning_effort"] == "medium"
        assert "thinking" not in kwargs

    def test_bedrock_kimi_gets_reasoning_config(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="bedrock/moonshotai.kimi-k2.5",
            thinking=True,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["reasoning_config"] == "high"
        assert "reasoning_effort" not in kwargs
        assert "thinking" not in kwargs
        assert "enable_thinking" not in kwargs

    def test_bedrock_kimi_thinking_false_no_reasoning_config(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="bedrock/moonshotai.kimi-k2.5",
            thinking=False,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert "reasoning_config" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_bedrock_qwen_gets_enable_thinking_true(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="bedrock/qwen.qwen3-next-80b-a3b",
            thinking=True,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["enable_thinking"] is True
        assert "reasoning_effort" not in kwargs
        assert "thinking" not in kwargs
        assert "think" not in kwargs

    def test_bedrock_qwen_thinking_false_omits_enable_thinking(self, anima_dir, tool_handler, memory):
        """thinking=False must NOT send enable_thinking — causes 'Please continue' on qwen3-next."""
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="bedrock/qwen.qwen3-next-80b-a3b",
            thinking=False,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert "enable_thinking" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_openai_gets_extra_body_enable_thinking_true(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="openai/qwen3.5-9b",
            thinking=True,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["extra_body"]["enable_thinking"] is True
        assert "think" not in kwargs
        assert "thinking" not in kwargs
        assert "enable_thinking" not in kwargs  # top-level, not extra_body

    def test_openai_gets_extra_body_enable_thinking_false(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="openai/qwen3.5-9b",
            thinking=False,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["extra_body"]["enable_thinking"] is False
        assert "think" not in kwargs

    def test_openai_thinking_none_no_extra_body(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="openai/qwen3.5-9b",
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert "extra_body" not in kwargs
        assert "think" not in kwargs

    def test_bedrock_glm_gets_enable_thinking_true(self, anima_dir, tool_handler, memory):
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="bedrock/zhipuai.glm-4.7-250414",
            thinking=True,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["enable_thinking"] is True
        assert "reasoning_effort" not in kwargs
        assert "thinking" not in kwargs
        assert "think" not in kwargs

    def test_bedrock_glm_thinking_false_omits_enable_thinking(self, anima_dir, tool_handler, memory):
        """thinking=False must NOT send enable_thinking — same fix as Bedrock Qwen."""
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="bedrock/zhipuai.glm-4.7-250414",
            thinking=False,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert "enable_thinking" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_openai_gpt_with_thinking_uses_extra_body(self, anima_dir, tool_handler, memory):
        """Non-Qwen openai/* models also use extra_body when thinking is set."""
        from core.execution.litellm_loop import LiteLLMExecutor

        cfg = ModelConfig(
            model="openai/gpt-4o",
            thinking=True,
            api_key="k",
        )
        ex = LiteLLMExecutor(
            model_config=cfg,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["extra_body"]["enable_thinking"] is True
        assert "think" not in kwargs
