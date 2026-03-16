"""Unit tests for frequency_penalty and presence_penalty in LiteLLM kwargs.

Verifies that penalty parameters flow from models.json through ModelConfig
to _build_llm_kwargs() for LLM degenerate repetition defense.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import ModelConfig

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Minimal anima directory structure required by executors."""
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    (d / "identity.md").write_text("# Test", encoding="utf-8")
    for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm"]:
        (d / sub).mkdir(exist_ok=True)
    return d


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    """Mock MemoryManager with minimal stubs."""
    from core.memory import MemoryManager

    m = MagicMock(spec=MemoryManager)
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    m.anima_dir = anima_dir
    return m


@pytest.fixture
def tool_handler(anima_dir: Path, memory: MagicMock) -> MagicMock:
    """Mock ToolHandler (avoids loading real tool schemas)."""
    from core.tooling.handler import ToolHandler

    th = MagicMock(spec=ToolHandler)
    th._human_notifier = None
    return th


def _make_litellm_executor(
    model_config: ModelConfig,
    anima_dir: Path,
    tool_handler: MagicMock,
    memory: MagicMock,
):
    """Instantiate a LiteLLMExecutor with minimal dependencies."""
    from core.execution.litellm_loop import LiteLLMExecutor

    return LiteLLMExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )


# ── resolve_penalties tests ────────────────────────────────────


class TestResolvePenalties:
    """Verify resolve_penalties() from core.config.models."""

    def test_resolve_penalties_from_models_json(self):
        """When models.json entry has penalties, returns correct dict."""
        from core.config.models import resolve_penalties

        entry = {"frequency_penalty": 0.3, "presence_penalty": 0.1}
        with patch("core.config.model_config._match_models_json", return_value=entry):
            result = resolve_penalties("claude-sonnet-4-6")
        assert result == {"frequency_penalty": 0.3, "presence_penalty": 0.1}

    def test_resolve_penalties_empty_when_not_configured(self):
        """When models.json entry has no penalties, returns empty dict."""
        from core.config.models import resolve_penalties

        entry = {"mode": "S", "context_window": 200000}
        with patch("core.config.model_config._match_models_json", return_value=entry):
            result = resolve_penalties("claude-sonnet-4-6")
        assert result == {}

    def test_resolve_penalties_no_match(self):
        """When no models.json match, returns empty dict."""
        from core.config.models import resolve_penalties

        with patch("core.config.model_config._match_models_json", return_value=None):
            result = resolve_penalties("unknown-model")
        assert result == {}

    def test_resolve_penalties_partial_entry(self):
        """When only one penalty is configured, returns only that key."""
        from core.config.models import resolve_penalties

        entry = {"frequency_penalty": 0.5}
        with patch("core.config.model_config._match_models_json", return_value=entry):
            result = resolve_penalties("openai/gpt-4o")
        assert result == {"frequency_penalty": 0.5}
        assert "presence_penalty" not in result

    def test_resolve_penalties_invalid_value_skipped(self):
        """Invalid float values are skipped (backward-compatible)."""
        from core.config.models import resolve_penalties

        entry = {"frequency_penalty": "not-a-number", "presence_penalty": 0.2}
        with patch("core.config.model_config._match_models_json", return_value=entry):
            result = resolve_penalties("claude-sonnet-4-6")
        assert result == {"presence_penalty": 0.2}

    def test_resolve_penalties_clamped_to_range(self):
        """Values outside [-2.0, 2.0] are clamped."""
        from core.config.models import resolve_penalties

        entry = {"frequency_penalty": 5.0, "presence_penalty": -3.0}
        with patch("core.config.model_config._match_models_json", return_value=entry):
            result = resolve_penalties("openai/gpt-4o")
        assert result == {"frequency_penalty": 2.0, "presence_penalty": -2.0}


# ── _build_llm_kwargs penalty injection tests ──────────────────


class TestBuildLlmKwargsPenalties:
    """Verify _build_llm_kwargs() includes penalty params when configured."""

    def test_build_llm_kwargs_includes_penalties(
        self, anima_dir, tool_handler, memory
    ):
        """When resolve_penalties returns values, kwargs include them."""
        cfg = ModelConfig(model="claude-sonnet-4-6", api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        with patch(
            "core.config.models.resolve_penalties",
            return_value={"frequency_penalty": 0.4, "presence_penalty": 0.2},
        ):
            kwargs = ex._build_llm_kwargs()
        assert kwargs["frequency_penalty"] == 0.4
        assert kwargs["presence_penalty"] == 0.2

    def test_build_llm_kwargs_model_config_overrides_models_json(
        self, anima_dir, tool_handler, memory
    ):
        """ModelConfig.frequency_penalty overrides models.json value."""
        cfg = ModelConfig(
            model="claude-sonnet-4-6",
            api_key="k",
            frequency_penalty=0.8,
            presence_penalty=0.5,
        )
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        with patch(
            "core.config.models.resolve_penalties",
            return_value={"frequency_penalty": 0.1, "presence_penalty": 0.1},
        ):
            kwargs = ex._build_llm_kwargs()
        assert kwargs["frequency_penalty"] == 0.8
        assert kwargs["presence_penalty"] == 0.5

    def test_build_llm_kwargs_no_penalties_when_unconfigured(
        self, anima_dir, tool_handler, memory
    ):
        """When no penalties configured, kwargs omit them (backward-compatible)."""
        cfg = ModelConfig(model="claude-sonnet-4-6", api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        with patch("core.config.models.resolve_penalties", return_value={}):
            kwargs = ex._build_llm_kwargs()
        assert "frequency_penalty" not in kwargs
        assert "presence_penalty" not in kwargs
