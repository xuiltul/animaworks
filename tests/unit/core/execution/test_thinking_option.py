# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Ollama thinking option in LiteLLMExecutor and AssistedExecutor.

The ``thinking`` field on ``ModelConfig`` controls the ``think`` kwarg passed
to ``litellm.acompletion``.  Behaviour:

- ``thinking=None`` + ollama/ model  → ``think=False`` (auto-off default)
- ``thinking=True``                  → ``think=True``  (explicit override)
- ``thinking=False``                 → ``think=False`` (explicit off)
- ``thinking=None`` + non-ollama     → ``think`` key absent
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


# ── Helpers ───────────────────────────────────────────────────


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


def _make_assisted_executor(
    model_config: ModelConfig,
    anima_dir: Path,
    tool_handler: MagicMock,
    memory: MagicMock,
):
    """Instantiate an AssistedExecutor with minimal dependencies."""
    from core.execution.assisted import AssistedExecutor
    return AssistedExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        memory=memory,
        messenger=None,
        tool_registry=[],
        personal_tools={},
    )


# ── LiteLLMExecutor._build_llm_kwargs tests ─────────────────


class TestLiteLLMThinkingOption:
    """Verify ``think`` kwarg in LiteLLMExecutor._build_llm_kwargs()."""

    def test_ollama_model_thinking_none_defaults_to_false(
        self, anima_dir, tool_handler, memory,
    ):
        """Ollama model + thinking=None → think=False (auto-off)."""
        cfg = ModelConfig(model="ollama/glm-4", thinking=None, api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        kwargs = ex._build_llm_kwargs()
        assert "think" in kwargs
        assert kwargs["think"] is False

    def test_ollama_model_thinking_true_overrides(
        self, anima_dir, tool_handler, memory,
    ):
        """Ollama model + thinking=True → think=True (explicit override)."""
        cfg = ModelConfig(model="ollama/qwen3", thinking=True, api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        kwargs = ex._build_llm_kwargs()
        assert kwargs["think"] is True

    def test_ollama_model_thinking_false_explicit(
        self, anima_dir, tool_handler, memory,
    ):
        """Ollama model + thinking=False → think=False (explicit off)."""
        cfg = ModelConfig(model="ollama/deepseek-r1", thinking=False, api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        kwargs = ex._build_llm_kwargs()
        assert kwargs["think"] is False

    def test_non_ollama_model_thinking_none_no_think_key(
        self, anima_dir, tool_handler, memory,
    ):
        """Non-ollama model + thinking=None → ``think`` key absent."""
        cfg = ModelConfig(model="openai/gpt-4o", thinking=None, api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        kwargs = ex._build_llm_kwargs()
        assert "think" not in kwargs

    def test_non_ollama_model_thinking_true_sets_think(
        self, anima_dir, tool_handler, memory,
    ):
        """Non-ollama model + thinking=True → think=True (explicit on works for any model)."""
        cfg = ModelConfig(model="openai/gpt-4o", thinking=True, api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        kwargs = ex._build_llm_kwargs()
        assert kwargs["think"] is True

    def test_non_ollama_model_thinking_false_sets_think(
        self, anima_dir, tool_handler, memory,
    ):
        """Non-ollama model + thinking=False → think=False."""
        cfg = ModelConfig(model="google/gemini-2.5-pro", thinking=False, api_key="k")
        ex = _make_litellm_executor(cfg, anima_dir, tool_handler, memory)
        kwargs = ex._build_llm_kwargs()
        assert kwargs["think"] is False


# ── AssistedExecutor._call_llm tests ────────────────────────


@pytest.mark.asyncio
class TestAssistedThinkingOption:
    """Verify ``think`` kwarg passed through AssistedExecutor._call_llm()."""

    async def test_ollama_model_thinking_none_defaults_to_false(
        self, anima_dir, tool_handler, memory,
    ):
        """Ollama model + thinking=None → think=False passed to litellm."""
        cfg = ModelConfig(model="ollama/glm-4", thinking=None, api_key="k", max_tokens=512)
        ex = _make_assisted_executor(cfg, anima_dir, tool_handler, memory)
        mock_acompletion = AsyncMock(return_value=MagicMock())
        with patch("litellm.acompletion", mock_acompletion):
            await ex._call_llm([{"role": "user", "content": "hi"}])
        _, kwargs = mock_acompletion.call_args
        assert kwargs.get("think") is False

    async def test_ollama_model_thinking_true_overrides(
        self, anima_dir, tool_handler, memory,
    ):
        """Ollama model + thinking=True → think=True passed to litellm."""
        cfg = ModelConfig(model="ollama/qwen3", thinking=True, api_key="k", max_tokens=512)
        ex = _make_assisted_executor(cfg, anima_dir, tool_handler, memory)
        mock_acompletion = AsyncMock(return_value=MagicMock())
        with patch("litellm.acompletion", mock_acompletion):
            await ex._call_llm([{"role": "user", "content": "hi"}])
        _, kwargs = mock_acompletion.call_args
        assert kwargs.get("think") is True

    async def test_ollama_model_thinking_false_explicit(
        self, anima_dir, tool_handler, memory,
    ):
        """Ollama model + thinking=False → think=False passed to litellm."""
        cfg = ModelConfig(model="ollama/deepseek-r1", thinking=False, api_key="k", max_tokens=512)
        ex = _make_assisted_executor(cfg, anima_dir, tool_handler, memory)
        mock_acompletion = AsyncMock(return_value=MagicMock())
        with patch("litellm.acompletion", mock_acompletion):
            await ex._call_llm([{"role": "user", "content": "hi"}])
        _, kwargs = mock_acompletion.call_args
        assert kwargs.get("think") is False

    async def test_non_ollama_model_thinking_none_no_think_key(
        self, anima_dir, tool_handler, memory,
    ):
        """Non-ollama model + thinking=None → ``think`` key absent from kwargs."""
        cfg = ModelConfig(model="openai/gpt-4o", thinking=None, api_key="k", max_tokens=512)
        ex = _make_assisted_executor(cfg, anima_dir, tool_handler, memory)
        mock_acompletion = AsyncMock(return_value=MagicMock())
        with patch("litellm.acompletion", mock_acompletion):
            await ex._call_llm([{"role": "user", "content": "hi"}])
        _, kwargs = mock_acompletion.call_args
        assert "think" not in kwargs

    async def test_non_ollama_model_thinking_true_sets_think(
        self, anima_dir, tool_handler, memory,
    ):
        """Non-ollama model + thinking=True → think=True passed to litellm."""
        cfg = ModelConfig(model="openai/gpt-4o", thinking=True, api_key="k", max_tokens=512)
        ex = _make_assisted_executor(cfg, anima_dir, tool_handler, memory)
        mock_acompletion = AsyncMock(return_value=MagicMock())
        with patch("litellm.acompletion", mock_acompletion):
            await ex._call_llm([{"role": "user", "content": "hi"}])
        _, kwargs = mock_acompletion.call_args
        assert kwargs.get("think") is True

    async def test_non_ollama_model_thinking_false_sets_think(
        self, anima_dir, tool_handler, memory,
    ):
        """Non-ollama model + thinking=False → think=False passed to litellm."""
        cfg = ModelConfig(model="google/gemini-2.5-pro", thinking=False, api_key="k", max_tokens=512)
        ex = _make_assisted_executor(cfg, anima_dir, tool_handler, memory)
        mock_acompletion = AsyncMock(return_value=MagicMock())
        with patch("litellm.acompletion", mock_acompletion):
            await ex._call_llm([{"role": "user", "content": "hi"}])
        _, kwargs = mock_acompletion.call_args
        assert kwargs.get("think") is False
