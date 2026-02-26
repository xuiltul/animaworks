from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for adaptive thinking support.

Verifies end-to-end adaptive thinking parameter flow through AgentCore:
- Claude 4.6 models get adaptive thinking with effort parameter
- Thinking effort defaults and clamping
- max_tokens enforcement with thinking enabled
- Backward compatibility with existing thinking=True/False
"""

import json
from pathlib import Path

import pytest

from tests.helpers.mocks import make_litellm_response, patch_litellm


@pytest.mark.e2e
class TestAdaptiveThinkingModeB:
    """Mode B (assisted) adaptive thinking tests via mocked LLM calls."""

    async def test_claude_46_thinking_true_gets_adaptive(
        self, make_agent_core, data_dir,
    ):
        """Claude 4.6 + thinking=True → adaptive thinking params in LLM call."""
        agent = make_agent_core(
            name="adaptive-claude46",
            model="claude-sonnet-4-6",
            execution_mode="assisted",
        )

        status_path = agent.anima_dir / "status.json"
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status_data["thinking"] = True
        status_data["thinking_effort"] = "medium"
        status_path.write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")

        from core.config import invalidate_cache
        invalidate_cache()
        from core.memory import MemoryManager
        memory = MemoryManager(agent.anima_dir)
        model_config = memory.read_model_config()
        agent._model_config = model_config
        agent._executor._model_config = model_config

        main_resp = make_litellm_response(content="Adaptive thinking response.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Think about this deeply.")

        assert mock_fn.call_count >= 1
        first_kwargs = mock_fn.call_args_list[0].kwargs
        assert first_kwargs.get("thinking") == {"type": "adaptive"}
        assert first_kwargs.get("reasoning_effort") == "medium"
        assert first_kwargs.get("temperature") == 1

    async def test_ollama_thinking_true_gets_think_param(self, make_agent_core):
        """Ollama + thinking=True → think=True (not adaptive)."""
        agent = make_agent_core(
            name="thinking-ollama",
            model="ollama/qwen3:14b",
            execution_mode="assisted",
        )

        status_path = agent.anima_dir / "status.json"
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status_data["thinking"] = True
        status_path.write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")

        from core.config import invalidate_cache
        invalidate_cache()
        from core.memory import MemoryManager
        memory = MemoryManager(agent.anima_dir)
        model_config = memory.read_model_config()
        agent._model_config = model_config
        agent._executor._model_config = model_config

        main_resp = make_litellm_response(content="Thinking response.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Hello.")

        first_kwargs = mock_fn.call_args_list[0].kwargs
        assert first_kwargs.get("think") is True
        assert "thinking" not in first_kwargs

    async def test_thinking_effort_default_high(self, make_agent_core):
        """When thinking_effort is not set, defaults to 'high'."""
        agent = make_agent_core(
            name="effort-default",
            model="claude-sonnet-4-6",
            execution_mode="assisted",
        )

        status_path = agent.anima_dir / "status.json"
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status_data["thinking"] = True
        status_path.write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")

        from core.config import invalidate_cache
        invalidate_cache()
        from core.memory import MemoryManager
        memory = MemoryManager(agent.anima_dir)
        model_config = memory.read_model_config()
        agent._model_config = model_config
        agent._executor._model_config = model_config

        main_resp = make_litellm_response(content="Response.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Hello.")

        first_kwargs = mock_fn.call_args_list[0].kwargs
        assert first_kwargs.get("reasoning_effort") == "high"

    async def test_max_tokens_resolved_with_thinking(self, make_agent_core):
        """Thinking enabled → resolve_max_tokens returns >= 16384 at config level.

        Note: Mode B preflight may clamp max_tokens further based on context
        window, so we verify the ModelConfig-level resolution instead.
        """
        agent = make_agent_core(
            name="maxtoken-thinking",
            model="claude-sonnet-4-6",
            execution_mode="assisted",
        )

        status_path = agent.anima_dir / "status.json"
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status_data["thinking"] = True
        status_path.write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")

        from core.config import invalidate_cache
        invalidate_cache()
        from core.memory import MemoryManager
        memory = MemoryManager(agent.anima_dir)
        model_config = memory.read_model_config()

        from core.config.models import resolve_max_tokens
        effective = resolve_max_tokens(
            model_config.model, model_config.max_tokens, model_config.thinking,
        )
        assert effective >= 16384

    async def test_backward_compat_thinking_null(self, make_agent_core):
        """thinking=null (unset) → no thinking params in call."""
        agent = make_agent_core(
            name="compat-null",
            model="claude-sonnet-4-6",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="Normal response.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Hello.")

        first_kwargs = mock_fn.call_args_list[0].kwargs
        assert "thinking" not in first_kwargs
        assert "think" not in first_kwargs

    async def test_effort_max_clamped_on_sonnet(self, make_agent_core):
        """effort='max' on Sonnet 4.6 → clamped to 'high'."""
        agent = make_agent_core(
            name="clamp-max",
            model="claude-sonnet-4-6",
            execution_mode="assisted",
        )

        status_path = agent.anima_dir / "status.json"
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status_data["thinking"] = True
        status_data["thinking_effort"] = "max"
        status_path.write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")

        from core.config import invalidate_cache
        invalidate_cache()
        from core.memory import MemoryManager
        memory = MemoryManager(agent.anima_dir)
        model_config = memory.read_model_config()
        agent._model_config = model_config
        agent._executor._model_config = model_config

        main_resp = make_litellm_response(content="Response.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Hello.")

        first_kwargs = mock_fn.call_args_list[0].kwargs
        assert first_kwargs.get("reasoning_effort") == "high"
