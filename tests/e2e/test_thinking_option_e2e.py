# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Ollama thinking option (think=False default).

Verifies that the full AgentCore.run_cycle() path passes the correct
``think`` parameter to ``litellm.acompletion`` depending on model and
explicit configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.mocks import make_litellm_response, patch_litellm


class TestThinkingOptionModeB:
    """Mode B (assisted) thinking option tests using mocked LLM calls."""

    async def test_ollama_model_gets_think_false_by_default(self, make_agent_core):
        """Ollama model should receive think=False when thinking is not configured."""
        agent = make_agent_core(
            name="think-ollama-default",
            model="ollama/glm-flash-q8:32k",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="Response from Ollama.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Hello from test.")

        # Mode B calls litellm.acompletion twice: main + knowledge extraction.
        # Check the FIRST call (main response).
        assert mock_fn.call_count >= 1
        first_call_kwargs = mock_fn.call_args_list[0].kwargs
        assert "think" in first_call_kwargs, (
            "Expected 'think' kwarg in litellm.acompletion call for ollama/ model"
        )
        assert first_call_kwargs["think"] is False

    async def test_non_ollama_model_no_think_param(self, make_agent_core):
        """Non-Ollama model should NOT receive a think parameter."""
        agent = make_agent_core(
            name="think-non-ollama",
            model="claude-sonnet-4-20250514",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(content="Response from Claude.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Hello from test.")

        assert mock_fn.call_count >= 1
        first_call_kwargs = mock_fn.call_args_list[0].kwargs
        assert "think" not in first_call_kwargs, (
            "Non-Ollama model should not have 'think' kwarg"
        )

    async def test_ollama_model_explicit_thinking_true(
        self, make_agent_core, data_dir,
    ):
        """Ollama model with explicit thinking=True should receive think=True."""
        agent = make_agent_core(
            name="think-ollama-explicit",
            model="ollama/glm-flash-q8:32k",
            execution_mode="assisted",
        )

        # Manually set thinking=True in config.json (not supported by
        # create_anima_dir helper) and rebuild model_config.
        config_path = data_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["animas"]["think-ollama-explicit"]["thinking"] = True
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Invalidate cache and reload model config
        from core.config import invalidate_cache
        invalidate_cache()

        from core.memory import MemoryManager
        memory = MemoryManager(agent.anima_dir)
        model_config = memory.read_model_config()
        agent._model_config = model_config
        agent._executor._model_config = model_config

        main_resp = make_litellm_response(content="Response with thinking.")
        extract_resp = make_litellm_response(content="なし")

        with patch_litellm(main_resp, extract_resp) as mock_fn:
            await agent.run_cycle("Hello, think about this.")

        assert mock_fn.call_count >= 1
        first_call_kwargs = mock_fn.call_args_list[0].kwargs
        assert "think" in first_call_kwargs, (
            "Expected 'think' kwarg when thinking=True is explicitly set"
        )
        assert first_call_kwargs["think"] is True
