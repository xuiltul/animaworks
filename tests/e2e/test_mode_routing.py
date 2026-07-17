# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for execution mode routing logic.

Verifies that AgentCore._resolve_execution_mode() correctly routes
to S, C, D, G, X, A, or B based on model name, SDK availability, and config.
No API calls are made in these tests.
"""

from __future__ import annotations


class TestModeRouting:
    """Mode detection: _resolve_execution_mode()."""

    def test_claude_model_with_sdk_routes_to_s(self, make_agent_core):
        """Claude model + SDK available → Mode S."""
        agent = make_agent_core(
            name="claude-s",
            model="claude-sonnet-4-6",
        )
        assert agent._resolve_execution_mode() == "s"

    def test_claude_model_explicit_assisted_routes_to_b(self, make_agent_core):
        """Claude model + execution_mode='assisted' → Mode B."""
        agent = make_agent_core(
            name="claude-b",
            model="claude-sonnet-4-6",
            execution_mode="assisted",
        )
        assert agent._resolve_execution_mode() == "b"

    def test_openai_model_routes_to_a(self, make_agent_core):
        """Non-Claude model (OpenAI) → Mode A."""
        agent = make_agent_core(
            name="openai-a",
            model="openai/gpt-4o",
        )
        assert agent._resolve_execution_mode() == "a"

    def test_cursor_model_routes_to_d(self, make_agent_core):
        """Cursor-prefixed model → Mode D."""
        agent = make_agent_core(
            name="cursor-d",
            model="cursor/auto",
        )
        assert agent._resolve_execution_mode() == "d"

    def test_gemini_cli_model_routes_to_g(self, make_agent_core):
        """Gemini CLI-prefixed model → Mode G."""
        agent = make_agent_core(
            name="gemini-g",
            model="gemini/gemini-2.5-pro",
        )
        assert agent._resolve_execution_mode() == "g"

    def test_grok_model_routes_to_x(self, make_agent_core):
        """Grok Build-prefixed model → Mode X."""
        agent = make_agent_core(
            name="grok-x",
            model="grok/grok-4.5",
        )
        assert agent._resolve_execution_mode() == "x"

    def test_ollama_model_routes_to_a(self, make_agent_core):
        """Ollama model with tool_use support → Mode A."""
        agent = make_agent_core(
            name="ollama-a",
            model="ollama/qwen3:14b",
        )
        assert agent._resolve_execution_mode() == "a"

    def test_ollama_model_explicit_assisted_routes_to_b(self, make_agent_core):
        """Ollama model + execution_mode='assisted' → Mode B."""
        agent = make_agent_core(
            name="ollama-b",
            model="ollama/qwen3:14b",
            execution_mode="assisted",
        )
        assert agent._resolve_execution_mode() == "b"

    def test_ollama_non_tool_model_routes_to_b(self, make_agent_core):
        """Ollama model without reliable tool_use → Mode B."""
        agent = make_agent_core(
            name="ollama-gemma-b",
            model="ollama/gemma3:27b",
        )
        assert agent._resolve_execution_mode() == "b"

    def test_claude_model_without_sdk_still_routes_to_s(self, make_agent_core):
        """Claude model + SDK unavailable → still Mode S (executor handles fallback)."""
        agent = make_agent_core(
            name="claude-nosdk",
            model="claude-sonnet-4-6",
        )
        # Force SDK to be unavailable — _resolve_execution_mode no longer
        # short-circuits to a; _create_executor handles the fallback chain.
        agent._sdk_available = False
        assert agent._resolve_execution_mode() == "s"

    def test_codex_model_routes_to_c(self, make_agent_core):
        """Codex-prefixed model → Mode C."""
        agent = make_agent_core(
            name="codex-c",
            model="codex/o4-mini",
        )
        assert agent._resolve_execution_mode() == "c"

    def test_codex_gpt41_routes_to_c(self, make_agent_core):
        """codex/gpt-4.1 → Mode C (not A)."""
        agent = make_agent_core(
            name="codex-gpt",
            model="codex/gpt-4.1",
        )
        assert agent._resolve_execution_mode() == "c"

    def test_openai_codex_routes_to_c(self, make_agent_core):
        """openai-codex/* alias → Mode C."""
        agent = make_agent_core(
            name="openai-codex-gpt",
            model="openai-codex/gpt-5.3-codex",
        )
        assert agent._resolve_execution_mode() == "c"

    def test_codex_explicit_a_override(self, make_agent_core):
        """Codex model + explicit execution_mode='A' → Mode A (override wins)."""
        agent = make_agent_core(
            name="codex-override",
            model="codex/o4-mini",
            execution_mode="A",
        )
        assert agent._resolve_execution_mode() == "a"
