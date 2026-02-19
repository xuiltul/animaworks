# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tool permission parsing in AgentCore._init_tool_registry().

Validates the default-all permission model:
  - No 外部ツール section -> returns all TOOL_MODULES keys
  - ``- all: yes`` -> returns all TOOL_MODULES keys
  - ``- all: yes`` + ``- chatwork: no`` -> all except chatwork
  - Individual ``- web_search: yes`` only -> whitelist (backward compat)
  - Empty permissions -> returns all tools
  - ``- all: yes`` + multiple denies -> correct exclusion
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import ModelConfig


# ── Helpers ───────────────────────────────────────────────


# Stable fake TOOL_MODULES for deterministic assertions
_FAKE_TOOL_MODULES: dict[str, str] = {
    "aws_collector": "core.tools.aws_collector",
    "chatwork": "core.tools.chatwork",
    "github": "core.tools.github",
    "gmail": "core.tools.gmail",
    "image_gen": "core.tools.image_gen",
    "local_llm": "core.tools.local_llm",
    "slack": "core.tools.slack",
    "transcribe": "core.tools.transcribe",
    "web_search": "core.tools.web_search",
    "x_search": "core.tools.x_search",
}

_ALL_TOOLS_SORTED = sorted(_FAKE_TOOL_MODULES.keys())


def _build_agent(tmp_path: Path, permissions_text: str) -> "AgentCore":
    """Construct AgentCore with a mocked MemoryManager returning *permissions_text*."""
    mc = ModelConfig(model="claude-sonnet-4-20250514", api_key="test-key")
    memory = MagicMock()
    memory.read_permissions.return_value = permissions_text
    memory.anima_dir = tmp_path

    with (
        patch("core.agent.ToolHandler"),
        patch("core.agent.AgentCore._check_sdk", return_value=False),
        patch("core.agent.AgentCore._discover_personal_tools", return_value={}),
        patch("core.agent.AgentCore._create_executor"),
        patch("core.tools.TOOL_MODULES", _FAKE_TOOL_MODULES),
        patch("core.tools.discover_core_tools", return_value=_FAKE_TOOL_MODULES),
    ):
        from core.agent import AgentCore
        agent = AgentCore(tmp_path, memory, mc)
    return agent


# ── Test cases ────────────────────────────────────────────


class TestToolPermissionsDefaultAll:
    """Test the default-all permission model."""

    def test_no_external_tools_section_returns_all(self, tmp_path: Path) -> None:
        """No 外部ツール section -> returns all TOOL_MODULES keys."""
        permissions = "# Permissions\n## 使えるツール\nRead, Write\n"
        agent = _build_agent(tmp_path, permissions)
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_all_yes_returns_all(self, tmp_path: Path) -> None:
        """``- all: yes`` -> returns all TOOL_MODULES keys."""
        permissions = "## 外部ツール\n- all: yes\n"
        agent = _build_agent(tmp_path, permissions)
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_all_yes_with_deny_excludes_tool(self, tmp_path: Path) -> None:
        """``- all: yes`` + ``- chatwork: no`` -> all except chatwork."""
        permissions = "## 外部ツール\n- all: yes\n- chatwork: no\n"
        agent = _build_agent(tmp_path, permissions)
        expected = [t for t in _ALL_TOOLS_SORTED if t != "chatwork"]
        assert agent._tool_registry == expected

    def test_individual_whitelist_backward_compat(self, tmp_path: Path) -> None:
        """Individual ``- web_search: yes`` only -> returns only web_search."""
        permissions = "## 外部ツール\n- web_search: yes\n"
        agent = _build_agent(tmp_path, permissions)
        assert agent._tool_registry == ["web_search"]

    def test_empty_permissions_returns_all(self, tmp_path: Path) -> None:
        """Empty permissions string -> returns all tools (no section found)."""
        agent = _build_agent(tmp_path, "")
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_all_yes_with_multiple_denies(self, tmp_path: Path) -> None:
        """``- all: yes`` + multiple deny entries -> correct exclusion."""
        permissions = (
            "## 外部ツール\n"
            "- all: yes\n"
            "- chatwork: no\n"
            "- slack: disabled\n"
            "- gmail: false\n"
        )
        agent = _build_agent(tmp_path, permissions)
        expected = [
            t for t in _ALL_TOOLS_SORTED
            if t not in {"chatwork", "slack", "gmail"}
        ]
        assert agent._tool_registry == expected

    def test_section_present_but_no_entries_returns_all(self, tmp_path: Path) -> None:
        """Section header present but no allow/deny lines -> returns all tools."""
        permissions = "## 外部ツール\n\n"
        agent = _build_agent(tmp_path, permissions)
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_deny_values_case_insensitive(self, tmp_path: Path) -> None:
        """Deny values are case-insensitive (No, NO, DENY, Disabled, False)."""
        permissions = (
            "## 外部ツール\n"
            "- all: yes\n"
            "- chatwork: No\n"
            "- slack: DENY\n"
            "- gmail: Disabled\n"
            "- github: False\n"
        )
        agent = _build_agent(tmp_path, permissions)
        expected = [
            t for t in _ALL_TOOLS_SORTED
            if t not in {"chatwork", "slack", "gmail", "github"}
        ]
        assert agent._tool_registry == expected

    def test_allow_values_case_insensitive(self, tmp_path: Path) -> None:
        """Allow values accept OK, yes, enabled, true (case-insensitive)."""
        permissions = (
            "## 外部ツール\n"
            "- web_search: OK\n"
            "- chatwork: Enabled\n"
            "- gmail: TRUE\n"
        )
        agent = _build_agent(tmp_path, permissions)
        assert sorted(agent._tool_registry) == ["chatwork", "gmail", "web_search"]

    def test_deny_unknown_tool_ignored(self, tmp_path: Path) -> None:
        """Deny entries for tools not in TOOL_MODULES are silently ignored."""
        permissions = (
            "## 外部ツール\n"
            "- all: yes\n"
            "- nonexistent_tool: no\n"
        )
        agent = _build_agent(tmp_path, permissions)
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_memory_none_returns_all(self, tmp_path: Path) -> None:
        """When memory is None, permissions is empty -> returns all tools."""
        mc = ModelConfig(model="claude-sonnet-4-20250514", api_key="test-key")

        with (
            patch("core.agent.ToolHandler"),
            patch("core.agent.AgentCore._check_sdk", return_value=False),
            patch("core.agent.AgentCore._discover_personal_tools", return_value={}),
            patch("core.agent.AgentCore._create_executor"),
            patch("core.tools.TOOL_MODULES", _FAKE_TOOL_MODULES),
            patch("core.tools.discover_core_tools", return_value=_FAKE_TOOL_MODULES),
        ):
            from core.agent import AgentCore
            agent = AgentCore(tmp_path, None, mc)  # type: ignore[arg-type]
        assert agent._tool_registry == _ALL_TOOLS_SORTED
