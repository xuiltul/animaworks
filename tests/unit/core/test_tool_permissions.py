# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tool permission parsing in AgentCore._init_tool_registry().

Validates the default-all permission model using permissions.json:
  - No permissions.json -> returns all TOOL_MODULES keys (open default)
  - allow_all: true -> returns all TOOL_MODULES keys
  - allow_all: true + deny: ["chatwork"] -> all except chatwork
  - allow_all: false + allow: ["web_search"] only -> whitelist
  - Empty external_tools -> returns all tools
  - allow_all: true + multiple denies -> correct exclusion
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.schemas import ModelConfig


# ── Helpers ───────────────────────────────────────────────


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


def _write_permissions(tmp_path: Path, config_dict: dict | None) -> None:
    """Write permissions.json to *tmp_path*. None means no file."""
    if config_dict is not None:
        (tmp_path / "permissions.json").write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _build_agent(tmp_path: Path, config_dict: dict | None = None) -> "AgentCore":
    """Construct AgentCore with optional permissions.json in tmp_path."""
    _write_permissions(tmp_path, config_dict)

    mc = ModelConfig(model="claude-sonnet-4-6", api_key="test-key")
    memory = MagicMock()
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
    """Test the default-all permission model with permissions.json."""

    def test_no_permissions_file_returns_all(self, tmp_path: Path) -> None:
        """No permissions.json -> returns all TOOL_MODULES keys (open default)."""
        agent = _build_agent(tmp_path, None)
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_allow_all_returns_all(self, tmp_path: Path) -> None:
        """allow_all: true -> returns all TOOL_MODULES keys."""
        agent = _build_agent(tmp_path, {"external_tools": {"allow_all": True}})
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_allow_all_with_deny_excludes_tool(self, tmp_path: Path) -> None:
        """allow_all: true + deny: ["chatwork"] -> all except chatwork."""
        agent = _build_agent(
            tmp_path,
            {"external_tools": {"allow_all": True, "deny": ["chatwork"]}},
        )
        expected = [t for t in _ALL_TOOLS_SORTED if t != "chatwork"]
        assert agent._tool_registry == expected

    def test_individual_allowlist(self, tmp_path: Path) -> None:
        """allow_all: false + allow: ["web_search"] -> only web_search."""
        agent = _build_agent(
            tmp_path,
            {"external_tools": {"allow_all": False, "allow": ["web_search"]}},
        )
        assert agent._tool_registry == ["web_search"]

    def test_empty_permissions_returns_all(self, tmp_path: Path) -> None:
        """Empty permissions.json (just {}) -> returns all tools."""
        agent = _build_agent(tmp_path, {})
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_allow_all_with_multiple_denies(self, tmp_path: Path) -> None:
        """allow_all: true + multiple deny entries -> correct exclusion."""
        agent = _build_agent(
            tmp_path,
            {"external_tools": {"allow_all": True, "deny": ["chatwork", "slack", "gmail"]}},
        )
        expected = [t for t in _ALL_TOOLS_SORTED if t not in {"chatwork", "slack", "gmail"}]
        assert agent._tool_registry == expected

    def test_deny_unknown_tool_ignored(self, tmp_path: Path) -> None:
        """Deny entries for tools not in TOOL_MODULES are silently ignored."""
        agent = _build_agent(
            tmp_path,
            {"external_tools": {"allow_all": True, "deny": ["nonexistent_tool"]}},
        )
        assert agent._tool_registry == _ALL_TOOLS_SORTED

    def test_memory_none_returns_all(self, tmp_path: Path) -> None:
        """When memory is None, returns all tools (default PermissionsConfig)."""
        mc = ModelConfig(model="claude-sonnet-4-6", api_key="test-key")

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
