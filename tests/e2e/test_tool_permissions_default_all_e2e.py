# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for default-all tool permissions.

Verifies that a real DigitalAnima / AgentCore instance populates the tool
registry correctly under the new default-all permission model, using
isolated filesystem fixtures.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ── Permissions fixtures ──────────────────────────────────

PERMISSIONS_ALL_YES = """\
# Permissions: test-agent

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 外部ツール
- all: yes
"""

PERMISSIONS_ALL_YES_WITH_DENY = """\
# Permissions: test-agent

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 外部ツール
- all: yes
- chatwork: no
- slack: disabled
"""

PERMISSIONS_WHITELIST_ONLY = """\
# Permissions: test-agent

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 外部ツール
- web_search: yes
- image_gen: yes
"""

PERMISSIONS_NO_SECTION = """\
# Permissions: test-agent

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob
"""

PERMISSIONS_EMPTY = ""


# ── Tests ─────────────────────────────────────────────────


class TestToolPermissionsDefaultAllE2E:
    """E2E: tool registry populated via AgentCore with real MemoryManager."""

    def test_all_yes_populates_all_tools(self, make_agent_core) -> None:
        """``- all: yes`` enables all core tool modules."""
        agent = make_agent_core("test-agent", permissions=PERMISSIONS_ALL_YES)
        from core.tools import TOOL_MODULES
        assert sorted(agent._tool_registry) == sorted(TOOL_MODULES.keys())

    def test_all_yes_with_deny_excludes_tools(self, make_agent_core) -> None:
        """``- all: yes`` + deny entries correctly excludes named tools."""
        agent = make_agent_core("test-agent", permissions=PERMISSIONS_ALL_YES_WITH_DENY)
        assert "chatwork" not in agent._tool_registry
        assert "slack" not in agent._tool_registry
        # Other tools remain
        assert "web_search" in agent._tool_registry
        assert "image_gen" in agent._tool_registry

    def test_whitelist_backward_compat(self, make_agent_core) -> None:
        """Individual allow entries work as whitelist (backward compat)."""
        agent = make_agent_core("test-agent", permissions=PERMISSIONS_WHITELIST_ONLY)
        assert sorted(agent._tool_registry) == ["image_gen", "web_search"]

    def test_no_section_returns_all(self, make_agent_core) -> None:
        """No 外部ツール section -> all tools enabled."""
        agent = make_agent_core("test-agent", permissions=PERMISSIONS_NO_SECTION)
        from core.tools import TOOL_MODULES
        assert sorted(agent._tool_registry) == sorted(TOOL_MODULES.keys())

    def test_empty_permissions_returns_all(self, make_agent_core) -> None:
        """Empty permissions file -> all tools enabled."""
        agent = make_agent_core("test-agent", permissions=PERMISSIONS_EMPTY)
        from core.tools import TOOL_MODULES
        assert sorted(agent._tool_registry) == sorted(TOOL_MODULES.keys())


class TestToolPermissionsDigitalAnimaE2E:
    """E2E: verify tool registry via DigitalAnima construction."""

    def test_digital_anima_default_all(self, make_digital_anima) -> None:
        """DigitalAnima with no 外部ツール section gets all tools."""
        anima = make_digital_anima("test-anima", permissions=PERMISSIONS_NO_SECTION)
        from core.tools import TOOL_MODULES
        assert sorted(anima.agent._tool_registry) == sorted(TOOL_MODULES.keys())

    def test_digital_anima_deny_list(self, make_digital_anima) -> None:
        """DigitalAnima with deny list correctly excludes tools."""
        anima = make_digital_anima("test-anima", permissions=PERMISSIONS_ALL_YES_WITH_DENY)
        assert "chatwork" not in anima.agent._tool_registry
        assert "slack" not in anima.agent._tool_registry
        assert "web_search" in anima.agent._tool_registry
