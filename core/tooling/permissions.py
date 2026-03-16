from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import PermissionsConfig

"""Shared permission parser for external tool access control.

Both MCP server (``core.mcp.server``) and AgentCore executor
(``core._agent_executor``) use :func:`get_permitted_tools` (or legacy
:func:`parse_permitted_tools`) to resolve which external tools an Anima
is allowed to invoke, keeping the logic in a single authoritative location.

Supports action-level gating: dangerous sub-actions (e.g. ``gmail_send``)
require explicit ``- gmail_send: yes`` in permissions even when
``- all: yes`` or ``- gmail: yes`` is present.
"""

import importlib
import logging
import re

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────

_PERMISSION_ALLOW_RE = re.compile(
    r"[-*]?\s*(\w+)\s*:\s*(OK|yes|enabled|true|全権限|読み取り.*)\s*$",
    re.IGNORECASE,
)
_PERMISSION_ALL_RE = re.compile(
    r"[-*]?\s*all\s*:\s*(OK|yes|enabled|true)\s*$",
    re.IGNORECASE,
)
_PERMISSION_DENY_RE = re.compile(
    r"[-*]?\s*(\w+)\s*:\s*(no|deny|disabled|false)\s*$",
    re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────


def get_permitted_tools(config: PermissionsConfig) -> set[str]:
    """Get permitted tool names from structured permissions config."""
    from core.tools import TOOL_MODULES

    all_tools = set(TOOL_MODULES.keys())

    if config.external_tools.allow_all:
        base = all_tools - set(config.external_tools.deny)
        action_permits = set(config.external_tools.allow) - all_tools - set(config.external_tools.deny)
        return base | action_permits
    if config.external_tools.allow:
        return set(config.external_tools.allow) - set(config.external_tools.deny)
    return all_tools - set(config.external_tools.deny)


def parse_permitted_tools(text: str) -> set[str]:
    """Parse permissions.md text and return permitted tool and action names.

    Deprecated: Prefer loading via :func:`load_permissions` and
    :func:`get_permitted_tools` for structured config. This wrapper parses
    legacy Markdown and delegates to :func:`get_permitted_tools`.

    Strategy:
      1. No ``外部ツール`` / ``External Tools`` section present → ALL tools (default-all)
      2. ``- all: yes`` found → ALL tools minus any deny entries, plus explicit
         action-level permits (``all: yes`` does NOT auto-allow gated actions)
      3. Individual ``- tool: yes`` / ``- tool_action: yes`` entries → whitelist mode
      4. Section present but no matching entries → ALL tools

    Returns:
        Set of permitted names: tool module names (keys from ``core.tools.TOOL_MODULES``)
        and action-level permits (e.g. ``gmail_send``).
    """
    from core.config.models import ExternalToolsPermission, PermissionsConfig

    if "外部ツール" not in text and "External Tools" not in text:
        return get_permitted_tools(PermissionsConfig())

    has_all_yes = False
    allowed: list[str] = []
    denied: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if _PERMISSION_ALL_RE.match(stripped):
            has_all_yes = True
            continue
        m_deny = _PERMISSION_DENY_RE.match(stripped)
        if m_deny:
            name = m_deny.group(1)
            denied.append(name)
            continue
        m_allow = _PERMISSION_ALLOW_RE.match(stripped)
        if m_allow:
            name = m_allow.group(1)
            allowed.append(name)

    ext = ExternalToolsPermission(
        allow_all=has_all_yes,
        allow=allowed,
        deny=denied,
    )
    config = PermissionsConfig(external_tools=ext)
    return get_permitted_tools(config)


# ── Action gating ──────────────────────────────────────────


def _load_execution_profile(tool_name: str) -> dict[str, dict[str, object]] | None:
    """Load EXECUTION_PROFILE from the tool module.

    Args:
        tool_name: Tool module name (e.g. ``gmail``).

    Returns:
        The module's EXECUTION_PROFILE dict, or None if not found or load fails.
    """
    try:
        from core.tools import TOOL_MODULES

        if tool_name not in TOOL_MODULES:
            return None
        mod = importlib.import_module(TOOL_MODULES[tool_name])
        return getattr(mod, "EXECUTION_PROFILE", None)
    except Exception:
        logger.debug("Failed to load EXECUTION_PROFILE for %s", tool_name, exc_info=True)
        return None


def is_action_gated(tool_name: str, action: str, permitted: set[str]) -> bool:
    """Check if a tool action is gated and not explicitly permitted.

    Args:
        tool_name: Tool module name (e.g. ``gmail``).
        action: Action/subcommand name (e.g. ``send``).
        permitted: Set from :func:`parse_permitted_tools`.

    Returns:
        True if the action is gated AND not in permitted (i.e. should be blocked).
        False if non-gated, permitted, or tool module not found.
    """
    profile = _load_execution_profile(tool_name)
    if profile is None:
        return False

    action_info = profile.get(action)
    if action_info is None:
        return False

    if action_info.get("gated") is not True:
        return False

    action_key = f"{tool_name}_{action}"
    return action_key not in permitted
