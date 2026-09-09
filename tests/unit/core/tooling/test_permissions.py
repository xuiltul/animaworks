# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for core.tooling.permissions — parse_permitted_tools and is_action_gated."""

from __future__ import annotations

from unittest.mock import patch


from core.tooling.permissions import (
    _load_execution_profile,
    is_action_gated,
    parse_permitted_tools,
)


# ── parse_permitted_tools: action-level entries ────────────────────────────


def test_parse_permitted_tools_includes_action_level_entries() -> None:
    """parse_permitted_tools includes action-level entries like gmail_send."""
    text = "## 外部ツール\n- gmail: yes\n- gmail_send: yes\n"
    result = parse_permitted_tools(text)
    assert "gmail" in result
    assert "gmail_send" in result


def test_parse_permitted_tools_action_only_whitelist() -> None:
    """Action-level only (no tool) in whitelist mode."""
    text = "## 外部ツール\n- gmail_send: yes\n"
    result = parse_permitted_tools(text)
    assert "gmail_send" in result
    assert "gmail" not in result  # gmail not explicitly allowed


# ── is_action_gated ────────────────────────────────────────────────────────


def test_is_action_gated_true_when_gated_and_not_allowed() -> None:
    """is_action_gated returns True for gated action not in permitted."""
    with patch(
        "core.tooling.permissions._load_execution_profile",
        return_value={"send": {"gated": True}},
    ):
        assert is_action_gated("gmail", "send", {"gmail"}) is True


def test_gmail_draft_update_is_gated_by_real_profile() -> None:
    assert is_action_gated("gmail", "draft-update", {"gmail"}) is True
    assert is_action_gated("gmail", "draft-update", {"gmail", "gmail_draft-update"}) is False
    assert is_action_gated("gmail", "draft_update", {"gmail"}) is True
    assert is_action_gated("gmail", "draft_update", {"gmail", "gmail_draft-update"}) is False


def test_is_action_gated_false_when_gated_and_allowed() -> None:
    """is_action_gated returns False for gated action in permitted."""
    with patch(
        "core.tooling.permissions._load_execution_profile",
        return_value={"send": {"gated": True}},
    ):
        assert is_action_gated("gmail", "send", {"gmail", "gmail_send"}) is False


def test_is_action_gated_false_when_non_gated() -> None:
    """is_action_gated returns False for non-gated actions."""
    with patch(
        "core.tooling.permissions._load_execution_profile",
        return_value={"read": {"gated": False}},
    ):
        assert is_action_gated("gmail", "read", {"gmail"}) is False


def test_is_action_gated_false_when_tool_module_not_found() -> None:
    """is_action_gated returns False when tool module not found."""
    with patch(
        "core.tooling.permissions._load_execution_profile",
        return_value=None,
    ):
        assert is_action_gated("nonexistent_tool", "send", set()) is False


def test_is_action_gated_respects_gated_as() -> None:
    """An action with gated_as gates under that name's permission key."""
    with patch(
        "core.tooling.permissions._load_execution_profile",
        return_value={"upload": {"gated": True, "gated_as": "send"}},
    ):
        # Allowed when gated_as permission (chatwork_send) is present.
        assert is_action_gated("chatwork", "upload", {"chatwork_send"}) is False
        # Blocked when not present.
        assert is_action_gated("chatwork", "upload", set()) is True


def test_is_action_gated_false_when_action_not_in_profile() -> None:
    """is_action_gated returns False when action not in EXECUTION_PROFILE."""
    with patch(
        "core.tooling.permissions._load_execution_profile",
        return_value={"read": {"gated": False}},
    ):
        assert is_action_gated("gmail", "send", {"gmail"}) is False


# ── all: yes does NOT auto-allow gated actions ─────────────────────────────


def test_all_yes_does_not_auto_allow_gated_actions() -> None:
    """all: yes does NOT automatically allow gated actions."""
    text = "## 外部ツール\n- all: yes\n"
    result = parse_permitted_tools(text)
    # gmail_send is not in result — all: yes only grants tool-level, not action-level
    assert "gmail_send" not in result
    assert "gmail" in result  # gmail is a tool, so it's in all_tools


def test_all_yes_with_explicit_gated_permit() -> None:
    """all: yes + explicit gmail_send: yes includes gmail_send."""
    text = "## 外部ツール\n- all: yes\n- gmail_send: yes\n"
    result = parse_permitted_tools(text)
    assert "gmail_send" in result
    assert "gmail" in result


# ── Action-level deny ───────────────────────────────────────────────────────


def test_action_level_deny_entries_work() -> None:
    """Action-level deny entries exclude from permitted."""
    text = "## 外部ツール\n- gmail: yes\n- gmail_send: yes\n- gmail_send: no\n"
    result = parse_permitted_tools(text)
    # Deny overrides allow
    assert "gmail_send" not in result
    assert "gmail" in result


def test_action_level_deny_with_all_yes() -> None:
    """Action-level deny with all: yes excludes that action."""
    text = "## 外部ツール\n- all: yes\n- gmail_send: no\n"
    result = parse_permitted_tools(text)
    assert "gmail_send" not in result
    assert "gmail" in result


# ── _load_execution_profile ─────────────────────────────────────────────────


def test_load_execution_profile_returns_profile_for_gmail() -> None:
    """_load_execution_profile returns EXECUTION_PROFILE for gmail."""
    profile = _load_execution_profile("gmail")
    assert profile is not None
    assert "unread" in profile
    assert "draft" in profile


def test_load_execution_profile_returns_none_for_unknown_tool() -> None:
    """_load_execution_profile returns None for unknown tool."""
    profile = _load_execution_profile("nonexistent_tool_xyz")
    assert profile is None
