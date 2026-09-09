"""Regression tests for pi-fix2 gated side-effect tools.

chatwork_send / discord_send / github_create-issue / github_create-pr
require explicit allow; allow_all alone is insufficient.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.models import ExternalToolsPermission, PermissionsConfig
from core.tooling.permissions import get_permitted_tools, is_action_gated


# ── EXECUTION_PROFILE flags ───────────────────────────────────


class TestExecutionProfileGatedFlags:
    """Newly gated actions must advertise gated=True."""

    def test_chatwork_send_gated(self) -> None:
        from core.tools.chatwork import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["send"].get("gated") is True

    def test_discord_send_gated(self) -> None:
        from core.tools.discord import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["send"].get("gated") is True

    def test_discord_channel_post_still_gated(self) -> None:
        from core.tools.discord import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["channel_post"].get("gated") is True

    def test_github_create_issue_gated(self) -> None:
        from core.tools.github import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["create-issue"].get("gated") is True

    def test_github_create_pr_gated(self) -> None:
        from core.tools.github import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["create-pr"].get("gated") is True


# ── is_action_gated: deny without allow ───────────────────────


@pytest.mark.parametrize(
    ("tool", "action", "permit_key"),
    [
        ("chatwork", "send", "chatwork_send"),
        ("discord", "send", "discord_send"),
        ("github", "create-issue", "github_create-issue"),
        ("github", "create-pr", "github_create-pr"),
    ],
)
class TestGatedBlockedWithoutAllow:
    def test_blocked_with_tool_only(
        self, tool: str, action: str, permit_key: str
    ) -> None:
        permitted = {tool}
        assert is_action_gated(tool, action, permitted) is True

    def test_blocked_with_allow_all_tool_set(
        self, tool: str, action: str, permit_key: str
    ) -> None:
        # Simulate allow_all: tool name present, action key absent
        config = PermissionsConfig(
            external_tools=ExternalToolsPermission(allow_all=True, allow=[], deny=[])
        )
        permitted = get_permitted_tools(config)
        # gated action key must not be auto-included
        assert permit_key not in permitted
        assert is_action_gated(tool, action, permitted) is True


# ── is_action_gated: pass with explicit allow ─────────────────


@pytest.mark.parametrize(
    ("tool", "action", "permit_key"),
    [
        ("chatwork", "send", "chatwork_send"),
        ("discord", "send", "discord_send"),
        ("github", "create-issue", "github_create-issue"),
        ("github", "create-pr", "github_create-pr"),
    ],
)
class TestGatedAllowedWithExplicitAllow:
    def test_allowed_with_explicit_key(
        self, tool: str, action: str, permit_key: str
    ) -> None:
        permitted = {tool, permit_key}
        assert is_action_gated(tool, action, permitted) is False

    def test_allow_all_plus_explicit_permit(
        self, tool: str, action: str, permit_key: str
    ) -> None:
        config = PermissionsConfig(
            external_tools=ExternalToolsPermission(
                allow_all=True,
                allow=[permit_key],
                deny=[],
            )
        )
        permitted = get_permitted_tools(config)
        assert permit_key in permitted
        assert is_action_gated(tool, action, permitted) is False


# ── Non-write github / discord reads stay non-gated ───────────


class TestReadActionsRemainOpen:
    def test_github_issues_not_gated(self) -> None:
        from core.tools.github import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["issues"].get("gated") is not True
        assert is_action_gated("github", "issues", {"github"}) is False

    def test_discord_messages_not_gated(self) -> None:
        from core.tools.discord import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["messages"].get("gated") is not True
        assert is_action_gated("discord", "messages", {"discord"}) is False

    def test_chatwork_rooms_not_gated(self) -> None:
        from core.tools.chatwork import EXECUTION_PROFILE

        assert EXECUTION_PROFILE["rooms"].get("gated") is not True
        assert is_action_gated("chatwork", "rooms", {"chatwork"}) is False


# ── Representative anima permission shapes ────────────────────


class TestRepresentativeAnimaShapes:
    """mei-like (chatwork) and engineer-like (github) allow lists."""

    def test_mei_shape_allows_chatwork_send(self) -> None:
        config = PermissionsConfig(
            external_tools=ExternalToolsPermission(
                allow_all=True,
                allow=["slack_send", "slack_channel_post", "chatwork_send"],
                deny=["machine"],
            )
        )
        permitted = get_permitted_tools(config)
        assert is_action_gated("chatwork", "send", permitted) is False

    def test_engineer_shape_allows_github_writes(self) -> None:
        config = PermissionsConfig(
            external_tools=ExternalToolsPermission(
                allow_all=True,
                allow=[
                    "discord_channel_post",
                    "discord_send",
                    "github_create-issue",
                    "github_create-pr",
                ],
                deny=["machine"],
            )
        )
        permitted = get_permitted_tools(config)
        assert is_action_gated("github", "create-issue", permitted) is False
        assert is_action_gated("github", "create-pr", permitted) is False
        assert is_action_gated("discord", "send", permitted) is False


# ── Migration script dry-run unit (no live ~/.animaworks writes) ──


class TestMigrationScriptDryRun:
    def test_dry_run_plans_adds_for_policy_animas(self, tmp_path: Path) -> None:
        from scripts.migrate_pi_fix2_gated_allows import DEFAULT_ALLOWS, run

        animas = tmp_path / "animas"
        # mei: should get chatwork_send + discord_send + github keys
        mei = animas / "mei"
        mei.mkdir(parents=True)
        (mei / "permissions.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "external_tools": {
                        "allow_all": True,
                        "allow": ["slack_send", "slack_channel_post"],
                        "deny": ["machine"],
                    },
                }
            ),
            encoding="utf-8",
        )
        # yoru: not on chatwork/discord/github lists → no adds
        yoru = animas / "yoru"
        yoru.mkdir(parents=True)
        (yoru / "permissions.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "external_tools": {
                        "allow_all": True,
                        "allow": [],
                        "deny": ["machine"],
                    },
                }
            ),
            encoding="utf-8",
        )
        # sumire: github only (not chatwork per PdM)
        sumire = animas / "sumire"
        sumire.mkdir(parents=True)
        (sumire / "permissions.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "external_tools": {
                        "allow_all": True,
                        "allow": ["slack_send"],
                        "deny": ["machine"],
                    },
                }
            ),
            encoding="utf-8",
        )

        results = run(tmp_path, apply=False, policy=DEFAULT_ALLOWS)
        by_name = {r["anima"]: r for r in results}

        assert by_name["mei"]["status"] == "would_apply"
        assert "chatwork_send" in by_name["mei"]["adds"]
        assert "discord_send" in by_name["mei"]["adds"]
        assert "github_create-issue" in by_name["mei"]["adds"]
        assert "github_create-pr" in by_name["mei"]["adds"]

        assert by_name["yoru"]["status"] == "noop"
        assert by_name["yoru"]["adds"] == []

        assert by_name["sumire"]["status"] == "would_apply"
        assert "chatwork_send" not in by_name["sumire"]["adds"]
        assert "github_create-issue" in by_name["sumire"]["adds"]
        assert "github_create-pr" in by_name["sumire"]["adds"]

        # dry-run must not write
        mei_after = json.loads((mei / "permissions.json").read_text(encoding="utf-8"))
        assert "chatwork_send" not in mei_after["external_tools"]["allow"]

    def test_apply_is_idempotent(self, tmp_path: Path) -> None:
        from scripts.migrate_pi_fix2_gated_allows import DEFAULT_ALLOWS, run

        animas = tmp_path / "animas"
        sakura = animas / "sakura"
        sakura.mkdir(parents=True)
        (sakura / "permissions.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "external_tools": {
                        "allow_all": True,
                        "allow": ["slack_send"],
                        "deny": ["machine"],
                    },
                }
            ),
            encoding="utf-8",
        )

        r1 = run(tmp_path, apply=True, policy=DEFAULT_ALLOWS)
        assert r1[0]["status"] == "applied"
        allow1 = json.loads((sakura / "permissions.json").read_text(encoding="utf-8"))[
            "external_tools"
        ]["allow"]
        assert allow1.count("chatwork_send") == 1

        r2 = run(tmp_path, apply=True, policy=DEFAULT_ALLOWS)
        assert r2[0]["status"] == "noop"
        allow2 = json.loads((sakura / "permissions.json").read_text(encoding="utf-8"))[
            "external_tools"
        ]["allow"]
        assert allow2 == allow1
