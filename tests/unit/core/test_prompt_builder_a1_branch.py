"""Unit tests for A1/A2 prompt template branching."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from core.paths import PROMPTS_DIR


# ── TestMessagingSectionBranching ─────────────────────────


class TestMessagingSectionBranching:
    """Verify _build_messaging_section selects the correct template by execution_mode."""

    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_a1_mode_loads_messaging_a1_template(self, mock_load: MagicMock) -> None:
        """When execution_mode='a1', the messaging_a1 template must be loaded."""
        from core.prompt.builder import _build_messaging_section

        anima_dir = Path("/tmp/animas/test_anima")
        _build_messaging_section(anima_dir, ["peer1"], execution_mode="a1")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "messaging_a1" in template_names

    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_a2_mode_loads_messaging_template(self, mock_load: MagicMock) -> None:
        """When execution_mode='a2', the standard messaging template must be loaded."""
        from core.prompt.builder import _build_messaging_section

        anima_dir = Path("/tmp/animas/test_anima")
        _build_messaging_section(anima_dir, ["peer1"], execution_mode="a2")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "messaging" in template_names
        assert "messaging_a1" not in template_names

    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_default_execution_mode_is_a1(self, mock_load: MagicMock) -> None:
        """Default execution_mode should be 'a1', loading messaging_a1 template."""
        from core.prompt.builder import _build_messaging_section

        anima_dir = Path("/tmp/animas/test_anima")
        _build_messaging_section(anima_dir, ["peer1"])

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "messaging_a1" in template_names


# ── TestOrgContextCommunicationRules ─────────────────────


class TestOrgContextCommunicationRules:
    """Verify _build_org_context selects the correct communication_rules template."""

    @patch("core.config.load_config")
    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_a1_mode_loads_communication_rules_a1(
        self, mock_load: MagicMock, mock_config: MagicMock
    ) -> None:
        """When execution_mode='a1' and other_animas exist, communication_rules_a1 is loaded."""
        from core.prompt.builder import _build_org_context

        # Set up a minimal config with animas that have supervisor relationships
        mock_anima_cfg = MagicMock()
        mock_anima_cfg.supervisor = None
        mock_anima_cfg.speciality = "engineer"

        mock_peer_cfg = MagicMock()
        mock_peer_cfg.supervisor = "test_anima"
        mock_peer_cfg.speciality = "writer"

        mock_config.return_value.animas = {
            "test_anima": mock_anima_cfg,
            "peer1": mock_peer_cfg,
        }

        _build_org_context("test_anima", ["peer1"], execution_mode="a1")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "communication_rules_a1" in template_names
        assert "communication_rules" not in template_names

    @patch("core.config.load_config")
    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_a2_mode_loads_communication_rules(
        self, mock_load: MagicMock, mock_config: MagicMock
    ) -> None:
        """When execution_mode='a2' and other_animas exist, communication_rules (not _a1) is loaded."""
        from core.prompt.builder import _build_org_context

        mock_anima_cfg = MagicMock()
        mock_anima_cfg.supervisor = None
        mock_anima_cfg.speciality = "engineer"

        mock_peer_cfg = MagicMock()
        mock_peer_cfg.supervisor = "test_anima"
        mock_peer_cfg.speciality = "writer"

        mock_config.return_value.animas = {
            "test_anima": mock_anima_cfg,
            "peer1": mock_peer_cfg,
        }

        _build_org_context("test_anima", ["peer1"], execution_mode="a2")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "communication_rules" in template_names
        assert "communication_rules_a1" not in template_names

    @patch("core.config.load_config")
    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_empty_other_animas_skips_communication_rules(
        self, mock_load: MagicMock, mock_config: MagicMock
    ) -> None:
        """When other_animas is empty, no communication_rules template should be loaded."""
        from core.prompt.builder import _build_org_context

        mock_anima_cfg = MagicMock()
        mock_anima_cfg.supervisor = None
        mock_anima_cfg.speciality = "engineer"

        mock_config.return_value.animas = {
            "test_anima": mock_anima_cfg,
        }

        _build_org_context("test_anima", [], execution_mode="a1")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "communication_rules_a1" not in template_names
        assert "communication_rules" not in template_names


# ── TestHumanNotificationGuidanceBranching ───────────────


class TestHumanNotificationGuidanceBranching:
    """Verify _build_human_notification_guidance selects the correct howto template."""

    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_a1_mode_loads_howto_a1(self, mock_load: MagicMock) -> None:
        """When execution_mode='a1', builder/human_notification_howto_a1 must be loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="a1")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification_howto_a1" in template_names
        assert "builder/human_notification_howto_other" not in template_names

    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_a2_mode_loads_howto_other(self, mock_load: MagicMock) -> None:
        """When execution_mode='a2', builder/human_notification_howto_other must be loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="a2")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification_howto_other" in template_names
        assert "builder/human_notification_howto_a1" not in template_names

    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_empty_mode_loads_howto_other(self, mock_load: MagicMock) -> None:
        """When execution_mode is empty string, builder/human_notification_howto_other must be loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification_howto_other" in template_names
        assert "builder/human_notification_howto_a1" not in template_names

    @patch("core.prompt.builder.load_prompt", return_value="mocked prompt")
    def test_all_modes_load_human_notification_wrapper(self, mock_load: MagicMock) -> None:
        """Regardless of mode, builder/human_notification wrapper template is always loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="a1")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification" in template_names


# ── TestA1TemplateContent ────────────────────────────────


class TestA1TemplateContent:
    """Verify actual A1 template files contain correct MCP tool references."""

    def test_messaging_a1_contains_mcp_send_message(self) -> None:
        """messaging_a1.md must reference the MCP tool name mcp__aw__send_message."""
        content = (PROMPTS_DIR / "messaging_a1.md").read_text(encoding="utf-8")
        assert "mcp__aw__send_message" in content

    def test_messaging_a1_does_not_contain_bash_send(self) -> None:
        """messaging_a1.md must NOT contain 'bash send' (legacy CLI approach removed)."""
        content = (PROMPTS_DIR / "messaging_a1.md").read_text(encoding="utf-8")
        assert "bash send" not in content

    def test_communication_rules_a1_contains_mcp_send_message(self) -> None:
        """communication_rules_a1.md must reference the MCP tool name mcp__aw__send_message."""
        content = (PROMPTS_DIR / "communication_rules_a1.md").read_text(encoding="utf-8")
        assert "mcp__aw__send_message" in content

    def test_communication_rules_a1_no_bare_send_message(self) -> None:
        """communication_rules_a1.md must NOT contain bare 'send_message' without 'mcp__aw__' prefix.

        Every occurrence of 'send_message' in the A1 template should be
        prefixed with 'mcp__aw__', indicating the MCP tool is used
        rather than a generic tool_use call.
        """
        content = (PROMPTS_DIR / "communication_rules_a1.md").read_text(encoding="utf-8")
        # Find all occurrences of 'send_message' and verify each is prefixed
        # with 'mcp__aw__'. We use a negative lookbehind to find bare instances.
        bare_matches = re.findall(r"(?<!mcp__aw__)send_message", content)
        assert bare_matches == [], (
            f"Found bare 'send_message' without 'mcp__aw__' prefix: "
            f"{len(bare_matches)} occurrence(s)"
        )
