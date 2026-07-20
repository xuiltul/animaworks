"""Unit tests for S/A prompt template branching."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.paths import TEMPLATES_DIR

# ── TestMessagingSectionBranching ─────────────────────────


class TestMessagingSectionBranching:
    """Verify _build_messaging_section selects the correct template by execution_mode."""

    @patch("core.prompt.messaging.load_prompt", return_value="mocked prompt")
    def test_s_mode_loads_messaging_s_template(self, mock_load: MagicMock) -> None:
        """When execution_mode='s', the messaging_s template must be loaded."""
        from core.prompt.builder import _build_messaging_section

        anima_dir = Path("/tmp/animas/test_anima")
        _build_messaging_section(anima_dir, ["peer1"], execution_mode="s")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "messaging_s" in template_names

    @patch("core.prompt.messaging.load_prompt", return_value="mocked prompt")
    def test_a_mode_loads_messaging_template(self, mock_load: MagicMock) -> None:
        """When execution_mode='a', the standard messaging template must be loaded."""
        from core.prompt.builder import _build_messaging_section

        anima_dir = Path("/tmp/animas/test_anima")
        _build_messaging_section(anima_dir, ["peer1"], execution_mode="a")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "messaging" in template_names
        assert "messaging_s" not in template_names

    @patch("core.prompt.messaging.load_prompt", return_value="mocked prompt")
    def test_default_execution_mode_is_s(self, mock_load: MagicMock) -> None:
        """Default execution_mode should be 's', loading messaging_s template."""
        from core.prompt.builder import _build_messaging_section

        anima_dir = Path("/tmp/animas/test_anima")
        _build_messaging_section(anima_dir, ["peer1"])

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "messaging_s" in template_names


# ── TestOrgContextCommunicationRules ─────────────────────


class TestOrgContextCommunicationRules:
    """Verify _build_org_context selects the correct communication_rules template."""

    @patch("core.config.load_config")
    @patch("core.prompt.org_context.load_prompt_text", return_value="mocked prompt")
    @patch("core.prompt.org_context.load_prompt", return_value="mocked prompt")
    def test_s_mode_loads_communication_rules_s(
        self, mock_load: MagicMock, mock_load_text: MagicMock, mock_config: MagicMock
    ) -> None:
        """When execution_mode='s' and other_animas exist, communication_rules_s is loaded."""
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

        _build_org_context("test_anima", ["peer1"], execution_mode="s")

        template_names = [c.args[0] for c in mock_load_text.call_args_list]
        assert "communication_rules_s" in template_names
        assert "communication_rules" not in template_names

    @patch("core.config.load_config")
    @patch("core.prompt.org_context.load_prompt_text", return_value="mocked prompt")
    @patch("core.prompt.org_context.load_prompt", return_value="mocked prompt")
    def test_a_mode_loads_communication_rules(
        self, mock_load: MagicMock, mock_load_text: MagicMock, mock_config: MagicMock
    ) -> None:
        """When execution_mode='a' and other_animas exist, communication_rules (not _s) is loaded."""
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

        _build_org_context("test_anima", ["peer1"], execution_mode="a")

        template_names = [c.args[0] for c in mock_load_text.call_args_list]
        assert "communication_rules" in template_names
        assert "communication_rules_s" not in template_names

    @patch("core.config.load_config")
    @patch("core.prompt.org_context.load_prompt_text", return_value="mocked prompt")
    @patch("core.prompt.org_context.load_prompt", return_value="mocked prompt")
    def test_empty_other_animas_skips_communication_rules(
        self, mock_load: MagicMock, mock_load_text: MagicMock, mock_config: MagicMock
    ) -> None:
        """When other_animas is empty, no communication_rules template should be loaded."""
        from core.prompt.builder import _build_org_context

        mock_anima_cfg = MagicMock()
        mock_anima_cfg.supervisor = None
        mock_anima_cfg.speciality = "engineer"

        mock_config.return_value.animas = {
            "test_anima": mock_anima_cfg,
        }

        _build_org_context("test_anima", [], execution_mode="s")

        template_names = [c.args[0] for c in mock_load_text.call_args_list]
        assert "communication_rules_s" not in template_names
        assert "communication_rules" not in template_names


# ── TestHumanNotificationGuidanceBranching ───────────────


class TestHumanNotificationGuidanceBranching:
    """Verify _build_human_notification_guidance selects the correct howto template."""

    @patch("core.prompt.messaging.load_prompt", return_value="mocked prompt")
    def test_s_mode_loads_howto_s(self, mock_load: MagicMock) -> None:
        """When execution_mode='s', builder/human_notification_howto_s must be loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="s")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification_howto_s" in template_names
        assert "builder/human_notification_howto_other" not in template_names

    @patch("core.prompt.messaging.load_prompt", return_value="mocked prompt")
    def test_a_mode_loads_howto_other(self, mock_load: MagicMock) -> None:
        """When execution_mode='a', builder/human_notification_howto_other must be loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="a")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification_howto_other" in template_names
        assert "builder/human_notification_howto_s" not in template_names

    @patch("core.prompt.messaging.load_prompt", return_value="mocked prompt")
    def test_empty_mode_loads_howto_other(self, mock_load: MagicMock) -> None:
        """When execution_mode is empty string, builder/human_notification_howto_other must be loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification_howto_other" in template_names
        assert "builder/human_notification_howto_s" not in template_names

    @patch("core.prompt.messaging.load_prompt", return_value="mocked prompt")
    def test_all_modes_load_human_notification_wrapper(self, mock_load: MagicMock) -> None:
        """Regardless of mode, builder/human_notification wrapper template is always loaded."""
        from core.prompt.builder import _build_human_notification_guidance

        _build_human_notification_guidance(execution_mode="s")

        template_names = [c.args[0] for c in mock_load.call_args_list]
        assert "builder/human_notification" in template_names


# ── TestSTemplateContent ────────────────────────────────


class TestSTemplateContent:
    """Verify actual S template files contain correct MCP tool references."""

    def test_messaging_s_contains_send_message(self) -> None:
        """messaging_s.md must reference send_message (unified name, no mcp__aw__ prefix)."""
        content = (TEMPLATES_DIR / "ja" / "prompts" / "messaging_s.md").read_text(encoding="utf-8")
        assert "send_message" in content
        assert "mcp__aw__" not in content

    def test_messaging_s_does_not_contain_bash_send(self) -> None:
        """messaging_s.md must NOT contain 'bash send' (legacy CLI approach removed)."""
        content = (TEMPLATES_DIR / "ja" / "prompts" / "messaging_s.md").read_text(encoding="utf-8")
        assert "bash send" not in content

    def test_communication_rules_s_has_routing_rules(self) -> None:
        """communication_rules_s.md must contain routing rules (compressed format)."""
        content = (TEMPLATES_DIR / "ja" / "prompts" / "communication_rules_s.md").read_text(encoding="utf-8")
        assert "経路" in content
