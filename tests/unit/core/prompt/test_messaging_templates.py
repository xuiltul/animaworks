"""Unit tests for messaging prompt templates — S vs A content validation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from core.paths import PROMPTS_DIR


class TestMessagingTemplates:
    def test_messaging_s_template_exists(self):
        """S mode messaging template file must exist."""
        path = PROMPTS_DIR / "messaging_s.md"
        assert path.exists(), f"Missing template: {path}"

    def test_messaging_s_has_mcp_send_message(self):
        """S mode template must reference MCP send_message tool."""
        content = (PROMPTS_DIR / "messaging_s.md").read_text(encoding="utf-8")
        assert "mcp__aw__send_message" in content

    def test_messaging_s_no_bash_send(self):
        """S mode template must NOT reference bash send command (abolished)."""
        content = (PROMPTS_DIR / "messaging_s.md").read_text(encoding="utf-8")
        assert "bash send" not in content

    def test_messaging_s_has_placeholders(self):
        """S mode template must have the required format placeholders."""
        content = (PROMPTS_DIR / "messaging_s.md").read_text(encoding="utf-8")
        assert "{animas_line}" in content

    def test_messaging_a_template_exists(self):
        """Standard messaging template (for A mode) must still exist."""
        path = PROMPTS_DIR / "messaging.md"
        assert path.exists(), f"Missing template: {path}"

    def test_messaging_a_has_send_message_tool(self):
        """A mode template should still reference send_message tool."""
        content = (PROMPTS_DIR / "messaging.md").read_text(encoding="utf-8")
        assert "send_message" in content


class TestHeartbeatDefaultChecklist:
    def test_has_tool_failure_escalation(self):
        """Default heartbeat checklist must include tool failure escalation."""
        content = (PROMPTS_DIR / "heartbeat_default_checklist.md").read_text(encoding="utf-8")
        assert "上司" in content or "報告" in content
        assert "外部ツール" in content or "アクセス" in content
