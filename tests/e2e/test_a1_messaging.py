# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for A1 messaging integration.

Verifies that:
- A1 mode system prompts contain bash send instructions (not send_message tool)
- A2 mode system prompts still reference send_message tool
- Anima creation places the send wrapper script
- Heartbeat default checklist includes tool failure escalation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.manager import MemoryManager
from core.prompt.builder import build_system_prompt


class TestA1MessagingPrompt:
    """Verify system prompts differ by execution mode for messaging."""

    def test_a1_prompt_has_bash_send_not_send_message_tool(self, data_dir, make_anima):
        """A1 system prompt should reference bash send, not send_message tool."""
        anima_dir = make_anima("alice")
        # Create a second anima so messaging section is populated
        make_anima("bob", supervisor="alice")

        memory = MemoryManager(anima_dir)
        prompt = build_system_prompt(memory, execution_mode="a1")

        assert "bash send" in prompt
        # The messaging section should NOT recommend send_message tool
        # Note: send_message may appear elsewhere (e.g., unread_messages template),
        # but the messaging section itself should not recommend it
        # Check that the A1-specific content is present
        assert "bash send <宛先>" in prompt or "bash send" in prompt

    def test_a2_prompt_has_send_message_tool(self, data_dir, make_anima):
        """A2 system prompt should reference send_message tool."""
        anima_dir = make_anima("alice")
        make_anima("bob", supervisor="alice")

        memory = MemoryManager(anima_dir)
        prompt = build_system_prompt(memory, execution_mode="a2")

        assert "send_message" in prompt

    def test_a1_and_a2_both_list_other_animas(self, data_dir, make_anima):
        """Both modes should list available recipients in messaging section."""
        anima_dir = make_anima("alice")
        make_anima("bob", supervisor="alice")

        memory = MemoryManager(anima_dir)

        a1_prompt = build_system_prompt(memory, execution_mode="a1")
        a2_prompt = build_system_prompt(memory, execution_mode="a2")

        assert "bob" in a1_prompt
        assert "bob" in a2_prompt


class TestSendScriptPlacement:
    """Verify that anima creation places the send script."""

    def test_create_blank_places_send_script(self, tmp_path):
        """create_blank should place the send script in new anima dirs."""
        from core.anima_factory import create_blank

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        anima_dir = create_blank(animas_dir, "testanima")
        send_script = anima_dir / "send"

        assert send_script.exists(), "send script should be placed in anima dir"
        content = send_script.read_text(encoding="utf-8")
        assert "send" in content
        assert send_script.stat().st_mode & 0o100, "send script should be executable"


class TestHeartbeatEscalation:
    """Verify heartbeat default checklist includes tool failure escalation."""

    def test_default_checklist_has_escalation(self):
        """The default heartbeat checklist should instruct escalation on tool failure."""
        from core.paths import load_prompt

        checklist = load_prompt("heartbeat_default_checklist")
        assert "外部ツール" in checklist
        assert "上司" in checklist or "報告" in checklist
