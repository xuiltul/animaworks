# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for A1 messaging integration.

Verifies that:
- A1 mode system prompts contain MCP tool instructions (mcp__aw__send_message)
- A2 mode system prompts still reference send_message tool
- Anima creation does NOT place legacy send/board wrapper scripts
- Heartbeat default checklist includes tool failure escalation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.manager import MemoryManager
from core.prompt.builder import build_system_prompt


class TestA1MessagingPrompt:
    """Verify system prompts differ by execution mode for messaging."""

    def test_a1_prompt_has_mcp_send_message(self, data_dir, make_anima):
        """A1 system prompt should reference mcp__aw__send_message tool."""
        anima_dir = make_anima("alice")
        # Create a second anima so messaging section is populated
        make_anima("bob", supervisor="alice")

        memory = MemoryManager(anima_dir)
        prompt = build_system_prompt(memory, execution_mode="a1")

        assert "mcp__aw__send_message" in prompt
        # bash send should NOT appear (abolished)
        assert "bash send" not in prompt

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


class TestSendScriptRemoved:
    """Verify that anima creation does NOT place legacy send/board scripts."""

    def test_create_blank_does_not_place_send_script(self, tmp_path):
        """create_blank should NOT place the send script (MCP replaces it)."""
        from core.anima_factory import create_blank

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        anima_dir = create_blank(animas_dir, "testanima")
        send_script = anima_dir / "send"

        assert not send_script.exists(), (
            "send script should NOT exist — MCP tools replaced bash send"
        )


class TestHeartbeatEscalation:
    """Verify heartbeat default checklist includes tool failure escalation."""

    def test_default_checklist_has_escalation(self):
        """The default heartbeat checklist should instruct escalation on tool failure."""
        from core.paths import load_prompt

        checklist = load_prompt("heartbeat_default_checklist")
        assert "外部ツール" in checklist
        assert "上司" in checklist or "報告" in checklist
