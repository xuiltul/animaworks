from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _build_reply_instruction() in core._anima_inbox.

Covers:
- Slack: full metadata (channel, ts, user_id)
- Slack: missing external_user_id (no @mention)
- Slack: missing source_message_id (no --thread)
- Chatwork: full metadata
- Edge case: empty external_channel_id (guard in caller, but function still safe)
- Unknown source: returns empty string
"""

from dataclasses import dataclass

import pytest

from core._anima_inbox import _build_reply_instruction


@dataclass
class _FakeMsg:
    """Minimal message stub for testing reply instruction generation."""

    source: str = "anima"
    external_channel_id: str = ""
    source_message_id: str = ""
    external_user_id: str = ""


class TestBuildReplyInstructionSlack:
    """Slack reply instruction generation."""

    def test_full_metadata(self):
        """Slack message with channel, ts, and user_id produces full command."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="1234567890.123456",
            external_user_id="U99999",
        )
        result = _build_reply_instruction(msg)
        assert "[reply_instruction:" in result
        assert "animaworks-tool slack send" in result
        assert "'C12345'" in result
        assert "<@U99999>" in result
        assert "--thread 1234567890.123456" in result

    def test_missing_user_id(self):
        """Slack without external_user_id omits @mention."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="1234567890.123456",
            external_user_id="",
        )
        result = _build_reply_instruction(msg)
        assert "<@" not in result
        assert "--thread 1234567890.123456" in result
        assert "'C12345'" in result

    def test_missing_source_message_id(self):
        """Slack without source_message_id omits --thread."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="",
            external_user_id="U99999",
        )
        result = _build_reply_instruction(msg)
        assert "--thread" not in result
        assert "<@U99999>" in result
        assert "'C12345'" in result

    def test_missing_both_optional_fields(self):
        """Slack with only channel_id: no @mention and no --thread."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="",
            external_user_id="",
        )
        result = _build_reply_instruction(msg)
        assert "[reply_instruction:" in result
        assert "'C12345'" in result
        assert "<@" not in result
        assert "--thread" not in result


class TestBuildReplyInstructionChatwork:
    """Chatwork reply instruction generation."""

    def test_full_metadata(self):
        """Chatwork message produces chatwork send command."""
        msg = _FakeMsg(
            source="chatwork",
            external_channel_id="12345678",
            source_message_id="",
            external_user_id="",
        )
        result = _build_reply_instruction(msg)
        assert "[reply_instruction:" in result
        assert "animaworks-tool chatwork send" in result
        assert "12345678" in result


class TestBuildReplyInstructionEdgeCases:
    """Edge cases and unknown sources."""

    def test_unknown_source_returns_empty(self):
        """Non-slack/chatwork source returns empty string."""
        msg = _FakeMsg(source="anima", external_channel_id="C123")
        assert _build_reply_instruction(msg) == ""

    def test_human_source_returns_empty(self):
        """Human source returns empty string."""
        msg = _FakeMsg(source="human", external_channel_id="C123")
        assert _build_reply_instruction(msg) == ""

    def test_slack_empty_channel(self):
        """Slack with empty channel_id still generates instruction (caller guards)."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="",
            source_message_id="ts123",
            external_user_id="U123",
        )
        result = _build_reply_instruction(msg)
        assert "[reply_instruction:" in result

    def test_reply_instruction_format(self):
        """Verify exact format: '  [reply_instruction: CMD]'."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C1",
            source_message_id="ts1",
            external_user_id="U1",
        )
        result = _build_reply_instruction(msg)
        assert result.startswith("  [reply_instruction: ")
        assert result.endswith("]")
