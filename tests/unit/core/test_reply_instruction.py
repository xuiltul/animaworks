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
from unittest.mock import patch

from core._anima_inbox import _build_reply_instruction


@dataclass
class _FakeMsg:
    """Minimal message stub for testing reply instruction generation."""

    source: str = "anima"
    external_channel_id: str = ""
    source_message_id: str = ""
    external_user_id: str = ""
    external_thread_ts: str = ""


_MOCK_AUTO_OFF = patch("core._anima_inbox._is_auto_response_enabled", return_value=False)
_MOCK_AUTO_ON = patch("core._anima_inbox._is_auto_response_enabled", return_value=True)


class TestBuildReplyInstructionSlack:
    """Slack reply instruction generation."""

    def test_full_metadata(self):
        """Slack message with channel, ts, and user_id produces tool instruction."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="1234567890.123456",
            external_user_id="U99999",
        )
        with _MOCK_AUTO_OFF:
            result = _build_reply_instruction(msg)
        assert "[reply_instruction:" in result
        assert "use tool slack_channel_post" in result
        assert 'channel_id="C12345"' in result
        assert 'text="<@U99999> {返信内容}"' in result
        assert 'thread_ts="1234567890.123456"' in result

    def test_missing_user_id(self):
        """Slack without external_user_id omits @mention."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="1234567890.123456",
            external_user_id="",
        )
        with _MOCK_AUTO_OFF:
            result = _build_reply_instruction(msg)
        assert "<@" not in result
        assert 'thread_ts="1234567890.123456"' in result
        assert 'channel_id="C12345"' in result

    def test_missing_source_message_id(self):
        """Slack without source_message_id omits thread_ts."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="",
            external_user_id="U99999",
        )
        with _MOCK_AUTO_OFF:
            result = _build_reply_instruction(msg)
        assert "thread_ts=" not in result
        assert 'text="<@U99999> {返信内容}"' in result
        assert 'channel_id="C12345"' in result

    def test_missing_both_optional_fields(self):
        """Slack with only channel_id: no @mention and no thread_ts."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="",
            external_user_id="",
        )
        with _MOCK_AUTO_OFF:
            result = _build_reply_instruction(msg)
        assert "[reply_instruction:" in result
        assert 'channel_id="C12345"' in result
        assert "<@" not in result
        assert "thread_ts=" not in result

    def test_auto_response_enabled(self):
        """When auto_response is enabled, returns auto_reply annotation."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C12345",
            source_message_id="1234567890.123456",
            external_user_id="U99999",
        )
        with _MOCK_AUTO_ON:
            result = _build_reply_instruction(msg)
        assert "[auto_reply:" in result
        assert "[reply_instruction:" not in result


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
        with _MOCK_AUTO_OFF:
            result = _build_reply_instruction(msg)
        assert "[reply_instruction:" in result
        assert "slack_channel_post" in result

    def test_reply_instruction_format(self):
        """Verify exact format: '  [reply_instruction: CMD]'."""
        msg = _FakeMsg(
            source="slack",
            external_channel_id="C1",
            source_message_id="ts1",
            external_user_id="U1",
        )
        with _MOCK_AUTO_OFF:
            result = _build_reply_instruction(msg)
        assert result.startswith("  [reply_instruction: ")
        assert result.endswith("]")
