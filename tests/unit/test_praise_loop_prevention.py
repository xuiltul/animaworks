from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for praise-loop prevention behavior.

Acknowledgements and thanks should not elicit another reply, while the
one-message-per-recipient run guard remains as a final safety net.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from core.tooling.handler import ToolHandler


# ── Test 1: praise-loop safeguards ──────────────────────────────


@pytest.mark.unit
class TestPraiseLoopPrevention:
    def test_inbox_prompt_says_not_to_reply_to_acknowledgements(self) -> None:
        prompt = (Path(__file__).resolve().parents[2] / "templates/ja/prompts/inbox_message.md").read_text(
            encoding="utf-8"
        )
        assert "単なる了解・感謝・称賛には返信せず" in prompt

    def test_depth_observation_never_discards_a_message(self, tmp_path: Path, caplog) -> None:
        from core.activity.logger import ActivityLogger
        from core.messaging.messenger import Messenger

        shared_dir = tmp_path / "shared"
        sender_dir = tmp_path / "animas" / "alice"
        (tmp_path / "animas" / "bob").mkdir(parents=True)
        (sender_dir / "activity_log").mkdir(parents=True)
        activity = ActivityLogger(sender_dir)
        for _ in range(6):
            activity.log("message_sent", content="Earlier update", to_person="bob", meta={"from_type": "anima"})

        messenger = Messenger(shared_dir, "alice")
        with caplog.at_level(logging.INFO, logger="animaworks.messenger"):
            message = messenger.send("bob", "Thanks for the update", msg_type="board_mention")

        assert message.type == "board_mention"
        inbox_files = list((shared_dir / "inbox" / "bob").glob("*.json"))
        assert len(inbox_files) == 1
        assert json.loads(inbox_files[0].read_text(encoding="utf-8"))["content"] == "Thanks for the update"
        assert "MESSAGE DEPTH OBSERVED" in caplog.text

    def test_second_dm_to_same_recipient_in_one_run_is_rejected(self, tmp_path: Path) -> None:
        from core.config.models import AnimaModelConfig, AnimaWorksConfig
        from core.memory import MemoryManager
        from core.messaging.messenger import Messenger
        from core.tooling.handler import ToolHandler

        animas_dir = tmp_path / "animas"
        sender_dir = animas_dir / "alice"
        target_dir = animas_dir / "bob"
        for anima_dir in (sender_dir, target_dir):
            anima_dir.mkdir(parents=True)
            (anima_dir / "status.json").write_text("{}", encoding="utf-8")
        (sender_dir / "permissions.json").write_text(
            '{"version":1,"file_roots":["/"],"commands":{"allow_all":true,"allow":[],"deny":[]},'
            '"external_tools":{"allow_all":true},"tool_creation":{"personal":true,"shared":false}}',
            encoding="utf-8",
        )
        shared_dir = tmp_path / "shared"
        memory = MagicMock(spec=MemoryManager)
        memory.read_permissions.return_value = ""
        config = AnimaWorksConfig(
            animas={"alice": AnimaModelConfig(), "bob": AnimaModelConfig()},
        )

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=config),
        ):
            handler = ToolHandler(
                anima_dir=sender_dir,
                memory=memory,
                messenger=Messenger(shared_dir, "alice"),
                tool_registry=[],
            )
            first = handler.handle("send_message", {"to": "bob", "content": "Thanks!", "intent": "report"})
            second = handler.handle("send_message", {"to": "bob", "content": "You're welcome!", "intent": "report"})

        assert "Message sent to bob" in first
        assert "Error" in second
        inbox_files = list((shared_dir / "inbox" / "bob").glob("*.json"))
        assert len(inbox_files) == 1


# ── Test 2: _suppress_board_fanout flag in handler ────────────────


@pytest.mark.unit
class TestSuppressBoardFanout:
    """Verify that _suppress_board_fanout flag controls fanout in _handle_post_channel."""

    @pytest.fixture(autouse=True)
    def _bypass_acl(self):
        """Bypass channel ACL checks — these tests use MagicMock messenger."""
        with patch("core.messaging.messenger.is_channel_member", return_value=True):
            yield

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test-anima"
        d.mkdir(parents=True)
        (d / "permissions.md").write_text("", encoding="utf-8")
        (d / "status.json").write_text("{}", encoding="utf-8")
        return d

    @pytest.fixture
    def memory(self) -> MagicMock:
        m = MagicMock()
        m.read_permissions.return_value = ""
        m.search_memory_text.return_value = []
        return m

    @pytest.fixture
    def messenger(self, tmp_path: Path) -> MagicMock:
        m = MagicMock()
        m.anima_name = "test-anima"
        m.shared_dir = tmp_path / "shared"
        (m.shared_dir / "channels").mkdir(parents=True)
        msg = MagicMock()
        msg.id = "msg_001"
        msg.thread_id = "thread_001"
        m.send.return_value = msg
        return m

    @pytest.fixture
    def handler(
        self,
        anima_dir: Path,
        memory: MagicMock,
        messenger: MagicMock,
    ) -> ToolHandler:
        from core.tooling.handler import ToolHandler

        return ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=[],
        )

    def test_post_channel_calls_fanout_when_flag_not_set(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """Without _suppress_board_fanout, _fanout_board_mentions should be called."""
        with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
            handler._handle_post_channel({"channel": "general", "text": "@all hello"})

        mock_fanout.assert_called_once_with("general", "@all hello")

    def test_post_channel_calls_fanout_when_flag_explicitly_false(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """suppress_board_fanout=False should still call fanout normally."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(False)
        try:
            with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
                handler._handle_post_channel({"channel": "dev", "text": "@bob check"})

            mock_fanout.assert_called_once_with("dev", "@bob check")
        finally:
            suppress_board_fanout.reset(token)

    def test_post_channel_suppresses_fanout_when_flag_true(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """suppress_board_fanout=True should skip _fanout_board_mentions."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(True)
        try:
            with patch.object(handler, "_fanout_board_mentions") as mock_fanout:
                handler._handle_post_channel({"channel": "general", "text": "@all thanks"})

            mock_fanout.assert_not_called()
        finally:
            suppress_board_fanout.reset(token)

    def test_post_channel_still_posts_when_fanout_suppressed(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """Even with fanout suppressed, the channel post itself should still succeed."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(True)
        try:
            result = handler._handle_post_channel({"channel": "general", "text": "hello"})

            messenger.post_channel.assert_called_once_with("general", "hello")
            assert "Posted to #general" in result
        finally:
            suppress_board_fanout.reset(token)

    def test_post_channel_logs_suppression(
        self,
        handler: ToolHandler,
        messenger: MagicMock,
    ) -> None:
        """Suppressed fanout should be logged."""
        from core.tooling.handler import suppress_board_fanout

        token = suppress_board_fanout.set(True)
        try:
            with patch("core.tooling.handler_comms.logger") as mock_logger:
                handler._handle_post_channel({"channel": "ops", "text": "@all acknowledged"})

            # Look for the suppression log message
            log_messages = [call.args[0] if call.args else "" for call in mock_logger.info.call_args_list]
            assert any("Suppressed board fanout" in msg for msg in log_messages), (
                f"Expected suppression log message, got: {log_messages}"
            )
        finally:
            suppress_board_fanout.reset(token)

    def test_suppress_flag_defaults_to_false_via_contextvar(
        self,
        handler: ToolHandler,
    ) -> None:
        """suppress_board_fanout ContextVar should default to False."""
        from core.tooling.handler import suppress_board_fanout

        assert suppress_board_fanout.get() is False
