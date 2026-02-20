"""Unit tests for heartbeat decomposition: 5 extracted private methods + InboxResult."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from core.schemas import CycleResult, Message


# ── Helpers ───────────────────────────────────────────────


def _make_cycle_result(**kwargs) -> CycleResult:
    defaults = dict(trigger="test", action="responded", summary="done", duration_ms=100)
    defaults.update(kwargs)
    return CycleResult(**defaults)


def _create_anima(anima_dir, shared_dir, **extra_patches):
    """Create a DigitalAnima with standard patches.  Returns (dp, mocks_dict)."""
    patcher_agent = patch("core.anima.AgentCore")
    patcher_mm = patch("core.anima.MemoryManager")
    patcher_msg = patch("core.anima.Messenger")
    patcher_lp = patch("core.anima.load_prompt", side_effect=lambda name, **kw: f"<{name}>")

    MockAgent = patcher_agent.start()
    MockMM = patcher_mm.start()
    MockMsg = patcher_msg.start()
    MockLP = patcher_lp.start()

    MockMM.return_value.read_model_config.return_value = MagicMock()
    MockMM.return_value.read_heartbeat_config.return_value = "default checklist"
    MockMsg.return_value.has_unread.return_value = False
    MockMsg.return_value.unread_count.return_value = 0

    from core.anima import DigitalAnima
    dp = DigitalAnima(anima_dir, shared_dir)

    # Ensure run dir exists for replied_to file
    (anima_dir / "run").mkdir(parents=True, exist_ok=True)

    mocks = {
        "agent": MockAgent,
        "memory_manager": MockMM,
        "messenger": MockMsg,
        "load_prompt": MockLP,
        "patchers": [patcher_agent, patcher_mm, patcher_msg, patcher_lp],
    }
    return dp, mocks


def _stop_patches(mocks):
    for p in mocks["patchers"]:
        p.stop()


def _make_inbox_item(from_person: str, content: str, path: Path | None = None):
    """Build a lightweight InboxItem-like object."""
    from core.messenger import InboxItem
    msg = Message(
        from_person=from_person,
        to_person="alice",
        content=content,
    )
    if path is None:
        path = Path(f"/tmp/fake_inbox/{from_person}_{id(msg)}.json")
    return InboxItem(msg=msg, path=path)


# ── InboxResult dataclass ─────────────────────────────────


class TestInboxResult:
    def test_default_values(self):
        from core.anima import InboxResult
        ir = InboxResult()
        assert ir.inbox_items == []
        assert ir.messages == []
        assert ir.senders == set()
        assert ir.unread_count == 0
        assert ir.prompt_parts == []

    def test_custom_values(self):
        from core.anima import InboxResult
        item = _make_inbox_item("bob", "hello")
        ir = InboxResult(
            inbox_items=[item],
            messages=[item.msg],
            senders={"bob"},
            unread_count=1,
            prompt_parts=["prompt part"],
        )
        assert len(ir.inbox_items) == 1
        assert ir.senders == {"bob"}
        assert ir.unread_count == 1
        assert ir.prompt_parts == ["prompt part"]

    def test_is_dataclass_with_expected_fields(self):
        from core.anima import InboxResult
        names = {f.name for f in fields(InboxResult)}
        assert names == {"inbox_items", "messages", "senders", "unread_count", "prompt_parts"}

    def test_mutable_defaults_are_independent(self):
        from core.anima import InboxResult
        ir1 = InboxResult()
        ir2 = InboxResult()
        ir1.inbox_items.append("x")
        assert ir2.inbox_items == []

    def test_senders_set_operations(self):
        from core.anima import InboxResult
        ir = InboxResult(senders={"alice", "bob"})
        assert "alice" in ir.senders
        assert len(ir.senders) == 2


# ── _build_heartbeat_prompt ──────────────────────────────


class TestBuildHeartbeatPrompt:
    async def test_basic_prompt_parts(self, data_dir, make_anima):
        """Basic call returns at least the heartbeat checklist prompt."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            # Stub methods that _build_heartbeat_prompt calls
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            assert isinstance(parts, list)
            assert len(parts) >= 1
            # First part should be the heartbeat prompt template
            assert "<heartbeat>" in parts[0]
        finally:
            _stop_patches(mocks)

    async def test_recovery_note_present(self, data_dir, make_anima):
        """Recovery note is read, appended, then deleted."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            recovery_path = anima_dir / "state" / "recovery_note.md"
            recovery_path.write_text("Previous crash info", encoding="utf-8")

            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            # Should contain recovery note content
            recovery_parts = [p for p in parts if "前回のハートビート障害情報" in p]
            assert len(recovery_parts) == 1
            assert "Previous crash info" in recovery_parts[0]
            # Recovery note file should be removed
            assert not recovery_path.exists()
        finally:
            _stop_patches(mocks)

    async def test_recovery_note_absent(self, data_dir, make_anima):
        """No recovery note file -- no recovery section in prompt."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            recovery_path = anima_dir / "state" / "recovery_note.md"
            if recovery_path.exists():
                recovery_path.unlink()

            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            recovery_parts = [p for p in parts if "前回のハートビート障害情報" in p]
            assert len(recovery_parts) == 0
        finally:
            _stop_patches(mocks)

    async def test_background_notifications_present(self, data_dir, make_anima):
        """Background notifications are included when present."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(
                return_value=["Task A done", "Task B done"]
            )

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            bg_parts = [p for p in parts if "バックグラウンドタスク完了通知" in p]
            assert len(bg_parts) == 1
            assert "Task A done" in bg_parts[0]
            assert "Task B done" in bg_parts[0]
        finally:
            _stop_patches(mocks)

    async def test_background_notifications_absent(self, data_dir, make_anima):
        """No background notification section when list is empty."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            bg_parts = [p for p in parts if "バックグラウンドタスク完了通知" in p]
            assert len(bg_parts) == 0
        finally:
            _stop_patches(mocks)

    async def test_heartbeat_history_present(self, data_dir, make_anima):
        """Heartbeat history text triggers a heartbeat_history prompt."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="- 10:00: checked email")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            # load_prompt("heartbeat_history", ...) is called; our stub returns "<heartbeat_history>"
            history_parts = [p for p in parts if "heartbeat_history" in p]
            assert len(history_parts) == 1
        finally:
            _stop_patches(mocks)

    async def test_heartbeat_history_absent(self, data_dir, make_anima):
        """Empty history string -- no heartbeat_history section."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            history_parts = [p for p in parts if "heartbeat_history" in p]
            assert len(history_parts) == 0
        finally:
            _stop_patches(mocks)

    async def test_dialogue_context_with_turns(self, data_dir, make_anima):
        """Recent conversation turns are injected as dialogue context."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            mock_turn = MagicMock()
            mock_turn.role = "human"
            mock_turn.content = "What's the status?"

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(
                    turns=[mock_turn]
                )
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            dialogue_parts = [p for p in parts if "直近の対話履歴" in p]
            assert len(dialogue_parts) == 1
            assert "What's the status?" in dialogue_parts[0]
        finally:
            _stop_patches(mocks)

    async def test_dialogue_context_no_turns(self, data_dir, make_anima):
        """Empty turns list -- no dialogue section."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            dialogue_parts = [p for p in parts if "直近の対話履歴" in p]
            assert len(dialogue_parts) == 0
        finally:
            _stop_patches(mocks)

    async def test_dialogue_context_exception_handled(self, data_dir, make_anima):
        """ConversationMemory failure is caught silently."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory", side_effect=RuntimeError("conv error")), \
                 patch("core.config.models.load_config") as MockCfg:
                MockCfg.return_value.animas = {}

                # Should not raise
                parts = await dp._build_heartbeat_prompt()

            dialogue_parts = [p for p in parts if "直近の対話履歴" in p]
            assert len(dialogue_parts) == 0
        finally:
            _stop_patches(mocks)

    async def test_delegation_check_with_subordinates(self, data_dir, make_anima):
        """Delegation check injected when alice has subordinates."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            sub_config = MagicMock()
            sub_config.supervisor = "alice"

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {"bob": sub_config, "charlie": sub_config}

                parts = await dp._build_heartbeat_prompt()

            delegation_parts = [p for p in parts if "heartbeat_delegation_check" in p]
            assert len(delegation_parts) == 1
        finally:
            _stop_patches(mocks)

    async def test_delegation_check_without_subordinates(self, data_dir, make_anima):
        """No delegation check when there are no subordinates."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            delegation_parts = [p for p in parts if "heartbeat_delegation_check" in p]
            assert len(delegation_parts) == 0
        finally:
            _stop_patches(mocks)

    async def test_delegation_check_exception_handled(self, data_dir, make_anima):
        """load_config failure in delegation check is caught silently."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config", side_effect=RuntimeError("config error")):
                MockConv.return_value.load.return_value = MagicMock(turns=[])

                # Should not raise
                parts = await dp._build_heartbeat_prompt()

            delegation_parts = [p for p in parts if "heartbeat_delegation_check" in p]
            assert len(delegation_parts) == 0
        finally:
            _stop_patches(mocks)

    async def test_heartbeat_config_fallback_to_default(self, data_dir, make_anima):
        """When memory.read_heartbeat_config() returns empty, fallback to default prompt."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.memory.read_heartbeat_config.return_value = ""
            dp._load_heartbeat_history = MagicMock(return_value="")
            dp.drain_background_notifications = MagicMock(return_value=[])

            with patch("core.anima.ConversationMemory") as MockConv, \
                 patch("core.config.models.load_config") as MockCfg:
                MockConv.return_value.load.return_value = MagicMock(turns=[])
                MockCfg.return_value.animas = {}

                parts = await dp._build_heartbeat_prompt()

            # load_prompt should have been called with "heartbeat_default_checklist"
            # first (to get the fallback checklist), then with "heartbeat"
            lp_calls = mocks["load_prompt"].call_args_list
            checklist_call = [c for c in lp_calls if c[0][0] == "heartbeat_default_checklist"]
            assert len(checklist_call) == 1
        finally:
            _stop_patches(mocks)


# ── _process_inbox_messages ──────────────────────────────


class TestProcessInboxMessages:
    async def test_empty_inbox(self, data_dir, make_anima):
        """No unread messages returns default InboxResult."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.messenger.has_unread.return_value = False

            from core.anima import InboxResult
            result = await dp._process_inbox_messages()

            assert isinstance(result, InboxResult)
            assert result.inbox_items == []
            assert result.messages == []
            assert result.senders == set()
            assert result.unread_count == 0
            assert result.prompt_parts == []
        finally:
            _stop_patches(mocks)

    async def test_messages_present(self, data_dir, make_anima):
        """Messages in inbox are processed and returned in InboxResult."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Create inbox directory and a real file for read_counts pruning
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg_file = inbox_dir / "msg_001.json"
        msg_file.write_text("{}", encoding="utf-8")

        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.messenger.has_unread.return_value = True

            item = _make_inbox_item("bob", "Hello Alice!", msg_file)
            dp.messenger.receive_with_paths.return_value = [item]

            # Mock dedup to avoid import issues
            with patch("core.memory.dedup.MessageDeduplicator") as MockDedup, \
                 patch("core.anima.ActivityLogger"):
                dedup_inst = MockDedup.return_value
                dedup_inst.load_deferred.return_value = []
                dedup_inst.apply_rate_limit.return_value = ([item.msg], [])
                dedup_inst.consolidate_messages.return_value = ([item.msg], [])
                dp.memory.read_resolutions = MagicMock(return_value=[])
                dp.memory.append_episode = MagicMock()

                result = await dp._process_inbox_messages()

            assert result.unread_count == 1
            assert "bob" in result.senders
            assert len(result.inbox_items) == 1
            assert len(result.prompt_parts) >= 1
        finally:
            _stop_patches(mocks)

    async def test_cascade_suppression(self, data_dir, make_anima):
        """Messages from cascade-suppressed senders are filtered out."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg_file_bob = inbox_dir / "msg_bob.json"
        msg_file_bob.write_text("{}", encoding="utf-8")
        msg_file_eve = inbox_dir / "msg_eve.json"
        msg_file_eve.write_text("{}", encoding="utf-8")

        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.messenger.has_unread.return_value = True

            item_bob = _make_inbox_item("bob", "Hi from bob", msg_file_bob)
            item_eve = _make_inbox_item("eve", "Hi from eve", msg_file_eve)
            dp.messenger.receive_with_paths.return_value = [item_bob, item_eve]

            with patch("core.memory.dedup.MessageDeduplicator") as MockDedup, \
                 patch("core.anima.ActivityLogger"):
                dedup_inst = MockDedup.return_value
                dedup_inst.load_deferred.return_value = []
                # After cascade filtering, only bob remains
                dedup_inst.apply_rate_limit.side_effect = lambda msgs: (msgs, [])
                dedup_inst.consolidate_messages.side_effect = lambda msgs: (msgs, [])
                dp.memory.read_resolutions = MagicMock(return_value=[])
                dp.memory.append_episode = MagicMock()

                result = await dp._process_inbox_messages(
                    cascade_suppressed_senders={"eve"},
                )

            # Eve should be suppressed; only Bob remains
            assert "bob" in result.senders
            assert "eve" not in result.senders
            assert result.unread_count == 1
        finally:
            _stop_patches(mocks)

    async def test_dedup_failure_uses_original_messages(self, data_dir, make_anima):
        """When dedup import fails, original messages are used."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg_file = inbox_dir / "msg_001.json"
        msg_file.write_text("{}", encoding="utf-8")

        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.messenger.has_unread.return_value = True

            item = _make_inbox_item("bob", "Hello!", msg_file)
            dp.messenger.receive_with_paths.return_value = [item]

            with patch("core.memory.dedup.MessageDeduplicator", side_effect=ImportError("no dedup")), \
                 patch("core.anima.ActivityLogger"):
                dp.memory.append_episode = MagicMock()

                # Should not raise
                result = await dp._process_inbox_messages()

            assert result.unread_count == 1
            assert "bob" in result.senders
        finally:
            _stop_patches(mocks)

    async def test_episode_recording_for_inbox_messages(self, data_dir, make_anima):
        """Received messages are recorded as episodes."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg_file = inbox_dir / "msg_001.json"
        msg_file.write_text("{}", encoding="utf-8")

        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.messenger.has_unread.return_value = True

            item = _make_inbox_item("bob", "Important message", msg_file)
            dp.messenger.receive_with_paths.return_value = [item]

            with patch("core.memory.dedup.MessageDeduplicator") as MockDedup, \
                 patch("core.anima.ActivityLogger"):
                dedup_inst = MockDedup.return_value
                dedup_inst.load_deferred.return_value = []
                dedup_inst.apply_rate_limit.return_value = ([item.msg], [])
                dedup_inst.consolidate_messages.return_value = ([item.msg], [])
                dp.memory.read_resolutions = MagicMock(return_value=[])
                dp.memory.append_episode = MagicMock()

                await dp._process_inbox_messages()

            dp.memory.append_episode.assert_called_once()
            episode_content = dp.memory.append_episode.call_args[0][0]
            assert "bob" in episode_content
            assert "Important message" in episode_content
        finally:
            _stop_patches(mocks)


# ── _execute_heartbeat_cycle ─────────────────────────────


class TestExecuteHeartbeatCycle:
    async def test_normal_cycle_done(self, data_dir, make_anima):
        """Normal streaming cycle with cycle_done event returns CycleResult."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp._heartbeat_stream_queue = None

            async def mock_stream(prompt, trigger="heartbeat"):
                yield {"type": "text_delta", "text": "Hello "}
                yield {"type": "text_delta", "text": "world"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Checked email and tasks",
                        "duration_ms": 500,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                result = await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=0,
                )

            assert isinstance(result, CycleResult)
            assert result.summary == "Checked email and tasks"
            assert result.duration_ms == 500
            assert result.trigger == "heartbeat"
            assert dp._last_activity is not None
        finally:
            _stop_patches(mocks)

    async def test_no_cycle_done_fallback(self, data_dir, make_anima):
        """When no cycle_done event is yielded, fallback CycleResult is created."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp._heartbeat_stream_queue = None

            async def mock_stream(prompt, trigger="heartbeat"):
                yield {"type": "text_delta", "text": "Some text"}
                # No cycle_done event

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                result = await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=0,
                )

            assert isinstance(result, CycleResult)
            assert result.trigger == "heartbeat"
            assert result.action == "responded"
        finally:
            _stop_patches(mocks)

    async def test_checkpoint_file_created_and_cleaned(self, data_dir, make_anima):
        """Checkpoint is written before execution and removed after success."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp._heartbeat_stream_queue = None

            checkpoint_path = anima_dir / "state" / "heartbeat_checkpoint.json"
            checkpoint_observed = {"exists_during_run": False}

            async def mock_stream(prompt, trigger="heartbeat"):
                # Check if checkpoint was written before streaming starts
                checkpoint_observed["exists_during_run"] = checkpoint_path.exists()
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "ok", "duration_ms": 10},
                }

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=0,
                )

            assert checkpoint_observed["exists_during_run"] is True
            # After success, checkpoint is removed
            assert not checkpoint_path.exists()
        finally:
            _stop_patches(mocks)

    async def test_relay_queue_receives_chunks(self, data_dir, make_anima):
        """When _heartbeat_stream_queue is set, text_delta chunks are relayed."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            queue = asyncio.Queue()
            dp._heartbeat_stream_queue = queue

            async def mock_stream(prompt, trigger="heartbeat"):
                yield {"type": "text_delta", "text": "chunk1"}
                yield {"type": "text_delta", "text": "chunk2"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "ok", "duration_ms": 10},
                }

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=0,
                )

            # Queue should have: chunk1, chunk2, None (sentinel)
            items = []
            while not queue.empty():
                items.append(await queue.get())
            assert len(items) == 3
            assert items[0]["text"] == "chunk1"
            assert items[1]["text"] == "chunk2"
            assert items[2] is None  # sentinel
        finally:
            _stop_patches(mocks)

    async def test_relay_queue_sentinel_without_queue(self, data_dir, make_anima):
        """When _heartbeat_stream_queue is None, no error on sentinel."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp._heartbeat_stream_queue = None

            async def mock_stream(prompt, trigger="heartbeat"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "ok", "duration_ms": 10},
                }

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                # Should not raise
                result = await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=0,
                )

            assert result.summary == "ok"
        finally:
            _stop_patches(mocks)

    async def test_episode_recorded_for_non_heartbeat_ok(self, data_dir, make_anima):
        """Heartbeat results without HEARTBEAT_OK are recorded as episodes."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp._heartbeat_stream_queue = None

            async def mock_stream(prompt, trigger="heartbeat"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "summary": "Replied to 3 messages",
                        "duration_ms": 200,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=2,
                )

            dp.memory.append_episode.assert_called_once()
            episode_text = dp.memory.append_episode.call_args[0][0]
            assert "Replied to 3 messages" in episode_text
            assert "2件のメッセージを処理" in episode_text
        finally:
            _stop_patches(mocks)

    async def test_episode_not_recorded_for_heartbeat_ok(self, data_dir, make_anima):
        """Results containing HEARTBEAT_OK are NOT recorded as episodes."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp._heartbeat_stream_queue = None

            async def mock_stream(prompt, trigger="heartbeat"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "summary": "HEARTBEAT_OK - nothing to do",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=0,
                )

            dp.memory.append_episode.assert_not_called()
        finally:
            _stop_patches(mocks)

    async def test_replied_to_file_cleared(self, data_dir, make_anima):
        """replied_to.jsonl is deleted at the start of execution."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp._heartbeat_stream_queue = None

            replied_to_path = anima_dir / "run" / "replied_to.jsonl"
            replied_to_path.write_text("old data", encoding="utf-8")

            async def mock_stream(prompt, trigger="heartbeat"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {"summary": "ok", "duration_ms": 10},
                }

            dp.agent.run_cycle_streaming = mock_stream

            with patch("core.anima.StreamingJournal") as MockSJ, \
                 patch("core.anima.ActivityLogger"), \
                 patch("core.anima.ConversationMemory") as MockConv:
                MockConv.return_value.finalize_if_session_ended = AsyncMock()
                dp.memory.append_episode = MagicMock()

                await dp._execute_heartbeat_cycle(
                    "test prompt", inbox_items=[], unread_count=0,
                )

            assert not replied_to_path.exists()
        finally:
            _stop_patches(mocks)


# ── _archive_processed_messages ──────────────────────────


class TestArchiveProcessedMessages:
    async def test_all_replied(self, data_dir, make_anima):
        """All senders replied to -- all messages archived."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            item_bob = _make_inbox_item("bob", "msg from bob")
            item_eve = _make_inbox_item("eve", "msg from eve")
            inbox_items = [item_bob, item_eve]
            senders = {"bob", "eve"}
            replied_to = {"bob", "eve"}

            dp.messenger.archive_paths = MagicMock(return_value=2)

            await dp._archive_processed_messages(inbox_items, senders, replied_to)

            dp.messenger.archive_paths.assert_called_once()
            archived_items = dp.messenger.archive_paths.call_args[0][0]
            assert len(archived_items) == 2
        finally:
            _stop_patches(mocks)

    async def test_some_unreplied_kept_in_inbox(self, data_dir, make_anima):
        """Unreplied senders' messages are kept in inbox."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            item_bob = _make_inbox_item("bob", "msg from bob")
            item_eve = _make_inbox_item("eve", "msg from eve")
            inbox_items = [item_bob, item_eve]
            senders = {"bob", "eve"}
            replied_to = {"bob"}  # Only replied to bob

            dp.messenger.archive_paths = MagicMock(return_value=1)

            await dp._archive_processed_messages(inbox_items, senders, replied_to)

            dp.messenger.archive_paths.assert_called_once()
            archived_items = dp.messenger.archive_paths.call_args[0][0]
            # Only bob's message should be archived
            archived_senders = {item.msg.from_person for item in archived_items}
            assert "bob" in archived_senders
            assert "eve" not in archived_senders
        finally:
            _stop_patches(mocks)

    async def test_stale_messages_force_archived(self, data_dir, make_anima):
        """Messages older than _STALE_MESSAGE_TIMEOUT_SEC are force-archived."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Create a real file with old mtime for the stale message
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        stale_file = inbox_dir / "stale_msg.json"
        stale_file.write_text("{}", encoding="utf-8")

        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            item_eve = _make_inbox_item("eve", "old msg from eve", stale_file)
            inbox_items = [item_eve]
            senders = {"eve"}
            replied_to = set()  # Nobody replied to

            dp.messenger.archive_paths = MagicMock(return_value=1)

            # Make the file appear old (> 600 seconds)
            import os
            old_time = time.time() - 700
            os.utime(stale_file, (old_time, old_time))

            await dp._archive_processed_messages(inbox_items, senders, replied_to)

            dp.messenger.archive_paths.assert_called_once()
            archived_items = dp.messenger.archive_paths.call_args[0][0]
            # Eve's stale message should be force-archived
            assert len(archived_items) == 1
            assert archived_items[0].msg.from_person == "eve"
        finally:
            _stop_patches(mocks)

    async def test_non_stale_unreplied_kept(self, data_dir, make_anima):
        """Non-stale unreplied messages remain in inbox."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        fresh_file = inbox_dir / "fresh_msg.json"
        fresh_file.write_text("{}", encoding="utf-8")

        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            item_eve = _make_inbox_item("eve", "recent msg", fresh_file)
            inbox_items = [item_eve]
            senders = {"eve"}
            replied_to = set()

            dp.messenger.archive_paths = MagicMock(return_value=0)

            # File is fresh (just created), so it won't be stale
            await dp._archive_processed_messages(inbox_items, senders, replied_to)

            dp.messenger.archive_paths.assert_called_once()
            archived_items = dp.messenger.archive_paths.call_args[0][0]
            # Eve's message should NOT be archived (not stale)
            assert len(archived_items) == 0
        finally:
            _stop_patches(mocks)

    async def test_system_messages_always_archived(self, data_dir, make_anima):
        """Messages from senders not in the senders set (system msgs) are archived."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            item_system = _make_inbox_item("system", "sys notification")
            item_bob = _make_inbox_item("bob", "hi from bob")
            inbox_items = [item_system, item_bob]
            senders = {"bob"}  # system is not in senders
            replied_to = {"bob"}

            dp.messenger.archive_paths = MagicMock(return_value=2)

            await dp._archive_processed_messages(inbox_items, senders, replied_to)

            dp.messenger.archive_paths.assert_called_once()
            archived_items = dp.messenger.archive_paths.call_args[0][0]
            archived_senders = {item.msg.from_person for item in archived_items}
            assert "system" in archived_senders
            assert "bob" in archived_senders

        finally:
            _stop_patches(mocks)

    async def test_empty_inbox_items(self, data_dir, make_anima):
        """Empty inbox items list does not crash."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            dp.messenger.archive_paths = MagicMock(return_value=0)

            await dp._archive_processed_messages([], set(), set())

            dp.messenger.archive_paths.assert_called_once()
            archived_items = dp.messenger.archive_paths.call_args[0][0]
            assert len(archived_items) == 0
        finally:
            _stop_patches(mocks)


# ── _handle_heartbeat_failure ────────────────────────────


class TestHandleHeartbeatFailure:
    async def test_crash_archive_inbox_messages(self, data_dir, make_anima):
        """Inbox messages are crash-archived on failure."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            item = _make_inbox_item("bob", "message from bob")
            error = RuntimeError("heartbeat boom")

            dp.messenger.archive_paths = MagicMock(return_value=1)
            dp._heartbeat_stream_queue = None

            with patch("core.anima.ActivityLogger"):
                await dp._handle_heartbeat_failure(error, [item], unread_count=1)

            dp.messenger.archive_paths.assert_called_once_with([item])
        finally:
            _stop_patches(mocks)

    async def test_empty_inbox_no_crash_archive(self, data_dir, make_anima):
        """When inbox_items is empty, archive_paths is not called."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            error = RuntimeError("boom")
            dp.messenger.archive_paths = MagicMock(return_value=0)
            dp._heartbeat_stream_queue = None

            with patch("core.anima.ActivityLogger"):
                await dp._handle_heartbeat_failure(error, [], unread_count=0)

            dp.messenger.archive_paths.assert_not_called()
        finally:
            _stop_patches(mocks)

    async def test_recovery_note_written(self, data_dir, make_anima):
        """Recovery note is saved to state/recovery_note.md."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            error = ValueError("something went wrong")
            dp._heartbeat_stream_queue = None

            with patch("core.anima.ActivityLogger"):
                await dp._handle_heartbeat_failure(error, [], unread_count=3)

            recovery_path = anima_dir / "state" / "recovery_note.md"
            assert recovery_path.exists()
            content = recovery_path.read_text(encoding="utf-8")
            assert "ValueError" in content
            assert "something went wrong" in content
            assert "未処理メッセージ数: 3" in content
        finally:
            _stop_patches(mocks)

    async def test_error_activity_logged(self, data_dir, make_anima):
        """Error is logged to activity log."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            error = TypeError("type issue")
            dp._heartbeat_stream_queue = None

            mock_activity = MagicMock()
            dp._activity = mock_activity

            await dp._handle_heartbeat_failure(error, [], unread_count=0)

            mock_activity.log.assert_called_once()
            call_args = mock_activity.log.call_args
            assert call_args[0][0] == "error"
            assert "TypeError" in call_args[1]["summary"]
        finally:
            _stop_patches(mocks)

    async def test_relay_queue_sentinel_sent(self, data_dir, make_anima):
        """Sentinel (None) is sent to relay queue on failure."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            error = RuntimeError("boom")
            queue = asyncio.Queue()
            dp._heartbeat_stream_queue = queue

            with patch("core.anima.ActivityLogger"):
                await dp._handle_heartbeat_failure(error, [], unread_count=0)

            # Queue should have the None sentinel
            assert not queue.empty()
            sentinel = await queue.get()
            assert sentinel is None
        finally:
            _stop_patches(mocks)

    async def test_no_relay_queue_no_error(self, data_dir, make_anima):
        """No relay queue set -- no error when sending sentinel."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            error = RuntimeError("boom")
            dp._heartbeat_stream_queue = None

            with patch("core.anima.ActivityLogger"):
                # Should not raise
                await dp._handle_heartbeat_failure(error, [], unread_count=0)
        finally:
            _stop_patches(mocks)

    async def test_crash_archive_failure_handled(self, data_dir, make_anima):
        """Even if crash-archive itself fails, recovery note is still written."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            item = _make_inbox_item("bob", "msg")
            error = RuntimeError("boom")
            dp._heartbeat_stream_queue = None

            dp.messenger.archive_paths = MagicMock(side_effect=OSError("archive failed"))

            with patch("core.anima.ActivityLogger"):
                # Should not raise despite archive failure
                await dp._handle_heartbeat_failure(error, [item], unread_count=1)

            # Recovery note should still be written
            recovery_path = anima_dir / "state" / "recovery_note.md"
            assert recovery_path.exists()
        finally:
            _stop_patches(mocks)

    async def test_recovery_note_with_long_error_truncated(self, data_dir, make_anima):
        """Error message longer than 200 chars is truncated in recovery note."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp, mocks = _create_anima(anima_dir, shared_dir)
        try:
            long_msg = "x" * 500
            error = RuntimeError(long_msg)
            dp._heartbeat_stream_queue = None

            with patch("core.anima.ActivityLogger"):
                await dp._handle_heartbeat_failure(error, [], unread_count=0)

            recovery_path = anima_dir / "state" / "recovery_note.md"
            content = recovery_path.read_text(encoding="utf-8")
            # The error content line should contain at most 200 chars of the error
            error_line = [l for l in content.split("\n") if "エラー内容" in l][0]
            # str(error)[:200] = first 200 'x' chars
            assert "x" * 200 in error_line
            # But not the full 500
            assert "x" * 201 not in error_line
        finally:
            _stop_patches(mocks)
