# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for heartbeat dialogue context injection and episode recording.

Covers:
- A-1: run_heartbeat() loads recent conversation turns into the heartbeat prompt
- A-2: load_recent_heartbeat_summary() returns formatted entries, filters HEARTBEAT_OK
- A-3: Non-HEARTBEAT_OK heartbeat results are recorded to episodes
- B-1: build_system_prompt() uses emphasized header for non-idle current_task
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import CycleResult


# ── Helpers ───────────────────────────────────────────────


def _make_cycle_result(**kwargs) -> CycleResult:
    defaults = dict(
        trigger="heartbeat",
        action="responded",
        summary="done",
        duration_ms=100,
    )
    defaults.update(kwargs)
    return CycleResult(**defaults)


def _make_digital_anima(anima_dir: Path, shared_dir: Path):
    """Create a DigitalAnima with all heavy deps mocked."""
    with patch("core.anima.AgentCore"), \
         patch("core.anima.MemoryManager") as MockMM, \
         patch("core.anima.Messenger"):
        MockMM.return_value.read_model_config.return_value = MagicMock()
        from core.anima import DigitalAnima
        return DigitalAnima(anima_dir, shared_dir)


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "alice"
    d.mkdir(parents=True)
    (d / "identity.md").write_text("# Alice", encoding="utf-8")
    return d


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def dp(anima_dir: Path, shared_dir: Path):
    """A DigitalAnima instance with mocked dependencies."""
    return _make_digital_anima(anima_dir, shared_dir)


@pytest.fixture
def mock_memory(anima_dir: Path, tmp_path: Path) -> MagicMock:
    """A mock MemoryManager with all required methods stubbed for build_system_prompt."""
    mm = MagicMock()
    mm.anima_dir = anima_dir
    mm.common_skills_dir = tmp_path / "common_skills"
    mm.common_skills_dir.mkdir(parents=True, exist_ok=True)
    mm.read_identity.return_value = "# Identity"
    mm.read_injection.return_value = ""
    mm.read_specialty_prompt.return_value = ""
    mm.read_bootstrap.return_value = ""
    mm.read_company_vision.return_value = ""
    mm.read_permissions.return_value = ""
    mm.read_current_state.return_value = "status: idle"
    mm.read_pending.return_value = ""
    mm.list_knowledge_files.return_value = []
    mm.list_episode_files.return_value = []
    mm.list_procedure_files.return_value = []
    mm.list_skill_summaries.return_value = []
    mm.list_common_skill_summaries.return_value = []
    mm.list_skill_metas.return_value = []
    mm.list_common_skill_metas.return_value = []
    mm.list_shared_users.return_value = []
    mm.load_recent_heartbeat_summary.return_value = ""
    mm.read_model_config.return_value = MagicMock(supervisor=None)
    return mm


# ══════════════════════════════════════════════════════════
# 1. TestLoadRecentHeartbeatSummary (A-2: MemoryManager)
# ══════════════════════════════════════════════════════════


class TestLoadRecentHeartbeatSummary:
    """Tests for MemoryManager.load_recent_heartbeat_summary()."""

    def test_empty_when_no_history_dir(self, tmp_path: Path):
        """Returns empty string when heartbeat_history directory does not exist."""
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Bob", encoding="utf-8")

        with patch("core.memory.manager.get_company_dir", return_value=tmp_path / "company"), \
             patch("core.memory.manager.get_common_skills_dir", return_value=tmp_path / "common_skills"), \
             patch("core.memory.manager.get_common_knowledge_dir", return_value=tmp_path / "common_knowledge"), \
             patch("core.memory.manager.get_shared_dir", return_value=tmp_path / "shared"):
            from core.memory.manager import MemoryManager
            mm = MemoryManager(anima_dir)

        result = mm.load_recent_heartbeat_summary()
        assert result == ""

    def test_returns_entries_from_jsonl(self, tmp_path: Path):
        """Creates JSONL entries and verifies parsed output."""
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Bob", encoding="utf-8")
        history_dir = anima_dir / "shortterm" / "heartbeat_history"
        history_dir.mkdir(parents=True)

        entries = []
        for i in range(3):
            entries.append(json.dumps({
                "timestamp": f"2026-02-17T{10 + i:02d}:00:00",
                "trigger": "heartbeat",
                "action": "checked",
                "summary": f"Performed action {i}",
                "duration_ms": 100 + i,
            }, ensure_ascii=False))
        (history_dir / f"{date.today().isoformat()}.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        with patch("core.memory.manager.get_company_dir", return_value=tmp_path / "company"), \
             patch("core.memory.manager.get_common_skills_dir", return_value=tmp_path / "common_skills"), \
             patch("core.memory.manager.get_common_knowledge_dir", return_value=tmp_path / "common_knowledge"), \
             patch("core.memory.manager.get_shared_dir", return_value=tmp_path / "shared"):
            from core.memory.manager import MemoryManager
            mm = MemoryManager(anima_dir)

        result = mm.load_recent_heartbeat_summary()
        assert result != ""
        # Each non-HEARTBEAT_OK entry should appear as a line
        lines = result.strip().splitlines()
        assert len(lines) == 3
        assert "Performed action 0" in lines[0]
        assert "[checked]" in lines[0]
        assert "Performed action 2" in lines[2]

    def test_filters_heartbeat_ok(self, tmp_path: Path):
        """HEARTBEAT_OK entries should be excluded from the summary."""
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Bob", encoding="utf-8")
        history_dir = anima_dir / "shortterm" / "heartbeat_history"
        history_dir.mkdir(parents=True)

        entries = [
            json.dumps({
                "timestamp": "2026-02-17T10:00:00",
                "action": "checked",
                "summary": "HEARTBEAT_OK: nothing to do",
            }, ensure_ascii=False),
            json.dumps({
                "timestamp": "2026-02-17T11:00:00",
                "action": "responded",
                "summary": "Replied to user message about deployment",
            }, ensure_ascii=False),
            json.dumps({
                "timestamp": "2026-02-17T12:00:00",
                "action": "checked",
                "summary": "HEARTBEAT_OK",
            }, ensure_ascii=False),
        ]
        (history_dir / f"{date.today().isoformat()}.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        with patch("core.memory.manager.get_company_dir", return_value=tmp_path / "company"), \
             patch("core.memory.manager.get_common_skills_dir", return_value=tmp_path / "common_skills"), \
             patch("core.memory.manager.get_common_knowledge_dir", return_value=tmp_path / "common_knowledge"), \
             patch("core.memory.manager.get_shared_dir", return_value=tmp_path / "shared"):
            from core.memory.manager import MemoryManager
            mm = MemoryManager(anima_dir)

        result = mm.load_recent_heartbeat_summary()
        lines = result.strip().splitlines()
        # Only the non-HEARTBEAT_OK entry should remain
        assert len(lines) == 1
        assert "Replied to user message" in lines[0]
        assert "HEARTBEAT_OK" not in result

    def test_limits_entries(self, tmp_path: Path):
        """Respects the limit parameter."""
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Bob", encoding="utf-8")
        history_dir = anima_dir / "shortterm" / "heartbeat_history"
        history_dir.mkdir(parents=True)

        # Write 10 entries
        entries = []
        for i in range(10):
            entries.append(json.dumps({
                "timestamp": f"2026-02-17T{10 + i:02d}:00:00",
                "action": "checked",
                "summary": f"Activity number {i}",
            }, ensure_ascii=False))
        (history_dir / f"{date.today().isoformat()}.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        with patch("core.memory.manager.get_company_dir", return_value=tmp_path / "company"), \
             patch("core.memory.manager.get_common_skills_dir", return_value=tmp_path / "common_skills"), \
             patch("core.memory.manager.get_common_knowledge_dir", return_value=tmp_path / "common_knowledge"), \
             patch("core.memory.manager.get_shared_dir", return_value=tmp_path / "shared"):
            from core.memory.manager import MemoryManager
            mm = MemoryManager(anima_dir)

        # With limit=3, only last 3 entries should appear
        result = mm.load_recent_heartbeat_summary(limit=3)
        lines = result.strip().splitlines()
        assert len(lines) == 3
        # Should contain the last 3 entries (Activity 7, 8, 9)
        assert "Activity number 7" in lines[0]
        assert "Activity number 8" in lines[1]
        assert "Activity number 9" in lines[2]


# ══════════════════════════════════════════════════════════
# 2. TestBuildSystemPromptCurrentTask (B-1)
# ══════════════════════════════════════════════════════════


class TestBuildSystemPromptCurrentTask:
    """Tests that build_system_prompt uses correct headers based on current_task state."""

    def test_idle_uses_normal_header(self, mock_memory: MagicMock):
        """When state is 'status: idle', uses the '## 現在の状態' header."""
        mock_memory.read_current_state.return_value = "status: idle"

        with patch("core.prompt.builder.load_prompt", return_value="prompt section"), \
             patch("core.prompt.builder._discover_other_animas", return_value=[]), \
             patch("core.prompt.builder._build_org_context", return_value=""):
            from core.prompt.builder import build_system_prompt
            result = build_system_prompt(mock_memory)

        assert "## 現在の状態" in result
        assert "⚠️ 進行中タスク" not in result

    def test_working_uses_emphasis_header(self, mock_memory: MagicMock):
        """When state is not 'status: idle', uses the '⚠️ 進行中タスク' header."""
        mock_memory.read_current_state.return_value = "status: working\ntask: Fix bug"

        def _mock_lp(name: str, **kwargs) -> str:
            if name == "builder/task_in_progress":
                return (
                    "## ⚠️ 進行中タスク（MUST: 最優先で確認すること）\n\n"
                    "以下のタスクが進行中です。\n\n"
                    f"{kwargs.get('state', '')}"
                )
            return "prompt section"

        with patch("core.prompt.builder.load_prompt", side_effect=_mock_lp), \
             patch("core.prompt.builder._discover_other_animas", return_value=[]), \
             patch("core.prompt.builder._build_org_context", return_value=""):
            from core.prompt.builder import build_system_prompt
            result = build_system_prompt(mock_memory)

        assert "⚠️ 進行中タスク" in result
        assert "最優先で確認すること" in result
        assert "Fix bug" in result
        assert "## 現在の状態" not in result


# ══════════════════════════════════════════════════════════
# 3. TestBuildSystemPromptHBSummary (A-2: prompt injection)
# ══════════════════════════════════════════════════════════


class TestBuildSystemPromptHBSummary:
    """Tests that activity summary is provided via PrimingEngine.

    Section 9 (_load_recent_activity_summary) was removed per Arch-1
    hippocampus model. PrimingEngine Channel B is now the sole
    activity reader for prompt construction.
    """

    def test_section9_removed_from_builder(self):
        """_load_recent_activity_summary no longer exists in builder module."""
        import core.prompt.builder as builder_mod
        assert not hasattr(builder_mod, "_load_recent_activity_summary")


# ══════════════════════════════════════════════════════════
# 4. TestHeartbeatEpisodeRecording (A-3)
# ══════════════════════════════════════════════════════════


class TestHeartbeatEpisodeRecording:
    """Tests that run_heartbeat records non-HEARTBEAT_OK results as episodes."""

    async def test_heartbeat_ok_not_recorded(self, data_dir, make_anima):
        """HEARTBEAT_OK result does not call append_episode."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockMsg.return_value.has_unread.return_value = False
            # Mock ConversationMemory.load to return empty state
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            # Simulate streaming that yields a cycle_done with HEARTBEAT_OK
            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "checked",
                        "summary": "HEARTBEAT_OK: nothing to do",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            result = await dp.run_heartbeat()
            assert "HEARTBEAT_OK" in result.summary
            # append_episode must NOT have been called
            MockMM.return_value.append_episode.assert_not_called()

    async def test_non_ok_result_recorded(self, data_dir, make_anima):
        """Non-HEARTBEAT_OK result calls append_episode with the summary."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockMsg.return_value.has_unread.return_value = False
            # Mock ConversationMemory.load to return empty state
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            # Simulate streaming that yields a cycle_done with actual activity
            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Processed user request and sent report",
                        "duration_ms": 500,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            result = await dp.run_heartbeat()
            assert "Processed user request" in result.summary
            # append_episode MUST have been called once
            MockMM.return_value.append_episode.assert_called_once()
            episode_text = MockMM.return_value.append_episode.call_args[0][0]
            assert "ハートビート活動" in episode_text
            assert "Processed user request" in episode_text


# ══════════════════════════════════════════════════════════
# 5. TestHeartbeatDialogueContext (A-1)
# ══════════════════════════════════════════════════════════


class TestHeartbeatDialogueContext:
    """Tests that run_heartbeat loads recent conversation turns into the prompt."""

    async def test_dialogue_context_injected_into_prompt(self, data_dir, make_anima):
        """Recent conversation turns are included in the heartbeat prompt."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockMsg.return_value.has_unread.return_value = False

            # Create mock conversation turns
            turn1 = MagicMock()
            turn1.role = "human"
            turn1.content = "Please deploy the staging server"
            turn2 = MagicMock()
            turn2.role = "assistant"
            turn2.content = "I will start the deployment process now"
            mock_state = MagicMock()
            mock_state.turns = [turn1, turn2]
            MockConv.return_value.load.return_value = mock_state

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            # Capture the prompt passed to run_cycle_streaming
            captured_prompts: list[str] = []

            async def mock_stream(prompt, trigger="manual", **kwargs):
                captured_prompts.append(prompt)
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "checked",
                        "summary": "HEARTBEAT_OK",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

            # Verify dialogue context is in the prompt
            assert len(captured_prompts) == 1
            prompt = captured_prompts[0]
            assert "直近の対話履歴" in prompt
            assert "deploy the staging server" in prompt
            assert "deployment process" in prompt

    async def test_no_dialogue_context_when_no_turns(self, data_dir, make_anima):
        """When conversation has no turns, dialogue context section is omitted."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockMsg.return_value.has_unread.return_value = False

            # Empty conversation
            mock_state = MagicMock()
            mock_state.turns = []
            MockConv.return_value.load.return_value = mock_state

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            captured_prompts: list[str] = []

            async def mock_stream(prompt, trigger="manual", **kwargs):
                captured_prompts.append(prompt)
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "checked",
                        "summary": "HEARTBEAT_OK",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

            assert len(captured_prompts) == 1
            prompt = captured_prompts[0]
            assert "直近の対話履歴" not in prompt

    async def test_dialogue_context_limits_to_last_5_turns(self, data_dir, make_anima):
        """Only the last 5 conversation turns are included in the heartbeat prompt."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.MemoryManager") as MockMM, \
             patch("core.anima.Messenger") as MockMsg, \
             patch("core.anima.load_prompt", return_value="prompt"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockMM.return_value.read_model_config.return_value = MagicMock()
            MockMM.return_value.read_heartbeat_config.return_value = "checklist"
            MockMM.return_value.append_episode = MagicMock()
            MockMsg.return_value.has_unread.return_value = False

            # Create 8 conversation turns
            turns = []
            for i in range(8):
                t = MagicMock()
                t.role = "human" if i % 2 == 0 else "assistant"
                t.content = f"Message number {i}"
                turns.append(t)
            mock_state = MagicMock()
            mock_state.turns = turns
            MockConv.return_value.load.return_value = mock_state

            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            captured_prompts: list[str] = []

            async def mock_stream(prompt, trigger="manual", **kwargs):
                captured_prompts.append(prompt)
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "checked",
                        "summary": "HEARTBEAT_OK",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

            assert len(captured_prompts) == 1
            prompt = captured_prompts[0]
            # Messages 0, 1, 2 should NOT be in the prompt (only last 5: 3-7)
            assert "Message number 0" not in prompt
            assert "Message number 1" not in prompt
            assert "Message number 2" not in prompt
            # Messages 3-7 should be present
            assert "Message number 3" in prompt
            assert "Message number 4" in prompt
            assert "Message number 5" in prompt
            assert "Message number 6" in prompt
            assert "Message number 7" in prompt
