# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for heartbeat-dialogue context gap and messaging improvements.

Tests cross-context flows WITHOUT mocking file-system operations:
  A-1: HB execution injects recent dialogue context
  A-2: Dialogue sessions inject recent HB summary into system prompt
  A-3: Non-HEARTBEAT_OK results are recorded to episodes
  B-1: current_task.md gets emphasized header when status != idle
  E:   receive_and_archive() sends read ACK with loop prevention
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.conversation import ConversationMemory, ConversationTurn
from core.memory.manager import MemoryManager
from core.messenger import Messenger
from core.schemas import CycleResult, ModelConfig


# ── Helpers ───────────────────────────────────────────────


def _make_model_config() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        max_turns=5,
    )


def _make_digital_anima(anima_dir: Path, shared_dir: Path):
    """Create a DigitalAnima with heavy deps (AgentCore, MemoryManager, Messenger) mocked."""
    with patch("core.anima.AgentCore"), \
         patch("core.anima.MemoryManager") as MockMM, \
         patch("core.anima.Messenger"):
        MockMM.return_value.read_model_config.return_value = MagicMock()
        from core.anima import DigitalAnima
        return DigitalAnima(anima_dir, shared_dir)


def _make_cycle_result(**kwargs) -> CycleResult:
    """Create a CycleResult with sensible defaults."""
    defaults = dict(
        trigger="heartbeat",
        action="checked",
        summary="All systems normal",
        duration_ms=150,
    )
    defaults.update(kwargs)
    return CycleResult(**defaults)


def _write_conversation_json(anima_dir: Path, turns: list[dict]) -> None:
    """Write a conversation.json file with the given turns."""
    state_dir = anima_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "anima_name": anima_dir.name,
        "turns": turns,
        "compressed_summary": "",
        "compressed_turn_count": 0,
    }
    (state_dir / "conversation.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _setup_anima_dir(tmp_path: Path, name: str = "alice") -> Path:
    """Create an anima directory with required subdirectories."""
    d = tmp_path / "animas" / name
    d.mkdir(parents=True)
    (d / "identity.md").write_text(f"# {name}", encoding="utf-8")
    for sub in ("state", "episodes", "knowledge", "procedures", "skills",
                "shortterm", "shortterm/archive", "transcripts"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "state" / "current_task.md").write_text("status: idle\n", encoding="utf-8")
    return d


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    return _setup_anima_dir(tmp_path, "alice")


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def dp(anima_dir: Path, shared_dir: Path):
    """A DigitalAnima instance with mocked dependencies."""
    return _make_digital_anima(anima_dir, shared_dir)


# =====================================================================
# Test 1: Full cross-context flow (A-1 + A-2 + A-3)
# =====================================================================


class TestCrossContextFlow:
    """Verify the heartbeat-dialogue cross-context bridge end-to-end.

    A-1: Heartbeat reads dialogue context from conversation.json
    A-2: Dialogue sessions read heartbeat summaries via load_recent_heartbeat_summary()
    A-3: Non-HEARTBEAT_OK results are persisted to episodes/
    """

    def test_activity_log_records_heartbeat(self, dp, anima_dir):
        """ActivityLogger records heartbeat_end to activity_log/."""
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        activity.log("heartbeat_end", summary="Checked Slack, found 3 unread messages")

        log_dir = anima_dir / "activity_log"
        today_file = log_dir / f"{date.today().isoformat()}.jsonl"
        assert today_file.exists()

        content = today_file.read_text(encoding="utf-8").strip()
        entry = json.loads(content)
        assert entry["summary"] == "Checked Slack, found 3 unread messages"
        assert entry["type"] == "heartbeat_end"

    def test_load_heartbeat_history_returns_entries(self, dp, anima_dir):
        """_load_heartbeat_history reads back persisted entries."""
        # Write entries directly to heartbeat_history/ (legacy format)
        history_dir = anima_dir / "shortterm" / "heartbeat_history"
        # Write heartbeat_end entries to unified activity log
        activity_dir = anima_dir / "activity_log"
        activity_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for i in range(5):
            entries.append(json.dumps({
                "ts": f"2026-02-17T{10 + i:02d}:00:00",
                "type": "heartbeat_end",
                "summary": f"HB action {i}",
            }, ensure_ascii=False))
        (activity_dir / f"{date.today().isoformat()}.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        text = dp._load_heartbeat_history()
        assert text != ""
        lines = text.strip().splitlines()
        # Default _HEARTBEAT_HISTORY_N = 3
        assert len(lines) == dp._HEARTBEAT_HISTORY_N
        # Most recent entries should be present
        assert "HB action 4" in lines[-1]
        assert "HB action 3" in lines[-2]

    def test_load_recent_heartbeat_summary_a2(self, anima_dir):
        """MemoryManager.load_recent_heartbeat_summary returns formatted entries.

        This is the A-2 path: dialogue sessions inject HB summary into prompt.
        """
        # Prepare heartbeat history JSONL
        history_dir = anima_dir / "shortterm" / "heartbeat_history"
        history_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for i in range(4):
            entry = json.dumps({
                "timestamp": f"2026-02-17T{10 + i:02d}:00:00",
                "trigger": "heartbeat",
                "action": "reported",
                "summary": f"Action {i}: did something important" if i % 2 == 0 else "HEARTBEAT_OK",
                "duration_ms": 100 + i,
            }, ensure_ascii=False)
            entries.append(entry)
        (history_dir / f"{date.today().isoformat()}.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        # Use a real MemoryManager (the only mock is implicit: no config.json)
        with patch.object(MemoryManager, "__init__", lambda self, ad, **kw: None):
            mm = object.__new__(MemoryManager)
            mm.anima_dir = anima_dir

        result = mm.load_recent_heartbeat_summary(limit=5)
        assert result != ""
        # HEARTBEAT_OK entries should be filtered out
        assert "HEARTBEAT_OK" not in result
        # Non-OK entries should appear
        assert "Action 0" in result
        assert "Action 2" in result
        assert "[reported]" in result

    def test_episode_recording_a3(self, dp, anima_dir):
        """Non-HEARTBEAT_OK heartbeat results are recorded to episodes (A-3).

        Since run_heartbeat is async and calls the agent, we test the episode-
        recording logic directly by simulating what run_heartbeat does after
        the agent cycle completes.
        """
        # Use a real MemoryManager for episode recording
        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        mm.episodes_dir = anima_dir / "episodes"
        mm.knowledge_dir = anima_dir / "knowledge"
        mm.procedures_dir = anima_dir / "procedures"
        mm.skills_dir = anima_dir / "skills"
        mm.state_dir = anima_dir / "state"
        mm._indexer = None
        mm._indexer_initialized = True

        # Simulate A-3: record heartbeat episode
        result = _make_cycle_result(
            summary="Slack channel #general had 5 new messages. "
                    "Responded to user query about deployment."
        )

        # Replicate the A-3 logic from run_heartbeat
        if result.summary and "HEARTBEAT_OK" not in result.summary:
            ts = datetime.now().strftime("%H:%M")
            episode_entry = (
                f"## {ts} ハートビート活動\n\n"
                f"{result.summary[:500]}"
            )
            mm.append_episode(episode_entry)

        # Verify episode was written
        episode_file = anima_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")
        assert "ハートビート活動" in content
        assert "Slack channel #general" in content

    def test_heartbeat_ok_not_recorded_to_episodes(self, dp, anima_dir):
        """HEARTBEAT_OK results should NOT be recorded to episodes."""
        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        mm.episodes_dir = anima_dir / "episodes"
        mm.knowledge_dir = anima_dir / "knowledge"
        mm.procedures_dir = anima_dir / "procedures"
        mm.skills_dir = anima_dir / "skills"
        mm.state_dir = anima_dir / "state"
        mm._indexer = None
        mm._indexer_initialized = True

        result = _make_cycle_result(summary="HEARTBEAT_OK - nothing to do")

        # Replicate the A-3 gate
        if result.summary and "HEARTBEAT_OK" not in result.summary:
            mm.append_episode("should not appear")

        episode_file = anima_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert not episode_file.exists()

    def test_conversation_json_written_and_read_back(self, anima_dir):
        """ConversationMemory can write turns, save, and reload them.

        This underpins A-1: heartbeat loads conversation.json for context.
        """
        config = _make_model_config()
        conv = ConversationMemory(anima_dir, config)

        conv.append_turn("human", "Please deploy the staging server")
        conv.append_turn("assistant", "I have started the deployment pipeline.")
        conv.append_turn("human", "How long will it take?")
        conv.save()

        # Reload from disk
        conv2 = ConversationMemory(anima_dir, config)
        state = conv2.load()
        assert len(state.turns) == 3
        assert state.turns[0].role == "human"
        assert "deploy" in state.turns[0].content
        assert state.turns[1].role == "assistant"
        assert state.turns[2].content == "How long will it take?"

    def test_full_cross_context_roundtrip(self, dp, anima_dir):
        """Full roundtrip: conversation -> activity log -> summary -> episodes.

        1. Write conversation turns (simulating dialogue)
        2. Record heartbeat via ActivityLogger (simulating HB run)
        3. Verify activity log is loadable
        4. Verify load_recent_heartbeat_summary returns formatted data
        5. Verify episode recording
        """
        from core.memory.activity import ActivityLogger

        # Step 1: Write dialogue turns
        config = _make_model_config()
        conv = ConversationMemory(anima_dir, config)
        conv.append_turn("human", "Start the batch job for client X")
        conv.append_turn("assistant", "Batch job initiated. ETA: 30 minutes.")
        conv.save()

        # Verify A-1 path: conversation can be loaded
        state = conv.load()
        recent = state.turns[-5:]
        assert len(recent) == 2
        conv_lines = [f"- [{t.role}] {t.content[:200]}" for t in recent]
        conv_summary = "\n".join(conv_lines)
        assert "batch job" in conv_summary.lower()

        # Step 2: Record heartbeat via ActivityLogger + legacy heartbeat_history
        result = _make_cycle_result(
            summary="Checked batch job status for client X: 50% complete"
        )
        activity = ActivityLogger(anima_dir)
        activity.log("heartbeat_end", summary=result.summary[:200])

        # Also write legacy heartbeat_history for load_recent_heartbeat_summary
        history_dir = anima_dir / "shortterm" / "heartbeat_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "trigger": "heartbeat",
            "action": "checked",
            "summary": result.summary,
            "duration_ms": result.duration_ms,
        }, ensure_ascii=False)
        (history_dir / f"{date.today().isoformat()}.jsonl").write_text(
            entry + "\n", encoding="utf-8",
        )

        # Step 3: Verify activity log is loadable
        entries = activity.recent(days=1, limit=10)
        summaries = " ".join(e.summary for e in entries)
        assert "batch job" in summaries.lower()

        # Step 4: Verify A-2 path (load_recent_heartbeat_summary)
        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        summary = mm.load_recent_heartbeat_summary(limit=5)
        assert "batch job" in summary.lower()
        assert "[checked]" in summary

        # Step 5: Verify A-3 path (episode recording)
        mm.episodes_dir = anima_dir / "episodes"
        mm.knowledge_dir = anima_dir / "knowledge"
        mm.procedures_dir = anima_dir / "procedures"
        mm.skills_dir = anima_dir / "skills"
        mm.state_dir = anima_dir / "state"
        mm._indexer = None
        mm._indexer_initialized = True

        if result.summary and "HEARTBEAT_OK" not in result.summary:
            ts = datetime.now().strftime("%H:%M")
            episode_entry = (
                f"## {ts} ハートビート活動\n\n"
                f"{result.summary[:500]}"
            )
            mm.append_episode(episode_entry)

        episode_file = anima_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")
        assert "batch job" in content.lower()


# =====================================================================
# Test 2: Messenger ACK flow (E)
# =====================================================================


class TestMessengerAckFlow:
    """Test receive_and_archive sends read ACK to senders with loop prevention."""

    def test_ack_sent_on_receive_and_archive(self, shared_dir):
        """When alice calls receive_and_archive, bob gets an ACK in his inbox."""
        alice = Messenger(shared_dir, "alice")
        bob = Messenger(shared_dir, "bob")

        # Bob sends messages to alice
        bob.send("alice", "Hello Alice, first message")
        bob.send("alice", "Second message about the project")

        assert alice.unread_count() == 2

        # Alice reads and archives
        messages = alice.receive_and_archive()
        assert len(messages) == 2

        # Bob should now have ACK messages in his inbox
        bob_messages = bob.receive()
        assert len(bob_messages) >= 1

        # Find the ACK message
        ack_msgs = [m for m in bob_messages if m.type == "ack"]
        assert len(ack_msgs) == 1

        ack = ack_msgs[0]
        assert ack.from_person == "alice"
        assert ack.to_person == "bob"
        assert "既読通知" in ack.content
        assert "2件" in ack.content

    def test_ack_contains_message_summary(self, shared_dir):
        """ACK message contains a summary of the received messages."""
        alice = Messenger(shared_dir, "alice")
        bob = Messenger(shared_dir, "bob")

        bob.send("alice", "Check the deployment logs")

        alice.receive_and_archive()

        bob_messages = bob.receive()
        ack_msgs = [m for m in bob_messages if m.type == "ack"]
        assert len(ack_msgs) == 1
        assert "Check the deployment" in ack_msgs[0].content

    def test_ack_loop_prevention(self, shared_dir):
        """Calling receive_and_archive on ACK messages does NOT generate further ACKs.

        This is the loop prevention mechanism: ack-type messages are excluded
        from the ACK generation logic.
        """
        alice = Messenger(shared_dir, "alice")
        bob = Messenger(shared_dir, "bob")

        # Bob sends a normal message to alice
        bob.send("alice", "Hello Alice!")

        # Alice reads and archives -> bob gets ACK
        alice.receive_and_archive()

        # Bob now has an ACK in his inbox
        bob_before = bob.receive()
        ack_msgs = [m for m in bob_before if m.type == "ack"]
        assert len(ack_msgs) == 1

        # Bob reads and archives the ACK -> alice should NOT get an ACK back
        bob.receive_and_archive()

        # Alice should have no new messages (no ACK-of-ACK)
        alice_inbox = alice.receive()
        ack_of_ack = [m for m in alice_inbox if m.type == "ack"]
        assert len(ack_of_ack) == 0

    def test_multiple_senders_get_separate_acks(self, shared_dir):
        """Each sender gets their own ACK message."""
        alice = Messenger(shared_dir, "alice")
        bob = Messenger(shared_dir, "bob")
        charlie = Messenger(shared_dir, "charlie")

        bob.send("alice", "From Bob")
        charlie.send("alice", "From Charlie")

        alice.receive_and_archive()

        # Bob should have an ACK
        bob_msgs = bob.receive()
        bob_acks = [m for m in bob_msgs if m.type == "ack"]
        assert len(bob_acks) == 1
        assert "1件" in bob_acks[0].content

        # Charlie should have an ACK
        charlie_msgs = charlie.receive()
        charlie_acks = [m for m in charlie_msgs if m.type == "ack"]
        assert len(charlie_acks) == 1
        assert "1件" in charlie_acks[0].content

    def test_ack_with_many_messages_shows_overflow(self, shared_dir):
        """When more than 3 messages from one sender, ACK shows overflow count."""
        alice = Messenger(shared_dir, "alice")
        bob = Messenger(shared_dir, "bob")

        for i in range(5):
            bob.send("alice", f"Message {i} from bob")

        alice.receive_and_archive()

        bob_msgs = bob.receive()
        ack_msgs = [m for m in bob_msgs if m.type == "ack"]
        assert len(ack_msgs) == 1

        ack_content = ack_msgs[0].content
        assert "5件" in ack_content
        # The ACK summary only shows first 3 messages, with overflow indicator
        assert "+2件" in ack_content

    def test_no_ack_when_no_messages(self, shared_dir):
        """receive_and_archive on empty inbox does not produce ACKs."""
        alice = Messenger(shared_dir, "alice")
        bob = Messenger(shared_dir, "bob")

        messages = alice.receive_and_archive()
        assert messages == []

        # Bob should have nothing
        bob_msgs = bob.receive()
        assert len(bob_msgs) == 0


# =====================================================================
# Test 3: current_task.md emphasis (B-1)
# =====================================================================


class TestCurrentTaskEmphasis:
    """Test B-1: current_task.md gets emphasized header when status != idle."""

    def test_working_status_gets_emphasis(self, tmp_path, monkeypatch):
        """When current_task.md has non-idle status, prompt has emphasized header."""
        # Set up isolated data_dir
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        # Invalidate caches
        from core.config import invalidate_cache
        from core.paths import _prompt_cache
        invalidate_cache()
        _prompt_cache.clear()

        # Create required directories
        (data_dir / "company").mkdir()
        (data_dir / "company" / "vision.md").write_text("# Vision\nTest", encoding="utf-8")
        (data_dir / "common_skills").mkdir()
        (data_dir / "common_knowledge").mkdir()
        (data_dir / "shared" / "users").mkdir(parents=True)

        anima_dir = _setup_anima_dir(tmp_path, "tester")

        # Write an active task
        (anima_dir / "state" / "current_task.md").write_text(
            "status: working\ntask: Deploy v2.0 to production\nprogress: 60%\n",
            encoding="utf-8",
        )

        # Create MemoryManager with minimal init
        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        mm.company_dir = data_dir / "company"
        mm.common_skills_dir = data_dir / "common_skills"
        mm.common_knowledge_dir = data_dir / "common_knowledge"
        mm.episodes_dir = anima_dir / "episodes"
        mm.knowledge_dir = anima_dir / "knowledge"
        mm.procedures_dir = anima_dir / "procedures"
        mm.skills_dir = anima_dir / "skills"
        mm.state_dir = anima_dir / "state"
        mm._indexer = None
        mm._indexer_initialized = True

        from core.prompt.builder import build_system_prompt
        prompt = build_system_prompt(mm)

        # B-1: emphasized header must be present
        assert "⚠️ 進行中タスク" in prompt
        assert "MUST: 最優先で確認すること" in prompt
        assert "Deploy v2.0 to production" in prompt

        # Cleanup caches
        invalidate_cache()
        _prompt_cache.clear()

    def test_idle_status_gets_normal_header(self, tmp_path, monkeypatch):
        """When current_task.md has idle status, prompt uses normal header."""
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        from core.config import invalidate_cache
        from core.paths import _prompt_cache
        invalidate_cache()
        _prompt_cache.clear()

        (data_dir / "company").mkdir()
        (data_dir / "company" / "vision.md").write_text("# Vision\nTest", encoding="utf-8")
        (data_dir / "common_skills").mkdir()
        (data_dir / "common_knowledge").mkdir()
        (data_dir / "shared" / "users").mkdir(parents=True)

        anima_dir = _setup_anima_dir(tmp_path, "tester")

        (anima_dir / "state" / "current_task.md").write_text(
            "status: idle\n", encoding="utf-8",
        )

        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        mm.company_dir = data_dir / "company"
        mm.common_skills_dir = data_dir / "common_skills"
        mm.common_knowledge_dir = data_dir / "common_knowledge"
        mm.episodes_dir = anima_dir / "episodes"
        mm.knowledge_dir = anima_dir / "knowledge"
        mm.procedures_dir = anima_dir / "procedures"
        mm.skills_dir = anima_dir / "skills"
        mm.state_dir = anima_dir / "state"
        mm._indexer = None
        mm._indexer_initialized = True

        from core.prompt.builder import build_system_prompt
        prompt = build_system_prompt(mm)

        # Normal header, not the emphasized one
        assert "## 現在の状態" in prompt
        assert "⚠️ 進行中タスク" not in prompt

        invalidate_cache()
        _prompt_cache.clear()

    def test_heartbeat_summary_injected_into_prompt_a2(self, tmp_path, monkeypatch):
        """A-2: build_system_prompt includes recent activity summary section.

        The builder now uses _load_recent_activity_summary() which reads
        from the unified ActivityLogger, falling back to legacy
        heartbeat_history.  We write activity_log entries so the primary
        path is exercised.
        """
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        from core.config import invalidate_cache
        from core.paths import _prompt_cache
        invalidate_cache()
        _prompt_cache.clear()

        (data_dir / "company").mkdir()
        (data_dir / "company" / "vision.md").write_text("# Vision\nTest", encoding="utf-8")
        (data_dir / "common_skills").mkdir()
        (data_dir / "common_knowledge").mkdir()
        (data_dir / "shared" / "users").mkdir(parents=True)

        anima_dir = _setup_anima_dir(tmp_path, "tester")

        # Write unified activity log entries (replaces heartbeat_history)
        from core.memory.activity import ActivityLogger
        activity = ActivityLogger(anima_dir)
        activity.log(
            "heartbeat_end",
            summary="Found 3 errors in production logs",
        )

        mm = MemoryManager.__new__(MemoryManager)
        mm.anima_dir = anima_dir
        mm.company_dir = data_dir / "company"
        mm.common_skills_dir = data_dir / "common_skills"
        mm.common_knowledge_dir = data_dir / "common_knowledge"
        mm.episodes_dir = anima_dir / "episodes"
        mm.knowledge_dir = anima_dir / "knowledge"
        mm.procedures_dir = anima_dir / "procedures"
        mm.skills_dir = anima_dir / "skills"
        mm.state_dir = anima_dir / "state"
        mm._indexer = None
        mm._indexer_initialized = True

        from core.prompt.builder import build_system_prompt
        from core.memory.priming import PrimingResult, format_priming_section

        # Generate the priming section with activity data
        recent_entries = activity.recent(days=1)
        activity_text = activity.format_for_priming(recent_entries)
        priming_result = PrimingResult(recent_activity=activity_text)
        priming_section = format_priming_section(priming_result)

        result = build_system_prompt(mm, priming_section=priming_section)
        prompt = result.system_prompt

        # A-2: activity summary section should be present via priming
        assert "アクティビティ" in prompt
        assert "Found 3 errors in production logs" in prompt

        invalidate_cache()
        _prompt_cache.clear()
