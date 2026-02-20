from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for memory data durability changes.

Covers:
- ConversationMemory.save() atomic writes
- MemoryManager.write_knowledge_with_meta() atomic writes
- MemoryManager.append_episode() fsync behaviour
- ActivityLogger per-write fsync behaviour
- StreamingJournal two-step recovery (recover + confirm_recovery)
- AnimaRunner._recover_streaming_journal() tool_use event logging
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from core.memory.streaming_journal import StreamingJournal, JournalRecovery
from core.schemas import ModelConfig


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory with required subdirectories."""
    anima_dir = tmp_path / "test_anima"
    anima_dir.mkdir()
    (anima_dir / "shortterm").mkdir()
    (anima_dir / "state").mkdir()
    (anima_dir / "episodes").mkdir()
    (anima_dir / "knowledge").mkdir()
    (anima_dir / "procedures").mkdir()
    (anima_dir / "skills").mkdir()
    return anima_dir


@pytest.fixture
def model_config() -> ModelConfig:
    """Create a minimal ModelConfig for ConversationMemory."""
    return ModelConfig(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        max_turns=20,
    )


# ── 1. TestConversationAtomicSave ───────────────────────────────────


class TestConversationAtomicSave:
    """Tests that ConversationMemory.save() uses atomic writes."""

    def test_save_creates_valid_json(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """After save(), the conversation.json file exists and is valid JSON."""
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, model_config)
        conv.append_turn("human", "Hello!")
        conv.append_turn("assistant", "Hi there!")
        conv.save()

        state_path = anima_dir / "state" / "conversation.json"
        assert state_path.exists(), "conversation.json should exist after save"

        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert data["anima_name"] == anima_dir.name
        assert len(data["turns"]) == 2
        assert data["turns"][0]["role"] == "human"
        assert data["turns"][0]["content"] == "Hello!"
        assert data["turns"][1]["role"] == "assistant"
        assert data["turns"][1]["content"] == "Hi there!"

    def test_no_tmp_files_remain_after_save(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """No .tmp files should remain in the state directory after save()."""
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, model_config)
        conv.append_turn("human", "test message")
        conv.save()

        state_dir = anima_dir / "state"
        tmp_files = list(state_dir.glob("*.tmp"))
        assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"

    def test_crash_during_save_preserves_original(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """If atomic_write_text raises mid-write, the original data is preserved.

        Write initial data, then simulate a crash by patching atomic_write_text
        to raise an exception.  The original file should remain intact.
        """
        from core.memory.conversation import ConversationMemory

        # Write initial valid state
        conv1 = ConversationMemory(anima_dir, model_config)
        conv1.append_turn("human", "initial message")
        conv1.save()

        state_path = anima_dir / "state" / "conversation.json"
        original_content = state_path.read_text(encoding="utf-8")
        original_data = json.loads(original_content)
        assert len(original_data["turns"]) == 1

        # Attempt a second save that crashes mid-write
        conv2 = ConversationMemory(anima_dir, model_config)
        conv2.load()
        conv2.append_turn("assistant", "this should not persist")

        with patch(
            "core.memory.conversation.atomic_write_text",
            side_effect=OSError("Simulated disk failure"),
        ):
            with pytest.raises(OSError, match="Simulated disk failure"):
                conv2.save()

        # Original data must still be intact
        preserved = json.loads(state_path.read_text(encoding="utf-8"))
        assert len(preserved["turns"]) == 1
        assert preserved["turns"][0]["content"] == "initial message"

        # No stale .tmp files
        tmp_files = list((anima_dir / "state").glob("*.tmp"))
        assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"


# ── 2. TestManagerKnowledgeAtomicWrite ──────────────────────────────


class TestManagerKnowledgeAtomicWrite:
    """Tests that MemoryManager.write_knowledge_with_meta() uses atomic writes."""

    @pytest.fixture
    def memory_manager(self, tmp_path: Path):
        """Create a MemoryManager with mocked path dependencies."""
        from core.memory.manager import MemoryManager

        anima_dir = tmp_path / "test_anima"
        anima_dir.mkdir()
        with (
            patch(
                "core.memory.manager.get_common_knowledge_dir",
                return_value=tmp_path / "ck",
            ),
            patch(
                "core.memory.manager.get_common_skills_dir",
                return_value=tmp_path / "cs",
            ),
            patch(
                "core.memory.manager.get_company_dir",
                return_value=tmp_path / "company",
            ),
            patch(
                "core.memory.manager.get_shared_dir",
                return_value=tmp_path / "shared",
            ),
        ):
            mgr = MemoryManager(anima_dir)
        return mgr

    def test_write_knowledge_creates_file_with_frontmatter(
        self, memory_manager,
    ):
        """write_knowledge_with_meta() creates a file with YAML frontmatter."""
        path = memory_manager.knowledge_dir / "test_topic.md"
        metadata = {"tags": ["test"], "source": "unit_test"}
        content = "# Test Knowledge\n\nSome content here."

        memory_manager.write_knowledge_with_meta(path, content, metadata)

        assert path.exists(), "Knowledge file should exist after write"

        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), "File should start with YAML frontmatter"
        assert "tags:" in text
        assert "source: unit_test" in text
        assert "# Test Knowledge" in text
        assert "Some content here." in text

    def test_no_tmp_files_after_knowledge_write(
        self, memory_manager,
    ):
        """No .tmp files should remain after write_knowledge_with_meta()."""
        path = memory_manager.knowledge_dir / "clean_write.md"
        metadata = {"version": 1}
        content = "Clean write test."

        memory_manager.write_knowledge_with_meta(path, content, metadata)

        tmp_files = list(memory_manager.knowledge_dir.glob("*.tmp"))
        assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"


# ── 3. TestManagerEpisodeFsync ──────────────────────────────────────


class TestManagerEpisodeFsync:
    """Tests that MemoryManager.append_episode() calls fsync."""

    @pytest.fixture
    def memory_manager(self, tmp_path: Path):
        """Create a MemoryManager with mocked path dependencies."""
        from core.memory.manager import MemoryManager

        anima_dir = tmp_path / "test_anima"
        anima_dir.mkdir()
        with (
            patch(
                "core.memory.manager.get_common_knowledge_dir",
                return_value=tmp_path / "ck",
            ),
            patch(
                "core.memory.manager.get_common_skills_dir",
                return_value=tmp_path / "cs",
            ),
            patch(
                "core.memory.manager.get_company_dir",
                return_value=tmp_path / "company",
            ),
            patch(
                "core.memory.manager.get_shared_dir",
                return_value=tmp_path / "shared",
            ),
        ):
            mgr = MemoryManager(anima_dir)
        return mgr

    def test_append_episode_calls_fsync(self, memory_manager):
        """append_episode() should call os.fsync after writing."""
        with patch("core.memory.manager.os.fsync") as mock_fsync:
            memory_manager.append_episode("## 10:00 — Test Episode\n\nSome content.")

        mock_fsync.assert_called_once()
        # Verify the argument is a valid file descriptor (integer)
        fd_arg = mock_fsync.call_args[0][0]
        assert isinstance(fd_arg, int), f"fsync should receive an int fd, got {type(fd_arg)}"


# ── 4. TestActivityLoggerBufferedFsync ──────────────────────────────


class TestActivityLoggerFsync:
    """Tests that ActivityLogger._append() calls fsync on every write."""

    @pytest.fixture
    def activity_logger(self, anima_dir: Path):
        """Create an ActivityLogger bound to the temp anima directory."""
        from core.memory.activity import ActivityLogger

        return ActivityLogger(anima_dir)

    def test_single_entry_triggers_fsync(
        self, activity_logger, anima_dir: Path,
    ):
        """Every single append should trigger fsync."""
        with patch("core.memory.activity.os.fsync") as mock_fsync:
            activity_logger.log("message_received", content="single entry")

        mock_fsync.assert_called_once()

    def test_multiple_entries_fsync_every_time(
        self, activity_logger, anima_dir: Path,
    ):
        """Each append call should produce its own fsync."""
        with patch("core.memory.activity.os.fsync") as mock_fsync:
            for i in range(5):
                activity_logger.log(
                    "message_received", content=f"entry {i}",
                )

        assert mock_fsync.call_count == 5, (
            f"Expected fsync called 5 times (once per entry), "
            f"but was called {mock_fsync.call_count} times"
        )


# ── 5. TestStreamingJournalConfirmRecovery ──────────────────────────


class TestStreamingJournalConfirmRecovery:
    """Tests the two-step recovery flow: recover() + confirm_recovery()."""

    @pytest.fixture
    def journal(self, anima_dir: Path) -> StreamingJournal:
        """Create a StreamingJournal instance."""
        return StreamingJournal(anima_dir)

    def test_recover_no_longer_deletes_journal(
        self, journal: StreamingJournal, anima_dir: Path,
    ):
        """recover() should read the journal but NOT delete it.

        The file must still exist after recover() returns, allowing
        the caller to persist data before confirming.
        """
        journal.open(trigger="chat", from_person="tester", session_id="s-1")
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("test content")
        journal.close()

        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"
        assert journal_path.exists(), "Journal should exist before recovery"

        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        assert recovery.recovered_text == "test content"

        # Key assertion: file still exists after recover()
        assert journal_path.exists(), (
            "Journal file should NOT be deleted by recover() — "
            "caller must use confirm_recovery()"
        )

    def test_confirm_recovery_deletes_journal(
        self, journal: StreamingJournal, anima_dir: Path,
    ):
        """confirm_recovery() should delete the journal file."""
        journal.open(trigger="chat")
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("content to recover")
        journal.close()

        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"
        assert journal_path.exists()

        # recover first, then confirm
        StreamingJournal.recover(anima_dir)
        assert journal_path.exists(), "recover() should not delete the file"

        StreamingJournal.confirm_recovery(anima_dir)
        assert not journal_path.exists(), (
            "Journal file should be deleted after confirm_recovery()"
        )

    def test_recover_then_confirm_full_sequence(
        self, journal: StreamingJournal, anima_dir: Path,
    ):
        """Full two-step sequence: recover() -> use data -> confirm_recovery()."""
        journal.open(trigger="heartbeat", from_person="cron")
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("recovered output")
        journal.write_tool_start("web_search", args_summary="q=test")
        journal.write_tool_end("web_search", result_summary="3 results")
        journal.close()

        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"

        # Step 1: recover
        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        assert recovery.recovered_text == "recovered output"
        assert len(recovery.tool_calls) == 1
        assert recovery.trigger == "heartbeat"
        assert journal_path.exists(), "File should survive recover()"

        # Step 2: simulate persisting the data (just assert we have it)
        assert recovery.recovered_text

        # Step 3: confirm
        StreamingJournal.confirm_recovery(anima_dir)
        assert not journal_path.exists(), "File should be gone after confirm"

        # has_orphan should now return False
        assert StreamingJournal.has_orphan(anima_dir) is False

    def test_confirm_recovery_noop_when_no_file(self, anima_dir: Path):
        """confirm_recovery() should be a no-op when no journal file exists."""
        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"
        assert not journal_path.exists()

        # Should not raise
        StreamingJournal.confirm_recovery(anima_dir)

        # Still no file (no error, no side effects)
        assert not journal_path.exists()


# ── 6. TestRunnerToolUseRecovery ────────────────────────────────────


def _make_runner(anima_dir: Path) -> "AnimaRunner":
    """Create a minimal AnimaRunner with mocked anima for recovery tests.

    Reuses the pattern from tests/test_streaming_journal.py.
    """
    from core.supervisor.runner import AnimaRunner

    runner = AnimaRunner(
        anima_name=anima_dir.name,
        socket_path=anima_dir / "sock",
        animas_dir=anima_dir.parent,
        shared_dir=anima_dir.parent / "shared",
    )
    mock_anima = MagicMock()
    mock_anima.model_config = MagicMock()
    runner.anima = mock_anima
    return runner


class TestRunnerToolUseRecovery:
    """Tests that AnimaRunner._recover_streaming_journal() logs tool_use events.

    Verifies the new tool_use logging added in the data durability changes.
    """

    @pytest.fixture
    def journal(self, anima_dir: Path) -> StreamingJournal:
        """Create a StreamingJournal instance."""
        return StreamingJournal(anima_dir)

    def test_tool_calls_logged_as_tool_use_events(
        self, journal: StreamingJournal, anima_dir: Path,
    ):
        """Orphaned journal with tool calls should produce tool_use activity log entries.

        Create an orphaned journal with multiple tool calls, run recovery,
        and verify ActivityLogger.log() is called with "tool_use" events
        for each tool call.
        """
        journal.open(trigger="chat", from_person="user")
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("some output")
        journal.write_tool_start("web_search", args_summary="q=hello")
        journal.write_tool_end("web_search", result_summary="2 results")
        journal.write_tool_start("github", args_summary="pr list")
        journal.write_tool_end("github", result_summary="5 PRs")
        journal.close()

        runner = _make_runner(anima_dir)
        mock_activity = MagicMock()

        with patch(
            "core.memory.conversation.ConversationMemory",
            return_value=MagicMock(),
        ), patch(
            "core.memory.activity.ActivityLogger",
            return_value=mock_activity,
        ):
            runner._recover_streaming_journal()

        # Find all log calls with event_type "tool_use"
        tool_use_calls = [
            c for c in mock_activity.log.call_args_list
            if c[0][0] == "tool_use"
        ]
        assert len(tool_use_calls) == 2, (
            f"Expected 2 tool_use log calls, got {len(tool_use_calls)}"
        )

        # Verify first tool_use entry
        first_call = tool_use_calls[0]
        assert first_call[1]["tool"] == "web_search"
        assert "[recovered]" in first_call[1]["summary"]
        assert first_call[1]["meta"]["recovered"] is True

        # Verify second tool_use entry
        second_call = tool_use_calls[1]
        assert second_call[1]["tool"] == "github"
        assert "[recovered]" in second_call[1]["summary"]
        assert second_call[1]["meta"]["recovered"] is True

    def test_confirm_recovery_called_after_save(
        self, journal: StreamingJournal, anima_dir: Path,
    ):
        """After saving recovered data, confirm_recovery() should be called
        to delete the journal file.
        """
        journal.open(trigger="chat", from_person="tester")
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("recovered text")
        journal.close()

        runner = _make_runner(anima_dir)
        mock_conv = MagicMock()

        with patch(
            "core.memory.conversation.ConversationMemory",
            return_value=mock_conv,
        ), patch(
            "core.memory.activity.ActivityLogger",
            return_value=MagicMock(),
        ), patch.object(
            StreamingJournal,
            "confirm_recovery",
            wraps=StreamingJournal.confirm_recovery,
        ) as mock_confirm:
            runner._recover_streaming_journal()

        # confirm_recovery should have been called
        mock_confirm.assert_called_once_with(anima_dir)

        # And conv.save() should have been called before confirm_recovery
        mock_conv.save.assert_called_once()

    def test_confirm_recovery_called_even_without_text(
        self, anima_dir: Path,
    ):
        """When journal has no text but exists, confirm_recovery is still called.

        The runner should clean up the journal file even when there is no
        recovered text to save to conversation memory.
        """
        # Create an empty orphaned journal (start event only, no text)
        journal = StreamingJournal(anima_dir)
        journal.open(trigger="chat")
        journal.close()

        runner = _make_runner(anima_dir)
        # Set anima to None to trigger the "no text" branch
        # (recovery.recovered_text will be "" and self.anima check passes
        # but recovered_text is empty)
        # Actually: the runner checks `recovery.recovered_text and self.anima`
        # With empty text, it goes to else branch

        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"
        assert journal_path.exists()

        with patch(
            "core.memory.activity.ActivityLogger",
            return_value=MagicMock(),
        ):
            runner._recover_streaming_journal()

        # Journal file should be cleaned up via confirm_recovery
        assert not journal_path.exists(), (
            "Journal file should be deleted even when no text is recovered"
        )


# ── 7. TestStartupTmpCleanup ──────────────────────────────────────


class TestStartupTmpCleanup:
    """Tests that AnimaRunner.run() cleans up stale .tmp files at startup."""

    def test_runner_run_contains_cleanup_calls(self):
        """AnimaRunner.run() must contain cleanup_tmp_files() calls for state/ and knowledge/.

        Since run() is a complex async method with many dependencies,
        we verify the code path via source inspection.  The actual
        cleanup_tmp_files() functionality is tested in
        test_stale_tmp_files_actually_removed and test_memory_io.py.
        """
        import inspect
        from core.supervisor.runner import AnimaRunner

        source = inspect.getsource(AnimaRunner.run)
        assert "cleanup_tmp_files" in source, (
            "AnimaRunner.run() must call cleanup_tmp_files()"
        )
        assert '"state"' in source, (
            "AnimaRunner.run() must clean up state/ directory"
        )
        assert '"knowledge"' in source, (
            "AnimaRunner.run() must clean up knowledge/ directory"
        )

    def test_stale_tmp_files_actually_removed(self, anima_dir: Path):
        """End-to-end: create stale .tmp files and verify cleanup removes them."""
        state_dir = anima_dir / "state"
        knowledge_dir = anima_dir / "knowledge"

        # Create stale .tmp files
        (state_dir / ".conversation.json.abc123.tmp").write_text("stale")
        (state_dir / ".pending.json.def456.tmp").write_text("stale")
        (knowledge_dir / ".topic.md.ghi789.tmp").write_text("stale")

        # Also create a non-.tmp file that should NOT be removed
        (state_dir / "conversation.json").write_text("{}")

        from core.memory._io import cleanup_tmp_files

        removed_state = cleanup_tmp_files(state_dir)
        removed_knowledge = cleanup_tmp_files(knowledge_dir)

        assert removed_state == 2
        assert removed_knowledge == 1
        assert (state_dir / "conversation.json").exists(), "Non-.tmp file should survive"
        assert list(state_dir.glob(".*.tmp")) == []
        assert list(knowledge_dir.glob(".*.tmp")) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
