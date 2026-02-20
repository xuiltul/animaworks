from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for streaming journal — crash-resilient write-ahead log.

Tests cover:
- Normal lifecycle (open → write → finalize)
- Crash recovery (orphaned journal detection and recovery)
- Tool event recording and recovery
- Buffered write behaviour
- Corrupted JSONL line handling
- Edge cases (empty journal, no orphan, double open, finalize-then-close)
- AnimaRunner._recover_streaming_journal() integration
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.streaming_journal import StreamingJournal, JournalRecovery


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory with shortterm/ subdirectory."""
    anima_dir = tmp_path / "test_anima"
    anima_dir.mkdir()
    (anima_dir / "shortterm").mkdir()
    return anima_dir


@pytest.fixture
def journal(anima_dir: Path) -> StreamingJournal:
    """Create a StreamingJournal instance bound to the temp anima directory."""
    return StreamingJournal(anima_dir)


# ── Normal Lifecycle ────────────────────────────────────────────────


class TestNormalLifecycle:
    """Test the happy path: open → write → finalize."""

    def test_normal_lifecycle(self, journal: StreamingJournal, anima_dir: Path):
        """Open, write multiple text chunks, then finalize.

        After finalize the journal file must be deleted and has_orphan()
        must return False.
        """
        journal.open(trigger="chat", from_person="user", session_id="sess-1")
        journal.write_text("Hello ")
        journal.write_text("world.")
        journal.finalize(summary="completed normally")

        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"
        assert not journal_path.exists(), "Journal file should be deleted after finalize"
        assert StreamingJournal.has_orphan(anima_dir) is False


# ── Crash Recovery ──────────────────────────────────────────────────


class TestCrashRecovery:
    """Test recovery from orphaned journals (simulated crash)."""

    def test_crash_recovery(self, journal: StreamingJournal, anima_dir: Path):
        """Open, write text, then close without finalize.

        has_orphan() must return True.  recover() must return a
        JournalRecovery whose recovered_text equals the concatenation
        of all written fragments.  After confirm_recovery() the file
        is deleted.
        """
        journal.open(trigger="chat", from_person="tester", session_id="s-crash")

        # Force immediate flush by patching the time threshold
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("chunk-A ")
            journal.write_text("chunk-B")

        journal.close()

        assert StreamingJournal.has_orphan(anima_dir) is True

        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        assert isinstance(recovery, JournalRecovery)
        assert recovery.recovered_text == "chunk-A chunk-B"
        assert recovery.trigger == "chat"
        assert recovery.from_person == "tester"
        assert recovery.session_id == "s-crash"
        assert recovery.is_complete is False

        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"
        # recover() no longer deletes the file; caller must confirm_recovery()
        assert journal_path.exists(), "Journal file should survive after recover()"
        StreamingJournal.confirm_recovery(anima_dir)
        assert not journal_path.exists(), "Journal file should be deleted after confirm_recovery()"


# ── Tool Events Recovery ────────────────────────────────────────────


class TestToolEventsRecovery:
    """Test that tool_start / tool_end events are properly recovered."""

    def test_tool_events_recovery(self, journal: StreamingJournal, anima_dir: Path):
        """Write text and tool events, then recover.

        The recovered tool_calls list must contain the correct tool name,
        args summary, result summary, and status.
        """
        journal.open(trigger="heartbeat")

        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("before-tool ")

        journal.write_tool_start("web_search", args_summary="query=test")
        journal.write_tool_end("web_search", result_summary="3 results found")

        journal.close()

        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        assert len(recovery.tool_calls) == 1

        tc = recovery.tool_calls[0]
        assert tc["tool"] == "web_search"
        assert tc["args_summary"] == "query=test"
        assert tc["result_summary"] == "3 results found"
        assert tc["status"] == "completed"

        assert "before-tool" in recovery.recovered_text


# ── Buffering ───────────────────────────────────────────────────────


class TestBuffering:
    """Test that small writes are buffered and flushed correctly."""

    def test_buffering(self, journal: StreamingJournal, anima_dir: Path):
        """Write small text fragments that stay below the flush threshold.

        After close() and recover(), all fragments must be present.
        """
        journal.open(trigger="chat")

        # Write small chunks that individually don't exceed the buffer.
        # Use the real buffer size (500 chars) — these are well below it.
        for i in range(10):
            journal.write_text(f"w{i} ")

        journal.close()

        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        expected = "".join(f"w{i} " for i in range(10))
        assert recovery.recovered_text == expected


# ── Corrupted Line Skip ─────────────────────────────────────────────


class TestCorruptedLineSkip:
    """Test that corrupted JSONL lines are silently skipped."""

    def test_corrupted_line_skip(self, anima_dir: Path):
        """Manually create a journal with valid and invalid JSONL lines.

        recover() must skip the corrupted lines and return data from
        the valid ones.
        """
        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"

        valid_start = json.dumps({
            "ev": "start",
            "trigger": "chat",
            "from": "user",
            "session_id": "s1",
            "ts": "2026-02-17T12:00:00",
        })
        valid_text = json.dumps({
            "ev": "text",
            "t": "recovered text",
            "ts": "2026-02-17T12:00:01",
        })
        corrupted_line = '{"ev": "text", "t": "broken'  # incomplete JSON

        lines = [valid_start, corrupted_line, valid_text, "not-json-at-all"]
        journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        assert recovery.recovered_text == "recovered text"
        assert recovery.trigger == "chat"
        assert recovery.from_person == "user"


# ── Empty Journal ───────────────────────────────────────────────────


class TestEmptyJournal:
    """Test behaviour when a journal is opened but nothing is written."""

    def test_empty_journal(self, journal: StreamingJournal, anima_dir: Path):
        """Open and immediately close without writing any text.

        has_orphan() must return True (the file exists).
        recover() must return a JournalRecovery with empty recovered_text.
        """
        journal.open(trigger="chat")
        journal.close()

        assert StreamingJournal.has_orphan(anima_dir) is True

        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        assert recovery.recovered_text == ""
        assert recovery.is_complete is False


# ── No Orphan ───────────────────────────────────────────────────────


class TestNoOrphan:
    """Test behaviour when no journal file exists."""

    def test_no_orphan(self, anima_dir: Path):
        """Without a journal file, has_orphan() is False and recover()
        returns None.
        """
        assert StreamingJournal.has_orphan(anima_dir) is False
        assert StreamingJournal.recover(anima_dir) is None


# ── Finalize Then Close ─────────────────────────────────────────────


class TestFinalizeThenClose:
    """Test that close() after finalize() is a harmless no-op."""

    def test_finalize_then_close_noop(
        self,
        journal: StreamingJournal,
        anima_dir: Path,
    ):
        """Open, write, finalize, then close.

        The file must already be deleted after finalize; close() should
        not raise or recreate it.
        """
        journal.open(trigger="chat")

        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("some text")

        journal.finalize(summary="done")

        journal_path = anima_dir / "shortterm" / "streaming_journal.jsonl"
        assert not journal_path.exists(), "File should be deleted after finalize"

        # close() after finalize — should be a no-op
        journal.close()
        assert not journal_path.exists(), "File should still not exist after close"


# ── Double Open ─────────────────────────────────────────────────────


class TestDoubleOpen:
    """Test that a second open() overwrites the previous journal."""

    def test_double_open(self, journal: StreamingJournal, anima_dir: Path):
        """Open, write, then open again.

        The second open must overwrite the journal file.  Only the data
        from the second session should be recoverable.
        """
        journal.open(trigger="first-session")

        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("first-data")

        # Second open overwrites the file (mode="w" in open())
        journal.open(trigger="second-session")

        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("second-data")

        journal.close()

        recovery = StreamingJournal.recover(anima_dir)
        assert recovery is not None
        assert recovery.trigger == "second-session"
        assert recovery.recovered_text == "second-data"
        assert "first-data" not in recovery.recovered_text


# ── AnimaRunner._recover_streaming_journal() Integration ───────────


def _make_runner(anima_dir: Path) -> "AnimaRunner":
    """Create a minimal AnimaRunner with mocked anima for recovery tests.

    Sets only the fields that ``_recover_streaming_journal()`` actually
    reads: ``anima_name``, ``_anima_dir``, and ``anima`` (with
    ``model_config``).
    """
    from core.supervisor.runner import AnimaRunner

    # AnimaRunner.__init__ requires socket_path / animas_dir / shared_dir,
    # but _recover_streaming_journal only uses self._anima_dir and
    # self.anima (for model_config).  We supply the real anima_dir via
    # animas_dir so that self._anima_dir resolves correctly.
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


class TestRecoverStreamingJournal:
    """Integration tests for AnimaRunner._recover_streaming_journal().

    Verifies that the runner correctly bridges StreamingJournal recovery
    data into ConversationMemory and ActivityLogger.
    """

    def test_recover_saves_to_conversation_memory(
        self,
        journal: StreamingJournal,
        anima_dir: Path,
    ):
        """Orphaned journal text is saved to ConversationMemory with crash marker.

        Create an orphaned journal (open, write, close without finalize),
        then invoke ``_recover_streaming_journal`` and verify that
        ConversationMemory.append_turn() receives the recovered text
        suffixed with the crash marker string.
        """
        # Create orphaned journal
        journal.open(trigger="chat", from_person="tester", session_id="s-1")
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("partial response")
        journal.close()

        runner = _make_runner(anima_dir)
        mock_conv = MagicMock()

        with patch(
            "core.memory.conversation.ConversationMemory",
            return_value=mock_conv,
        ) as conv_cls, patch(
            "core.memory.activity.ActivityLogger",
        ):
            runner._recover_streaming_journal()

        # ConversationMemory was instantiated with anima_dir and model_config
        conv_cls.assert_called_once_with(anima_dir, runner.anima.model_config)

        # append_turn received the recovered text + crash marker
        mock_conv.append_turn.assert_called_once()
        call_args = mock_conv.append_turn.call_args
        assert call_args[0][0] == "assistant"
        saved_text = call_args[0][1]
        assert "partial response" in saved_text
        assert "[応答が中断されました]" in saved_text

        # save() was called after append_turn
        mock_conv.save.assert_called_once()

    def test_recover_logs_to_activity_logger(
        self,
        journal: StreamingJournal,
        anima_dir: Path,
    ):
        """Crash recovery event is logged via ActivityLogger.log().

        Verify that ActivityLogger.log() is called with event_type "error",
        a summary containing the crash message, and metadata containing
        recovered_chars, trigger, tool_calls count, from_person, and
        timing information.
        """
        # Create orphaned journal with tool call
        journal.open(trigger="heartbeat", from_person="cron")
        with patch("core.memory.streaming_journal._FLUSH_SIZE_CHARS", 0):
            journal.write_text("some output")
        journal.write_tool_start("web_search", args_summary="q=hello")
        journal.write_tool_end("web_search", result_summary="2 results")
        journal.close()

        runner = _make_runner(anima_dir)
        mock_activity = MagicMock()

        with patch(
            "core.memory.conversation.ConversationMemory",
            return_value=MagicMock(),
        ), patch(
            "core.memory.activity.ActivityLogger",
            return_value=mock_activity,
        ) as activity_cls:
            runner._recover_streaming_journal()

        # ActivityLogger was instantiated twice (crash event + tool_use events)
        assert activity_cls.call_count == 2

        # log() was called multiple times: 1 error + 1 tool_use
        assert mock_activity.log.call_count == 2

        # First call: error event
        error_call = mock_activity.log.call_args_list[0]
        assert error_call[0][0] == "error"
        assert "応答が中断されました" in error_call[1]["summary"]

        meta = error_call[1]["meta"]
        assert meta["recovered_chars"] == len("some output")
        assert meta["trigger"] == "heartbeat"
        assert meta["tool_calls"] == 1
        assert meta["from_person"] == "cron"
        assert "started_at" in meta
        assert "last_event_at" in meta

        # Second call: tool_use event
        tool_call = mock_activity.log.call_args_list[1]
        assert tool_call[0][0] == "tool_use"
        assert "[recovered]" in tool_call[1]["summary"]
        assert tool_call[1]["tool"] == "web_search"
        assert tool_call[1]["meta"]["recovered"] is True

    def test_recover_no_orphan_noop(self, anima_dir: Path):
        """When no orphaned journal exists, nothing is called.

        Neither ConversationMemory nor ActivityLogger should be
        instantiated when there is no journal file to recover.
        """
        runner = _make_runner(anima_dir)

        with patch(
            "core.memory.conversation.ConversationMemory",
        ) as conv_cls, patch(
            "core.memory.activity.ActivityLogger",
        ) as activity_cls:
            runner._recover_streaming_journal()

        conv_cls.assert_not_called()
        activity_cls.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
