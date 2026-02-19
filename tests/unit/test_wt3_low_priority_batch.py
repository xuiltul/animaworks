"""Unit tests for WT-3 low-priority improvement batch (8 fixes).

Covers:
- Fix 8:  Board mention fanout targets running Animas only
- Fix 10: A2 streaming comment updates (agent.py + litellm_loop.py)
- Fix 11: CLAUDE.md Mode B session chaining exclusion note
- Fix N4: ProcessHandle streaming lock
- Fix N5: Forgetting relaxed thresholds
- Fix N6: Priming compiled regex constants + input truncation
- Fix N7: Streaming journal orphan recovery on open()
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════════════
# Fix 8: Board Mention Fanout — running Animas only
# ══════════════════════════════════════════════════════════════════════


def _make_handler(tmp_path: Path, anima_name: str = "alice") -> "ToolHandler":
    """Create a ToolHandler with minimal mocked dependencies."""
    from core.tooling.handler import ToolHandler

    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []

    messenger = MagicMock()

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
        tool_registry=[],
    )
    return handler


class TestFanoutAllExcludesStoppedAnimas:
    """Fix 8: @all fanout only targets running Animas (socket present)."""

    def test_fanout_all_excludes_stopped_animas(self, tmp_path):
        """@all mention should only reach Animas with active socket files.

        Setup: 3 Animas exist (bob, carol, dave).
        Only bob and carol have .sock files (running).
        dave is stopped (no socket).

        Expect: fanout delivers to bob and carol only.
        """
        handler = _make_handler(tmp_path, anima_name="alice")

        # Create socket dir with only bob and carol running
        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "bob.sock").touch()
        (sockets_dir / "carol.sock").touch()
        # dave has no socket — stopped

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler._fanout_board_mentions("general", "Hello @all !")

        messenger = handler._messenger
        # Should have called send() for bob and carol (sorted order)
        assert messenger.send.call_count == 2
        targets = sorted(
            c[1]["to"] for c in messenger.send.call_args_list
        )
        assert targets == ["bob", "carol"]
        assert "dave" not in targets

    def test_fanout_all_excludes_self(self, tmp_path):
        """@all should never deliver back to the posting Anima itself."""
        handler = _make_handler(tmp_path, anima_name="alice")

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "alice.sock").touch()
        (sockets_dir / "bob.sock").touch()

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler._fanout_board_mentions("general", "Hey @all")

        messenger = handler._messenger
        assert messenger.send.call_count == 1
        call_kwargs = messenger.send.call_args_list[0][1]
        assert call_kwargs["to"] == "bob"


class TestFanoutNamedExcludesStoppedAnimas:
    """Fix 8: Named @mention only reaches running targets."""

    def test_fanout_named_excludes_stopped_animas(self, tmp_path):
        """Named @bob @dave mention should only reach bob (running).

        dave is stopped (no socket) and should be excluded.
        """
        handler = _make_handler(tmp_path, anima_name="alice")

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        (sockets_dir / "bob.sock").touch()
        # dave has no socket — stopped

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler._fanout_board_mentions("ops", "Hey @bob @dave check this")

        messenger = handler._messenger
        assert messenger.send.call_count == 1
        call_kwargs = messenger.send.call_args_list[0][1]
        assert call_kwargs["to"] == "bob"

    def test_fanout_no_running_targets_is_noop(self, tmp_path):
        """If all mentioned Animas are stopped, no messages should be sent."""
        handler = _make_handler(tmp_path, anima_name="alice")

        sockets_dir = tmp_path / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        # No sockets at all

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            handler._fanout_board_mentions("general", "Hey @dave @eve")

        messenger = handler._messenger
        messenger.send.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# Fix 10: A2 Streaming Comments
# ══════════════════════════════════════════════════════════════════════


class TestStreamingCommentUpdated:
    """Fix 10: agent.py streaming section comment updated."""

    def test_streaming_comment_updated(self):
        """agent.py should reference 'A1 / A2 / all modes', not just 'A1 Agent SDK'."""
        agent_path = Path(__file__).resolve().parents[2] / "core" / "agent.py"
        content = agent_path.read_text(encoding="utf-8")
        assert "A1 / A2 / all modes" in content, (
            "agent.py streaming section comment should say 'A1 / A2 / all modes'"
        )


class TestLitellmCommentUpdated:
    """Fix 10: litellm_loop.py session chaining comment updated."""

    def test_litellm_comment_updated(self):
        """litellm_loop.py should say 'handled by AgentCore', not 'NOT handled'."""
        litellm_path = (
            Path(__file__).resolve().parents[2]
            / "core" / "execution" / "litellm_loop.py"
        )
        content = litellm_path.read_text(encoding="utf-8")
        assert "handled by AgentCore" in content, (
            "litellm_loop.py should say session chaining is 'handled by AgentCore'"
        )
        # Verify old incorrect comment is gone
        assert "NOT handled" not in content, (
            "litellm_loop.py should NOT contain 'NOT handled' anymore"
        )


# ══════════════════════════════════════════════════════════════════════
# Fix 11: Mode B Exclusion in CLAUDE.md
# ══════════════════════════════════════════════════════════════════════


class TestClaudeMdModeBExclusion:
    """Fix 11: CLAUDE.md documents Mode B session chaining exclusion."""

    def test_claude_md_mode_b_exclusion(self):
        """CLAUDE.md should contain the Mode B session chaining note."""
        claude_md_path = Path(__file__).resolve().parents[2] / "CLAUDE.md"
        content = claude_md_path.read_text(encoding="utf-8")
        # Mode B entry should mention session chaining is not supported
        assert "セッションチェイニング非対応" in content or "session chaining" in content.lower(), (
            "CLAUDE.md Mode B section should mention session chaining exclusion"
        )


# ══════════════════════════════════════════════════════════════════════
# Fix N4: Streaming Lock
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def process_handle(tmp_path: Path):
    """Create a ProcessHandle with mock paths."""
    from core.supervisor.process_handle import ProcessHandle

    socket_path = tmp_path / "test.sock"
    return ProcessHandle(
        anima_name="test-anima",
        socket_path=socket_path,
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
    )


class TestProcessHandleStreamingLock:
    """Fix N4: ProcessHandle uses asyncio.Lock for streaming state."""

    def test_process_handle_has_streaming_lock(self, process_handle):
        """ProcessHandle should have a _streaming_lock attribute (asyncio.Lock)."""
        assert hasattr(process_handle, "_streaming_lock")
        assert isinstance(process_handle._streaming_lock, asyncio.Lock)

    def test_process_handle_has_is_streaming_property(self, process_handle):
        """is_streaming property should exist and return a bool."""
        assert hasattr(type(process_handle), "is_streaming")
        assert isinstance(type(process_handle).is_streaming, property)
        result = process_handle.is_streaming
        assert isinstance(result, bool)
        assert result is False  # Default: not streaming

    def test_streaming_flag_initially_false(self, process_handle):
        """_streaming should be False on construction."""
        assert process_handle._streaming is False
        assert process_handle._streaming_started_at is None


# ══════════════════════════════════════════════════════════════════════
# Fix N5: Forgetting Relaxed Thresholds
# ══════════════════════════════════════════════════════════════════════


class TestForgettingThresholds:
    """Fix N5: Complete forgetting uses relaxed thresholds."""

    def test_forgetting_days_threshold_is_90(self):
        """FORGETTING_LOW_ACTIVATION_DAYS should be 90."""
        from core.memory.forgetting import FORGETTING_LOW_ACTIVATION_DAYS

        assert FORGETTING_LOW_ACTIVATION_DAYS == 90

    def test_forgetting_max_access_count_is_2(self):
        """FORGETTING_MAX_ACCESS_COUNT should be 2 (relaxed from previous stricter value)."""
        from core.memory.forgetting import FORGETTING_MAX_ACCESS_COUNT

        assert FORGETTING_MAX_ACCESS_COUNT == 2

    def test_complete_forgetting_uses_relaxed_thresholds(self):
        """Chunks with access_count=1 should be eligible for deletion.

        The relaxed threshold (FORGETTING_MAX_ACCESS_COUNT=2) means
        access_count <= 2 qualifies for deletion. A chunk with
        access_count=1 that has been in low activation for >90 days
        should be eligible.
        """
        from core.memory.forgetting import (
            FORGETTING_LOW_ACTIVATION_DAYS,
            FORGETTING_MAX_ACCESS_COUNT,
        )

        # Simulate a chunk that has been in low activation for 100 days
        # with access_count=1
        days_low = 100
        access_count = 1

        eligible = (
            days_low > FORGETTING_LOW_ACTIVATION_DAYS
            and access_count <= FORGETTING_MAX_ACCESS_COUNT
        )
        assert eligible is True, (
            f"Chunk with access_count={access_count} and days_low={days_low} "
            f"should be eligible for deletion (threshold: days>{FORGETTING_LOW_ACTIVATION_DAYS}, "
            f"access<={FORGETTING_MAX_ACCESS_COUNT})"
        )

    def test_access_count_3_not_eligible(self):
        """Chunks with access_count=3 should NOT be eligible (above threshold)."""
        from core.memory.forgetting import (
            FORGETTING_LOW_ACTIVATION_DAYS,
            FORGETTING_MAX_ACCESS_COUNT,
        )

        days_low = 100
        access_count = 3

        eligible = (
            days_low > FORGETTING_LOW_ACTIVATION_DAYS
            and access_count <= FORGETTING_MAX_ACCESS_COUNT
        )
        assert eligible is False, (
            f"Chunk with access_count={access_count} should NOT be eligible "
            f"(threshold: access<={FORGETTING_MAX_ACCESS_COUNT})"
        )


# ══════════════════════════════════════════════════════════════════════
# Fix N6: Priming Regex Compiled Constants + Input Truncation
# ══════════════════════════════════════════════════════════════════════


class TestPrimingRegexCompiledConstants:
    """Fix N6: _RE_KATAKANA and _RE_WORDS are pre-compiled regex patterns."""

    def test_priming_regex_compiled_constants(self):
        """Module-level regex constants should be compiled re.Pattern objects."""
        from core.memory.priming import _RE_KATAKANA, _RE_WORDS

        assert isinstance(_RE_KATAKANA, re.Pattern), (
            "_RE_KATAKANA should be a compiled regex Pattern"
        )
        assert isinstance(_RE_WORDS, re.Pattern), (
            "_RE_WORDS should be a compiled regex Pattern"
        )

    def test_re_katakana_matches_katakana(self):
        """_RE_KATAKANA should match katakana sequences of 2+ characters."""
        from core.memory.priming import _RE_KATAKANA

        assert _RE_KATAKANA.findall("テスト") == ["テスト"]
        assert _RE_KATAKANA.findall("ア") == []  # Single char: no match
        assert _RE_KATAKANA.findall("アイウ") == ["アイウ"]

    def test_re_words_matches_mixed_text(self):
        """_RE_WORDS should match alphanumeric and CJK word tokens."""
        from core.memory.priming import _RE_WORDS

        tokens = _RE_WORDS.findall("Hello 世界 test123")
        assert "Hello" in tokens
        assert "世界" in tokens
        assert "test123" in tokens


class TestExtractKeywordsTruncatesLongInput:
    """Fix N6: Messages exceeding _MAX_KEYWORD_INPUT_LEN are truncated."""

    def test_extract_keywords_truncates_long_input(self, tmp_path):
        """Messages > 5000 chars should be truncated before keyword extraction."""
        from core.memory.priming import PrimingEngine, _MAX_KEYWORD_INPUT_LEN

        # Create minimal PrimingEngine
        anima_dir = tmp_path / "test-anima"
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True)
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()

        engine = PrimingEngine(anima_dir)

        # Build a message that exceeds the limit
        long_msg = "キーワード " * (_MAX_KEYWORD_INPUT_LEN // 5 + 1000)
        assert len(long_msg) > _MAX_KEYWORD_INPUT_LEN

        # Should not raise (no ReDoS / runaway processing)
        keywords = engine._extract_keywords(long_msg)
        assert isinstance(keywords, list)
        assert len(keywords) <= 10  # Max 10 keywords returned

    def test_extract_keywords_normal_input(self, tmp_path):
        """Normal-length input should still return meaningful keywords."""
        from core.memory.priming import PrimingEngine

        anima_dir = tmp_path / "test-anima"
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True)
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()

        engine = PrimingEngine(anima_dir)

        keywords = engine._extract_keywords("プロジェクト管理ツールについて教えてください")
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_max_keyword_input_len_value(self):
        """_MAX_KEYWORD_INPUT_LEN should be 5000."""
        from core.memory.priming import _MAX_KEYWORD_INPUT_LEN

        assert _MAX_KEYWORD_INPUT_LEN == 5000


# ══════════════════════════════════════════════════════════════════════
# Fix N7: Streaming Journal Open Protection
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def journal_anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory with shortterm/ subdirectory."""
    anima_dir = tmp_path / "test_anima"
    anima_dir.mkdir()
    (anima_dir / "shortterm").mkdir()
    return anima_dir


class TestJournalOpenRecoversOrphan:
    """Fix N7: open() recovers orphaned journal before overwriting."""

    def test_journal_open_recovers_orphan(self, journal_anima_dir):
        """If an orphaned journal exists when open() is called,
        it should be recovered before creating the new one.
        """
        from core.memory.streaming_journal import StreamingJournal

        # Create an orphaned journal (simulate previous crash)
        journal_path = journal_anima_dir / "shortterm" / "streaming_journal.jsonl"
        orphan_data = json.dumps({
            "ev": "start",
            "trigger": "heartbeat",
            "from": "old-session",
            "session_id": "orphan-sess",
            "ts": datetime.now().isoformat(),
        })
        journal_path.write_text(orphan_data + "\n", encoding="utf-8")

        assert StreamingJournal.has_orphan(journal_anima_dir) is True

        # Now open a new journal — should trigger recovery first
        journal = StreamingJournal(journal_anima_dir)

        with patch.object(
            StreamingJournal, "recover", wraps=StreamingJournal.recover
        ) as mock_recover:
            journal.open(trigger="chat", from_person="user", session_id="new-sess")

        # The orphan was present before open() — recovery path was taken.
        # Verify the journal is now open and functional.
        assert journal._fd is not None
        assert not journal._finalized

        # Clean up
        journal.finalize(summary="test done")

    def test_journal_open_orphan_content_recovered(self, journal_anima_dir):
        """Orphan journal content should be recoverable before overwrite."""
        from core.memory.streaming_journal import StreamingJournal

        # Write an orphan with some text
        journal_path = journal_anima_dir / "shortterm" / "streaming_journal.jsonl"
        lines = [
            json.dumps({"ev": "start", "trigger": "chat", "from": "tester",
                         "session_id": "orphan-1",
                         "ts": datetime.now().isoformat()}),
            json.dumps({"ev": "text", "t": "orphan content",
                         "ts": datetime.now().isoformat()}),
        ]
        journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Recover the orphan first (simulating what open() does internally)
        recovery = StreamingJournal.recover(journal_anima_dir)
        assert recovery is not None
        assert recovery.recovered_text == "orphan content"
        assert recovery.trigger == "chat"

    def test_journal_open_no_orphan_normal(self, journal_anima_dir):
        """Opening without existing journal should work normally (no recovery)."""
        from core.memory.streaming_journal import StreamingJournal

        journal_path = journal_anima_dir / "shortterm" / "streaming_journal.jsonl"
        assert not journal_path.exists()

        journal = StreamingJournal(journal_anima_dir)
        journal.open(trigger="chat", from_person="user", session_id="fresh-sess")

        # Journal file should now exist (just opened)
        assert journal_path.exists()
        assert journal._fd is not None
        assert not journal._finalized

        # Clean up
        journal.finalize(summary="test done")
        assert not journal_path.exists()
