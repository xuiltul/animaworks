from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for memory file I/O protection — graceful handling of OSError."""

from pathlib import Path
from unittest.mock import patch

import pytest

from core.exceptions import MemoryWriteError
from core.memory.conversation.memory import ConversationMemory
from core.memory.conversation.shortterm import SessionState, ShortTermMemory
from core.memory.conversation.streaming_journal import StreamingJournal
from core.memory.manager import MemoryManager
from core.schemas import ModelConfig

# ── Test 6: manager._read() returns empty on OSError ────────────────────


def test_manager_read_returns_empty_on_oserror(tmp_path: Path) -> None:
    """Mock path.read_text to raise OSError. Verify _read() returns '' instead of crashing."""
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "IDENTITY.md").write_text("content", encoding="utf-8")

    mm = MemoryManager(anima_dir)

    with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
        result = mm._read(anima_dir / "IDENTITY.md")

    assert result == ""


# ── Test 7: shortterm.save() raises MemoryWriteError on JSON write OSError ─


def test_shortterm_save_raises_memory_write_error_on_oserror(tmp_path: Path) -> None:
    """Mock the atomic JSON writer to raise OSError and verify it is wrapped."""
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "shortterm" / "chat").mkdir(parents=True)

    st = ShortTermMemory(anima_dir, session_type="chat")
    state = SessionState(
        session_id="s1",
        timestamp="2026-01-01T00:00:00",
        trigger="chat",
        original_prompt="test",
        accumulated_response="",
    )

    with (
        patch("core.platform.state_writer.atomic_write_json", side_effect=OSError("disk full")),
        pytest.raises(MemoryWriteError, match="Short-term memory save failed"),
    ):
        st.save(state)


# ── Test 8: shortterm.load() handles OSError gracefully ───────────────────


def test_shortterm_load_handles_oserror_gracefully(tmp_path: Path) -> None:
    """Mock read_text to raise OSError. Verify load() doesn't crash, returns empty/default state."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "shortterm" / "chat").mkdir(parents=True)
    (anima_dir / "shortterm" / "chat" / "session_state.json").write_text(
        '{"session_id":"x"}',
        encoding="utf-8",
    )

    st = ShortTermMemory(anima_dir, session_type="chat")

    with patch.object(Path, "read_text", side_effect=OSError("read failed")):
        result = st.load()

    assert result is None


# ── Test 9: conversation.load() handles OSError ──────────────────────────


def test_conversation_load_handles_oserror(tmp_path: Path) -> None:
    """Mock read_text to raise OSError. Verify load() handles it (same as JSONDecodeError path)."""
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()
    (anima_dir / "state").mkdir()
    (anima_dir / "state" / "conversation.json").write_text(
        '{"anima_name":"test","turns":[]}',
        encoding="utf-8",
    )

    model_config = ModelConfig(model="claude-sonnet-4-6")
    conv = ConversationMemory(anima_dir, model_config)

    with patch.object(Path, "read_text", side_effect=OSError("read failed")):
        result = conv.load()

    assert result is not None
    assert result.anima_name == anima_dir.name
    assert result.turns == []
    assert result.compressed_summary == ""


# ── Test 10: streaming_journal.open() handles OSError ────────────────────


def test_streaming_journal_open_handles_oserror(tmp_path: Path) -> None:
    """Mock builtins.open to raise OSError. Verify _fd is None after open()."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "shortterm").mkdir(parents=True)

    journal = StreamingJournal(anima_dir, session_type="chat")

    with patch("builtins.open", side_effect=OSError("cannot open")):
        journal.open(trigger="chat", from_person="human")

    assert journal._fd is None


def test_streaming_journal_write_event_does_nothing_when_fd_none(tmp_path: Path) -> None:
    """Verify _write_event() does nothing when _fd is None."""
    anima_dir = tmp_path / "anima"
    (anima_dir / "shortterm").mkdir(parents=True)

    journal = StreamingJournal(anima_dir, session_type="chat")
    journal._fd = None

    # Should not raise; should be a no-op
    journal._write_event({"ev": "text", "t": "hello"})

    # No file should be created
    journal_path = anima_dir / "shortterm" / "streaming_journal_chat.jsonl"
    assert not journal_path.exists()
