# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for stream checkpoint and retry feature.

Covers StreamCheckpoint dataclass, ShortTermMemory checkpoint methods,
build_stream_retry_prompt helper, and StreamDisconnectedError.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from core.memory.shortterm import ShortTermMemory, StreamCheckpoint
from core.execution._session import build_stream_retry_prompt
from core.execution.agent_sdk import StreamDisconnectedError


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create an isolated anima directory with shortterm subdirectory."""
    d = tmp_path / "animas" / "test-anima"
    (d / "shortterm").mkdir(parents=True)
    return d


@pytest.fixture
def shortterm(anima_dir: Path) -> ShortTermMemory:
    """Return a ShortTermMemory instance rooted at the test anima dir."""
    return ShortTermMemory(anima_dir)


@pytest.fixture
def sample_checkpoint() -> StreamCheckpoint:
    """Return a StreamCheckpoint with representative data."""
    return StreamCheckpoint(
        timestamp="2026-02-16T10:30:00",
        trigger="heartbeat",
        original_prompt="Deploy the new configuration to staging.",
        completed_tools=[
            {"tool_name": "Bash", "summary": "Listed files in /app"},
            {"tool_name": "Read", "summary": "Read config.yaml"},
        ],
        accumulated_text="Step 1 complete. Step 2 in progress...",
        retry_count=1,
    )


# ── StreamCheckpoint dataclass ──────────────────────────────


class TestStreamCheckpoint:
    """Tests for the StreamCheckpoint dataclass itself."""

    def test_create_with_defaults(self) -> None:
        """Default-constructed checkpoint has empty fields and zero retry."""
        cp = StreamCheckpoint()
        assert cp.timestamp == ""
        assert cp.trigger == ""
        assert cp.original_prompt == ""
        assert cp.completed_tools == []
        assert cp.accumulated_text == ""
        assert cp.retry_count == 0

    def test_create_with_values(self, sample_checkpoint: StreamCheckpoint) -> None:
        """Checkpoint created with explicit values retains them."""
        assert sample_checkpoint.timestamp == "2026-02-16T10:30:00"
        assert sample_checkpoint.trigger == "heartbeat"
        assert sample_checkpoint.original_prompt == "Deploy the new configuration to staging."
        assert len(sample_checkpoint.completed_tools) == 2
        assert sample_checkpoint.completed_tools[0]["tool_name"] == "Bash"
        assert sample_checkpoint.accumulated_text == "Step 1 complete. Step 2 in progress..."
        assert sample_checkpoint.retry_count == 1

    def test_serialize_to_dict(self, sample_checkpoint: StreamCheckpoint) -> None:
        """asdict() produces a plain dict suitable for JSON serialization."""
        d = asdict(sample_checkpoint)
        assert isinstance(d, dict)
        assert d["timestamp"] == "2026-02-16T10:30:00"
        assert d["retry_count"] == 1
        # Verify JSON round-trip works without errors
        json_str = json.dumps(d, ensure_ascii=False)
        assert isinstance(json_str, str)

    def test_deserialize_from_dict(self) -> None:
        """StreamCheckpoint can be reconstructed from a deserialized dict."""
        raw: dict[str, Any] = {
            "timestamp": "2026-02-16T11:00:00",
            "trigger": "cron",
            "original_prompt": "Check server health",
            "completed_tools": [{"tool_name": "Bash", "summary": "ran uptime"}],
            "accumulated_text": "Server is healthy.",
            "retry_count": 0,
        }
        cp = StreamCheckpoint(**raw)
        assert cp.timestamp == "2026-02-16T11:00:00"
        assert cp.trigger == "cron"
        assert cp.completed_tools[0]["tool_name"] == "Bash"

    def test_round_trip_json(self, sample_checkpoint: StreamCheckpoint) -> None:
        """Serialize to JSON and back produces an equal checkpoint."""
        json_str = json.dumps(asdict(sample_checkpoint), ensure_ascii=False)
        restored = StreamCheckpoint(**json.loads(json_str))
        assert restored.timestamp == sample_checkpoint.timestamp
        assert restored.trigger == sample_checkpoint.trigger
        assert restored.original_prompt == sample_checkpoint.original_prompt
        assert restored.completed_tools == sample_checkpoint.completed_tools
        assert restored.accumulated_text == sample_checkpoint.accumulated_text
        assert restored.retry_count == sample_checkpoint.retry_count


# ── ShortTermMemory.save_checkpoint() ────────────────────────


class TestSaveCheckpoint:
    """Tests for ShortTermMemory.save_checkpoint()."""

    def test_saves_json_file(
        self,
        shortterm: ShortTermMemory,
        sample_checkpoint: StreamCheckpoint,
    ) -> None:
        """save_checkpoint() creates stream_checkpoint.json in shortterm dir."""
        path = shortterm.save_checkpoint(sample_checkpoint)
        assert path.exists()
        assert path.name == "stream_checkpoint.json"
        assert path.parent == shortterm.shortterm_dir

    def test_saved_content_is_valid_json(
        self,
        shortterm: ShortTermMemory,
        sample_checkpoint: StreamCheckpoint,
    ) -> None:
        """The saved file contains valid JSON matching the checkpoint data."""
        path = shortterm.save_checkpoint(sample_checkpoint)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["timestamp"] == "2026-02-16T10:30:00"
        assert data["trigger"] == "heartbeat"
        assert data["retry_count"] == 1
        assert len(data["completed_tools"]) == 2

    def test_creates_shortterm_dir_if_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """save_checkpoint() creates the shortterm directory when absent."""
        anima_dir = tmp_path / "animas" / "new-anima"
        anima_dir.mkdir(parents=True)
        # Do NOT create shortterm/ — let save_checkpoint handle it
        stm = ShortTermMemory(anima_dir)
        cp = StreamCheckpoint(timestamp="2026-02-16T12:00:00")
        path = stm.save_checkpoint(cp)
        assert path.exists()
        assert (anima_dir / "shortterm").is_dir()

    def test_overwrites_existing_checkpoint(
        self,
        shortterm: ShortTermMemory,
    ) -> None:
        """A second save overwrites the previous checkpoint file."""
        cp1 = StreamCheckpoint(timestamp="first", retry_count=0)
        cp2 = StreamCheckpoint(timestamp="second", retry_count=1)
        shortterm.save_checkpoint(cp1)
        path = shortterm.save_checkpoint(cp2)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["timestamp"] == "second"
        assert data["retry_count"] == 1


# ── ShortTermMemory.load_checkpoint() ────────────────────────


class TestLoadCheckpoint:
    """Tests for ShortTermMemory.load_checkpoint()."""

    def test_loads_saved_checkpoint(
        self,
        shortterm: ShortTermMemory,
        sample_checkpoint: StreamCheckpoint,
    ) -> None:
        """load_checkpoint() returns a StreamCheckpoint matching what was saved."""
        shortterm.save_checkpoint(sample_checkpoint)
        loaded = shortterm.load_checkpoint()
        assert loaded is not None
        assert loaded.timestamp == sample_checkpoint.timestamp
        assert loaded.trigger == sample_checkpoint.trigger
        assert loaded.original_prompt == sample_checkpoint.original_prompt
        assert loaded.completed_tools == sample_checkpoint.completed_tools
        assert loaded.accumulated_text == sample_checkpoint.accumulated_text
        assert loaded.retry_count == sample_checkpoint.retry_count

    def test_returns_none_when_no_file(
        self,
        shortterm: ShortTermMemory,
    ) -> None:
        """load_checkpoint() returns None when no checkpoint file exists."""
        assert shortterm.load_checkpoint() is None

    def test_returns_none_on_corrupt_json(
        self,
        shortterm: ShortTermMemory,
    ) -> None:
        """load_checkpoint() returns None when the file contains invalid JSON."""
        corrupt_path = shortterm.shortterm_dir / "stream_checkpoint.json"
        corrupt_path.write_text("{invalid json!!!", encoding="utf-8")
        result = shortterm.load_checkpoint()
        assert result is None

    def test_returns_none_on_wrong_schema(
        self,
        shortterm: ShortTermMemory,
    ) -> None:
        """load_checkpoint() returns None when JSON has unexpected keys."""
        bad_schema_path = shortterm.shortterm_dir / "stream_checkpoint.json"
        bad_schema_path.write_text(
            json.dumps({"unexpected_key": True}),
            encoding="utf-8",
        )
        result = shortterm.load_checkpoint()
        # StreamCheckpoint(**data) raises TypeError for unexpected keys
        assert result is None


# ── ShortTermMemory.clear_checkpoint() ───────────────────────


class TestClearCheckpoint:
    """Tests for ShortTermMemory.clear_checkpoint()."""

    def test_removes_checkpoint_file(
        self,
        shortterm: ShortTermMemory,
        sample_checkpoint: StreamCheckpoint,
    ) -> None:
        """clear_checkpoint() removes the checkpoint file."""
        path = shortterm.save_checkpoint(sample_checkpoint)
        assert path.exists()
        shortterm.clear_checkpoint()
        assert not path.exists()

    def test_no_error_when_no_file(
        self,
        shortterm: ShortTermMemory,
    ) -> None:
        """clear_checkpoint() is a no-op when no checkpoint file exists."""
        # Should not raise
        shortterm.clear_checkpoint()

    def test_load_returns_none_after_clear(
        self,
        shortterm: ShortTermMemory,
        sample_checkpoint: StreamCheckpoint,
    ) -> None:
        """After clearing, load_checkpoint() returns None."""
        shortterm.save_checkpoint(sample_checkpoint)
        shortterm.clear_checkpoint()
        assert shortterm.load_checkpoint() is None


# ── build_stream_retry_prompt() ──────────────────────────────


class TestBuildStreamRetryPrompt:
    """Tests for the build_stream_retry_prompt() helper."""

    def test_constructs_prompt_with_completed_tools(self) -> None:
        """Prompt includes numbered completed tool entries."""
        cp = StreamCheckpoint(
            original_prompt="Run deployment pipeline",
            completed_tools=[
                {"tool_name": "Bash", "summary": "git pull completed"},
                {"tool_name": "Write", "summary": "Updated config.yaml"},
            ],
            accumulated_text="Pulling latest code...",
        )
        prompt = build_stream_retry_prompt(cp)

        # Contains original prompt
        assert "Run deployment pipeline" in prompt
        # Contains numbered tool entries
        assert "1." in prompt
        assert "Bash" in prompt
        assert "git pull completed" in prompt
        assert "2." in prompt
        assert "Write" in prompt
        assert "Updated config.yaml" in prompt
        # Contains accumulated text
        assert "Pulling latest code..." in prompt

    def test_empty_tools_shows_none_marker(self) -> None:
        """With no completed tools, the prompt uses the (なし) placeholder."""
        cp = StreamCheckpoint(
            original_prompt="Check system status",
            completed_tools=[],
            accumulated_text="",
        )
        prompt = build_stream_retry_prompt(cp)

        assert "Check system status" in prompt
        assert "(なし)" in prompt

    def test_long_accumulated_text_truncated(self) -> None:
        """Accumulated text longer than 2000 chars is truncated with prefix."""
        long_text = "x" * 5000
        cp = StreamCheckpoint(
            original_prompt="Process data",
            completed_tools=[],
            accumulated_text=long_text,
        )
        prompt = build_stream_retry_prompt(cp)

        # The truncated text should contain the suffix marker
        assert "...(前半省略)..." in prompt
        # The full 5000-char string should NOT appear verbatim
        assert long_text not in prompt
        # The tail 2000 chars should be present
        assert "x" * 2000 in prompt

    def test_accumulated_text_at_exactly_2000_not_truncated(self) -> None:
        """Text of exactly 2000 chars is kept as-is (no truncation)."""
        exact_text = "y" * 2000
        cp = StreamCheckpoint(
            original_prompt="Exact boundary test",
            completed_tools=[],
            accumulated_text=exact_text,
        )
        prompt = build_stream_retry_prompt(cp)

        assert "...(前半省略)..." not in prompt
        assert exact_text in prompt

    def test_prompt_includes_retry_instructions(self) -> None:
        """Prompt contains instructions to avoid repeating completed steps."""
        cp = StreamCheckpoint(
            original_prompt="Deploy",
            completed_tools=[{"tool_name": "Bash", "summary": "done"}],
            accumulated_text="partial output",
        )
        prompt = build_stream_retry_prompt(cp)

        # Check for key instruction phrases
        assert "完了済みステップを繰り返さないでください" in prompt
        assert "中断前の作業の続きを実行してください" in prompt

    def test_prompt_structure_has_all_sections(self) -> None:
        """The prompt includes all expected markdown sections."""
        cp = StreamCheckpoint(
            original_prompt="Test task",
            completed_tools=[{"tool_name": "Read", "summary": "read file"}],
            accumulated_text="some output",
        )
        prompt = build_stream_retry_prompt(cp)

        assert "## 元の指示" in prompt
        assert "## 完了済みステップ" in prompt
        assert "## これまでの出力" in prompt
        assert "## 注意" in prompt


# ── StreamDisconnectedError ──────────────────────────────────


class TestStreamDisconnectedError:
    """Tests for StreamDisconnectedError exception class."""

    def test_carries_partial_text(self) -> None:
        """The exception stores partial_text from the interrupted stream."""
        err = StreamDisconnectedError(
            "connection lost",
            partial_text="partial response here",
        )
        assert err.partial_text == "partial response here"
        assert str(err) == "connection lost"

    def test_default_partial_text_is_empty(self) -> None:
        """Default partial_text is an empty string."""
        err = StreamDisconnectedError("disconnected")
        assert err.partial_text == ""

    def test_default_message(self) -> None:
        """Default message is 'Stream disconnected'."""
        err = StreamDisconnectedError()
        assert str(err) == "Stream disconnected"
        assert err.partial_text == ""

    def test_is_exception_subclass(self) -> None:
        """StreamDisconnectedError is a proper Exception subclass."""
        err = StreamDisconnectedError("test")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """The error can be raised, caught, and partial_text retrieved."""
        with pytest.raises(StreamDisconnectedError) as exc_info:
            raise StreamDisconnectedError(
                "stream broke",
                partial_text="data before break",
            )
        assert exc_info.value.partial_text == "data before break"
        assert "stream broke" in str(exc_info.value)
