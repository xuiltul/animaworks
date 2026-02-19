# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for conversation data loss protection.

Verifies that conversation data is persisted to disk even when errors
occur during agent execution.  Uses REAL ConversationMemory (no mocks)
against a real filesystem, but mocks AgentCore, MemoryManager, and
Messenger to isolate the persistence logic.

Changes under test:
  1. process_message()  — user input pre-saved; error marker on failure
  2. process_message_stream() — user input pre-saved; partial + error marker
  3. process_greet()  — error marker on failure
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import CycleResult, ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_conversation(anima_dir: Path) -> list[dict]:
    """Load turns from the persisted conversation.json."""
    conv_path = anima_dir / "state" / "conversation.json"
    assert conv_path.exists(), f"conversation.json not found at {conv_path}"
    data = json.loads(conv_path.read_text(encoding="utf-8"))
    return data.get("turns", [])


def _read_transcript(anima_dir: Path) -> list[dict]:
    """Load today's JSONL transcript entries."""
    transcript_path = anima_dir / "transcripts" / f"{date.today().isoformat()}.jsonl"
    if not transcript_path.exists():
        return []
    entries = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _make_anima_with_mocks(anima_dir: Path, shared_dir: Path):
    """Construct a DigitalAnima with mocked AgentCore/MemoryManager/Messenger.

    ConversationMemory is NOT mocked — it writes to the real filesystem
    so we can verify persistence.
    """
    model_config = ModelConfig(model="claude-sonnet-4-20250514")

    with (
        patch("core.anima.AgentCore") as MockAgent,
        patch("core.anima.MemoryManager") as MockMM,
        patch("core.anima.Messenger") as MockMessenger,
    ):
        # MemoryManager.read_model_config() must return a real ModelConfig
        # so ConversationMemory receives correct configuration.
        MockMM.return_value.read_model_config.return_value = model_config

        # Messenger.unread_count() is called by the status property
        MockMessenger.return_value.unread_count.return_value = 0

        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir, shared_dir)

    return anima


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConversationDataLossProtection:
    """Verify that conversation turns survive agent errors."""

    async def test_process_message_error_preserves_user_input(
        self, make_anima, data_dir,
    ):
        """When agent.run_cycle raises, the user's input and an error
        marker must already be persisted to conversation.json."""
        anima_dir = make_anima(name="dataloss-msg-err")
        shared_dir = data_dir / "shared"
        dp = _make_anima_with_mocks(anima_dir, shared_dir)

        # Agent execution fails
        dp.agent.run_cycle = AsyncMock(
            side_effect=RuntimeError("LLM unavailable"),
        )

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await dp.process_message("test message")

        turns = _read_conversation(anima_dir)

        # Human turn must be present (pre-saved before agent execution)
        assert len(turns) >= 1
        human_turns = [t for t in turns if t["role"] == "human"]
        assert len(human_turns) == 1
        assert human_turns[0]["content"] == "test message"

        # Error marker turn must also be present
        error_turns = [
            t for t in turns
            if t["role"] == "assistant" and "ERROR" in t["content"]
        ]
        assert len(error_turns) == 1
        assert "エラー" in error_turns[0]["content"]

    async def test_process_message_success_saves_both_turns(
        self, make_anima, data_dir,
    ):
        """On success, both the human input and the assistant response
        are persisted to conversation.json."""
        anima_dir = make_anima(name="dataloss-msg-ok")
        shared_dir = data_dir / "shared"
        dp = _make_anima_with_mocks(anima_dir, shared_dir)

        # Agent execution succeeds
        dp.agent.run_cycle = AsyncMock(
            return_value=CycleResult(
                trigger="message:human",
                action="responded",
                summary="Hello! Nice to meet you.",
            ),
        )

        result = await dp.process_message("hello")
        assert result == "Hello! Nice to meet you."

        turns = _read_conversation(anima_dir)

        assert len(turns) == 2
        assert turns[0]["role"] == "human"
        assert turns[0]["content"] == "hello"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hello! Nice to meet you."

    async def test_process_message_stream_error_preserves_data(
        self, make_anima, data_dir,
    ):
        """When streaming fails mid-way, user input + partial response
        with error marker are persisted."""
        anima_dir = make_anima(name="dataloss-stream-err")
        shared_dir = data_dir / "shared"
        dp = _make_anima_with_mocks(anima_dir, shared_dir)

        # Streaming yields some text_delta chunks, then raises
        async def _failing_stream(prompt, trigger="", **kwargs):
            yield {"type": "text_delta", "text": "Partial "}
            yield {"type": "text_delta", "text": "response"}
            raise RuntimeError("Stream interrupted")

        dp.agent.run_cycle_streaming = _failing_stream

        # Consume the generator — it should yield an error event
        chunks = []
        async for chunk in dp.process_message_stream("stream test"):
            chunks.append(chunk)

        # The last yielded chunk should be the error event
        assert any(c.get("type") == "error" for c in chunks)

        turns = _read_conversation(anima_dir)

        # Human turn must be persisted (pre-saved)
        human_turns = [t for t in turns if t["role"] == "human"]
        assert len(human_turns) == 1
        assert human_turns[0]["content"] == "stream test"

        # Assistant turn should contain partial text + error marker
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert len(assistant_turns) == 1
        assert "Partial response" in assistant_turns[0]["content"]
        assert "応答が中断されました" in assistant_turns[0]["content"]

    async def test_process_greet_error_preserves_data(
        self, make_anima, data_dir,
    ):
        """When process_greet fails, an error marker is saved to
        conversation.json."""
        anima_dir = make_anima(name="dataloss-greet-err")
        shared_dir = data_dir / "shared"
        dp = _make_anima_with_mocks(anima_dir, shared_dir)

        # Agent execution fails during greet
        dp.agent.run_cycle = AsyncMock(
            side_effect=RuntimeError("Greet failed"),
        )

        with pytest.raises(RuntimeError, match="Greet failed"):
            await dp.process_greet()

        turns = _read_conversation(anima_dir)

        # Error marker turn must be present
        assert len(turns) >= 1
        error_turns = [
            t for t in turns
            if t["role"] == "assistant" and "ERROR" in t["content"]
        ]
        assert len(error_turns) == 1
        assert "エラー" in error_turns[0]["content"]

    async def test_transcript_not_written_by_append_turn(
        self, make_anima, data_dir,
    ):
        """append_turn() no longer writes to transcript (replaced by
        unified activity log).  Verify that no transcript JSONL is
        created when processing a message, while conversation.json
        still records the turns."""
        anima_dir = make_anima(name="dataloss-transcript")
        shared_dir = data_dir / "shared"
        dp = _make_anima_with_mocks(anima_dir, shared_dir)

        # Agent execution fails
        dp.agent.run_cycle = AsyncMock(
            side_effect=RuntimeError("Transcript test error"),
        )

        with pytest.raises(RuntimeError, match="Transcript test error"):
            await dp.process_message("transcript input")

        # Transcript JSONL should NOT be written (replaced by activity log)
        entries = _read_transcript(anima_dir)
        assert len(entries) == 0

        # But conversation.json must still have the turns
        turns = _read_conversation(anima_dir)
        human_turns = [t for t in turns if t["role"] == "human"]
        assert len(human_turns) >= 1
        assert human_turns[0]["content"] == "transcript input"
