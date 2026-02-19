# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the character greeting feature.

Tests the full greet flow: DigitalAnima.process_greet() with mocked LLM,
verifying response format, caching, emotion extraction, and conversation
memory recording.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from core.schemas import CycleResult


def _make_cycle_result(**kwargs) -> CycleResult:
    defaults = dict(trigger="greet:user", action="responded", summary="done", duration_ms=50)
    defaults.update(kwargs)
    return CycleResult(**defaults)


@pytest.mark.asyncio
class TestGreetE2E:
    """End-to-end tests for DigitalAnima.process_greet()."""

    async def test_greet_full_flow(self, make_digital_anima):
        """Test complete greeting flow: prompt → LLM → response with emotion."""
        dp = make_digital_anima("greeter")

        with patch.object(
            dp.agent, "run_cycle",
            new=AsyncMock(return_value=_make_cycle_result(
                summary='やあ！今はのんびりしてるよ。<!-- emotion: {"emotion": "smile"} -->'
            )),
        ):
            result = await dp.process_greet()

        assert result["response"] == "やあ！今はのんびりしてるよ。"
        assert result["emotion"] == "smile"
        assert result["cached"] is False
        assert dp._last_greet_text == "やあ！今はのんびりしてるよ。"

    async def test_greet_caching_within_cooldown(self, make_digital_anima):
        """Test that repeated greet calls within 5 minutes return cached response."""
        dp = make_digital_anima("greeter")

        with patch.object(
            dp.agent, "run_cycle",
            new=AsyncMock(return_value=_make_cycle_result(summary="Hello!")),
        ) as mock_cycle:
            # First call — fresh
            result1 = await dp.process_greet()
            assert result1["cached"] is False

            # Second call — should be cached
            result2 = await dp.process_greet()
            assert result2["cached"] is True
            assert result2["response"] == result1["response"]

            # LLM should have been called exactly once
            assert mock_cycle.await_count == 1

    async def test_greet_cache_expires_after_cooldown(self, make_digital_anima):
        """Test that cache expires after the cooldown period."""
        dp = make_digital_anima("greeter")

        with patch.object(
            dp.agent, "run_cycle",
            new=AsyncMock(return_value=_make_cycle_result(summary="Hello again!")),
        ) as mock_cycle:
            await dp.process_greet()

            # Simulate cache expiry (set last greet to beyond the 1-hour cooldown)
            dp._last_greet_at = time.time() - 3601

            result = await dp.process_greet()
            assert result["cached"] is False
            assert mock_cycle.await_count == 2

    async def test_greet_records_visit_and_assistant_turns(self, make_digital_anima):
        """Test that visit marker (system) and assistant turn are both recorded."""
        dp = make_digital_anima("greeter")

        with patch.object(
            dp.agent, "run_cycle",
            new=AsyncMock(return_value=_make_cycle_result(summary="Hi there!")),
        ):
            await dp.process_greet()

        # Check conversation state file
        conv_path = dp.anima_dir / "state" / "conversation.json"
        assert conv_path.exists()

        import json
        conv_data = json.loads(conv_path.read_text(encoding="utf-8"))
        turns = conv_data.get("turns", [])
        assert len(turns) == 2
        assert turns[0]["role"] == "system"
        assert turns[0]["content"] == "[デスクを訪問]"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hi there!"

    async def test_greet_preserves_status(self, make_digital_anima):
        """Test that greeting restores the previous status after completion."""
        dp = make_digital_anima("greeter")
        dp._status = "working"
        dp._current_task = "Building report"

        with patch.object(
            dp.agent, "run_cycle",
            new=AsyncMock(return_value=_make_cycle_result(summary="Busy!")),
        ):
            result = await dp.process_greet()

        assert dp._status == "working"
        assert dp._current_task == "Building report"

    async def test_greet_with_busy_status_in_prompt(self, make_digital_anima):
        """Test that current status is injected into the greet prompt."""
        dp = make_digital_anima("greeter")
        dp._status = "checking"
        dp._current_task = "Morning heartbeat"

        prompt_received = []

        async def capture_prompt(prompt, trigger="manual"):
            prompt_received.append(prompt)
            return _make_cycle_result(summary="I'm busy!")

        with patch.object(dp.agent, "run_cycle", new=capture_prompt):
            await dp.process_greet()

        assert len(prompt_received) == 1
        assert "checking" in prompt_received[0]
        assert "Morning heartbeat" in prompt_received[0]

    async def test_greet_emotion_invalid_falls_back(self, make_digital_anima):
        """Test that invalid emotion values fall back to neutral."""
        dp = make_digital_anima("greeter")

        with patch.object(
            dp.agent, "run_cycle",
            new=AsyncMock(return_value=_make_cycle_result(
                summary='Hi! <!-- emotion: {"emotion": "angry"} -->'
            )),
        ):
            result = await dp.process_greet()

        assert result["emotion"] == "neutral"
        assert result["response"] == "Hi!"

    async def test_greet_no_emotion_tag(self, make_digital_anima):
        """Test response without emotion tag defaults to neutral."""
        dp = make_digital_anima("greeter")

        with patch.object(
            dp.agent, "run_cycle",
            new=AsyncMock(return_value=_make_cycle_result(
                summary="Plain greeting without emotion"
            )),
        ):
            result = await dp.process_greet()

        assert result["emotion"] == "neutral"
        assert result["response"] == "Plain greeting without emotion"

    async def test_greet_prompt_includes_guidance(self, make_digital_anima):
        """Test that the greet prompt includes expanded guidance."""
        dp = make_digital_anima("greeter")

        prompt_received = []

        async def capture_prompt(prompt, trigger="manual"):
            prompt_received.append(prompt)
            return _make_cycle_result(summary="OK")

        with patch.object(dp.agent, "run_cycle", new=capture_prompt):
            await dp.process_greet()

        assert len(prompt_received) == 1
        prompt = prompt_received[0]
        assert "困っていること" in prompt
        assert "3〜4文" in prompt

    async def test_greet_visit_marker_recorded_before_llm_call(self, make_digital_anima):
        """Test that visit marker is written before the LLM call."""
        dp = make_digital_anima("greeter")

        import json
        conv_path = dp.anima_dir / "state" / "conversation.json"
        turns_at_llm_call = []

        async def capture_state(prompt, trigger="manual"):
            # Read conv state at the moment LLM is called
            if conv_path.exists():
                data = json.loads(conv_path.read_text(encoding="utf-8"))
                turns_at_llm_call.extend(data.get("turns", []))
            return _make_cycle_result(summary="Hello!")

        with patch.object(dp.agent, "run_cycle", new=capture_state):
            await dp.process_greet()

        # Visit marker should exist before LLM call
        assert len(turns_at_llm_call) == 1
        assert turns_at_llm_call[0]["role"] == "system"
        assert turns_at_llm_call[0]["content"] == "[デスクを訪問]"
