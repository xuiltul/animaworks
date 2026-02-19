# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for Mode A1 (Claude Agent SDK) execution.

Mode A1 uses claude_agent_sdk.query() as an async generator.
Mock tests patch sys.modules to inject a mock SDK module.
"""

from __future__ import annotations

import pytest

from tests.helpers.mocks import (
    MockResultMessage,
    patch_agent_sdk,
    patch_agent_sdk_streaming,
)


class TestModeA1Mock:
    """Mode A1 tests using mocked Claude Agent SDK."""

    async def test_basic_response(self, make_agent_core):
        """A1: basic prompt produces a text response."""
        with patch_agent_sdk(
            response_text="Hello from Claude Agent SDK.",
            usage={"input_tokens": 500, "output_tokens": 100},
        ):
            agent = make_agent_core(
                name="a1-basic",
                model="claude-sonnet-4-20250514",
            )
            # Force SDK available after patching sys.modules
            agent._sdk_available = True

            result = await agent.run_cycle("Say hello")

        assert result.action == "responded"
        assert "Hello from Claude Agent SDK." in result.summary

    async def test_streaming_events(self, make_agent_core):
        """A1 streaming: events are yielded in expected order."""
        with patch_agent_sdk_streaming(
            text_deltas=["Hello ", "world", "!"],
            usage={"input_tokens": 500, "output_tokens": 50},
        ):
            agent = make_agent_core(
                name="a1-stream",
                model="claude-sonnet-4-20250514",
            )
            agent._sdk_available = True

            events = []
            async for chunk in agent.run_cycle_streaming("Say hello"):
                events.append(chunk)

        # Should have text_delta events and a final cycle_done
        event_types = [e["type"] for e in events]
        assert "text_delta" in event_types
        assert "cycle_done" in event_types

        # The cycle_done event should contain the cycle result
        cycle_done = [e for e in events if e["type"] == "cycle_done"][0]
        assert "cycle_result" in cycle_done

    async def test_session_chaining(self, make_agent_core):
        """A1: session chaining triggers when context threshold is exceeded."""
        # First call: high usage to trigger threshold
        with patch_agent_sdk(
            response_text="First session response.",
            usage={"input_tokens": 150_000, "output_tokens": 10_000},
            num_turns=5,
        ):
            agent = make_agent_core(
                name="a1-chain",
                model="claude-sonnet-4-20250514",
                context_threshold=0.01,  # Very low to force chaining
                max_chains=1,
            )
            agent._sdk_available = True

            result = await agent.run_cycle("Do something complex")

        assert result.summary
        assert "First session response." in result.summary


class TestModeA1Live:
    """Mode A1 tests using real API calls."""

    @pytest.mark.live
    @pytest.mark.timeout(90)
    async def test_live_basic_response(self, make_agent_core):
        """Live A1: real Claude Agent SDK call."""
        agent = make_agent_core(
            name="a1-live",
            model="claude-sonnet-4-20250514",
        )

        if not agent._sdk_available:
            pytest.skip("Claude Agent SDK not installed")

        result = await agent.run_cycle(
            "Reply with exactly: ANIMAWORKS_A1_TEST_OK"
        )

        assert result.summary
        assert result.action == "responded"
