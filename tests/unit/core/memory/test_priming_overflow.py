"""Unit tests for Channel C full-search priming behavior."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.memory.priming import PrimingEngine


@pytest.fixture
def priming_engine(tmp_path: Path) -> PrimingEngine:
    """Create a PrimingEngine with isolated temp directories."""
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    (anima_dir / "knowledge").mkdir()
    (anima_dir / "skills").mkdir()
    (anima_dir / "episodes").mkdir()
    return PrimingEngine(anima_dir)


class TestChannelCFullSearchOnly:
    def test_channel_c_related_knowledge_has_no_restrict_parameter(self) -> None:
        params = inspect.signature(PrimingEngine._channel_c_related_knowledge).parameters
        assert "restrict_to" not in params

    @pytest.mark.asyncio
    async def test_prime_memories_passes_no_restriction_to_channel_c(
        self,
        priming_engine: PrimingEngine,
    ) -> None:
        with (
            patch.object(
                priming_engine,
                "_channel_c_related_knowledge",
                new_callable=AsyncMock,
                return_value=("", ""),
            ) as mock_channel_c,
            patch.object(priming_engine, "_channel_a_sender_profile", new_callable=AsyncMock, return_value=""),
            patch.object(priming_engine, "_channel_b_recent_activity", new_callable=AsyncMock, return_value=""),
            patch.object(priming_engine, "_channel_c0_important_knowledge", new_callable=AsyncMock, return_value=""),
            patch.object(priming_engine, "_channel_e_pending_tasks", new_callable=AsyncMock, return_value=""),
            patch.object(priming_engine, "_collect_recent_outbound", new_callable=AsyncMock, return_value=""),
            patch.object(priming_engine, "_channel_f_episodes", new_callable=AsyncMock, return_value=""),
            patch.object(
                priming_engine,
                "_collect_pending_human_notifications",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch.object(priming_engine, "_channel_g_graph_context", new_callable=AsyncMock, return_value=""),
        ):
            await priming_engine.prime_memories(
                message="test message",
                sender_name="human",
            )

        mock_channel_c.assert_called_once()
        assert "restrict_to" not in mock_channel_c.call_args.kwargs
