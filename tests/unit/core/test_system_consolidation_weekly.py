from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.lifecycle.system_consolidation import run_weekly_full_contradiction_scan


@pytest.mark.asyncio
async def test_weekly_contradiction_scan_logs_dict_resolution_stats(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector = MagicMock()
    detector.scan_contradictions = AsyncMock(return_value=[{"left": "a", "right": "b"}])
    detector.resolve_contradictions = AsyncMock(
        return_value={"superseded": 2, "merged": 0, "coexisted": 0, "failed": 0}
    )
    detector.last_scan_stats = {"candidate_pairs": 1}

    with (
        patch("core.memory.activity.ActivityLogger", return_value=MagicMock()),
        patch("core.memory.manager.MemoryManager", return_value=MagicMock()),
        patch("core.memory.contradiction.ContradictionDetector", return_value=detector),
        caplog.at_level(logging.INFO, logger="animaworks.lifecycle"),
    ):
        await run_weekly_full_contradiction_scan(
            tmp_path / "animas" / "sakura",
            "sakura",
            SimpleNamespace(weekly_full_contradiction_max_pairs=3),
            model="test-model",
        )

    detector.resolve_contradictions.assert_awaited_once()
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Weekly full contradiction scan for sakura: detected=1" in message
        and "resolved={'superseded': 2" in message
        for message in messages
    )
