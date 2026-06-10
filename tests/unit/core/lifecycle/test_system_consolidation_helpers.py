from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo


def test_activity_log_only_anima_passes_daily_consolidation_gate(tmp_path: Path) -> None:
    from core.lifecycle.system_consolidation import evaluate_daily_consolidation_gate
    from core.memory.activity import ActivityLogger

    anima_dir = tmp_path / "animas" / "ritsu"
    anima_dir.mkdir(parents=True)
    with patch("core.memory.activity.now_iso", return_value="2026-06-10T12:00:00+09:00"):
        ActivityLogger(anima_dir).log(
            "response_sent",
            summary="worked from activity log only",
            content="activity exists before any episode has been generated",
        )

    with patch(
        "core.memory.consolidation.now_local",
        return_value=datetime(2026, 6, 11, 2, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
    ):
        gate = evaluate_daily_consolidation_gate(
            anima_dir,
            "ritsu",
            threshold=1,
            hours=24,
        )

    assert gate.should_run is True
    assert gate.activity_count == 1
    assert gate.episode_count == 0
