from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.slim_runtime_report import summarize


def test_report_preserves_unknown_cost_and_avoids_business_content(tmp_path: Path) -> None:
    anima = tmp_path / "animas" / "private-customer-name"
    anima.mkdir(parents=True)
    (anima / "status.json").write_text("{}")
    for family in ("activity_log", "prompt_logs", "token_usage"):
        (anima / family).mkdir()
    (anima / "activity_log/2026-09-01.jsonl").write_text(
        json.dumps({"type": "task_exec_end", "meta": {"task_id": "one", "status": "undeclared"}})
        + "\ninvalid\n"
        + json.dumps({"type": "task_exec_end", "meta": {"task_id": "one", "status": "completed"}})
        + "\n"
    )
    (anima / "token_usage/2026-09-01.jsonl").write_text(
        json.dumps({"trigger": "cron:private business title", "estimated_cost_usd": 0, "input_tokens": 10}) + "\n"
    )
    result = summarize(tmp_path, date(2026, 9, 1), date(2026, 9, 1))
    assert result["invalid_lines"] == {"activity_log": 1}
    assert result["distinct_available_correlation_ids"] == {"task_id": 1}
    assert result["task_end_events_by_status"] == {"undeclared": 1, "completed": 1}
    group = next(iter(result["groups"].values()))
    assert group["unknown_or_zero_estimated_cost_records"] == 1
    assert "private" not in json.dumps(result)
