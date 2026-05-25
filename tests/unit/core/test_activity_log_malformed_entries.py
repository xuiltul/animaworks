from __future__ import annotations

import json
from pathlib import Path

from core.memory.activity import ActivityLogger
from core.time_utils import now_iso, today_local


def _write_jsonl(path: Path, entries: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_recent_page_skips_non_object_json_entries(tmp_path: Path) -> None:
    """Historical activity logs may contain JSON arrays; readers must not crash."""

    anima_dir = tmp_path / "animas" / "hina"
    log_path = anima_dir / "activity_log" / f"{today_local().isoformat()}.jsonl"
    _write_jsonl(
        log_path,
        [
            {"ts": now_iso(), "type": "cron", "summary": "before malformed entry"},
            [
                {
                    "timestamp": "2026-05-21T18:21:01Z",
                    "task": "異常値検知（毎時15分）",
                    "result": "normal",
                }
            ],
            {"ts": now_iso(), "type": "heartbeat_start", "summary": "after malformed entry"},
        ],
    )

    page = ActivityLogger(anima_dir).recent_page(days=1, limit=10)

    assert page.total == 2
    assert [entry.summary for entry in page.entries] == [
        "after malformed entry",
        "before malformed entry",
    ]
