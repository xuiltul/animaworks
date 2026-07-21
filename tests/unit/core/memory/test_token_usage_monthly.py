from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from core.memory.token_usage import TokenUsageLogger


def _write_usage(anima_dir: Path, day: str, *totals: int) -> None:
    usage_dir = anima_dir / "token_usage"
    usage_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "ts": f"{day}T12:00:00+00:00",
            "input_tokens": total,
            "output_tokens": 0,
            "total_tokens": total,
        }
        for total in totals
    ]
    (usage_dir / f"{day}.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    path = tmp_path / "animas" / "alice"
    path.mkdir(parents=True)
    return path


class TestMonthlyTotal:
    def test_rollup_matches_full_jsonl_scan(self, anima_dir: Path) -> None:
        _write_usage(anima_dir, "2026-06-30", 1_000)
        _write_usage(anima_dir, "2026-07-01", 100)
        _write_usage(anima_dir, "2026-07-21", 200)
        _write_usage(anima_dir, "2026-07-22", 300)
        logger = TokenUsageLogger(anima_dir)

        actual = logger.monthly_total(datetime(2026, 7, 22, 15, tzinfo=UTC))
        entries = logger.read_entries(days=22, target_date=date(2026, 7, 22))
        expected = sum(entry.get("total_tokens", 0) for entry in entries)

        assert actual == expected == 600
        assert json.loads((anima_dir / "token_usage" / "rollup.json").read_text()) == {
            "2026-07-01": 100,
            "2026-07-21": 200,
        }

    def test_only_today_is_reaggregated_after_rollup_exists(
        self, anima_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_usage(anima_dir, "2026-07-21", 100)
        _write_usage(anima_dir, "2026-07-22", 200)
        logger = TokenUsageLogger(anima_dir)
        now = datetime(2026, 7, 22, 15, tzinfo=UTC)
        assert logger.monthly_total(now) == 300

        _write_usage(anima_dir, "2026-07-22", 200, 50)
        read_days: list[str] = []
        original = TokenUsageLogger._read_daily_total

        def track_read(path: Path) -> int:
            read_days.append(path.stem)
            return original(path)

        monkeypatch.setattr(TokenUsageLogger, "_read_daily_total", staticmethod(track_read))

        assert logger.monthly_total(now) == 350
        assert read_days == ["2026-07-22"]

    @pytest.mark.parametrize("broken_rollup", ["not json", "[]", '{"2026-07-01": "bad"}'])
    def test_broken_rollup_is_rebuilt(self, anima_dir: Path, broken_rollup: str) -> None:
        _write_usage(anima_dir, "2026-07-01", 125)
        rollup_path = anima_dir / "token_usage" / "rollup.json"
        rollup_path.write_text(broken_rollup, encoding="utf-8")

        total = TokenUsageLogger(anima_dir).monthly_total(datetime(2026, 7, 22, tzinfo=UTC))

        assert total == 125
        assert json.loads(rollup_path.read_text()) == {"2026-07-01": 125}

    def test_missing_rollup_day_is_reconstructed(self, anima_dir: Path) -> None:
        _write_usage(anima_dir, "2026-07-01", 100)
        _write_usage(anima_dir, "2026-07-02", 200)
        rollup_path = anima_dir / "token_usage" / "rollup.json"
        rollup_path.write_text(json.dumps({"2026-07-01": 100}), encoding="utf-8")

        total = TokenUsageLogger(anima_dir).monthly_total(datetime(2026, 7, 22, tzinfo=UTC))

        assert total == 300
        assert json.loads(rollup_path.read_text()) == {"2026-07-01": 100, "2026-07-02": 200}

    def test_rollup_and_total_only_include_current_month(self, anima_dir: Path) -> None:
        _write_usage(anima_dir, "2026-06-30", 1_000)
        _write_usage(anima_dir, "2026-07-01", 100)
        _write_usage(anima_dir, "2026-07-22", 200)
        _write_usage(anima_dir, "2026-07-23", 400)
        rollup_path = anima_dir / "token_usage" / "rollup.json"
        rollup_path.write_text(
            json.dumps({"2026-06-30": 1_000, "2026-07-01": 100, "2026-07-22": 9_999}),
            encoding="utf-8",
        )

        total = TokenUsageLogger(anima_dir).monthly_total(datetime(2026, 7, 22, tzinfo=UTC))

        assert total == 300
        assert json.loads(rollup_path.read_text()) == {"2026-07-01": 100}
