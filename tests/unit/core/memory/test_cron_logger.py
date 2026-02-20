"""Unit tests for core/memory/cron_logger.py — CronLogger."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from core.memory.cron_logger import CronLogger


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "anima"
    d.mkdir()
    return d


@pytest.fixture
def cl(anima_dir: Path) -> CronLogger:
    return CronLogger(anima_dir)


# ── append_cron_log ──────────────────────────────────────


class TestAppendCronLog:
    def test_creates_jsonl_file(self, cl: CronLogger, anima_dir: Path) -> None:
        cl.append_cron_log("daily-report", summary="Generated report", duration_ms=123)
        log_dir = anima_dir / "state" / "cron_logs"
        path = log_dir / f"{date.today().isoformat()}.jsonl"
        assert path.exists()

    def test_writes_correct_json_fields(self, cl: CronLogger, anima_dir: Path) -> None:
        cl.append_cron_log("daily-report", summary="Generated report", duration_ms=456)
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["task"] == "daily-report"
        assert entry["summary"] == "Generated report"
        assert entry["duration_ms"] == 456
        assert "timestamp" in entry

    def test_appends_multiple_entries(self, cl: CronLogger, anima_dir: Path) -> None:
        cl.append_cron_log("task-a", summary="First", duration_ms=100)
        cl.append_cron_log("task-b", summary="Second", duration_ms=200)
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["task"] == "task-a"
        assert json.loads(lines[1])["task"] == "task-b"

    def test_truncates_summary_at_500_chars(self, cl: CronLogger, anima_dir: Path) -> None:
        long_summary = "x" * 1000
        cl.append_cron_log("task", summary=long_summary, duration_ms=0)
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert len(entry["summary"]) == 500

    def test_bounding_at_max_lines(self, cl: CronLogger, anima_dir: Path) -> None:
        """Writing more than MAX_LINES entries keeps only the last 50."""
        for i in range(55):
            cl.append_cron_log(f"task-{i}", summary=f"entry {i}", duration_ms=i)
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 50
        # The first 5 entries should be trimmed; first remaining is task-5
        first_entry = json.loads(lines[0])
        assert first_entry["task"] == "task-5"

    def test_creates_parent_directories(self, cl: CronLogger, anima_dir: Path) -> None:
        log_dir = anima_dir / "state" / "cron_logs"
        assert not log_dir.exists()
        cl.append_cron_log("task", summary="ok", duration_ms=0)
        assert log_dir.is_dir()


# ── append_cron_command_log ──────────────────────────────


class TestAppendCronCommandLog:
    def test_writes_correct_json_fields(self, cl: CronLogger, anima_dir: Path) -> None:
        cl.append_cron_command_log(
            "git-pull",
            exit_code=0,
            stdout="Already up to date.\n",
            stderr="",
            duration_ms=350,
        )
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["task"] == "git-pull"
        assert entry["exit_code"] == 0
        assert entry["stdout_lines"] == 1
        assert entry["stderr_lines"] == 0
        assert entry["stdout_preview"] == "Already up to date."
        assert entry["stderr_preview"] == ""
        assert entry["duration_ms"] == 350
        assert "timestamp" in entry

    def test_preview_short_output(self, cl: CronLogger, anima_dir: Path) -> None:
        """Output with <=10 lines is included in full."""
        stdout = "\n".join(f"line {i}" for i in range(8))
        cl.append_cron_command_log(
            "task", exit_code=0, stdout=stdout, stderr="", duration_ms=0,
        )
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["stdout_lines"] == 8
        assert "..." not in entry["stdout_preview"]

    def test_preview_long_output(self, cl: CronLogger, anima_dir: Path) -> None:
        """Output with >10 lines shows first 5 + '...' + last 5."""
        stdout = "\n".join(f"line {i}" for i in range(20))
        cl.append_cron_command_log(
            "task", exit_code=0, stdout=stdout, stderr="", duration_ms=0,
        )
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["stdout_lines"] == 20
        preview_lines = entry["stdout_preview"].splitlines()
        assert preview_lines[5] == "..."
        assert preview_lines[0] == "line 0"
        assert preview_lines[-1] == "line 19"
        assert len(preview_lines) == 11  # 5 + 1 ("...") + 5

    def test_preview_truncated_at_1000_chars(self, cl: CronLogger, anima_dir: Path) -> None:
        """Preview is capped at 1000 characters."""
        # Create long lines so preview exceeds 1000 chars
        stdout = "\n".join("A" * 200 for _ in range(20))
        cl.append_cron_command_log(
            "task", exit_code=0, stdout=stdout, stderr="", duration_ms=0,
        )
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert len(entry["stdout_preview"]) <= 1000

    def test_empty_stdout_stderr(self, cl: CronLogger, anima_dir: Path) -> None:
        cl.append_cron_command_log(
            "task", exit_code=0, stdout="", stderr="", duration_ms=0,
        )
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["stdout_lines"] == 0
        assert entry["stderr_lines"] == 0
        assert entry["stdout_preview"] == ""
        assert entry["stderr_preview"] == ""

    def test_nonzero_exit_code(self, cl: CronLogger, anima_dir: Path) -> None:
        cl.append_cron_command_log(
            "failing-task",
            exit_code=1,
            stdout="",
            stderr="Error: something went wrong",
            duration_ms=50,
        )
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["exit_code"] == 1
        assert entry["stderr_lines"] == 1
        assert "Error" in entry["stderr_preview"]

    def test_bounding_at_max_lines(self, cl: CronLogger, anima_dir: Path) -> None:
        """Writing more than MAX_LINES command entries keeps only the last 50."""
        for i in range(55):
            cl.append_cron_command_log(
                f"task-{i}", exit_code=0, stdout=f"out-{i}", stderr="", duration_ms=i,
            )
        path = anima_dir / "state" / "cron_logs" / f"{date.today().isoformat()}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 50


# ── read_cron_log ────────────────────────────────────────


class TestReadCronLog:
    def test_reads_todays_entries(self, cl: CronLogger) -> None:
        cl.append_cron_log("task-a", summary="did thing A", duration_ms=100)
        cl.append_cron_log("task-b", summary="did thing B", duration_ms=200)
        result = cl.read_cron_log(days=1)
        assert "task-a" in result
        assert "task-b" in result
        assert "did thing A" in result
        assert "100ms" in result

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        cl = CronLogger(tmp_path / "nonexistent")
        result = cl.read_cron_log(days=1)
        assert result == ""

    def test_respects_days_parameter(self, cl: CronLogger, anima_dir: Path) -> None:
        """Only log files within the requested day range are read."""
        log_dir = anima_dir / "state" / "cron_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        today = date.today()
        # Write today's log
        today_entry = json.dumps({
            "timestamp": "2026-01-20T10:00:00+09:00",
            "task": "today-task",
            "summary": "today summary",
            "duration_ms": 100,
        })
        (log_dir / f"{today.isoformat()}.jsonl").write_text(
            today_entry + "\n", encoding="utf-8",
        )

        # Write yesterday's log
        yesterday = today - timedelta(days=1)
        yesterday_entry = json.dumps({
            "timestamp": "2026-01-19T10:00:00+09:00",
            "task": "yesterday-task",
            "summary": "yesterday summary",
            "duration_ms": 200,
        })
        (log_dir / f"{yesterday.isoformat()}.jsonl").write_text(
            yesterday_entry + "\n", encoding="utf-8",
        )

        # Write 3-days-ago log
        three_days = today - timedelta(days=3)
        old_entry = json.dumps({
            "timestamp": "2026-01-17T10:00:00+09:00",
            "task": "old-task",
            "summary": "old summary",
            "duration_ms": 300,
        })
        (log_dir / f"{three_days.isoformat()}.jsonl").write_text(
            old_entry + "\n", encoding="utf-8",
        )

        # days=1 should only get today
        result_1 = cl.read_cron_log(days=1)
        assert "today-task" in result_1
        assert "yesterday-task" not in result_1

        # days=2 should get today + yesterday
        result_2 = cl.read_cron_log(days=2)
        assert "today-task" in result_2
        assert "yesterday-task" in result_2
        assert "old-task" not in result_2

    def test_skips_malformed_json(self, cl: CronLogger, anima_dir: Path) -> None:
        """Malformed JSON lines are silently skipped."""
        log_dir = anima_dir / "state" / "cron_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{date.today().isoformat()}.jsonl"

        valid_entry = json.dumps({
            "timestamp": "2026-01-20T10:00:00+09:00",
            "task": "good-task",
            "summary": "good",
            "duration_ms": 100,
        })
        path.write_text(
            "not valid json\n" + valid_entry + "\n",
            encoding="utf-8",
        )
        result = cl.read_cron_log(days=1)
        assert "good-task" in result

    def test_skips_entries_missing_keys(self, cl: CronLogger, anima_dir: Path) -> None:
        """Entries missing required keys (e.g. 'task') are silently skipped."""
        log_dir = anima_dir / "state" / "cron_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{date.today().isoformat()}.jsonl"

        incomplete = json.dumps({"timestamp": "2026-01-20T10:00:00+09:00"})
        valid = json.dumps({
            "timestamp": "2026-01-20T11:00:00+09:00",
            "task": "valid-task",
            "summary": "ok",
            "duration_ms": 50,
        })
        path.write_text(incomplete + "\n" + valid + "\n", encoding="utf-8")
        result = cl.read_cron_log(days=1)
        assert "valid-task" in result

    def test_empty_log_file(self, cl: CronLogger, anima_dir: Path) -> None:
        """An empty log file returns empty string."""
        log_dir = anima_dir / "state" / "cron_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{date.today().isoformat()}.jsonl").write_text(
            "", encoding="utf-8",
        )
        result = cl.read_cron_log(days=1)
        assert result == ""

    def test_command_log_entries_skipped_by_read(
        self, cl: CronLogger, anima_dir: Path,
    ) -> None:
        """read_cron_log expects 'summary' key; command entries lack it and are skipped."""
        cl.append_cron_command_log(
            "cmd-task", exit_code=0, stdout="output", stderr="", duration_ms=10,
        )
        result = cl.read_cron_log(days=1)
        # Command log entries lack the 'summary' key, so they cause KeyError
        # and are skipped by the except clause
        assert "cmd-task" not in result
