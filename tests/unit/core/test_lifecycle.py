"""Unit tests for lifecycle schedule parsing re-exports."""

from __future__ import annotations

from core.lifecycle import _parse_cron_md, _parse_schedule


class TestParseCronMd:
    def test_empty_content(self) -> None:
        assert _parse_cron_md("") == []

    def test_single_task(self) -> None:
        content = """\
## Daily Report
schedule: 0 9 * * *
Prepare the morning report.
"""
        tasks = _parse_cron_md(content)

        assert len(tasks) == 1
        assert tasks[0].name == "Daily Report"
        assert tasks[0].schedule == "0 9 * * *"
        assert tasks[0].type == "llm"
        assert "morning report" in tasks[0].description

    def test_command_task(self) -> None:
        content = """\
## Cleanup
schedule: 0 3 * * *
type: command
command: animaworks memory status
"""
        tasks = _parse_cron_md(content)

        assert len(tasks) == 1
        assert tasks[0].type == "command"
        assert tasks[0].command == "animaworks memory status"


class TestParseSchedule:
    def test_parse_cron_schedule(self) -> None:
        trigger = _parse_schedule("0 9 * * *")

        assert trigger is not None

    def test_parse_empty_schedule_returns_none(self) -> None:
        assert _parse_schedule("") is None
