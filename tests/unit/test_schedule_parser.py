"""Unit tests for core.schedule_parser module."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from apscheduler.triggers.cron import CronTrigger

import pytest

from core.schedule_parser import (
    parse_heartbeat_config,
    parse_cron_md,
    parse_schedule,
    _posix_dow_to_apsched,
)


class TestParseHeartbeatConfig:
    """Tests for parse_heartbeat_config()."""

    def test_basic_active_hours(self):
        content = """# Heartbeat
## 活動時間
8:00 - 23:00（JST）
"""
        start, end = parse_heartbeat_config(content)
        assert start == 8
        assert end == 23

    def test_standard_active_hours(self):
        content = "活動時間\n9:00 - 22:00"
        start, end = parse_heartbeat_config(content)
        assert start == 9
        assert end == 22

    def test_defaults_when_no_match(self):
        content = "some random content without patterns"
        start, end = parse_heartbeat_config(content)
        assert start is None
        assert end is None

    def test_empty_content(self):
        start, end = parse_heartbeat_config("")
        assert start is None
        assert end is None


class TestParseCronMd:
    """Tests for parse_cron_md() with new cron expression format."""

    def test_single_llm_task(self):
        content = """\
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].name == "毎朝の業務計画"
        assert tasks[0].schedule == "0 9 * * *"
        assert tasks[0].type == "llm"
        assert "長期記憶" in tasks[0].description

    def test_multiple_tasks(self):
        content = """\
## Task A
schedule: 0 8 * * *
type: llm
Description A

## Task B
schedule: 0 17 * * 5
type: llm
Description B
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 2
        assert tasks[0].name == "Task A"
        assert tasks[1].name == "Task B"

    def test_command_type(self):
        content = """\
## Backup
schedule: 0 2 * * *
type: command
command: /usr/bin/backup.sh
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].type == "command"
        assert tasks[0].command == "/usr/bin/backup.sh"

    def test_default_type_is_llm(self):
        content = """\
## Task
schedule: 0 9 * * *
Description without type
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].type == "llm"

    def test_empty_content(self):
        tasks = parse_cron_md("")
        assert tasks == []

    def test_tool_type_with_args(self):
        content = """\
## Slack通知
schedule: 0 9 * * 1-5
type: command
tool: slack_post
args:
  channel: general
  message: おはようございます
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].tool == "slack_post"
        assert tasks[0].args == {"channel": "general", "message": "おはようございます"}


class TestParseSchedule:
    """Tests for parse_schedule() with standard cron expressions."""

    def test_daily(self):
        trigger = parse_schedule("0 9 * * *")
        assert trigger is not None
        assert isinstance(trigger, CronTrigger)

    def test_weekday(self):
        trigger = parse_schedule("0 9 * * 1-5")
        assert trigger is not None

    def test_weekly(self):
        trigger = parse_schedule("0 17 * * 5")
        assert trigger is not None

    def test_every_5_minutes(self):
        trigger = parse_schedule("*/5 * * * *")
        assert trigger is not None

    def test_monthly_first_day(self):
        trigger = parse_schedule("0 9 1 * *")
        assert trigger is not None

    def test_complex_cron(self):
        trigger = parse_schedule("0,30 9-17 * * 1-5")
        assert trigger is not None

    def test_invalid_returns_none(self):
        trigger = parse_schedule("invalid schedule string")
        assert trigger is None

    def test_empty_returns_none(self):
        trigger = parse_schedule("")
        assert trigger is None

    def test_japanese_schedule_no_longer_supported(self):
        """Old Japanese format is not supported by the new parser."""
        trigger = parse_schedule("毎日 9:00 JST")
        assert trigger is None

    def test_sunday_start_range_does_not_raise(self):
        """parse_schedule('30 17 * * 0-4') must not raise ValueError.

        Regression test for the wrap-around bug: POSIX '0-4' (Sun-Thu) was
        naively converted to APScheduler '6-3' which is an invalid range.
        """
        trigger = parse_schedule("30 17 * * 0-4")
        assert trigger is not None
        assert isinstance(trigger, CronTrigger)


class TestPosixDowToApsched:
    """Unit tests for _posix_dow_to_apsched() conversion logic."""

    def test_wildcard_passthrough(self):
        assert _posix_dow_to_apsched("*") == "*"

    def test_single_value(self):
        """POSIX 4 (Thu) → ISO 3."""
        assert _posix_dow_to_apsched("4") == "3"

    def test_single_value_monday(self):
        """POSIX 1 (Mon) → ISO 0."""
        assert _posix_dow_to_apsched("1") == "0"

    def test_single_value_sunday_posix0(self):
        """POSIX 0 (Sun) → ISO 6."""
        assert _posix_dow_to_apsched("0") == "6"

    def test_single_value_sunday_posix7(self):
        """POSIX 7 (Sun alias) → ISO 6."""
        assert _posix_dow_to_apsched("7") == "6"

    def test_comma_separated(self):
        """POSIX '1,4' (Mon,Thu) → ISO '0,3'."""
        assert _posix_dow_to_apsched("1,4") == "0,3"

    def test_normal_range_no_wrap(self):
        """POSIX '1-5' (Mon–Fri) → ISO '0-4' (no wrap-around)."""
        assert _posix_dow_to_apsched("1-5") == "0-4"

    def test_sunday_start_range(self):
        """POSIX '0-4' (Sun–Thu) → ISO '6,0-3' (wrap-around fix).

        This is the primary regression test for the bug introduced in commit
        020367d9: previously '0-4' produced '6-3' which raises ValueError in
        APScheduler because the minimum of a range must not exceed the maximum.
        """
        assert _posix_dow_to_apsched("0-4") == "6,0-3"

    def test_full_week_range(self):
        """POSIX '0-6' (all days) → APScheduler representation covering all days."""
        result = _posix_dow_to_apsched("0-6")
        # '6,0-5' covers all 7 ISO days; verify APScheduler accepts it
        trigger = parse_schedule(f"0 9 * * {result}")
        assert trigger is not None

    def test_range_ending_at_sunday(self):
        """POSIX '5-7' (Fri–Sun) → ISO '4-6' (consecutive, no wrap)."""
        assert _posix_dow_to_apsched("5-7") == "4-6"

    def test_step_normal_range(self):
        """POSIX '1-5/2' (Mon,Wed,Fri) → ISO values without wrap."""
        result = _posix_dow_to_apsched("1-5/2")
        # POSIX [1,3,5] → ISO [0,2,4]; no consecutive pairs → individual values
        assert result == "0,2,4"

    @pytest.mark.parametrize("expr", [
        "30 17 * * 0-4",   # sec's broken schedule — the main bug
        "0 9 * * 0",       # Sunday only
        "0 9 * * 0-6",     # All days via full POSIX range
        "0 9 * * 5-7",     # Fri–Sun via POSIX 5-7
    ])
    def test_parse_schedule_accepts_sunday_ranges(self, expr: str):
        """parse_schedule must return a valid CronTrigger for Sunday-containing ranges."""
        trigger = parse_schedule(expr)
        assert trigger is not None, f"parse_schedule({expr!r}) returned None"
