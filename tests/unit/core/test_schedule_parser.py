"""Unit tests for core.schedule_parser with standard cron expression format."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.schedule_parser import (
    parse_cron_md,
    parse_schedule,
    parse_heartbeat_config,
)
from core.schemas import CronTask


# ── parse_cron_md tests ───────────────────────────────────


class TestParseCronMd:
    """Tests for the new cron.md format with ``schedule:`` directives."""

    def test_basic_llm_task(self):
        """Basic LLM-type task with schedule directive."""
        content = """\
## Morning Standup
schedule: 0 9 * * *
type: llm
Check yesterday's progress and plan today.
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "Morning Standup"
        assert task.schedule == "0 9 * * *"
        assert task.type == "llm"
        assert "progress" in task.description

    def test_command_type_with_bash(self):
        """Command-type task with bash command."""
        content = """\
## DB Backup
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "DB Backup"
        assert task.schedule == "0 2 * * *"
        assert task.type == "command"
        assert task.command == "/usr/local/bin/backup.sh"

    def test_command_type_with_tool_and_args(self):
        """Command-type task with tool and YAML args."""
        content = """\
## Deploy
schedule: 0 2 * * 1-5
type: command
tool: run_deploy
args:
  env: staging
  dry_run: true
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "Deploy"
        assert task.schedule == "0 2 * * 1-5"
        assert task.type == "command"
        assert task.tool == "run_deploy"
        assert task.args == {"env": "staging", "dry_run": True}

    def test_multiple_tasks(self):
        """Multiple tasks in one cron.md."""
        content = """\
## Morning Report
schedule: 0 9 * * *
type: llm
Summarize overnight events.

## Evening Cleanup
schedule: 30 17 * * 1-5
type: command
command: /opt/cleanup.sh
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 2
        assert tasks[0].name == "Morning Report"
        assert tasks[0].schedule == "0 9 * * *"
        assert tasks[1].name == "Evening Cleanup"
        assert tasks[1].schedule == "30 17 * * 1-5"

    def test_every_5_minutes(self):
        """Task running every 5 minutes."""
        content = """\
## Health Check
schedule: */5 * * * *
type: command
tool: health_check
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].schedule == "*/5 * * * *"

    def test_no_schedule_line(self):
        """Task without schedule: line gets empty schedule."""
        content = """\
## Orphan Task
type: llm
Do something without a schedule.
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].schedule == ""
        assert tasks[0].name == "Orphan Task"

    def test_default_type_is_llm(self):
        """When type: is missing, defaults to llm."""
        content = """\
## Simple Task
schedule: 0 12 * * *
Just do something at noon.
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].type == "llm"

    def test_empty_content(self):
        """Empty content returns no tasks."""
        assert parse_cron_md("") == []
        assert parse_cron_md("   \n  \n") == []


# ── HTML comment tests ────────────────────────────────────


class TestHtmlCommentExclusion:
    """Tests for HTML comment stripping before cron.md parsing."""

    def test_html_comment_single_line_excluded(self):
        """A task fully wrapped in a single HTML comment block is excluded."""
        content = """\
<!-- ## Disabled Task
schedule: 0 9 * * *
type: llm
Do something disabled -->
"""
        tasks = parse_cron_md(content)
        assert tasks == []

    def test_html_comment_multiline_excluded(self):
        """Multiple tasks inside one HTML comment block are all excluded."""
        content = """\
<!--
## Task A
schedule: 0 8 * * *
type: llm
Description A

## Task B
schedule: 0 17 * * 5
type: llm
Description B
-->
"""
        tasks = parse_cron_md(content)
        assert tasks == []

    def test_html_comment_partial_exclusion(self):
        """Only commented-out tasks are excluded; tasks outside remain."""
        content = """\
## Active Task
schedule: 0 9 * * *
type: llm
I should be parsed.

<!-- ## Disabled Task
schedule: 0 10 * * *
type: llm
I should NOT be parsed. -->

## Another Active
schedule: 0 8 * * 1-5
type: llm
I should also be parsed.
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 2
        assert tasks[0].name == "Active Task"
        assert tasks[1].name == "Another Active"

    def test_no_comments_unchanged(self):
        """Content without HTML comments parses normally (regression)."""
        content = """\
## Daily Report
schedule: 0 18 * * *
type: llm
Summarize the day.
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].name == "Daily Report"
        assert tasks[0].schedule == "0 18 * * *"
        assert tasks[0].type == "llm"
        assert "Summarize" in tasks[0].description

    def test_nested_comment_markers(self):
        """Greedy-minimal match: <!-- ... <!-- ... --> stops at first -->."""
        content = """\
<!-- outer <!-- inner --> still visible
## Visible Task
schedule: 0 7 * * *
type: llm
Should be parsed.
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].name == "Visible Task"


# ── parse_schedule tests ──────────────────────────────────


class TestParseSchedule:
    """Tests for parse_schedule with standard cron expressions."""

    def test_daily_at_nine(self):
        """Standard daily 9am cron expression."""
        trigger = parse_schedule("0 9 * * *")
        assert trigger is not None

    def test_every_5_minutes(self):
        """Every 5 minutes cron expression."""
        trigger = parse_schedule("*/5 * * * *")
        assert trigger is not None

    def test_weekday_at_two_am(self):
        """Weekdays at 2am."""
        trigger = parse_schedule("0 2 * * 1-5")
        assert trigger is not None

    def test_friday_at_five_thirty(self):
        """Fridays at 5:30pm."""
        trigger = parse_schedule("30 17 * * 5")
        assert trigger is not None

    def test_first_of_month(self):
        """First of every month at midnight."""
        trigger = parse_schedule("0 0 1 * *")
        assert trigger is not None

    def test_complex_expression(self):
        """Complex cron with ranges and lists."""
        trigger = parse_schedule("0,30 9-17 * * 1-5")
        assert trigger is not None

    def test_every_hour(self):
        """Every hour at minute 0."""
        trigger = parse_schedule("0 * * * *")
        assert trigger is not None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert parse_schedule("") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string returns None."""
        assert parse_schedule("   ") is None

    def test_invalid_expression_returns_none(self, caplog):
        """Invalid expression returns None with warning."""
        with caplog.at_level(logging.WARNING):
            result = parse_schedule("not a cron expression")
        assert result is None
        assert "Invalid cron expression" in caplog.text

    def test_japanese_schedule_returns_none(self, caplog):
        """Old Japanese format is no longer supported."""
        with caplog.at_level(logging.WARNING):
            result = parse_schedule("毎日 9:00 JST")
        assert result is None
        assert "Invalid cron expression" in caplog.text

    def test_too_few_fields_returns_none(self, caplog):
        """Fewer than 5 fields returns None."""
        with caplog.at_level(logging.WARNING):
            result = parse_schedule("0 9 *")
        assert result is None

    def test_too_many_fields_returns_none(self, caplog):
        """More than 5 fields returns None."""
        with caplog.at_level(logging.WARNING):
            result = parse_schedule("0 9 * * * *")
        assert result is None

    def test_leading_trailing_whitespace_handled(self):
        """Leading/trailing whitespace is stripped."""
        trigger = parse_schedule("  0 9 * * *  ")
        assert trigger is not None


# ── parse_heartbeat_config tests ──────────────────────────


class TestParseHeartbeatConfig:
    """Tests for heartbeat.md parsing (unchanged from previous format)."""

    def test_basic_active_hours(self):
        """Parse active hours from heartbeat content."""
        start, end = parse_heartbeat_config("30分ごとにチェック\n9:00-22:00")
        assert start == 9
        assert end == 22

    def test_defaults_without_time_range(self):
        """None returned when no time range in content."""
        start, end = parse_heartbeat_config("15分間隔")
        assert start is None
        assert end is None


# ── skip_pattern tests ───────────────────────────────────


class TestSkipPattern:
    """Tests for skip_pattern directive parsing."""

    def test_skip_pattern_parsed(self):
        """skip_pattern directive is correctly parsed."""
        content = """\
## Check
schedule: */5 * * * *
type: command
tool: check_something
skip_pattern: ^\\[\\s*\\]$
args:
  sync: true
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].skip_pattern == "^\\[\\s*\\]$"

    def test_skip_pattern_none_when_absent(self):
        """skip_pattern is None when not specified."""
        content = """\
## Simple
schedule: 0 9 * * *
type: command
tool: simple_tool
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].skip_pattern is None

    def test_empty_skip_pattern_normalized_to_none(self):
        """Empty skip_pattern: line (no value) is normalized to None."""
        content = """\
## Empty
schedule: 0 9 * * *
type: command
tool: some_tool
skip_pattern:
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].skip_pattern is None

    def test_invalid_regex_skip_pattern_normalized_to_none(self, caplog):
        """Invalid regex in skip_pattern is warned and set to None."""
        content = """\
## Bad Regex
schedule: 0 9 * * *
type: command
tool: some_tool
skip_pattern: [unterminated
"""
        with caplog.at_level(logging.WARNING):
            tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].skip_pattern is None
        assert "Invalid skip_pattern" in caplog.text

    def test_skip_pattern_with_tool_and_args(self):
        """skip_pattern works alongside tool and args."""
        content = """\
## Full
schedule: */5 * * * *
type: command
tool: chatwork_unreplied
skip_pattern: ^\\[\\s*\\]$
args:
  sync: true
  sync_limit: 10
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].tool == "chatwork_unreplied"
        assert tasks[0].skip_pattern == "^\\[\\s*\\]$"
        assert tasks[0].args == {"sync": True, "sync_limit": 10}


# ── Blank template format tests ──────────────────────────


class TestBlankTemplateFormat:
    """Tests for the updated blank cron.md template."""

    def test_blank_template_parses_correctly(self):
        """Updated blank template with schedule: lines parses correctly."""
        content = (Path(__file__).resolve().parents[3] / "templates/anima_templates/_blank/cron.md").read_text()
        # Replace {name} placeholder
        content = content.replace("{name}", "test")
        tasks = parse_cron_md(content)
        # Active section has 2 tasks (commented-out ones are excluded)
        assert len(tasks) == 2
        assert tasks[0].name == "毎朝の業務計画"
        assert tasks[0].schedule == "0 9 * * *"
        assert tasks[0].type == "llm"
        assert tasks[1].name == "週次振り返り"
        assert tasks[1].schedule == "0 17 * * 4"
        assert tasks[1].type == "llm"

    def test_blank_template_schedules_are_valid(self):
        """All schedules in the blank template are valid cron expressions."""
        content = (Path(__file__).resolve().parents[3] / "templates/anima_templates/_blank/cron.md").read_text()
        content = content.replace("{name}", "test")
        tasks = parse_cron_md(content)
        for task in tasks:
            if task.schedule:
                trigger = parse_schedule(task.schedule)
                assert trigger is not None, f"Invalid schedule for {task.name}: '{task.schedule}'"

    def test_h3_heading_schedule_not_parsed(self):
        """H3 heading with cron expression should NOT be parsed as schedule."""
        content = """\
## Bad Task
### */5 * * * *
type: command
command: echo hello
"""
        tasks = parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].schedule == ""  # Should NOT have schedule
        assert tasks[0].name == "Bad Task"

    def test_natural_language_schedule_not_parsed(self):
        """Natural language time like '09:00' should NOT be a valid schedule."""
        assert parse_schedule("09:00") is None
        assert parse_schedule("毎週金曜 17:00") is None
        assert parse_schedule("every 5 minutes") is None
