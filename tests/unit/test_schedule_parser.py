"""Unit tests for core.schedule_parser module."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from apscheduler.triggers.cron import CronTrigger

from core.schedule_parser import (
    parse_heartbeat_config,
    parse_cron_md,
    parse_schedule,
)


class TestParseHeartbeatConfig:
    """Tests for parse_heartbeat_config()."""

    def test_basic_interval_and_active_hours(self):
        content = """# Heartbeat
## 実行間隔
5分ごと
## 活動時間
8:00 - 23:00（JST）
"""
        interval, start, end = parse_heartbeat_config(content)
        assert interval == 5
        assert start == 8
        assert end == 23

    def test_30min_interval(self):
        content = "30分ごと\n活動時間\n9:00 - 22:00"
        interval, start, end = parse_heartbeat_config(content)
        assert interval == 30
        assert start == 9
        assert end == 22

    def test_defaults_when_no_match(self):
        content = "some random content without patterns"
        interval, start, end = parse_heartbeat_config(content)
        assert interval == 30
        assert start == 9
        assert end == 22

    def test_empty_content(self):
        interval, start, end = parse_heartbeat_config("")
        assert interval == 30
        assert start == 9
        assert end == 22


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
