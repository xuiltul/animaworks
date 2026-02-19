"""Unit tests for cron.md migration from Japanese text schedules to cron expressions."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pathlib import Path

from core.config.migrate import (
    _convert_jp_schedule_to_cron,
    _is_already_migrated,
    migrate_cron_format,
    migrate_all_cron,
)


# ── _convert_jp_schedule_to_cron tests ────────────────────


class TestConvertJpScheduleToCron:
    """Tests for individual Japanese schedule pattern conversion."""

    def test_daily(self):
        """毎日 HH:MM -> MM HH * * *"""
        assert _convert_jp_schedule_to_cron("毎日 9:00") == "0 9 * * *"
        assert _convert_jp_schedule_to_cron("毎日 18:30") == "30 18 * * *"

    def test_daily_with_timezone(self):
        """毎日 HH:MM JST -> MM HH * * * (timezone stripped)."""
        assert _convert_jp_schedule_to_cron("毎日 9:00 JST") == "0 9 * * *"
        assert _convert_jp_schedule_to_cron("毎日 8:00 UTC") == "0 8 * * *"

    def test_weekday(self):
        """平日 HH:MM -> MM HH * * 1-5"""
        assert _convert_jp_schedule_to_cron("平日 9:00") == "0 9 * * 1-5"
        assert _convert_jp_schedule_to_cron("平日 8:30 JST") == "30 8 * * 1-5"

    def test_weekly_with_day(self):
        """毎週X曜 HH:MM -> MM HH * * N"""
        assert _convert_jp_schedule_to_cron("毎週月曜 9:00") == "0 9 * * 1"
        assert _convert_jp_schedule_to_cron("毎週金曜 17:00") == "0 17 * * 5"
        assert _convert_jp_schedule_to_cron("毎週日曜 10:00") == "0 10 * * 0"
        assert _convert_jp_schedule_to_cron("毎週火曜 14:30") == "30 14 * * 2"

    def test_weekly_short_day_names(self):
        """毎週X HH:MM with short day names (月, 火, etc.)."""
        assert _convert_jp_schedule_to_cron("毎週月 9:00") == "0 9 * * 1"
        assert _convert_jp_schedule_to_cron("毎週金 17:00") == "0 17 * * 5"
        assert _convert_jp_schedule_to_cron("毎週土 10:00") == "0 10 * * 6"

    def test_monthly_day(self):
        """毎月DD日 HH:MM -> MM HH DD * *"""
        assert _convert_jp_schedule_to_cron("毎月1日 9:00") == "0 9 1 * *"
        assert _convert_jp_schedule_to_cron("毎月15日 10:00") == "0 10 15 * *"

    def test_every_n_minutes(self):
        """X分毎 -> */X * * * *"""
        assert _convert_jp_schedule_to_cron("5分毎") == "*/5 * * * *"
        assert _convert_jp_schedule_to_cron("30分毎") == "*/30 * * * *"

    def test_every_n_hours(self):
        """X時間毎 -> 0 */X * * *"""
        assert _convert_jp_schedule_to_cron("2時間毎") == "0 */2 * * *"
        assert _convert_jp_schedule_to_cron("6時間毎") == "0 */6 * * *"

    def test_biweekly_returns_none(self):
        """隔週 patterns cannot be expressed in standard cron."""
        assert _convert_jp_schedule_to_cron("隔週金曜 17:00") is None

    def test_last_day_of_month_returns_none(self):
        """毎月最終日 cannot be expressed in standard cron."""
        assert _convert_jp_schedule_to_cron("毎月最終日 18:00") is None

    def test_nth_weekday_returns_none(self):
        """第NX曜 patterns cannot be expressed in standard cron."""
        assert _convert_jp_schedule_to_cron("第2火曜 10:00") is None

    def test_empty_string(self):
        """Empty string returns None."""
        assert _convert_jp_schedule_to_cron("") is None

    def test_unrecognized_pattern_returns_none(self):
        """Completely unrecognized patterns return None."""
        assert _convert_jp_schedule_to_cron("random text") is None


# ── _is_already_migrated tests ────────────────────────────


class TestIsAlreadyMigrated:
    """Tests for detecting already-migrated cron.md files."""

    def test_new_format_detected(self):
        """File with schedule: directive is detected as already migrated."""
        content = """\
## Morning Task
schedule: 0 9 * * *
type: llm
Do something.
"""
        assert _is_already_migrated(content) is True

    def test_old_format_not_detected(self):
        """File with old Japanese format is not detected as migrated."""
        content = """\
## Morning Task（毎日 9:00 JST）
type: llm
Do something.
"""
        assert _is_already_migrated(content) is False

    def test_schedule_inside_comment_not_counted(self):
        """schedule: inside HTML comment does not count as migrated."""
        content = """\
<!-- ## Old Task
schedule: 0 9 * * *
type: llm -->

## Active Task（毎日 10:00）
type: llm
Do something.
"""
        assert _is_already_migrated(content) is False


# ── migrate_cron_format tests ─────────────────────────────


class TestMigrateCronFormat:
    """Tests for full cron.md file migration."""

    def test_basic_migration(self, tmp_path):
        """Basic migration converts Japanese schedule to cron expression."""
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "## 毎朝の業務計画（毎日 9:00 JST）\n"
            "type: llm\n"
            "昨日の進捗を確認し、今日のタスクを計画する。\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "schedule: 0 9 * * *" in migrated
        assert "## 毎朝の業務計画" in migrated
        # The old parenthesized schedule should be gone
        assert "（毎日 9:00 JST）" not in migrated

    def test_multiple_tasks_migration(self, tmp_path):
        """Multiple tasks are all migrated."""
        anima_dir = tmp_path / "bob"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "## 朝会（毎日 9:00）\n"
            "type: llm\n"
            "朝のチェック。\n"
            "\n"
            "## 週報（毎週金曜 17:00）\n"
            "type: llm\n"
            "週次レポートを作成。\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "schedule: 0 9 * * *" in migrated
        assert "schedule: 0 17 * * 5" in migrated

    def test_weekday_migration(self, tmp_path):
        """平日 schedule is migrated correctly."""
        anima_dir = tmp_path / "carol"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "## 日次チェック（平日 8:30）\n"
            "type: llm\n"
            "日次確認。\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "schedule: 30 8 * * 1-5" in migrated

    def test_monthly_migration(self, tmp_path):
        """毎月DD日 schedule is migrated correctly."""
        anima_dir = tmp_path / "dave"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "## 月次レポート（毎月1日 10:00）\n"
            "type: llm\n"
            "月次報告。\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "schedule: 0 10 1 * *" in migrated

    def test_already_migrated_skipped(self, tmp_path):
        """Already-migrated file returns False."""
        anima_dir = tmp_path / "eve"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        content = (
            "## Morning Task\n"
            "schedule: 0 9 * * *\n"
            "type: llm\n"
            "Do something.\n"
        )
        cron_md.write_text(content, encoding="utf-8")

        result = migrate_cron_format(anima_dir)
        assert result is False
        # Content should be unchanged
        assert cron_md.read_text(encoding="utf-8") == content

    def test_missing_cron_md(self, tmp_path):
        """Missing cron.md returns False."""
        anima_dir = tmp_path / "frank"
        anima_dir.mkdir()
        assert migrate_cron_format(anima_dir) is False

    def test_empty_cron_md(self, tmp_path):
        """Empty cron.md returns False."""
        anima_dir = tmp_path / "grace"
        anima_dir.mkdir()
        (anima_dir / "cron.md").write_text("", encoding="utf-8")
        assert migrate_cron_format(anima_dir) is False

    def test_unconvertible_schedule_noted(self, tmp_path):
        """Unconvertible schedules get a migration note comment."""
        anima_dir = tmp_path / "henry"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "## 隔週ミーティング（隔週金曜 17:00）\n"
            "type: llm\n"
            "隔週のミーティング。\n"
            "\n"
            "## 朝会（毎日 9:00）\n"
            "type: llm\n"
            "朝のチェック。\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "MIGRATION NOTE" in migrated
        assert "隔週金曜 17:00" in migrated
        # The convertible task should be migrated
        assert "schedule: 0 9 * * *" in migrated

    def test_html_comments_preserved(self, tmp_path):
        """HTML comment blocks are preserved during migration."""
        anima_dir = tmp_path / "iris"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "<!-- This is a top-level comment -->\n"
            "\n"
            "## 朝会（毎日 9:00）\n"
            "type: llm\n"
            "朝のチェック。\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "<!-- This is a top-level comment -->" in migrated
        assert "schedule: 0 9 * * *" in migrated

    def test_minutes_interval_migration(self, tmp_path):
        """X分毎 pattern is migrated correctly."""
        anima_dir = tmp_path / "jack"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "## ヘルスチェック（5分毎）\n"
            "type: command\n"
            "tool: health_check\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "schedule: */5 * * * *" in migrated

    def test_task_with_ascii_parens(self, tmp_path):
        """Task with ASCII parentheses (not fullwidth) is also handled."""
        anima_dir = tmp_path / "kate"
        anima_dir.mkdir()
        cron_md = anima_dir / "cron.md"
        cron_md.write_text(
            "## Morning Task (毎日 9:00 JST)\n"
            "type: llm\n"
            "Do something.\n",
            encoding="utf-8",
        )

        result = migrate_cron_format(anima_dir)
        assert result is True

        migrated = cron_md.read_text(encoding="utf-8")
        assert "schedule: 0 9 * * *" in migrated
        assert "(毎日 9:00 JST)" not in migrated


# ── migrate_all_cron tests ────────────────────────────────


class TestMigrateAllCron:
    """Tests for bulk cron migration across all animas."""

    def test_migrate_multiple_animas(self, tmp_path):
        """Migrates cron.md for multiple animas."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        for name in ("alice", "bob"):
            d = animas_dir / name
            d.mkdir()
            (d / "cron.md").write_text(
                f"## {name}のタスク（毎日 9:00）\n"
                "type: llm\n"
                "タスク実行。\n",
                encoding="utf-8",
            )

        count = migrate_all_cron(animas_dir)
        assert count == 2

    def test_skips_non_directories(self, tmp_path):
        """Non-directory entries are skipped."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        (animas_dir / "not_a_dir.txt").write_text("hello", encoding="utf-8")

        count = migrate_all_cron(animas_dir)
        assert count == 0

    def test_nonexistent_directory(self, tmp_path):
        """Non-existent directory returns 0."""
        count = migrate_all_cron(tmp_path / "nonexistent")
        assert count == 0

    def test_mix_of_migrated_and_unmigrated(self, tmp_path):
        """Only unmigrated animas are counted."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        # Already migrated
        alice = animas_dir / "alice"
        alice.mkdir()
        (alice / "cron.md").write_text(
            "## Task\nschedule: 0 9 * * *\ntype: llm\nDo it.\n",
            encoding="utf-8",
        )

        # Needs migration
        bob = animas_dir / "bob"
        bob.mkdir()
        (bob / "cron.md").write_text(
            "## タスク（毎日 10:00）\ntype: llm\nやる。\n",
            encoding="utf-8",
        )

        count = migrate_all_cron(animas_dir)
        assert count == 1
