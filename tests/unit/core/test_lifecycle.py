"""Unit tests for core/lifecycle.py — LifecycleManager and schedule parsing."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.lifecycle import (
    LifecycleManager,
    _parse_cron_md,
    _parse_schedule,
)
from core.schemas import CronTask, Message


# ── _parse_cron_md ────────────────────────────────────────


class TestParseCronMd:
    def test_empty_content(self):
        assert _parse_cron_md("") == []

    def test_single_task(self):
        content = """\
## 日次レポート
schedule: 0 9 * * *
毎朝の報告を作成する。
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].name == "日次レポート"
        assert tasks[0].schedule == "0 9 * * *"
        assert "毎朝の報告" in tasks[0].description

    def test_multiple_tasks(self):
        content = """\
## タスクA
schedule: 0 8 * * *
内容A

## タスクB
schedule: 0 17 * * 1-5
内容B
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 2
        assert tasks[0].name == "タスクA"
        assert tasks[1].name == "タスクB"

    def test_task_without_schedule(self):
        content = """\
## No Schedule Here
Just a description.
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].name == "No Schedule Here"
        assert tasks[0].schedule == ""

    def test_llm_type_explicit(self):
        """Test explicit LLM-type task parsing."""
        content = """\
## 業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認する。
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].type == "llm"
        assert tasks[0].name == "業務計画"
        assert "長期記憶" in tasks[0].description
        assert tasks[0].command is None
        assert tasks[0].tool is None

    def test_llm_type_default(self):
        """Test LLM-type task with default type (no explicit type field)."""
        content = """\
## 日次報告
schedule: 0 17 * * *
報告書を作成する。
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].type == "llm"  # Default type
        assert "報告書" in tasks[0].description

    def test_command_type_bash(self):
        """Test command-type task with bash command."""
        content = """\
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].type == "command"
        assert tasks[0].command == "/usr/local/bin/backup.sh"
        assert tasks[0].tool is None
        assert tasks[0].args is None

    def test_command_type_tool_with_args(self):
        """Test command-type task with internal tool and YAML args."""
        content = """\
## Slack通知
schedule: 0 9 * * 1-5
type: command
tool: slack_send_message
args:
  channel: "#general"
  message: "おはようございます！"
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 1
        assert tasks[0].type == "command"
        assert tasks[0].tool == "slack_send_message"
        assert tasks[0].command is None
        assert tasks[0].args is not None
        assert tasks[0].args["channel"] == "#general"
        assert tasks[0].args["message"] == "おはようございます！"

    def test_mixed_types(self):
        """Test parsing multiple tasks with different types."""
        content = """\
## 業務計画
schedule: 0 9 * * *
type: llm
計画を立てる。

## バックアップ
schedule: 0 2 * * *
type: command
command: /bin/backup.sh

## 通知送信
schedule: 0 10 * * 1-5
type: command
tool: send_notification
args:
  message: "Start working"
"""
        tasks = _parse_cron_md(content)
        assert len(tasks) == 3
        assert tasks[0].type == "llm"
        assert tasks[1].type == "command"
        assert tasks[1].command == "/bin/backup.sh"
        assert tasks[2].type == "command"
        assert tasks[2].tool == "send_notification"
        assert tasks[2].args["message"] == "Start working"


# ── _parse_schedule ───────────────────────────────────────


class TestParseSchedule:
    def test_daily(self):
        trigger = _parse_schedule("0 9 * * *")
        assert trigger is not None

    def test_weekday(self):
        trigger = _parse_schedule("0 9 * * 1-5")
        assert trigger is not None

    def test_weekly(self):
        trigger = _parse_schedule("0 17 * * 5")
        assert trigger is not None

    def test_monthly_day(self):
        trigger = _parse_schedule("0 9 1 * *")
        assert trigger is not None

    def test_every_5_minutes(self):
        trigger = _parse_schedule("*/5 * * * *")
        assert trigger is not None

    def test_every_hour(self):
        trigger = _parse_schedule("0 * * * *")
        assert trigger is not None

    def test_invalid_schedule(self):
        trigger = _parse_schedule("whenever I feel like it")
        assert trigger is None

    def test_japanese_format_rejected(self):
        trigger = _parse_schedule("毎日 9:00 JST")
        assert trigger is None

    def test_empty_schedule(self):
        trigger = _parse_schedule("")
        assert trigger is None


# ── LifecycleManager ──────────────────────────────────────


class TestLifecycleManager:
    def test_init(self):
        lm = LifecycleManager()
        assert lm.animas == {}
        assert lm._ws_broadcast is None

    def test_set_broadcast(self):
        lm = LifecycleManager()
        fn = AsyncMock()
        lm.set_broadcast(fn)
        assert lm._ws_broadcast is fn

    def test_register_anima(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.memory.read_heartbeat_config.return_value = ""
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()

        lm.register_anima(dp)
        assert "alice" in lm.animas
        dp.set_on_lock_released.assert_called_once()

    def test_unregister_anima(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.memory.read_heartbeat_config.return_value = ""
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()

        lm.register_anima(dp)
        lm.unregister_anima("alice")
        assert "alice" not in lm.animas

    def test_unregister_nonexistent(self):
        lm = LifecycleManager()
        lm.unregister_anima("nobody")  # should not raise


class TestHeartbeatWrapper:
    async def test_heartbeat_with_broadcast(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.run_heartbeat = AsyncMock(return_value=MagicMock())
        dp.run_heartbeat.return_value.model_dump.return_value = {}
        lm.animas["alice"] = dp

        broadcast = AsyncMock()
        lm._ws_broadcast = broadcast

        await lm._heartbeat_wrapper("alice")
        dp.run_heartbeat.assert_called_once()
        broadcast.assert_called_once()

    async def test_heartbeat_no_anima(self):
        lm = LifecycleManager()
        # Should return silently
        await lm._heartbeat_wrapper("nobody")


class TestCronWrapper:
    async def test_cron_with_broadcast(self):
        import asyncio

        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.run_cron_task = AsyncMock(return_value=MagicMock())
        dp.run_cron_task.return_value.model_dump.return_value = {}
        lm.animas["alice"] = dp

        broadcast = AsyncMock()
        lm._ws_broadcast = broadcast

        # Create CronTask object (LLM type)
        task = CronTask(
            name="daily_report",
            schedule="0 9 * * *",
            type="llm",
            description="Generate report",
        )
        await lm._cron_wrapper("alice", task)
        # _cron_wrapper creates a background task; let it run
        await asyncio.sleep(0)
        dp.run_cron_task.assert_called_once_with("daily_report", "Generate report")
        broadcast.assert_called_once()

    async def test_cron_no_anima(self):
        lm = LifecycleManager()
        task = CronTask(name="task", schedule="0 10 * * *", description="desc")
        await lm._cron_wrapper("nobody", task)


class TestSetupHeartbeat:
    def test_interval_is_always_30_minutes(self):
        """Heartbeat interval is fixed at 30 minutes regardless of config."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        # Even if config says 15 minutes, interval should remain 30
        dp.memory.read_heartbeat_config.return_value = "巡回間隔: 15分"
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()

        lm.register_anima(dp)
        jobs = lm.scheduler.get_jobs()
        hb_job = next(j for j in jobs if j.id == "alice_heartbeat")
        # CronTrigger fields: verify minute is */30
        assert str(hb_job.trigger).find("*/30") != -1

    def test_interval_fixed_with_5min_config(self):
        """Ensure 5-minute config is ignored; interval stays 30."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "bob"
        dp.memory.read_heartbeat_config.return_value = "5分ごと"
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()

        lm.register_anima(dp)
        jobs = lm.scheduler.get_jobs()
        hb_job = next(j for j in jobs if j.id == "bob_heartbeat")
        assert str(hb_job.trigger).find("*/30") != -1

    def test_default_24h_when_no_time_range(self):
        """No time range in heartbeat.md means 24h heartbeat (hour='*')."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "carol"
        dp.memory.read_heartbeat_config.return_value = "- チェック項目"
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()

        lm.register_anima(dp)
        jobs = lm.scheduler.get_jobs()
        hb_job = next(j for j in jobs if j.id == "carol_heartbeat")
        trigger_str = str(hb_job.trigger)
        # hour should not be restricted — no "hour=" range like "9-21"
        assert "*/30" in trigger_str

    def test_parses_active_hours_from_heartbeat_md(self):
        """Time range in heartbeat.md restricts heartbeat hours."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "bob"
        dp.memory.read_heartbeat_config.return_value = "稼働時間: 8:00 - 20:00"
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()

        lm.register_anima(dp)
        jobs = lm.scheduler.get_jobs()
        assert any(j.id == "bob_heartbeat" for j in jobs)


class TestMessageTriggeredHeartbeat:
    async def test_triggered_heartbeat_success(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.run_heartbeat = AsyncMock(return_value=MagicMock())
        dp.run_heartbeat.return_value.model_dump.return_value = {}
        # Provide an actionable message so intent filter passes
        msg = Message(
            from_person="bob", to_person="alice",
            content="please do X", intent="delegation", source="anima",
        )
        dp.messenger.receive.return_value = [msg]
        lm.animas["alice"] = dp
        lm._pending_triggers.add("alice")

        await lm._message_triggered_heartbeat("alice")
        dp.run_heartbeat.assert_called_once()
        assert "alice" not in lm._pending_triggers

    async def test_triggered_heartbeat_no_anima(self):
        lm = LifecycleManager()
        lm._pending_triggers.add("nobody")
        await lm._message_triggered_heartbeat("nobody")
        assert "nobody" not in lm._pending_triggers


class TestOnAnimaLockReleased:
    async def test_schedules_deferred_when_unread(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = True
        lm.animas["alice"] = dp

        with patch.object(lm, "_schedule_deferred_trigger") as mock_sched:
            await lm._on_anima_lock_released("alice")
            mock_sched.assert_called_once_with("alice")

    async def test_no_action_when_no_unread(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = False
        lm.animas["alice"] = dp

        with patch.object(lm, "_schedule_deferred_trigger") as mock_sched:
            await lm._on_anima_lock_released("alice")
            mock_sched.assert_not_called()

    async def test_no_action_when_pending_trigger(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = True
        lm.animas["alice"] = dp
        lm._pending_triggers.add("alice")

        with patch.object(lm, "_schedule_deferred_trigger") as mock_sched:
            await lm._on_anima_lock_released("alice")
            mock_sched.assert_not_called()

    async def test_no_action_when_anima_not_registered(self):
        lm = LifecycleManager()
        await lm._on_anima_lock_released("nobody")


class TestScheduleDeferredTrigger:
    def test_schedules_timer(self):
        """_schedule_deferred_trigger creates a timer in _deferred_timers."""
        import asyncio
        lm = LifecycleManager()
        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.get_running_loop", return_value=loop):
                lm._schedule_deferred_trigger("alice")
            assert "alice" in lm._deferred_timers
            # Clean up
            lm._deferred_timers["alice"].cancel()
        finally:
            loop.close()

    def test_noop_when_already_scheduled(self):
        """Second call is a no-op when timer already exists."""
        lm = LifecycleManager()
        sentinel = MagicMock()
        lm._deferred_timers["alice"] = sentinel

        lm._schedule_deferred_trigger("alice")
        # Timer should still be the same sentinel object
        assert lm._deferred_timers["alice"] is sentinel

    def test_unregister_cancels_timer(self):
        """unregister_anima cancels any deferred timer."""
        lm = LifecycleManager()
        mock_timer = MagicMock()
        lm._deferred_timers["alice"] = mock_timer

        dp = MagicMock()
        dp.name = "alice"
        dp.memory.read_heartbeat_config.return_value = ""
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()
        lm.animas["alice"] = dp

        lm.unregister_anima("alice")
        mock_timer.cancel.assert_called_once()
        assert "alice" not in lm._deferred_timers


class TestTryDeferredTrigger:
    async def test_triggers_heartbeat_when_ready(self):
        """Fires heartbeat when not in cooldown and lock not held."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = True
        dp._lock = MagicMock()
        dp._lock.locked.return_value = False
        lm.animas["alice"] = dp
        # Set timer entry so pop works
        lm._deferred_timers["alice"] = MagicMock()

        with patch.object(lm, "_is_in_cooldown", return_value=False), \
             patch.object(lm, "_message_triggered_heartbeat", new_callable=AsyncMock) as mock_hb:
            await lm._try_deferred_trigger("alice")
            assert "alice" in lm._pending_triggers

    async def test_reschedules_when_in_cooldown(self):
        """Re-schedules if still in cooldown."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = True
        lm.animas["alice"] = dp
        lm._deferred_timers["alice"] = MagicMock()

        with patch.object(lm, "_is_in_cooldown", return_value=True), \
             patch.object(lm, "_schedule_deferred_trigger") as mock_sched:
            await lm._try_deferred_trigger("alice")
            mock_sched.assert_called_once_with("alice")
            assert "alice" not in lm._pending_triggers

    async def test_reschedules_when_lock_held(self):
        """Re-schedules if anima lock is held."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = True
        dp._lock = MagicMock()
        dp._lock.locked.return_value = True
        lm.animas["alice"] = dp
        lm._deferred_timers["alice"] = MagicMock()

        with patch.object(lm, "_is_in_cooldown", return_value=False), \
             patch.object(lm, "_schedule_deferred_trigger") as mock_sched:
            await lm._try_deferred_trigger("alice")
            mock_sched.assert_called_once_with("alice")

    async def test_noop_when_no_unread(self):
        """Does nothing if inbox is empty."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = False
        lm.animas["alice"] = dp
        lm._deferred_timers["alice"] = MagicMock()

        await lm._try_deferred_trigger("alice")
        assert "alice" not in lm._pending_triggers

    async def test_noop_when_pending_trigger(self):
        """Does nothing if trigger already pending."""
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.messenger.has_unread.return_value = True
        lm.animas["alice"] = dp
        lm._deferred_timers["alice"] = MagicMock()
        lm._pending_triggers.add("alice")

        await lm._try_deferred_trigger("alice")
        # Should not have scheduled anything new


class TestLifecycleStartShutdown:
    async def test_start_and_shutdown(self):
        lm = LifecycleManager()
        lm.start()
        assert lm._inbox_watcher_task is not None
        assert lm.scheduler.running
        lm.shutdown()
        # After shutdown, the inbox watcher task should be in cancelling state
        assert lm._inbox_watcher_task.cancelling() > 0 or lm._inbox_watcher_task.cancelled()

    async def test_shutdown_cancels_deferred_timers(self):
        lm = LifecycleManager()
        mock_timer = MagicMock()
        lm._deferred_timers["alice"] = mock_timer
        lm.start()
        lm.shutdown()
        mock_timer.cancel.assert_called_once()
        assert lm._deferred_timers == {}


# ── Command-type cron execution ───────────────────────────


class TestCommandTypeCron:
    """Test command-type cron execution (bash and tool)."""

    async def test_run_cron_command_bash_success(self):
        """Test successful bash command execution in cron."""
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from core.anima import DigitalAnima
        from core.memory import MemoryManager

        # Create mock anima
        anima_dir = Path("/tmp/test_anima")
        shared_dir = Path("/tmp/shared")
        anima_dir.mkdir(parents=True, exist_ok=True)
        shared_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(MemoryManager, "read_model_config"):
            dp = DigitalAnima(anima_dir, shared_dir)

        # Mock append_cron_command_log
        dp.memory.append_cron_command_log = MagicMock()

        # Execute command
        result = await dp.run_cron_command(
            "test_task",
            command="echo 'Hello World'",
        )

        # Verify result
        assert result["exit_code"] == 0
        assert "Hello World" in result["stdout"]
        assert result["stderr"] == ""
        dp.memory.append_cron_command_log.assert_called_once()

    async def test_run_cron_command_bash_failure(self):
        """Test bash command that returns non-zero exit code."""
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from core.anima import DigitalAnima
        from core.memory import MemoryManager

        anima_dir = Path("/tmp/test_anima2")
        shared_dir = Path("/tmp/shared2")
        anima_dir.mkdir(parents=True, exist_ok=True)
        shared_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(MemoryManager, "read_model_config"):
            dp = DigitalAnima(anima_dir, shared_dir)

        dp.memory.append_cron_command_log = MagicMock()

        # Execute failing command
        result = await dp.run_cron_command(
            "failing_task",
            command="exit 1",
        )

        # Verify non-zero exit code
        assert result["exit_code"] == 1
        dp.memory.append_cron_command_log.assert_called_once()

    async def test_run_cron_command_tool(self):
        """Test internal tool execution in cron."""
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from core.anima import DigitalAnima
        from core.memory import MemoryManager

        anima_dir = Path("/tmp/test_anima3")
        shared_dir = Path("/tmp/shared3")
        anima_dir.mkdir(parents=True, exist_ok=True)
        shared_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(MemoryManager, "read_model_config"):
            dp = DigitalAnima(anima_dir, shared_dir)

        dp.memory.append_cron_command_log = MagicMock()

        # Mock tool_handler.handle
        dp.agent._tool_handler.handle = MagicMock(return_value="Tool executed successfully")

        # Execute tool
        result = await dp.run_cron_command(
            "tool_task",
            tool="test_tool",
            args={"key": "value"},
        )

        # Verify result
        assert result["exit_code"] == 0
        assert "Tool executed successfully" in result["stdout"]
        dp.agent._tool_handler.handle.assert_called_once_with(
            "test_tool",
            {"key": "value"},
        )
        dp.memory.append_cron_command_log.assert_called_once()


# ── ReloadAnimaSchedule ─────────────────────────────────


class TestReloadAnimaSchedule:
    def test_reload_nonexistent_anima(self):
        lm = LifecycleManager()
        result = lm.reload_anima_schedule("nobody")
        assert "error" in result

    def test_reload_registered_anima(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.memory.read_heartbeat_config.return_value = "9:00 - 22:00"
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()
        dp.set_on_schedule_changed = MagicMock()

        lm.register_anima(dp)
        initial_jobs = [j.id for j in lm.scheduler.get_jobs() if j.id.startswith("alice_")]
        assert len(initial_jobs) >= 1

        # Change active hours only; interval stays 30
        dp.memory.read_heartbeat_config.return_value = "8:00 - 23:00"
        result = lm.reload_anima_schedule("alice")

        assert result["reloaded"] == "alice"
        assert result["removed"] >= 1
        assert len(result["new_jobs"]) >= 1
        # Verify interval is still 30 after reload
        hb_job = next(
            j for j in lm.scheduler.get_jobs() if j.id == "alice_heartbeat"
        )
        assert str(hb_job.trigger).find("*/30") != -1

    def test_reload_with_cron_tasks(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "bob"
        dp.memory.read_heartbeat_config.return_value = "30分ごと\n9:00 - 22:00"
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()
        dp.set_on_schedule_changed = MagicMock()

        lm.register_anima(dp)

        # Add a cron task and reload
        dp.memory.read_cron_config.return_value = """\
## ログチェック
schedule: 0 10 * * *
type: llm
サーバーログを確認する。
"""
        result = lm.reload_anima_schedule("bob")
        assert result["reloaded"] == "bob"
        # Should have heartbeat + cron job
        assert len(result["new_jobs"]) >= 2

    def test_register_anima_wires_schedule_callback(self):
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.memory.read_heartbeat_config.return_value = ""
        dp.memory.read_cron_config.return_value = ""
        dp.set_on_lock_released = MagicMock()
        dp.set_on_schedule_changed = MagicMock()

        lm.register_anima(dp)
        dp.set_on_schedule_changed.assert_called_once_with(lm.reload_anima_schedule)
