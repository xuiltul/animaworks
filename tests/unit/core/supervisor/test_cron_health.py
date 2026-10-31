"""Tests for cron.md parse diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.runtime.scheduler_manager import SchedulerManager
from core.schemas import CronTask


@pytest.fixture
def scheduler_mgr(tmp_path: Path) -> SchedulerManager:
    """Create a SchedulerManager with a temp anima dir."""
    anima = MagicMock()
    anima.memory.read_heartbeat_config.return_value = ""
    anima.memory.read_cron_config.return_value = ""
    mgr = SchedulerManager(
        anima=anima,
        anima_name="test_anima",
        anima_dir=tmp_path,
        emit_event=MagicMock(),
    )
    return mgr


def _notif_dir(tmp_path: Path) -> Path:
    return tmp_path / "state" / "background_notifications"


def _notif_files(tmp_path: Path) -> list[Path]:
    d = _notif_dir(tmp_path)
    if not d.exists():
        return []
    return sorted(d.glob("cron_health_*.md"))


def _make_task(name: str, schedule: str = "") -> CronTask:
    return CronTask(name=name, schedule=schedule, type="llm", description="")


# ── Layer 1: _check_cron_parse_health ─────────────────────────


class TestCheckCronParseHealth:
    """Layer 1 — immediate detection at setup/reload time."""

    def test_no_notification_when_all_registered_no_issues(
        self, scheduler_mgr: SchedulerManager, tmp_path: Path
    ) -> None:
        tasks = [_make_task("t1", "0 9 * * *")]
        scheduler_mgr._check_cron_parse_health("schedule: 0 9 * * *", tasks, registered=1)
        assert _notif_files(tmp_path) == []

    def test_all_schedules_invalid(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        tasks = [_make_task("t1", "bad"), _make_task("t2", "also bad")]
        scheduler_mgr._check_cron_parse_health("## t1\nschedule: bad\n## t2\nschedule: also bad", tasks, registered=0)
        files = _notif_files(tmp_path)
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "2" in content  # task_count=2

    def test_indented_schedule_detected(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        raw = "## Good\nschedule: 0 9 * * *\n## Example\n  schedule: 0 10 * * *"
        tasks = [_make_task("good", "0 9 * * *")]
        scheduler_mgr._check_cron_parse_health(raw, tasks, registered=1)
        files = _notif_files(tmp_path)
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "schedule:" in content

    def test_template_documentation_does_not_trigger_warning(
        self, scheduler_mgr: SchedulerManager, tmp_path: Path
    ) -> None:
        from core.runtime.schedule_parser import parse_cron_md

        template_path = Path(__file__).parents[4] / "templates" / "ja" / "anima_templates" / "_blank" / "cron.md"
        raw = template_path.read_text(encoding="utf-8").replace("{name}", "test_anima")
        tasks = parse_cron_md(raw)
        scheduler_mgr._check_cron_parse_health(raw, tasks, registered=len(tasks))
        assert _notif_files(tmp_path) == []

    def test_indented_schedule_detected_even_with_valid_jobs(
        self, scheduler_mgr: SchedulerManager, tmp_path: Path
    ) -> None:
        """Indented schedule: lines are warned even when some jobs register."""
        raw = "## Good\nschedule: 0 9 * * *\n## Bad\n  schedule: 0 10 * * *"
        tasks = [_make_task("good", "0 9 * * *"), _make_task("bad")]
        scheduler_mgr._check_cron_parse_health(raw, tasks, registered=1)
        files = _notif_files(tmp_path)
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "schedule:" in content

    def test_multiple_issues_combined_in_single_file(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        """All-invalid + indented should produce exactly one file."""
        raw = "## t1\n  schedule: bad"
        tasks = [_make_task("t1", "bad")]
        scheduler_mgr._check_cron_parse_health(raw, tasks, registered=0)
        files = _notif_files(tmp_path)
        assert len(files) == 1

    def test_unrecognized_schedule_no_tasks(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        raw = 'schedule: "0 9 * * *"'
        scheduler_mgr._check_cron_parse_health(raw, tasks=[], registered=0)
        files = _notif_files(tmp_path)
        assert len(files) == 1

    def test_documentation_only_no_notification(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        raw = "<!--\n  schedule: 0 9 * * *\n-->\n```yaml\n  schedule: 0 10 * * *\n```"
        scheduler_mgr._check_cron_parse_health(raw, tasks=[], registered=0)
        assert _notif_files(tmp_path) == []

    def test_empty_config_no_notification(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        scheduler_mgr._check_cron_parse_health("", tasks=[], registered=0)
        assert _notif_files(tmp_path) == []


# ── Layer 1 integration: _setup_cron_tasks ────────────────────


class TestSetupCronTasksHealthIntegration:
    """_setup_cron_tasks invokes _check_cron_parse_health."""

    def test_invalid_cron_triggers_notification(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        scheduler_mgr._anima.memory.read_cron_config.return_value = (
            "## My Task\nschedule: INVALID\ntype: llm\nDo something\n"
        )
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        scheduler_mgr.scheduler = mock_scheduler

        scheduler_mgr._setup_cron_tasks()

        files = _notif_files(tmp_path)
        assert len(files) >= 1
        content = files[0].read_text(encoding="utf-8")
        assert "1" in content  # 1 task defined

    def test_valid_cron_no_notification(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        scheduler_mgr._anima.memory.read_cron_config.return_value = (
            "## My Task\nschedule: 0 9 * * *\ntype: llm\nDo something\n"
        )
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        scheduler_mgr.scheduler = mock_scheduler

        scheduler_mgr._setup_cron_tasks()

        assert _notif_files(tmp_path) == []

    def test_legacy_disabled_state_does_not_skip_cron_registration(
        self, scheduler_mgr: SchedulerManager, tmp_path: Path
    ) -> None:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "cron_disabled.json").write_text('{"Daily": {"reason": "legacy"}}', encoding="utf-8")
        scheduler_mgr._anima.memory.read_cron_config.return_value = (
            "## Daily\nschedule: 0 9 * * *\ntype: llm\nDo something\n"
        )
        mock_scheduler = MagicMock()
        scheduler_mgr.scheduler = mock_scheduler

        scheduler_mgr._setup_cron_tasks()

        mock_scheduler.add_job.assert_called_once()

    def test_registration_result_records_registered_and_rejected(
        self, scheduler_mgr: SchedulerManager, tmp_path: Path
    ) -> None:
        scheduler_mgr._anima.memory.read_cron_config.return_value = (
            "## Notes\nNot a cron job\n"
            "## Broken\ntype: command\ncommand: echo hi\n"
            "## Daily\nschedule: 0 9 * * *\ntype: llm\nDo something\n"
        )
        scheduler_mgr.scheduler = MagicMock()

        scheduler_mgr._setup_cron_tasks()

        result = json.loads((tmp_path / "state" / "cron_registration.json").read_text(encoding="utf-8"))
        assert result["registered"] == [{"name": "Daily", "schedule": "0 9 * * *"}]
        # Prose sections are ignored; only a section with an action but no
        # schedule is a real misconfiguration worth nagging the anima about.
        assert result["rejected"] == [{"name": "Broken", "reason": "Empty schedule expression"}]
        assert result["parsed_at"]
        scheduler_mgr._anima.memory.append_cron_event.assert_called_once_with(
            "Daily",
            "scheduled",
            reason="",
            schedule="0 9 * * *",
        )

    def test_invalid_rejected_but_llm_codeblock_registered(
        self, scheduler_mgr: SchedulerManager, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        scheduler_mgr._anima.memory.read_cron_config.return_value = (
            "## Invalid\nschedule: tomorrow\ntype: llm\nNo\n"
            "## Scripted\nschedule: 0 9 * * *\ntype: llm\n```sh\necho ok\n```\n"
        )
        scheduler_mgr.scheduler = MagicMock()

        scheduler_mgr._setup_cron_tasks()

        result = json.loads((tmp_path / "state" / "cron_registration.json").read_text(encoding="utf-8"))
        assert result["registered"] == [{"name": "Scripted", "schedule": "0 9 * * *"}]
        assert result["rejected"] == [{"name": "Invalid", "reason": "Invalid cron expression"}]
        assert "contains a code block" in caplog.text

    def test_registration_write_failure_does_not_stop_registration(self, scheduler_mgr: SchedulerManager) -> None:
        scheduler_mgr._anima.memory.read_cron_config.return_value = (
            "## Daily\nschedule: 0 9 * * *\ntype: llm\nDo something\n"
        )
        scheduler_mgr.scheduler = MagicMock()

        with patch("core.runtime.scheduler_manager.atomic_write_json", side_effect=OSError("read-only")):
            scheduler_mgr._setup_cron_tasks()

        scheduler_mgr.scheduler.add_job.assert_called_once()


# ── _write_cron_health_notification ───────────────────────────


class TestWriteCronHealthNotification:
    def test_creates_md_file(self, scheduler_mgr: SchedulerManager, tmp_path: Path) -> None:
        scheduler_mgr._write_cron_health_notification("Test warning message")

        files = _notif_files(tmp_path)
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "Test warning message" in content
        assert "⚠️" in content

    def test_write_failure_does_not_raise(self, scheduler_mgr: SchedulerManager) -> None:
        scheduler_mgr._anima_dir = Path("/nonexistent/path/should/fail")
        scheduler_mgr._write_cron_health_notification("msg")
