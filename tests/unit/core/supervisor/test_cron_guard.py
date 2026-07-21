"""Unit tests for cron execution guard statistics and enforcement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli.commands.cron_guard import cmd_cron_guard_enable, cmd_cron_guard_list
from core.config.schemas import AnimaWorksConfig, CronGuardConfig
from core.schemas import CronTask
from core.supervisor.scheduler_manager import SchedulerManager


@pytest.fixture
def scheduler_mgr(tmp_path: Path) -> SchedulerManager:
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    anima = MagicMock()
    anima._activity = MagicMock()
    anima.memory.read_cron_config.return_value = ""
    anima.run_cron_task = AsyncMock()
    anima.run_cron_command = AsyncMock()
    manager = SchedulerManager(anima, "test", anima_dir, MagicMock())
    manager.scheduler = MagicMock()
    manager.scheduler.get_jobs.return_value = []
    return manager


def _config(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(cron_guard=CronGuardConfig(**overrides))


def _stats(manager: SchedulerManager) -> dict[str, object]:
    return json.loads((manager._anima_dir / "state" / "cron_stats.json").read_text(encoding="utf-8"))


def _task(name: str = "guarded") -> CronTask:
    return CronTask(
        name=name,
        schedule="*/5 * * * *",
        type="llm",
        description="Run guarded work",
    )


def test_config_defaults_are_conservative() -> None:
    config = AnimaWorksConfig()
    assert config.cron_guard == CronGuardConfig(
        mode="warn",
        max_fires_per_window=60,
        window_minutes=60,
        max_consecutive_failures=5,
    )


def test_frequency_threshold_records_warning(scheduler_mgr: SchedulerManager) -> None:
    with patch("core.supervisor.scheduler_manager.load_config", return_value=_config(max_fires_per_window=1)):
        scheduler_mgr._record_cron_result("frequent", success=True)
        scheduler_mgr._record_cron_result("frequent", success=True)

    scheduler_mgr._anima._activity.log.assert_called_once()
    call = scheduler_mgr._anima._activity.log.call_args
    assert call.args[0] == "cron_guard_warning"
    assert call.kwargs["meta"]["task_name"] == "frequent"
    assert len(_stats(scheduler_mgr)["frequent"]["fire_timestamps"]) == 2  # type: ignore[index]
    assert len(list((scheduler_mgr._anima_dir / "state" / "background_notifications").glob("cron_guard_*.md"))) == 1


def test_consecutive_failure_threshold_records_warning(scheduler_mgr: SchedulerManager) -> None:
    config = _config(max_fires_per_window=100, max_consecutive_failures=2)
    with patch("core.supervisor.scheduler_manager.load_config", return_value=config):
        scheduler_mgr._record_cron_result("failing", success=False)
        scheduler_mgr._record_cron_result("failing", success=False)

    call = scheduler_mgr._anima._activity.log.call_args
    assert call.args[0] == "cron_guard_warning"
    assert call.kwargs["meta"]["stats"]["consecutive_failures"] == 2


def test_normal_defaults_and_off_mode_do_not_evaluate(scheduler_mgr: SchedulerManager) -> None:
    with patch("core.supervisor.scheduler_manager.load_config", return_value=_config()):
        for _ in range(3):
            scheduler_mgr._record_cron_result("normal", success=True)
    with patch(
        "core.supervisor.scheduler_manager.load_config",
        return_value=_config(mode="off", max_fires_per_window=1, max_consecutive_failures=1),
    ):
        scheduler_mgr._record_cron_result("off", success=False)
        scheduler_mgr._record_cron_result("off", success=False)

    scheduler_mgr._anima._activity.log.assert_not_called()
    assert _stats(scheduler_mgr)["off"]["consecutive_failures"] == 2  # type: ignore[index]


def test_warning_is_suppressed_for_24_hours(scheduler_mgr: SchedulerManager) -> None:
    with patch("core.supervisor.scheduler_manager.load_config", return_value=_config(max_fires_per_window=1)):
        scheduler_mgr._record_cron_result("frequent", success=True)
        scheduler_mgr._record_cron_result("frequent", success=True)
        scheduler_mgr._record_cron_result("frequent", success=True)

    scheduler_mgr._anima._activity.log.assert_called_once()
    assert len(list((scheduler_mgr._anima_dir / "state" / "background_notifications").glob("cron_guard_*.md"))) == 1


def test_recent_run_history_is_limited_to_ten(scheduler_mgr: SchedulerManager) -> None:
    with patch("core.supervisor.scheduler_manager.load_config", return_value=_config(mode="off")):
        for _ in range(12):
            scheduler_mgr._record_cron_result("history", success=True)

    assert len(_stats(scheduler_mgr)["history"]["recent_runs"]) == 10  # type: ignore[index]


@pytest.mark.asyncio
async def test_corrupt_stats_fail_open_and_are_rebuilt(scheduler_mgr: SchedulerManager) -> None:
    stats_path = scheduler_mgr._anima_dir / "state" / "cron_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text("{broken", encoding="utf-8")
    result = MagicMock()
    result.action = "completed"
    result.usage = {"input_tokens": 12, "output_tokens": 3}
    result.model_dump.return_value = {"summary": "done"}
    scheduler_mgr._anima.run_cron_task.return_value = result

    with patch("core.supervisor.scheduler_manager.load_config", return_value=_config()):
        await scheduler_mgr._run_cron_task(_task("recovered"))

    recovered = _stats(scheduler_mgr)["recovered"]  # type: ignore[index]
    assert recovered["consecutive_failures"] == 0
    assert recovered["recent_runs"][0]["usage"] == {"input_tokens": 12, "output_tokens": 3}


def test_disable_skip_cli_enable_and_reregister(
    scheduler_mgr: SchedulerManager,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _task("unstable")
    job = SimpleNamespace(id="test_cron_0", args=[task])
    scheduler_mgr.scheduler.get_jobs.return_value = [job]

    with patch(
        "core.supervisor.scheduler_manager.load_config",
        return_value=_config(mode="disable", max_consecutive_failures=1),
    ):
        scheduler_mgr._record_cron_result(task.name, success=False)

    disabled_path = scheduler_mgr._anima_dir / "state" / "cron_disabled.json"
    disabled = json.loads(disabled_path.read_text(encoding="utf-8"))
    assert disabled[task.name]["reason"]
    assert disabled[task.name]["stats"]["consecutive_failures"] == 1
    scheduler_mgr.scheduler.remove_job.assert_called_once_with("test_cron_0")
    event = scheduler_mgr._anima._activity.log.call_args
    assert event.args[0] == "cron_auto_disabled"

    cron_config = "## unstable\nschedule: */5 * * * *\ntype: llm\nRun guarded work\n"
    skipped_manager = SchedulerManager(scheduler_mgr._anima, "test", scheduler_mgr._anima_dir, MagicMock())
    skipped_manager.scheduler = MagicMock()
    skipped_manager._anima.memory.read_cron_config.return_value = cron_config
    skipped_manager._setup_cron_tasks()
    skipped_manager.scheduler.add_job.assert_not_called()

    with patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"):
        cmd_cron_guard_list(argparse.Namespace(anima="test"))
        assert "unstable" in capsys.readouterr().out
        cmd_cron_guard_enable(argparse.Namespace(anima="test", task="unstable"))
    assert task.name not in json.loads(disabled_path.read_text(encoding="utf-8"))

    restored_manager = SchedulerManager(scheduler_mgr._anima, "test", scheduler_mgr._anima_dir, MagicMock())
    restored_manager.scheduler = MagicMock()
    restored_manager._anima.memory.read_cron_config.return_value = cron_config
    restored_manager._setup_cron_tasks()
    restored_manager.scheduler.add_job.assert_called_once()
