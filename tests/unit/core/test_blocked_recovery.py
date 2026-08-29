from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core._anima_heartbeat import HeartbeatMixin
from core.blocked_recovery import revalidate_blocked_tasks
from core.config.schemas import BackgroundTaskConfig
from core.memory.task_queue import TaskQueueManager


@pytest.fixture(autouse=True)
def _no_ssh_dropin_tmpfs():
    # Host-dependent bwrap args (see foreign_owned_ssh_config_dirs) would
    # otherwise leak into the exact-argv assertions below.
    with patch("core.file_access_policy.foreign_owned_ssh_config_dirs", return_value=[]):
        yield
from core.time_utils import now_local


def _config(**overrides):
    defaults = {
        "blocked_recovery_enabled": True,
        "blocked_reprobe_after_hours": 6.0,
        "blocked_reprobe_batch_limit": 3,
        "blocked_max_reprobes": 4,
        "blocked_check_timeout_seconds": 60,
        "blocked_checkless_reprobe_enabled": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(background_task=SimpleNamespace(**defaults))


def _blocked_task(anima_dir: Path, *, task_id: str, meta: dict) -> TaskQueueManager:
    manager = TaskQueueManager(anima_dir)
    manager.add_task(
        source="human",
        original_instruction="finish the task",
        assignee=anima_dir.name,
        summary="finish task",
        task_id=task_id,
        meta=meta,
    )
    manager.update_status(task_id, "blocked", summary="waiting")
    return manager


def _heartbeat(anima_dir: Path) -> HeartbeatMixin:
    heartbeat = HeartbeatMixin()
    heartbeat.anima_dir = anima_dir
    heartbeat.name = anima_dir.name
    heartbeat.memory = SimpleNamespace(read_heartbeat_config=lambda: "checklist")
    heartbeat._build_state_cleanup_instruction = lambda: None
    heartbeat._build_background_context_parts = lambda: []
    return heartbeat


def test_check_success_republishes_without_consuming_retry(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id="check-pass",
        meta={
            "unblock_check": "test -w .",
            "retry_count": 3,
        },
    )

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch(
            "core.blocked_recovery.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run,
    ):
        result = revalidate_blocked_tasks(anima_dir, "worker")

    assert result == ["check-pass"]
    current = manager.get_task_by_id("check-pass")
    assert current is not None
    assert current.status == "pending"
    assert current.meta["retry_count"] == 3
    pending = anima_dir / "state" / "pending" / "check-pass.json"
    assert json.loads(pending.read_text(encoding="utf-8"))["description"] == "finish the task"
    assert run.call_args.args[0] == [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--die-with-parent",
        "--",
        "/bin/bash",
        "-c",
        "test -w .",
    ]
    kwargs = run.call_args.kwargs
    assert kwargs["cwd"] == anima_dir
    assert kwargs["timeout"] == 60
    assert set(kwargs["env"]) == {"PATH", "HOME", "ANIMAWORKS_ANIMA_DIR"}
    assert kwargs["env"]["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)


def test_recovery_batch_limit_uses_oldest_blocked_tasks(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    base = now_local() - timedelta(days=1)
    for index in range(5):
        _blocked_task(
            anima_dir,
            task_id=f"task-{index}",
            meta={"blocked_at": (base + timedelta(hours=index)).isoformat(), "unblock_check": "true"},
        )

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch(
            "core.blocked_recovery.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == ["task-0", "task-1", "task-2"]

    manager = TaskQueueManager(anima_dir)
    assert [manager.get_task_by_id(f"task-{index}").status for index in range(5)] == [
        "pending",
        "pending",
        "pending",
        "blocked",
        "blocked",
    ]
    activity_files = list((anima_dir / "activity_log").glob("*.jsonl"))
    events = [json.loads(line) for line in activity_files[0].read_text(encoding="utf-8").splitlines()]
    assert [event["meta"]["task_id"] for event in events] == ["task-0", "task-1", "task-2"]
    assert all(event["type"] == "blocked_recovery" for event in events)


def test_blocked_reprobe_batch_limit_default_and_validation() -> None:
    assert BackgroundTaskConfig().blocked_reprobe_batch_limit == 3
    with pytest.raises(ValueError):
        BackgroundTaskConfig(blocked_reprobe_batch_limit=0)


def test_missing_bwrap_fails_closed_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id="no-bwrap",
        meta={
            "unblock_check": "touch escaped",
            "task_desc": {"title": "finish"},
        },
    )

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch("core.blocked_recovery.subprocess.run", side_effect=FileNotFoundError("bwrap")),
        caplog.at_level("WARNING", logger="animaworks.blocked_recovery"),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    current = manager.get_task_by_id("no-bwrap")
    assert current is not None
    assert current.status == "blocked"
    assert current.meta["unblock_check_failures"] == 1
    assert not (anima_dir / "escaped").exists()
    assert "sandbox unavailable" in caplog.text


def test_taskboard_suppression_skips_recovery(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id="suppressed",
        meta={"unblock_check": "true", "task_desc": {"title": "finish"}},
    )
    should_execute = Mock(return_value=SimpleNamespace(executable=False))
    resolver = SimpleNamespace(should_execute=should_execute)

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch("core.taskboard.attention_resolver.resolver_for_anima_dir", return_value=resolver) as factory,
        patch("core.blocked_recovery.subprocess.run") as run,
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    factory.assert_called_once_with(anima_dir)
    should_execute.assert_called_once_with("worker", "suppressed", queue_status="pending")
    run.assert_not_called()
    current = manager.get_task_by_id("suppressed")
    assert current is not None
    assert current.status == "blocked"


def test_publish_failure_restores_blocked_status(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id="publish-fails",
        meta={"unblock_check": "true"},
    )

    def fail_publish(*_args, **_kwargs):
        assert manager.get_task_by_id("publish-fails").status == "pending"
        raise OSError("read-only filesystem")

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch("core.blocked_recovery.regenerate_pending_json", side_effect=fail_publish),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    assert manager.get_task_by_id("publish-fails").status == "blocked"


async def test_heartbeat_does_not_revalidate_blocked_tasks(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    anima_dir.mkdir(parents=True)

    with (
        patch("core.blocked_recovery.revalidate_blocked_tasks") as revalidate,
        patch("core._anima_heartbeat.load_prompt", return_value="heartbeat"),
        patch("core._anima_heartbeat._build_curator_review_part", return_value=None),
    ):
        assert await _heartbeat(anima_dir)._build_heartbeat_prompt() == ["heartbeat"]
    revalidate.assert_not_called()


def test_checkless_task_waits_for_reprobe_interval(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id="too-new",
        meta={
            "blocked_at": now_local().isoformat(),
            "unblock_check": "   ",
            "task_desc": {"title": "finish"},
        },
    )

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch("core.blocked_recovery.subprocess.run") as run,
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    run.assert_not_called()
    current = manager.get_task_by_id("too-new")
    assert current is not None
    assert current.status == "blocked"
    assert "blocked_reprobe_count" not in current.meta


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_failed_check_stays_blocked_and_counts_failure(tmp_path: Path, failure: str) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id=f"check-{failure}",
        meta={
            "unblock_check": "false",
            "task_desc": {"title": "finish"},
        },
    )
    outcome = subprocess.CompletedProcess([], 1) if failure == "nonzero" else subprocess.TimeoutExpired("false", 60)

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch(
            "core.blocked_recovery.subprocess.run",
            return_value=outcome if failure == "nonzero" else None,
            side_effect=outcome if failure == "timeout" else None,
        ),
    ):
        result = revalidate_blocked_tasks(anima_dir, "worker")

    current = manager.get_task_by_id(f"check-{failure}")
    assert result == []
    assert current is not None
    assert current.status == "blocked"
    assert current.meta["unblock_check_failures"] == 1
    assert not (anima_dir / "state" / "pending" / f"check-{failure}.json").exists()


def test_failing_check_hands_task_back_for_self_review(tmp_path: Path) -> None:
    """A long-failing check goes back to the anima to re-judge, not to a human."""
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "worker"
    supervisor_dir = animas_dir / "boss"
    supervisor_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text('{"supervisor": "boss"}', encoding="utf-8")
    manager = _blocked_task(
        anima_dir,
        task_id="stale-check",
        meta={
            "blocked_at": (now_local() - timedelta(hours=7)).isoformat(),
            "unblock_check": "test 1 = 2",
            "task_desc": {"title": "finish", "description": "finish the task"},
        },
    )

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch(
            "core.blocked_recovery.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1),
        ),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == ["stale-check"]

    current = manager.get_task_by_id("stale-check")
    assert current is not None
    assert current.status == "pending"
    assert current.meta["unblock_check_failures"] == 1
    assert current.meta["blocked_reprobe_count"] == 1
    assert current.meta.get("blocked_recovery_alerted") is None
    assert not TaskQueueManager(supervisor_dir).list_tasks()
    description = json.loads((anima_dir / "state" / "pending" / "stale-check.json").read_text(encoding="utf-8"))[
        "description"
    ]
    assert "unblock_check" in description


def test_failing_check_alerts_after_self_reviews_are_used_up(tmp_path: Path) -> None:
    """Escalate only once the anima has had its bounded attempts to fix itself."""
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "worker"
    supervisor_dir = animas_dir / "boss"
    supervisor_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text('{"supervisor": "boss"}', encoding="utf-8")
    manager = _blocked_task(
        anima_dir,
        task_id="dead-check",
        meta={
            "blocked_at": (now_local() - timedelta(hours=7)).isoformat(),
            "unblock_check": "test 1 = 2",
            "blocked_reprobe_count": 4,
        },
    )

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch(
            "core.blocked_recovery.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1),
        ),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    current = manager.get_task_by_id("dead-check")
    assert current is not None
    assert current.status == "blocked"
    assert current.meta.get("blocked_recovery_alerted") is True
    alerts = [
        task
        for task in TaskQueueManager(supervisor_dir).list_tasks()
        if task.meta.get("kind") == "blocked_task_manual_intervention_required"
    ]
    assert len(alerts) == 1


def test_fresh_failing_check_does_not_alert(tmp_path: Path) -> None:
    """Below the threshold the check just retries; no supervisor noise."""
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "worker"
    supervisor_dir = animas_dir / "boss"
    supervisor_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text('{"supervisor": "boss"}', encoding="utf-8")
    manager = _blocked_task(
        anima_dir,
        task_id="fresh-check",
        meta={
            "blocked_at": (now_local() - timedelta(hours=1)).isoformat(),
            "unblock_check": "test 1 = 2",
        },
    )

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch(
            "core.blocked_recovery.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1),
        ),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    current = manager.get_task_by_id("fresh-check")
    assert current is not None
    assert current.meta.get("blocked_recovery_alerted") is None
    assert not TaskQueueManager(supervisor_dir).list_tasks()


def test_checkless_task_stays_blocked_and_alerts_once(tmp_path: Path) -> None:
    """Fail closed: checkless tasks never auto-reprobe; alert once past threshold."""
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "worker"
    supervisor_dir = animas_dir / "boss"
    supervisor_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text('{"supervisor": "boss"}', encoding="utf-8")
    manager = _blocked_task(
        anima_dir,
        task_id="no-check",
        meta={
            "blocked_at": (now_local() - timedelta(hours=7)).isoformat(),
            "task_desc": {"title": "finish", "description": "finish the task"},
        },
    )

    with patch("core.config.models.load_config", return_value=_config()):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []
        assert revalidate_blocked_tasks(anima_dir, "worker") == []
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    current = manager.get_task_by_id("no-check")
    assert current is not None
    assert current.status == "blocked"
    assert current.meta.get("blocked_recovery_alerted") is True
    assert "blocked_reprobe_count" not in current.meta
    assert not (anima_dir / "state" / "pending" / "no-check.json").exists()
    alerts = [
        task
        for task in TaskQueueManager(supervisor_dir).list_tasks()
        if task.meta.get("kind") == "blocked_task_manual_intervention_required"
    ]
    assert len(alerts) == 1
    assert "unblock_check を持たない" in alerts[0].original_instruction


def test_checkless_legacy_task_uses_updated_at_for_alert_only(tmp_path: Path) -> None:
    """Legacy checkless (no blocked_at) falls back to updated_at; still fail closed."""
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "worker"
    supervisor_dir = animas_dir / "boss"
    supervisor_dir.mkdir(parents=True)
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text('{"supervisor": "boss"}', encoding="utf-8")
    manager = _blocked_task(
        anima_dir,
        task_id="legacy-blocked",
        meta={"task_desc": {"title": "finish"}},
    )
    entry = manager.get_task_by_id("legacy-blocked")
    assert entry is not None
    future = datetime.fromisoformat(entry.updated_at) + timedelta(hours=7)

    with (
        patch("core.config.models.load_config", return_value=_config()),
        patch("core.blocked_recovery.now_local", return_value=future),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    current = manager.get_task_by_id("legacy-blocked")
    assert current is not None
    assert current.status == "blocked"
    assert current.meta.get("blocked_recovery_alerted") is True
    assert "blocked_reprobe_count" not in current.meta
    assert not (anima_dir / "state" / "pending" / "legacy-blocked.json").exists()
    alerts = [
        task
        for task in TaskQueueManager(supervisor_dir).list_tasks()
        if task.meta.get("kind") == "blocked_task_manual_intervention_required"
    ]
    assert len(alerts) == 1


def test_checkless_reprobe_enabled_restores_legacy_time_based_reprobe(
    tmp_path: Path,
) -> None:
    """blocked_checkless_reprobe_enabled=True restores pre-fail-closed behavior."""
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id="legacy-on",
        meta={
            "blocked_at": (now_local() - timedelta(hours=7)).isoformat(),
            "task_desc": {"title": "finish", "description": "finish the task"},
        },
    )

    with patch(
        "core.config.models.load_config",
        return_value=_config(blocked_checkless_reprobe_enabled=True),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == ["legacy-on"]

    current = manager.get_task_by_id("legacy-on")
    assert current is not None
    assert current.status == "pending"
    assert current.meta["blocked_reprobe_count"] == 1
    desc = json.loads((anima_dir / "state" / "pending" / "legacy-on.json").read_text(encoding="utf-8"))["description"]
    assert "blockerが解消済みか確認" in desc


def test_checkless_many_tasks_generate_zero_pending_json(tmp_path: Path) -> None:
    """Many checkless blocked tasks must not produce any pending JSON under fail closed."""
    anima_dir = tmp_path / "animas" / "worker"
    for index in range(50):
        _blocked_task(
            anima_dir,
            task_id=f"checkless-{index:02d}",
            meta={
                "blocked_at": (now_local() - timedelta(hours=24)).isoformat(),
                "task_desc": {"title": f"finish-{index}"},
            },
        )

    with patch("core.config.models.load_config", return_value=_config()):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    pending_dir = anima_dir / "state" / "pending"
    pending_files = list(pending_dir.glob("*.json")) if pending_dir.is_dir() else []
    assert pending_files == []
    manager = TaskQueueManager(anima_dir)
    for index in range(50):
        current = manager.get_task_by_id(f"checkless-{index:02d}")
        assert current is not None
        assert current.status == "blocked"


def test_blocked_checkless_reprobe_enabled_default_is_true() -> None:
    """Self review before escalation: the anima gets bounded attempts of its own."""
    assert BackgroundTaskConfig().blocked_checkless_reprobe_enabled is True


def test_recovery_can_be_disabled(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "worker"
    manager = _blocked_task(
        anima_dir,
        task_id="disabled",
        meta={"unblock_check": "true", "task_desc": {"title": "finish"}},
    )

    with patch(
        "core.config.models.load_config",
        return_value=_config(blocked_recovery_enabled=False),
    ):
        assert revalidate_blocked_tasks(anima_dir, "worker") == []

    current = manager.get_task_by_id("disabled")
    assert current is not None
    assert current.status == "blocked"
