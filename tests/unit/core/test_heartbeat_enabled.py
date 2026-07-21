"""Tests for the per-Anima periodic heartbeat switch."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.config.models import (
    AnimaDefaults,
    AnimaModelConfig,
    AnimaWorksConfig,
    CredentialConfig,
    resolve_anima_config,
)
from core.lifecycle.scheduler import SchedulerMixin
from core.memory.config_reader import ConfigReader
from core.schemas import ModelConfig
from core.time_utils import get_app_timezone


class _SchedulerHarness(SchedulerMixin):
    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone=get_app_timezone())

    async def _heartbeat_wrapper(self, name: str) -> None:
        pass


def _anima(name: str, *, heartbeat_enabled: bool) -> MagicMock:
    anima = MagicMock()
    anima.name = name
    anima.memory.read_model_config.return_value = ModelConfig(heartbeat_enabled=heartbeat_enabled)
    anima.memory.read_heartbeat_config.return_value = ""
    return anima


def test_heartbeat_enabled_defaults_are_backward_compatible() -> None:
    assert AnimaDefaults().heartbeat_enabled is True
    assert AnimaModelConfig().heartbeat_enabled is None


@pytest.mark.parametrize("override", [False, True])
def test_heartbeat_enabled_per_anima_override_wins(override: bool) -> None:
    config = AnimaWorksConfig(
        anima_defaults=AnimaDefaults(heartbeat_enabled=not override),
        animas={"alice": AnimaModelConfig(heartbeat_enabled=override)},
    )

    resolved, _ = resolve_anima_config(config, "alice")

    assert resolved.heartbeat_enabled is override


def test_heartbeat_enabled_unspecified_falls_back_to_defaults() -> None:
    config = AnimaWorksConfig(
        anima_defaults=AnimaDefaults(heartbeat_enabled=False),
        animas={"alice": AnimaModelConfig()},
    )

    resolved, _ = resolve_anima_config(config, "alice")

    assert resolved.heartbeat_enabled is False


def test_heartbeat_enabled_status_json_override_is_resolved(tmp_path) -> None:
    (tmp_path / "status.json").write_text(json.dumps({"heartbeat_enabled": False}), encoding="utf-8")

    resolved, _ = resolve_anima_config(AnimaWorksConfig(), "alice", anima_dir=tmp_path)

    assert resolved.heartbeat_enabled is False


def test_heartbeat_enabled_propagates_to_runtime_model_config(tmp_path) -> None:
    resolved = AnimaDefaults(heartbeat_enabled=False)
    credential = CredentialConfig()
    config_path = MagicMock()
    config_path.exists.return_value = True

    with (
        patch("core.config.get_config_path", return_value=config_path),
        patch("core.config.load_config", return_value=AnimaWorksConfig()),
        patch("core.config.resolve_anima_config", return_value=(resolved, credential)),
        patch("core.config.resolve_execution_mode", return_value="A"),
    ):
        model_config = ConfigReader(tmp_path).read_model_config()

    assert model_config.heartbeat_enabled is False


def test_heartbeat_disable_skips_periodic_job() -> None:
    harness = _SchedulerHarness()
    anima = _anima("alice", heartbeat_enabled=False)

    with patch("core.lifecycle.scheduler.load_config", return_value=SimpleNamespace(heartbeat=SimpleNamespace(interval_minutes=30))):
        harness._setup_heartbeat(anima)

    assert harness.scheduler.get_job("alice_heartbeat") is None


def test_heartbeat_enabled_registers_periodic_job() -> None:
    harness = _SchedulerHarness()
    anima = _anima("alice", heartbeat_enabled=True)

    with patch("core.lifecycle.scheduler.load_config", return_value=SimpleNamespace(heartbeat=SimpleNamespace(interval_minutes=30))):
        harness._setup_heartbeat(anima)

    assert harness.scheduler.get_job("alice_heartbeat") is not None


def test_heartbeat_disable_removes_existing_periodic_job() -> None:
    harness = _SchedulerHarness()
    anima = _anima("alice", heartbeat_enabled=True)

    with patch("core.lifecycle.scheduler.load_config", return_value=SimpleNamespace(heartbeat=SimpleNamespace(interval_minutes=30))):
        harness._setup_heartbeat(anima)
        assert harness.scheduler.get_job("alice_heartbeat") is not None

        anima.memory.read_model_config.return_value = ModelConfig(heartbeat_enabled=False)
        harness._setup_heartbeat(anima)

    assert harness.scheduler.get_job("alice_heartbeat") is None


# ── Supervisor-mode scheduler (core/supervisor/scheduler_manager.py) ─────────


def _supervisor_mgr(tmp_path, *, heartbeat_enabled: bool):
    from core.supervisor.scheduler_manager import SchedulerManager

    anima = MagicMock()
    anima.memory.read_model_config.return_value = ModelConfig(heartbeat_enabled=heartbeat_enabled)
    anima.memory.read_heartbeat_config.return_value = "# heartbeat"
    mgr = SchedulerManager(anima, "alice", tmp_path, lambda _e, _d: None)
    mgr.scheduler = AsyncIOScheduler(timezone=get_app_timezone())
    return mgr, anima


def test_supervisor_heartbeat_disable_skips_periodic_job(tmp_path) -> None:
    mgr, _ = _supervisor_mgr(tmp_path, heartbeat_enabled=False)

    with patch(
        "core.supervisor.scheduler_manager.load_config",
        return_value=SimpleNamespace(heartbeat=SimpleNamespace(interval_minutes=30), activity_level=100),
    ):
        mgr._setup_heartbeat()

    assert mgr.scheduler.get_job("alice_heartbeat") is None


def test_supervisor_heartbeat_enabled_registers_periodic_job(tmp_path) -> None:
    mgr, _ = _supervisor_mgr(tmp_path, heartbeat_enabled=True)

    with patch(
        "core.supervisor.scheduler_manager.load_config",
        return_value=SimpleNamespace(heartbeat=SimpleNamespace(interval_minutes=30), activity_level=100),
    ):
        mgr._setup_heartbeat()

    assert mgr.scheduler.get_job("alice_heartbeat") is not None


def test_supervisor_heartbeat_disable_removes_existing_periodic_job(tmp_path) -> None:
    mgr, anima = _supervisor_mgr(tmp_path, heartbeat_enabled=True)

    with patch(
        "core.supervisor.scheduler_manager.load_config",
        return_value=SimpleNamespace(heartbeat=SimpleNamespace(interval_minutes=30), activity_level=100),
    ):
        mgr._setup_heartbeat()
        assert mgr.scheduler.get_job("alice_heartbeat") is not None

        anima.memory.read_model_config.return_value = ModelConfig(heartbeat_enabled=False)
        mgr._setup_heartbeat()

    assert mgr.scheduler.get_job("alice_heartbeat") is None
