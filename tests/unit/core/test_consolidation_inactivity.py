from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config.schemas import AnimaDefaults, AnimaWorksConfig, ConsolidationConfig
from core.lifecycle.system_consolidation import (
    SystemConsolidationMixin,
    has_recent_activity,
    is_consolidation_enabled,
    should_skip_inactive_consolidation,
)


def test_consolidation_config_defaults() -> None:
    assert AnimaDefaults().consolidation_enabled is True
    config = ConsolidationConfig()
    assert config.inactivity_skip_enabled is True
    assert config.inactivity_days == 7


def test_per_anima_consolidation_disabled(tmp_path, monkeypatch) -> None:
    (tmp_path / "status.json").write_text('{"consolidation_enabled": false}', encoding="utf-8")
    monkeypatch.setattr("core.lifecycle.system_consolidation.load_config", lambda: AnimaWorksConfig())

    assert is_consolidation_enabled(tmp_path) is False


def test_per_anima_disable_does_not_depend_on_credential_resolution(tmp_path, monkeypatch) -> None:
    (tmp_path / "status.json").write_text(
        '{"consolidation_enabled": false, "credential": "missing"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("core.lifecycle.system_consolidation.load_config", lambda: AnimaWorksConfig())

    assert is_consolidation_enabled(tmp_path) is False


def test_per_anima_disable_takes_priority_over_inactivity_toggle(tmp_path, monkeypatch, caplog) -> None:
    (tmp_path / "status.json").write_text('{"consolidation_enabled": false}', encoding="utf-8")
    monkeypatch.setattr("core.lifecycle.system_consolidation.load_config", lambda: AnimaWorksConfig())
    config = SimpleNamespace(inactivity_skip_enabled=False, inactivity_days=7)

    with caplog.at_level(logging.INFO, logger="animaworks.lifecycle"):
        skipped = should_skip_inactive_consolidation(tmp_path, "disabled", config)

    assert skipped is True
    assert "consolidation_enabled=false" in caplog.text


def test_recent_activity_uses_entry_timestamp_not_file_mtime(tmp_path) -> None:
    now = datetime(2026, 7, 12, tzinfo=UTC)
    log_dir = tmp_path / "activity_log"
    log_dir.mkdir()
    log_file = log_dir / "old-name.jsonl"
    log_file.write_text(json.dumps({"ts": (now - timedelta(days=2)).isoformat()}) + "\n", encoding="utf-8")

    assert has_recent_activity(tmp_path, days=7, now=now) is True


def test_activity_older_than_window_is_inactive(tmp_path) -> None:
    now = datetime(2026, 7, 12, tzinfo=UTC)
    log_dir = tmp_path / "activity_log"
    log_dir.mkdir()
    (log_dir / "2026-07-01.jsonl").write_text(
        json.dumps({"ts": (now - timedelta(days=8)).isoformat()}) + "\n",
        encoding="utf-8",
    )

    assert has_recent_activity(tmp_path, days=7, now=now) is False


def test_inactivity_guard_logs_skip(tmp_path, caplog) -> None:
    (tmp_path / "status.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(inactivity_skip_enabled=True, inactivity_days=7)

    with caplog.at_level(logging.INFO, logger="animaworks.lifecycle"):
        skipped = should_skip_inactive_consolidation(tmp_path, "sleepy", config)

    assert skipped is True
    assert "no activity_log entries in the last 7 days" in caplog.text


def test_inactivity_guard_can_be_disabled(tmp_path) -> None:
    (tmp_path / "status.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(inactivity_skip_enabled=False, inactivity_days=7)

    assert should_skip_inactive_consolidation(tmp_path, "sleepy", config) is False


def test_anima_default_consolidation_disable_is_resolved(tmp_path, monkeypatch) -> None:
    (tmp_path / "status.json").write_text("{}", encoding="utf-8")
    config = AnimaWorksConfig(anima_defaults=AnimaDefaults(consolidation_enabled=False))
    monkeypatch.setattr("core.lifecycle.system_consolidation.load_config", lambda: config)

    assert is_consolidation_enabled(tmp_path) is False


def test_status_override_wins_over_disabled_anima_default(tmp_path, monkeypatch) -> None:
    (tmp_path / "status.json").write_text('{"consolidation_enabled": true}', encoding="utf-8")
    config = AnimaWorksConfig(anima_defaults=AnimaDefaults(consolidation_enabled=False))
    monkeypatch.setattr("core.lifecycle.system_consolidation.load_config", lambda: config)

    assert is_consolidation_enabled(tmp_path) is True


@pytest.mark.asyncio
async def test_system_mixin_daily_and_weekly_skip_inactive_anima(tmp_path, monkeypatch) -> None:
    (tmp_path / "status.json").write_text("{}", encoding="utf-8")
    run_consolidation = AsyncMock()
    anima = SimpleNamespace(
        memory=SimpleNamespace(anima_dir=tmp_path),
        run_consolidation=run_consolidation,
    )
    runner = SystemConsolidationMixin()
    runner.animas = {"sleepy": anima}
    runner._ws_broadcast = None
    config = SimpleNamespace(
        consolidation=SimpleNamespace(
            daily_enabled=True,
            weekly_enabled=True,
            inactivity_skip_enabled=True,
            inactivity_days=7,
        )
    )
    monkeypatch.setattr("core.lifecycle.system_consolidation.load_config", lambda: config)

    await runner._handle_daily_consolidation()
    await runner._handle_weekly_integration()

    run_consolidation.assert_not_awaited()
