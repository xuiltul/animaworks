from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config.models import HousekeepingConfig


def test_taskboard_housekeeping_config_defaults() -> None:
    cfg = HousekeepingConfig()

    assert cfg.pending_processing_stale_hours == 24
    assert cfg.background_running_stale_hours == 48
    assert cfg.current_state_stale_hours == 24
    assert cfg.taskboard_suppressed_retention_days == 30
    assert cfg.taskboard_orphan_metadata_stale_hours == 24


def test_taskboard_housekeeping_config_custom_values() -> None:
    cfg = HousekeepingConfig(
        pending_processing_stale_hours=6,
        background_running_stale_hours=12,
        current_state_stale_hours=18,
        taskboard_suppressed_retention_days=45,
        taskboard_orphan_metadata_stale_hours=12,
    )

    assert cfg.pending_processing_stale_hours == 6
    assert cfg.background_running_stale_hours == 12
    assert cfg.current_state_stale_hours == 18
    assert cfg.taskboard_suppressed_retention_days == 45
    assert cfg.taskboard_orphan_metadata_stale_hours == 12


def test_destructive_taskboard_housekeeping_thresholds_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        HousekeepingConfig(pending_processing_stale_hours=0)
    with pytest.raises(ValidationError):
        HousekeepingConfig(background_running_stale_hours=0)
    with pytest.raises(ValidationError):
        HousekeepingConfig(current_state_stale_hours=0)
    with pytest.raises(ValidationError):
        HousekeepingConfig(taskboard_suppressed_retention_days=0)
    with pytest.raises(ValidationError):
        HousekeepingConfig(taskboard_orphan_metadata_stale_hours=0)
