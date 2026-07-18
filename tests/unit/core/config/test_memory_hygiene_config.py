from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config.schemas import ConsolidationConfig, HousekeepingConfig


def test_memory_hygiene_config_defaults() -> None:
    consolidation = ConsolidationConfig()
    housekeeping = HousekeepingConfig()

    assert consolidation.episode_retention_batch_limit == 200
    assert housekeeping.shortterm_archive_retention_days == 30
    assert housekeeping.shortterm_thread_gc_days == 30
    assert housekeeping.facts_lock_stale_hours == 24


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shortterm_archive_retention_days", 0),
        ("shortterm_thread_gc_days", 0),
        ("facts_lock_stale_hours", 0),
    ],
)
def test_housekeeping_memory_hygiene_fields_require_positive_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        HousekeepingConfig(**{field: value})


def test_episode_retention_batch_limit_allows_zero_but_not_negative() -> None:
    assert ConsolidationConfig(episode_retention_batch_limit=0).episode_retention_batch_limit == 0
    with pytest.raises(ValidationError):
        ConsolidationConfig(episode_retention_batch_limit=-1)
