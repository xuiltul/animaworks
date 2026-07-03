from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""F22: housekeeping scheduler guard + config field.

- ``archive_superseded_retention_days`` is now configurable (was hardcoded 7).
- The cron run and the startup catch-up cannot run housekeeping concurrently.
"""

import asyncio

import pytest

from core.config.schemas import HousekeepingConfig
from core.supervisor._mgr_scheduler import SchedulerMixin


def test_archive_superseded_retention_days_default() -> None:
    assert HousekeepingConfig().archive_superseded_retention_days == 7


def test_archive_superseded_retention_days_configurable() -> None:
    cfg = HousekeepingConfig.model_validate({"archive_superseded_retention_days": 14})
    assert cfg.archive_superseded_retention_days == 14


class _GuardHarness(SchedulerMixin):
    """Minimal SchedulerMixin subclass exercising only the housekeeping guard."""

    def __init__(self) -> None:
        self.impl_calls = 0

    async def _run_housekeeping_impl(self) -> None:  # type: ignore[override]
        self.impl_calls += 1
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_concurrent_housekeeping_runs_are_mutually_exclusive() -> None:
    harness = _GuardHarness()
    # Cron run and catch-up firing together: the second must skip, not double-run.
    await asyncio.gather(harness._run_housekeeping(), harness._run_housekeeping())
    assert harness.impl_calls == 1


@pytest.mark.asyncio
async def test_housekeeping_lock_is_reused() -> None:
    harness = _GuardHarness()
    lock_a = harness._get_housekeeping_lock()
    lock_b = harness._get_housekeeping_lock()
    assert lock_a is lock_b


@pytest.mark.asyncio
async def test_sequential_housekeeping_runs_each_execute() -> None:
    harness = _GuardHarness()
    await harness._run_housekeeping()
    await harness._run_housekeeping()
    assert harness.impl_calls == 2
