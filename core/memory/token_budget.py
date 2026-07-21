from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared monthly token-budget status calculation."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.time_utils import now_local


@dataclass(frozen=True)
class TokenBudgetStatus:
    """Current monthly consumption relative to an optional budget."""

    budget: int | None
    consumed: int
    remaining: int | None
    exceeded: bool


def calculate_token_budget_status(
    budget: int | None,
    consumed: int,
) -> TokenBudgetStatus:
    """Build a consistent budget status from a configured limit and usage."""
    return TokenBudgetStatus(
        budget=budget,
        consumed=consumed,
        remaining=None if budget is None else max(budget - consumed, 0),
        # A cycle starting at the limit would necessarily exceed the cap.
        exceeded=budget is not None and consumed >= budget,
    )


def read_token_budget_status(
    anima_dir: Path,
    now: datetime | None = None,
) -> TokenBudgetStatus:
    """Load an Anima's resolved budget and current monthly consumption.

    This reporting helper intentionally reads consumption even when the budget
    is unlimited.  Runtime gates must first check for ``budget is None`` and
    avoid this helper so unlimited Animas incur no aggregation I/O.
    """
    from core.config.model_config import load_model_config
    from core.memory.token_usage import TokenUsageLogger

    model_config = load_model_config(anima_dir)
    consumed = TokenUsageLogger(anima_dir).monthly_total(now or now_local())
    return calculate_token_budget_status(model_config.token_budget_monthly, consumed)
