from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Canonical LoCoMo benchmark protocol helpers."""

import argparse
from typing import Any

ERROR_RATE_INVALID_THRESHOLD = 0.02
PROTOCOL_VERSION = "locomo-integrity-v1"


def attach_standard_protocol_summary(summary: dict[str, Any], config: dict[str, Any]) -> None:
    """Attach the canonical LoCoMo comparison metric block to ``summary``."""
    cat5_summary = summary.get("cat5_excluded")
    if not isinstance(cat5_summary, dict):
        cat5_summary = {}
    answer_model = str(config.get("answer_model", "") or "")
    judge_model = str(config.get("judge_model", "") or "")
    judge_enabled = bool(config.get("judge_enabled", False))
    leakage_free = not bool(config.get("leakage_alias_map_enabled", False)) and not bool(
        config.get("category_branches_enabled", False)
    )
    judge_model_independent = bool(judge_model and answer_model and judge_model.casefold() != answer_model.casefold())
    error_rate = float(summary.get("error_rate", 0.0) or 0.0)
    primary_value = cat5_summary.get("overall_judge")
    scope_all = str(config.get("search_mode", "") or "") == "scope_all"
    cat5_excluded = bool(config.get("exclude_cat5", False))
    max_questions = int(config.get("max_questions", 0) or 0)
    summary["standard_protocol"] = {
        "primary_metric": "cat5_excluded.overall_judge",
        "primary_value": primary_value,
        "secondary_metric": "overall_f1",
        "secondary_value": summary.get("overall_f1"),
        "search_mode": config.get("search_mode"),
        "scope_all": scope_all,
        "cat5_excluded": cat5_excluded,
        "max_questions": max_questions,
        "judge_enabled": judge_enabled,
        "judge_model_independent": judge_model_independent,
        "leakage_free": leakage_free,
        "error_rate": error_rate,
        "invalid_due_to_error_rate": error_rate > ERROR_RATE_INVALID_THRESHOLD,
        "valid": (
            scope_all
            and cat5_excluded
            and int(config.get("conversations", 0) or 0) == 10
            and max_questions == 0
            and judge_enabled
            and judge_model_independent
            and leakage_free
            and error_rate <= ERROR_RATE_INVALID_THRESHOLD
        ),
    }


def apply_standard_protocol_args(args: argparse.Namespace) -> None:
    """Mutate ``args`` for the canonical protocol shortcut."""
    if not bool(getattr(args, "standard_protocol", False)):
        return
    args.conversations = 10
    args.mode = "scope_all"
    args.judge = True
    args.exclude_cat5 = True
    args.enable_locomo_alias = False
    args.enable_locomo_category_branches = False
