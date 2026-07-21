from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command: ``animaworks cost`` — token usage and cost estimation."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def cmd_cost(args: argparse.Namespace) -> None:
    """Show token usage and estimated cost."""
    from core.memory.token_usage import TokenUsageLogger
    from core.paths import get_data_dir

    data_dir = get_data_dir()
    animas_dir = data_dir / "animas"

    days = 1 if args.today else args.days

    if args.anima:
        anima_dir = animas_dir / args.anima
        if not anima_dir.is_dir():
            print(f"Error: Anima '{args.anima}' not found at {anima_dir}")
            sys.exit(1)
        _show_anima_cost(args.anima, anima_dir, days, args.json_output)
    else:
        anima_dirs = sorted(p for p in animas_dir.iterdir() if p.is_dir() and _has_usage_or_budget(p))
        if not anima_dirs:
            print("No token usage data found. Start the server and send some messages.")
            return

        if args.json_output:
            result: dict = {}
            for ad in anima_dirs:
                logger = TokenUsageLogger(ad)
                summary = logger.summarize(days)
                summary.update(_read_budget_fields(ad))
                result[ad.name] = summary
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            grand_total_cost = 0.0
            grand_total_input = 0
            grand_total_output = 0
            grand_total_cache_read = 0
            grand_total_cache_write = 0
            grand_total_sessions = 0

            for ad in anima_dirs:
                logger = TokenUsageLogger(ad)
                summary = logger.summarize(days)
                budget_fields = _read_budget_fields(ad)
                if summary["total_sessions"] == 0 and budget_fields["budget"] is None:
                    continue
                grand_total_cost += summary["total_estimated_cost_usd"]
                grand_total_input += summary["total_input_tokens"]
                grand_total_output += summary["total_output_tokens"]
                grand_total_cache_read += summary.get("total_cache_read_tokens", 0)
                grand_total_cache_write += summary.get("total_cache_write_tokens", 0)
                grand_total_sessions += summary["total_sessions"]
                _print_anima_summary(ad.name, summary, budget_fields)

            if grand_total_sessions > 0:
                print("=" * 60)
                print(
                    f"{'TOTAL':>20s}  {grand_total_sessions:>6d} sessions  "
                    f"in:{_fmt_tokens(grand_total_input):>10s}  "
                    f"CacheR:{_fmt_tokens(grand_total_cache_read):>10s}  "
                    f"CacheW:{_fmt_tokens(grand_total_cache_write):>10s}  "
                    f"out:{_fmt_tokens(grand_total_output):>10s}  "
                    f"${grand_total_cost:.4f}"
                )
                print()


def _show_anima_cost(name: str, anima_dir: Path, days: int, json_output: bool) -> None:
    from core.memory.token_usage import TokenUsageLogger

    logger = TokenUsageLogger(anima_dir)
    summary = logger.summarize(days)
    budget_fields = _read_budget_fields(anima_dir)

    if json_output:
        summary.update(budget_fields)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if summary["total_sessions"] == 0:
        print(f"No token usage data for '{name}' in the last {days} day(s).")
        _print_budget_summary(budget_fields)
        return

    _print_anima_summary(name, summary, budget_fields)

    by_model = summary.get("by_model", {})
    if by_model:
        print(
            f"  {'Model':<35s} {'Sessions':>8s} {'Input':>10s} {'CacheR':>10s} {'CacheW':>10s} "
            f"{'Output':>10s} {'Cost':>10s}"
        )
        print(f"  {'-' * 35} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
        for model, data in sorted(by_model.items()):
            print(
                f"  {model:<35s} {data['sessions']:>8d} "
                f"{_fmt_tokens(data.get('input_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('cache_read_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('cache_write_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('output_tokens', 0)):>10s} "
                f"${data['cost_usd']:.4f}"
            )
        print()

    by_trigger = summary.get("by_trigger", {})
    if by_trigger:
        print(
            f"  {'Trigger':<20s} {'Sessions':>8s} {'Input':>10s} {'CacheR':>10s} {'CacheW':>10s} "
            f"{'Output':>10s} {'Cost':>10s}"
        )
        print(f"  {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
        for trigger, data in sorted(by_trigger.items()):
            print(
                f"  {trigger:<20s} {data['sessions']:>8d} "
                f"{_fmt_tokens(data.get('input_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('cache_read_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('cache_write_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('output_tokens', 0)):>10s} "
                f"${data['cost_usd']:.4f}"
            )
        print()

    by_date = summary.get("by_date", {})
    if by_date:
        print(
            f"  {'Date':<12s} {'Sessions':>8s} {'Input':>10s} {'CacheR':>10s} {'CacheW':>10s} "
            f"{'Output':>10s} {'Cost':>10s}"
        )
        print(f"  {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
        for day, data in sorted(by_date.items()):
            print(
                f"  {day:<12s} {data['sessions']:>8d} "
                f"{_fmt_tokens(data.get('input_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('cache_read_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('cache_write_tokens', 0)):>10s} "
                f"{_fmt_tokens(data.get('output_tokens', 0)):>10s} "
                f"${data['cost_usd']:.4f}"
            )
        print()


def _read_budget_fields(anima_dir: Path, *, now: datetime | None = None) -> dict[str, object]:
    """Return JSON-ready current-month budget fields for one Anima."""
    from core.memory.token_budget import read_token_budget_status
    from core.time_utils import now_local

    current = now or now_local()
    status = read_token_budget_status(anima_dir, now=current)
    return {
        "month": current.strftime("%Y-%m"),
        "budget": status.budget,
        "consumed": status.consumed,
        "remaining": status.remaining,
        "exceeded": status.exceeded,
    }


def _has_usage_or_budget(anima_dir: Path) -> bool:
    """Keep legacy usage rows while including configured zero-usage budgets."""
    if (anima_dir / "token_usage").is_dir():
        return True
    try:
        from core.config.model_config import load_model_config

        return load_model_config(anima_dir).token_budget_monthly is not None
    except Exception:
        return False


def _print_budget_summary(budget_fields: dict[str, object]) -> None:
    budget = budget_fields["budget"]
    remaining = budget_fields["remaining"]
    budget_text = "-" if budget is None else _fmt_tokens(int(budget))
    remaining_text = "-" if remaining is None else _fmt_tokens(int(remaining))
    exceeded_text = "yes" if budget_fields["exceeded"] else "no"
    print(
        f"  Monthly budget: {budget_text}   "
        f"Consumed: {_fmt_tokens(int(budget_fields['consumed']))}   "
        f"Remaining: {remaining_text}   Exceeded: {exceeded_text}"
    )


def _print_anima_summary(name: str, summary: dict, budget_fields: dict[str, object]) -> None:
    print(f"\n  {name}")
    print(f"  {'─' * 56}")
    cache_r = summary.get("total_cache_read_tokens", 0)
    cache_w = summary.get("total_cache_write_tokens", 0)
    print(
        f"  Sessions: {summary['total_sessions']:,d}   "
        f"Input: {_fmt_tokens(summary['total_input_tokens'])}   "
        f"CacheR: {_fmt_tokens(cache_r)}   "
        f"CacheW: {_fmt_tokens(cache_w)}   "
        f"Output: {_fmt_tokens(summary['total_output_tokens'])}"
    )
    print(f"  Estimated cost: ${summary['total_estimated_cost_usd']:.4f}")
    _print_budget_summary(budget_fields)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
