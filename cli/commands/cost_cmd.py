from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command: ``animaworks cost`` — token usage and cost estimation."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def cmd_cost(args: argparse.Namespace) -> None:
    """Show token usage and estimated cost."""
    from core.paths import get_data_dir
    from core.memory.token_usage import TokenUsageLogger

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
        anima_dirs = sorted(
            p for p in animas_dir.iterdir()
            if p.is_dir() and (p / "token_usage").is_dir()
        )
        if not anima_dirs:
            print("No token usage data found. Start the server and send some messages.")
            return

        if args.json_output:
            result: dict = {}
            for ad in anima_dirs:
                logger = TokenUsageLogger(ad)
                result[ad.name] = logger.summarize(days)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            grand_total_cost = 0.0
            grand_total_input = 0
            grand_total_output = 0
            grand_total_sessions = 0

            for ad in anima_dirs:
                logger = TokenUsageLogger(ad)
                summary = logger.summarize(days)
                if summary["total_sessions"] == 0:
                    continue
                grand_total_cost += summary["total_estimated_cost_usd"]
                grand_total_input += summary["total_input_tokens"]
                grand_total_output += summary["total_output_tokens"]
                grand_total_sessions += summary["total_sessions"]
                _print_anima_summary(ad.name, summary)

            if grand_total_sessions > 0:
                print("=" * 60)
                print(f"{'TOTAL':>20s}  {grand_total_sessions:>6d} sessions  "
                      f"in:{_fmt_tokens(grand_total_input):>10s}  "
                      f"out:{_fmt_tokens(grand_total_output):>10s}  "
                      f"${grand_total_cost:.4f}")
                print()


def _show_anima_cost(name: str, anima_dir: Path, days: int, json_output: bool) -> None:
    from core.memory.token_usage import TokenUsageLogger

    logger = TokenUsageLogger(anima_dir)
    summary = logger.summarize(days)

    if json_output:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if summary["total_sessions"] == 0:
        print(f"No token usage data for '{name}' in the last {days} day(s).")
        return

    _print_anima_summary(name, summary)

    by_model = summary.get("by_model", {})
    if by_model:
        print(f"  {'Model':<35s} {'Sessions':>8s} {'Input':>10s} {'Output':>10s} {'Cost':>10s}")
        print(f"  {'-' * 35} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
        for model, data in sorted(by_model.items()):
            print(f"  {model:<35s} {data['sessions']:>8d} "
                  f"{_fmt_tokens(data['input_tokens']):>10s} "
                  f"{_fmt_tokens(data['output_tokens']):>10s} "
                  f"${data['cost_usd']:.4f}")
        print()

    by_trigger = summary.get("by_trigger", {})
    if by_trigger:
        print(f"  {'Trigger':<20s} {'Sessions':>8s} {'Input':>10s} {'Output':>10s} {'Cost':>10s}")
        print(f"  {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
        for trigger, data in sorted(by_trigger.items()):
            print(f"  {trigger:<20s} {data['sessions']:>8d} "
                  f"{_fmt_tokens(data['input_tokens']):>10s} "
                  f"{_fmt_tokens(data['output_tokens']):>10s} "
                  f"${data['cost_usd']:.4f}")
        print()

    by_date = summary.get("by_date", {})
    if by_date:
        print(f"  {'Date':<12s} {'Sessions':>8s} {'Input':>10s} {'Output':>10s} {'Cost':>10s}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
        for day, data in sorted(by_date.items()):
            print(f"  {day:<12s} {data['sessions']:>8d} "
                  f"{_fmt_tokens(data['input_tokens']):>10s} "
                  f"{_fmt_tokens(data['output_tokens']):>10s} "
                  f"${data['cost_usd']:.4f}")
        print()


def _print_anima_summary(name: str, summary: dict) -> None:
    print(f"\n  {name}")
    print(f"  {'─' * 56}")
    print(f"  Sessions: {summary['total_sessions']:,d}   "
          f"Input: {_fmt_tokens(summary['total_input_tokens'])}   "
          f"Output: {_fmt_tokens(summary['total_output_tokens'])}   "
          f"Total: {_fmt_tokens(summary['total_tokens'])}")
    print(f"  Estimated cost: ${summary['total_estimated_cost_usd']:.4f}")


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
