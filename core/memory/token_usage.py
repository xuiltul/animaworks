from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Token usage logger — persistent per-session JSONL logging with cost estimation.

Writes to ``{anima_dir}/token_usage/{date}.jsonl``.  Each entry represents
one execution cycle (chat, heartbeat, cron, inbox, task).
"""

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.memory.fact_observability import warn_rate_limited
from core.time_utils import now_local, today_local

logger = logging.getLogger("animaworks.token_usage")

# ── Default pricing (USD per 1M tokens, as of 2026-03) ─────
# Override via ~/.animaworks/pricing.json
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # Opus 4.7 / 4.8 share Opus 4.5/4.6 pricing ($5/$25 per 1M in/out).
    # Source: ~/.claude/skills/claude-api (SKILL.md model table, 2026-07).
    # cache_read = 0.1x input, cache_write = 1.25x input (5-min TTL).
    "claude-opus-4-8": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-opus-4-7": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-opus-4-6": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-sonnet-4": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-opus-4-5": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-sonnet-4-5": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
    "claude-haiku-3-5": {
        "input": 0.80,
        "output": 4.0,
        "cache_read": 0.08,
        "cache_write": 1.0,
    },
    "gpt-4.1": {
        "input": 2.0,
        "output": 8.0,
        "cache_read": 0.50,
        "cache_write": 2.0,
    },
    "gpt-4.1-mini": {
        "input": 0.40,
        "output": 1.60,
        "cache_read": 0.10,
        "cache_write": 0.40,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.0,
        "cache_read": 1.25,
        "cache_write": 2.50,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.075,
        "cache_write": 0.15,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.0,
        "cache_read": 0.315,
        "cache_write": 1.25,
    },
    "gemini-2.5-flash": {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.0375,
        "cache_write": 0.15,
    },
}


# ── TokenUsageLogger ──────────────────────────────────────────


class TokenUsageLogger:
    """Append-only JSONL logger for per-session token usage."""

    def __init__(self, anima_dir: Path) -> None:
        self._dir = anima_dir / "token_usage"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pricing: dict[str, dict[str, float]] | None = None

    def log(
        self,
        *,
        model: str,
        trigger: str,
        mode: str,
        auth: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        turns: int = 0,
        chains: int = 0,
        duration_ms: int = 0,
    ) -> None:
        """Log a single session's token usage."""
        now = now_local()
        cost = self.estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        entry: dict[str, Any] = {
            "ts": now.isoformat(),
            "model": model,
            "trigger": trigger,
            "mode": mode,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "turns": turns,
            "duration_ms": duration_ms,
            "estimated_cost_usd": round(cost, 6),
        }
        if auth:
            entry["auth"] = auth
        if cache_read_tokens:
            entry["cache_read_tokens"] = cache_read_tokens
        if cache_write_tokens:
            entry["cache_write_tokens"] = cache_write_tokens
        if chains:
            entry["chains"] = chains

        path = self._dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Failed to write token usage log", exc_info=True)

    # ── Cost estimation ────────────────────────────────────

    def estimate_cost(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Return estimated cost in USD."""
        pricing = self._resolve_pricing(model)
        if not pricing:
            return 0.0
        cost = (
            input_tokens * pricing.get("input", 0)
            + output_tokens * pricing.get("output", 0)
            + cache_read_tokens * pricing.get("cache_read", 0)
            + cache_write_tokens * pricing.get("cache_write", 0)
        ) / 1_000_000
        return cost

    def _resolve_pricing(self, model: str) -> dict[str, float] | None:
        """Resolve pricing for a model (longest prefix match)."""
        table = self._load_pricing_table()
        bare = model.split("/", 1)[-1] if "/" in model else model
        bare = re.sub(r"^[a-z]{2}\.anthropic\.", "", bare)
        matches = [(p, v) for p, v in table.items() if bare.startswith(p)]
        if not matches:
            warn_rate_limited(
                logger,
                f"token_usage.unknown_pricing.{bare}",
                "No pricing entry for model %r; cost will be estimated as 0.0. "
                "Add it to DEFAULT_PRICING or pricing.json.",
                bare,
            )
            return None
        return max(matches, key=lambda x: len(x[0]))[1]

    def _load_pricing_table(self) -> dict[str, dict[str, float]]:
        if self._pricing is not None:
            return self._pricing
        from core.paths import get_data_dir

        custom = get_data_dir() / "pricing.json"
        if custom.is_file():
            try:
                with custom.open(encoding="utf-8") as f:
                    self._pricing = json.load(f)
                    return self._pricing
            except Exception:
                logger.warning("Failed to load pricing.json", exc_info=True)
        self._pricing = DEFAULT_PRICING
        return self._pricing

    # ── Aggregation ────────────────────────────────────────

    def read_entries(
        self,
        days: int = 30,
        *,
        target_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Read log entries for a date range."""
        entries: list[dict[str, Any]] = []
        end = target_date or today_local()
        for i in range(days):
            d = end - timedelta(days=i)
            path = self._dir / f"{d.isoformat()}.jsonl"
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Failed to read %s", path, exc_info=True)
                continue
            for line in raw.splitlines():
                line = line.strip().strip("\x00")
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries

    def monthly_total(self, now: datetime) -> int:
        """Return total tokens consumed in the month containing *now*.

        Completed days are cached as daily totals in ``rollup.json``.  The
        current day's JSONL remains mutable, so it is always read directly.
        JSONL files are authoritative: a missing, malformed, or incomplete
        rollup is rebuilt from the corresponding completed-day logs.
        """
        current_day = now.date()
        month_prefix = current_day.strftime("%Y-%m-")
        rollup_path = self._dir / "rollup.json"

        past_logs: dict[str, Path] = {}
        for path in self._dir.glob(f"{month_prefix}*.jsonl"):
            try:
                log_day = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if log_day < current_day and log_day.year == current_day.year and log_day.month == current_day.month:
                past_logs[log_day.isoformat()] = path

        cached = self._read_rollup(rollup_path)
        rollup: dict[str, int] = {}
        if cached is not None:
            rollup = {day: cached[day] for day in past_logs if day in cached}

        for day, path in past_logs.items():
            if day not in rollup:
                rollup[day] = self._read_daily_total(path)

        if cached is None or rollup != cached:
            self._write_rollup(rollup_path, rollup)

        today_total = self._read_daily_total(self._dir / f"{current_day.isoformat()}.jsonl")
        return sum(rollup.values()) + today_total

    @staticmethod
    def _read_daily_total(path: Path) -> int:
        """Read the ``total_tokens`` sum from one daily JSONL file."""
        if not path.is_file():
            return 0
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("Failed to read %s", path, exc_info=True)
            raise

        total = 0
        for line in raw.splitlines():
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = entry.get("total_tokens", 0) if isinstance(entry, dict) else 0
            if isinstance(value, int) and not isinstance(value, bool):
                total += value
        return total

    @staticmethod
    def _read_rollup(path: Path) -> dict[str, int] | None:
        """Load a valid rollup, returning ``None`` when it needs rebuilding."""
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if any(
            not isinstance(day, str) or not isinstance(value, int) or isinstance(value, bool)
            for day, value in data.items()
        ):
            return None
        return data

    @staticmethod
    def _write_rollup(path: Path, rollup: dict[str, int]) -> None:
        """Persist daily totals without allowing cache I/O to break callers."""
        try:
            atomic_write_text(path, json.dumps(dict(sorted(rollup.items())), ensure_ascii=False, indent=2) + "\n")
        except Exception:
            logger.warning("Failed to write token usage rollup", exc_info=True)

    def summarize(
        self,
        days: int = 30,
        *,
        target_date: date | None = None,
    ) -> dict[str, Any]:
        """Return a summary of token usage over the given period."""
        entries = self.read_entries(days, target_date=target_date)
        if not entries:
            return {
                "period_days": days,
                "total_sessions": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cache_read_tokens": 0,
                "total_cache_write_tokens": 0,
                "total_tokens": 0,
                "total_estimated_cost_usd": 0.0,
                "by_model": {},
                "by_trigger": {},
                "by_date": {},
            }

        total_input = sum(e.get("input_tokens", 0) for e in entries)
        total_output = sum(e.get("output_tokens", 0) for e in entries)
        total_cache_read = sum(e.get("cache_read_tokens", 0) for e in entries)
        total_cache_write = sum(e.get("cache_write_tokens", 0) for e in entries)
        total_cost = sum(e.get("estimated_cost_usd", 0) for e in entries)

        by_model: dict[str, dict[str, Any]] = {}
        by_trigger: dict[str, dict[str, Any]] = {}
        by_date: dict[str, dict[str, Any]] = {}

        for e in entries:
            model = e.get("model", "unknown")
            trigger = e.get("trigger", "unknown")
            ts = e.get("ts", "")
            day = ts[:10] if ts else "unknown"
            cost = e.get("estimated_cost_usd", 0)
            inp = e.get("input_tokens", 0)
            out = e.get("output_tokens", 0)
            cache_r = e.get("cache_read_tokens", 0)
            cache_w = e.get("cache_write_tokens", 0)

            for key, bucket in ((model, by_model), (trigger, by_trigger), (day, by_date)):
                if key not in bucket:
                    bucket[key] = {
                        "sessions": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "cost_usd": 0.0,
                    }
                bucket[key]["sessions"] += 1
                bucket[key]["input_tokens"] += inp
                bucket[key]["output_tokens"] += out
                bucket[key]["cache_read_tokens"] += cache_r
                bucket[key]["cache_write_tokens"] += cache_w
                bucket[key]["cost_usd"] = round(bucket[key]["cost_usd"] + cost, 6)

        return {
            "period_days": days,
            "total_sessions": len(entries),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "total_tokens": total_input + total_output,
            "total_estimated_cost_usd": round(total_cost, 4),
            "by_model": by_model,
            "by_trigger": by_trigger,
            "by_date": dict(sorted(by_date.items())),
        }
