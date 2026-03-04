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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.token_usage")

JST = timezone(timedelta(hours=9))

# ── Default pricing (USD per 1M tokens, as of 2026-03) ─────
# Override via ~/.animaworks/pricing.json
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
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
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-sonnet-4-5": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-haiku-4": {
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
        now = datetime.now(JST)
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
        end = target_date or date.today()
        for i in range(days):
            d = end - timedelta(days=i)
            path = self._dir / f"{d.isoformat()}.jsonl"
            if not path.is_file():
                continue
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            except (OSError, json.JSONDecodeError):
                logger.warning("Failed to read %s", path, exc_info=True)
        return entries

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
                "total_tokens": 0,
                "total_estimated_cost_usd": 0.0,
                "by_model": {},
                "by_trigger": {},
                "by_date": {},
            }

        total_input = sum(e.get("input_tokens", 0) for e in entries)
        total_output = sum(e.get("output_tokens", 0) for e in entries)
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

            for key, bucket in ((model, by_model), (trigger, by_trigger), (day, by_date)):
                if key not in bucket:
                    bucket[key] = {"sessions": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                bucket[key]["sessions"] += 1
                bucket[key]["input_tokens"] += inp
                bucket[key]["output_tokens"] += out
                bucket[key]["cost_usd"] = round(bucket[key]["cost_usd"] + cost, 6)

        return {
            "period_days": days,
            "total_sessions": len(entries),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_estimated_cost_usd": round(total_cost, 4),
            "by_model": by_model,
            "by_trigger": by_trigger,
            "by_date": dict(sorted(by_date.items())),
        }
