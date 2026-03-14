#!/usr/bin/env python3
"""Investigate Heartbeat execution times for all Animas."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ANIMAS_ROOT = Path.home() / ".animaworks" / "animas"
DATES = ["2026-03-10", "2026-03-11", "2026-03-12"]
LONG_SESSION_THRESHOLD_SEC = 30 * 60  # 30 minutes


def parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def main() -> None:
    if not ANIMAS_ROOT.exists():
        print(f"Animas root not found: {ANIMAS_ROOT}")
        return

    anima_dirs = [d for d in ANIMAS_ROOT.iterdir() if d.is_dir()]
    if not anima_dirs:
        print("No anima directories found.")
        return

    results: list[dict] = []

    for anima_dir in sorted(anima_dirs):
        name = anima_dir.name
        log_dir = anima_dir / "activity_log"
        if not log_dir.exists():
            continue

        sessions: list[tuple[datetime, datetime, float]] = []

        for date in DATES:
            log_file = log_dir / f"{date}.jsonl"
            if not log_file.exists():
                continue

            pending_start: datetime | None = None

            for line in log_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                typ = entry.get("type")
                ts_str = entry.get("ts")
                if not ts_str:
                    continue

                ts = parse_ts(ts_str)

                if typ == "heartbeat_start":
                    pending_start = ts
                elif typ == "heartbeat_end" and pending_start is not None:
                    duration_sec = (ts - pending_start).total_seconds()
                    if duration_sec >= 0:
                        sessions.append((pending_start, ts, duration_sec))
                    pending_start = None

        if not sessions:
            continue

        durations = [s[2] for s in sessions]
        max_idx = max(range(len(durations)), key=lambda i: durations[i])
        max_session = sessions[max_idx]

        results.append({
            "name": name,
            "count": len(sessions),
            "avg_sec": sum(durations) / len(durations),
            "max_sec": max(durations),
            "max_ts": max_session[0].isoformat(),
            "long_sessions": sum(1 for d in durations if d >= LONG_SESSION_THRESHOLD_SEC),
        })

    results.sort(key=lambda r: r["max_sec"], reverse=True)

    print("=" * 100)
    print("Heartbeat Execution Time Investigation (2026-03-10 ~ 2026-03-12)")
    print("=" * 100)
    print(f"{'Anima':<20} {'Sessions':>8} {'Avg (sec)':>10} {'Max (sec)':>10} {'Max (min)':>10} {'Max TS':<28} {'30min+':>8}")
    print("-" * 100)

    for r in results:
        flag = "***" if r["long_sessions"] > 0 else ""
        print(
            f"{r['name']:<20} {r['count']:>8} {r['avg_sec']:>10.1f} {r['max_sec']:>10.1f} "
            f"{r['max_sec']/60:>10.1f} {r['max_ts'][:26]:<28} {r['long_sessions']:>5} {flag}"
        )

    print("-" * 100)
    print("*** = Has session(s) >= 30 minutes")
    print()

    if any(r["long_sessions"] > 0 for r in results):
        print("Animas with 30+ minute sessions:")
        for r in results:
            if r["long_sessions"] > 0:
                print(f"  - {r['name']}: {r['long_sessions']} session(s), max {r['max_sec']/60:.1f} min at {r['max_ts']}")


if __name__ == "__main__":
    main()
