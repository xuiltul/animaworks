#!/usr/bin/env python3
"""Analyze Heartbeat execution times across all Animas."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ANIMAS_ROOT = Path.home() / ".animaworks" / "animas"
DATES = ("2026-03-11", "2026-03-12")
LONG_SESSION_THRESHOLD_SEC = 600  # 10 minutes


def parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def main() -> None:
    anima_dirs = sorted(
        d for d in ANIMAS_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    results: list[dict] = []

    for anima_dir in anima_dirs:
        name = anima_dir.name
        log_dir = anima_dir / "activity_log"
        if not log_dir.exists():
            continue

        sessions: list[dict] = []

        for date in DATES:
            log_file = log_dir / f"{date}.jsonl"
            if not log_file.exists():
                continue

            pending_start: dict | None = None
            events_in_session: list[dict] = []

            for line in log_file.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ev_type = ev.get("type")
                ts_str = ev.get("ts")
                if not ts_str:
                    continue

                if ev_type == "heartbeat_start":
                    if pending_start:
                        pending_start = None
                    pending_start = {"ts": ts_str, "events": []}
                    events_in_session = pending_start["events"]

                elif ev_type == "heartbeat_end" and pending_start:
                    start_ts = parse_ts(pending_start["ts"])
                    end_ts = parse_ts(ts_str)
                    duration_sec = (end_ts - start_ts).total_seconds()
                    tool_count = sum(
                        1 for e in events_in_session if e.get("type") == "tool_use"
                    )
                    sessions.append(
                        {
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "duration_sec": duration_sec,
                            "tool_use_count": tool_count,
                        }
                    )
                    pending_start = None

                elif pending_start and ev_type == "tool_use":
                    events_in_session.append(ev)

        if not sessions:
            continue

        durations = [s["duration_sec"] for s in sessions]
        avg_sec = sum(durations) / len(durations)
        max_sec = max(durations)
        latest = max(sessions, key=lambda s: s["start_ts"])
        tool_use_counts = [s["tool_use_count"] for s in sessions]
        total_tool_use = sum(tool_use_counts)
        long_sessions = [s for s in sessions if s["duration_sec"] >= LONG_SESSION_THRESHOLD_SEC]

        results.append(
            {
                "name": name,
                "session_count": len(sessions),
                "avg_sec": avg_sec,
                "max_sec": max_sec,
                "latest_ts": latest["start_ts"],
                "latest_duration_sec": latest["duration_sec"],
                "total_tool_use": total_tool_use,
                "long_session_count": len(long_sessions),
                "long_sessions": long_sessions,
            }
        )

    results.sort(key=lambda r: r["avg_sec"], reverse=True)

    print("=" * 140)
    print("AnimaWorks Heartbeat 実行時間調査 (2026-03-11, 2026-03-12)")
    print("=" * 140)
    print(
        f"{'Anima':<12} {'HB数':>6} {'平均(秒)':>10} {'最大(秒)':>10} "
        f"{'最新HB時刻':>22} {'最新所要(秒)':>12} {'tool_use合計':>12} {'10分超':>8}"
    )
    print("-" * 140)

    for r in results:
        latest_str = r["latest_ts"].strftime("%Y-%m-%d %H:%M:%S")
        long_flag = f"⚠️ {r['long_session_count']}" if r["long_session_count"] else ""
        print(
            f"{r['name']:<12} {r['session_count']:>6} {r['avg_sec']:>10.1f} {r['max_sec']:>10.1f} "
            f"{latest_str:>22} {r['latest_duration_sec']:>12.1f} {r['total_tool_use']:>12} {long_flag:>8}"
        )

    print("-" * 140)

    long_any = [r for r in results if r["long_session_count"]]
    if long_any:
        print("\n【10分以上のHBセッション詳細】")
        for r in long_any:
            for s in r["long_sessions"]:
                start_str = s["start_ts"].strftime("%Y-%m-%d %H:%M:%S")
                dur_min = s["duration_sec"] / 60
                print(f"  {r['name']}: {start_str} ({dur_min:.1f}分, tool_use={s['tool_use_count']})")


if __name__ == "__main__":
    main()
