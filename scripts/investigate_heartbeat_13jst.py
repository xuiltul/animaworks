#!/usr/bin/env python3
"""2026-03-12 13:00 JST以降のHeartbeat実行時間を調査するスクリプト。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ANIMAS_DIR = Path.home() / ".animaworks" / "animas"
ACTIVITY_LOG = "activity_log"
LOG_FILE = "2026-03-12.jsonl"
CUTOFF_JST = datetime.fromisoformat("2026-03-12T13:00:00+09:00")


def parse_ts(ts_str: str) -> datetime | None:
    """ISO形式タイムスタンプをパース。タイムゾーンなしはJSTとして扱う。"""
    if not ts_str:
        return None
    try:
        if ts_str.endswith("Z"):
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if "+" in ts_str or (len(ts_str) >= 6 and ts_str[-6] in "-+"):
            return datetime.fromisoformat(ts_str)
        return datetime.fromisoformat(ts_str + "+09:00")
    except (ValueError, TypeError):
        return None


def is_after_cutoff(ts: datetime | None) -> bool:
    return ts is not None and ts >= CUTOFF_JST


def main() -> None:
    anima_dirs = [d for d in ANIMAS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    anima_dirs.sort(key=lambda d: d.name)

    results: list[dict] = []

    for anima_dir in anima_dirs:
        log_path = anima_dir / ACTIVITY_LOG / LOG_FILE
        if not log_path.exists():
            continue

        events: list[dict] = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        sessions: list[dict] = []
        pending_start: dict | None = None

        for ev in events:
            ev_type = ev.get("type")
            ts = parse_ts(ev.get("ts", ""))

            if ev_type == "heartbeat_start":
                if is_after_cutoff(ts):
                    pending_start = {"ts": ts, "ev": ev}
                else:
                    pending_start = None

            elif ev_type == "heartbeat_end" and pending_start is not None:
                if is_after_cutoff(ts):
                    duration_sec = (ts - pending_start["ts"]).total_seconds()
                    sessions.append({
                        "start_ts": pending_start["ts"],
                        "end_ts": ts,
                        "duration_sec": duration_sec,
                    })
                pending_start = None

        tool_counts: dict[str, int] = {}
        for sess in sessions:
            start_ts = sess["start_ts"]
            end_ts = sess["end_ts"]
            for ev in events:
                if ev.get("type") != "tool_use":
                    continue
                t = parse_ts(ev.get("ts", ""))
                if t is not None and start_ts <= t <= end_ts:
                    tool = ev.get("tool") or ev.get("tool_name") or "(unknown)"
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

        if not sessions:
            continue

        durations = [s["duration_sec"] for s in sessions]
        avg_dur = sum(durations) / len(durations)
        max_dur = max(durations)

        results.append({
            "anima": anima_dir.name,
            "session_count": len(sessions),
            "avg_sec": avg_dur,
            "max_sec": max_dur,
            "sessions": sessions,
            "tool_counts": tool_counts,
        })

    results.sort(key=lambda r: r["max_sec"], reverse=True)

    print("=" * 80)
    print("2026-03-12 13:00 JST 以降の Heartbeat 実行時間調査")
    print("=" * 80)

    for r in results:
        print(f"\n--- {r['anima']} ---")
        print(f"  HBセッション数: {r['session_count']}")
        print(f"  平均所要時間: {r['avg_sec']:.1f} 秒")
        print(f"  最大所要時間: {r['max_sec']:.1f} 秒")
        if r["tool_counts"]:
            sorted_tools = sorted(r["tool_counts"].items(), key=lambda x: x[1], reverse=True)[:10]
            print("  上位ツール使用回数:")
            for tool, count in sorted_tools:
                print(f"    - {tool}: {count}回")

    print("\n" + "=" * 80)
    print("サマリ（所要時間降順）")
    print("=" * 80)
    print(f"{'アニマ':<12} {'セッション数':>8} {'平均(秒)':>10} {'最大(秒)':>10}")
    print("-" * 45)
    for r in results:
        print(f"{r['anima']:<12} {r['session_count']:>8} {r['avg_sec']:>10.1f} {r['max_sec']:>10.1f}")


if __name__ == "__main__":
    main()
