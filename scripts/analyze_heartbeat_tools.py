#!/usr/bin/env python3
"""Analyze Heartbeat tool usage patterns from activity_log.

Usage: python scripts/analyze_heartbeat_tools.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ANIMAS_ROOT = Path.home() / ".animaworks" / "animas"
DATES = ["2026-03-10", "2026-03-11", "2026-03-12"]
HEAVY_TOOLS = {"list_tasks", "search_memory", "read_memory_file", "mcp__aw__list_tasks",
               "mcp__aw__search_memory", "mcp__aw__read_memory_file", "mcp__aw__list_background_tasks",
               "list_background_tasks", "plan_tasks", "mcp__aw__plan_tasks"}


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_events(anima: str, date: str) -> list[dict]:
    path = ANIMAS_ROOT / anima / "activity_log" / f"{date}.jsonl"
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def extract_hb_sessions(events: list[dict]) -> list[tuple[datetime, datetime | None, list[dict]]]:
    """Extract (start_ts, end_ts, events_in_session) for each HB session."""
    sessions = []
    current_start = None
    current_events = []

    for ev in events:
        t = ev.get("type")
        ts = parse_ts(ev["ts"]) if ev.get("ts") else None

        if t == "heartbeat_start":
            if current_start is not None:
                sessions.append((current_start, None, current_events))
            current_start = ts
            current_events = [ev]
        elif t == "heartbeat_end":
            if current_start is not None:
                current_events.append(ev)
                sessions.append((current_start, ts, current_events))
            current_start = None
            current_events = []
        elif current_start is not None:
            current_events.append(ev)

    if current_start is not None:
        sessions.append((current_start, None, current_events))

    return sessions


def main() -> None:
    animas = [d.name for d in ANIMAS_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")]
    animas = sorted(animas)

    all_hb_tool_counts: dict[str, int] = defaultdict(int)
    all_hb_non_tool: dict[str, int] = defaultdict(int)
    anima_send_post: dict[str, int] = defaultdict(int)
    long_sessions: list[dict] = []

    for anima in animas:
        for date in DATES:
            events = load_events(anima, date)
            if not events:
                continue

            sessions = extract_hb_sessions(events)
            for start_ts, end_ts, sess_events in sessions:
                if end_ts is None:
                    continue
                duration_min = (end_ts - start_ts).total_seconds() / 60

                tool_counts: dict[str, int] = defaultdict(int)
                non_tool_counts: dict[str, int] = defaultdict(int)
                send_post_count = 0

                for ev in sess_events:
                    t = ev.get("type")
                    if t == "tool_use":
                        tool = ev.get("tool") or ev.get("meta", {}).get("tool")
                        if tool:
                            tool_counts[tool] += 1
                            all_hb_tool_counts[tool] += 1
                            if tool in ("send_message", "mcp__aw__send_message", "post_channel",
                                        "mcp__aw__post_channel"):
                                send_post_count += 1
                    elif t in ("message_sent", "channel_post", "channel_read", "memory_write",
                                "human_notify", "error", "tool_result"):
                        non_tool_counts[t] += 1
                        all_hb_non_tool[t] += 1

                anima_send_post[anima] += send_post_count

                if duration_min >= 20:
                    heavy_count = sum(tool_counts[t] for t in HEAVY_TOOLS if t in tool_counts)
                    long_sessions.append({
                        "anima": anima,
                        "date": date,
                        "start": start_ts.isoformat(),
                        "end": end_ts.isoformat(),
                        "duration_min": round(duration_min, 1),
                        "tool_counts": dict(tool_counts),
                        "non_tool_counts": dict(non_tool_counts),
                        "heavy_tool_count": heavy_count,
                        "send_post_count": send_post_count,
                    })

    # Report
    print("=" * 70)
    print("Heartbeat ツール使用パターン分析 (2026-03-10 〜 12)")
    print("=" * 70)

    print("\n## 1. HB中に最も頻繁に使用されるツール TOP10")
    print("-" * 50)
    sorted_tools = sorted(all_hb_tool_counts.items(), key=lambda x: -x[1])
    for i, (tool, count) in enumerate(sorted_tools[:10], 1):
        print(f"  {i:2}. {tool}: {count}回")

    print("\n## 2. HB中に send_message / post_channel が多いアニマ（実行寄り）")
    print("-" * 50)
    sorted_anima = sorted(anima_send_post.items(), key=lambda x: -x[1])
    for anima, count in sorted_anima:
        if count > 0:
            print(f"  {anima}: {count}回")

    print("\n## 3. 20分以上かかったHBセッション一覧")
    print("-" * 50)
    if not long_sessions:
        print("  （該当なし）")
    else:
        for s in sorted(long_sessions, key=lambda x: -x["duration_min"]):
            print(f"\n  [{s['anima']}] {s['date']} | {s['duration_min']}分")
            print(f"    開始: {s['start']}")
            print(f"    終了: {s['end']}")
            print(f"    ツール呼び出し:")
            for tool, cnt in sorted(s["tool_counts"].items(), key=lambda x: -x[1])[:15]:
                heavy = " (重い)" if tool in HEAVY_TOOLS else ""
                print(f"      - {tool}: {cnt}回{heavy}")
            print(f"    重いツール合計: {s['heavy_tool_count']}回")
            print(f"    tool_use以外: {dict(s['non_tool_counts'])}")
            print(f"    send_message/post_channel: {s['send_post_count']}回")

    print("\n## 4. 全HB中の tool_use 以外イベント集計")
    print("-" * 50)
    for t, count in sorted(all_hb_non_tool.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}回")

    print("\n## 5. 重いツール（list_tasks, search_memory, read_memory_file等）の総呼び出し")
    print("-" * 50)
    heavy_total = sum(all_hb_tool_counts[t] for t in HEAVY_TOOLS if t in all_hb_tool_counts)
    print(f"  合計: {heavy_total}回")
    for t in HEAVY_TOOLS:
        if t in all_hb_tool_counts:
            print(f"    {t}: {all_hb_tool_counts[t]}回")


if __name__ == "__main__":
    main()
