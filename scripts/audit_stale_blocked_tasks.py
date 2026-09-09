#!/usr/bin/env python3
"""List tasks stuck in blocked status for more than N days.

Blocked is not a terminal status, so TaskBoard archive never closes these;
they silently pile up (owner directive 2026-08-13). Weekly cron feeds the
output to rin for triage.

Usage: audit_stale_blocked_tasks.py [--days 7] [--data-dir ~/.animaworks]
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))


def latest_status(queue_path: Path) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    try:
        with queue_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = d.get("task_id") or d.get("id")
                if tid:
                    tasks.setdefault(tid, {}).update(d)
    except OSError:
        pass
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--data-dir", default="~/.animaworks")
    args = ap.parse_args()

    cutoff = datetime.now(JST) - timedelta(days=args.days)
    data_dir = Path(args.data_dir).expanduser()
    stale: list[str] = []

    for queue in sorted(data_dir.glob("animas/*/state/task_queue.jsonl")):
        anima = queue.parent.parent.name
        for tid, task in latest_status(queue).items():
            if task.get("status") != "blocked":
                continue
            # blocked_at, not updated_at: the unblock_check retry bumps updated_at
            # every heartbeat, which hid every stale task from this audit.
            meta = task.get("meta") or {}
            ts_raw = meta.get("blocked_at") or task.get("ts") or task.get("updated_at") or ""
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=JST)
            if ts < cutoff:
                summary = (task.get("summary") or task.get("description") or "")[:120]
                stale.append(f"- {anima} {tid} (since {ts_raw[:10]}): {summary}")

    if stale:
        print(f"blocked のまま {args.days} 日以上経過したタスク: {len(stale)}件")
        print("\n".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
