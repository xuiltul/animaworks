#!/usr/bin/env python3
"""全アニマのタスク関連データ蓄積状況を調査するスクリプト。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ANIMAS_ROOT = Path.home() / ".animaworks" / "animas"

# フラグ閾値
TASK_QUEUE_SIZE_WARN = 100_000  # 100KB
TASK_QUEUE_LINES_WARN = 500
TASK_RESULTS_FILES_WARN = 50
TASK_RESULTS_SIZE_WARN = 500_000  # 500KB
PENDING_FILES_WARN = 10
PENDING_FAILED_WARN = 5
CURRENT_TASK_SIZE_WARN = 5_000  # 5KB
PENDING_MD_SIZE_WARN = 10_000  # 10KB
BG_NOTIF_WARN = 20


def get_animas() -> list[str]:
    """animasディレクトリ内のアニマ名一覧（ディレクトリのみ）。"""
    if not ANIMAS_ROOT.exists():
        return []
    return sorted(
        d.name
        for d in ANIMAS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def analyze_task_queue(path: Path) -> dict:
    """task_queue.jsonl を解析。replayして各task_idの最新statusを集計。"""
    result = {
        "size": 0,
        "lines": 0,
        "pending": 0,
        "in_progress": 0,
        "done": 0,
        "cancelled": 0,
        "failed": 0,
        "blocked": 0,
        "delegated": 0,
        "other": 0,
    }
    if not path.exists():
        return result

    result["size"] = path.stat().st_size
    latest: dict[str, str] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result["lines"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = obj.get("task_id")
            status = obj.get("status", "unknown")
            if task_id:
                latest[task_id] = status

    for status in latest.values():
        s = (status or "").lower()
        if s == "pending":
            result["pending"] += 1
        elif s == "in_progress":
            result["in_progress"] += 1
        elif s == "done":
            result["done"] += 1
        elif s == "cancelled":
            result["cancelled"] += 1
        elif s == "failed":
            result["failed"] += 1
        elif s == "blocked":
            result["blocked"] += 1
        elif s == "delegated":
            result["delegated"] += 1
        else:
            result["other"] += 1

    return result


def analyze_dir(path: Path, exclude: set[str] | None = None) -> tuple[int, int, datetime | None, datetime | None]:
    """ディレクトリ内のファイル数、合計サイズ、最古/最新日付。"""
    exclude = exclude or set()
    count = 0
    total = 0
    oldest: datetime | None = None
    newest: datetime | None = None

    if not path.exists() or not path.is_dir():
        return 0, 0, None, None

    for f in path.iterdir():
        if f.name in exclude:
            continue
        if f.is_file():
            count += 1
            total += f.stat().st_size
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if oldest is None or mtime < oldest:
                oldest = mtime
            if newest is None or mtime > newest:
                newest = mtime
        elif f.is_dir():
            sub_count, sub_size, sub_old, sub_new = analyze_dir(f, exclude)
            count += sub_count
            total += sub_size
            if sub_old and (oldest is None or sub_old < oldest):
                oldest = sub_old
            if sub_new and (newest is None or sub_new > newest):
                newest = sub_new

    return count, total, oldest, newest


def analyze_pending(path: Path) -> dict:
    """state/pending/ を解析（.wake除く）。"""
    result = {
        "files": 0,
        "size": 0,
        "oldest": None,
        "failed_count": 0,
        "failed_oldest": None,
        "processing_count": 0,
    }
    if not path.exists():
        return result

    exclude = {".wake"}
    result["files"], result["size"], result["oldest"], _ = analyze_dir(path, exclude)

    failed_dir = path / "failed"
    if failed_dir.exists():
        fc, _, fo, _ = analyze_dir(failed_dir)
        result["failed_count"] = fc
        result["failed_oldest"] = fo

    proc_dir = path / "processing"
    if proc_dir.exists():
        result["processing_count"], _, _, _ = analyze_dir(proc_dir)

    return result


def analyze_anima(name: str) -> dict:
    """1アニマ分の調査結果。"""
    base = ANIMAS_ROOT / name
    state = base / "state"

    task_queue_path = state / "task_queue.jsonl"
    tq = analyze_task_queue(task_queue_path)

    task_results_path = state / "task_results"
    tr_count, tr_size, tr_old, tr_new = analyze_dir(task_results_path)

    pending_path = state / "pending"
    pend = analyze_pending(pending_path)

    current_task_path = state / "current_task.md"
    ct_size = current_task_path.stat().st_size if current_task_path.exists() else 0

    pending_md_path = state / "pending.md"
    pm_size = pending_md_path.stat().st_size if pending_md_path.exists() else 0

    bg_path = state / "background_notifications"
    bg_count, _, _, _ = analyze_dir(bg_path)

    flags: list[str] = []
    if tq["size"] > TASK_QUEUE_SIZE_WARN:
        flags.append("task_queue大")
    if tq["lines"] > TASK_QUEUE_LINES_WARN:
        flags.append("task_queue行数多")
    if tq["done"] > 100:
        flags.append("done残多")
    if tr_count > TASK_RESULTS_FILES_WARN:
        flags.append("task_results多")
    if tr_size > TASK_RESULTS_SIZE_WARN:
        flags.append("task_results大")
    if pend["files"] > PENDING_FILES_WARN:
        flags.append("pending多")
    if pend["failed_count"] > PENDING_FAILED_WARN:
        flags.append("failed多")
    if ct_size > CURRENT_TASK_SIZE_WARN:
        flags.append("current_task大")
    if pm_size > PENDING_MD_SIZE_WARN:
        flags.append("pending.md大")
    if bg_count > BG_NOTIF_WARN:
        flags.append("bg_notif多")

    return {
        "name": name,
        "task_queue": tq,
        "task_results": {"count": tr_count, "size": tr_size, "oldest": tr_old, "newest": tr_new},
        "pending": pend,
        "current_task_size": ct_size,
        "pending_md_size": pm_size,
        "background_notifications": bg_count,
        "flags": flags,
    }


def fmt_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d")


def main() -> None:
    animas = get_animas()
    if not animas:
        print("No animas found in", ANIMAS_ROOT)
        return

    results: list[dict] = []
    for name in animas:
        try:
            r = analyze_anima(name)
            results.append(r)
        except Exception as e:
            results.append({
                "name": name,
                "error": str(e),
                "flags": ["ERROR"],
            })

    # テーブル出力
    print("\n" + "=" * 140)
    print("AnimaWorks タスク関連データ蓄積状況")
    print("=" * 140)

    header = (
        f"{'アニマ名':<12} | "
        f"{'task_queue':<35} | "
        f"{'task_results':<18} | "
        f"{'pending/':<18} | "
        f"{'current_task':<10} | "
        f"{'pending.md':<10} | "
        f"フラグ"
    )
    print(header)
    print("-" * 140)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<12} | ERROR: {r['error']}")
            continue

        tq = r["task_queue"]
        other = tq["cancelled"] + tq["failed"] + tq["blocked"] + tq["delegated"] + tq["other"]
        tq_str = f"{fmt_size(tq['size'])}/{tq['lines']} p:{tq['pending']} ip:{tq['in_progress']} d:{tq['done']} o:{other}"

        tr = r["task_results"]
        tr_str = f"{tr['count']}f/{fmt_size(tr['size'])}"

        pend = r["pending"]
        pend_str = f"{pend['files']}f (failed:{pend['failed_count']})"

        ct = fmt_size(r["current_task_size"])
        pm = fmt_size(r["pending_md_size"])
        flags_str = ", ".join(r["flags"]) if r["flags"] else "-"

        print(f"{r['name']:<12} | {tq_str:<35} | {tr_str:<18} | {pend_str:<18} | {ct:<10} | {pm:<10} | {flags_str}")

    # 詳細サマリ（task_results日付、pending日付等）
    print("\n" + "=" * 140)
    print("詳細サマリ（日付・サイズ）")
    print("=" * 140)

    for r in results:
        if "error" in r:
            continue
        pend = r["pending"]
        tr = r["task_results"]
        lines = [
            f"  {r['name']}:",
            f"    task_results: {tr['count']} files, {fmt_size(tr['size'])} (oldest: {fmt_date(tr.get('oldest'))}, newest: {fmt_date(tr.get('newest'))})",
            f"    pending: {pend['files']} files, failed={pend['failed_count']} (oldest: {fmt_date(pend.get('failed_oldest'))}), processing={pend['processing_count']}",
            f"    background_notifications: {r['background_notifications']} files",
        ]
        if r["flags"]:
            lines.append(f"    ⚠️ {', '.join(r['flags'])}")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
