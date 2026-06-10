#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Report per-Anima atomic-facts adoption from runtime JSONL files."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FactsHealthRow:
    anima: str
    status: str
    facts_count: int
    facts_files: int
    last_fact_date: str
    last_fact_mtime: str
    last_episode_date: str


def collect_facts_health(data_dir: Path) -> list[FactsHealthRow]:
    animas_dir = data_dir / "animas"
    if not animas_dir.is_dir():
        return []

    rows: list[FactsHealthRow] = []
    for anima_dir in sorted(path for path in animas_dir.iterdir() if path.is_dir()):
        status = _runtime_status(anima_dir)
        fact_files = sorted((anima_dir / "facts").glob("*.jsonl"))
        facts_count = sum(_count_jsonl_records(path) for path in fact_files)
        last_fact_file = _latest_mtime_file(fact_files)
        rows.append(
            FactsHealthRow(
                anima=anima_dir.name,
                status=status,
                facts_count=facts_count,
                facts_files=len(fact_files),
                last_fact_date=_last_fact_date(fact_files),
                last_fact_mtime=_mtime_iso(last_fact_file) if last_fact_file else "",
                last_episode_date=_last_episode_date(anima_dir),
            )
        )
    return rows


def _runtime_status(anima_dir: Path) -> str:
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        return "unknown"
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return "status_error"
    if data.get("enabled") is False:
        return "dormant"
    return "active"


def _count_jsonl_records(path: Path) -> int:
    count = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    return count


def _last_fact_date(paths: list[Path]) -> str:
    dates = [path.stem for path in paths if path.stem]
    return max(dates, default="")


def _last_episode_date(anima_dir: Path) -> str:
    episodes_dir = anima_dir / "episodes"
    if not episodes_dir.is_dir():
        return ""
    dates = [path.stem for path in episodes_dir.glob("*.md") if path.stem]
    return max(dates, default="")


def _mtime_iso(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return ""


def _latest_mtime_file(paths: list[Path]) -> Path | None:
    latest: tuple[float, Path] | None = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if latest is None or mtime > latest[0]:
            latest = (mtime, path)
    return latest[1] if latest else None


def _default_data_dir() -> Path:
    return Path(os.environ.get("ANIMAWORKS_DATA_DIR") or Path.home() / ".animaworks").expanduser()


def _summary(rows: list[FactsHealthRow]) -> dict[str, Any]:
    active = [row for row in rows if row.status == "active"]
    active_with_facts = [row for row in active if row.facts_count > 0]
    return {
        "animas": len(rows),
        "active": len(active),
        "active_with_facts": len(active_with_facts),
        "dormant": sum(1 for row in rows if row.status == "dormant"),
        "facts_count": sum(row.facts_count for row in rows),
    }


def render_table(rows: list[FactsHealthRow]) -> str:
    headers = ("anima", "status", "facts", "files", "last_fact", "last_episode", "last_mtime_utc")
    widths = [len(header) for header in headers]
    values = [
        (
            row.anima,
            row.status,
            str(row.facts_count),
            str(row.facts_files),
            row.last_fact_date or "-",
            row.last_episode_date or "-",
            row.last_fact_mtime or "-",
        )
        for row in rows
    ]
    for value in values:
        widths = [max(width, len(cell)) for width, cell in zip(widths, value, strict=True)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(cell.ljust(width) for cell, width in zip(value, widths, strict=True)) for value in values)
    summary = _summary(rows)
    lines.append("")
    lines.append(
        "summary: "
        f"active_with_facts={summary['active_with_facts']}/{summary['active']} "
        f"dormant={summary['dormant']} facts_count={summary['facts_count']}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    rows = collect_facts_health(args.data_dir.expanduser())
    if args.json:
        print(json.dumps({"summary": _summary(rows), "animas": [asdict(row) for row in rows]}, ensure_ascii=False))
    else:
        print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
