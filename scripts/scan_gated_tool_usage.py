#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan live Anima permissions + activity_log for gated-candidate tool usage.

Efficient: uses ``rg -c --fixed-strings`` per anima (no full-file loads).
Target runtime under ~2 minutes for a typical multi-Anima data dir.

Usage::

    python scripts/scan_gated_tool_usage.py
    python scripts/scan_gated_tool_usage.py --data-dir ~/.animaworks
    python scripts/scan_gated_tool_usage.py --out docs/records/YYYYMMDD_gated-tool-usage-scan.md
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Needles counted in activity_log/*.jsonl (fixed-string, case-sensitive).
# Counts are presence signals for allow-list planning, not exact call tallies
# (skill docs / knowledge mentions may also match).
SCAN_NEEDLES: dict[str, tuple[str, ...]] = {
    "chatwork_send": ("chatwork_send", "chatwork send"),
    "discord_send": ("discord_send", "discord send"),
    "github_create-issue": ("github create-issue", "github_create-issue"),
    "github_create-pr": ("github create-pr", "github_create-pr"),
}

# Gated action keys that this pi-fix2 change introduces.
GATED_ACTIONS: tuple[str, ...] = (
    "chatwork_send",
    "discord_send",
    "github_create-issue",
    "github_create-pr",
)


def _rg_count(needle: str, path: Path) -> int:
    """Return match count for fixed-string needle under path (0 if none/rg missing)."""
    if not path.is_dir():
        return 0
    try:
        proc = subprocess.run(
            ["rg", "-c", "--fixed-strings", needle, str(path), "-g", "*.jsonl"],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    total = 0
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        try:
            total += int(line.rsplit(":", 1)[1])
        except ValueError:
            continue
    return total


def _load_external_tools(perm_path: Path) -> dict:
    if not perm_path.is_file():
        return {}
    try:
        data = json.loads(perm_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("external_tools") or {}


def scan_animas(data_dir: Path) -> dict:
    animas_dir = data_dir / "animas"
    if not animas_dir.is_dir():
        raise SystemExit(f"animas dir not found: {animas_dir}")

    rows: list[dict] = []
    for anima_dir in sorted(p for p in animas_dir.iterdir() if p.is_dir()):
        name = anima_dir.name
        ext = _load_external_tools(anima_dir / "permissions.json")
        allow = list(ext.get("allow") or [])
        deny = list(ext.get("deny") or [])
        allow_all = bool(ext.get("allow_all", False))
        log_dir = anima_dir / "activity_log"
        usage: dict[str, int] = {}
        for action, needles in SCAN_NEEDLES.items():
            usage[action] = sum(_rg_count(n, log_dir) for n in needles)
        rows.append(
            {
                "anima": name,
                "allow_all": allow_all,
                "allow": allow,
                "deny": deny,
                "usage": usage,
            }
        )
    return {"data_dir": str(data_dir), "animas": rows}


def _has_usage(usage: dict[str, int], action: str) -> bool:
    return int(usage.get(action, 0)) > 0


def recommend_allows(scan: dict) -> dict[str, list[str]]:
    """Map gated action → anima names that need explicit allow after gating."""
    rec: dict[str, list[str]] = {a: [] for a in GATED_ACTIONS}
    for row in scan["animas"]:
        name = row["anima"]
        allow = set(row["allow"])
        deny = set(row["deny"])
        usage = row["usage"]
        for action in GATED_ACTIONS:
            tool = action.split("_", 1)[0]
            if tool in deny or action in deny:
                continue
            if action in allow:
                continue
            if _has_usage(usage, action):
                rec[action].append(name)
    return rec


def format_report(scan: dict, rec: dict[str, list[str]], elapsed_s: float) -> str:
    lines: list[str] = []
    lines.append("# Gated tool usage scan")
    lines.append("")
    lines.append(f"- data_dir: `{scan['data_dir']}`")
    lines.append(f"- animas: {len(scan['animas'])}")
    lines.append(f"- elapsed_s: {elapsed_s:.1f}")
    lines.append("")
    lines.append("## Per-Anima permissions (external_tools)")
    lines.append("")
    lines.append("| anima | allow_all | allow | deny |")
    lines.append("|---|---|---|---|")
    for row in scan["animas"]:
        lines.append(
            f"| {row['anima']} | {row['allow_all']} | "
            f"{', '.join(row['allow']) or '—'} | "
            f"{', '.join(row['deny']) or '—'} |"
        )
    lines.append("")
    lines.append("## activity_log needle counts (signals, not exact call tallies)")
    lines.append("")
    header = "| anima | " + " | ".join(GATED_ACTIONS) + " |"
    sep = "|---|" + "|".join(["---"] * len(GATED_ACTIONS)) + "|"
    lines.append(header)
    lines.append(sep)
    for row in scan["animas"]:
        cells = [str(row["usage"].get(a, 0)) for a in GATED_ACTIONS]
        lines.append(f"| {row['anima']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Recommended allow additions (usage > 0 and not already allowed/denied)")
    lines.append("")
    for action in GATED_ACTIONS:
        names = rec.get(action) or []
        lines.append(f"- **{action}**: {', '.join(names) if names else '(none)'}")
    lines.append("")
    lines.append("## PdM allow policy (pi-fix2, 2026-08-01)")
    lines.append("")
    lines.append(
        "- `chatwork_send`: animas with chatwork send usage "
        "(exclude sumire/ayame/yoru when usage is noise-only / PdM list)"
    )
    lines.append(
        "- `discord_send` / `github_create-*`: usage-based only "
        "(see recommended list above)"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".animaworks",
        help="AnimaWorks data directory (default: ~/.animaworks)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write markdown report to this path (also prints to stdout)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write raw JSON scan result",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    scan = scan_animas(args.data_dir.expanduser())
    rec = recommend_allows(scan)
    elapsed = time.time() - t0
    report = format_report(scan, rec, elapsed)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nWrote report: {args.out}", file=sys.stderr)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scan": scan, "recommended_allows": rec, "elapsed_s": elapsed}
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote json: {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
