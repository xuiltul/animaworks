#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Idempotent migration: add explicit allow entries for newly gated actions.

pi-fix2 gates chatwork_send / discord_send / github_create-issue /
github_create-pr. Live Animas that already use these actions
need action-level allows so day-to-day work is not blocked.

Default is **dry-run** (print planned changes only). Pass ``--apply`` to write.

Does **not** modify permissions.py semantics, EXECUTION_PROFILE, or templates.

Usage::

    python scripts/migrate_pi_fix2_gated_allows.py
    python scripts/migrate_pi_fix2_gated_allows.py --apply
    python scripts/migrate_pi_fix2_gated_allows.py --data-dir ~/.animaworks --apply
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Allow distribution policy (2026-08-01 PdM + usage scan) ──────────────
#
# chatwork_send: PdM list of animas with real send usage (exclude
#   sumire/ayame/yoru).
# discord_send: animas with discord send usage signals.
# github_create-issue / github_create-pr: animas with github create usage.

DEFAULT_ALLOWS: dict[str, list[str]] = {
    "chatwork_send": [
        "aoi",
        "kotoha",
        "mei",
        "mio",
        "nagi",
        "natsume",
        "rin",
        "ritsu",
        "sakura",
        "sora",
    ],
    "discord_send": [
        "aoi",
        "mei",
        "mio",
        "nagi",
        "natsume",
        "rin",
        "ritsu",
        "sakura",
        "sora",
    ],
    "github_create-issue": [
        "aoi",
        "ayame",
        "kotoha",
        "mei",
        "mio",
        "natsume",
        "rin",
        "ritsu",
        "sakura",
        "sora",
        "sumire",
    ],
    "github_create-pr": [
        "aoi",
        "ayame",
        "kotoha",
        "mei",
        "mio",
        "natsume",
        "rin",
        "ritsu",
        "sakura",
        "sora",
        "sumire",
    ],
}


def _print(msg: str) -> None:
    print(msg)


def _load_permissions(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _print(f"ERROR reading {path}: {exc}")
        return None
    if not isinstance(data, dict):
        _print(f"ERROR {path}: root is not an object")
        return None
    return data


def _planned_adds(
    anima: str,
    allow: list[str],
    deny: list[str],
    policy: dict[str, list[str]],
) -> list[str]:
    """Return action keys to append for this anima (order-stable, deduped)."""
    allow_set = set(allow)
    deny_set = set(deny)
    adds: list[str] = []
    for action, animas in policy.items():
        if anima not in animas:
            continue
        tool = action.split("_", 1)[0]
        if action in deny_set or tool in deny_set:
            continue
        if action in allow_set:
            continue
        adds.append(action)
    return adds


def migrate_one(
    anima_dir: Path,
    *,
    apply: bool,
    policy: dict[str, list[str]],
) -> dict[str, Any]:
    name = anima_dir.name
    perm_path = anima_dir / "permissions.json"
    data = _load_permissions(perm_path)
    if data is None:
        return {"anima": name, "status": "skip", "reason": "no permissions.json", "adds": []}

    ext = data.get("external_tools")
    if not isinstance(ext, dict):
        ext = {}
        data["external_tools"] = ext
    allow = list(ext.get("allow") or [])
    deny = list(ext.get("deny") or [])
    adds = _planned_adds(name, allow, deny, policy)
    if not adds:
        return {"anima": name, "status": "noop", "adds": [], "allow": allow, "deny": deny}

    new_allow = allow + adds
    if apply:
        ext["allow"] = new_allow
        # Preserve other keys; keep formatting stable
        perm_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "anima": name,
            "status": "applied",
            "adds": adds,
            "allow_before": allow,
            "allow_after": new_allow,
            "deny": deny,
        }
    return {
        "anima": name,
        "status": "would_apply",
        "adds": adds,
        "allow_before": allow,
        "allow_after": new_allow,
        "deny": deny,
    }


def run(
    data_dir: Path,
    *,
    apply: bool,
    policy: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    policy = policy or DEFAULT_ALLOWS
    animas_dir = data_dir / "animas"
    if not animas_dir.is_dir():
        raise SystemExit(f"animas dir not found: {animas_dir}")

    results: list[dict[str, Any]] = []
    mode = "APPLY" if apply else "DRY-RUN"
    _print(f"=== pi-fix2 gated allows migration ({mode}) ===")
    _print(f"data_dir: {data_dir}")
    _print("")

    for anima_dir in sorted(p for p in animas_dir.iterdir() if p.is_dir()):
        result = migrate_one(anima_dir, apply=apply, policy=policy)
        results.append(result)
        status = result["status"]
        adds = result.get("adds") or []
        if status == "noop":
            _print(f"  {result['anima']}: OK (no changes)")
        elif status == "skip":
            _print(f"  {result['anima']}: SKIP ({result.get('reason')})")
        elif status == "would_apply":
            _print(f"  {result['anima']}: WOULD ADD {adds}")
            _print(f"    allow: {result.get('allow_before')} → {result.get('allow_after')}")
        elif status == "applied":
            _print(f"  {result['anima']}: APPLIED +{adds}")
            _print(f"    allow: {result.get('allow_before')} → {result.get('allow_after')}")
        else:
            _print(f"  {result['anima']}: {status} {adds}")

    would = [r for r in results if r["status"] in ("would_apply", "applied")]
    _print("")
    _print(f"summary: {len(would)} anima(s) with allow additions, {len(results)} total")
    if not apply:
        _print("dry-run only — re-run with --apply to write permissions.json")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".animaworks",
        help="AnimaWorks data directory (default: ~/.animaworks)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default: dry-run)",
    )
    args = parser.parse_args(argv)
    run(args.data_dir.expanduser(), apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
