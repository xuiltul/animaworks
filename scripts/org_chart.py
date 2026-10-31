#!/usr/bin/env python3
"""AnimaWorks Organisation Chart CLI tool.

Reads config.json and anima directories to display the org tree.

Usage:
    python scripts/org_chart.py              # Text tree (default)
    python scripts/org_chart.py --json       # JSON output
    python scripts/org_chart.py --all        # Include disabled animas
    python scripts/org_chart.py --api        # Fetch from running server API
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))

DEFAULT_DATA_DIR = Path.home() / ".animaworks"
DEFAULT_CONFIG = DEFAULT_DATA_DIR / "config.json"
DEFAULT_ANIMAS_DIR = DEFAULT_DATA_DIR / "animas"


def load_org_data(
    config_path: Path = DEFAULT_CONFIG,
    animas_dir: Path = DEFAULT_ANIMAS_DIR,
    include_disabled: bool = True,
) -> dict:
    """Build org chart data from config.json and anima directories."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    animas_cfg = config.get("animas", {})
    defaults = config.get("anima_defaults", {})

    # Discover all anima names from disk
    all_names: list[str] = []
    if animas_dir.exists():
        for d in sorted(animas_dir.iterdir()):
            if d.is_dir() and (d / "IDENTITY.md").exists():
                all_names.append(d.name)

    # Check enabled status
    def is_enabled(name: str) -> bool:
        status_file = animas_dir / name / "status.json"
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text(encoding="utf-8"))
                return data.get("enabled", True)
            except (json.JSONDecodeError, OSError):
                pass
        return True

    # Build flat lookup
    flat: dict[str, dict] = {}
    for name in all_names:
        enabled = is_enabled(name)
        if not include_disabled and not enabled:
            continue

        acfg = animas_cfg.get(name, {})
        flat[name] = {
            "name": name,
            "speciality": acfg.get("speciality", defaults.get("speciality")),
            "supervisor": acfg.get("supervisor", defaults.get("supervisor")),
            "model": acfg.get("model", defaults.get("model")),
            "role": _read_role(animas_dir / name),
            "enabled": enabled,
        }

    return flat


def _read_role(anima_dir: Path) -> str | None:
    """Read role from status.json."""
    status_file = anima_dir / "status.json"
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            return data.get("role")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def build_tree(flat: dict[str, dict]) -> list[dict]:
    """Build tree structure from flat lookup."""

    def _build_node(name: str) -> dict:
        info = flat[name]
        children_names = sorted(
            n for n, d in flat.items() if d["supervisor"] == name
        )
        return {
            "name": name,
            "speciality": info["speciality"],
            "model": info["model"],
            "enabled": info["enabled"],
            "children": [_build_node(c) for c in children_names],
        }

    roots = sorted(n for n, d in flat.items() if d["supervisor"] is None)
    return [_build_node(r) for r in roots]


def render_text_tree(tree: list[dict], total: int) -> str:
    """Render tree as ASCII art."""
    lines: list[str] = []
    lines.append("AnimaWorks Organisation Chart")
    lines.append(f"Generated: {datetime.now(tz=JST).strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 50)

    def _render(node: dict, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        if not node["enabled"]:
            mark = "✗"
        else:
            mark = "✓"
        spec = node["speciality"] or "?"
        label = f"{node['name']} [{spec}] ({mark})"
        lines.append(f"{prefix}{connector}{label}")
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node["children"]):
            _render(child, child_prefix, i == len(node["children"]) - 1)

    for i, root in enumerate(tree):
        _render(root, "", i == len(tree) - 1)

    lines.append("")
    lines.append(f"Total: {total} animas")
    return "\n".join(lines)


def fetch_from_api(base_url: str = "http://localhost:18500", include_disabled: bool = False) -> dict:
    """Fetch org chart from running server API."""
    import urllib.request

    params = "?include_disabled=true" if include_disabled else ""
    url = f"{base_url}/api/org/chart{params}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="AnimaWorks Organisation Chart")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Include disabled animas (default: true)")
    parser.add_argument("--enabled-only", action="store_true",
                        help="Show only enabled animas")
    parser.add_argument("--api", action="store_true",
                        help="Fetch from running server API instead of config files")
    parser.add_argument("--url", default="http://localhost:18500",
                        help="Server URL (for --api mode)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to config.json")
    parser.add_argument("--animas-dir", type=Path, default=DEFAULT_ANIMAS_DIR,
                        help="Path to animas directory")
    args = parser.parse_args()

    include_disabled = not args.enabled_only

    if args.api:
        try:
            data = fetch_from_api(args.url, include_disabled=include_disabled)
        except Exception as e:
            print(f"Error fetching from API: {e}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            # Render text from API response
            tree = data.get("tree", [])
            total = data.get("total", 0)
            # API tree nodes have slightly different structure, normalize
            def _normalize(node: dict) -> dict:
                return {
                    "name": node["name"],
                    "speciality": node.get("speciality"),
                    "model": node.get("model"),
                    "enabled": node.get("enabled", node.get("status") != "disabled"),
                    "children": [_normalize(c) for c in node.get("children", [])],
                }
            normalized = [_normalize(n) for n in tree]
            print(render_text_tree(normalized, total))
        return

    # Direct file read mode
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    flat = load_org_data(args.config, args.animas_dir, include_disabled=include_disabled)
    tree = build_tree(flat)

    if args.json:
        output = {
            "generated_at": datetime.now(tz=JST).isoformat(),
            "total": len(flat),
            "tree": tree,
            "flat": flat,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(render_text_tree(tree, len(flat)))


if __name__ == "__main__":
    main()
