#!/usr/bin/env python3
"""Format AnimaWorks prompt log JSONL into readable Markdown.

Usage:
    python tools/format_prompt_log.py <input.jsonl> [-n NUM] [-o output.md] [--all]

Options:
    -n NUM      Number of entries to format from the end (default: 1)
    --all       Format all entries
    -o FILE     Output file (default: <input_stem>_formatted.md)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _header(entry: dict) -> str:
    lines = [
        "# Prompt Log Entry",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Timestamp | `{entry.get('ts', '')}` |",
        f"| Trigger | `{entry.get('trigger', '')}` |",
        f"| From | `{entry.get('from', '')}` |",
        f"| Model | `{entry.get('model', '')}` |",
        f"| Mode | `{entry.get('mode', '')}` |",
        f"| Session ID | `{entry.get('session_id', '')}` |",
        f"| System Prompt Length | {entry.get('system_prompt_length', len(entry.get('system_prompt', '')))} chars |",
        f"| User Message Length | {len(entry.get('user_message', ''))} chars |",
        f"| Tools | {', '.join(entry.get('tools', []))} |",
        "",
    ]
    return "\n".join(lines)


def _section(title: str, content: str, lang: str = "") -> str:
    sep = "=" * 60
    lines = [
        f"## {title}",
        "",
        f"<!-- {sep} -->",
        "",
        content,
        "",
    ]
    return "\n".join(lines)


def format_entry(entry: dict, index: int) -> str:
    parts: list[str] = []
    if index > 0:
        parts.append("\n---\n")

    parts.append(_header(entry))
    parts.append(_section("System Prompt", entry.get("system_prompt", "")))
    parts.append(_section("User Message", entry.get("user_message", "")))

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format AnimaWorks prompt log JSONL into readable Markdown.",
    )
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("-n", type=int, default=1, help="Number of entries from the end (default: 1)")
    parser.add_argument("--all", action="store_true", help="Format all entries")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file")
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_name(input_path.stem + "_formatted.md")

    entries: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not args.all:
        entries = entries[-args.n:]

    parts: list[str] = []
    for i, entry in enumerate(entries):
        parts.append(format_entry(entry, i))

    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Formatted {len(entries)} entries -> {output_path}")


if __name__ == "__main__":
    main()
