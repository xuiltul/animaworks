#!/usr/bin/env python3
"""Dump the full system prompt for a given Anima.

Usage:
    python tools/dump_prompt.py [anima_name] [test_message]

Examples:
    python tools/dump_prompt.py sakura
    python tools/dump_prompt.py sakura "mcpサーバーツールはあなたは知っていますか"

Output:
    - Writes the prompt to ~/.animaworks/{anima_name}_prompt_dump.md
    - Prints summary statistics to stdout
"""
from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

# ── Ensure project root is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    anima_name = sys.argv[1] if len(sys.argv) > 1 else "sakura"
    test_message = sys.argv[2] if len(sys.argv) > 2 else "mcpサーバーツールはあなたは知っていますか"

    # ── Resolve paths ──────────────────────────────────────
    from core.paths import get_data_dir, get_shared_dir

    data_dir = get_data_dir()
    anima_dir = data_dir / "animas" / anima_name

    if not anima_dir.exists():
        print(f"ERROR: Anima directory not found: {anima_dir}")
        sys.exit(1)

    print(f"Anima directory : {anima_dir}")
    print(f"Test message    : {test_message}")
    print()

    # ── Initialize MemoryManager ───────────────────────────
    from core.memory.manager import MemoryManager

    memory = MemoryManager(anima_dir)

    # ── Build tool_registry (mimic agent._init_tool_registry) ──
    tool_registry: list[str] = []
    try:
        from core.tools import TOOL_MODULES

        all_tools = sorted(TOOL_MODULES.keys())
        permissions_text = memory.read_permissions()

        if "外部ツール" not in permissions_text:
            tool_registry = all_tools
        elif "all: yes" in permissions_text:
            tool_registry = all_tools
        else:
            tool_registry = all_tools  # fallback: default-all
    except Exception as e:
        print(f"WARN: Could not load tool_registry: {e}")
        tool_registry = []

    print(f"Tool registry   : {len(tool_registry)} tools")

    # ── Run PrimingEngine (optional, with fallback) ────────
    priming_section = ""
    try:
        from core.memory.priming import PrimingEngine, format_priming_section

        shared_dir = get_shared_dir()
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

        # Derive channel from trigger arg
        _trigger = sys.argv[3] if len(sys.argv) > 3 else ""
        _channel = (
            "heartbeat" if _trigger == "heartbeat"
            else "cron" if _trigger.startswith("cron")
            else "chat"
        )
        priming_result = asyncio.run(
            engine.prime_memories(
                message=test_message,
                sender_name="human",
                channel=_channel,
                enable_dynamic_budget=True,
            )
        )

        priming_section = format_priming_section(priming_result, sender_name="human")
        print(f"Priming section : {len(priming_section)} chars")
        if priming_result.sender_profile:
            print(f"  - sender_profile   : {len(priming_result.sender_profile)} chars")
        if priming_result.recent_activity:
            print(f"  - recent_activity  : {len(priming_result.recent_activity)} chars")
        if priming_result.related_knowledge:
            print(f"  - related_knowledge: {len(priming_result.related_knowledge)} chars")
        if priming_result.matched_skills:
            print(f"  - matched_skills   : {priming_result.matched_skills}")
        if priming_result.pending_tasks:
            print(f"  - pending_tasks    : {len(priming_result.pending_tasks)} chars")
    except Exception as e:
        print(f"WARN: Priming failed (using empty section): {e}")
        priming_section = ""

    print()

    # ── Build system prompt ────────────────────────────────
    from core.prompt.builder import build_system_prompt

    # Determine trigger from CLI args (default: user message)
    trigger = sys.argv[3] if len(sys.argv) > 3 else ""

    result = build_system_prompt(
        memory=memory,
        tool_registry=tool_registry,
        execution_mode="a1",
        message=test_message,
        priming_section=priming_section,
        trigger=trigger,
    )

    prompt_text = result.system_prompt

    # ── Analysis ───────────────────────────────────────────
    # Count section headers (## or # lines)
    section_headers = re.findall(r"^#{1,3}\s+.+$", prompt_text, re.MULTILINE)
    # Count separator blocks
    separator_count = prompt_text.count("\n\n---\n\n")
    # Check for notable features
    has_newstaff = "newstaff" in prompt_text.lower()
    has_bootstrap = "bootstrap" in prompt_text.lower() or "初回起動" in prompt_text
    has_priming = "あなたが思い出していること" in prompt_text
    has_org_context = "組織構成" in prompt_text or "上司" in prompt_text
    has_emotion = "EMOTION" in prompt_text or "表情" in prompt_text
    has_tools_guide = "外部ツール" in prompt_text
    has_hiring_rules = "雇用ルール" in prompt_text or "create_anima" in prompt_text
    has_human_notification = "人間通知" in prompt_text or "call_human" in prompt_text

    # ── Write dump file ────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = data_dir / f"{anima_name}_prompt_dump.md"

    header = f"""---
generated: {timestamp}
anima: {anima_name}
test_message: "{test_message}"
execution_mode: a1
total_length: {len(prompt_text)}
sections: {separator_count + 1}
section_headers: {len(section_headers)}
has_priming: {has_priming}
has_newstaff: {has_newstaff}
has_bootstrap: {has_bootstrap}
has_tools_guide: {has_tools_guide}
---

"""

    output_path.write_text(header + prompt_text, encoding="utf-8")
    print(f"Dump written to : {output_path}")
    print(f"File size       : {output_path.stat().st_size:,} bytes")
    print()

    # ── Print summary ──────────────────────────────────────
    print("=" * 60)
    print("SYSTEM PROMPT SUMMARY")
    print("=" * 60)
    print(f"Total length    : {len(prompt_text):,} chars")
    print(f"Sections (---): {separator_count + 1}")
    print(f"Section headers : {len(section_headers)}")
    print()
    print("Feature flags:")
    print(f"  bootstrap       : {'YES' if has_bootstrap else 'no'}")
    print(f"  priming         : {'YES' if has_priming else 'no'}")
    print(f"  org_context     : {'YES' if has_org_context else 'no'}")
    print(f"  emotion         : {'YES' if has_emotion else 'no'}")
    print(f"  tools_guide     : {'YES' if has_tools_guide else 'no'}")
    print(f"  newstaff        : {'YES' if has_newstaff else 'no'}")
    print(f"  hiring_rules    : {'YES' if has_hiring_rules else 'no'}")
    print(f"  human_notification: {'YES' if has_human_notification else 'no'}")
    print()

    print("Section headers found:")
    for i, h in enumerate(section_headers, 1):
        print(f"  {i:2d}. {h.strip()}")
    print()

    if result.injected_procedures:
        print("Injected procedures:")
        for p in result.injected_procedures:
            print(f"  - {p}")
        print()


if __name__ == "__main__":
    main()
