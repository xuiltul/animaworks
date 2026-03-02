#!/usr/bin/env python3
"""Debug script: extract priming layer output for each trigger mode.

Usage:
    python scripts/debug_priming.py [anima_name] [--message "text"]

Outputs separate files under /tmp/priming_debug/ for each trigger mode:
  - chat.md        (channel="chat", sender="human")
  - inbox.md       (channel="chat", sender="rin")
  - heartbeat.md   (channel="heartbeat")
  - cron.md        (channel="cron")
  - task.md        (skipped by design)

Also outputs raw PrimingResult fields for inspection.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import get_data_dir, get_shared_dir, get_animas_dir
from core.prompt.context import resolve_context_window
from core.prompt.builder import (
    build_system_prompt,
    resolve_prompt_tier,
    TIER_FULL,
    TIER_STANDARD,
    TIER_LIGHT,
    TIER_MINIMAL,
)
from core.memory import MemoryManager
from core.memory.priming import PrimingEngine, PrimingResult, format_priming_section
from core.i18n import t

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("debug_priming")

OUTPUT_DIR = Path("/tmp/priming_debug")

# ── Trigger definitions ───────────────────────────────────────

TRIGGERS = [
    {
        "name": "chat",
        "trigger": "message:human",
        "channel": "chat",
        "sender": "human",
        "message": None,  # filled at runtime
        "description": "人間からのチャットメッセージ",
    },
    {
        "name": "inbox",
        "trigger": "inbox:rin",
        "channel": "chat",
        "sender": "rin",
        "message": "進捗報告です。APIの実装が完了しました。",
        "description": "他Animaからの受信メッセージ (inbox)",
    },
    {
        "name": "heartbeat",
        "trigger": "heartbeat",
        "channel": "heartbeat",
        "sender": "system",
        "message": "定期巡回を実行してください。",
        "description": "ハートビート（定期巡回）",
    },
    {
        "name": "cron",
        "trigger": "cron:daily_report",
        "channel": "cron",
        "sender": "system",
        "message": "日次レポート生成タスクを実行してください。",
        "description": "Cronタスク実行",
    },
    {
        "name": "task",
        "trigger": "task:pending_123",
        "channel": "chat",
        "sender": "system",
        "message": "ログ分析タスクを実行してください。",
        "description": "TaskExec（プライミングはスキップされる）",
    },
]


def _result_to_dict(result: PrimingResult) -> dict:
    """Convert PrimingResult to a serializable dict with char counts."""
    return {
        "sender_profile": {
            "chars": len(result.sender_profile),
            "content": result.sender_profile[:500] + ("..." if len(result.sender_profile) > 500 else ""),
        },
        "recent_activity": {
            "chars": len(result.recent_activity),
            "content": result.recent_activity[:1000] + ("..." if len(result.recent_activity) > 1000 else ""),
        },
        "related_knowledge": {
            "chars": len(result.related_knowledge),
            "content": result.related_knowledge[:500] + ("..." if len(result.related_knowledge) > 500 else ""),
        },
        "related_knowledge_untrusted": {
            "chars": len(result.related_knowledge_untrusted),
            "content": result.related_knowledge_untrusted[:500]
            + ("..." if len(result.related_knowledge_untrusted) > 500 else ""),
        },
        "matched_skills": result.matched_skills,
        "pending_tasks": {
            "chars": len(result.pending_tasks),
            "content": result.pending_tasks[:500] + ("..." if len(result.pending_tasks) > 500 else ""),
        },
        "recent_outbound": {
            "chars": len(result.recent_outbound),
            "content": result.recent_outbound[:500] + ("..." if len(result.recent_outbound) > 500 else ""),
        },
        "total_chars": result.total_chars(),
        "estimated_tokens": result.estimated_tokens(),
        "is_empty": result.is_empty(),
    }


def _apply_tier_filter(
    result: PrimingResult,
    formatted: str,
    sender_name: str,
    tier: str,
) -> str:
    """Apply the same tier-based filtering as _agent_priming.py."""
    if tier == TIER_MINIMAL:
        return "(TIER_MINIMAL: プライミングはスキップ)"

    if tier == TIER_LIGHT:
        if result.sender_profile:
            return (
                t("agent.priming_tier_light_header", sender_name=sender_name)
                + result.sender_profile
            )
        return "(TIER_LIGHT: sender_profileが空のためスキップ)"

    if tier == TIER_STANDARD and len(formatted) > 4000:
        return formatted[:4000] + t("agent.omitted_rest")

    return formatted


async def run_priming_for_trigger(
    engine: PrimingEngine,
    trigger_def: dict,
    default_message: str,
    tier: str,
) -> tuple[PrimingResult | None, str, str]:
    """Run priming for a single trigger definition.

    Returns (raw_result, formatted_section, tier_filtered_section).
    """
    is_task = trigger_def["trigger"].startswith("task:")
    if is_task:
        return None, "", "(task: トリガーではプライミングはスキップされる)"

    message = trigger_def["message"] or default_message
    channel = trigger_def["channel"]
    sender = trigger_def["sender"]

    result = await engine.prime_memories(
        message,
        sender,
        channel=channel,
        enable_dynamic_budget=True,
    )

    if result.is_empty():
        return result, "", "(結果が空)"

    formatted = format_priming_section(result, sender)
    filtered = _apply_tier_filter(result, formatted, sender, tier)

    return result, formatted, filtered


async def main() -> None:
    parser = argparse.ArgumentParser(description="Debug priming layer output")
    parser.add_argument("anima", nargs="?", help="Anima name (auto-detects first available)")
    parser.add_argument("--message", "-m", default="最近のシステムの状態を確認して、問題があれば教えて。", help="Test message for chat trigger")
    parser.add_argument("--context-window", "-c", type=int, default=0, help="Override context window size")
    args = parser.parse_args()

    # Resolve anima
    animas_dir = get_animas_dir()
    if args.anima:
        anima_dir = animas_dir / args.anima
    else:
        available = sorted(
            [d.name for d in animas_dir.iterdir() if d.is_dir() and (d / "status.json").exists()],
        )
        if not available:
            logger.error("No animas found in %s", animas_dir)
            sys.exit(1)
        anima_dir = animas_dir / available[0]
        logger.info("Auto-selected anima: %s (from %d available: %s)", anima_dir.name, len(available), ", ".join(available[:5]))

    if not anima_dir.is_dir():
        logger.error("Anima directory not found: %s", anima_dir)
        sys.exit(1)

    # Setup
    shared_dir = get_shared_dir()
    memory = MemoryManager(anima_dir)
    model_config = memory.read_model_config()
    model_name = model_config.model or "claude-sonnet-4-6"

    if args.context_window:
        ctx_window = args.context_window
    else:
        ctx_window = resolve_context_window(model_name)

    tier = resolve_prompt_tier(ctx_window)

    logger.info("=" * 60)
    logger.info("Priming Debug Report")
    logger.info("=" * 60)
    logger.info("Anima: %s", anima_dir.name)
    logger.info("Model: %s", model_name)
    logger.info("Context window: %d", ctx_window)
    logger.info("Prompt tier: %s", tier)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = PrimingEngine(anima_dir, shared_dir, context_window=ctx_window)

    summary_lines: list[str] = []
    summary_lines.append(f"# Priming Debug Report\n")
    summary_lines.append(f"- **Anima**: {anima_dir.name}")
    summary_lines.append(f"- **Model**: {model_name}")
    summary_lines.append(f"- **Context window**: {ctx_window:,}")
    summary_lines.append(f"- **Prompt tier**: {tier}")
    summary_lines.append(f"- **Test message (chat)**: {args.message}")
    summary_lines.append("")

    for trigger_def in TRIGGERS:
        name = trigger_def["name"]
        logger.info("--- Running: %s (%s) ---", name, trigger_def["description"])

        result, formatted, filtered = await run_priming_for_trigger(
            engine, trigger_def, args.message, tier,
        )

        # Write formatted priming section
        out_file = OUTPUT_DIR / f"{name}.md"
        header = (
            f"# Priming: {name}\n\n"
            f"- Trigger: `{trigger_def['trigger']}`\n"
            f"- Channel: `{trigger_def['channel']}`\n"
            f"- Sender: `{trigger_def['sender']}`\n"
            f"- Description: {trigger_def['description']}\n"
            f"- Tier: {tier}\n\n"
            f"---\n\n"
        )
        out_file.write_text(header + filtered, encoding="utf-8")

        # Write raw result JSON
        if result is not None:
            raw_file = OUTPUT_DIR / f"{name}_raw.json"
            raw_file.write_text(
                json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Summary
        if result is None:
            summary_lines.append(f"## {name} ({trigger_def['description']})")
            summary_lines.append(f"- **Skipped** (task trigger)")
            summary_lines.append("")
        else:
            summary_lines.append(f"## {name} ({trigger_def['description']})")
            summary_lines.append(f"- Trigger: `{trigger_def['trigger']}`")
            summary_lines.append(f"- Channel: `{trigger_def['channel']}`")
            summary_lines.append(f"- Sender: `{trigger_def['sender']}`")
            summary_lines.append(f"- Empty: {result.is_empty()}")
            summary_lines.append(f"- Total chars: {result.total_chars():,}")
            summary_lines.append(f"- Estimated tokens: {result.estimated_tokens():,}")
            summary_lines.append(f"- Formatted length: {len(formatted):,} chars")
            summary_lines.append(f"- After tier filter: {len(filtered):,} chars")
            summary_lines.append(f"- Channels:")
            summary_lines.append(f"  - sender_profile: {len(result.sender_profile):,} chars")
            summary_lines.append(f"  - recent_activity: {len(result.recent_activity):,} chars")
            summary_lines.append(f"  - related_knowledge: {len(result.related_knowledge):,} chars")
            summary_lines.append(f"  - related_knowledge_untrusted: {len(result.related_knowledge_untrusted):,} chars")
            summary_lines.append(f"  - matched_skills: {result.matched_skills}")
            summary_lines.append(f"  - pending_tasks: {len(result.pending_tasks):,} chars")
            summary_lines.append(f"  - recent_outbound: {len(result.recent_outbound):,} chars")
            summary_lines.append("")

        logger.info("  -> Written to %s", out_file)

    # Write summary
    summary_file = OUTPUT_DIR / "summary.md"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Summary written to %s", summary_file)

    # Also build full system prompts for chat and heartbeat for comparison
    for mode_name, trigger_val in [("chat_full_prompt", "message:human"), ("heartbeat_full_prompt", "heartbeat")]:
        is_task = trigger_val.startswith("task:")
        if is_task:
            priming_section = ""
        else:
            channel = "heartbeat" if trigger_val == "heartbeat" else "chat"
            sender = "human" if channel == "chat" else "system"
            msg = args.message if channel == "chat" else "定期巡回"
            r = await engine.prime_memories(msg, sender, channel=channel, enable_dynamic_budget=True)
            if r.is_empty():
                priming_section = ""
            else:
                priming_section = format_priming_section(r, sender)
                priming_section = _apply_tier_filter(r, priming_section, sender, tier)

        build_result = build_system_prompt(
            memory,
            priming_section=priming_section,
            execution_mode="a",
            message=args.message,
            trigger=trigger_val,
            context_window=ctx_window,
        )

        full_file = OUTPUT_DIR / f"{mode_name}.md"
        full_file.write_text(build_result.system_prompt, encoding="utf-8")
        logger.info("Full system prompt (%s) written to %s (%d chars)", mode_name, full_file, len(build_result.system_prompt))

    logger.info("=" * 60)
    logger.info("Done. All files in %s", OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
