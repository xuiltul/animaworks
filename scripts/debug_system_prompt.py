#!/usr/bin/env python3
"""Debug system prompt visualizer — build & render as color-coded HTML.

Usage:
    python scripts/debug_system_prompt.py [anima_name] [test_message] [trigger]

Examples:
    python scripts/debug_system_prompt.py sakura
    python scripts/debug_system_prompt.py sakura "mcpサーバーツールはあなたは知っていますか"
    python scripts/debug_system_prompt.py sakura "" heartbeat

Output:
    - /tmp/prompt_debug_{anima_name}.html
    - Optionally copies to server/static/files/ if the directory exists
"""
from __future__ import annotations

import asyncio
import html
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ── Ensure project root is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Section Categories ──────────────────────────────────────

@dataclass
class CategoryStyle:
    bg: str
    fg: str


CATEGORIES: dict[str, CategoryStyle] = {
    "header":       CategoryStyle("#1f2937", "#f9fafb"),
    "framework":    CategoryStyle("#dbeafe", "#1e3a5f"),
    "identity":     CategoryStyle("#d1fae5", "#065f46"),
    "injection":    CategoryStyle("#fef3c7", "#92400e"),
    "time":         CategoryStyle("#f9fafb", "#111827"),
    "company":      CategoryStyle("#ede9fe", "#5b21b6"),
    "permissions":  CategoryStyle("#fee2e2", "#991b1b"),
    "state":        CategoryStyle("#fef9c3", "#854d0e"),
    "priming":      CategoryStyle("#cffafe", "#155e75"),
    "tools":        CategoryStyle("#e0e7ff", "#3730a3"),
    "memory":       CategoryStyle("#fce7f3", "#9d174d"),
    "organization": CategoryStyle("#ccfbf1", "#134e4a"),
    "meta":         CategoryStyle("#f3f4f6", "#374151"),
}


@dataclass
class SectionInfo:
    name: str
    category: str
    source: str
    content: str
    chars: int
    tokens_est: int


# ── Section identification heuristics ───────────────────────

# Each rule: (pattern_or_callable, name, category, source)
# pattern is matched against the first 500 chars of the section.
# Rules are tried in order; first match wins.

def _identify_section(text: str, idx: int, anima_name: str) -> SectionInfo:
    """Identify a single section by matching heuristics on its content.

    Rules are ordered so that more specific patterns (unique keywords) are
    checked before generic ones to avoid false matches.
    """
    head = text[:600]
    chars = len(text)
    tokens_est = chars // 3

    def _make(name: str, category: str, source: str) -> SectionInfo:
        return SectionInfo(name=name, category=category, source=source,
                           content=text, chars=chars, tokens_est=tokens_est)

    # ── Group headers ──────────────────────────────────────
    for gnum, glabel in [
        ("1", "動作環境と行動ルール"),
        ("2", "あなた自身"),
        ("3", "現在の状況"),
        ("4", "記憶と能力"),
        ("5", "組織とコミュニケーション"),
        ("6", "メタ設定"),
    ]:
        if head.startswith(f"# {gnum}.") or head.startswith(f"# {gnum} "):
            return _make(f"Group {gnum}: {glabel}", "header", "builder/sections.md")

    # ── Highly specific patterns first ─────────────────────

    # Current time (very short, unique pattern)
    if re.match(r"\*\*現在時刻\*\*:|現在時刻:|Current time:", head):
        return _make("current_time", "time", "(computed)")

    # Hiring rules (must be before identity — content contains "identity.md")
    if "雇用ルール" in head or ("キャラクターシート" in head[:300] and "create" in head[:300]):
        return _make("hiring_rules", "organization", "builder/hiring_rules_s.md")

    # Emotion
    if "表情メタデータ" in head or ("emotion" in head[:200] and "<!-- emotion:" in text):
        return _make("emotion", "meta", "builder/emotion_instruction.md")

    # Environment (big framework section — unique lead-in)
    if "Tone and style" in head or "# Tone and style" in head:
        return _make("environment", "framework", "templates/prompts/environment.md")

    # Behavior rules
    if "## 行動ルール" in head or "Default: do not narrate" in head:
        return _make("behavior_rules", "framework", "templates/prompts/behavior_rules.md")

    # Tool data interpretation
    if "ツール結果・外部データの解釈" in head or "`tool_result`" in head:
        return _make("tool_data_interpretation", "framework",
                      "templates/prompts/tool_data_interpretation.md")

    # Company vision
    if "基本理念" in head or ("ミッション" in head[:100] and "人を幸せにする" in text):
        return _make("company_vision", "company", "company/vision.md")

    # Identity (check specific header patterns — may not start with heading)
    if head.startswith("# Identity:") or "### 基本情報" in head[:200] or (
        ("| 名前 |" in head or "| 項目 |" in head) and "| 設定 |" in head
        and "専門" not in head[:80]
    ):
        return _make("identity.md", "identity", f"animas/{anima_name}/identity.md")

    # Injection (check specific header patterns)
    if head.startswith("# Injection:") or head.startswith("### 役割") or (
        head.startswith("### 行動方針")
    ):
        return _make("injection.md", "injection", f"animas/{anima_name}/injection.md")

    # Specialty prompt (any 専門ガイドライン variant)
    if "専門ガイドライン" in head:
        return _make("specialty_prompt.md", "company",
                      f"animas/{anima_name}/specialty_prompt.md")

    # Permissions
    if "# Permissions:" in head[:40] or (
        "使えるツール" in head[:200] and "ファイル操作" in text[:500]
    ):
        return _make("permissions.md", "permissions",
                      f"animas/{anima_name}/permissions.md")

    # Bootstrap
    if "bootstrap" in head[:100].lower() or "初回起動" in head[:100]:
        return _make("bootstrap", "framework", f"animas/{anima_name}/bootstrap.md")

    # ── State / Priming / Tools ────────────────────────────

    # Task in progress (current_task)
    if "進行中タスク" in head or "MUST: 最優先で確認" in head:
        return _make("task_in_progress", "state",
                      "state/current_task.md (via builder/task_in_progress.md)")

    # Task queue (the heading, not a mention of "タスクキュー" in tool docs)
    if "## タスクキュー" in head[:40] or "## Task Queue" in head[:40] or (
        "## Active Task Queue" in head[:40]
    ):
        return _make("task_queue", "state", "(computed: TaskQueueManager)")

    # Resolution registry
    if "解決済み案件" in head or "Resolution Registry" in head[:60]:
        return _make("resolution_registry", "state", "(computed: ResolutionTracker)")

    # Priming
    if "あなたが思い出していること" in head or "<priming" in head:
        return _make("priming", "priming", "(computed: PrimingEngine)")

    # Recent tool results
    if "Recent Tool Results" in head or "直近のツール結果" in head:
        return _make("recent_tool_results", "tools", "(computed: ConversationMemory)")

    # Pending tasks
    if "## Pending" in head[:30] or "保留中のタスク" in head[:60]:
        return _make("pending_tasks", "state", "state/pending/")

    # ── Memory ─────────────────────────────────────────────

    # Memory guide
    if "あなたの記憶（書庫）" in head or "記憶（書庫）" in head or (
        "エピソード記憶" in head[:300] and "ディレクトリ" in head[:400]
    ):
        return _make("memory_guide", "memory", "templates/prompts/memory_guide.md")

    # Procedures
    if "## Procedures" in head[:30] or "## 手順書" in head[:30]:
        return _make("procedures", "memory", "procedures/ (distilled)")

    # Distilled Knowledge
    if "## Distilled Knowledge" in head[:40] or "## 蒸留された知識" in head[:40]:
        return _make("distilled_knowledge", "memory", "knowledge/ (distilled)")

    # Common knowledge hint
    if "共有リファレンス" in head or ("common_knowledge" in head[:100] and "共有ナレッジ" in text):
        return _make("common_knowledge_hint", "memory", "builder/common_knowledge_hint.md")

    # ── Organization (before Tools — messaging contains mcp__aw__ refs) ──

    # Org context
    if "組織上の位置" in head or "あなたの専門" in head[:100]:
        return _make("org_context", "organization", "(computed: _build_org_context)")

    # Messaging (must be checked before mcp_tools — content includes mcp__aw__*)
    if "メッセージ送信（社員間通信）" in head or "社員間通信" in head[:100] or (
        "送信可能な相手" in text[:600] and "## Board" in text
    ) or "## メッセージ送信" in head[:100]:
        return _make("messaging", "organization", "templates/prompts/messaging_s.md")

    # Human notification
    if "人間への連絡" in head or ("call_human" in head[:300] and "トップレベル" in head[:200]):
        return _make("human_notification", "organization", "builder/human_notification.md")

    # ── Tools ──────────────────────────────────────────────

    # MCP tools (heading specifically, not just any mention of mcp__aw__)
    if "## MCPツール" in head[:30] or ("MCPツール" in head[:100] and "mcp__aw__" in head):
        return _make("mcp_tools", "tools", "(computed: tool guides — s_mcp / prompt_db)")

    # Heartbeat tool instruction
    if "Heartbeatでは" in head or "heartbeat_tool" in head[:100].lower():
        return _make("heartbeat_tool_instruction", "tools",
                      "builder/heartbeat_tool_instruction.md")

    # S builtin tools
    if "ネイティブツール" in head[:200] or "Agent SDK" in head[:200]:
        return _make("s_builtin_tools", "tools", "(computed: tool guides — s_builtin)")

    # External tools
    if "外部ツール" in head[:60] or "External Tools" in head[:60]:
        return _make("external_tools", "tools", "(computed: build_tools_guide)")

    # Hiring context (solo anima)
    if "人材採用" in head[:100]:
        return _make("hiring_context", "organization", "templates/prompts/hiring_context.md")

    # Light tier org
    if "他のアニマとはsend_message" in head:
        return _make("light_tier_org", "organization", "builder/light_tier_org.md")

    # ── Meta ───────────────────────────────────────────────

    # A-mode reflection
    if "振り返り" in head[:100] and "## 振り返り" in head[:30]:
        return _make("a_reflection", "meta", "builder/a_reflection.md")

    # Codex response requirement
    if "応答要件" in head[:30]:
        return _make("response_requirement", "meta", "(computed: codex guidance)")

    # Fallback: unknown
    # Try to extract a heading for the name
    heading_match = re.match(r"^#{1,3}\s+(.+)$", head, re.MULTILINE)
    fallback_name = heading_match.group(1).strip()[:60] if heading_match else f"section_{idx}"
    return SectionInfo(
        name=fallback_name, category="meta",
        source="(unknown)",
        content=text, chars=chars, tokens_est=tokens_est,
    )


# ── HTML Rendering ──────────────────────────────────────────

def _render_badge(cat: str) -> str:
    s = CATEGORIES.get(cat, CategoryStyle("#f3f4f6", "#374151"))
    return (
        f'<span style="background:{s.bg};color:{s.fg};'
        f'padding:1px 6px;border-radius:3px;font-size:12px">{cat}</span>'
    )


def _render_legend_badge(cat: str) -> str:
    s = CATEGORIES.get(cat, CategoryStyle("#f3f4f6", "#374151"))
    return (
        f'<span style="background:{s.bg};color:{s.fg};'
        f'padding:2px 8px;border-radius:4px;margin:2px;'
        f'display:inline-block;font-size:13px">{cat}</span>'
    )


def render_html(
    anima_name: str,
    trigger: str,
    sections: list[SectionInfo],
    total_chars: int,
) -> str:
    total_tokens = total_chars // 3
    n_sections = len(sections)

    # Category legend (unique, ordered by first appearance)
    seen_cats: list[str] = []
    for sec in sections:
        if sec.category not in seen_cats:
            seen_cats.append(sec.category)
    legend_html = "".join(_render_legend_badge(c) for c in seen_cats)

    # Size distribution bar
    bar_parts: list[str] = []
    for sec in sections:
        pct = sec.chars / total_chars * 100 if total_chars else 0
        if pct < 0.1:
            continue
        s = CATEGORIES.get(sec.category, CategoryStyle("#f3f4f6", "#374151"))
        bar_parts.append(
            f'<div title="{html.escape(sec.name)} ({pct:.1f}%)" '
            f'style="width:{pct}%;background:{s.bg};height:100%;display:inline-block"></div>'
        )
    bar_html = "".join(bar_parts)

    # Table rows
    rows: list[str] = []
    for i, sec in enumerate(sections):
        pct = sec.chars / total_chars * 100 if total_chars else 0
        badge = _render_badge(sec.category)
        rows.append(
            f'<tr>'
            f'<td style="text-align:right;padding:2px 8px;color:#6b7280">{i+1}</td>'
            f'<td><a href="#section-{i}" style="text-decoration:none">'
            f'{badge} {html.escape(sec.name)}</a></td>'
            f'<td style="text-align:right;padding:2px 8px;font-family:monospace;font-size:13px">'
            f'{sec.chars:,}</td>'
            f'<td style="text-align:right;padding:2px 8px;font-family:monospace;font-size:13px">'
            f'~{sec.tokens_est:,}</td>'
            f'<td style="text-align:right;padding:2px 8px;font-family:monospace;font-size:13px">'
            f'{pct:.1f}%</td>'
            f'<td style="color:#9ca3af;font-size:12px;padding:2px 8px">'
            f'{html.escape(sec.source)}</td>'
            f'</tr>'
        )
    table_body = "\n".join(rows)

    # Section content blocks
    content_blocks: list[str] = []
    for i, sec in enumerate(sections):
        s = CATEGORIES.get(sec.category, CategoryStyle("#f3f4f6", "#374151"))
        content_blocks.append(f"""\
<div id="section-{i}" style="margin-bottom:16px;">
  <div style="background:{s.bg};color:{s.fg};padding:6px 12px;border-radius:8px 8px 0 0;
              display:flex;justify-content:space-between;align-items:center;
              position:sticky;top:0;z-index:10">
    <div>
      <strong>#{i+1} {html.escape(sec.name)}</strong>
      <span style="opacity:0.7;margin-left:12px;font-size:13px">{sec.category}</span>
    </div>
    <div style="font-size:12px;opacity:0.8">
      {sec.chars:,} chars (~{sec.tokens_est:,} tokens)
      &middot; {html.escape(sec.source)}
    </div>
  </div>
  <pre style="background:#1e1e2e;color:#cdd6f4;padding:12px;margin:0;
              border-radius:0 0 8px 8px;overflow-x:auto;font-size:13px;
              line-height:1.5;white-space:pre-wrap;word-wrap:break-word;
              border:1px solid {s.bg};border-top:none">{html.escape(sec.content)}</pre>
</div>""")
    content_html = "\n".join(content_blocks)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>System Prompt Debug: {html.escape(anima_name)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f0f23;
    color: #e0e0e0;
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
  }}
  h1 {{ color: #f9fafb; margin-bottom: 8px; }}
  a {{ color: #93c5fd; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #374151;
       color: #9ca3af; font-size: 13px; }}
  tr:hover {{ background: #1f2937; }}
</style>
</head>
<body>

<h1>System Prompt Debug: {html.escape(anima_name)}</h1>
<p style="color:#9ca3af;margin-bottom:4px">
  Generated: {timestamp} &middot;
  Trigger: <code>{html.escape(trigger or "chat")}</code> &middot;
  Sections: {n_sections} &middot;
  Total: {total_chars:,} chars (~{total_tokens:,} tokens)
</p>

<div style="margin:16px 0">
  <p style="color:#9ca3af;font-size:13px;margin-bottom:4px">Categories:</p>
  <div>{legend_html}</div>
</div>

<div style="margin:16px 0">
  <p style="color:#9ca3af;font-size:13px;margin-bottom:4px">Size distribution:</p>
  <div style="height:24px;border-radius:6px;overflow:hidden;background:#1f2937;
              display:flex">{bar_html}</div>
</div>

<details style="margin:16px 0" open>
  <summary style="cursor:pointer;color:#93c5fd;font-size:15px;margin-bottom:8px">
    Section Table ({n_sections} sections)
  </summary>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Section</th><th>Chars</th><th>Tokens</th><th>%</th><th>Source</th>
      </tr>
    </thead>
    <tbody>
{table_body}
    </tbody>
  </table>
</details>

<hr style="border:1px solid #374151;margin:24px 0">

{content_html}

</body>
</html>"""


# ── Main ────────────────────────────────────────────────────

def main() -> None:
    anima_name = sys.argv[1] if len(sys.argv) > 1 else "sakura"
    test_message = sys.argv[2] if len(sys.argv) > 2 else "mcpサーバーツールはあなたは知っていますか"
    trigger = sys.argv[3] if len(sys.argv) > 3 else ""

    # ── Resolve paths ──────────────────────────────────────
    from core.paths import get_data_dir, get_shared_dir

    data_dir = get_data_dir()
    anima_dir = data_dir / "animas" / anima_name

    if not anima_dir.exists():
        print(f"ERROR: Anima directory not found: {anima_dir}")
        sys.exit(1)

    print(f"Anima directory : {anima_dir}")
    print(f"Test message    : {test_message}")
    print(f"Trigger         : {trigger or '(chat)'}")
    print()

    # ── Initialize MemoryManager ───────────────────────────
    from core.memory.manager import MemoryManager

    memory = MemoryManager(anima_dir)

    # ── Build tool_registry ────────────────────────────────
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

    # ── Run PrimingEngine ──────────────────────────────────
    priming_section = ""
    try:
        from core.memory.priming import PrimingEngine, format_priming_section

        shared_dir = get_shared_dir()
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

        _channel = (
            "heartbeat" if trigger == "heartbeat"
            else "cron" if trigger.startswith("cron")
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

    result = build_system_prompt(
        memory=memory,
        tool_registry=tool_registry,
        execution_mode="s",
        message=test_message,
        priming_section=priming_section,
        trigger=trigger,
    )

    prompt_text = result.system_prompt

    # ── Split into sections ────────────────────────────────
    # Split by --- separators, then merge sub-sections (### headings)
    # back into their parent section.  Individual procedure/knowledge
    # entries use --- internally and start with ### , so they must be
    # re-attached to the preceding ## Procedures / ## Distilled Knowledge.
    raw_sections = prompt_text.split("\n\n---\n\n")

    # Pass 1: identify each raw piece
    raw_identified = [_identify_section(r, i, anima_name) for i, r in enumerate(raw_sections)]

    # Pass 2: merge sub-sections into their parent.
    # Merge conditions:
    #   a) Starts with ### and is not a known anima-specific section
    #   b) Is an (unknown) section shorter than 120 chars (likely an internal
    #      fragment, e.g. company_vision split by its internal ---)
    _ANIMA_SPECIFIC_CATEGORIES = {"identity", "injection", "permissions", "company", "time"}
    sections: list[SectionInfo] = []
    for sec in raw_identified:
        is_subsection = sec.content.lstrip().startswith("### ")
        is_anima_specific = sec.category in _ANIMA_SPECIFIC_CATEGORIES
        is_group_header = sec.category == "header"
        is_short_unknown = sec.source == "(unknown)" and sec.chars < 120

        should_merge = (
            sections
            and not is_group_header
            and (
                (is_subsection and not is_anima_specific)
                or is_short_unknown
            )
        )

        if should_merge:
            prev = sections[-1]
            merged_content = prev.content + "\n\n---\n\n" + sec.content
            sections[-1] = SectionInfo(
                name=prev.name,
                category=prev.category,
                source=prev.source,
                content=merged_content,
                chars=len(merged_content),
                tokens_est=len(merged_content) // 3,
            )
        else:
            sections.append(sec)

    total_chars = len(prompt_text)
    unknown_count = sum(1 for s in sections if s.source == "(unknown)")

    # ── Render HTML ────────────────────────────────────────
    html_content = render_html(anima_name, trigger, sections, total_chars)

    # Determine suffix for trigger
    suffix = f"_{trigger}" if trigger and trigger not in ("", "chat") else ""
    out_name = f"prompt_debug_{anima_name}{suffix}.html"
    out_path = Path(f"/tmp/{out_name}")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"HTML written to  : {out_path}")

    # Copy to server static files if directory exists
    static_dir = PROJECT_ROOT / "server" / "static" / "files"
    if static_dir.is_dir():
        static_path = static_dir / out_name
        shutil.copy2(out_path, static_path)
        print(f"Copied to        : {static_path}")

    print()

    # ── Print summary ──────────────────────────────────────
    print("=" * 60)
    print("SYSTEM PROMPT SUMMARY")
    print("=" * 60)
    print(f"Total length    : {total_chars:,} chars (~{total_chars // 3:,} tokens)")
    print(f"Sections        : {len(sections)}")
    print(f"Unknown sections: {unknown_count}")
    print()

    print("Sections:")
    for i, sec in enumerate(sections):
        pct = sec.chars / total_chars * 100 if total_chars else 0
        print(f"  {i+1:2d}. [{sec.category:13s}] {sec.name:30s} {sec.chars:>6,} chars ({pct:4.1f}%) <- {sec.source}")
    print()

    if result.injected_procedures:
        print("Injected procedures:")
        for p in result.injected_procedures:
            print(f"  - {p}")
        print()

    if result.injected_knowledge_files:
        print("Injected knowledge:")
        for k in result.injected_knowledge_files:
            print(f"  - {k}")
        print()

    if result.overflow_files:
        print("Overflow (budget exceeded):")
        for o in result.overflow_files:
            print(f"  - {o}")
        print()


if __name__ == "__main__":
    main()
