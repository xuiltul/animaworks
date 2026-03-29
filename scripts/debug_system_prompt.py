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


# ── Section identification ──────────────────────────────────

# Mapping from builder section id → (category, source hint)
_SECTION_META: dict[str, tuple[str, str]] = {
    "environment":              ("framework",    "templates/prompts/environment.md"),
    "identity":                 ("identity",     "animas/{name}/identity.md"),
    "injection":                ("injection",    "animas/{name}/injection.md"),
    "current_time":             ("time",         "(computed)"),
    "behavior_rules":           ("framework",    "templates/prompts/behavior_rules.md"),
    "tool_data_interpretation":  ("framework",    "templates/prompts/tool_data_interpretation.md"),
    "company_vision":           ("company",      "company/vision.md"),
    "specialty_prompt":         ("company",      "animas/{name}/specialty_prompt.md"),
    "permissions":              ("permissions",  "animas/{name}/permissions.json"),
    "bootstrap":                ("framework",    "animas/{name}/bootstrap.md"),
    "task_in_progress":         ("state",        "state/current_state.md"),
    "task_queue":               ("state",        "(computed: TaskQueueManager)"),
    "resolution_registry":      ("state",        "(computed: ResolutionTracker)"),
    "priming":                  ("priming",      "(computed: PrimingEngine)"),
    "recent_tool_results":      ("tools",        "(computed: ConversationMemory)"),
    "pending_tasks":            ("state",        "state/pending/"),
    "memory_guide":             ("memory",       "templates/prompts/memory_guide.md"),
    "dk_procedures":            ("memory",       "procedures/ (distilled)"),
    "dk_knowledge":             ("memory",       "knowledge/ (distilled)"),
    "common_knowledge_hint":    ("memory",       "builder/common_knowledge_hint.md"),
    "org_context":              ("organization", "(computed: _build_org_context)"),
    "messaging":                ("organization", "templates/prompts/messaging_s.md"),
    "human_notification":       ("organization", "builder/human_notification.md"),
    "tool_guides":              ("tools",        "(computed: prompt_db)"),
    "external_tools":           ("tools",        "(computed: build_tools_guide)"),
    "emotion_instruction":      ("meta",         "builder/emotion_instruction.md"),
    "a_reflection":             ("meta",         "builder/a_reflection.md"),
    "c_response_requirement":   ("meta",         "(computed: codex guidance)"),
    "light_tier_org":           ("organization", "builder/light_tier_org.md"),
}

_RE_GROUP_OPEN = re.compile(r'<group_(\d+)\s+title="([^"]*)">')
_RE_GROUP_CLOSE = re.compile(r"</group_(\d+)>")
_RE_SECTION = re.compile(
    r'<section\s+name="([^"]*)">\n(.*?)\n</section>', re.DOTALL
)


def _parse_xml_sections(prompt_text: str, anima_name: str) -> list[SectionInfo]:
    """Parse XML-tagged prompt into SectionInfo list (groups + sections)."""
    items: list[tuple[int, SectionInfo]] = []

    for m in _RE_GROUP_OPEN.finditer(prompt_text):
        gnum, title = m.group(1), m.group(2)
        items.append((m.start(), SectionInfo(
            name=f"Group {gnum}: {title}",
            category="header",
            source="builder/sections.md",
            content=title,
            chars=len(title),
            tokens_est=len(title) // 3,
        )))

    for m in _RE_SECTION.finditer(prompt_text):
        sid, body = m.group(1), m.group(2)
        cat, src = _SECTION_META.get(sid, ("meta", "(unknown)"))
        src = src.replace("{name}", anima_name)
        items.append((m.start(), SectionInfo(
            name=sid,
            category=cat,
            source=src,
            content=body,
            chars=len(body),
            tokens_est=len(body) // 3,
        )))

    items.sort(key=lambda t: t[0])
    return [info for _, info in items]


# ── Legacy heuristic fallback (pre-XML prompts) ─────────────

def _identify_section_legacy(text: str, idx: int, anima_name: str) -> SectionInfo:
    """Identify a section by content heuristics (legacy --- split)."""
    head = text[:600]
    chars = len(text)
    tokens_est = chars // 3

    def _make(name: str, category: str, source: str) -> SectionInfo:
        return SectionInfo(name=name, category=category, source=source,
                           content=text, chars=chars, tokens_est=tokens_est)

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

    if re.match(r"\*\*現在時刻\*\*:|現在時刻:|Current time:", head):
        return _make("current_time", "time", "(computed)")
    if "Tone and style" in head:
        return _make("environment", "framework", "templates/prompts/environment.md")
    if "## 行動ルール" in head:
        return _make("behavior_rules", "framework", "templates/prompts/behavior_rules.md")
    if "表情メタデータ" in head:
        return _make("emotion", "meta", "builder/emotion_instruction.md")
    if head.startswith("# Identity:") or "### 基本情報" in head[:200]:
        return _make("identity.md", "identity", f"animas/{anima_name}/identity.md")
    if head.startswith("# Injection:") or head.startswith("### 役割"):
        return _make("injection.md", "injection", f"animas/{anima_name}/injection.md")
    if "## MCPツール" in head[:30]:
        return _make("mcp_tools", "tools", "(computed: prompt_db)")
    if "メッセージ送信" in head[:100]:
        return _make("messaging", "organization", "templates/prompts/messaging_s.md")
    if "解決済み案件" in head:
        return _make("resolution_registry", "state", "(computed: ResolutionTracker)")
    if "<priming" in head:
        return _make("priming", "priming", "(computed: PrimingEngine)")

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

    # Section content blocks — group headers render as banner, sections as collapsible
    content_blocks: list[str] = []
    for i, sec in enumerate(sections):
        s = CATEGORIES.get(sec.category, CategoryStyle("#f3f4f6", "#374151"))
        if sec.category == "header":
            content_blocks.append(f"""\
<div id="section-{i}" style="margin:24px 0 8px 0;">
  <div style="background:{s.bg};color:{s.fg};padding:10px 16px;border-radius:8px;
              font-size:16px;font-weight:bold;
              border-left:4px solid #60a5fa">
    {html.escape(sec.name)}
  </div>
</div>""")
        else:
            content_blocks.append(f"""\
<div id="section-{i}" style="margin-bottom:12px;margin-left:12px;">
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

    # Normalize shorthand triggers to match builder.py's startswith() checks
    _TRIGGER_NORMALIZE: dict[str, str] = {
        "inbox": "inbox:debug",
        "task": "task:debug",
        "cron": "cron:debug",
    }
    trigger = _TRIGGER_NORMALIZE.get(trigger, trigger)

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
        tool_registry = all_tools
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
    if "<group_" in prompt_text and "<section " in prompt_text:
        sections = _parse_xml_sections(prompt_text, anima_name)
        print(f"Parsing mode    : XML tags ({len(sections)} entries)")
    else:
        print("Parsing mode    : Legacy (--- separators)")
        raw_sections = prompt_text.split("\n\n---\n\n")
        raw_identified = [
            _identify_section_legacy(r, i, anima_name)
            for i, r in enumerate(raw_sections)
        ]
        _ANIMA_SPECIFIC = {"identity", "injection", "permissions", "company", "time"}
        sections = []
        for sec in raw_identified:
            is_sub = sec.content.lstrip().startswith("### ")
            is_specific = sec.category in _ANIMA_SPECIFIC
            is_header = sec.category == "header"
            is_short_unk = sec.source == "(unknown)" and sec.chars < 120
            should_merge = (
                sections and not is_header
                and ((is_sub and not is_specific) or is_short_unk)
            )
            if should_merge:
                prev = sections[-1]
                merged = prev.content + "\n\n---\n\n" + sec.content
                sections[-1] = SectionInfo(
                    name=prev.name, category=prev.category,
                    source=prev.source, content=merged,
                    chars=len(merged), tokens_est=len(merged) // 3,
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
