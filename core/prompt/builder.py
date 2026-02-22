from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.memory import MemoryManager
from core.paths import PROJECT_DIR, get_data_dir, load_prompt
from core.memory.shortterm import ShortTermMemory
from core.time_utils import now_jst

logger = logging.getLogger("animaworks.prompt_builder")


@dataclass
class BuildResult:
    """Result of system prompt building."""

    system_prompt: str
    injected_procedures: list[Path] = field(default_factory=list)
    injected_knowledge_files: list[str] = field(default_factory=list)
    overflow_files: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Backward compatibility: str() returns prompt."""
        return self.system_prompt

    def __len__(self) -> int:
        """Backward compatibility: len() returns prompt length."""
        return len(self.system_prompt)

    def encode(self, encoding: str = "utf-8") -> bytes:
        """Backward compatibility: encode() encodes prompt."""
        return self.system_prompt.encode(encoding)

    def __contains__(self, item: str) -> bool:
        """Backward compatibility: 'x in result' checks prompt."""
        return item in self.system_prompt

    def __add__(self, other: str) -> str:
        """Backward compatibility: result + str concatenates prompt."""
        return self.system_prompt + other

    def __radd__(self, other: str) -> str:
        """Backward compatibility: str + result concatenates prompt."""
        return other + self.system_prompt

    def index(self, sub: str, *args: int) -> int:
        """Backward compatibility: result.index(x) searches prompt."""
        return self.system_prompt.index(sub, *args)

    def count(self, sub: str, *args: int) -> int:
        """Backward compatibility: result.count(x) counts in prompt."""
        return self.system_prompt.count(sub, *args)


_CURRENT_TASK_MAX_CHARS = 3000

# ── Emotion Instruction ───────────────────────────────────

def _build_emotion_instruction() -> str:
    """Build EMOTION_INSTRUCTION with the canonical emotion list."""
    from core.schemas import VALID_EMOTIONS
    emotion_list = ", ".join(sorted(VALID_EMOTIONS))
    return load_prompt("builder/emotion_instruction", emotion_list=emotion_list)


EMOTION_INSTRUCTION = _build_emotion_instruction()


def _discover_other_animas(anima_dir: Path) -> list[str]:
    """List sibling anima directories."""
    animas_root = anima_dir.parent
    self_name = anima_dir.name
    others = []
    for d in sorted(animas_root.iterdir()):
        if d.is_dir() and d.name != self_name and (d / "identity.md").exists():
            others.append(d.name)
    return others


# ── Organisation Context ─────────────────────────────────


def _format_anima_entry(name: str, speciality: str | None) -> str:
    """Format an anima name with optional speciality annotation."""
    if speciality:
        return f"{name} ({speciality})"
    return name


def _build_full_org_tree(
    anima_name: str,
    all_animas: dict[str, Any],
) -> str:
    """Build an indented full organization tree for top-level animas."""
    # Build children map: parent -> list of children
    children: dict[str | None, list[str]] = {}
    for name, pcfg in all_animas.items():
        parent = pcfg.supervisor
        children.setdefault(parent, []).append(name)
    for k in children:
        children[k].sort()

    lines: list[str] = []

    def _render(name: str, prefix: str, is_last: bool, is_root: bool) -> None:
        spec = all_animas[name].speciality if name in all_animas else None
        label = _format_anima_entry(name, spec)
        if is_root:
            marker = ""
            suffix = "  ← あなた" if name == anima_name else ""
        else:
            marker = "└── " if is_last else "├── "
            suffix = "  ← あなた" if name == anima_name else ""
        lines.append(f"{prefix}{marker}{label}{suffix}")
        kids = children.get(name, [])
        for i, child in enumerate(kids):
            child_is_last = (i == len(kids) - 1)
            if is_root:
                child_prefix = prefix
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            _render(child, child_prefix, child_is_last, False)

    roots = children.get(None, [])
    for i, root in enumerate(roots):
        _render(root, "", i == len(roots) - 1, True)

    return "\n".join(lines)


def _build_org_context(anima_name: str, other_animas: list[str], execution_mode: str = "a1") -> str:
    """Build organisation context section from supervisor chain.

    Reads config.json to derive each Anima's relationship
    (supervisor / subordinate / peer) relative to *anima_name*
    and returns a formatted prompt section.
    """
    from core.config import load_config
    from core.tooling.prompt_db import get_prompt_store

    try:
        config = load_config()
    except Exception:
        logger.debug("Could not load config for org context", exc_info=True)
        return ""

    all_animas = config.animas
    my_config = all_animas.get(anima_name)
    my_supervisor = my_config.supervisor if my_config else None
    my_speciality = my_config.speciality if my_config else None
    is_top_level = my_supervisor is None

    # Top-level anima with subordinates: show full org tree
    if is_top_level and len(all_animas) > 1:
        anima_speciality = my_speciality or "(未設定)"
        tree_text = _build_full_org_tree(anima_name, all_animas)
        parts = [
            load_prompt(
                "builder/org_context_toplevel",
                anima_speciality=anima_speciality,
                tree_text=tree_text,
            ),
        ]
        if other_animas:
            cr_key = "communication_rules_a1" if execution_mode == "a1" else "communication_rules"
            _cr_store = get_prompt_store()
            _cr = (
                _cr_store.get_section(cr_key) if _cr_store else None
            ) or load_prompt(cr_key)
            if _cr:
                parts.append(_cr)
        return "\n\n".join(parts)

    # Non-top-level: existing logic
    # Supervisor
    if my_supervisor:
        sup_spec = None
        if my_supervisor in all_animas:
            sup_spec = all_animas[my_supervisor].speciality
        supervisor_line = _format_anima_entry(my_supervisor, sup_spec)
    else:
        supervisor_line = "(なし — あなたがトップです)"

    # Subordinates: animas whose supervisor is me
    subordinates: list[str] = []
    for name in sorted(all_animas):
        if name == anima_name:
            continue
        pcfg = all_animas[name]
        if pcfg.supervisor == anima_name:
            subordinates.append(_format_anima_entry(name, pcfg.speciality))

    # Peers: animas with the same supervisor (excluding self)
    peers: list[str] = []
    if my_supervisor is not None:
        for name in sorted(all_animas):
            if name == anima_name:
                continue
            pcfg = all_animas[name]
            if pcfg.supervisor == my_supervisor:
                peers.append(_format_anima_entry(name, pcfg.speciality))

    subordinates_line = ", ".join(subordinates) if subordinates else "(なし)"
    peers_line = ", ".join(peers) if peers else "(なし)"
    anima_speciality = my_speciality or "(未設定)"

    parts = [
        load_prompt(
            "org_context",
            supervisor_line=supervisor_line,
            subordinates_line=subordinates_line,
            peers_line=peers_line,
            anima_speciality=anima_speciality,
        ),
    ]

    # Communication rules: only when there are other animas
    if other_animas:
        cr_key = "communication_rules_a1" if execution_mode == "a1" else "communication_rules"
        _cr_store = get_prompt_store()
        _cr = (
            _cr_store.get_section(cr_key) if _cr_store else None
        ) or load_prompt(cr_key)
        if _cr:
            parts.append(_cr)

    return "\n\n".join(parts)


def _build_messaging_section(
    anima_dir: Path,
    other_animas: list[str],
    execution_mode: str = "a1",
) -> str:
    """Build the messaging instructions with resolved paths."""
    from core.tooling.prompt_db import get_prompt_store

    self_name = anima_dir.name
    main_py = PROJECT_DIR / "main.py"
    animas_line = (
        ", ".join(other_animas) if other_animas else "(まだ他の社員はいません)"
    )

    db_key = "messaging_a1" if execution_mode == "a1" else "messaging"
    _msg_store = get_prompt_store()
    raw = (_msg_store.get_section(db_key) if _msg_store else None)
    if raw:
        try:
            return raw.format(
                animas_line=animas_line, main_py=main_py, self_name=self_name,
            )
        except (KeyError, IndexError):
            return raw
    return load_prompt(
        db_key,
        animas_line=animas_line,
        main_py=main_py,
        self_name=self_name,
    )


def _load_a2_reflection() -> str:
    """Load the A2 reflection/retry prompt template."""
    try:
        return load_prompt("a2_reflection")
    except Exception:
        logger.debug("a2_reflection template not found, skipping")
        return ""


def _build_recent_tool_section(anima_dir: Path, model_config: Any) -> str:
    """Build a summary of recent tool results for system prompt injection.

    Reads the last few turns from ConversationMemory and extracts tool
    records with result summaries, constrained to a ~2000 token budget.
    """
    try:
        from core.memory.conversation import ConversationMemory
        conv_memory = ConversationMemory(anima_dir, model_config)
        state = conv_memory.load()
    except Exception:
        return ""
    if not state.turns:
        return ""

    tool_lines: list[str] = []
    budget_remaining = 2000  # approximate token budget (~8000 chars)
    for turn in reversed(state.turns[-3:]):
        for tr in turn.tool_records[:5]:
            if not tr.result_summary:
                continue
            line = f"- {tr.tool_name}: {tr.result_summary[:500]}"
            budget_remaining -= len(line) // 4
            if budget_remaining <= 0:
                break
            tool_lines.append(line)
        if budget_remaining <= 0:
            break

    if not tool_lines:
        return ""
    return "## Recent Tool Results\n\n" + "\n".join(tool_lines)


def _build_human_notification_guidance(execution_mode: str = "") -> str:
    """Build the human notification instruction for top-level Animas."""
    if execution_mode == "a1":
        how_to = load_prompt("builder/human_notification_howto_a1")
    else:
        how_to = load_prompt("builder/human_notification_howto_other")

    return load_prompt("builder/human_notification", how_to=how_to)


def build_system_prompt(
    memory: MemoryManager,
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
    priming_section: str = "",
    execution_mode: str = "a1",
    message: str = "",
    retriever: object | None = None,
    *,
    trigger: str = "",
) -> BuildResult:
    """Construct the full system prompt from Markdown files.

    Sections are organised into 6 hierarchical groups:
      1. 動作環境と行動ルール
      2. あなた自身
      3. 現在の状況
      4. 記憶と能力
      5. 組織とコミュニケーション
      6. メタ設定
    """
    parts: list[str] = []
    pd = memory.anima_dir
    data_dir = get_data_dir()

    # ── Pre-compute values needed across multiple groups ──────────

    # DB-first prompt store (singleton); used for system sections & tool guides
    from core.tooling.prompt_db import DEFAULT_GUIDES, get_prompt_store
    _prompt_store = get_prompt_store()

    # other_animas is needed by Group 5 (org_context, messaging)
    other_animas = _discover_other_animas(pd)

    # Skill/procedure metadata needed by Group 4
    skill_metas = memory.list_skill_metas()
    common_skill_metas = memory.list_common_skill_metas()
    procedure_metas = memory.list_procedure_metas()

    # Read permissions early (needed by Group 2 and Group 4 external tools check)
    permissions = memory.read_permissions()

    # ── Group 1: 動作環境と行動ルール ─────────────────────────
    parts.append("# 1. 動作環境と行動ルール")

    _env = (_prompt_store.get_section("environment") if _prompt_store else None)
    if _env:
        try:
            _env = _env.format(data_dir=data_dir, anima_name=pd.name)
        except (KeyError, IndexError):
            pass
    else:
        _env = load_prompt("environment", data_dir=data_dir, anima_name=pd.name)
    if _env:
        parts.append(_env)

    current_time = now_jst().strftime("%Y-%m-%d %H:%M (%Z)")
    parts.append(f"**現在時刻**: {current_time}")

    _br = (
        _prompt_store.get_section("behavior_rules") if _prompt_store else None
    ) or load_prompt("behavior_rules")
    if _br:
        parts.append(_br)

    # ── Group 2: あなた自身 ───────────────────────────────────
    parts.append("# 2. あなた自身")

    bootstrap = memory.read_bootstrap()
    if bootstrap:
        parts.append(bootstrap)

    company_vision = memory.read_company_vision()
    if company_vision:
        parts.append(company_vision)

    identity = memory.read_identity()
    if identity:
        parts.append(identity)

    injection = memory.read_injection()
    if injection:
        parts.append(injection)

    specialty = memory.read_specialty_prompt()
    if specialty:
        parts.append(specialty)

    if permissions:
        parts.append(permissions)

    # ── Group 3: 現在の状況 ───────────────────────────────────
    parts.append("# 3. 現在の状況")

    # current_state with size limit
    state = memory.read_current_state()
    if state and state.strip() != "status: idle":
        if len(state) > _CURRENT_TASK_MAX_CHARS:
            truncated = state[-_CURRENT_TASK_MAX_CHARS:]
            first_nl = truncated.find("\n")
            if first_nl != -1 and first_nl < _CURRENT_TASK_MAX_CHARS * 0.2:
                truncated = truncated[first_nl + 1:]
            state = "（前半省略）\n\n" + truncated
        parts.append(load_prompt("builder/task_in_progress", state=state))
    elif state:
        parts.append(f"## 現在の状態\n\n{state}")

    pending = memory.read_pending()
    if pending:
        parts.append(f"## 未完了タスク\n\n{pending}")

    # Task Queue (structured persistent queue)
    try:
        from core.memory.task_queue import TaskQueueManager
        task_queue = TaskQueueManager(memory.anima_dir)
        task_summary = task_queue.format_for_priming()
        if task_summary:
            parts.append(load_prompt("builder/task_queue", task_summary=task_summary))
    except Exception:
        logger.debug("Failed to inject task queue", exc_info=True)

    # Resolution registry with dedup (cross-org resolved issues)
    try:
        resolutions = memory.read_resolutions(days=7)
        if resolutions:
            seen_issues: dict[str, dict] = {}
            for r in resolutions:
                key = r.get("issue", "")
                seen_issues[key] = r  # 後勝ち = 最新
            deduped = sorted(seen_issues.values(), key=lambda x: x.get("ts", ""))
            res_lines = []
            for r in deduped[-10:]:
                ts_short = r.get("ts", "")[:16]
                resolver = r.get("resolver", "unknown")
                issue = r.get("issue", "")
                res_lines.append(f"- [{ts_short}] {resolver}: {issue}")
            parts.append(load_prompt(
                "builder/resolution_registry",
                res_lines="\n".join(res_lines),
            ))
    except Exception:
        logger.debug("Failed to inject resolution registry", exc_info=True)

    # Priming section (automatic memory recall)
    if priming_section:
        parts.append(priming_section)

    # Recent tool results (last few turns)
    try:
        _model_cfg = memory.read_model_config()
        recent_tools = _build_recent_tool_section(pd, _model_cfg)
        if recent_tools:
            parts.append(recent_tools)
    except Exception:
        logger.debug("Failed to inject recent tool results", exc_info=True)

    # ── Group 4: 記憶と能力 ───────────────────────────────────
    parts.append("# 4. 記憶と能力")

    # Memory directory guide
    knowledge_list = ", ".join(memory.list_knowledge_files()) or "(なし)"
    episode_list = ", ".join(memory.list_episode_files()[:7]) or "(なし)"
    procedure_list = ", ".join(memory.list_procedure_files()) or "(なし)"
    all_skill_names = [m.name for m in skill_metas] + [
        f"{m.name}(共通)" for m in common_skill_metas
    ]
    skill_names = ", ".join(all_skill_names) or "(なし)"
    shared_users_list = ", ".join(memory.list_shared_users()) or "(なし)"

    parts.append(load_prompt(
        "memory_guide",
        anima_dir=pd,
        knowledge_list=knowledge_list,
        episode_list=episode_list,
        procedure_list=procedure_list,
        skill_names=skill_names,
        shared_users_list=shared_users_list,
    ))

    # ── Distilled Knowledge Injection ─────────────────────
    # CLS theory: knowledge/ + procedures/ = neocortex (always-active)
    from core.prompt.context import resolve_context_window

    injected_knowledge_files: list[str] = []
    overflow_files: list[str] = []

    try:
        _model_config = memory.read_model_config()
        ctx_window = resolve_context_window(_model_config.model)
    except Exception:
        ctx_window = 128_000

    knowledge_budget = int(ctx_window * 0.10)
    distilled = memory.collect_distilled_knowledge()

    used_tokens = 0
    injection_parts: list[str] = []
    for entry in distilled:
        est_tokens = len(entry["content"]) // 3
        if used_tokens + est_tokens <= knowledge_budget:
            injection_parts.append(
                f"### {entry['name']}\n\n{entry['content']}"
            )
            used_tokens += est_tokens
            injected_knowledge_files.append(entry["name"])
        else:
            overflow_files.append(entry["name"])

    if injection_parts:
        parts.append(
            "## Distilled Knowledge\n\n"
            + "\n\n---\n\n".join(injection_parts)
        )

    # Common knowledge reference hint
    common_knowledge_dir = data_dir / "common_knowledge"
    if common_knowledge_dir.exists() and any(common_knowledge_dir.rglob("*.md")):
        parts.append(load_prompt("builder/common_knowledge_hint"))

    # Commander hiring guardrail: force create_anima tool/CLI usage
    has_newstaff = any(m.name == "newstaff" for m in skill_metas)
    if has_newstaff:
        if execution_mode == "a1":
            parts.append(load_prompt("builder/hiring_rules_a1"))
        else:
            parts.append(load_prompt("builder/hiring_rules_other"))

    # ── Tool usage guides from DB (with hardcoded fallback) ──
    if not _prompt_store:
        logger.warning("Tool prompt DB unavailable; using hardcoded fallback guides")

    if execution_mode == "a1":
        _a1_builtin = (
            _prompt_store.get_guide("a1_builtin") if _prompt_store else None
        ) or DEFAULT_GUIDES.get("a1_builtin", "")
        if _a1_builtin:
            parts.append(_a1_builtin)
        _a1_mcp = (
            _prompt_store.get_guide("a1_mcp") if _prompt_store else None
        ) or DEFAULT_GUIDES.get("a1_mcp", "")
        if _a1_mcp:
            parts.append(_a1_mcp)
    else:
        _non_a1 = (
            _prompt_store.get_guide("non_a1") if _prompt_store else None
        ) or DEFAULT_GUIDES.get("non_a1", "")
        if _non_a1:
            parts.append(_non_a1)

    # External tools guide (filtered by registry)
    if permissions and "外部ツール" in permissions and (tool_registry or personal_tools):
        if execution_mode == "a2":
            categories = ", ".join(sorted(tool_registry or []))
            if personal_tools:
                personal_cats = ", ".join(sorted(personal_tools.keys()))
                categories = f"{categories}, {personal_cats}" if categories else personal_cats
            parts.append(load_prompt(
                "builder/external_tools_guide",
                categories=categories,
            ))
        else:
            from core.tooling.guide import build_tools_guide
            tools_guide = build_tools_guide(tool_registry or [], personal_tools)
            if tools_guide:
                parts.append(tools_guide)

    # ── Group 5: 組織とコミュニケーション ─────────────────────
    parts.append("# 5. 組織とコミュニケーション")

    # Hiring context: suggest team building when top-level anima has no peers
    if not other_animas:
        try:
            model_config = memory.read_model_config()
            if model_config.supervisor is None:
                _hc = (
                    _prompt_store.get_section("hiring_context")
                    if _prompt_store else None
                ) or load_prompt("hiring_context")
                if _hc:
                    parts.append(_hc)
        except Exception:
            logger.debug("Skipped hiring context injection", exc_info=True)

    org_context = _build_org_context(pd.name, other_animas, execution_mode)
    if org_context:
        parts.append(org_context)

    parts.append(_build_messaging_section(pd, other_animas, execution_mode))

    # Human notification guidance for top-level Animas
    try:
        from core.config import load_config as _load_cfg
        _cfg = _load_cfg()
        _my_pcfg = _cfg.animas.get(pd.name)
        _is_top_level = _my_pcfg is None or _my_pcfg.supervisor is None
        if _is_top_level and _cfg.human_notification.enabled:
            parts.append(_build_human_notification_guidance(execution_mode))
    except Exception:
        logger.debug("Skipped human notification guidance injection", exc_info=True)

    # ── Group 6: メタ設定 ─────────────────────────────────────
    parts.append("# 6. メタ設定")

    _ei = (
        _prompt_store.get_section("emotion_instruction")
        if _prompt_store else None
    ) or EMOTION_INSTRUCTION
    if _ei:
        parts.append(_ei)

    if execution_mode == "a2":
        _ar = (
            _prompt_store.get_section("a2_reflection")
            if _prompt_store else None
        ) or _load_a2_reflection()
        if _ar:
            parts.append(_ar)

    # ── Final assembly ────────────────────────────────────────
    prompt = "\n\n---\n\n".join(parts)
    logger.debug(
        "System prompt built: %d sections, total_len=%d",
        len(parts), len(prompt),
    )
    return BuildResult(
        system_prompt=prompt,
        injected_knowledge_files=injected_knowledge_files,
        overflow_files=overflow_files,
    )


def inject_shortterm(
    base_prompt: str,
    shortterm: ShortTermMemory,
) -> str:
    """Append short-term memory content to the system prompt.

    If the shortterm folder has a ``session_state.md``, its content is
    appended after a separator so the agent can pick up where it left off.
    """
    md_content = shortterm.load_markdown()
    if not md_content:
        return base_prompt
    return base_prompt + "\n\n---\n\n" + md_content