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

# Modes that use MCP-style tool access (built-in + mcp__aw__*).
_MCP_MODES = frozenset({"s", "c"})


def _is_mcp_mode(execution_mode: str) -> bool:
    """Return True for modes using built-in tools + MCP (S and C)."""
    return execution_mode in _MCP_MODES


def _parse_kv_template(raw: str) -> dict[str, str]:
    """Parse ``[key]: value`` lines from a template string.

    Only the first ``]: `` occurrence per line is used as delimiter,
    so values may safely contain ``]: ``.
    """
    result: dict[str, str] = {}
    for line in raw.strip().splitlines():
        if not line.startswith("["):
            continue
        bracket_end = line.find("]")
        if bracket_end < 0:
            continue
        sep = line.find("]: ", bracket_end)
        if sep < 0:
            continue
        key = line[1:bracket_end]
        value = line[sep + 3:]
        result[key] = value
    return result


def _load_section_strings(locale: str | None = None) -> dict[str, str]:
    """Load section headers and labels from template."""
    try:
        raw = load_prompt("builder/sections", locale=locale)
    except FileNotFoundError:
        return {}
    return _parse_kv_template(raw)


def _load_fallback_strings(locale: str | None = None) -> dict[str, str]:
    """Load fallback/placeholder texts from template."""
    try:
        raw = load_prompt("builder/fallbacks", locale=locale)
    except FileNotFoundError:
        return {}
    return _parse_kv_template(raw)


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

# ── Prompt tier constants ─────────────────────────────────────
TIER_FULL = "full"          # 128k+
TIER_STANDARD = "standard"  # 32k–128k
TIER_LIGHT = "light"        # 16k–32k
TIER_MINIMAL = "minimal"    # <16k


def resolve_prompt_tier(context_window: int) -> str:
    """Determine prompt tier from context window size.

    Boundaries chosen based on measured full prompt size (~12k tokens):
    - 128k+: all components injected (existing behaviour)
    - 32k–128k: DK budget reduced, priming budget halved
    - 16k–32k: bootstrap/vision/specialty/DK/memory_guide omitted
    - <16k: additionally permissions/priming/org/messaging/emotion omitted
    """
    if context_window >= 128_000:
        return TIER_FULL
    if context_window >= 32_000:
        return TIER_STANDARD
    if context_window >= 16_000:
        return TIER_LIGHT
    return TIER_MINIMAL

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


def _shorten_model_name(model: str | None) -> str | None:
    """Convert a raw model identifier to a short human-readable label."""
    if not model or not isinstance(model, str):
        return None
    # Strip provider prefix after last "/" (e.g. bedrock/jp.anthropic.claude-... -> jp.anthropic.claude-...)
    m = model.rsplit("/", 1)[-1]
    # Strip dot-separated provider namespaces (e.g. jp.anthropic.claude-sonnet-4-6 -> claude-sonnet-4-6)
    # Only strip segments that look like provider names (all-alpha, no digits)
    while "." in m:
        head, _, tail = m.partition(".")
        if head.isalpha():
            m = tail
        else:
            break
    low = m.lower()
    # Claude family
    if "opus" in low:
        return "Opus"
    if "sonnet" in low:
        return "Sonnet"
    if "haiku" in low:
        return "Haiku"
    # GPT family — extract version number
    if low.startswith("gpt-"):
        import re
        ver = re.match(r"gpt-([\d.]+)", low)
        label = f"GPT-{ver.group(1)}" if ver else "GPT"
        if "mini" in low:
            label += "m"
        return label
    # Fallback: return as-is
    return m


def _format_anima_entry(name: str, speciality: str | None, model: str | None = None) -> str:
    """Format an anima name with optional speciality and model annotation."""
    short_model = _shorten_model_name(model)
    parts = []
    if speciality:
        parts.append(speciality)
    if short_model:
        parts.append(short_model)
    if parts:
        return f"{name} ({', '.join(parts)})"
    return name


def _build_full_org_tree(
    anima_name: str,
    all_animas: dict[str, Any],
) -> str:
    """Build an indented full organization tree for top-level animas."""
    _ss = _load_section_strings()
    you_marker = _ss.get("you_marker", "  ← you")

    # Build children map: parent -> list of children
    children: dict[str | None, list[str]] = {}
    for name, pcfg in all_animas.items():
        parent = pcfg.supervisor
        children.setdefault(parent, []).append(name)
    for k in children:
        children[k].sort()

    lines: list[str] = []

    def _render(name: str, prefix: str, is_last: bool, is_root: bool) -> None:
        acfg = all_animas.get(name)
        spec = acfg.speciality if acfg else None
        mdl = acfg.model if acfg else None
        label = _format_anima_entry(name, spec, mdl)
        if is_root:
            marker = ""
            suffix = you_marker if name == anima_name else ""
        else:
            marker = "└── " if is_last else "├── "
            suffix = you_marker if name == anima_name else ""
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


def _scan_all_animas(animas_dir: Path) -> dict[str, Any]:
    """Scan anima directories and merge with config.json for a complete org map.

    Reads ``status.json`` from each anima directory and merges with
    config.json entries (config overrides where present).
    Returns a dict of ``name -> AnimaModelConfig``-like objects.
    """
    from core.config import load_config
    from core.config.models import AnimaModelConfig

    try:
        config = load_config()
    except Exception:
        config = None

    config_animas = config.animas if config else {}
    result: dict[str, AnimaModelConfig] = {}

    if not animas_dir.is_dir():
        return config_animas

    for d in sorted(animas_dir.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_", "tmp")):
            continue
        if not (d / "identity.md").exists():
            continue

        name = d.name
        supervisor: str | None = None
        speciality: str | None = None
        role: str | None = None
        model: str | None = None

        # status.json is the SSoT; config.animas is fallback only.
        status_has_supervisor = False
        status_has_speciality = False
        status_path = d / "status.json"
        if status_path.exists():
            try:
                import json
                data = json.loads(status_path.read_text(encoding="utf-8"))
                if "supervisor" in data:
                    supervisor = data["supervisor"] or None
                    status_has_supervisor = True
                if "speciality" in data:
                    speciality = data["speciality"] or None
                    status_has_speciality = True
                role = data.get("role") or None
                model = data.get("model") or None
            except Exception:
                pass

        # Fallback to config.animas when status.json omits the field
        if name in config_animas:
            cfg = config_animas[name]
            if not status_has_supervisor and cfg.supervisor is not None:
                supervisor = cfg.supervisor
            if not status_has_speciality and cfg.speciality is not None:
                speciality = cfg.speciality

        if not speciality and role:
            speciality = role

        result[name] = AnimaModelConfig(supervisor=supervisor, speciality=speciality, model=model)

    for name, cfg in config_animas.items():
        if name not in result:
            result[name] = cfg

    return result


def _build_org_context(anima_name: str, other_animas: list[str], execution_mode: str = "s") -> str:
    """Build organisation context section from directory scan + config.json.

    Scans anima directories for ``status.json`` supervisor relationships
    and merges with config.json for a complete org tree.
    """
    from core.tooling.prompt_db import get_prompt_store
    from core.paths import get_data_dir

    _fs = _load_fallback_strings()

    animas_dir = get_data_dir() / "animas"
    all_animas = _scan_all_animas(animas_dir)

    if not all_animas:
        return ""

    my_config = all_animas.get(anima_name)
    my_supervisor = my_config.supervisor if my_config else None
    my_speciality = my_config.speciality if my_config else None
    is_top_level = my_supervisor is None

    # Top-level anima with subordinates: show full org tree
    if is_top_level and len(all_animas) > 1:
        anima_speciality = my_speciality or _fs.get("unset", "(not set)")
        tree_text = _build_full_org_tree(anima_name, all_animas)
        parts = [
            load_prompt(
                "builder/org_context_toplevel",
                anima_speciality=anima_speciality,
                tree_text=tree_text,
            ),
        ]
        if other_animas:
            cr_key = "communication_rules_s" if _is_mcp_mode(execution_mode) else "communication_rules"
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
        sup_cfg = all_animas.get(my_supervisor)
        supervisor_line = _format_anima_entry(
            my_supervisor,
            sup_cfg.speciality if sup_cfg else None,
            sup_cfg.model if sup_cfg else None,
        )
    else:
        supervisor_line = _fs.get("none_top_level", "(none — you are top-level)")

    # Subordinates: animas whose supervisor is me
    subordinates: list[str] = []
    for name in sorted(all_animas):
        if name == anima_name:
            continue
        pcfg = all_animas[name]
        if pcfg.supervisor == anima_name:
            subordinates.append(_format_anima_entry(name, pcfg.speciality, pcfg.model))

    # Peers: animas with the same supervisor (excluding self)
    peers: list[str] = []
    if my_supervisor is not None:
        for name in sorted(all_animas):
            if name == anima_name:
                continue
            pcfg = all_animas[name]
            if pcfg.supervisor == my_supervisor:
                peers.append(_format_anima_entry(name, pcfg.speciality, pcfg.model))

    subordinates_line = ", ".join(subordinates) if subordinates else _fs.get("none", "(none)")
    peers_line = ", ".join(peers) if peers else _fs.get("none", "(none)")
    anima_speciality = my_speciality or _fs.get("unset", "(not set)")

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
        cr_key = "communication_rules_s" if _is_mcp_mode(execution_mode) else "communication_rules"
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
    execution_mode: str = "s",
) -> str:
    """Build the messaging instructions with resolved paths."""
    from core.tooling.prompt_db import get_prompt_store

    _fs = _load_fallback_strings()
    self_name = anima_dir.name
    main_py = PROJECT_DIR / "main.py"
    animas_line = (
        ", ".join(other_animas) if other_animas else _fs.get("no_other_animas", "(no other employees yet)")
    )

    db_key = "messaging_s" if _is_mcp_mode(execution_mode) else "messaging"
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


def _load_a_reflection() -> str:
    """Load the A mode reflection/retry prompt template."""
    try:
        return load_prompt("a_reflection")
    except Exception:
        logger.debug("a_reflection template not found, skipping")
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
    _ss = _load_section_strings()
    header = _ss.get("recent_tool_results_header", "## Recent Tool Results")
    return f"{header}\n\n" + "\n".join(tool_lines)



def _build_human_notification_guidance(execution_mode: str = "") -> str:
    """Build the human notification instruction for top-level Animas."""
    if _is_mcp_mode(execution_mode):
        how_to = load_prompt("builder/human_notification_howto_s")
    else:
        how_to = load_prompt("builder/human_notification_howto_other")

    return load_prompt("builder/human_notification", how_to=how_to)


def build_system_prompt(
    memory: MemoryManager,
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
    priming_section: str = "",
    execution_mode: str = "s",
    message: str = "",
    retriever: object | None = None,
    *,
    trigger: str = "",
    context_window: int = 200_000,
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
    tier = resolve_prompt_tier(context_window)
    _ss = _load_section_strings()
    _fs = _load_fallback_strings()

    # ── Trigger-based section filtering ──────────────────────────
    is_inbox = trigger.startswith("inbox:")
    is_heartbeat = trigger == "heartbeat"
    is_cron = trigger.startswith("cron:")
    is_task = trigger.startswith("task:")
    is_background_auto = is_heartbeat or is_cron
    is_chat = not (is_inbox or is_background_auto or is_task)

    # ── Pre-compute values needed across multiple groups ──────────

    # DB-first prompt store (singleton); used for system sections & tool guides
    from core.tooling.prompt_db import get_default_guide, get_prompt_store
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
    parts.append(_ss.get("group1_header", "# 1. Environment and Action Rules"))

    if is_task:
        parts.append(f"Anima: {pd.name}\nData directory: {data_dir}")
    else:
        _env = (_prompt_store.get_section("environment") if _prompt_store else None)
        if _env:
            try:
                _env = _env.format(data_dir=data_dir, anima_name=pd.name)
            except (KeyError, IndexError):
                pass
        else:
            _env = load_prompt("environment", data_dir=data_dir, anima_name=pd.name)
        if _env:
            if tier == TIER_LIGHT and len(_env) > 8000:
                _env = _env[:8000]
            elif tier == TIER_MINIMAL and len(_env) > 4800:
                _env = _env[:4800]
            parts.append(_env)

    # ── Identity + Injection: immediately after environment ──────
    # Placed here so the "# Identity" teaser in environment.md is
    # resolved without intervening system boilerplate.
    identity = memory.read_identity()
    if identity:
        parts.append(identity)

    injection = memory.read_injection()
    if injection:
        parts.append(injection)

    current_time = now_jst().strftime("%Y-%m-%d %H:%M (%Z)")
    parts.append(f"{_ss.get('current_time_label', '**Current time**:')} {current_time}")

    if tier in (TIER_FULL, TIER_STANDARD):
        _br = (
            _prompt_store.get_section("behavior_rules") if _prompt_store else None
        ) or load_prompt("behavior_rules")
        if _br:
            parts.append(_br)

    if not is_task:
        _tdi = load_prompt("tool_data_interpretation")
        if _tdi:
            parts.append(_tdi)

    # ── Group 2: あなた自身（補足） ─────────────────────────────
    # NOTE: identity.md / injection.md は Group 1 直後に注入済み。
    #       ここでは bootstrap, vision, specialty, permissions を追加。
    parts.append(_ss.get("group2_header", "# 2. Yourself"))

    if not is_task and tier in (TIER_FULL, TIER_STANDARD):
        bootstrap = memory.read_bootstrap()
        if bootstrap:
            parts.append(bootstrap)

    if not is_task and tier in (TIER_FULL, TIER_STANDARD):
        company_vision = memory.read_company_vision()
        if company_vision:
            parts.append(company_vision)

    if not is_inbox and not is_background_auto and not is_task and tier in (TIER_FULL, TIER_STANDARD):
        specialty = memory.read_specialty_prompt()
        if specialty:
            parts.append(specialty)

    if tier != TIER_MINIMAL:
        if permissions:
            parts.append(permissions)

    # ── Group 3: 現在の状況 ───────────────────────────────────
    parts.append(_ss.get("group3_header", "# 3. Current Situation"))

    # current_state: skip for task; summary (500 chars) for inbox
    if not is_task:
        _state_max = {
            TIER_FULL: _CURRENT_TASK_MAX_CHARS,
            TIER_STANDARD: _CURRENT_TASK_MAX_CHARS,
            TIER_LIGHT: 1000,
            TIER_MINIMAL: 500,
        }[tier]
        if is_inbox:
            _state_max = min(_state_max, 500)
        state = memory.read_current_state()
        if state and state.strip() != "status: idle":
            if len(state) > _state_max:
                truncated = state[-_state_max:]
                first_nl = truncated.find("\n")
                if first_nl != -1 and first_nl < _state_max * 0.2:
                    truncated = truncated[first_nl + 1:]
                state = f"{_fs.get('truncated', '(earlier portion omitted)')}\n\n{truncated}"
            parts.append(load_prompt("builder/task_in_progress", state=state))
        elif state:
            parts.append(f"{_ss.get('current_state_header', '## Current State')}\n\n{state}")

    # pending: skip for inbox and task
    if not is_inbox and not is_task:
        pending = memory.read_pending()
        if pending:
            parts.append(f"{_ss.get('pending_tasks_header', '## Pending Tasks')}\n\n{pending}")

    # Task Queue, Resolution Registry, Recent Outbound: skip for inbox and task
    if not is_inbox and not is_task and tier in (TIER_FULL, TIER_STANDARD):
        try:
            from core.memory.task_queue import TaskQueueManager
            task_queue = TaskQueueManager(memory.anima_dir)
            task_summary = task_queue.format_for_priming()
            if task_summary:
                parts.append(load_prompt("builder/task_queue", task_summary=task_summary))
        except Exception:
            logger.debug("Failed to inject task queue", exc_info=True)

        try:
            resolutions = memory.read_resolutions(days=7)
            if resolutions:
                seen_issues: dict[str, dict] = {}
                for r in resolutions:
                    key = r.get("issue", "")
                    seen_issues[key] = r
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

    # Priming: skip for task
    if not is_task and priming_section:
        parts.append(priming_section)

    # Recent tool results: only for chat
    if is_chat and tier in (TIER_FULL, TIER_STANDARD):
        try:
            _model_cfg = memory.read_model_config()
            recent_tools = _build_recent_tool_section(pd, _model_cfg)
            if recent_tools:
                parts.append(recent_tools)
        except Exception:
            logger.debug("Failed to inject recent tool results", exc_info=True)

    # ── Group 4: 記憶と能力 ───────────────────────────────────
    parts.append(_ss.get("group4_header", "# 4. Memory and Capabilities"))

    if tier in (TIER_FULL, TIER_STANDARD):
        # Memory directory guide
        _none = _fs.get("none", "(none)")
        _common = _ss.get("common_label", "(shared)")
        knowledge_list = ", ".join(memory.list_knowledge_files()) or _none
        episode_list = ", ".join(memory.list_episode_files()[:7]) or _none
        procedure_list = ", ".join(memory.list_procedure_files()) or _none
        skill_lines: list[str] = []
        for m in skill_metas:
            desc = f": {m.description}" if m.description else ""
            skill_lines.append(f"- {m.name}{desc}")
        for m in common_skill_metas:
            desc = f": {m.description}" if m.description else ""
            skill_lines.append(f"- {m.name}{_common}{desc}")
        for m in procedure_metas:
            desc = f": {m.description}" if m.description else ""
            skill_lines.append(f"- {m.name} (手順){desc}")
        skill_names = "\n".join(skill_lines) or _none
        shared_users_list = ", ".join(memory.list_shared_users()) or _none

        parts.append(load_prompt(
            "memory_guide",
            anima_dir=pd,
            knowledge_list=knowledge_list,
            episode_list=episode_list,
            procedure_list=procedure_list,
            skill_names=skill_names,
            shared_users_list=shared_users_list,
        ))

    # ── Distilled Knowledge Injection (skip for task) ─────
    injected_knowledge_files: list[str] = []
    injected_procedures: list[Path] = []
    overflow_files: list[str] = []

    if is_task:
        knowledge_budget = 0
    elif tier == TIER_FULL:
        from core.prompt.context import resolve_context_window
        try:
            _model_config = memory.read_model_config()
            ctx_window = resolve_context_window(_model_config.model)
        except Exception:
            ctx_window = 128_000
        knowledge_budget = min(int(ctx_window * 0.05), 4000)
    elif tier == TIER_STANDARD:
        knowledge_budget = min(int(context_window * 0.03), 2000)
    else:
        knowledge_budget = 0

    procedures_list, knowledge_list = memory.collect_distilled_knowledge_separated()

    used_tokens = 0

    proc_parts: list[str] = []
    for entry in procedures_list:
        est_tokens = len(entry["content"]) // 3
        if used_tokens + est_tokens <= knowledge_budget:
            proc_parts.append(
                f"### {entry['name']}\n\n{entry['content']}"
            )
            used_tokens += est_tokens
            injected_procedures.append(Path(entry["path"]))
        else:
            overflow_files.append(entry["name"])

    if proc_parts:
        parts.append(
            f"{_ss.get('procedures_header', '## Procedures')}\n\n"
            + "\n\n---\n\n".join(proc_parts)
        )

    know_parts: list[str] = []
    for entry in knowledge_list:
        est_tokens = len(entry["content"]) // 3
        if used_tokens + est_tokens <= knowledge_budget:
            know_parts.append(
                f"### {entry['name']}\n\n{entry['content']}"
            )
            used_tokens += est_tokens
            injected_knowledge_files.append(entry["name"])
        else:
            overflow_files.append(entry["name"])

    if know_parts:
        parts.append(
            f"{_ss.get('distilled_knowledge_header', '## Distilled Knowledge')}\n\n"
            + "\n\n---\n\n".join(know_parts)
        )

    if not is_task and tier in (TIER_FULL, TIER_STANDARD):
        common_knowledge_dir = data_dir / "common_knowledge"
        if common_knowledge_dir.exists() and any(common_knowledge_dir.rglob("*.md")):
            parts.append(load_prompt("builder/common_knowledge_hint"))

        has_newstaff = any(m.name == "newstaff" for m in skill_metas)
        if has_newstaff:
            if _is_mcp_mode(execution_mode):
                parts.append(load_prompt("builder/hiring_rules_s"))
            else:
                parts.append(load_prompt("builder/hiring_rules_other"))

    # ── Tool usage guides from DB (with hardcoded fallback) ──
    if not _prompt_store:
        logger.warning("Tool prompt DB unavailable; using hardcoded fallback guides")

    if is_heartbeat:
        try:
            parts.append(load_prompt("builder/heartbeat_tool_instruction"))
        except FileNotFoundError:
            parts.append(
                "Heartbeatでは**観察・報告・計画・フォローアップ**にツールを使ってください。\n"
                "- OK: チャネル読み取り、記憶検索、メッセージ送信、タスク更新、pending作成、外部ツール確認\n"
                "- NG: コード変更、ファイル大量編集、長時間の分析・調査\n"
                "重い作業が必要な場合は state/pending/ にタスクファイルを書き出してください。"
            )
    else:
        if _is_mcp_mode(execution_mode):
            _s_builtin = (
                _prompt_store.get_guide("s_builtin") if _prompt_store else None
            ) or get_default_guide("s_builtin")
            if _s_builtin:
                parts.append(_s_builtin)
            _s_mcp = (
                _prompt_store.get_guide("s_mcp") if _prompt_store else None
            ) or get_default_guide("s_mcp")
            if _s_mcp:
                parts.append(_s_mcp)
        else:
            _non_s = (
                _prompt_store.get_guide("non_s") if _prompt_store else None
            ) or get_default_guide("non_s")
            if _non_s:
                parts.append(_non_s)

    # External tools guide (include for heartbeat too for observation)
    _EXTERNAL_TOOLS_KEYWORDS = {"外部ツール", "External Tools", "external tools"}
    if permissions and any(kw in permissions for kw in _EXTERNAL_TOOLS_KEYWORDS) and (tool_registry or personal_tools):
        if execution_mode == "a":
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
    parts.append(_ss.get("group5_header", "# 5. Organization and Communication"))

    # hiring_context: skip for inbox and task
    if not is_inbox and not is_task and tier in (TIER_FULL, TIER_STANDARD):
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

    if tier in (TIER_FULL, TIER_STANDARD):
        org_context = _build_org_context(pd.name, other_animas, execution_mode)
        if org_context:
            parts.append(org_context)

        if not is_task:
            _msg = _build_messaging_section(pd, other_animas, execution_mode)
            if is_background_auto and len(_msg) > 500:
                _msg = _msg[:500] + "\n" + _fs.get("summary", "(summary)")
            parts.append(_msg)

            # human_notification: skip for inbox and task
            if not is_inbox:
                try:
                    from core.config import load_config as _load_cfg
                    _cfg = _load_cfg()
                    _my_pcfg = _cfg.animas.get(pd.name)
                    _is_top_level = _my_pcfg is None or _my_pcfg.supervisor is None
                    if _is_top_level and _cfg.human_notification.enabled:
                        parts.append(_build_human_notification_guidance(execution_mode))
                except Exception:
                    logger.debug("Skipped human notification guidance injection", exc_info=True)
    elif not is_task and tier == TIER_LIGHT:
        try:
            parts.append(load_prompt("builder/light_tier_org", anima_name=pd.name))
        except FileNotFoundError:
            parts.append(f"あなたは{pd.name}です。他のアニマとはsend_messageで通信できます。")
            parts.append("メッセージはsend_messageツールで送信してください。")

    # ── Group 6: メタ設定 ─────────────────────────────────────
    parts.append(_ss.get("group6_header", "# 6. Meta Settings"))

    # emotion: skip for background-auto (heartbeat/cron) and task
    if not is_background_auto and not is_task and tier in (TIER_FULL, TIER_STANDARD):
        _ei = (
            _prompt_store.get_section("emotion_instruction")
            if _prompt_store else None
        ) or EMOTION_INSTRUCTION
        if _ei:
            parts.append(_ei)

    # a_reflection: skip for inbox and background-auto (heartbeat/cron)
    if not is_inbox and not is_background_auto and tier in (TIER_FULL, TIER_STANDARD):
        if execution_mode == "a":
            _ar = (
                _prompt_store.get_section("a_reflection")
                if _prompt_store else None
            ) or _load_a_reflection()
            if _ar:
                parts.append(_ar)

    # c_response_requirement: Codex models need explicit text-output guidance
    # because model_instructions_file replaces the CLI's built-in preamble
    # prompt.  Only injected for chat / inbox (not background / task).
    if execution_mode == "c" and not is_background_auto and not is_task:
        parts.append(
            "## 応答要件\n"
            "あなたはユーザーとの対話において、**必ずテキストで応答**してください。\n"
            "ツール呼び出しを行った場合でも、その結果の要約やユーザーへの返答を\n"
            "テキストメッセージとして出力してください。\n"
            "挨拶・質問・雑談などの会話メッセージには、ツール呼び出しの前後に\n"
            "自然なテキスト応答を必ず含めてください。"
        )

    # ── Final assembly ────────────────────────────────────────
    prompt = "\n\n---\n\n".join(parts)
    logger.debug(
        "System prompt built: %d sections, total_len=%d, tier=%s, context_window=%d",
        len(parts), len(prompt), tier, context_window,
    )
    return BuildResult(
        system_prompt=prompt,
        injected_procedures=injected_procedures,
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