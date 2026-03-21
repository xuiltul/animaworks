from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""System prompt builder — facade module.

The heavy lifting is delegated to submodules:
  - :mod:`core.prompt.sections`   — template loading helpers
  - :mod:`core.prompt.org_context` — organisation tree/context
  - :mod:`core.prompt.messaging`  — messaging & notification sections
  - :mod:`core.prompt.assembler`  — budget allocation & XML assembly

This module re-exports every symbol that tests reference via
``patch("core.prompt.builder.XXX", ...)``.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.i18n import t
from core.memory import MemoryManager
from core.memory.shortterm import ShortTermMemory
from core.paths import get_data_dir, load_prompt  # noqa: F401 — tests patch core.prompt.builder.load_prompt
from core.prompt.assembler import (
    _KNOW_SUMMARY_BUDGET,  # noqa: F401
    _MIN_SYSTEM_BUDGET,  # noqa: F401
    _PROC_SUMMARY_BUDGET,
    _REFERENCE_WINDOW,
    SectionEntry,  # noqa: F401
    _allocate_sections,
    _assemble_with_tags,
    _compute_system_budget,
    _normalize_headings,  # noqa: F401
)
from core.prompt.messaging import (
    _build_human_notification_guidance,  # noqa: F401
    _build_messaging_section,
    _build_recent_tool_section,
    _load_a_reflection,
)
from core.prompt.org_context import (
    _build_full_org_tree,  # noqa: F401
    _build_org_context,
    _discover_other_animas,
    _format_anima_entry,  # noqa: F401
    _is_mcp_mode,
)
from core.prompt.sections import (
    _extract_entry_summary,
    _load_fallback_strings,
    _load_section_strings,
)
from core.time_utils import now_local

logger = logging.getLogger("animaworks.prompt_builder")

# Re-exported constants
_MCP_MODES = frozenset({"s", "c", "d", "g"})
_CURRENT_STATE_MAX_CHARS = 3000

# ── Prompt tier constants ─────────────────────────────────────
TIER_FULL = "full"
TIER_STANDARD = "standard"
TIER_LIGHT = "light"
TIER_MINIMAL = "minimal"


def resolve_prompt_tier(context_window: int) -> str:
    """Determine prompt tier from context window size."""
    if context_window >= 128_000:
        return TIER_FULL
    if context_window >= 32_000:
        return TIER_STANDARD
    if context_window >= 16_000:
        return TIER_LIGHT
    return TIER_MINIMAL


def _build_emotion_instruction() -> str:
    """Build EMOTION_INSTRUCTION with the canonical emotion list."""
    from core.schemas import VALID_EMOTIONS

    emotion_list = ", ".join(sorted(VALID_EMOTIONS))
    return load_prompt("builder/emotion_instruction", emotion_list=emotion_list)


EMOTION_INSTRUCTION = _build_emotion_instruction()


def _read_default_workspace(anima_dir: Path) -> str:
    """Read default_workspace from status.json and resolve via workspace registry."""
    from core.workspace import resolve_default_workspace

    resolved, alias = resolve_default_workspace(anima_dir)
    if not alias:
        return ""
    if resolved:
        return t("builder.default_workspace", path=str(resolved), alias=alias)
    return t("builder.default_workspace_unresolved", alias=alias)


@dataclass
class BuildResult:
    """Result of system prompt building."""

    system_prompt: str
    injected_procedures: list[Path] = field(default_factory=list)
    injected_knowledge_files: list[str] = field(default_factory=list)
    overflow_files: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.system_prompt

    def __len__(self) -> int:
        return len(self.system_prompt)

    def encode(self, encoding: str = "utf-8") -> bytes:
        return self.system_prompt.encode(encoding)

    def __contains__(self, item: str) -> bool:
        return item in self.system_prompt

    def __add__(self, other: str) -> str:
        return self.system_prompt + other

    def __radd__(self, other: str) -> str:
        return other + self.system_prompt

    def index(self, sub: str, *args: int) -> int:
        return self.system_prompt.index(sub, *args)

    def count(self, sub: str, *args: int) -> int:
        return self.system_prompt.count(sub, *args)


# ── Per-group section builders ────────────────────────────────
# Private helpers — kept in builder.py so they share the same
# ``load_prompt`` binding that tests patch.


def _build_group1(
    pd: Path,
    data_dir: Path,
    memory: MemoryManager,
    is_task: bool,
    prompt_store: Any,
    _ss: dict[str, str],
) -> list[SectionEntry]:
    """Group 1: Environment, identity, injection, time, behaviour rules."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group1_header", "# 1. Environment and Action Rules"), "group1_header", 1)

    dw = _read_default_workspace(pd)
    if dw:
        _add(dw, "default_workspace", 2)

    if is_task:
        _add(f"Anima: {pd.name}\nData directory: {data_dir}", "environment", 1)
    else:
        _env = prompt_store.get_section("environment") if prompt_store else None
        if _env:
            try:
                _env = _env.format(data_dir=data_dir, anima_name=pd.name)
            except (KeyError, IndexError):
                pass
        else:
            _env = load_prompt("environment", data_dir=data_dir, anima_name=pd.name)
        if _env:
            _add(_env, "environment", 2)

    identity = memory.read_identity()
    if identity:
        _add(identity, "identity", 1)

    injection = memory.read_injection()
    if injection:
        _add(injection, "injection", 1)
        try:
            from core.config import load_config

            config = load_config()
            threshold = config.prompt.injection_size_warning_chars
            if len(injection) > threshold:
                _add(
                    t("builder.injection_size_warning", size=len(injection), threshold=threshold),
                    "injection_size_warning",
                    1,
                )
                logger.warning("injection.md oversized: %d chars (threshold=%d)", len(injection), threshold)
        except Exception:
            pass

    current_time = now_local().strftime("%Y-%m-%d %H:%M (%Z)")
    _add(f"{_ss.get('current_time_label', '**Current time**:')} {current_time}", "current_time", 1)

    _br = (prompt_store.get_section("behavior_rules") if prompt_store else None) or load_prompt("behavior_rules")
    if _br:
        _add(_br, "behavior_rules", 2)

    if not is_task:
        _tdi = load_prompt("tool_data_interpretation")
        if _tdi:
            _add(_tdi, "tool_data_interpretation", 2)

    return out


def _build_group2(
    memory: MemoryManager,
    permissions: str,
    is_inbox: bool,
    is_background_auto: bool,
    is_task: bool,
    _ss: dict[str, str],
) -> list[SectionEntry]:
    """Group 2: Bootstrap, vision, specialty, permissions."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group2_header", "# 2. Yourself"), "group2_header", 1)
    if not is_task:
        b = memory.read_bootstrap()
        if b:
            _add(b, "bootstrap", 3)
    if not is_task:
        v = memory.read_company_vision()
        if v:
            _add(v, "vision", 3)
    if not is_inbox and not is_background_auto and not is_task:
        sp = memory.read_specialty_prompt()
        if sp:
            _add(sp, "specialty", 3)
    if permissions:
        _add(permissions, "permissions", 2)
    return out


def _build_group3(
    pd: Path,
    memory: MemoryManager,
    scale: float,
    priming_section: str,
    pending_human_notifications: str,
    execution_mode: str,
    is_inbox: bool,
    is_heartbeat: bool,
    is_chat: bool,
    is_task: bool,
    _ss: dict[str, str],
    _fs: dict[str, str],
) -> list[SectionEntry]:
    """Group 3: Current state, resolutions, priming, notifications, recent tools."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group3_header", "# 3. Current Situation"), "group3_header", 1)

    if not is_task:
        _state_max = max(int(_CURRENT_STATE_MAX_CHARS * scale), 500)
        if is_inbox:
            _state_max = min(_state_max, 500)
        state = memory.read_current_state()
        state_content = ""
        if state and state.strip() != "status: idle":
            if len(state) > _state_max:
                truncated = state[-_state_max:]
                first_nl = truncated.find("\n")
                if first_nl != -1 and first_nl < _state_max * 0.2:
                    truncated = truncated[first_nl + 1 :]
                state = f"{_fs.get('truncated', '(earlier portion omitted)')}\n\n{truncated}"
            state_content = load_prompt("builder/task_in_progress", state=state)
        elif state:
            state_content = f"{_ss.get('current_state_header', '## Current State')}\n\n{state}"
        if state_content:
            _add(state_content, "current_state", 2, "elastic")

    if not is_inbox and not is_task:
        try:
            resolutions = memory.read_resolutions(days=7)
            if resolutions:
                seen: dict[str, dict] = {}
                for r in resolutions:
                    seen[r.get("issue", "")] = r
                deduped = sorted(seen.values(), key=lambda x: x.get("ts", ""))
                lines = [
                    f"- [{r.get('ts', '')[:16]}] {r.get('resolver', 'unknown')}: {r.get('issue', '')}"
                    for r in deduped[-10:]
                ]
                _add(
                    load_prompt("builder/resolution_registry", res_lines="\n".join(lines)),
                    "resolution_registry",
                    3,
                    "elastic",
                )
        except Exception:
            logger.debug("Failed to inject resolution registry", exc_info=True)

    if not is_task and priming_section:
        _add(priming_section, "priming", 2, "elastic")
    if pending_human_notifications and (is_chat or is_heartbeat):
        _add(pending_human_notifications, "pending_human_notifications", 3, "elastic")
    if is_chat and execution_mode.upper() == "B":
        try:
            recent = _build_recent_tool_section(pd, memory.read_model_config())
            if recent:
                _add(recent, "recent_tools", 3, "elastic")
        except Exception:
            logger.debug("Failed to inject recent tool results", exc_info=True)
    return out


def _build_group4(
    pd: Path,
    data_dir: Path,
    memory: MemoryManager,
    scale: float,
    execution_mode: str,
    skill_metas: list[Any],
    prompt_store: Any,
    is_heartbeat: bool,
    is_task: bool,
    tool_registry: list[str] | None,
    personal_tools: dict[str, str] | None,
    _ss: dict[str, str],
    _fs: dict[str, str],
) -> tuple[list[SectionEntry], list[Path], list[str], list[str]]:
    """Group 4: Memory guide, DK, common knowledge, tool guides.

    Returns (sections, injected_procedures, injected_knowledge_files, overflow_files).
    """
    from core.tooling.prompt_db import get_default_guide

    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group4_header", "# 4. Memory and Capabilities"), "group4_header", 1)

    _none = _fs.get("none", "(none)")
    mg = load_prompt(
        "memory_guide",
        anima_dir=pd,
        knowledge_count=len(memory.list_knowledge_files()),
        procedure_count=len(memory.list_procedure_files()),
        shared_users_list=", ".join(memory.list_shared_users()) or _none,
    )
    if mg:
        _add(mg, "memory_guide", 3)

    # ── Distilled Knowledge ───
    injected_procs: list[Path] = []
    injected_know: list[str] = []
    overflow: list[str] = []
    proc_budget = 0 if is_task else max(int(_PROC_SUMMARY_BUDGET * scale), 0)
    know_budget = 0 if is_task else max(int(_KNOW_SUMMARY_BUDGET * scale), 0)
    procedures_list, knowledge_list = memory.collect_distilled_knowledge_separated()

    proc_parts, proc_used = [], 0
    for e in procedures_list:
        line = f"- **{e['name']}**: {_extract_entry_summary(e)}"
        est = len(line) // 3
        if proc_used + est <= proc_budget:
            proc_parts.append(line)
            proc_used += est
            injected_procs.append(Path(e["path"]))
        else:
            overflow.append(e["name"])
    if proc_parts:
        _add(
            f"{_ss.get('procedures_header', '## Procedures')}\n\n" + "\n".join(proc_parts),
            "dk_procedures",
            3,
            "elastic",
        )

    know_parts, know_used = [], 0
    for e in knowledge_list:
        line = f"- **{e['name']}**: {_extract_entry_summary(e)}"
        est = len(line) // 3
        if know_used + est <= know_budget:
            know_parts.append(line)
            know_used += est
            injected_know.append(e["name"])
        else:
            overflow.append(e["name"])
    if know_parts:
        _add(
            f"{_ss.get('distilled_knowledge_header', '## Distilled Knowledge')}\n\n" + "\n".join(know_parts),
            "dk_knowledge",
            3,
            "elastic",
        )

    if not is_task:
        ck_dir = data_dir / "common_knowledge"
        if ck_dir.exists() and any(ck_dir.rglob("*.md")):
            _add(load_prompt("builder/common_knowledge_hint"), "common_knowledge_hint", 4)
        ref_dir = data_dir / "reference"
        if ref_dir.exists() and any(ref_dir.rglob("*.md")):
            _add(load_prompt("builder/reference_hint"), "reference_hint", 4)

    # ── Tool guides ───
    if not prompt_store:
        logger.warning("Tool prompt DB unavailable; using hardcoded fallback guides")
    if is_heartbeat:
        try:
            hb_tool = load_prompt("builder/heartbeat_tool_instruction")
        except FileNotFoundError:
            hb_tool = t("builder.heartbeat_tool_fallback")
        _add(hb_tool, "tool_guides", 2)
    elif _is_mcp_mode(execution_mode):
        sb = (prompt_store.get_guide("s_builtin") if prompt_store else None) or get_default_guide("s_builtin")
        sm = (prompt_store.get_guide("s_mcp") if prompt_store else None) or get_default_guide("s_mcp")
        g = "\n\n".join(p for p in (sb, sm) if p)
        if g:
            _add(g, "tool_guides", 2)
    else:
        ns = (prompt_store.get_guide("non_s") if prompt_store else None) or get_default_guide("non_s")
        if ns:
            _add(ns, "tool_guides", 2)

    if not is_heartbeat:
        _add(
            "\n## CLI Tools\n"
            "For supervisor management, vault, channel management, "
            "background tasks, and external tools (Slack, Chatwork, Gmail, GitHub, etc.):\n"
            "```\nBash: animaworks-tool <tool> <subcommand> [args]\n```\n"
            "Use `skill machine-tool` to see available commands.",
            "tool_guides",
            1,
        )
    if not is_heartbeat and (tool_registry or personal_tools):
        cats = sorted(set((tool_registry or []) + list((personal_tools or {}).keys())))
        if cats:
            et = (
                f"## External Tools\nAvailable categories: {', '.join(cats)}\n"
                f"Use the `skill` tool to look up CLI usage, "
                f"then execute via Bash: `animaworks-tool <tool> <subcommand>`."
            )
            if "machine" in cats:
                et += t("builder.machine_hint")
            _add(et, "external_tools", 2)

    return out, injected_procs, injected_know, overflow


def _build_group5(
    pd: Path,
    memory: MemoryManager,
    other_animas: list[str],
    execution_mode: str,
    prompt_store: Any,
    is_inbox: bool,
    is_background_auto: bool,
    is_task: bool,
    _ss: dict[str, str],
    _fs: dict[str, str],
) -> list[SectionEntry]:
    """Group 5: Org context, messaging, human notification."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group5_header", "# 5. Organization and Communication"), "group5_header", 1)

    oc = _build_org_context(pd.name, other_animas, execution_mode)
    if oc:
        _add(oc, "org_context", 2)

    if not is_task:
        msg = _build_messaging_section(pd, other_animas, execution_mode)
        if is_background_auto and len(msg) > 500:
            msg = msg[:500] + "\n" + _fs.get("summary", "(summary)")
        _add(msg, "messaging", 2)
        if not is_inbox:
            try:
                from core.config import load_config as _lc

                cfg = _lc()
                pcfg = cfg.animas.get(pd.name)
                if (pcfg is None or pcfg.supervisor is None) and cfg.human_notification.enabled:
                    _add(_build_human_notification_guidance(execution_mode), "human_notification", 4)
            except Exception:
                logger.debug("Skipped human notification guidance injection", exc_info=True)
    return out


def _build_group6(
    execution_mode: str,
    prompt_store: Any,
    is_inbox: bool,
    is_chat: bool,
    is_background_auto: bool,
    is_task: bool,
    _ss: dict[str, str],
) -> list[SectionEntry]:
    """Group 6: Emotion, reflection, Codex response requirement."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    if not is_task:
        _add(_ss.get("group6_header", "# 6. Meta Settings"), "group6_header", 1)
    if is_chat:
        ei = (prompt_store.get_section("emotion_instruction") if prompt_store else None) or EMOTION_INSTRUCTION
        if ei:
            _add(ei, "emotion_instruction", 4)
    if not is_inbox and not is_background_auto and execution_mode == "a":
        ar = (prompt_store.get_section("a_reflection") if prompt_store else None) or _load_a_reflection()
        if ar:
            _add(ar, "a_reflection", 4)
    if execution_mode == "c" and not is_background_auto and not is_task:
        _add(t("builder.c_response_requirement"), "c_response_requirement", 2)
    return out


# ── Main entry point ──────────────────────────────────────────


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
    system_budget: int | None = None,
    pending_human_notifications: str = "",
) -> BuildResult:
    """Construct the full system prompt from Markdown files.

    Sections are organised into 6 hierarchical groups.
    """
    pd = memory.anima_dir
    data_dir = get_data_dir()
    budget = _compute_system_budget(context_window, system_budget)
    scale = min(context_window / _REFERENCE_WINDOW, 1.0)
    tier = resolve_prompt_tier(context_window)
    _ss = _load_section_strings()
    _fs = _load_fallback_strings()

    # Trigger flags
    is_inbox = trigger.startswith("inbox:")
    is_heartbeat = trigger == "heartbeat"
    is_cron = trigger.startswith("cron:")
    is_task = trigger.startswith("task:")
    is_consolidation = trigger.startswith("consolidation:")
    is_background_auto = is_heartbeat or is_cron or is_consolidation
    is_chat = not (is_inbox or is_background_auto or is_task)

    from core.tooling.prompt_db import get_prompt_store

    prompt_store = get_prompt_store()
    other_animas = _discover_other_animas(pd)
    skill_metas = memory.list_skill_metas()
    permissions = memory.read_permissions()

    # Assemble sections from all 6 groups
    sections = _build_group1(pd, data_dir, memory, is_task, prompt_store, _ss)
    sections += _build_group2(memory, permissions, is_inbox, is_background_auto, is_task, _ss)
    sections += _build_group3(
        pd,
        memory,
        scale,
        priming_section,
        pending_human_notifications,
        execution_mode,
        is_inbox,
        is_heartbeat,
        is_chat,
        is_task,
        _ss,
        _fs,
    )
    g4, injected_procedures, injected_knowledge_files, overflow_files = _build_group4(
        pd,
        data_dir,
        memory,
        scale,
        execution_mode,
        skill_metas,
        prompt_store,
        is_heartbeat,
        is_task,
        tool_registry,
        personal_tools,
        _ss,
        _fs,
    )
    sections += g4
    sections += _build_group5(
        pd,
        memory,
        other_animas,
        execution_mode,
        prompt_store,
        is_inbox,
        is_background_auto,
        is_task,
        _ss,
        _fs,
    )
    sections += _build_group6(
        execution_mode,
        prompt_store,
        is_inbox,
        is_chat,
        is_background_auto,
        is_task,
        _ss,
    )

    # Budget allocation + Final assembly
    allocated = _allocate_sections(sections, budget)
    prompt = _assemble_with_tags(allocated)
    logger.debug(
        "System prompt built: %d/%d sections, total_len=%d, budget=%d, tier=%s, cw=%d",
        len(allocated),
        len(sections),
        len(prompt),
        budget,
        tier,
        context_window,
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
    """Append short-term memory content to the system prompt."""
    md_content = shortterm.load_markdown()
    if not md_content:
        return base_prompt
    return base_prompt + "\n\n---\n\n" + md_content
