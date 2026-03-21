from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Organization context building for prompt injection."""

import re
from pathlib import Path
from typing import Any

from core.paths import get_data_dir, load_prompt
from core.prompt.sections import _load_fallback_strings, _load_section_strings

# Modes that use MCP-style tool access (built-in + mcp__aw__*).
_MCP_MODES = frozenset({"s", "c", "d", "g"})


def _is_mcp_mode(execution_mode: str) -> bool:
    """Return True for modes using built-in tools + MCP (S, C, D, and G)."""
    return execution_mode in _MCP_MODES


def _discover_other_animas(anima_dir: Path) -> list[str]:
    """List sibling anima directories."""
    animas_root = anima_dir.parent
    self_name = anima_dir.name
    others = []
    for d in sorted(animas_root.iterdir()):
        if d.is_dir() and d.name != self_name and (d / "identity.md").exists():
            others.append(d.name)
    return others


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
            child_is_last = i == len(kids) - 1
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
            _cr = (_cr_store.get_section(cr_key) if _cr_store else None) or load_prompt(cr_key)
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
        _cr = (_cr_store.get_section(cr_key) if _cr_store else None) or load_prompt(cr_key)
        if _cr:
            parts.append(_cr)

    return "\n\n".join(parts)
