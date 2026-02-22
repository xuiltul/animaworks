from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

import logging
from pathlib import Path
from typing import Any

from core.time_utils import now_jst

logger = logging.getLogger("animaworks.skill_tool")

# ── Builtin placeholders ─────────────────────────────────


def _resolve_builtins(anima_dir: Path) -> dict[str, str]:
    """Return builtin placeholder values."""
    return {
        "now_jst": now_jst().isoformat(),
        "anima_name": anima_dir.name,
        "anima_dir": str(anima_dir),
    }


def apply_builtins(content: str, builtins: dict[str, str]) -> str:
    """Replace {{key}} placeholders with builtin values."""
    for key, value in builtins.items():
        content = content.replace(f"{{{{{key}}}}}", value)
    return content


# ── Description builder ──────────────────────────────────

_DESCRIPTION_BUDGET = 8000


def build_skill_tool_description(
    skill_metas: list[Any],
    common_skill_metas: list[Any],
    procedure_metas: list[Any],
) -> str:
    """Build dynamic description for the skill tool.

    Aggregates all skill/procedure name+description into an
    <available_skills> block within the budget.
    """
    lines = [
        "スキル・手順書をオンデマンドでロードする。",
        "スキルを発動すると、詳細な手順がこのツールのレスポンスとして提供される。",
        "該当するスキルがある場合に使用すること。",
        "",
        "<available_skills>",
    ]
    total = sum(len(l) for l in lines)

    for meta in skill_metas:
        entry = f"- {meta.name}: {meta.description}"
        if total + len(entry) > _DESCRIPTION_BUDGET:
            lines.append("(以降省略)")
            break
        lines.append(entry)
        total += len(entry)

    for meta in common_skill_metas:
        entry = f"- {meta.name} (共通): {meta.description}"
        if total + len(entry) > _DESCRIPTION_BUDGET:
            lines.append("(以降省略)")
            break
        lines.append(entry)
        total += len(entry)

    for meta in procedure_metas:
        entry = f"- {meta.name} (手順): {meta.description}"
        if total + len(entry) > _DESCRIPTION_BUDGET:
            lines.append("(以降省略)")
            break
        lines.append(entry)
        total += len(entry)

    lines.append("</available_skills>")
    return "\n".join(lines)


# ── Skill loader ─────────────────────────────────────────

def load_and_render_skill(
    skill_name: str,
    anima_dir: Path,
    skills_dir: Path,
    common_skills_dir: Path,
    procedures_dir: Path,
    context: str = "",
) -> str:
    """Load a skill file, apply builtins, and build the response.

    Resolution order: personal skills -> common skills -> procedures.
    Personal skills take priority over common skills with the same name.
    """
    # Resolve skill file
    path, skill_type = _resolve_skill_path(
        skill_name, skills_dir, common_skills_dir, procedures_dir
    )
    if path is None:
        available = _list_available_names(
            skills_dir, common_skills_dir, procedures_dir
        )
        return (
            f"スキル '{skill_name}' が見つかりません。\n"
            f"利用可能なスキル: {', '.join(available)}"
        )

    # Read and strip frontmatter
    raw = path.read_text(encoding="utf-8")
    content, frontmatter = _strip_frontmatter(raw)

    # Apply builtin placeholders
    builtins = _resolve_builtins(anima_dir)
    content = apply_builtins(content, builtins)

    # Build response
    parts: list[str] = []
    parts.append(content)

    # Append context if provided
    if context:
        parts.append(f"\n## コンテキスト\n{context}")

    # Append allowed_tools soft constraint
    allowed_tools = frontmatter.get("allowed_tools", [])
    if allowed_tools:
        parts.append("\n## ツール制約")
        parts.append("このスキルの実行中は以下のツールのみ使用してください:")
        for tool in allowed_tools:
            parts.append(f"- {tool}")

    return "\n".join(parts)


# ── Internal helpers ─────────────────────────────────────

def _resolve_skill_path(
    name: str,
    skills_dir: Path,
    common_skills_dir: Path,
    procedures_dir: Path,
) -> tuple[Path | None, str]:
    """Resolve skill name to file path. Returns (path, type)."""
    # Guard against path traversal
    if "/" in name or "\\" in name or ".." in name:
        return None, ""

    # 1. Personal skills (highest priority)
    candidate = skills_dir / f"{name}.md"
    if candidate.is_file():
        return candidate, "個人"

    # 2. Common skills
    candidate = common_skills_dir / f"{name}.md"
    if candidate.is_file():
        return candidate, "共通"

    # 3. Procedures
    candidate = procedures_dir / f"{name}.md"
    if candidate.is_file():
        return candidate, "手順"

    return None, ""


def _list_available_names(
    skills_dir: Path,
    common_skills_dir: Path,
    procedures_dir: Path,
) -> list[str]:
    """List all available skill/procedure names."""
    names: list[str] = []
    for d in (skills_dir, common_skills_dir, procedures_dir):
        if d.is_dir():
            names.extend(f.stem for f in sorted(d.glob("*.md")))
    return names


def _strip_frontmatter(text: str) -> tuple[str, dict[str, Any]]:
    """Strip YAML frontmatter and return (content, parsed_frontmatter)."""
    if not text.startswith("---"):
        return text, {}

    parts = text.split("---", 2)
    if len(parts) < 3:
        return text, {}

    import yaml
    try:
        fm = yaml.safe_load(parts[1])
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}

    return parts[2].lstrip("\n"), fm
