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

from core.i18n import t
from core.time_utils import now_local

logger = logging.getLogger("animaworks.skill_tool")

# ── Builtin placeholders ─────────────────────────────────


def _resolve_builtins(anima_dir: Path) -> dict[str, str]:
    """Return builtin placeholder values."""
    return {
        "now_local": now_local().isoformat(),
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
        t("skill.desc_line1"),
        t("skill.desc_line2"),
        t("skill.desc_line3"),
        "",
        "<available_skills>",
    ]
    total = sum(len(line) for line in lines)

    truncated = t("skill.truncated")

    for meta in skill_metas:
        entry = f"- {meta.name}: {meta.description}"
        if total + len(entry) > _DESCRIPTION_BUDGET:
            lines.append(truncated)
            break
        lines.append(entry)
        total += len(entry)

    common_label = t("skill.label_common")
    for meta in common_skill_metas:
        entry = f"- {meta.name} ({common_label}): {meta.description}"
        if total + len(entry) > _DESCRIPTION_BUDGET:
            lines.append(truncated)
            break
        lines.append(entry)
        total += len(entry)

    procedure_label = t("skill.label_procedure")
    for meta in procedure_metas:
        entry = f"- {meta.name} ({procedure_label}): {meta.description}"
        if total + len(entry) > _DESCRIPTION_BUDGET:
            lines.append(truncated)
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
    path, skill_type = _resolve_skill_path(skill_name, skills_dir, common_skills_dir, procedures_dir)
    if path is None:
        available = _list_available_names(skills_dir, common_skills_dir, procedures_dir)
        return t("skill.not_found", skill_name=skill_name, available=", ".join(available))

    # Read and strip frontmatter
    raw = path.read_text(encoding="utf-8")
    content, frontmatter = _strip_frontmatter(raw)

    # Apply builtin placeholders
    builtins = _resolve_builtins(anima_dir)
    content = apply_builtins(content, builtins)

    # Filter gated CLI lines for external tool skills
    tool_name = _skill_name_to_tool_name(skill_name)
    if tool_name:
        permitted = _load_permitted_tools(anima_dir)
        content = _filter_guide_for_tool(content, tool_name, permitted)

    # Build response
    parts: list[str] = []
    parts.append(content)

    # Append context if provided
    if context:
        parts.append(f"\n{t('skill.context_header')}\n{context}")

    # Append allowed_tools soft constraint
    allowed_tools = frontmatter.get("allowed_tools", [])
    if allowed_tools:
        parts.append(f"\n{t('skill.tool_constraint_header')}")
        parts.append(t("skill.tool_constraint_desc"))
        for tool in allowed_tools:
            parts.append(f"- {tool}")

    return "\n".join(parts)


# ── Internal helpers ─────────────────────────────────────


def _skill_name_to_tool_name(skill_name: str) -> str | None:
    """Map skill name to tool module name if it is an external tool skill.

    e.g. gmail-tool -> gmail, image-gen-tool -> image_gen.
    """
    if not skill_name.endswith("-tool"):
        return None
    base = skill_name[:-5]
    return base.replace("-", "_") if base else None


def _load_permitted_tools(anima_dir: Path) -> set[str]:
    """Load permitted tool/action set from permissions config."""
    try:
        from core.config.models import load_permissions
        from core.tooling.permissions import get_permitted_tools

        config = load_permissions(anima_dir)
        return get_permitted_tools(config)
    except Exception:
        logger.debug("Failed to load permissions", exc_info=True)
        return set()


def _filter_guide_for_tool(
    content: str,
    tool_name: str,
    permitted: set[str],
) -> str:
    """Filter gated action lines from content for the given tool."""
    from core.tooling.guide import filter_gated_from_guide

    return filter_gated_from_guide(content, tool_name, permitted)


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

    # 1. Personal skills: {name}/SKILL.md
    candidate = skills_dir / name / "SKILL.md"
    if candidate.is_file():
        return candidate, t("skill.type_personal")

    # 2. Common skills: {name}/SKILL.md
    candidate = common_skills_dir / name / "SKILL.md"
    if candidate.is_file():
        return candidate, t("skill.type_common")

    # 3. Procedures (flat file, unchanged)
    candidate = procedures_dir / f"{name}.md"
    if candidate.is_file():
        return candidate, t("skill.type_procedure")

    return None, ""


def _list_available_names(
    skills_dir: Path,
    common_skills_dir: Path,
    procedures_dir: Path,
) -> list[str]:
    """List all available skill/procedure names."""
    names: list[str] = []
    for d in (skills_dir, common_skills_dir):
        if d.is_dir():
            names.extend(f.parent.name for f in sorted(d.glob("*/SKILL.md")))
    if procedures_dir.is_dir():
        names.extend(f.stem for f in sorted(procedures_dir.glob("*.md")))
    return names


def _strip_frontmatter(text: str) -> tuple[str, dict[str, Any]]:
    """Strip YAML frontmatter and return (content, parsed_frontmatter).

    Delegates to the canonical line-based parser to avoid false splits
    when YAML values contain ``---``.
    """
    from core.memory.frontmatter import parse_frontmatter

    meta, body = parse_frontmatter(text)
    return body, meta
