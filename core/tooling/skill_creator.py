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

import yaml

from core.i18n import t

logger = logging.getLogger("animaworks.skill_creator")


def _validate_filename(fname: str, parent_dir: Path) -> bool:
    """Validate filename to prevent path traversal."""
    if not fname or "/" in fname or "\\" in fname or ".." in fname:
        return False
    resolved = (parent_dir / fname).resolve()
    return resolved.is_relative_to(parent_dir.resolve())


def create_skill_directory(
    skill_name: str,
    description: str,
    body: str,
    base_dir: Path,
    *,
    references: list[dict[str, str]] | None = None,
    templates: list[dict[str, str]] | None = None,
    allowed_tools: list[str] | None = None,
    trust_level: str | None = None,
    source_type: str | None = None,
    source_origin: str | None = None,
    source_owner_anima: str | None = None,
    category: str | None = None,
    promotion_status: str | None = None,
    skill_policy: dict[str, str] | None = None,
    use_when: list[str] | None = None,
    trigger_phrases: list[str] | None = None,
    negative_phrases: list[str] | None = None,
    domains: list[str] | None = None,
    routing_examples: list[str] | None = None,
    trusted_by: str | None = None,
    trusted_at: str | None = None,
    trust_reason: str | None = None,
) -> str:
    """Create skill directory structure with SKILL.md and optional sub-files."""
    if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
        return t("skill_creator.invalid_name", skill_name=skill_name)

    skill_dir = base_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    fm: dict[str, Any] = {"name": skill_name, "description": description}
    if category:
        fm["category"] = category
    fm["trust_level"] = trust_level or "trusted"
    source: dict[str, str] = {"type": source_type or "anima"}
    if source_owner_anima:
        source["owner_anima"] = source_owner_anima
    source["origin"] = source_origin or "manual"
    fm["source"] = source
    fm["version"] = 1
    if promotion_status:
        fm["promotion_status"] = promotion_status
    if skill_policy:
        fm["skill_policy"] = skill_policy
    if allowed_tools:
        fm["allowed_tools"] = allowed_tools
    if use_when:
        fm["use_when"] = use_when
    if trigger_phrases:
        fm["trigger_phrases"] = trigger_phrases
    if negative_phrases:
        fm["negative_phrases"] = negative_phrases
    if domains:
        fm["domains"] = domains
    if routing_examples:
        fm["routing_examples"] = routing_examples
    if trusted_by is not None:
        fm["trusted_by"] = trusted_by
    if trusted_at is not None:
        fm["trusted_at"] = trusted_at
    if trust_reason is not None:
        fm["trust_reason"] = trust_reason
    frontmatter = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()

    skill_md = f"---\n{frontmatter}\n---\n\n{body}\n"
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    created_files = ["SKILL.md"]

    if references:
        ref_dir = skill_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        for ref in references:
            fname = ref.get("filename", "")
            content = ref.get("content", "")
            if fname and _validate_filename(fname, ref_dir):
                (ref_dir / fname).write_text(content, encoding="utf-8")
                created_files.append(f"references/{fname}")

    if templates:
        tpl_dir = skill_dir / "templates"
        tpl_dir.mkdir(exist_ok=True)
        for tpl in templates:
            fname = tpl.get("filename", "")
            content = tpl.get("content", "")
            if fname and _validate_filename(fname, tpl_dir):
                (tpl_dir / fname).write_text(content, encoding="utf-8")
                created_files.append(f"templates/{fname}")

    files_str = ", ".join(created_files)
    return t("skill_creator.created", skill_name=skill_name, skill_dir=skill_dir, files_str=files_str)
