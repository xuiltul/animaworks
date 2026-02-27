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
) -> str:
    """Create skill directory structure with SKILL.md and optional sub-files."""
    if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
        return f"無効なスキル名: '{skill_name}'（パス区切り文字は使用不可）"

    skill_dir = base_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    fm: dict[str, Any] = {"name": skill_name, "description": description}
    if allowed_tools:
        fm["allowed_tools"] = allowed_tools
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
    return f"スキル '{skill_name}' を作成しました: {skill_dir}\n作成ファイル: {files_str}"
