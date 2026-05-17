from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Local filesystem source adapter for Skill Hub imports."""

import shutil
from pathlib import Path

from core.skills.guard import MAX_FILES_PER_SKILL, MAX_SKILL_DIR_SIZE, MAX_SKILL_FILE_SIZE


def stage_local_source(source: str, staging_root: Path) -> Path:
    """Copy a local skill file or directory into *staging_root*."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Skill source not found: {source_path}")

    staged = staging_root / "skill"
    if source_path.is_file():
        if source_path.name != "SKILL.md":
            raise ValueError("Local file sources must be named SKILL.md")
        _validate_file_size(source_path)
        staged.mkdir(parents=True)
        shutil.copy2(source_path, staged / "SKILL.md")
        return staged

    skill_md = source_path / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"Skill directory must contain SKILL.md: {source_path}")
    _validate_skill_tree(source_path)
    _copy_skill_tree(source_path, staged)
    return staged


def _validate_skill_tree(source_dir: Path) -> None:
    source_root = source_dir.resolve()
    total_size = 0
    file_count = 0
    for item in source_root.rglob("*"):
        rel = item.relative_to(source_root)
        if item.is_symlink() or item.resolve().is_relative_to(source_root) is False:
            raise ValueError(f"Unsafe skill bundle entry: {rel}")
        if not item.is_file():
            continue
        file_count += 1
        if file_count > MAX_FILES_PER_SKILL:
            raise ValueError(f"Skill bundle file count exceeds limit ({MAX_FILES_PER_SKILL})")
        size = _validate_file_size(item)
        total_size += size
        if total_size > MAX_SKILL_DIR_SIZE:
            raise ValueError(f"Skill bundle total size exceeds limit ({MAX_SKILL_DIR_SIZE})")


def _validate_file_size(path: Path) -> int:
    size = path.stat().st_size
    if size > MAX_SKILL_FILE_SIZE:
        raise ValueError(f"Skill bundle file exceeds limit ({MAX_SKILL_FILE_SIZE}): {path.name}")
    return size


def _copy_skill_tree(source_dir: Path, target_dir: Path) -> None:
    source_root = source_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=False)
    for item in source_root.rglob("*"):
        rel = item.relative_to(source_root)
        if item.is_symlink() or item.resolve().is_relative_to(source_root) is False:
            raise ValueError(f"Unsafe skill bundle entry: {rel}")
        target = target_dir / rel
        if item.is_dir():
            target.mkdir(exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
