#!/usr/bin/env python3
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Auto-generate reference sections in common_knowledge documents from code.

Reads Pydantic models and tool schemas from the codebase and injects
formatted tables into AUTO-GENERATED marker sections in common_knowledge
Markdown files.

Usage:
    python scripts/generate_reference.py [--data-dir ~/.animaworks]
    python scripts/generate_reference.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logger = logging.getLogger("animaworks.generate_reference")

# ── Marker pattern ─────────────────────────────────────────

_MARKER_RE = re.compile(
    r"(<!-- AUTO-GENERATED:START (\w+) -->)\n"
    r".*?"
    r"(<!-- AUTO-GENERATED:END -->)",
    re.DOTALL,
)


# ── Generators ─────────────────────────────────────────────


def _generate_tool_parameters() -> str:
    """Generate tool parameter reference from core/tooling/schemas.py."""
    from core.tooling.schemas import MEMORY_TOOLS

    lines: list[str] = ["### ツールパラメータリファレンス（自動生成）", ""]

    for tool in MEMORY_TOOLS:
        name = tool["name"]
        desc = tool["description"]
        params = tool["parameters"]
        props = params.get("properties", {})
        required = set(params.get("required", []))

        lines.append(f"#### `{name}`")
        lines.append(f"")
        lines.append(f"{desc}")
        lines.append(f"")
        lines.append("| パラメータ | 型 | 必須 | 説明 |")
        lines.append("|-----------|-----|------|------|")

        for pname, pschema in props.items():
            ptype = pschema.get("type", "string")
            if "enum" in pschema:
                ptype = " \\| ".join(f'`{v}`' for v in pschema["enum"])
            pdesc = pschema.get("description", "")
            req = "Yes" if pname in required else "No"
            lines.append(f"| `{pname}` | {ptype} | {req} | {pdesc} |")

        lines.append("")

    return "\n".join(lines)


def _generate_config_fields() -> str:
    """Generate config field reference from core/config/models.py."""
    from core.config.models import AnimaWorksConfig, AnimaDefaults, AnimaModelConfig

    lines: list[str] = ["### 設定項目リファレンス（自動生成）", ""]

    for cls, title in [
        (AnimaModelConfig, "Anima設定 (per-anima overrides)"),
        (AnimaDefaults, "デフォルト値 (anima_defaults)"),
    ]:
        lines.append(f"#### {title}")
        lines.append("")
        lines.append("| フィールド | 型 | デフォルト | 説明 |")
        lines.append("|-----------|-----|----------|------|")

        for fname, finfo in cls.model_fields.items():
            ftype = _format_type(finfo.annotation)
            default = finfo.default
            if default is None:
                default_str = "None"
            elif isinstance(default, str):
                default_str = f'`"{default}"`'
            else:
                default_str = f"`{default}`"
            desc = finfo.description or ""
            lines.append(f"| `{fname}` | {ftype} | {default_str} | {desc} |")

        lines.append("")

    # Top-level config sections
    lines.append("#### AnimaWorksConfig トップレベル")
    lines.append("")
    lines.append("| セクション | 説明 |")
    lines.append("|-----------|------|")
    section_descs = {
        "version": "設定ファイルバージョン",
        "setup_complete": "セットアップ完了フラグ",
        "locale": "ロケール設定",
        "system": "システム設定（モード、ログレベル）",
        "credentials": "API認証情報",
        "model_modes": "モデル名→実行モードマッピング",
        "anima_defaults": "Anima設定デフォルト値",
        "animas": "Anima別設定オーバーライド",
        "consolidation": "記憶統合設定",
        "rag": "RAG（検索拡張生成）設定",
        "priming": "プライミング（自動記憶想起）設定",
        "image_gen": "画像生成設定",
    }
    for fname in AnimaWorksConfig.model_fields:
        desc = section_descs.get(fname, "")
        lines.append(f"| `{fname}` | {desc} |")
    lines.append("")

    return "\n".join(lines)


def _generate_cron_fields() -> str:
    """Generate CronTask field reference from core/schemas.py."""
    from core.schemas import CronTask

    lines: list[str] = ["### CronTaskフィールドリファレンス（自動生成）", ""]
    lines.append("| フィールド | 型 | デフォルト | 説明 |")
    lines.append("|-----------|-----|----------|------|")

    field_descs = {
        "name": "タスク名（一意の識別子）",
        "schedule": "cron式スケジュール（例: `*/30 * * * *`）",
        "type": "タスク種別: `llm`（LLM実行）または `command`（コマンド実行）",
        "description": "LLM型: LLMへの指示文",
        "command": "Command型: 実行するBashコマンド",
        "tool": "Command型: 呼び出す内部ツール名",
        "args": "Command型: ツールの引数（JSON形式）",
    }

    for fname, finfo in CronTask.model_fields.items():
        ftype = _format_type(finfo.annotation)
        default = finfo.default
        if default is None:
            default_str = "None"
        elif isinstance(default, str):
            default_str = f'`"{default}"`'
        else:
            default_str = f"`{default}`"
        desc = field_descs.get(fname, finfo.description or "")
        lines.append(f"| `{fname}` | {ftype} | {default_str} | {desc} |")

    lines.append("")
    return "\n".join(lines)


def _format_type(annotation: Any) -> str:
    """Format a type annotation for display."""
    if annotation is None:
        return "Any"
    type_str = str(annotation)
    # Clean up common patterns
    type_str = type_str.replace("typing.", "")
    type_str = type_str.replace("<class '", "").replace("'>", "")
    type_str = type_str.replace("NoneType", "None")
    return f"`{type_str}`"


# ── Injection engine ───────────────────────────────────────

_GENERATORS: dict[str, Any] = {
    "tool_parameters": _generate_tool_parameters,
    "config_fields": _generate_config_fields,
    "cron_fields": _generate_cron_fields,
}


def inject_auto_generated(file_path: Path, *, dry_run: bool = False) -> bool:
    """Inject auto-generated content into a single Markdown file.

    Returns True if the file was modified.
    """
    content = file_path.read_text(encoding="utf-8")
    modified = False

    def _replacer(match: re.Match[str]) -> str:
        nonlocal modified
        start_marker = match.group(1)
        section_name = match.group(2)
        end_marker = match.group(3)

        generator = _GENERATORS.get(section_name)
        if generator is None:
            logger.warning(
                "Unknown AUTO-GENERATED section '%s' in %s",
                section_name, file_path,
            )
            return match.group(0)

        generated = generator()
        modified = True
        return f"{start_marker}\n{generated}\n{end_marker}"

    new_content = _MARKER_RE.sub(_replacer, content)

    if modified and not dry_run:
        file_path.write_text(new_content, encoding="utf-8")
        logger.info("Updated auto-generated sections in %s", file_path)

    return modified


def process_directory(
    common_knowledge_dir: Path,
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Process all Markdown files in common_knowledge directory.

    Returns list of modified file paths.
    """
    modified_files: list[Path] = []

    for md_file in sorted(common_knowledge_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if "AUTO-GENERATED:START" not in content:
            continue

        if inject_auto_generated(md_file, dry_run=dry_run):
            modified_files.append(md_file)
            action = "Would update" if dry_run else "Updated"
            print(f"  {action}: {md_file.relative_to(common_knowledge_dir)}")

    return modified_files


# ── CLI ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate reference sections in common_knowledge documents.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Runtime data directory (default: ~/.animaworks)",
    )
    parser.add_argument(
        "--templates",
        action="store_true",
        help="Process templates/common_knowledge/ instead of runtime directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.templates:
        target_dir = PROJECT_DIR / "templates" / "common_knowledge"
    elif args.data_dir:
        target_dir = args.data_dir / "common_knowledge"
    else:
        from core.paths import get_data_dir
        target_dir = get_data_dir() / "common_knowledge"

    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing: {target_dir}")
    modified = process_directory(target_dir, dry_run=args.dry_run)

    if not modified:
        print("No AUTO-GENERATED sections found or no changes needed.")
    else:
        count = len(modified)
        action = "would be updated" if args.dry_run else "updated"
        print(f"\n{count} file(s) {action}.")


if __name__ == "__main__":
    main()
