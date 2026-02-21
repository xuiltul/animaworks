from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Dynamic tool guide generation for A1 (CLI) and A2 (schema) modes."""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

from core.exceptions import ToolConfigError  # noqa: F401

logger = logging.getLogger("animaworks.tool_guide")


# ── Public API ───────────────────────────────────────────────────


def build_tools_guide(
    tool_registry: list[str],
    personal_tools: dict[str, str] | None = None,
) -> str:
    """Build a compact summary table of allowed external tools.

    Replaces the previous verbose CLI-example format with a concise
    table that saves ~60-80% of system prompt tokens.  Agents can use
    ``animaworks-tool <tool> --help`` or the ``discover_tools`` MCP tool
    for detailed usage.
    """
    if not tool_registry and not personal_tools:
        return ""

    from core.tools import TOOL_MODULES
    from core.tools._base import load_execution_profiles

    profiles = load_execution_profiles(TOOL_MODULES, personal_tools)

    parts: list[str] = [
        "## 外部ツール",
        "",
        "以下の外部ツールが利用可能です。使い方の詳細は `discover_tools` ツールまたは "
        "`animaworks-tool <ツール名> --help` で確認してください。",
        "",
        "| ツール | 概要 | サブコマンド |",
        "|--------|------|------------|",
    ]

    bg_tools: list[str] = []

    for tool_name in sorted(tool_registry):
        if tool_name not in TOOL_MODULES:
            continue
        row = _get_tool_summary(tool_name, TOOL_MODULES[tool_name])
        if row:
            parts.append(row)
        # Track background-eligible subcommands
        profile = profiles.get(tool_name, {})
        bg_subcmds = [name for name, info in profile.items() if info.get("background_eligible")]
        if bg_subcmds:
            bg_tools.append(f"{tool_name} ({', '.join(sorted(bg_subcmds))})")

    if personal_tools:
        for tool_name in sorted(personal_tools):
            row = _get_tool_summary_from_file(tool_name, personal_tools[tool_name])
            if row:
                parts.append(row)
            profile = profiles.get(tool_name, {})
            bg_subcmds = [name for name, info in profile.items() if info.get("background_eligible")]
            if bg_subcmds:
                bg_tools.append(f"{tool_name} ({', '.join(sorted(bg_subcmds))})")

    parts.append("")

    if bg_tools:
        parts.append(f"長時間ツール: {', '.join(bg_tools)}")
        parts.append(
            "`animaworks-tool submit <tool> <subcommand>` で非同期実行すること。"
            "直接実行するとロックが保持される。"
        )
        parts.append("")

    parts.append("使えるツールは上記のみ（permissions.mdで許可されたもの）。APIキー未設定はエラーになる。")

    return "\n".join(parts)


def load_tool_schemas(
    tool_registry: list[str],
    personal_tools: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load structured schemas for A2 mode.

    Delegates to ``schemas.load_all_tool_schemas()`` which handles both
    core and personal tool modules with consistent normalisation.
    """
    from core.tooling.schemas import load_all_tool_schemas

    return load_all_tool_schemas(
        tool_registry=tool_registry,
        personal_tools=personal_tools,
    )


# ── Summary table helpers ────────────────────────────────────────


def _get_tool_summary(tool_name: str, module_path: str) -> str | None:
    """Generate a table row: ``| name | description | subcommands |``."""
    try:
        mod = importlib.import_module(module_path)
        return _build_summary_row(tool_name, mod)
    except Exception:
        logger.debug("Failed to get summary for %s", tool_name, exc_info=True)
        return None


def _get_tool_summary_from_file(tool_name: str, file_path: str) -> str | None:
    """Generate a table row from a file-based tool."""
    try:
        mod = _import_file(tool_name, file_path)
        return _build_summary_row(tool_name, mod)
    except Exception:
        logger.debug("Failed to get summary for personal tool %s", tool_name, exc_info=True)
        return None


def _build_summary_row(tool_name: str, mod: Any) -> str | None:
    """Build a Markdown table row from a loaded module."""
    if not hasattr(mod, "get_tool_schemas"):
        return None
    schemas = mod.get_tool_schemas()
    if not schemas:
        return None
    subcmds = _extract_subcommand_names(tool_name, schemas)
    desc = _get_module_description(mod, tool_name)
    subcmd_str = ", ".join(subcmds[:6])
    if len(subcmds) > 6:
        subcmd_str += ", ..."
    return f"| {tool_name} | {desc} | {subcmd_str} |"


def _extract_subcommand_names(tool_name: str, schemas: list[dict]) -> list[str]:
    """Extract clean subcommand names from tool schemas."""
    names: list[str] = []
    for s in schemas:
        raw = s.get("name", s.get("function", {}).get("name", ""))
        # Remove tool prefix (e.g. "chatwork_send" -> "send")
        clean = raw.replace(f"{tool_name}_", "").replace(f"{tool_name}.", "")
        if clean:
            names.append(clean)
    return names


def _get_module_description(mod: Any, tool_name: str) -> str:
    """Extract a short description from a tool module."""
    # Try explicit TOOL_DESCRIPTION constant
    if hasattr(mod, "TOOL_DESCRIPTION"):
        return str(mod.TOOL_DESCRIPTION)
    # Use first line of module docstring
    if mod.__doc__:
        first_line = mod.__doc__.strip().split("\n")[0]
        # Strip trailing period for table consistency
        first_line = first_line.rstrip(".")
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        return first_line
    return tool_name


# ── Utilities ────────────────────────────────────────────────────


def _import_file(name: str, file_path: str) -> Any:
    """Import a Python module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(
        f"animaworks_personal_tool_{name}", file_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod
