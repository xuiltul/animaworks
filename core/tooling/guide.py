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


def build_tools_guide(
    tool_registry: list[str],
    personal_tools: dict[str, str] | None = None,
) -> str:
    """Build a markdown CLI guide for allowed tools."""
    if not tool_registry and not personal_tools:
        return ""

    from core.tools import TOOL_MODULES
    from core.tools._base import auto_cli_guide, load_execution_profiles

    # Load execution profiles for background task annotations
    profiles = load_execution_profiles(TOOL_MODULES, personal_tools)

    parts: list[str] = [
        "## 外部ツール",
        "",
        "以下の外部ツールが `animaworks-tool` コマンド経由で使えます。",
        "Bashツールから実行してください。出力はJSON形式（`-j` オプション）を推奨します。",
        "",
    ]

    for tool_name in sorted(tool_registry):
        if tool_name not in TOOL_MODULES:
            continue
        guide = _guide_from_module_path(tool_name, TOOL_MODULES[tool_name])
        if guide:
            # Add background task warning if any subcommand is eligible
            profile = profiles.get(tool_name, {})
            bg_warning = _build_background_warning(tool_name, profile)
            if bg_warning:
                parts.append(bg_warning)
            parts.append(guide)
            parts.append("")

    if personal_tools:
        for tool_name in sorted(personal_tools):
            guide = _guide_from_file(tool_name, personal_tools[tool_name])
            if guide:
                profile = profiles.get(tool_name, {})
                bg_warning = _build_background_warning(tool_name, profile)
                if bg_warning:
                    parts.append(bg_warning)
                parts.append(guide)
                parts.append("")

    # Check if any tool has background-eligible subcommands
    has_bg_tools = any(
        any(info.get("background_eligible") for info in subcmds.values())
        for subcmds in profiles.values()
    )

    parts.extend([
        "### 注意事項",
        "- 使えるツールは上記のみ（permissions.mdで許可されたもの）",
        "- APIキーが未設定のツールはエラーになる。エラー内容を確認して報告すること",
    ])

    if has_bg_tools:
        parts.append(
            "- ⚠ マークのある長時間ツールは `animaworks-tool submit <tool> <args...>` で"
            "非同期実行すること。直接実行するとロックが保持され、"
            "メッセージ受信やheartbeatが停止する"
        )
        parts.append(
            "- submit したタスクの結果は state/background_notifications/ に通知される。"
            "次回の heartbeat で確認すること"
        )

    parts.append(
        "- 検索結果やメッセージ一覧は記憶に保存すべきか判断すること"
    )

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


# ── Helpers ───────────────────────────────────────────────────


def _build_background_warning(
    tool_name: str,
    profile: dict[str, dict[str, object]],
) -> str:
    """Build a warning block for tools with background-eligible subcommands."""
    if not profile:
        return ""

    bg_subcmds = {
        name: info
        for name, info in profile.items()
        if info.get("background_eligible")
    }

    if not bg_subcmds:
        return ""

    max_seconds = max(
        int(info.get("expected_seconds", 60)) for info in bg_subcmds.values()
    )

    if max_seconds >= 600:
        time_desc = f"最大{max_seconds // 60}分"
    else:
        time_desc = f"最大{max_seconds}秒"

    subcmd_list = ", ".join(sorted(bg_subcmds.keys()))

    lines = [
        f"⚠ **長時間ツール** ({tool_name}): "
        f"以下のサブコマンドは実行に{time_desc}かかります。",
        f"`animaworks-tool submit {tool_name} <subcommand> ...` で非同期実行してください。",
        f"結果は次回のheartbeatで通知されます。",
        f"対象サブコマンド: {subcmd_list}",
        "",
    ]
    return "\n".join(lines)


def _guide_from_module_path(tool_name: str, module_path: str) -> str | None:
    """Generate a CLI guide from a package-importable module."""
    from core.tools._base import auto_cli_guide

    try:
        mod = importlib.import_module(module_path)
        return _extract_guide(tool_name, mod)
    except Exception:
        logger.debug("Failed to generate guide for %s", tool_name, exc_info=True)
        return None


def _guide_from_file(tool_name: str, file_path: str) -> str | None:
    """Generate a CLI guide from a file-based personal tool module."""
    from core.tools._base import auto_cli_guide

    try:
        mod = _import_file(tool_name, file_path)
        return _extract_guide(tool_name, mod)
    except Exception:
        logger.debug(
            "Failed to generate guide for personal tool %s",
            tool_name, exc_info=True,
        )
        return None


def _extract_guide(tool_name: str, mod: Any) -> str | None:
    """Extract CLI guide from a loaded module (hand-crafted or auto)."""
    from core.tools._base import auto_cli_guide

    if hasattr(mod, "get_cli_guide"):
        return mod.get_cli_guide()
    if hasattr(mod, "get_tool_schemas"):
        schemas = mod.get_tool_schemas()
        return auto_cli_guide(tool_name, schemas)
    return None


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
