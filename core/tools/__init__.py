# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks external tools package."""
from __future__ import annotations
import logging
import sys
from pathlib import Path

logger = logging.getLogger("animaworks.tools")


def discover_core_tools() -> dict[str, str]:
    """Scan core/tools/ for tool modules.

    Returns: Mapping of tool_name → module path (e.g., "core.tools.web_search").
    Skips files starting with _ (private/internal modules).
    """
    tools_dir = Path(__file__).parent
    core: dict[str, str] = {}
    for f in sorted(tools_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        tool_name = f.stem
        core[tool_name] = f"core.tools.{tool_name}"
    return core


# Backward-compatible module-level variable
TOOL_MODULES = discover_core_tools()


def discover_common_tools(data_dir: Path | None = None) -> dict[str, str]:
    """Scan ~/.animaworks/common_tools/ for shared tool modules.

    Returns: Mapping of tool_name → absolute file path.
    """
    if data_dir is None:
        from core.paths import get_data_dir
        data_dir = get_data_dir()
    tools_dir = data_dir / "common_tools"
    if not tools_dir.is_dir():
        return {}
    common: dict[str, str] = {}
    for f in sorted(tools_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        tool_name = f.stem
        if tool_name in TOOL_MODULES:
            logger.warning(
                "Common tool '%s' shadows core tool — skipped", tool_name,
            )
            continue
        common[tool_name] = str(f)
    if common:
        logger.info("Discovered common tools: %s", list(common.keys()))
    return common


def discover_personal_tools(anima_dir: Path) -> dict[str, str]:
    """Scan ``{anima_dir}/tools/`` for personal tool modules.

    Returns:
        Mapping of tool_name → absolute file path.
        Skips files starting with ``_`` (including ``__init__.py``).
    """
    tools_dir = anima_dir / "tools"
    if not tools_dir.is_dir():
        return {}
    personal: dict[str, str] = {}
    for f in sorted(tools_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        tool_name = f.stem
        if tool_name in TOOL_MODULES:
            logger.warning(
                "Personal tool '%s' shadows core tool — skipped", tool_name,
            )
            continue
        personal[tool_name] = str(f)
    if personal:
        logger.info("Discovered personal tools: %s", list(personal.keys()))
    return personal


_SUBMIT_TASK_ID_LENGTH = 12


def _handle_submit(argv: list[str]) -> None:
    """Handle ``animaworks-tool submit <tool> <args...>``.

    Writes a pending task descriptor to ``state/background_tasks/pending/``
    and exits immediately.  The runner's pending watcher will pick it up.
    """
    import json
    import os
    import time
    import uuid

    if len(argv) < 1:
        print("Usage: animaworks-tool submit <tool_name> [args...]")
        print("Submits a long-running tool for background execution.")
        print("Results are delivered to your inbox on completion.")
        sys.exit(1)

    tool_name = argv[0]
    tool_args = argv[1:]

    anima_dir = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
    if not anima_dir:
        print(
            "Error: ANIMAWORKS_ANIMA_DIR not set. "
            "Cannot determine pending directory.",
        )
        sys.exit(1)

    anima_dir_path = Path(anima_dir)
    anima_name = anima_dir_path.name

    # Generate task ID
    task_id = uuid.uuid4().hex[:_SUBMIT_TASK_ID_LENGTH]

    # Determine subcommand (first non-flag argument after tool_name)
    subcommand = ""
    for arg in tool_args:
        if not arg.startswith("-"):
            subcommand = arg
            break

    # Optional: check EXECUTION_PROFILE for warning
    try:
        if tool_name in TOOL_MODULES:
            import importlib

            mod = importlib.import_module(TOOL_MODULES[tool_name])
            profile = getattr(mod, "EXECUTION_PROFILE", None)
            if profile and subcommand and subcommand in profile:
                info = profile[subcommand]
                if not info.get("background_eligible"):
                    print(
                        f"Warning: {tool_name} {subcommand} is not marked "
                        f"as long-running "
                        f"(expected ~{info.get('expected_seconds', '?')}s). "
                        f"Consider running directly instead.",
                        file=sys.stderr,
                    )
    except Exception:
        logger.debug("Profile check failed for %s %s", tool_name, subcommand, exc_info=True)

    # Write pending task descriptor
    pending_dir = anima_dir_path / "state" / "background_tasks" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    task_desc = {
        "task_id": task_id,
        "tool_name": tool_name,
        "subcommand": subcommand,
        "raw_args": tool_args,
        "anima_name": anima_name,
        "anima_dir": str(anima_dir_path),
        "submitted_at": time.time(),
        "status": "pending",
    }

    task_path = pending_dir / f"{task_id}.json"
    task_path.write_text(
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Output result
    result = {
        "task_id": task_id,
        "status": "submitted",
        "tool": tool_name,
        "subcommand": subcommand,
        "message": (
            f"バックグラウンドタスクを投入しました。"
            f"完了時にinboxに通知されます。(task_id: {task_id})"
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cli_dispatch():
    """Entry point for ``animaworks-tool`` CLI command.

    Supports core tools (from ``TOOL_MODULES``), common tools
    (from ``common_tools/``), and personal tools discovered via
    the ``ANIMAWORKS_ANIMA_DIR`` environment variable.
    """
    import os

    # Discover common tools
    common = discover_common_tools()

    # Discover personal tools if anima_dir is set
    anima_dir_str = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
    personal: dict[str, str] = {}
    if anima_dir_str:
        personal = discover_personal_tools(Path(anima_dir_str))

    all_tools = set(TOOL_MODULES.keys()) | set(common.keys()) | set(personal.keys())

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        tools = ", ".join(sorted(all_tools))
        print(f"Usage: animaworks-tool <tool_name> [args...]")
        print(f"Available tools: {tools}")
        sys.exit(0 if "--help" in sys.argv else 1)

    tool_name = sys.argv[1]

    # Handle submit subcommand — write pending task and exit immediately
    if tool_name == "submit":
        _handle_submit(sys.argv[2:])
        return

    # Try core tools first
    if tool_name in TOOL_MODULES:
        import importlib
        mod = importlib.import_module(TOOL_MODULES[tool_name])
        if not hasattr(mod, "cli_main"):
            print(f"Tool '{tool_name}' has no CLI interface")
            sys.exit(1)
        mod.cli_main(sys.argv[2:])
        return

    # Try common or personal tools (loaded from file path)
    file_tool = personal.get(tool_name) or common.get(tool_name)
    if file_tool:
        import importlib.util
        origin = "personal" if tool_name in personal else "common"
        spec = importlib.util.spec_from_file_location(
            f"animaworks_{origin}_tool_{tool_name}", file_tool,
        )
        if spec is None or spec.loader is None:
            print(f"Cannot load {origin} tool: {tool_name}")
            sys.exit(1)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if not hasattr(mod, "cli_main"):
            print(f"{origin.capitalize()} tool '{tool_name}' has no CLI interface")
            sys.exit(1)
        mod.cli_main(sys.argv[2:])
        return

    print(f"Unknown tool: {tool_name}")
    print(f"Available: {', '.join(sorted(all_tools))}")
    sys.exit(1)