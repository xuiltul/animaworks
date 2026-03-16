from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""External tool dispatcher — unified dispatch convention.

All tool modules (core, common, personal) follow the same dispatch
convention: either a ``dispatch(name, args)`` function or individual
functions matching schema names.  The old ``_DISPATCH_TABLE`` has been
removed in favour of module-level ``dispatch()`` functions.
"""

import json
import logging
from pathlib import Path
from typing import Any

from core.exceptions import ToolExecutionError  # noqa: F401
from core.i18n import t

logger = logging.getLogger("animaworks.external_tools")


# ── ExternalToolDispatcher ───────────────────────────────────


class ExternalToolDispatcher:
    """Dispatch tool calls to external tool modules.

    Uses a unified dispatch convention for all tool sources:
    core (importable modules), common and personal (file-based modules).
    """

    def __init__(
        self,
        tool_registry: list[str],
        personal_tools: dict[str, str] | None = None,
    ) -> None:
        self._registry = tool_registry
        self._personal_tools = personal_tools or {}

    @property
    def registry(self) -> list[str]:
        """Currently registered core tool names."""
        return self._registry

    def update_personal_tools(self, personal_tools: dict[str, str]) -> None:
        """Hot-reload: replace the personal/common tools mapping."""
        self._personal_tools = personal_tools

    def _check_gated(self, name: str, args: dict[str, Any]) -> str | None:
        """Check if a schema name is gated and not permitted.

        Parses tool_name and action from schema name (e.g. gmail_send),
        reads permissions.md from anima_dir, and returns an error string
        if the action is gated and not explicitly permitted.

        Returns:
            Error string if blocked, None if allowed or anima_dir missing.
        """
        anima_dir = args.get("anima_dir")
        if not anima_dir:
            return None

        tool_name, action = self._split_schema_name(name)
        if tool_name is None or action is None:
            return None

        try:
            from core.config.models import load_permissions
            from core.tooling.permissions import get_permitted_tools, is_action_gated

            perm_config = load_permissions(Path(anima_dir))
            permitted = get_permitted_tools(perm_config)
            if is_action_gated(tool_name, action, permitted):
                return json.dumps(
                    {
                        "status": "error",
                        "error_type": "PermissionDenied",
                        "message": t("tooling.gated_action_denied", tool=tool_name, action=action),
                    },
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.debug("Gated check failed for %s: %s", name, e)
        return None

    def _split_schema_name(self, name: str) -> tuple[str | None, str | None]:
        """Split schema name into tool_name and action.

        Convention: name is {tool}_{action}. Matches against registry
        and personal tools, preferring longest match (e.g. image_gen over image).
        """
        from core.tools import TOOL_MODULES

        all_tools = set(self._registry) | set(self._personal_tools.keys()) | set(TOOL_MODULES.keys())
        best_tool: str | None = None
        best_len = 0
        for tool in all_tools:
            prefix = f"{tool}_"
            if name.startswith(prefix) and len(tool) > best_len:
                best_tool = tool
                best_len = len(tool)
        if best_tool is None:
            return None, None
        action = name[len(best_tool) + 1 :]
        return best_tool, action if action else None

    def dispatch(self, name: str, args: dict[str, Any]) -> str | None:
        """Execute a tool by schema name.

        Tries core tools first (from TOOL_MODULES), then file-based tools
        (common + personal).  Returns None if no matching tool found.
        """
        err = self._check_gated(name, args)
        if err is not None:
            return err

        result = self._dispatch_from_registry(name, args)
        if result is not None:
            return result
        result = self._dispatch_from_files(name, args)
        if result is not None:
            return result
        return None

    # ── Core tools (importable modules) ──────────────────────

    def _dispatch_from_registry(self, name: str, args: dict[str, Any]) -> str | None:
        """Dispatch to core tool modules registered in TOOL_MODULES."""
        if not self._registry:
            return None

        import importlib

        from core.tools import TOOL_MODULES

        for tool_name, module_path in TOOL_MODULES.items():
            if tool_name not in self._registry:
                continue
            try:
                mod = importlib.import_module(module_path)
                schemas = mod.get_tool_schemas() if hasattr(mod, "get_tool_schemas") else []
                if name not in [s["name"] for s in schemas]:
                    continue
                return self._call_module(mod, name, args)
            except Exception as e:
                logger.warning("External tool %s failed: %s", name, e)
                return f"Error executing {name}: {e}"

        return None

    # ── File-based tools (common + personal) ─────────────────

    def _dispatch_from_files(self, name: str, args: dict[str, Any]) -> str | None:
        """Dispatch to file-based tool modules (common and personal)."""
        if not self._personal_tools:
            return None

        import importlib.util

        for tool_name, file_path in self._personal_tools.items():
            try:
                spec = importlib.util.spec_from_file_location(
                    f"animaworks_tool_{tool_name}",
                    file_path,
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]

                schemas = mod.get_tool_schemas() if hasattr(mod, "get_tool_schemas") else []
                if name not in [s["name"] for s in schemas]:
                    continue
                return self._call_module(mod, name, args)
            except Exception as e:
                logger.warning("Tool %s (%s) failed: %s", tool_name, name, e)
                return f"Error executing {name}: {e}"

        return None

    # ── Unified module call ──────────────────────────────────

    @staticmethod
    def _call_module(mod: Any, name: str, args: dict[str, Any]) -> str:
        """Call a tool module using the unified dispatch convention.

        Tries, in order:
        1. ``mod.dispatch(name, args)`` — recommended for multi-schema modules
        2. ``getattr(mod, name)(**args)`` — for single-schema modules
        """
        try:
            if hasattr(mod, "dispatch"):
                result = mod.dispatch(name, args)
            elif hasattr(mod, name):
                result = getattr(mod, name)(**args)
            else:
                logger.warning(
                    "Module has schema '%s' but no dispatch or matching function",
                    name,
                )
                return f"Error: no handler for '{name}'"

            if isinstance(result, (dict, list)):
                return json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            return str(result) if result is not None else "(no output)"
        except Exception as e:
            logger.warning("Tool %s failed: %s", name, e)
            return f"Error executing {name}: {e}"
