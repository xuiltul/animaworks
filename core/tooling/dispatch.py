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
from typing import Any

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

    def dispatch(self, name: str, args: dict[str, Any]) -> str | None:
        """Execute a tool by schema name.

        Tries core tools first (from TOOL_MODULES), then file-based tools
        (common + personal).  Returns None if no matching tool found.
        """
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
                schemas = (
                    mod.get_tool_schemas()
                    if hasattr(mod, "get_tool_schemas")
                    else []
                )
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
                    f"animaworks_tool_{tool_name}", file_path,
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]

                schemas = (
                    mod.get_tool_schemas()
                    if hasattr(mod, "get_tool_schemas")
                    else []
                )
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
                    result, ensure_ascii=False, indent=2, default=str,
                )
            return str(result) if result is not None else "(no output)"
        except Exception as e:
            logger.warning("Tool %s failed: %s", name, e)
            return f"Error executing {name}: {e}"
