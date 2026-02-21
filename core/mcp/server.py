from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Stdio MCP server exposing AnimaWorks tools for A1 mode.

Launched as ``python -m core.mcp.server``.  Receives configuration via
environment variables:

- ``ANIMAWORKS_ANIMA_DIR`` -- path to the running anima's data directory
- ``ANIMAWORKS_PROJECT_DIR`` -- path to the AnimaWorks project root

The server name is ``aw`` so tools appear as ``mcp__aw__send_message`` etc.
in the Claude Agent SDK tool namespace.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ── Logging (stderr only — stdout is MCP JSON-RPC) ──────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── MCP server instance ─────────────────────────────────
server = Server("aw")

# ── Tool selection ───────────────────────────────────────
#
# The 12 tools to expose, drawn from canonical schema lists in
# ``core/tooling/schemas.py``.  We pick them by name to build a
# stable, curated subset suitable for A1 mode.

_EXPOSED_TOOL_NAMES: frozenset[str] = frozenset({
    "send_message",
    "post_channel",
    "read_channel",
    "read_dm_history",
    "add_task",
    "update_task",
    "list_tasks",
    "call_human",
    "search_memory",
    "report_procedure_outcome",
    "report_knowledge_outcome",
    "discover_tools",
})


def _build_mcp_tools() -> list[Tool]:
    """Convert canonical AnimaWorks schemas to MCP Tool objects.

    Reads all relevant schema lists from ``core.tooling.schemas`` and
    filters to the 12 exposed tools.
    """
    from core.tooling.schemas import (
        CHANNEL_TOOLS,
        DISCOVERY_TOOLS,
        KNOWLEDGE_TOOLS,
        MEMORY_TOOLS,
        NOTIFICATION_TOOLS,
        PROCEDURE_TOOLS,
        TASK_TOOLS,
    )

    all_schemas: list[dict[str, Any]] = [
        *MEMORY_TOOLS,
        *CHANNEL_TOOLS,
        *TASK_TOOLS,
        *NOTIFICATION_TOOLS,
        *PROCEDURE_TOOLS,
        *KNOWLEDGE_TOOLS,
        *DISCOVERY_TOOLS,
    ]

    tools: list[Tool] = []
    for schema in all_schemas:
        name = schema["name"]
        if name not in _EXPOSED_TOOL_NAMES:
            continue
        tools.append(
            Tool(
                name=name,
                description=schema.get("description", ""),
                inputSchema=schema.get("parameters", {"type": "object", "properties": {}}),
            )
        )

    # Verify we found all 12
    found = {t.name for t in tools}
    missing = _EXPOSED_TOOL_NAMES - found
    if missing:
        logger.warning("MCP tool schemas missing for: %s", ", ".join(sorted(missing)))

    return tools


# Build once at import time (schemas are static dicts, no I/O needed).
MCP_TOOLS: list[Tool] = _build_mcp_tools()

# ── Lazy ToolHandler initialisation ──────────────────────

_tool_handler: Any = None  # core.tooling.handler.ToolHandler | None
_init_error: str | None = None


def _get_tool_handler() -> Any:
    """Return the singleton ToolHandler, initialising on first call.

    Lazy initialisation keeps MCP server startup fast.  If initialisation
    fails the error is cached so subsequent calls return immediately.
    """
    global _tool_handler, _init_error

    if _tool_handler is not None:
        return _tool_handler
    if _init_error is not None:
        return None

    try:
        anima_dir_env = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
        if not anima_dir_env:
            _init_error = "ANIMAWORKS_ANIMA_DIR environment variable is not set"
            logger.error(_init_error)
            return None

        anima_dir = Path(anima_dir_env).resolve()
        if not anima_dir.is_dir():
            _init_error = f"ANIMAWORKS_ANIMA_DIR does not exist: {anima_dir}"
            logger.error(_init_error)
            return None

        # ── MemoryManager ──
        from core.memory import MemoryManager

        memory = MemoryManager(anima_dir)

        # ── Messenger ──
        from core.paths import get_shared_dir
        from core.messenger import Messenger

        shared_dir = get_shared_dir()
        messenger = Messenger(shared_dir=shared_dir, anima_name=anima_dir.name)

        # ── HumanNotifier (optional) ──
        human_notifier = None
        try:
            from core.config.models import load_config
            from core.notification.notifier import HumanNotifier

            config = load_config()
            human_notifier = HumanNotifier.from_config(config.human_notification)
            if human_notifier.channel_count == 0:
                human_notifier = None
        except Exception:
            logger.debug("HumanNotifier init skipped", exc_info=True)

        # ── Tool registry and personal tools (filesystem discovery) ──
        tool_registry: list[str] = []
        personal_tools: dict[str, str] = {}
        try:
            from core.tools import (
                TOOL_MODULES,
                discover_common_tools,
                discover_personal_tools,
            )

            tool_registry = sorted(TOOL_MODULES.keys())
            common = discover_common_tools()
            personal = discover_personal_tools(anima_dir)
            personal_tools = {**common, **personal}
        except Exception:
            logger.debug("Tool discovery failed", exc_info=True)

        # ── ToolHandler ──
        from core.tooling.handler import ToolHandler

        _tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=tool_registry,
            personal_tools=personal_tools,
            on_message_sent=None,
            on_schedule_changed=None,
            human_notifier=human_notifier,
            background_manager=None,
        )

        logger.info(
            "ToolHandler initialised for anima '%s' (%s)",
            anima_dir.name,
            anima_dir,
        )
        return _tool_handler

    except Exception as exc:
        _init_error = f"ToolHandler initialisation failed: {exc}"
        logger.exception(_init_error)
        return None


# ── MCP handlers ─────────────────────────────────────────


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the static list of exposed AnimaWorks tools."""
    return MCP_TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    """Dispatch a tool call to ToolHandler.handle() in a thread.

    ToolHandler.handle() is synchronous and may perform blocking I/O
    (file reads, subprocess calls), so we run it via ``asyncio.to_thread``
    to keep the MCP event loop responsive.
    """
    # Defense-in-depth: reject tool names not in our exposed set.
    # The Agent SDK should only call tools from list_tools(), but
    # ToolHandler.handle() would fall through to external dispatch
    # for unrecognised names, so we gate here explicitly.
    if name not in _EXPOSED_TOOL_NAMES:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "error",
                        "error_type": "ToolNotFound",
                        "message": f"Tool '{name}' is not exposed via MCP",
                    },
                    ensure_ascii=False,
                ),
            )
        ]

    handler = _get_tool_handler()
    if handler is None:
        error_msg = _init_error or "ToolHandler is not available"
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"status": "error", "error_type": "InitError", "message": error_msg},
                    ensure_ascii=False,
                ),
            )
        ]

    try:
        result = await asyncio.to_thread(handler.handle, name, arguments or {})
        return [TextContent(type="text", text=result)]
    except Exception as exc:
        logger.exception("Unhandled error calling tool '%s'", name)
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "error",
                        "error_type": "UnhandledError",
                        "message": f"Tool execution failed: {name}: {exc}",
                    },
                    ensure_ascii=False,
                ),
            )
        ]


# ── Entry point ──────────────────────────────────────────


async def main() -> None:
    """Run the MCP stdio server."""
    logger.info("AnimaWorks MCP server starting (name=aw)")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
