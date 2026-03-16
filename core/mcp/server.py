from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Stdio MCP server exposing AnimaWorks tools for Mode S.

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
# The tools to expose, drawn from canonical schema lists in
# ``core/tooling/schemas.py``.  We pick them by name to build a
# stable, curated subset suitable for Mode S.

_EXPOSED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # AW-essential: memory
        "search_memory",
        "read_memory_file",
        "write_memory_file",
        "archive_memory_file",
        # AW-essential: messaging
        "send_message",
        "post_channel",
        # AW-essential: notification
        "call_human",
        # AW-essential: task management
        "delegate_task",
        "submit_tasks",
        "update_task",
        # AW-essential: skill/CLI manual
        "skill",
    }
)


def _get_supervisor_tool_names() -> frozenset[str]:
    """Supervisor tool names — derived from SUPERVISOR_TOOLS at import time.

    Used by list_tools() to dynamically filter supervisor-only tools.
    """
    from core.tooling.schemas import _supervisor_tools

    return frozenset(t["name"] for t in _supervisor_tools())


_SUPERVISOR_TOOL_NAMES: frozenset[str] = _get_supervisor_tool_names()


# Cached original parameter schemas (before relaxation) for type coercion
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {}


def _relax_integer_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Accept string values for integer parameters in MCP inputSchema.

    LLMs sometimes serialize numbers as strings (e.g. ``"10"`` instead
    of ``10``).  The MCP SDK validates strictly against the inputSchema,
    so we widen ``"type": "integer"`` to ``["integer", "string"]`` to
    let the call through.  Actual coercion happens in ``_coerce_integers``.
    """
    import copy

    schema = copy.deepcopy(schema)
    for _name, prop in schema.get("properties", {}).items():
        if prop.get("type") == "integer":
            prop["type"] = ["integer", "string"]
    return schema


def _coerce_integers(
    arguments: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    """Coerce string-valued integer arguments to actual ints.

    Matches against the canonical schema to find integer fields,
    then converts any string values that look numeric.
    """
    schema = _TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return arguments
    props = schema.get("properties", {})
    for key, prop in props.items():
        if prop.get("type") == "integer" and key in arguments:
            val = arguments[key]
            if isinstance(val, str):
                try:
                    arguments[key] = int(val)
                except (ValueError, TypeError):
                    pass
    return arguments


def _load_permitted_categories(anima_dir: Path) -> set[str]:
    """Load permitted external tool categories from permissions.json."""
    from core.config.models import load_permissions
    from core.tooling.permissions import get_permitted_tools

    config = load_permissions(anima_dir)
    return get_permitted_tools(config)


def _build_mcp_tools() -> tuple[list[Tool], frozenset[str]]:
    """Convert canonical AnimaWorks schemas to MCP Tool objects.

    Reads all relevant schema lists from ``core.tooling.schemas`` and
    filters to the exposed tools.  Mode S uses skill+CLI for external
    tools; ``use_tool`` is Mode B only.

    Returns:
        Tuple of (tool_list, exposed_name_set) where exposed_name_set
        is the set of internal tool names.
    """
    from core.tooling.schemas import (
        KNOWLEDGE_TOOLS,
        MEMORY_TOOLS,
        PROCEDURE_TOOLS,
        SUBMIT_TASKS_TOOLS,
        _background_task_tools,
        _channel_tools,
        _check_permissions_tools,
        _notification_tools,
        _skill_tools,
        _supervisor_tools,
        _task_tools,
        _vault_tools,
    )

    all_schemas: list[dict[str, Any]] = [
        *MEMORY_TOOLS,
        *_channel_tools(),
        *_task_tools(),
        *_notification_tools(),
        *PROCEDURE_TOOLS,
        *KNOWLEDGE_TOOLS,
        *_supervisor_tools(),
        *_skill_tools(),
        *SUBMIT_TASKS_TOOLS,
        *_background_task_tools(),
        *_vault_tools(),
        *_check_permissions_tools(),
    ]

    exposed = _EXPOSED_TOOL_NAMES

    # Apply DB description overrides
    from core.tooling.schemas import apply_db_descriptions

    all_schemas = apply_db_descriptions(all_schemas)

    # Cache original schemas for type coercion lookup
    for schema in all_schemas:
        if schema["name"] in exposed:
            _TOOL_SCHEMAS[schema["name"]] = schema.get("parameters", {})

    # Generate dynamic description for the skill tool
    _skill_description = _build_skill_description()

    tools: list[Tool] = []
    for schema in all_schemas:
        name = schema["name"]
        if name not in exposed:
            continue
        desc = schema.get("description", "")
        # Override skill tool description with dynamic content
        if name == "skill" and _skill_description:
            desc = _skill_description
        input_schema = schema.get("parameters", {"type": "object", "properties": {}})
        input_schema = _relax_integer_types(input_schema)
        tools.append(
            Tool(
                name=name,
                description=desc,
                inputSchema=input_schema,
            )
        )

    # Verify we found all expected internal tools
    found = {t.name for t in tools}
    missing = _EXPOSED_TOOL_NAMES - found
    if missing:
        logger.warning("MCP tool schemas missing for: %s", ", ".join(sorted(missing)))

    return tools, exposed


def _build_skill_description() -> str:
    """Build dynamic skill tool description from available skills.

    Called once at MCP server startup.  Returns empty string on failure
    (the static empty description from SKILL_TOOLS will be used instead).
    """
    try:
        anima_dir_env = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
        if not anima_dir_env:
            return ""

        anima_dir = Path(anima_dir_env).resolve()
        if not anima_dir.is_dir():
            return ""

        from core.memory.skill_metadata import SkillMetadataService
        from core.paths import get_common_skills_dir
        from core.tooling.skill_tool import build_skill_tool_description

        svc = SkillMetadataService(
            skills_dir=anima_dir / "skills",
            common_skills_dir=get_common_skills_dir(),
        )
        skill_metas = svc.list_skill_metas()
        common_metas = svc.list_common_skill_metas()

        # Procedures use the same SkillMetadataService extraction
        procedures_dir = anima_dir / "procedures"
        procedure_metas = []
        if procedures_dir.is_dir():
            procedure_metas = [SkillMetadataService.extract_skill_meta(f) for f in sorted(procedures_dir.glob("*.md"))]

        return build_skill_tool_description(skill_metas, common_metas, procedure_metas)

    except Exception:
        logger.debug("Failed to build skill description for MCP", exc_info=True)
        return ""


# Build once at import time.  DB descriptions are baked in at this point;
# WebUI edits to tool descriptions will not take effect until the MCP
# subprocess is restarted (i.e. the parent Anima process restarts).
MCP_TOOLS: list[Tool]
_EXPOSED_NAMES: frozenset[str]
MCP_TOOLS, _EXPOSED_NAMES = _build_mcp_tools()

# ── BackgroundTaskManager for MCP ────────────────────────


def _build_background_manager(anima_dir: Path) -> Any:
    """Build a BackgroundTaskManager for the MCP server process.

    Mirrors ``AgentCore._build_background_manager()`` but operates
    without the full AgentCore.  Returns None when disabled in config.
    """
    try:
        from core.config.models import load_config

        config = load_config()
        if not config.background_task.enabled:
            return None

        from core.background import BackgroundTaskManager
        from core.tools import TOOL_MODULES
        from core.tools._base import load_execution_profiles

        profiles = load_execution_profiles(TOOL_MODULES)
        config_eligible = {name: tc.threshold_s for name, tc in config.background_task.eligible_tools.items()}

        mgr = BackgroundTaskManager.from_profiles(
            anima_dir=anima_dir,
            anima_name=anima_dir.name,
            profiles=profiles,
            config_eligible=config_eligible or None,
        )

        mgr.on_complete = _make_on_complete_callback(anima_dir)
        logger.info("BackgroundTaskManager initialised for MCP (anima=%s)", anima_dir.name)
        return mgr

    except Exception:
        logger.debug("BackgroundTaskManager init skipped in MCP", exc_info=True)
        return None


def _make_on_complete_callback(anima_dir: Path) -> Any:
    """Create an on_complete callback that writes notification files.

    In the MCP subprocess we don't have access to WebSocket or
    HumanNotifier, so we only write a notification file under
    ``state/background_notifications/`` for the next heartbeat to pick up.
    """
    from core.i18n import t as _t

    async def _on_complete(task: Any) -> None:
        try:
            subject = _t("anima.bg_task_done", tool=task.tool_name)
            if task.status.value == "failed":
                subject = _t("anima.bg_task_failed", tool=task.tool_name)

            notif_dir = anima_dir / "state" / "background_notifications"
            notif_dir.mkdir(parents=True, exist_ok=True)
            notif_path = notif_dir / f"{task.task_id}.md"
            notif_content = (
                f"# {subject}\n\n"
                f"- task_id: {task.task_id}\n"
                f"- tool: {task.tool_name}\n"
                f"- status: {task.status.value}\n"
                f"- result: {task.summary()}\n"
            )
            notif_path.write_text(notif_content, encoding="utf-8")
            logger.info(
                "MCP bg task notification written: %s (tool=%s, status=%s)",
                task.task_id,
                task.tool_name,
                task.status.value,
            )
        except Exception:
            logger.exception(
                "Failed to write MCP bg task notification for %s",
                task.task_id,
            )

    return _on_complete


# ── Lazy ToolHandler initialisation ──────────────────────

_tool_handler: Any = None  # core.tooling.handler.ToolHandler | None
_init_error: str | None = None
_is_supervisor: bool | None = None


def _has_subordinates_for_anima() -> bool:
    """Check if this Anima has subordinates via config.json.

    Evaluated once at first call and cached. Falls back to False (safe side —
    hides supervisor tools when check fails).
    """
    global _is_supervisor
    if _is_supervisor is not None:
        return _is_supervisor

    try:
        anima_dir_env = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
        if not anima_dir_env:
            _is_supervisor = False
            return False
        anima_name = Path(anima_dir_env).name

        from core.paths import get_data_dir

        config_path = get_data_dir() / "config.json"
        if not config_path.is_file():
            _is_supervisor = False
            return False

        import json as _json

        config_data = _json.loads(config_path.read_text(encoding="utf-8"))
        animas = config_data.get("animas", {})

        for other_name, other_cfg in animas.items():
            if other_name == anima_name:
                continue
            if isinstance(other_cfg, dict) and other_cfg.get("supervisor") == anima_name:
                _is_supervisor = True
                return True

        _is_supervisor = False
        return False
    except Exception:
        logger.debug("Failed to check subordinate status, defaulting to False")
        _is_supervisor = False
        return False


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
        from core.messenger import Messenger
        from core.paths import get_shared_dir

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
                discover_common_tools,
                discover_personal_tools,
            )

            permitted = _load_permitted_categories(anima_dir)
            tool_registry = sorted(permitted)
            common = discover_common_tools()
            personal = discover_personal_tools(anima_dir)
            personal_tools = {**common, **personal}
        except Exception:
            logger.debug("Tool discovery failed", exc_info=True)

        # ── ToolHandler ──
        from core.tooling.handler import ToolHandler

        # Check debug_superuser flag from status.json
        _superuser = False
        _status_path = anima_dir / "status.json"
        if _status_path.is_file():
            try:
                import json as _json_mod

                _su_data = _json_mod.loads(_status_path.read_text(encoding="utf-8"))
                _superuser = bool(_su_data.get("debug_superuser"))
            except (ValueError, OSError):
                pass

        # ── BackgroundTaskManager ──
        bg_manager = _build_background_manager(anima_dir)

        _tool_handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=tool_registry,
            personal_tools=personal_tools,
            on_message_sent=None,
            on_schedule_changed=None,
            human_notifier=human_notifier,
            background_manager=bg_manager,
            superuser=_superuser,
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


# ── Trust boundary labeling ───────────────────────────────


def _wrap_result(tool_name: str, result: str) -> str:
    """Apply trust boundary tag to a successful tool result.

    Falls back to the raw *result* when ``wrap_tool_result`` is
    unavailable (should never happen in practice, but keeps the
    MCP server resilient).
    """
    try:
        from core.execution._sanitize import wrap_tool_result

        return wrap_tool_result(tool_name, result)
    except Exception:
        return result


# ── MCP handlers ─────────────────────────────────────────


def _is_consolidation_mode() -> bool:
    """Check whether this Anima is currently running memory consolidation.

    Reads a flag file written by ``run_consolidation()`` in the main process.
    """
    anima_dir_env = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
    if not anima_dir_env:
        return False
    return (Path(anima_dir_env) / "state" / ".consolidation_mode").exists()


_CONSOLIDATION_BLOCKED_NAMES: frozenset[str] = frozenset(
    {"delegate_task", "submit_tasks", "send_message", "post_channel"}
)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return exposed AnimaWorks tools, filtering supervisor tools dynamically."""
    if _is_consolidation_mode():
        return [t for t in MCP_TOOLS if t.name not in _CONSOLIDATION_BLOCKED_NAMES]
    if _has_subordinates_for_anima():
        return MCP_TOOLS
    return [t for t in MCP_TOOLS if t.name not in _SUPERVISOR_TOOL_NAMES]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    """Dispatch a tool call to ToolHandler.handle() in a thread.

    ToolHandler.handle() is synchronous and may perform blocking I/O
    (file reads, subprocess calls), so we run it via ``asyncio.to_thread``
    to keep the MCP event loop responsive.
    """
    # Defense-in-depth: block delegation/messaging tools during consolidation
    if name in _CONSOLIDATION_BLOCKED_NAMES and _is_consolidation_mode():
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "error",
                        "error_type": "ToolBlocked",
                        "message": f"Tool '{name}' is not available during memory consolidation",
                    },
                    ensure_ascii=False,
                ),
            )
        ]

    # Defense-in-depth: reject tool names not in our exposed set.
    # The Agent SDK should only call tools from list_tools(), but
    # ToolHandler.handle() would fall through to external dispatch
    # for unrecognised names, so we gate here explicitly.
    if name not in _EXPOSED_NAMES:
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

    coerced_args = _coerce_integers(dict(arguments or {}), name)

    try:
        result = await asyncio.to_thread(handler.handle, name, coerced_args)
        wrapped = _wrap_result(name, result)
        return [TextContent(type="text", text=wrapped)]
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
