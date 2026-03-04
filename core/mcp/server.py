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
import re
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

_EXPOSED_TOOL_NAMES: frozenset[str] = frozenset({
    "send_message",
    "post_channel",
    "read_channel",
    "manage_channel",
    "read_dm_history",
    "add_task",
    "update_task",
    "list_tasks",
    "call_human",
    "search_memory",
    "report_procedure_outcome",
    "report_knowledge_outcome",
    "disable_subordinate",
    "enable_subordinate",
    "skill",
    "plan_tasks",
    "check_background_task",
    "list_background_tasks",
})


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


# Regex matching permission lines like "- chatwork: 全権限" or "- slack: 読み取りのみ"
_PERMISSION_ALLOW_RE = re.compile(
    r"[-*]?\s*(\w+)\s*:\s*(OK|yes|enabled|true|全権限|読み取り.*)\s*$",
    re.IGNORECASE,
)
_PERMISSION_ALL_RE = re.compile(
    r"[-*]?\s*all\s*:\s*(OK|yes|enabled|true)\s*$",
    re.IGNORECASE,
)
_PERMISSION_DENY_RE = re.compile(
    r"[-*]?\s*(\w+)\s*:\s*(no|deny|disabled|false)\s*$",
    re.IGNORECASE,
)


def _load_permitted_categories(anima_dir: Path) -> set[str]:
    """Parse permissions.md to extract permitted external tool categories.

    Mirrors the logic of ``AgentCore._init_tool_registry()`` but operates
    on the raw file without requiring MemoryManager.

    Returns a set of permitted category names (module names from TOOL_MODULES).
    """
    from core.tools import TOOL_MODULES

    all_tools = set(TOOL_MODULES.keys())
    permissions_path = anima_dir / "permissions.md"
    if not permissions_path.is_file():
        return all_tools

    try:
        text = permissions_path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("Cannot read permissions.md from %s", anima_dir)
        return all_tools

    if "外部ツール" not in text:
        return all_tools

    has_all_yes = False
    allowed: list[str] = []
    denied: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if _PERMISSION_ALL_RE.match(stripped):
            has_all_yes = True
            continue
        m_deny = _PERMISSION_DENY_RE.match(stripped)
        if m_deny:
            name = m_deny.group(1)
            if name in all_tools:
                denied.append(name)
            continue
        m_allow = _PERMISSION_ALLOW_RE.match(stripped)
        if m_allow:
            name = m_allow.group(1)
            if name in all_tools:
                allowed.append(name)

    if has_all_yes:
        return all_tools - set(denied)
    if allowed:
        return set(allowed)
    return all_tools


def _build_mcp_tools() -> tuple[list[Tool], frozenset[str]]:
    """Convert canonical AnimaWorks schemas to MCP Tool objects.

    Reads all relevant schema lists from ``core.tooling.schemas`` and
    filters to the exposed tools.  Additionally loads permitted external
    tool schemas (chatwork, slack, etc.) from permissions.md.

    Returns:
        Tuple of (tool_list, exposed_name_set) where exposed_name_set
        is the union of internal and external tool names.
    """
    from core.tooling.schemas import (
        BACKGROUND_TASK_TOOLS,
        CHANNEL_TOOLS,
        KNOWLEDGE_TOOLS,
        MEMORY_TOOLS,
        NOTIFICATION_TOOLS,
        PLAN_TASKS_TOOLS,
        PROCEDURE_TOOLS,
        SKILL_TOOLS,
        SUPERVISOR_TOOLS,
        TASK_TOOLS,
    )

    all_schemas: list[dict[str, Any]] = [
        *MEMORY_TOOLS,
        *CHANNEL_TOOLS,
        *TASK_TOOLS,
        *NOTIFICATION_TOOLS,
        *PROCEDURE_TOOLS,
        *KNOWLEDGE_TOOLS,
        *SUPERVISOR_TOOLS,
        *SKILL_TOOLS,
        *PLAN_TASKS_TOOLS,
        *BACKGROUND_TASK_TOOLS,
    ]

    # Load permitted external tool schemas from permissions.md
    external_schemas: list[dict[str, Any]] = []
    anima_dir_env = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
    if anima_dir_env:
        anima_dir = Path(anima_dir_env).resolve()
        if anima_dir.is_dir():
            try:
                from core.tooling.schemas import load_external_schemas_by_category

                permitted = _load_permitted_categories(anima_dir)
                if permitted:
                    external_schemas = load_external_schemas_by_category(permitted)
                    all_schemas.extend(external_schemas)
                    logger.info(
                        "Loaded %d external tool schemas for categories: %s",
                        len(external_schemas),
                        ", ".join(sorted(permitted)),
                    )
            except Exception:
                logger.debug("External tool schema loading failed", exc_info=True)

    # Build the dynamic exposed set: internal + external tool names
    external_names = frozenset(s["name"] for s in external_schemas)
    exposed = _EXPOSED_TOOL_NAMES | external_names

    # Apply DB description overrides
    from core.tooling.schemas import apply_db_descriptions

    all_schemas = apply_db_descriptions(all_schemas)

    # Cache original schemas for type coercion lookup
    for schema in all_schemas:
        if schema["name"] in exposed:
            _TOOL_SCHEMAS[schema["name"]] = schema.get(
                "parameters", {}
            )

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
            procedure_metas = [
                SkillMetadataService.extract_skill_meta(f)
                for f in sorted(procedures_dir.glob("*.md"))
            ]

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
        config_eligible = {
            name: tc.threshold_s
            for name, tc in config.background_task.eligible_tools.items()
        }

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
                task.task_id, task.tool_name, task.status.value,
            )
        except Exception:
            logger.exception(
                "Failed to write MCP bg task notification for %s", task.task_id,
            )

    return _on_complete


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
