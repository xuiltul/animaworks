from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Build a standalone ``ToolHandler`` for non-Web processes.

Shared by the MCP subprocess (``core.mcp.server``) and anima-context CLI
adapters, which need a fully assembled ``ToolHandler`` from an ``anima_dir``
without a running ``AgentCore``.
"""

import json as _json
import logging
import sys
from functools import cache
from pathlib import Path
from typing import Any

from core.platform.env import anima_dir_env, has_env

logger = logging.getLogger(__name__)


def current_anima_dir() -> Path | None:
    """Return the calling anima's directory, or ``None`` outside a tool context."""
    value = anima_dir_env() or ""
    if not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


def require_anima_dir() -> Path:
    """Return the calling anima's directory or exit with a usage error."""
    anima_dir = current_anima_dir()
    if anima_dir is None:
        print(
            "Error: ANIMAWORKS_ANIMA_DIR not set or missing (set automatically inside an anima's tool context)",
            file=sys.stderr,
        )
        sys.exit(1)
    return anima_dir


@cache
def _standalone_handler(anima_dir: Path) -> Any:
    return build_standalone_tool_handler(anima_dir, for_mcp=False)


def run_tool_for_current_anima(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    anima_dir: Path | None = None,
) -> str:
    """Run *tool_name* through the current anima's standalone ``ToolHandler``."""
    return _standalone_handler((anima_dir or require_anima_dir()).resolve()).handle(tool_name, tool_args)


_ERROR_PREFIXES = ("error", "unknown tool", "エラー", "오류", "错误", "錯誤")
# Plain-text refusals from send_message that do not start with an error prefix.
_ERROR_HINTS = (
    "cannot send to '",
    "宛先 '",
    "dm send failed:",
    "dm送信失敗:",
)


def tool_result_is_error(result: str) -> bool:
    """Recognize ToolHandler errors, including results with appended action rules."""
    text = result.lstrip()
    try:
        payload, _ = _json.JSONDecoder().raw_decode(text)
    except (ValueError, TypeError):
        payload = text
    if isinstance(payload, dict):
        return str(payload.get("status", "")).casefold() == "error"
    if not isinstance(payload, str):
        return False
    first_line = payload.lstrip().split("\n", 1)[0].casefold()
    return first_line.startswith(_ERROR_PREFIXES) or any(hint in first_line for hint in _ERROR_HINTS)


def notify_server_message_sent(
    from_person: str,
    to_person: str,
    content: str,
    message_id: str = "",
) -> None:
    """Broadcast a sent DM or Board post through the server, if it is running."""
    from core.platform.pid import is_server_running

    if not is_server_running():
        return

    try:
        from core.internal_api import host_api

        response = host_api.post(
            "/api/internal/message-sent",
            json={
                "from_person": from_person,
                "to_person": to_person,
                "content": content[:200],
                "message_id": message_id,
            },
            timeout=5.0,
        )
        if response.status_code == 200:
            logger.debug("Server notified of message: %s -> %s", from_person, to_person)
        else:
            logger.debug("Server notification failed: %s", response.status_code)
    except Exception:
        logger.debug("Could not notify server of message", exc_info=True)


def _load_permitted_categories(anima_dir: Path) -> set[str]:
    """Load permitted external tool categories from permissions.json."""
    from core.config.models import load_permissions
    from core.tooling.permissions import get_permitted_tools

    config = load_permissions(anima_dir)
    return get_permitted_tools(config)


def _build_background_manager(anima_dir: Path) -> Any:
    """Build a BackgroundTaskManager for a standalone ToolHandler process.

    Mirrors ``AgentCore._build_background_manager()`` but operates
    without the full AgentCore.  Returns None when disabled in config.
    """
    try:
        from core.config.models import load_config

        config = load_config()
        if not config.background_task.enabled:
            return None

        from core.integrations._base import load_execution_profiles
        from core.tasks.background import BackgroundTaskManager
        from core.tooling.policy.registry import get_tool_modules

        profiles = load_execution_profiles(get_tool_modules())
        config_eligible = {name: tc.threshold_s for name, tc in config.background_task.eligible_tools.items()}

        mgr = BackgroundTaskManager.from_profiles(
            anima_dir=anima_dir,
            anima_name=anima_dir.name,
            profiles=profiles,
            config_eligible=config_eligible or None,
        )

        mgr.on_complete = _make_on_complete_callback(anima_dir)
        logger.info("BackgroundTaskManager initialised for standalone (anima=%s)", anima_dir.name)
        return mgr

    except Exception:
        logger.debug("BackgroundTaskManager init skipped in standalone", exc_info=True)
        return None


def _make_on_complete_callback(anima_dir: Path) -> Any:
    """Create an on_complete callback that writes notification files.

    Without access to WebSocket or HumanNotifier, we only write a
    notification file under ``state/background_notifications/`` for the
    next heartbeat to pick up.
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
                "Standalone bg task notification written: %s (tool=%s, status=%s)",
                task.task_id,
                task.tool_name,
                task.status.value,
            )
        except Exception:
            logger.exception("Failed to write standalone bg task notification for %s", task.task_id)

    return _on_complete


def _wrap_cli_handler_with_runtime_session(handler: Any) -> Any:
    """Bind each short-lived CLI invocation to the caller's runtime session.

    Records such as ``run/replied_to`` are then written under the session the
    executor reads. The per-run "already sent" guards are not seeded from those
    files: they accumulate across runs and would block later runs.
    """
    original_handle = handler.handle

    def _handle_with_runtime_session(
        name: str,
        args: dict[str, Any],
        tool_use_id: str | None = None,
    ) -> str:
        from core.execution.session.session_context import RuntimeSessionContext, runtime_session_scope

        ctx = RuntimeSessionContext.from_env()
        session_type = ctx.session_type if ctx is not None else "unknown"
        if ctx is not None:
            handler.bind_runtime_session(ctx)
        session_token = handler.set_active_session_type(session_type)
        try:
            if ctx is None:
                return original_handle(name, args, tool_use_id=tool_use_id)
            with runtime_session_scope(ctx):
                return original_handle(name, args, tool_use_id=tool_use_id)
        finally:
            session_token.var.reset(session_token)

    handler.handle = _handle_with_runtime_session
    return handler


def build_standalone_tool_handler(anima_dir: Path, *, for_mcp: bool) -> Any:
    """Assemble a fully configured ``ToolHandler`` from *anima_dir*.

    Used by the MCP subprocess (``for_mcp=True``) and anima-context CLI
    adapters (``for_mcp=False``). The flag controls the ``debug_superuser``
    gate and whether each CLI invocation is wrapped in its runtime session.
    """
    anima_dir = Path(anima_dir).resolve()

    # ── MemoryManager ──
    from core.memory import MemoryManager

    memory = MemoryManager(anima_dir)

    # ── Messenger ──
    from core.messaging.messenger import Messenger
    from core.paths import get_shared_dir

    messenger = Messenger(shared_dir=get_shared_dir(), anima_name=anima_dir.name)

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
        from core.integrations import discover_common_tools, discover_personal_tools

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
    if not (for_mcp and has_env("ANIMAWORKS_MCP_TOOLS")) and _status_path.is_file():
        try:
            _su_data = _json.loads(_status_path.read_text(encoding="utf-8"))
            _superuser = bool(_su_data.get("debug_superuser"))
        except (ValueError, OSError):
            pass

    # ── BackgroundTaskManager ──
    bg_manager = _build_background_manager(anima_dir)

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
        tool_registry=tool_registry,
        personal_tools=personal_tools,
        # MCP keeps its previous behaviour; the CLI notified the server itself.
        on_message_sent=None if for_mcp else notify_server_message_sent,
        on_schedule_changed=None,
        human_notifier=human_notifier,
        background_manager=bg_manager,
        superuser=_superuser,
    )

    if not for_mcp:
        handler = _wrap_cli_handler_with_runtime_session(handler)

    logger.info("ToolHandler initialised for anima '%s' (%s)", anima_dir.name, anima_dir)
    return handler
