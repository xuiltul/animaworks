from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0


"""Tool call dispatcher and permission enforcement.

``ToolHandler`` is the single entry-point for all synchronous tool execution.
It owns permission checks, memory/file/command operations, and delegates
external tool calls to ``ExternalToolDispatcher``.

The class is composed from responsibility-specific Mixins:
  - MemoryToolsMixin  (handler_memory.py)
  - CommsToolsMixin   (handler_comms.py)
  - OrgToolsMixin     (handler_org.py)
  - SkillsToolsMixin  (handler_skills.py)
  - FileToolsMixin    (handler_files.py)
  - PermissionsMixin  (handler_perms.py)
"""

import json as _json
import logging
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.background import BackgroundTaskManager
from core.exceptions import AnimaWorksError, ConfigError
from core.i18n import t
from core.memory import MemoryManager
from core.memory.activity import ActivityLogger
from core.messenger import Messenger
from core.notification.notifier import HumanNotifier
from core.tooling.dispatch import ExternalToolDispatcher

# ── Re-export all handler_base symbols for backward compatibility ──
from core.tooling.handler_base import (  # noqa: F401
    _BLOCKED_CMD_PATTERNS,
    _EPISODE_FILENAME_RE,
    _INJECTION_RE,
    _NEEDS_SHELL_RE,
    _PROTECTED_DIRS,
    _PROTECTED_FILES,
    _READ_AVG_LINE_LENGTH,
    _READ_CHARS_PER_TOKEN,
    _READ_CONTEXT_FRACTION,
    _READ_FILE_SAFETY_NOTICE,
    _READ_MAX_LINE_CHARS,
    _READ_MAX_LINES,
    _READ_MIN_LINES,
    MemoryWriteError,
    OnMessageSentFn,
    ToolExecutionError,
    _error_result,
    _extract_first_heading,
    _is_protected_write,
    _validate_episode_path,
    _validate_procedure_format,
    _validate_skill_format,
    active_session_type,
    suppress_board_fanout,
)

# ── Import Mixins ──
from core.tooling.handler_comms import CommsToolsMixin
from core.tooling.handler_files import FileToolsMixin
from core.tooling.handler_memory import MemoryToolsMixin
from core.tooling.handler_org import OrgToolsMixin
from core.tooling.handler_perms import PermissionsMixin
from core.tooling.handler_skills import SkillsToolsMixin

logger = logging.getLogger("animaworks.tool_handler")


@runtime_checkable
class ProcessSupervisorLike(Protocol):
    """Minimal interface of ``ProcessSupervisor`` used by ``ToolHandler``."""

    def get_process_status(self, anima_name: str) -> dict[str, Any]: ...


class ToolHandler(
    MemoryToolsMixin,
    CommsToolsMixin,
    OrgToolsMixin,
    SkillsToolsMixin,
    FileToolsMixin,
    PermissionsMixin,
):
    """Dispatches tool calls to the appropriate handler.

    Handles memory tools, file operations, command execution,
    delegation, and external tool dispatch.
    """

    def __init__(
        self,
        anima_dir: Path,
        memory: MemoryManager,
        messenger: Messenger | None = None,
        tool_registry: list[str] | None = None,
        personal_tools: dict[str, str] | None = None,
        on_message_sent: OnMessageSentFn | None = None,
        on_schedule_changed: Callable[[str], None] | None = None,
        human_notifier: HumanNotifier | None = None,
        background_manager: BackgroundTaskManager | None = None,
        context_window: int = 32_000,
        process_supervisor: ProcessSupervisorLike | None = None,
        superuser: bool = False,
    ) -> None:
        self._anima_dir = anima_dir
        self._superuser = superuser
        self._anima_name = anima_dir.name
        self._memory = memory
        self._messenger = messenger
        self._on_message_sent = on_message_sent
        self._on_schedule_changed = on_schedule_changed
        self._human_notifier = human_notifier
        self._background_manager = background_manager
        self._context_window = context_window
        self._process_supervisor = process_supervisor
        self._pending_notifications: list[dict[str, Any]] = []
        self._replied_to: dict[str, set[str]] = {"chat": set(), "background": set()}
        self._posted_channels: dict[str, set[str]] = {"chat": set(), "background": set()}
        self._read_paths: set[str] = set()
        self._session_id: str = uuid.uuid4().hex[:12]
        self._activity = ActivityLogger(self._anima_dir)
        self._state_file_lock: threading.Lock | None = None
        self._external = ExternalToolDispatcher(
            tool_registry or [],
            personal_tools=personal_tools,
        )

        # ── Session origin tracking (provenance Phase 3) ──
        self._session_origin: str = ""
        self._session_origin_chain: list[str] = []

        # ── TaskExec CWD override ──
        self._task_cwd: Path | None = None

        # ── Session trust tracking (security: min trust across all tools used) ──
        # 2 = trusted, 1 = medium, 0 = untrusted; default trusted (no tools used yet)
        self._min_trust_seen: int = 2

        # ── Cache subordinate paths for permission checks ──
        self._subordinate_activity_dirs: list[Path] = []
        self._subordinate_management_files: list[Path] = []
        self._subordinate_root_dirs: list[Path] = []
        self._descendant_activity_dirs: list[Path] = []
        self._descendant_state_files: list[Path] = []
        self._descendant_state_dirs: list[Path] = []
        self._peer_activity_dirs: list[Path] = []
        try:
            from core.config.models import load_config
            from core.paths import get_animas_dir

            _cfg = load_config()
            _animas_dir = get_animas_dir()
            _my_supervisor = None
            if self._anima_name in _cfg.animas:
                _my_supervisor = _cfg.animas[self._anima_name].supervisor
            for _sub_name, _sub_cfg in _cfg.animas.items():
                if _sub_cfg.supervisor == self._anima_name:
                    _sub_dir = (_animas_dir / _sub_name).resolve()
                    self._subordinate_activity_dirs.append(_sub_dir / "activity_log")
                    self._subordinate_management_files.append(_sub_dir / "cron.md")
                    self._subordinate_management_files.append(_sub_dir / "heartbeat.md")
                    self._subordinate_management_files.append(_sub_dir / "status.json")
                    self._subordinate_management_files.append(_sub_dir / "injection.md")
                    self._subordinate_root_dirs.append(_sub_dir)
            # Cache peer activity_log dirs (same supervisor, excluding self)
            for _peer_name, _peer_cfg in _cfg.animas.items():
                if _peer_name != self._anima_name and _peer_cfg.supervisor == _my_supervisor:
                    _peer_dir = (_animas_dir / _peer_name).resolve()
                    self._peer_activity_dirs.append(_peer_dir / "activity_log")
            _all_descendants = self._get_all_descendants()
            for _desc_name in _all_descendants:
                _desc_dir = (_animas_dir / _desc_name).resolve()
                self._descendant_activity_dirs.append(_desc_dir / "activity_log")
                self._descendant_state_files.append(_desc_dir / "state" / "current_task.md")
                self._descendant_state_files.append(_desc_dir / "state" / "pending.md")
                self._descendant_state_files.append(_desc_dir / "status.json")
                self._descendant_state_files.append(_desc_dir / "identity.md")
                self._descendant_state_files.append(_desc_dir / "injection.md")
                self._descendant_state_files.append(_desc_dir / "state" / "task_queue.jsonl")
                self._descendant_state_dirs.append(_desc_dir / "state" / "pending")
        except (ConfigError, OSError, PermissionError, KeyError, AttributeError):
            logger.debug("Failed to cache subordinate paths for %s", self._anima_name, exc_info=True)

        # ── Dispatch table: tool name → handler method ──
        self._dispatch: dict[str, Callable[[dict[str, Any]], str]] = {
            "search_memory": self._handle_search_memory,
            "read_memory_file": self._handle_read_memory_file,
            "write_memory_file": self._handle_write_memory_file,
            "archive_memory_file": self._handle_archive_memory_file,
            "send_message": self._handle_send_message,
            "post_channel": self._handle_post_channel,
            "read_channel": self._handle_read_channel,
            "manage_channel": self._handle_manage_channel,
            "read_dm_history": self._handle_read_dm_history,
            "read_file": self._handle_read_file,
            "write_file": self._handle_write_file,
            "edit_file": self._handle_edit_file,
            "execute_command": self._handle_execute_command,
            "search_code": self._handle_search_code,
            "list_directory": self._handle_list_directory,
            "web_fetch": self._handle_web_fetch,
            "call_human": self._handle_call_human,
            "create_anima": self._handle_create_anima,
            "disable_subordinate": self._handle_disable_subordinate,
            "enable_subordinate": self._handle_enable_subordinate,
            "set_subordinate_model": self._handle_set_subordinate_model,
            "set_subordinate_background_model": self._handle_set_subordinate_background_model,
            "restart_subordinate": self._handle_restart_subordinate,
            "org_dashboard": self._handle_org_dashboard,
            "ping_subordinate": self._handle_ping_subordinate,
            "read_subordinate_state": self._handle_read_subordinate_state,
            "check_permissions": self._handle_check_permissions,
            "delegate_task": self._handle_delegate_task,
            "task_tracker": self._handle_task_tracker,
            "audit_subordinate": self._handle_audit_subordinate,
            "refresh_tools": self._handle_refresh_tools,
            "share_tool": self._handle_share_tool,
            "report_procedure_outcome": self._handle_report_procedure_outcome,
            "report_knowledge_outcome": self._handle_report_knowledge_outcome,
            "skill": self._handle_skill,
            "create_skill": self._handle_create_skill,
            "backlog_task": self._handle_backlog_task,
            "update_task": self._handle_update_task,
            "list_tasks": self._handle_list_tasks,
            "submit_tasks": self._handle_submit_tasks,
            "use_tool": self._handle_use_tool,
            "check_background_task": self._handle_check_background_task,
            "list_background_tasks": self._handle_list_background_tasks,
            "vault_get": self._handle_vault_get,
            "vault_store": self._handle_vault_store,
            "vault_list": self._handle_vault_list,
            # CC-compatible aliases (Mode A/B unified schema)
            "Read": self._handle_read_file,
            "Write": self._handle_write_file,
            "Edit": self._handle_edit_file,
            "Bash": self._handle_execute_command,
            "Grep": self._handle_search_code,
            "Glob": self._handle_glob,
            "WebSearch": self._handle_web_search,
            "WebFetch": self._handle_web_fetch,
        }

    # ── Properties and session management ─────────────────────

    @property
    def on_message_sent(self) -> OnMessageSentFn | None:
        return self._on_message_sent

    @on_message_sent.setter
    def on_message_sent(self, fn: OnMessageSentFn | None) -> None:
        self._on_message_sent = fn

    @property
    def on_schedule_changed(self) -> Callable[[str], None] | None:
        return self._on_schedule_changed

    @on_schedule_changed.setter
    def on_schedule_changed(self, fn: Callable[[str], None] | None) -> None:
        self._on_schedule_changed = fn

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notification events."""
        events = self._pending_notifications
        self._pending_notifications = []
        return events

    @property
    def replied_to(self) -> set[str]:
        """Names already replied to in the current cycle (union of all sessions)."""
        result: set[str] = set()
        for s in self._replied_to.values():
            result |= s
        return result

    def replied_to_for(self, session_type: str) -> set[str]:
        """Names already replied to in a specific session type."""
        return self._replied_to.get(session_type, set())

    def set_state_file_lock(self, lock: threading.Lock) -> None:
        """Attach a state-file lock from DigitalAnima for concurrent write protection."""
        self._state_file_lock = lock

    def set_pending_executor_wake(self, wake_fn: Callable[[], Any]) -> None:
        """Attach the PendingTaskExecutor's wake callback for submit_tasks."""
        self._pending_executor_wake = wake_fn

    def set_task_cwd(self, cwd: Path | None) -> None:
        """Set override cwd for TaskExec command execution."""
        self._task_cwd = cwd

    def _is_state_file(self, path: Path) -> bool:
        """Return True if *path* resolves to state/current_task.md or state/pending.md."""
        try:
            resolved = path.resolve()
            anima_resolved = self._anima_dir.resolve()
            if not resolved.is_relative_to(anima_resolved):
                return False
            rel = str(resolved.relative_to(anima_resolved))
            return rel in ("state/current_task.md", "state/pending.md")
        except (OSError, ValueError):
            return False

    def set_active_session_type(self, session_type: str):
        """Set the active session type for the current context.

        Returns a reset token for use with ``active_session_type.reset(token)``.
        """
        return active_session_type.set(session_type)

    def set_session_origin(self, origin: str, origin_chain: list[str] | None = None) -> None:
        """Set the origin context for the current session.

        Called at the start of each execution path (chat, inbox, heartbeat,
        cron) so that outgoing ``send_message`` / ``delegate_task`` calls
        can propagate provenance information.
        """
        self._session_origin = origin
        self._session_origin_chain = origin_chain or []

    @property
    def session_id(self) -> str:
        """Unique session identifier for double-count prevention."""
        return self._session_id

    def reset_session_id(self) -> None:
        """Generate a new session ID (call at start of each interaction cycle)."""
        self._session_id = uuid.uuid4().hex[:12]
        self._min_trust_seen = 2
        self._read_paths.clear()

    def reset_read_paths(self) -> None:
        """Reset read-path tracking for new session."""
        self._read_paths.clear()

    def reset_replied_to(self, session_type: str | None = None) -> None:
        """Reset replied-to tracking. If session_type given, clear only that session."""
        if session_type:
            self._replied_to.get(session_type, set()).clear()
        else:
            for s in self._replied_to.values():
                s.clear()

    def posted_channels_for(self, session_type: str) -> set[str]:
        """Channels already posted to in a specific session type."""
        return self._posted_channels.get(session_type, set())

    def reset_posted_channels(self, session_type: str | None = None) -> None:
        """Reset posted-channels tracking. If session_type given, clear only that session."""
        if session_type:
            self._posted_channels.get(session_type, set()).clear()
        else:
            for s in self._posted_channels.values():
                s.clear()

    def _persist_replied_to(self, to: str, *, success: bool) -> None:
        """Persist replied_to entry to file for cross-mode tracking."""
        if not self._anima_dir:
            return
        replied_to_path = self._anima_dir / "run" / "replied_to.jsonl"
        try:
            replied_to_path.parent.mkdir(parents=True, exist_ok=True)
            entry = _json.dumps({"to": to, "success": success}, ensure_ascii=False)
            with replied_to_path.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to persist replied_to for '%s': %s", to, e)

    def merge_replied_to(self, names: set[str], session_type: str = "chat") -> None:
        """Merge a set of names into replied-to for a given session."""
        self._replied_to.setdefault(session_type, set()).update(names)

    # ── Main dispatch ────────────────────────────────────────

    _MAX_TOOL_OUTPUT_BYTES = 512_000

    def handle(self, name: str, args: dict[str, Any], tool_use_id: str | None = None) -> str:
        """Synchronous tool call dispatch.

        Routes by tool name to the appropriate handler method via
        ``self._dispatch`` dict lookup.  Falls back to external tool
        dispatch (or background execution) for unregistered names.
        Returns the tool result as a string (truncated if >500KB).
        """
        try:
            logger.debug("tool_call name=%s args_keys=%s", name, list(args.keys()))

            handler = self._dispatch.get(name)
            if handler is not None:
                result = handler(args)
            else:
                # ── Background execution for eligible external tools ──
                if self._background_manager and self._background_manager.is_eligible(name):
                    ext_args = {**args, "anima_dir": str(self._anima_dir)}
                    task_id = self._background_manager.submit(
                        name,
                        ext_args,
                        self._external.dispatch,
                    )
                    result = _json.dumps(
                        {
                            "status": "background",
                            "task_id": task_id,
                            "message": t("handler.background_task_started", task_id=task_id),
                        },
                        ensure_ascii=False,
                    )
                else:
                    ext_args = {**args, "anima_dir": str(self._anima_dir)}
                    result = self._external.dispatch(name, ext_args)
                    if result is None:
                        logger.warning("Unknown tool requested: %s", name)
                        result = f"Unknown tool: {name}"

            self._log_tool_activity(name, args, tool_use_id=tool_use_id)
            self._log_tool_result_activity(name, result, tool_use_id=tool_use_id)
            return self._truncate_output(result)

        except ToolExecutionError:
            raise
        except MemoryWriteError:
            raise
        except Exception as e:
            logger.exception("Unhandled tool error in %s", name)
            raise ToolExecutionError(f"Tool execution failed: {name}: {e}") from e

    def _truncate_output(self, output: str) -> str:
        """Truncate tool output if it exceeds the size limit."""
        size = len(output.encode("utf-8"))
        if size <= self._MAX_TOOL_OUTPUT_BYTES:
            return output
        truncated = output[: self._MAX_TOOL_OUTPUT_BYTES]
        while len(truncated.encode("utf-8")) > self._MAX_TOOL_OUTPUT_BYTES:
            truncated = truncated[:-1000]
        logger.warning(
            "Tool output truncated: original=%d bytes, limit=%d bytes",
            size,
            self._MAX_TOOL_OUTPUT_BYTES,
        )
        return truncated + "\n\n" + t("handler.output_truncated", size=size)

    # ── Activity logging ──────────────────────────────────────

    _ACTIVITY_TYPE_MAP: dict[str, str] = {
        "post_channel": "channel_post",
        "read_channel": "channel_read",
        "read_dm_history": "channel_read",
        "call_human": "human_notify",
    }

    def _log_tool_activity(self, name: str, args: dict[str, Any], *, tool_use_id: str | None = None) -> None:
        """Record tool usage in unified activity log."""
        try:
            activity_type = self._ACTIVITY_TYPE_MAP.get(name)
            meta: dict[str, Any] = {}
            if tool_use_id:
                meta["tool_use_id"] = tool_use_id

            if activity_type is None:
                self._activity.log("tool_use", tool=name, summary=str(args)[:200], meta=meta or None)
            elif name == "post_channel":
                ch = args.get("channel", "")
                txt = args.get("text", "")
                self._activity.log(
                    activity_type,
                    content=txt[:200],
                    channel=ch,
                    summary=f"#{ch}: {txt[:80]}" if ch else txt[:80],
                    meta=meta or None,
                )
            elif name == "read_channel":
                self._activity.log(
                    activity_type,
                    channel=args.get("channel", ""),
                    summary=t("handler.activity_recent_items", limit=args.get("limit", 20)),
                    meta=meta or None,
                )
            elif name == "read_dm_history":
                self._activity.log(
                    activity_type,
                    channel=f"dm:{args.get('peer', '')}",
                    summary=t("handler.activity_dm_history"),
                    meta=meta or None,
                )
            elif name == "call_human":
                self._activity.log(
                    activity_type, content=args.get("body", "")[:1000], via="configured_channels", meta=meta or None
                )
        except Exception as e:
            logger.warning("Activity logging failed for tool '%s': %s", name, e)

    def _log_tool_result_activity(self, name: str, result: str, *, tool_use_id: str | None = None) -> None:
        """Record tool result in unified activity log with structured meta."""
        try:
            meta: dict[str, Any] = {}
            if tool_use_id:
                meta["tool_use_id"] = tool_use_id

            is_error = result.startswith("Error") or result.startswith("error")
            meta["result_status"] = "fail" if is_error else "ok"
            meta["result_bytes"] = len(result.encode("utf-8", errors="replace"))

            if not is_error:
                try:
                    parsed = _json.loads(result)
                    if isinstance(parsed, list):
                        meta["result_count"] = len(parsed)
                    elif isinstance(parsed, dict) and "count" in parsed:
                        meta["result_count"] = parsed["count"]
                except (_json.JSONDecodeError, TypeError):
                    pass

            self._activity.log(
                "tool_result",
                tool=name,
                content=result,
                meta=meta,
            )
        except Exception as e:
            logger.warning("Activity result logging failed for tool '%s': %s", name, e)

    # ── use_tool dispatcher ──────────────────────────────────

    def _handle_use_tool(self, args: dict[str, Any]) -> str:
        """Dispatch to an external tool module via unified use_tool interface.

        Resolves ``tool_name + "_" + action`` as the schema name and
        delegates to the tool module's ``dispatch()`` function directly.
        Supports core tools (TOOL_MODULES), common tools, and personal tools.
        """
        import importlib

        from core.tools import TOOL_MODULES

        tool_name = args.get("tool_name", "")
        action = args.get("action", "")
        tool_args = args.get("args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        if not tool_name or not action:
            return _error_result(
                "InvalidArguments",
                "use_tool requires both 'tool_name' and 'action'",
            )

        personal_tools = self._external._personal_tools or {}
        is_core = tool_name in (self._external.registry or [])
        is_personal = tool_name in personal_tools

        if not is_core and not is_personal:
            return _error_result(
                "PermissionDenied",
                f"Tool '{tool_name}' is not permitted. Check permissions.md for allowed external tools.",
            )

        schema_name = f"{tool_name}_{action}"

        # Check gated action permission
        permitted: set[str] = set()
        try:
            permissions_text = self._memory.read_permissions()
            from core.tooling.permissions import parse_permitted_tools

            permitted = parse_permitted_tools(permissions_text)
        except Exception:
            logger.debug("Failed to parse permissions for gated action check; defaulting to empty set")

        from core.tooling.permissions import is_action_gated

        if is_action_gated(tool_name, action, permitted):
            return _error_result(
                "PermissionDenied",
                t("tooling.gated_action_denied", tool=tool_name, action=action),
            )

        dispatch_args = {**tool_args, "anima_dir": str(self._anima_dir)}

        try:
            if is_personal and tool_name not in TOOL_MODULES:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    f"animaworks_tool_{tool_name}",
                    personal_tools[tool_name],
                )
                if spec is None or spec.loader is None:
                    return _error_result(
                        "LoadError",
                        f"Cannot load personal tool: {tool_name}",
                    )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            else:
                if tool_name not in TOOL_MODULES:
                    return _error_result(
                        "InvalidArguments",
                        f"Unknown tool module: {tool_name}",
                    )
                mod = importlib.import_module(TOOL_MODULES[tool_name])

            result = ExternalToolDispatcher._call_module(mod, schema_name, dispatch_args)
            return result
        except AnimaWorksError:
            raise
        except Exception as e:
            logger.warning("use_tool dispatch failed: %s %s – %s", tool_name, action, e)
            raise ToolExecutionError(
                f"use_tool execution failed: {tool_name}/{action}: {e}",
            ) from e

    # ── Vault tools ──────────────────────────────────────────

    def _handle_vault_get(self, args: dict[str, Any]) -> str:
        """Retrieve a decrypted value from the credential vault."""
        from core.config.vault import get_vault_manager

        section = args.get("section", "")
        key = args.get("key", "")
        if not section or not key:
            return _error_result("InvalidArguments", "section and key are required")

        vault = get_vault_manager()
        value = vault.get(section, key)
        if value is None:
            return _error_result("NotFound", f"No entry for {section}/{key}")
        return value

    def _handle_vault_store(self, args: dict[str, Any]) -> str:
        """Store an encrypted value in the credential vault."""
        from core.config.vault import get_vault_manager

        section = args.get("section", "")
        key = args.get("key", "")
        value = args.get("value", "")
        if not section or not key or not value:
            return _error_result(
                "InvalidArguments",
                "section, key, and value are required",
            )

        vault = get_vault_manager()
        vault.store(section, key, value)
        return _json.dumps(
            {"status": "ok", "message": f"Stored {section}/{key}"},
            ensure_ascii=False,
        )

    def _handle_vault_list(self, args: dict[str, Any]) -> str:
        """List vault sections and keys (values are never shown)."""
        from core.config.vault import get_vault_manager

        vault = get_vault_manager()
        data = vault.load_vault()
        section = args.get("section")
        if section:
            keys = list(data.get(section, {}).keys())
            return _json.dumps(
                {"section": section, "keys": keys},
                ensure_ascii=False,
            )
        sections = {s: list(v.keys()) for s, v in data.items()}
        return _json.dumps({"sections": sections}, ensure_ascii=False)
