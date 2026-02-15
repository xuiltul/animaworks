from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Tool call dispatcher and permission enforcement.

``ToolHandler`` is the single entry-point for all synchronous tool execution.
It owns permission checks, memory/file/command operations, and delegates
external tool calls to ``ExternalToolDispatcher``.
"""

import json as _json
import logging
import re
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.background import BackgroundTaskManager
from core.tooling.dispatch import ExternalToolDispatcher
from core.memory import MemoryManager
from core.messenger import Messenger
from core.notification.notifier import HumanNotifier

logger = logging.getLogger("animaworks.tool_handler")

# Type alias for the message-sent callback (from, to, content).
OnMessageSentFn = Callable[[str, str, str], None]

# Shell metacharacters that indicate injection attempts.
_SHELL_METACHAR_RE = re.compile(r"[;&|`$(){}]")


def _error_result(
    error_type: str,
    message: str,
    *,
    context: dict[str, Any] | None = None,
    suggestion: str = "",
) -> str:
    """Build a structured error response for LLM consumption."""
    import json as _json
    result: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    if context:
        result["context"] = context
    if suggestion:
        result["suggestion"] = suggestion
    return _json.dumps(result, ensure_ascii=False)


class ToolHandler:
    """Dispatches tool calls to the appropriate handler.

    Handles memory tools, file operations, command execution,
    delegation, and external tool dispatch.
    """

    def __init__(
        self,
        person_dir: Path,
        memory: MemoryManager,
        messenger: Messenger | None = None,
        tool_registry: list[str] | None = None,
        personal_tools: dict[str, str] | None = None,
        on_message_sent: OnMessageSentFn | None = None,
        on_schedule_changed: Callable[[str], Any] | None = None,
        human_notifier: HumanNotifier | None = None,
        background_manager: BackgroundTaskManager | None = None,
    ) -> None:
        self._person_dir = person_dir
        self._person_name = person_dir.name
        self._memory = memory
        self._messenger = messenger
        self._on_message_sent = on_message_sent
        self._on_schedule_changed = on_schedule_changed
        self._human_notifier = human_notifier
        self._background_manager = background_manager
        self._replied_to: set[str] = set()
        self._external = ExternalToolDispatcher(
            tool_registry or [],
            personal_tools=personal_tools,
        )

    @property
    def on_message_sent(self) -> OnMessageSentFn | None:
        return self._on_message_sent

    @on_message_sent.setter
    def on_message_sent(self, fn: OnMessageSentFn | None) -> None:
        self._on_message_sent = fn

    @property
    def on_schedule_changed(self) -> Callable[[str], Any] | None:
        return self._on_schedule_changed

    @on_schedule_changed.setter
    def on_schedule_changed(self, fn: Callable[[str], Any] | None) -> None:
        self._on_schedule_changed = fn

    @property
    def replied_to(self) -> set[str]:
        """Person names this person has sent messages to in the current cycle."""
        return self._replied_to

    def reset_replied_to(self) -> None:
        """Clear reply tracking (call at start of each heartbeat cycle)."""
        self._replied_to.clear()

    def merge_replied_to(self, names: set[str]) -> None:
        """Merge externally detected reply targets into tracking."""
        self._replied_to.update(names)

    # ── Main dispatch ────────────────────────────────────────

    # Maximum tool output size before truncation (500KB)
    _MAX_TOOL_OUTPUT_BYTES = 512_000

    def handle(self, name: str, args: dict[str, Any]) -> str:
        """Synchronous tool call dispatch.

        Routes by tool name to the appropriate handler method.
        Returns the tool result as a string (truncated if >500KB).
        """
        try:
            logger.debug("tool_call name=%s args_keys=%s", name, list(args.keys()))

            result: str | None = None

            # Memory tools
            if name == "search_memory":
                result = self._handle_search_memory(args)
            elif name == "read_memory_file":
                result = self._handle_read_memory_file(args)
            elif name == "write_memory_file":
                result = self._handle_write_memory_file(args)
            elif name == "send_message":
                result = self._handle_send_message(args)
            # File operation tools
            elif name == "read_file":
                result = self._handle_read_file(args)
            elif name == "write_file":
                result = self._handle_write_file(args)
            elif name == "edit_file":
                result = self._handle_edit_file(args)
            elif name == "execute_command":
                result = self._handle_execute_command(args)
            # Search tools
            elif name == "search_code":
                result = self._handle_search_code(args)
            elif name == "list_directory":
                result = self._handle_list_directory(args)
            # Human notification
            elif name == "notify_human":
                result = self._handle_notify_human(args)
            else:
                # ── Background execution for eligible external tools ──
                if self._background_manager and self._background_manager.is_eligible(name):
                    ext_args = {**args, "person_dir": str(self._person_dir)}
                    task_id = self._background_manager.submit(
                        name, ext_args, self._external.dispatch,
                    )
                    result = _json.dumps({
                        "status": "background",
                        "task_id": task_id,
                        "message": f"タスクをバックグラウンドで実行開始しました (task_id: {task_id})",
                    }, ensure_ascii=False)
                else:
                    # External tool dispatch -- inject person_dir for tools that need it
                    ext_args = {**args, "person_dir": str(self._person_dir)}
                    result = self._external.dispatch(name, ext_args)
                    if result is None:
                        logger.warning("Unknown tool requested: %s", name)
                        result = f"Unknown tool: {name}"

            return self._truncate_output(result)

        except Exception as e:
            logger.exception("Unhandled tool error in %s", name)
            return _error_result(
                "UnhandledError", f"Tool execution failed: {name}: {e}",
            )

    def _truncate_output(self, output: str) -> str:
        """Truncate tool output if it exceeds the size limit."""
        size = len(output.encode("utf-8"))
        if size <= self._MAX_TOOL_OUTPUT_BYTES:
            return output
        # Truncate by character estimate (UTF-8 can be multi-byte)
        # Use a conservative ratio to ensure we stay under the byte limit
        truncated = output[:self._MAX_TOOL_OUTPUT_BYTES]
        while len(truncated.encode("utf-8")) > self._MAX_TOOL_OUTPUT_BYTES:
            truncated = truncated[:-1000]
        logger.warning(
            "Tool output truncated: original=%d bytes, limit=%d bytes",
            size, self._MAX_TOOL_OUTPUT_BYTES,
        )
        return (
            truncated
            + f"\n\n[出力が500KBを超えたためトランケーションしました。元のサイズ: {size}]"
        )

    # ── Memory tool handlers ─────────────────────────────────

    def _handle_search_memory(self, args: dict[str, Any]) -> str:
        scope = args.get("scope", "all")
        query = args.get("query", "")
        results = self._memory.search_memory_text(query, scope=scope)
        logger.debug(
            "search_memory query=%s scope=%s results=%d",
            query, scope, len(results),
        )
        if not results:
            return f"No results for '{query}'"
        return "\n".join(f"- {fname}: {line}" for fname, line in results[:10])

    def _handle_read_memory_file(self, args: dict[str, Any]) -> str:
        rel = args["path"]
        # Support common_knowledge/ prefix — resolve to shared dir
        if rel.startswith("common_knowledge/"):
            from core.paths import get_common_knowledge_dir
            suffix = rel[len("common_knowledge/"):]
            path = get_common_knowledge_dir() / suffix
        else:
            path = self._person_dir / rel
        if path.exists() and path.is_file():
            logger.debug("read_memory_file path=%s", rel)
            return path.read_text(encoding="utf-8")
        logger.debug("read_memory_file NOT FOUND path=%s", rel)
        return f"File not found: {rel}"

    def _handle_write_memory_file(self, args: dict[str, Any]) -> str:
        path = self._person_dir / args["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.get("mode") == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(args["content"])
        else:
            path.write_text(args["content"], encoding="utf-8")
        logger.info(
            "write_memory_file path=%s mode=%s",
            args["path"], args.get("mode", "overwrite"),
        )

        # Trigger schedule reload if heartbeat or cron config changed
        if args["path"] in ("heartbeat.md", "cron.md") and self._on_schedule_changed:
            try:
                self._on_schedule_changed(self._person_name)
                logger.info("Schedule reload triggered for '%s'", self._person_name)
            except Exception:
                logger.exception("Schedule reload failed for '%s'", self._person_name)

        return f"Written to {args['path']}"

    def _handle_send_message(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        msg = self._messenger.send(
            to=args["to"],
            content=args["content"],
            thread_id=args.get("thread_id", ""),
            reply_to=args.get("reply_to", ""),
        )
        logger.info("send_message to=%s thread=%s", args["to"], msg.thread_id)
        self._replied_to.add(args["to"])

        if self._on_message_sent:
            try:
                self._on_message_sent(
                    self._messenger.person_name, args["to"], args["content"],
                )
            except Exception:
                logger.exception("on_message_sent callback failed")

        return f"Message sent to {args['to']} (id: {msg.id}, thread: {msg.thread_id})"

    # ── Human notification handler ────────────────────────────

    def _handle_notify_human(self, args: dict[str, Any]) -> str:
        if not self._human_notifier:
            return _error_result(
                "NotConfigured",
                "Human notification is not configured",
                suggestion="Enable human_notification in config.json",
            )
        if self._human_notifier.channel_count == 0:
            return _error_result(
                "NotConfigured",
                "No notification channels configured",
                suggestion="Add channels to human_notification.channels in config.json",
            )

        import asyncio

        subject = args.get("subject", "")
        body = args.get("body", "")
        priority = args.get("priority", "normal")

        if not subject or not body:
            return _error_result(
                "InvalidArguments",
                "subject and body are required",
            )

        try:
            coro = self._human_notifier.notify(
                subject, body, priority,
                person_name=self._person_name,
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                # Already inside an async context — run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(asyncio.run, coro).result(timeout=60)
            else:
                results = asyncio.run(coro)
        except Exception as e:
            return _error_result("NotificationError", f"Failed to send notification: {e}")

        import json as _json
        return _json.dumps(
            {"status": "sent", "results": results},
            ensure_ascii=False,
        )

    # ── File operation handlers ──────────────────────────────

    def _handle_read_file(self, args: dict[str, Any]) -> str:
        path_str = args.get("path", "")
        err = self._check_file_permission(path_str)
        if err:
            return err
        path = Path(path_str)
        if not path.exists():
            return _error_result("FileNotFound", f"File not found: {path_str}", suggestion="Use list_directory to find the correct path")
        if not path.is_file():
            return _error_result("InvalidArguments", f"Not a file: {path_str}", suggestion="Provide a file path, not a directory")
        try:
            content = path.read_text(encoding="utf-8")
            logger.info("read_file path=%s len=%d", path_str, len(content))
            return content[:100_000]  # cap at 100k chars
        except Exception as e:
            return _error_result("ReadError", f"Error reading {path_str}: {e}")

    def _handle_write_file(self, args: dict[str, Any]) -> str:
        path_str = args.get("path", "")
        err = self._check_file_permission(path_str)
        if err:
            return err
        path = Path(path_str)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.get("content", ""), encoding="utf-8")
            logger.info("write_file path=%s", path_str)
            return f"Written to {path_str}"
        except Exception as e:
            return _error_result("WriteError", f"Error writing {path_str}: {e}")

    def _handle_edit_file(self, args: dict[str, Any]) -> str:
        path_str = args.get("path", "")
        err = self._check_file_permission(path_str)
        if err:
            return err
        path = Path(path_str)
        if not path.exists():
            return _error_result("FileNotFound", f"File not found: {path_str}", suggestion="Use list_directory to find the correct path")
        try:
            content = path.read_text(encoding="utf-8")
            old = args.get("old_string", "")
            new = args.get("new_string", "")
            if old not in content:
                return _error_result("StringNotFound", f"old_string not found in {path_str}", suggestion="Use search_code to find the exact string")
            count = content.count(old)
            if count > 1:
                return _error_result("AmbiguousMatch", f"old_string matches {count} locations", context={"match_count": count}, suggestion="Provide more surrounding context to make it unique")
            content = content.replace(old, new, 1)
            path.write_text(content, encoding="utf-8")
            logger.info("edit_file path=%s", path_str)
            return f"Edited {path_str}"
        except Exception as e:
            return _error_result("EditError", f"Error editing {path_str}: {e}")

    def _handle_execute_command(self, args: dict[str, Any]) -> str:
        command = args.get("command", "")
        err = self._check_command_permission(command)
        if err:
            return err
        timeout = args.get("timeout", 30)
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return _error_result("InvalidArguments", f"Error parsing command: {e}")
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._person_dir),
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            logger.info(
                "execute_command cmd=%s rc=%d", command[:80], proc.returncode,
            )
            return output[:50_000] or f"(exit code {proc.returncode})"
        except subprocess.TimeoutExpired:
            return _error_result("Timeout", f"Command timed out after {timeout}s", suggestion="Increase timeout or break the command into smaller steps")
        except Exception as e:
            return _error_result("ExecutionError", f"Error executing command: {e}")

    # ── Search tool handlers ──────────────────────────────────

    def _handle_search_code(self, args: dict[str, Any]) -> str:
        import re as _re

        pattern_str = args.get("pattern", "")
        if not pattern_str:
            return _error_result(
                "InvalidArguments", "pattern is required",
                suggestion="Provide a regex pattern to search for",
            )

        try:
            regex = _re.compile(pattern_str)
        except _re.error as e:
            return _error_result(
                "InvalidArguments", f"Invalid regex: {e}",
                suggestion="Use a valid Python regex pattern",
            )

        search_path_str = args.get("path", "")
        if search_path_str:
            search_path = Path(search_path_str)
            err = self._check_file_permission(search_path_str)
            if err:
                return err
        else:
            search_path = self._person_dir

        if not search_path.exists():
            return _error_result(
                "FileNotFound", f"Path not found: {search_path}",
                suggestion="Use list_directory to find the correct path",
            )

        glob_pattern = args.get("glob", "")
        matches: list[str] = []
        max_matches = 50

        if search_path.is_file():
            files = [search_path]
        elif glob_pattern:
            files = list(search_path.rglob(glob_pattern))
        else:
            files = list(search_path.rglob("*"))

        for fpath in files:
            if not fpath.is_file():
                continue
            if len(matches) >= max_matches:
                break
            try:
                for line_num, line in enumerate(
                    fpath.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if regex.search(line):
                        rel = fpath.relative_to(search_path) if search_path.is_dir() else fpath.name
                        matches.append(f"{rel}:{line_num}: {line.rstrip()}")
                        if len(matches) >= max_matches:
                            break
            except (OSError, UnicodeDecodeError):
                continue

        logger.info("search_code pattern=%s matches=%d", pattern_str, len(matches))
        if not matches:
            return f"No matches for pattern '{pattern_str}'"
        result = "\n".join(matches)
        if len(matches) == max_matches:
            result += f"\n(truncated at {max_matches} matches)"
        return result

    def _handle_list_directory(self, args: dict[str, Any]) -> str:
        dir_path_str = args.get("path", "")
        if dir_path_str:
            dir_path = Path(dir_path_str)
            err = self._check_file_permission(dir_path_str)
            if err:
                return err
        else:
            dir_path = self._person_dir

        if not dir_path.exists():
            return _error_result(
                "FileNotFound", f"Directory not found: {dir_path}",
                suggestion="Check the path and try again",
            )
        if not dir_path.is_dir():
            return _error_result(
                "InvalidArguments", f"Not a directory: {dir_path}",
                suggestion="Provide a directory path, not a file path",
            )

        pattern = args.get("pattern", "")
        recursive = args.get("recursive", False)

        entries: list[str] = []
        max_entries = 200

        if pattern:
            if recursive:
                items = list(dir_path.rglob(pattern))
            else:
                items = list(dir_path.glob(pattern))
        elif recursive:
            items = list(dir_path.rglob("*"))
        else:
            items = list(dir_path.iterdir())

        for item in sorted(items)[:max_entries]:
            suffix = "/" if item.is_dir() else ""
            try:
                rel = item.relative_to(dir_path)
            except ValueError:
                rel = item
            entries.append(f"{rel}{suffix}")

        logger.info("list_directory path=%s entries=%d", dir_path, len(entries))
        if not entries:
            return "(empty directory)"
        result = "\n".join(entries)
        if len(items) > max_entries:
            result += f"\n(truncated at {max_entries} entries, total: {len(items)})"
        return result

    # ── Permission checks ────────────────────────────────────

    def _parse_permission_section(self, section_header: str) -> list[str]:
        """Parse a section from permissions.md and return allowed items.

        Scans for a line containing *section_header*, then collects list
        items (lines starting with ``-``) until the next ``#`` heading or
        EOF.  Each item is the text before the first ``:`` on the line.

        Args:
            section_header: Text to search for in a section heading
                (e.g. ``"ファイル操作"`` or ``"コマンド実行"``).

        Returns:
            List of extracted item strings (may be empty).
        """
        permissions = self._memory.read_permissions()
        items: list[str] = []
        in_section = False
        for line in permissions.splitlines():
            stripped = line.strip()
            if section_header in stripped:
                in_section = True
                continue
            if in_section and stripped.startswith("#"):
                break
            if in_section and stripped.startswith("-"):
                item = stripped.lstrip("- ").split(":")[0].strip()
                if item:
                    items.append(item)
        return items

    def _check_file_permission(self, path: str) -> str | None:
        """Check if the file path is allowed by permissions.md.

        Returns ``None`` if allowed, or an error message string if denied.

        Access rules (evaluated in order):
          1. Own person_dir -- always allowed
          2. Paths listed under ``ファイル操作`` section in permissions.md
          3. Everything else -- denied
        """
        resolved = Path(path).resolve()

        # Always allow access to own person_dir
        if resolved.is_relative_to(self._person_dir.resolve()):
            return None

        permissions = self._memory.read_permissions()
        if "ファイル操作" not in permissions:
            return _error_result("PermissionDenied", "File operations not enabled in permissions.md")

        # Parse allowed directory whitelist from permissions.md
        raw_items = self._parse_permission_section("ファイル操作")
        allowed_dirs = [
            Path(item).resolve() for item in raw_items if item.startswith("/")
        ]

        if not allowed_dirs:
            return _error_result("PermissionDenied", "No allowed paths listed under ファイル操作", suggestion="Add directory paths to permissions.md")

        for allowed in allowed_dirs:
            if resolved.is_relative_to(allowed):
                return None

        return _error_result("PermissionDenied", f"'{path}' is not under any allowed directory", context={"allowed_dirs": [str(d) for d in allowed_dirs]})

    def _check_command_permission(self, command: str) -> str | None:
        """Check if the command is in the allowed list from permissions.md.

        Returns ``None`` if allowed, or an error message string if denied.
        Rejects commands containing shell metacharacters to prevent injection.
        """
        if not command or not command.strip():
            return _error_result("PermissionDenied", "Empty command")

        # Reject shell metacharacters regardless of permissions
        if _SHELL_METACHAR_RE.search(command):
            return _error_result("PermissionDenied", f"Command contains shell metacharacters ({_SHELL_METACHAR_RE.pattern})", suggestion="Use separate tool calls instead of chaining commands")

        permissions = self._memory.read_permissions()
        if "コマンド実行" not in permissions:
            return _error_result("PermissionDenied", "Command execution not enabled in permissions.md")

        # Parse the command safely
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return _error_result("PermissionDenied", f"Invalid command syntax: {e}")

        if not argv:
            return "Permission denied: empty command after parsing"

        # Extract allowed commands (lines like "- git: OK" or "- npm: OK")
        allowed = self._parse_permission_section("コマンド実行")
        if not allowed:
            return None  # No explicit list = allow all (section exists)

        cmd_base = argv[0]
        if cmd_base not in allowed:
            return _error_result("PermissionDenied", f"Command '{cmd_base}' not in allowed list", context={"allowed_commands": allowed})
        return None
