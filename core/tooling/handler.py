from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


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
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.exceptions import ToolExecutionError, MemoryWriteError, ProcessError, DeliveryError  # noqa: F401
from core.time_utils import now_iso

from core.background import BackgroundTaskManager
from core.memory.activity import ActivityLogger
from core.tooling.dispatch import ExternalToolDispatcher
from core.memory import MemoryManager
from core.messenger import Messenger
from core.notification.notifier import HumanNotifier

logger = logging.getLogger("animaworks.tool_handler")

# Type alias for the message-sent callback (from, to, content).
OnMessageSentFn = Callable[[str, str, str], None]

# Shell metacharacters that indicate injection attempts.
_SHELL_METACHAR_RE = re.compile(r"[;&|`$(){}]")

# Files that animas cannot modify themselves (identity/privilege protection).
_PROTECTED_FILES = frozenset({
    "permissions.md",
    "identity.md",
    "bootstrap.md",
})

# Standard episode filename: YYYY-MM-DD.md or YYYY-MM-DD_suffix.md
_EPISODE_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(_.+)?\.md$")


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


def _validate_episode_path(rel_path: str) -> str | None:
    """Return a warning if *rel_path* targets ``episodes/`` with a non-standard name.

    Standard patterns (no warning):
      - ``episodes/YYYY-MM-DD.md``
      - ``episodes/YYYY-MM-DD_suffix.md``

    Non-standard patterns (warning returned):
      - ``episodes/random_name.md``
      - ``episodes/2026-99-99.md`` (invalid date chars are caught by regex)

    The warning does NOT block the write — it is appended to the tool
    response so the LLM can learn the convention for future calls.
    """
    if not rel_path.startswith("episodes/"):
        return None

    # Only validate direct children (not subdirectories)
    parts = rel_path.split("/")
    if len(parts) != 2:
        return None

    filename = parts[1]
    if _EPISODE_FILENAME_RE.match(filename):
        return None

    from datetime import date

    return (
        f"WARNING: エピソードファイル名 '{filename}' は標準パターン "
        f"(YYYY-MM-DD.md または YYYY-MM-DD_suffix.md) に合致しません。"
        f" 推奨: episodes/{date.today().isoformat()}.md に"
        f" '## HH:MM — タイトル' 形式で追記してください。"
    )


def _validate_skill_format(content: str) -> str:
    """Validate skill file content format (soft validation).

    Checks for YAML frontmatter with required fields and warns about
    legacy section formats.  Returns an empty string if everything is
    fine, or a newline-joined string of warnings/errors otherwise.

    This is a *soft* validation — the caller should still write the file
    and append the returned messages to the tool response so the LLM
    can self-correct on subsequent calls.
    """
    messages: list[str] = []

    # ── Check YAML frontmatter presence ──
    if not content.startswith("---"):
        return "スキルファイルにはYAMLフロントマター(---)が必要です。"

    # ── Parse frontmatter ──
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return "スキルファイルにはYAMLフロントマター(---)が必要です。"

    frontmatter_raw = content[3:end_idx].strip()
    try:
        import yaml
        frontmatter = yaml.safe_load(frontmatter_raw)
        if not isinstance(frontmatter, dict):
            frontmatter = {}
    except Exception:
        logger.debug("YAML parse fallback for skill validation", exc_info=True)
        # Fallback: simple key: value parsing
        frontmatter = {}
        for line in frontmatter_raw.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                frontmatter[key.strip()] = val.strip()

    # ── Required fields ──
    if "name" not in frontmatter:
        messages.append("`name` フィールドが必要です。")
    if "description" not in frontmatter:
        messages.append("`description` フィールドが必要です。")

    # ── Description quality check (only if description exists) ──
    desc = str(frontmatter.get("description", ""))
    if desc and ("「" not in desc or "」" not in desc):
        messages.append(
            "descriptionに「」キーワードがありません。"
            "自動マッチング精度が低下する可能性があります。"
        )

    # ── Legacy section detection ──
    body = content[end_idx + 3:]
    if "## 概要" in body or "## 発動条件" in body:
        messages.append(
            "旧形式のセクション(## 概要 / ## 発動条件)が検出されました。"
            "Claude Code形式(YAMLフロントマター)への移行を推奨します。"
        )

    return "\n".join(messages)


def _validate_procedure_format(content: str) -> str:
    """Validate procedure file content format (soft validation).

    Checks for YAML frontmatter with a ``description`` field.
    Returns an empty string if everything is fine, or a newline-joined
    string of warnings otherwise.

    This is a *soft* validation -- warnings are appended to the tool
    response but the write is never blocked.
    """
    messages: list[str] = []

    if not content.startswith("---"):
        messages.append(
            "手順書ファイルにはYAMLフロントマター(---)を推奨します。"
            "description フィールドで自動マッチングが有効になります。"
        )
        return "\n".join(messages)

    end_idx = content.find("---", 3)
    if end_idx == -1:
        messages.append(
            "手順書ファイルにはYAMLフロントマター(---)を推奨します。"
        )
        return "\n".join(messages)

    frontmatter_raw = content[3:end_idx].strip()
    try:
        import yaml
        frontmatter = yaml.safe_load(frontmatter_raw)
        if not isinstance(frontmatter, dict):
            frontmatter = {}
    except Exception:
        logger.debug("YAML parse fallback for procedure validation", exc_info=True)
        frontmatter = {}
        for line in frontmatter_raw.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                frontmatter[key.strip()] = val.strip()

    if "description" not in frontmatter:
        messages.append(
            "`description` フィールドがありません。"
            "自動マッチングを有効にするために description を追加してください。"
        )

    return "\n".join(messages)


def _extract_first_heading(text: str) -> str:
    """Extract the first Markdown heading as description."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.lstrip("# ").strip()
    return ""


def _is_protected_write(anima_dir: Path, target: Path) -> str | None:
    """Check if a write target is a protected file or outside anima_dir.

    Returns error message string if blocked, None if allowed.
    """
    resolved = target.resolve()
    anima_resolved = anima_dir.resolve()

    # Path traversal: target must be within anima_dir
    if not resolved.is_relative_to(anima_resolved):
        return _error_result(
            "PermissionDenied",
            "Path resolves outside anima directory",
        )

    # Protected file check
    rel = str(resolved.relative_to(anima_resolved))
    if rel in _PROTECTED_FILES:
        return _error_result(
            "PermissionDenied",
            f"'{rel}' is a protected file and cannot be modified by the anima itself",
        )

    return None


class ToolHandler:
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
        on_schedule_changed: Callable[[str], Any] | None = None,
        human_notifier: HumanNotifier | None = None,
        background_manager: BackgroundTaskManager | None = None,
    ) -> None:
        self._anima_dir = anima_dir
        self._anima_name = anima_dir.name
        self._memory = memory
        self._messenger = messenger
        self._on_message_sent = on_message_sent
        self._on_schedule_changed = on_schedule_changed
        self._human_notifier = human_notifier
        self._background_manager = background_manager
        self._pending_notifications: list[dict[str, Any]] = []
        self._replied_to: set[str] = set()
        self._session_id: str = uuid.uuid4().hex[:12]
        self._activity = ActivityLogger(self._anima_dir)
        self._external = ExternalToolDispatcher(
            tool_registry or [],
            personal_tools=personal_tools,
        )

        # ── Dispatch table: tool name → handler method ──
        self._dispatch: dict[str, Callable[[dict[str, Any]], str]] = {
            "search_memory": self._handle_search_memory,
            "read_memory_file": self._handle_read_memory_file,
            "write_memory_file": self._handle_write_memory_file,
            "send_message": self._handle_send_message,
            "post_channel": self._handle_post_channel,
            "read_channel": self._handle_read_channel,
            "read_dm_history": self._handle_read_dm_history,
            "read_file": self._handle_read_file,
            "write_file": self._handle_write_file,
            "edit_file": self._handle_edit_file,
            "execute_command": self._handle_execute_command,
            "search_code": self._handle_search_code,
            "list_directory": self._handle_list_directory,
            "call_human": self._handle_call_human,
            "create_anima": self._handle_create_anima,
            "refresh_tools": self._handle_refresh_tools,
            "share_tool": self._handle_share_tool,
            "report_procedure_outcome": self._handle_report_procedure_outcome,
            "report_knowledge_outcome": self._handle_report_knowledge_outcome,
            "add_task": self._handle_add_task,
            "update_task": self._handle_update_task,
            "list_tasks": self._handle_list_tasks,
        }

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

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notification events."""
        events = self._pending_notifications
        self._pending_notifications = []
        return events

    @property
    def replied_to(self) -> set[str]:
        """Anima names this anima has sent messages to in the current cycle."""
        return self._replied_to

    @property
    def session_id(self) -> str:
        """Unique session identifier for double-count prevention."""
        return self._session_id

    def reset_session_id(self) -> None:
        """Generate a new session ID (call at start of each interaction cycle)."""
        self._session_id = uuid.uuid4().hex[:12]

    def reset_replied_to(self) -> None:
        """Clear reply tracking (call at start of each heartbeat cycle)."""
        self._replied_to.clear()

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
        except Exception as e:
            logger.warning("Failed to persist replied_to for '%s': %s", to, e)

    def merge_replied_to(self, names: set[str]) -> None:
        """Merge externally detected reply targets into tracking."""
        self._replied_to.update(names)

    # ── Main dispatch ────────────────────────────────────────

    # Maximum tool output size before truncation (500KB)
    _MAX_TOOL_OUTPUT_BYTES = 512_000

    def handle(self, name: str, args: dict[str, Any]) -> str:
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
                        name, ext_args, self._external.dispatch,
                    )
                    result = _json.dumps({
                        "status": "background",
                        "task_id": task_id,
                        "message": f"タスクをバックグラウンドで実行開始しました (task_id: {task_id})",
                    }, ensure_ascii=False)
                else:
                    # External tool dispatch -- inject anima_dir for tools that need it
                    ext_args = {**args, "anima_dir": str(self._anima_dir)}
                    result = self._external.dispatch(name, ext_args)
                    if result is None:
                        logger.warning("Unknown tool requested: %s", name)
                        result = f"Unknown tool: {name}"

            self._log_tool_activity(name, args)
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

    # Activity type mapping: tool name → (activity_type, kwargs_builder)
    _ACTIVITY_TYPE_MAP: dict[str, str] = {
        "post_channel": "channel_post",
        "read_channel": "channel_read",
        "read_dm_history": "channel_read",
        "call_human": "human_notify",
    }

    def _log_tool_activity(self, name: str, args: dict[str, Any]) -> None:
        """Record tool usage in unified activity log."""
        try:
            activity_type = self._ACTIVITY_TYPE_MAP.get(name)

            if activity_type is None:
                self._activity.log("tool_use", tool=name, summary=str(args)[:200])
            elif name == "post_channel":
                self._activity.log(activity_type, content=args.get("text", "")[:200], channel=args.get("channel", ""))
            elif name == "read_channel":
                self._activity.log(activity_type, channel=args.get("channel", ""), summary=f"最新{args.get('limit', 20)}件を確認")
            elif name == "read_dm_history":
                self._activity.log(activity_type, channel=f"dm:{args.get('peer', '')}", summary="DM履歴を確認")
            elif name == "call_human":
                self._activity.log(activity_type, content=args.get("body", "")[:200], via="configured_channels")
        except Exception as e:
            logger.warning("Activity logging failed for tool '%s': %s", name, e)

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
            path = self._anima_dir / rel
            # Prevent path traversal outside anima_dir
            if not path.resolve().is_relative_to(self._anima_dir.resolve()):
                return _error_result(
                    "PermissionDenied",
                    "Path resolves outside anima directory",
                )
        if path.exists() and path.is_file():
            logger.debug("read_memory_file path=%s", rel)
            return path.read_text(encoding="utf-8")
        logger.debug("read_memory_file NOT FOUND path=%s", rel)
        return f"File not found: {rel}"

    def _handle_write_memory_file(self, args: dict[str, Any]) -> str:
        rel = args["path"]
        path = self._anima_dir / rel

        # Security check: block protected files and path traversal
        err = _is_protected_write(self._anima_dir, path)
        if err:
            return err

        # Tool creation permission check
        if rel.startswith("tools/") and rel.endswith(".py"):
            if not self._check_tool_creation_permission("個人ツール"):
                return _error_result(
                    "PermissionDenied",
                    "ツール作成が許可されていません。permissions.md に「ツール作成」セクションを追加してください。",
                )

        content = args["content"]
        mode = args.get("mode", "overwrite")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-add YAML frontmatter for procedure overwrite writes
        auto_frontmatter_applied = False
        if (rel.startswith("procedures/") and rel.endswith(".md")
                and mode == "overwrite"
                and not content.lstrip().startswith("---")):
            desc = _extract_first_heading(content)
            metadata = {
                "description": desc,
                "success_count": 0,
                "failure_count": 0,
                "confidence": 0.5,
            }
            self._memory.write_procedure_with_meta(path, content, metadata)
            auto_frontmatter_applied = True
        elif mode == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")
        logger.info(
            "write_memory_file path=%s mode=%s",
            args["path"], args.get("mode", "overwrite"),
        )

        # Activity log: memory write
        self._activity.log(
            "memory_write",
            summary=f"{rel} ({args.get('mode', 'overwrite')})",
            meta={"path": rel, "mode": args.get("mode", "overwrite")},
        )

        # Trigger schedule reload if heartbeat or cron config changed
        if args["path"] in ("heartbeat.md", "cron.md") and self._on_schedule_changed:
            try:
                self._on_schedule_changed(self._anima_name)
                logger.info("Schedule reload triggered for '%s'", self._anima_name)
            except Exception:
                logger.exception("Schedule reload failed for '%s'", self._anima_name)

        result = f"Written to {args['path']}"

        # Warn (but don't block) if episode filename is non-standard
        episode_warning = _validate_episode_path(args["path"])
        if episode_warning:
            logger.warning("Non-standard episode path: %s", args["path"])
            result = f"{result}\n\n{episode_warning}"

        # Validate skill file format (soft validation: warn but don't block)
        if (rel.startswith("skills/") or rel.startswith("common_skills/")) and rel.endswith(".md"):
            validation_msg = _validate_skill_format(args["content"])
            if validation_msg:
                result = f"{result}\n\n⚠️ スキルフォーマット検証:\n{validation_msg}"

        # Validate procedure file format (soft validation: warn but don't block)
        # Skip when auto-frontmatter was just applied (content already structured)
        if rel.startswith("procedures/") and rel.endswith(".md") and not auto_frontmatter_applied:
            validation_msg = _validate_procedure_format(args["content"])
            if validation_msg:
                result = f"{result}\n\n⚠️ 手順書フォーマット検証:\n{validation_msg}"

        # Auto-update RAG index for skill/procedure writes
        if rel.startswith(("skills/", "procedures/")) and rel.endswith(".md"):
            indexer = self._memory._get_indexer()
            if indexer:
                memory_type = "skills" if rel.startswith("skills/") else "procedures"
                try:
                    indexer.index_file(path, memory_type=memory_type, force=True)
                except Exception as e:
                    logger.warning("Failed to update RAG index for %s: %s", rel, e)

        return result

    def _handle_send_message(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"

        to = args["to"]
        content = args["content"]
        intent = args.get("intent", "")[:50]

        # ── Resolve recipient: internal Anima or external user ──
        try:
            from core.outbound import resolve_recipient, send_external
            from core.config.models import load_config
            from core.paths import get_animas_dir

            config = load_config()
            animas_dir = get_animas_dir()
            known_animas = {
                d.name for d in animas_dir.iterdir() if d.is_dir()
            } if animas_dir.exists() else set()

            resolved = resolve_recipient(
                to, known_animas, config.external_messaging,
            )
        except ValueError as e:
            # Unknown recipient — return helpful error
            return _error_result("UnknownRecipient", str(e))
        except Exception as e:
            logger.warning(
                "Recipient resolution failed for '%s': %s",
                to, e, exc_info=True,
            )
            return _error_result(
                "RecipientResolutionError",
                f"Failed to resolve recipient '{to}': {e}",
                suggestion="Check config.json external_messaging settings",
            )

        # ── External routing ──
        if resolved is not None and not resolved.is_internal:
            logger.info(
                "send_message routed externally: to=%s channel=%s",
                to, resolved.channel,
            )
            self._replied_to.add(to)
            self._persist_replied_to(to, success=True)

            # Log to dm_logs via messenger (Activity Timeline)
            msg = self._messenger.send(
                to=to,
                content=content,
                thread_id=args.get("thread_id", ""),
                reply_to=args.get("reply_to", ""),
                intent=intent,
            )

            if self._on_message_sent:
                try:
                    self._on_message_sent(
                        self._messenger.anima_name, to, content,
                    )
                except Exception:
                    logger.exception("on_message_sent callback failed")

            # Dispatch to external platform
            result = send_external(
                resolved, content, sender_name=self._anima_name,
            )
            return result

        # ── Internal messaging (existing behavior) ──
        internal_to = resolved.name if resolved else to
        msg = self._messenger.send(
            to=internal_to,
            content=content,
            thread_id=args.get("thread_id", ""),
            reply_to=args.get("reply_to", ""),
            intent=intent,
        )

        # Depth limiter may return an error Message without delivery
        if msg.type == "error":
            return f"Error: {msg.content}"

        logger.info("send_message to=%s thread=%s", internal_to, msg.thread_id)
        self._replied_to.add(internal_to)
        self._persist_replied_to(internal_to, success=True)

        if self._on_message_sent:
            try:
                self._on_message_sent(
                    self._messenger.anima_name, internal_to, content,
                )
            except Exception:
                logger.exception("on_message_sent callback failed")

        return f"Message sent to {internal_to} (id: {msg.id}, thread: {msg.thread_id})"

    # ── Channel tool handlers ────────────────────────────────

    def _handle_post_channel(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        channel = args.get("channel", "")
        text = args.get("text", "")
        if not channel or not text:
            return _error_result("InvalidArguments", "channel and text are required")
        self._messenger.post_channel(channel, text)
        logger.info("post_channel channel=%s anima=%s", channel, self._anima_name)

        # ── Board mention fanout ──────────────────────────────
        self._fanout_board_mentions(channel, text)

        return f"Posted to #{channel}"

    def _fanout_board_mentions(self, channel: str, text: str) -> None:
        """Send DM notifications to mentioned Animas when posting to a board channel.

        Parses @all and @name mentions from the posted text and delivers
        board_mention DMs to running Animas (detected via socket files).
        The posting Anima is always excluded from fanout targets.
        """
        if not self._messenger:
            return

        mentions = re.findall(r"@(\w+)", text)
        if not mentions:
            return

        is_all = "all" in mentions

        # Determine running Animas via socket files
        from core.paths import get_data_dir
        sockets_dir = get_data_dir() / "run" / "sockets"
        if sockets_dir.exists():
            running = {p.stem for p in sockets_dir.glob("*.sock")}
        else:
            running = set()

        # Determine fanout targets (running only — stopped Animas have no inbox consumer)
        if is_all:
            targets = running - {self._anima_name}
        else:
            named = {m for m in mentions if m != "all"}
            targets = (named & running) - {self._anima_name}

        if not targets:
            return

        from_name = self._anima_name
        fanout_content = (
            f"[board_reply:channel={channel},from={from_name}]\n"
            f"{from_name}さんがボード #{channel} であなたをメンションしました:\n\n"
            f"{text}\n\n"
            f'返信するには post_channel(channel="{channel}", text="返信内容") を使ってください。'
        )

        for target in sorted(targets):
            try:
                self._messenger.send(
                    to=target,
                    content=fanout_content,
                    msg_type="board_mention",
                )
                logger.info(
                    "board_mention fanout: %s -> %s (channel=%s)",
                    from_name, target, channel,
                )
            except Exception:
                logger.warning(
                    "Failed to fanout board_mention to %s", target, exc_info=True,
                )

    def _handle_read_channel(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        channel = args.get("channel", "")
        if not channel:
            return _error_result("InvalidArguments", "channel is required")
        limit = args.get("limit", 20)
        human_only = args.get("human_only", False)
        messages = self._messenger.read_channel(channel, limit=limit, human_only=human_only)
        if not messages:
            return f"No messages in #{channel}"
        import json as _json
        return _json.dumps(messages, ensure_ascii=False, indent=2)

    def _handle_read_dm_history(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        peer = args.get("peer", "")
        if not peer:
            return _error_result("InvalidArguments", "peer is required")
        limit = args.get("limit", 20)
        messages = self._messenger.read_dm_history(peer, limit=limit)
        if not messages:
            return f"No DM history with {peer}"
        import json as _json
        return _json.dumps(messages, ensure_ascii=False, indent=2)

    # ── Human notification handler ────────────────────────────

    def _handle_call_human(self, args: dict[str, Any]) -> str:
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
                anima_name=self._anima_name,
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

        # Queue proactive message for Web UI chat display + notification broadcast
        notif_data = {
            "anima": self._anima_name,
            "subject": subject,
            "body": body,
            "priority": priority,
            "timestamp": now_iso(),
        }
        self._pending_notifications.append(notif_data)

        import json as _json
        return _json.dumps(
            {"status": "sent", "results": results},
            ensure_ascii=False,
        )

    # ── Admin tool handlers ────────────────────────────────

    def _handle_create_anima(self, args: dict[str, Any]) -> str:
        """Create a new anima from a character sheet via anima_factory.

        Accepts either ``character_sheet_content`` (preferred) or
        ``character_sheet_path``.  The ``supervisor`` parameter overrides the
        character sheet's 上司 field; if omitted the sheet value is used, and
        if the sheet also lacks a supervisor the calling anima is used as
        fallback.
        """
        import json as _json

        from core.anima_factory import create_from_md
        from core.paths import get_data_dir, get_animas_dir

        content = args.get("character_sheet_content")
        sheet_path_raw = args.get("character_sheet_path")
        name = args.get("name")
        explicit_supervisor = args.get("supervisor")

        # Resolve content source
        if content:
            # Content provided directly — preferred path
            md_path = None
        elif sheet_path_raw:
            md_path = Path(sheet_path_raw).expanduser()
            if not md_path.is_absolute():
                md_path = self._anima_dir / md_path
            if not md_path.exists():
                return _error_result(
                    "FileNotFound",
                    f"Character sheet not found: {md_path}",
                    suggestion=(
                        "Use character_sheet_content to pass content directly, "
                        "or ensure the file exists"
                    ),
                )
        else:
            return _error_result(
                "MissingParameter",
                "Either character_sheet_content or character_sheet_path is required",
            )

        try:
            anima_dir = create_from_md(
                get_animas_dir(),
                md_path,
                name=name,
                content=content,
                supervisor=explicit_supervisor,
            )
        except FileExistsError as e:
            return _error_result(
                "AnimaExists",
                str(e),
                suggestion="Choose a different name",
            )
        except ValueError as e:
            return _error_result("InvalidCharacterSheet", str(e))

        # Supervisor fallback: if neither explicit param nor sheet provided
        # a supervisor, use the calling anima as default.
        status_path = anima_dir / "status.json"
        if status_path.exists() and self._anima_name:
            try:
                status_data = _json.loads(status_path.read_text(encoding="utf-8"))
                if not status_data.get("supervisor"):
                    status_data["supervisor"] = self._anima_name
                    status_path.write_text(
                        _json.dumps(status_data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    logger.debug(
                        "Set fallback supervisor '%s' for '%s'",
                        self._anima_name,
                        anima_dir.name,
                    )
            except (OSError, _json.JSONDecodeError):
                logger.warning("Failed to set fallback supervisor", exc_info=True)

        # Register in config.json
        try:
            from cli.commands.init_cmd import _register_anima_in_config
            _register_anima_in_config(get_data_dir(), anima_dir.name)
        except Exception:
            logger.warning("Failed to register anima in config.json", exc_info=True)

        logger.info("create_anima: created '%s' at %s", anima_dir.name, anima_dir)
        return f"Anima '{anima_dir.name}' created successfully at {anima_dir}. Reload the server to activate."

    # ── Tool management handlers ─────────────────────────────

    def _check_tool_creation_permission(self, kind: str) -> bool:
        """Check if tool creation is permitted via permissions.md."""
        permissions = self._memory.read_permissions() if self._memory else ""
        if "ツール作成" not in permissions:
            return False
        _perm_re = re.compile(
            rf"[-*]?\s*{re.escape(kind)}\s*:\s*(OK|yes|enabled|true)\s*$",
            re.IGNORECASE,
        )
        for line in permissions.splitlines():
            if _perm_re.match(line.strip()):
                return True
        return False

    def _handle_refresh_tools(self, args: dict[str, Any]) -> str:
        """Re-discover personal and common tools, update dispatcher."""
        from core.tools import discover_common_tools, discover_personal_tools

        personal = discover_personal_tools(self._anima_dir)
        common = discover_common_tools()
        merged = {**common, **personal}
        self._external.update_personal_tools(merged)

        if not merged:
            return "No personal or common tools found."

        names = ", ".join(sorted(merged.keys()))
        logger.info("refresh_tools: discovered %d tools: %s", len(merged), names)
        return (
            f"Refreshed tools ({len(merged)} discovered): {names}\n"
            "These tools are now available for use."
        )

    def _handle_share_tool(self, args: dict[str, Any]) -> str:
        """Copy a personal tool to common_tools/ for all animas."""
        import shutil

        from core.paths import get_data_dir

        tool_name = args["tool_name"]

        # Sanitise: tool_name must be a valid Python identifier (no path traversal)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool_name):
            return _error_result(
                "InvalidArguments",
                f"Invalid tool name '{tool_name}'. Must be a valid Python identifier.",
                suggestion="Use only letters, digits, and underscores",
            )

        src = self._anima_dir / "tools" / f"{tool_name}.py"
        if not src.exists():
            return _error_result(
                "FileNotFound",
                f"Personal tool '{tool_name}' not found at {src}",
                suggestion="Check tool name with refresh_tools first",
            )

        # Permission check
        if not self._check_tool_creation_permission("共有ツール"):
            return _error_result(
                "PermissionDenied",
                "共有ツール作成が許可されていません。",
            )

        common_dir = get_data_dir() / "common_tools"
        common_dir.mkdir(parents=True, exist_ok=True)
        dst = common_dir / f"{tool_name}.py"
        if dst.exists():
            return _error_result(
                "FileExists",
                f"Common tool '{tool_name}' already exists at {dst}",
                suggestion="Choose a different name or remove the existing tool",
            )

        shutil.copy2(src, dst)
        logger.info("share_tool: copied %s → %s", src, dst)
        return f"Shared tool '{tool_name}' to common_tools/. All animas can now use it after refresh_tools."

    # ── Procedure outcome tracking ──────────────────────────

    def _handle_report_procedure_outcome(self, args: dict[str, Any]) -> str:
        """Report success/failure of a procedure and update its metadata.

        Updates frontmatter fields: success_count, failure_count, last_used,
        and recalculates confidence = success / max(1, success + failure).
        """
        rel = args.get("path", "")
        success = args.get("success", True)
        notes = args.get("notes", "")

        if not rel:
            return _error_result("InvalidArguments", "path is required")

        target = self._anima_dir / rel
        if not target.exists():
            return _error_result(
                "FileNotFound",
                f"File not found: {rel}",
                suggestion="Check the path (e.g. procedures/deploy.md)",
            )

        # Security: must be within anima_dir
        if not target.resolve().is_relative_to(self._anima_dir.resolve()):
            return _error_result("PermissionDenied", "Path resolves outside anima directory")

        # Read existing metadata
        meta = self._memory.read_procedure_metadata(target)

        # Update counts
        if success:
            meta["success_count"] = meta.get("success_count", 0) + 1
        else:
            meta["failure_count"] = meta.get("failure_count", 0) + 1

        # Update last_used
        meta["last_used"] = now_iso()

        # Recalculate confidence
        s = meta.get("success_count", 0)
        f = meta.get("failure_count", 0)
        meta["confidence"] = s / max(1, s + f)

        # Flag for auto-tracking skip (prevents double-counting)
        meta["_reported_session_id"] = self._session_id

        # Read body and rewrite with updated metadata
        body = self._memory.read_procedure_content(target)
        self._memory.write_procedure_with_meta(target, body, meta)

        logger.info(
            "report_procedure_outcome path=%s success=%s confidence=%.2f",
            rel, success, meta["confidence"],
        )

        outcome_label = "成功" if success else "失敗"
        result = (
            f"Procedure outcome recorded: {rel} -> {outcome_label}\n"
            f"confidence: {meta['confidence']:.2f} "
            f"(success: {meta['success_count']}, failure: {meta['failure_count']})"
        )
        if notes:
            result += f"\nnotes: {notes}"

        return result

    # ── Knowledge outcome tracking ──────────────────────────

    def _handle_report_knowledge_outcome(self, args: dict[str, Any]) -> str:
        """Report success/failure of a knowledge file and update its metadata.

        Updates frontmatter fields: success_count, failure_count, last_used,
        and recalculates confidence = success / max(1, success + failure).
        """
        rel = args.get("path", "")
        success = args.get("success", True)
        notes = args.get("notes", "")

        if not rel:
            return _error_result("InvalidArguments", "path is required")

        target = self._anima_dir / rel
        if not target.exists():
            return _error_result(
                "FileNotFound",
                f"File not found: {rel}",
                suggestion="Check the path (e.g. knowledge/topic.md)",
            )

        # Security: must be within anima_dir
        if not target.resolve().is_relative_to(self._anima_dir.resolve()):
            return _error_result("PermissionDenied", "Path resolves outside anima directory")

        # Read existing metadata
        meta = self._memory.read_knowledge_metadata(target)

        # Update counts
        if success:
            meta["success_count"] = meta.get("success_count", 0) + 1
        else:
            meta["failure_count"] = meta.get("failure_count", 0) + 1

        # Update last_used
        meta["last_used"] = datetime.now().isoformat()

        # Recalculate confidence
        s = meta.get("success_count", 0)
        f = meta.get("failure_count", 0)
        meta["confidence"] = s / max(1, s + f)

        # Read body and rewrite with updated metadata
        content = self._memory.read_knowledge_content(target)
        self._memory.write_knowledge_with_meta(target, content, meta)

        logger.info(
            "report_knowledge_outcome path=%s success=%s confidence=%.2f",
            rel, success, meta["confidence"],
        )

        # Activity log: knowledge outcome
        self._activity.log(
            "knowledge_outcome",
            summary=f"{'成功' if success else '失敗'}: {rel}",
            meta={
                "path": rel,
                "success": success,
                "confidence": meta["confidence"],
                "notes": notes[:200] if notes else "",
            },
        )

        outcome_label = "成功" if success else "失敗"
        result = (
            f"Knowledge outcome recorded: {rel} -> {outcome_label}\n"
            f"confidence: {meta.get('confidence', 0):.2f} "
            f"(success: {meta.get('success_count', 0)}, failure: {meta.get('failure_count', 0)})"
        )
        if notes:
            result += f"\nnotes: {notes}"

        return result

    # ── Task queue handlers ─────────────────────────────────

    def _handle_add_task(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)
        source = args.get("source", "anima")
        instruction = args.get("original_instruction", "")
        assignee = args.get("assignee", "")
        summary = args.get("summary", "") or instruction[:100]
        deadline = args.get("deadline")
        relay_chain = args.get("relay_chain", [])

        if not instruction:
            return _error_result("InvalidArguments", "original_instruction is required")
        if not assignee:
            return _error_result("InvalidArguments", "assignee is required")
        if not deadline:
            return _error_result(
                "InvalidArguments",
                "deadline is required. Use relative format ('30m', '2h', '1d') or ISO8601.",
            )

        try:
            entry = manager.add_task(
                source=source,
                original_instruction=instruction,
                assignee=assignee,
                summary=summary,
                deadline=deadline,
                relay_chain=relay_chain,
            )
        except ValueError as e:
            return _error_result("InvalidArguments", str(e))

        # Activity log: task created
        self._activity.log(
            "task_created",
            summary=f"タスク追加: {summary[:100]}",
            meta={"task_id": entry.task_id, "source": source, "assignee": assignee},
        )

        return _json.dumps(entry.model_dump(), ensure_ascii=False, indent=2)

    def _handle_update_task(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)
        task_id = args.get("task_id", "")
        status = args.get("status", "")
        summary = args.get("summary")

        if not task_id:
            return _error_result("InvalidArguments", "task_id is required")
        if not status:
            return _error_result("InvalidArguments", "status is required")

        entry = manager.update_status(task_id, status, summary=summary)
        if entry is None:
            return _error_result(
                "TaskNotFound",
                f"Task not found or invalid status: {task_id}",
            )

        # Activity log: task updated
        self._activity.log(
            "task_updated",
            summary=f"タスク更新: {entry.summary[:100]} → {status}",
            meta={"task_id": task_id, "status": status},
        )

        return _json.dumps(entry.model_dump(), ensure_ascii=False, indent=2)

    def _handle_list_tasks(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)
        status_filter = args.get("status")
        tasks = manager.list_tasks(status=status_filter)
        result = [t.model_dump() for t in tasks]
        return _json.dumps(result, ensure_ascii=False, indent=2)

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
        err = self._check_file_permission(path_str, write=True)
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
        err = self._check_file_permission(path_str, write=True)
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
                cwd=str(self._anima_dir),
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
            search_path = self._anima_dir

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
            dir_path = self._anima_dir

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

    def _check_file_permission(self, path: str, *, write: bool = False) -> str | None:
        """Check if the file path is allowed by permissions.md.

        Returns ``None`` if allowed, or an error message string if denied.

        Access rules (evaluated in order):
          1. Own anima_dir -- always allowed for reads; writes to protected files blocked
          2. Paths listed under ``ファイル操作`` section in permissions.md
          3. Everything else -- denied
        """
        resolved = Path(path).resolve()

        # Own anima_dir
        if resolved.is_relative_to(self._anima_dir.resolve()):
            if write:
                err = _is_protected_write(self._anima_dir, resolved)
                if err:
                    logger.warning("permission_denied anima=%s path=%s reason=protected_file", self._anima_name, path)
                    return err
            return None

        permissions = self._memory.read_permissions()
        if "ファイル操作" not in permissions:
            logger.warning("permission_denied anima=%s path=%s reason=file_ops_not_enabled", self._anima_name, path)
            return _error_result("PermissionDenied", "File operations not enabled in permissions.md")

        # Parse allowed directory whitelist from permissions.md
        raw_items = self._parse_permission_section("ファイル操作")
        allowed_dirs = [
            Path(item).resolve() for item in raw_items if item.startswith("/")
        ]

        if not allowed_dirs:
            logger.warning("permission_denied anima=%s path=%s reason=no_allowed_dirs", self._anima_name, path)
            return _error_result("PermissionDenied", "No allowed paths listed under ファイル操作", suggestion="Add directory paths to permissions.md")

        for allowed in allowed_dirs:
            if resolved.is_relative_to(allowed):
                return None

        logger.warning("permission_denied anima=%s path=%s reason=outside_allowed_dirs", self._anima_name, path)
        return _error_result("PermissionDenied", f"'{path}' is not under any allowed directory", context={"allowed_dirs": [str(d) for d in allowed_dirs]})

    def _check_command_permission(self, command: str) -> str | None:
        """Check if the command is in the allowed list from permissions.md.

        Returns ``None`` if allowed, or an error message string if denied.
        Rejects commands containing shell metacharacters to prevent injection.
        """
        if not command or not command.strip():
            logger.warning("permission_denied anima=%s command=<empty>", self._anima_name)
            return _error_result("PermissionDenied", "Empty command")

        # Reject shell metacharacters regardless of permissions
        if _SHELL_METACHAR_RE.search(command):
            logger.warning("permission_denied anima=%s command=%s reason=shell_metacharacters", self._anima_name, command[:80])
            return _error_result("PermissionDenied", f"Command contains shell metacharacters ({_SHELL_METACHAR_RE.pattern})", suggestion="Use separate tool calls instead of chaining commands")

        permissions = self._memory.read_permissions()
        if "コマンド実行" not in permissions:
            logger.warning("permission_denied anima=%s command=%s reason=cmd_not_enabled", self._anima_name, command[:80])
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
            logger.warning("permission_denied anima=%s command=%s reason=not_in_allowed_list", self._anima_name, cmd_base)
            return _error_result("PermissionDenied", f"Command '{cmd_base}' not in allowed list", context={"allowed_commands": allowed})

        # Block arguments with path traversal targeting other animas
        for arg in argv[1:]:
            if ".." in arg:
                try:
                    resolved = (self._anima_dir / arg).resolve()
                    if not resolved.is_relative_to(self._anima_dir.resolve()):
                        return _error_result(
                            "PermissionDenied",
                            f"Command argument resolves outside anima directory",
                        )
                except (ValueError, OSError):
                    pass

        return None
