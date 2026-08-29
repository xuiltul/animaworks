from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Mode X executor: Grok Build CLI over ACP stdio.

The Grok CLI exposes the Agent Client Protocol as JSON-RPC 2.0 NDJSON.
This executor owns one short-lived ACP connection per AnimaWorks turn while
persisting Grok's session ID for conversational triggers.
"""

import asyncio
import inspect
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.execution.base import (
    BaseExecutor,
    ExecutionResult,
    TokenUsage,
    ToolCallRecord,
    _truncate_for_record,
)
from core.execution.error_classifier import (
    FailoverReason,
    classify_llm_error_message,
    guard_key,
    provider_family_of,
)
from core.execution.rate_guard import get_rate_guard
from core.i18n import t
from core.memory.shortterm import ShortTermMemory
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger("animaworks.execution.grok_cli")

__all__ = [
    "GrokCLIExecutor",
    "GrokResultMessage",
    "is_grok_cli_available",
    "_MAX_RESUME_TURNS",
    "_clear_session_id",
    "_find_grok_binary",
    "_load_session_id",
    "_resolve_real_error",
    "_resolve_grok_model",
    "_resolve_session_type",
    "_save_session_id",
    "_session_id_path",
]

_GROK_BINARY_NAMES = ("grok",)
_DEFAULT_TIMEOUT_SECONDS = 600
# Progress-aware (sidecar-style) timeouts: instead of one absolute wall-clock
# limit that kills healthy long-running tasks, we bound the *idle* gap between
# ACP stream events. Every event (thinking chunk, message chunk, tool update,
# response) counts as progress and resets the idle clock. A generous absolute
# hard cap remains only as a runaway backstop.
_IDLE_TIMEOUT_SECONDS = 900  # max silence between stream events before kill
_HARD_CAP_SECONDS = 14400  # 4h absolute backstop against a wedged child
# ACP NDJSON lines carry whole tool outputs / context blobs in one line;
# asyncio's default 64KiB StreamReader limit truncates them (LimitOverrunError).
_STDOUT_LIMIT_BYTES = 16 * 1024 * 1024
_GRACEFUL_KILL_WAIT = 3.0
_MAX_RESUME_TURNS = 10
_RESUMABLE_TRIGGERS = frozenset({"chat"})
_AUTH_ERROR_WORDS = (
    "auth",
    "login",
    "unauthorized",
    "unauthenticated",
    "authentication",
    "credential",
)
_REAL_ERROR_LOG_BYTES = 256 * 1024
_REAL_ERROR_CLOCK_SKEW_SECONDS = 5.0
_REAL_ERROR_MESSAGES = frozenset({"shell.turn.inference_failed", "turn.terminal_failure"})


def _find_grok_binary() -> str | None:
    """Return the Grok CLI path, or ``None`` when it is unavailable."""
    for name in _GROK_BINARY_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def is_grok_cli_available() -> bool:
    """Return whether the Grok CLI is available on ``PATH``."""
    return _find_grok_binary() is not None


def _resolve_grok_model(model: str) -> str:
    """Strip AnimaWorks' ``grok/`` provider prefix for the CLI."""
    return model[len("grok/") :] if model.startswith("grok/") else model


def _resolve_session_type(trigger: str) -> str:
    """Normalize human-message triggers to the shared ``chat`` type."""
    if not trigger or trigger.startswith(("chat", "message")):
        return "chat"
    return trigger.split(":", 1)[0]


def _session_id_path(anima_dir: Path, session_type: str, thread_id: str = "default") -> Path:
    """Return the per-trigger, per-thread Grok session state path."""
    return anima_dir / "shortterm" / session_type / thread_id / "grok_session_id.txt"


def _save_session_id(
    anima_dir: Path,
    session_id: str,
    session_type: str,
    thread_id: str = "default",
    turn_count: int = 1,
) -> None:
    path = _session_id_path(anima_dir, session_type, thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{session_id}\n{turn_count}", encoding="utf-8")


def _load_session_id(
    anima_dir: Path,
    session_type: str,
    thread_id: str = "default",
) -> tuple[str | None, int]:
    """Load a session ID and turn count, accepting a legacy one-line file."""
    path = _session_id_path(anima_dir, session_type, thread_id)
    if not path.is_file():
        return (None, 0)
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return (None, 0)
    session_id = lines[0].strip() if lines else ""
    if not session_id:
        return (None, 0)
    try:
        turn_count = int(lines[1].strip()) if len(lines) > 1 else 0
    except (ValueError, IndexError):
        turn_count = 0
    return (session_id, turn_count)


def _clear_session_id(anima_dir: Path, session_type: str, thread_id: str = "default") -> None:
    _session_id_path(anima_dir, session_type, thread_id).unlink(missing_ok=True)


def _resolve_real_error(
    pid: int,
    session_id: str,
    since_ts: float,
    log_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the newest matching Grok CLI failure context, if available.

    The CLI only exposes detailed provider failures in its unified log.  Read
    at most the tail of that file and fail open on every format or I/O error so
    an unofficial log format change cannot turn into an executor failure.
    """
    path = log_path or Path.home() / ".grok" / "logs" / "unified.jsonl"
    try:
        with path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            start = max(0, size - _REAL_ERROR_LOG_BYTES)
            log_file.seek(start)
            tail = log_file.read(_REAL_ERROR_LOG_BYTES)
        if start:
            newline = tail.find(b"\n")
            tail = tail[newline + 1 :] if newline >= 0 else b""

        threshold = float(since_ts) - _REAL_ERROR_CLOCK_SKEW_SECONDS
        newest: tuple[float, dict[str, Any]] | None = None
        for raw_line in tail.splitlines():
            try:
                entry = json.loads(raw_line)
                if not isinstance(entry, dict) or entry.get("msg") not in _REAL_ERROR_MESSAGES:
                    continue
                pid_matches = entry.get("pid") == pid
                sid_matches = bool(session_id) and entry.get("sid") == session_id
                if not (pid_matches or sid_matches):
                    continue
                raw_ts = entry.get("ts")
                if not isinstance(raw_ts, str):
                    continue
                parsed_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if parsed_ts.tzinfo is None:
                    parsed_ts = parsed_ts.replace(tzinfo=UTC)
                timestamp = parsed_ts.timestamp()
                if timestamp < threshold:
                    continue
                ctx = entry.get("ctx")
                if not isinstance(ctx, dict) or not (ctx.get("message") or ctx.get("status_code")):
                    continue
                if newest is None or timestamp >= newest[0]:
                    newest = (timestamp, dict(ctx))
            except Exception:  # noqa: BLE001
                continue
        return newest[1] if newest is not None else None
    except Exception:  # noqa: BLE001
        logger.debug("Failed to resolve Grok provider error from unified log", exc_info=True)
        return None


def _grok_error_metadata(message: str, model: str) -> dict[str, Any]:
    """Classify a Grok failure, report fleet blocks, and return chunk metadata."""
    reason, _hint = classify_llm_error_message(message)
    guarded_reasons = {
        FailoverReason.RATE_LIMIT,
        FailoverReason.OVERLOADED,
        FailoverReason.QUOTA_EXHAUSTED,
    }
    if reason in guarded_reasons:
        try:
            guard = get_rate_guard()
            cfg = guard.config
            block_seconds = (
                cfg.quota_block_seconds if reason is FailoverReason.QUOTA_EXHAUSTED else cfg.default_block_seconds
            )
            guard.report_block(
                guard_key(provider_family_of(model), "grok"),
                block_seconds,
                reason.value,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to report Grok error to rate guard", exc_info=True)

    # At this point the short-lived Grok ACP process has already failed and
    # exited.  Unlike an in-flight SDK notification, every such failure is a
    # terminal outcome for this attempt, including otherwise retryable classes.
    return {"terminal": True, "reason": reason.value}


@dataclass
class GrokResultMessage:
    """Session metadata adapter required by :class:`BaseExecutor`."""

    num_turns: int = 0
    session_id: str = ""
    usage: dict[str, int] | None = None


@dataclass
class _RunState:
    """Mutable outcome populated while the ACP async generator is consumed."""

    session_id: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_records: list[ToolCallRecord] = field(default_factory=list)
    full_text: str = ""
    stderr: str = ""
    error_text: str = ""
    failed: bool = False
    resume_failed: bool = False
    interrupted: bool = False
    completed: bool = False
    # Last per-response usage from ACP ``usage_update`` (current context size).
    # Distinct from ``usage`` which is the cumulative PromptUsage for cost logs.
    last_response_usage: dict[str, int] | None = None


class _ACPError(RuntimeError):
    """An error response returned by the ACP agent."""

    def __init__(self, method: str, error: Any) -> None:
        self.method = method
        self.error = error
        if isinstance(error, dict):
            detail = error.get("message") or json.dumps(error, ensure_ascii=False)
        else:
            detail = str(error)
        super().__init__(f"{method}: {detail}")


class GrokCLIExecutor(BaseExecutor):
    """Execute Grok Build CLI turns through ACP stdio (Mode X)."""

    @property
    def supports_streaming(self) -> bool:  # noqa: D102
        return True

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        tool_registry: list[str] | None = None,
        personal_tools: dict[str, str] | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(model_config, anima_dir, interrupt_event=interrupt_event)
        self._tool_registry = tool_registry or []
        self._personal_tools = personal_tools or {}
        self._workspace = anima_dir / ".grok-workspace"

    def _ensure_workspace(self) -> None:
        self._workspace.mkdir(parents=True, exist_ok=True)

    def _write_sandbox_config(self) -> bool:
        """Write the per-turn Grok sandbox profile and return whether to use it.

        Grok applies one profile to both its model-facing shell and child MCP
        server.  Unlike Mode C's shell profile, this profile deliberately does
        not include ``shell_internal_deny_paths()``: the MCP child needs direct
        access to Anima state and vector data.  Its deny rules therefore match
        Mode C's MCP profile and contain only explicit ``file_roots_denied``.

        Protection against writes to ``permissions.json`` is intentionally
        omitted because Grok CLI 0.2.102 silently fails to apply Landlock when
        ``read_only`` contains a file path.  As a residual risk, an Anima can
        therefore rewrite its own ``permissions.json``.
        """
        if (self._model_config.extra_keys or {}).get("grok_sandbox") == "off":
            return False

        from core.config.models import load_permissions
        from core.file_access_policy import (
            effective_write_roots,
            resolve_effective_denied_roots,
            shared_tool_cache_write_root,
        )

        permissions_config = load_permissions(self._anima_dir)
        write_roots = effective_write_roots(
            self._anima_dir,
            permissions_config.file_roots,
            self._task_cwd,
        )
        denied_roots = list(
            resolve_effective_denied_roots(
                self._anima_dir,
                getattr(permissions_config, "file_roots_denied", []),
            )
        )
        if not denied_roots and "/" in permissions_config.file_roots:
            return False

        read_write = [str(self._anima_dir.resolve()), *(str(root) for root in write_roots)]
        # External-tool caches (Chatwork/Slack message DBs and the identity
        # map) sit outside the Anima directory; without write access even a
        # plain inbox read fails with EROFS.
        tool_cache_root = shared_tool_cache_write_root(self._anima_dir)
        if tool_cache_root is not None:
            read_write.append(str(tool_cache_root))

        deny = [str(Path(root).resolve()) for root in denied_roots]

        def toml_array(values: list[str]) -> str:
            # JSON double-quoted strings use the same escapes required by TOML
            # for quotes, backslashes, and control characters.
            return ", ".join(json.dumps(value, ensure_ascii=False) for value in values)

        config = (
            "[profiles.animaworks]\n"
            'extends = "workspace"\n'
            f"read_write = [{toml_array(read_write)}]\n"
            f"deny = [{toml_array(deny)}]\n"
        )
        config_dir = self._workspace / ".grok"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "sandbox.toml").write_text(config, encoding="utf-8")
        return True

    def _log_sandbox_status(self) -> None:
        """Best-effort log of the newest Grok sandbox event for this workspace."""
        events_path = Path.home() / ".grok" / "sandbox-events.jsonl"
        try:
            with events_path.open("rb") as events_file:
                events_file.seek(0, os.SEEK_END)
                size = events_file.tell()
                events_file.seek(max(0, size - 64 * 1024))
                lines = events_file.read().splitlines()
        except (FileNotFoundError, OSError):
            return

        workspace = str(self._workspace.resolve())
        parsed_any = False
        latest: dict[str, Any] | None = None
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            parsed_any = True
            if event.get("workspace") == workspace:
                latest = event
                break

        if latest is not None and latest.get("event_type") == "ProfileApplied" and latest.get("enforced") is True:
            logger.debug("Grok sandbox enforced for workspace %s", workspace)
        elif parsed_any:
            logger.warning("Grok sandbox not enforced (Landlock); deny paths remain bwrap-enforced")

    def _build_command(self) -> list[str]:
        binary = _find_grok_binary()
        if not binary:
            return []
        return [
            binary,
            "agent",
            "--always-approve",
            "--no-leader",
            "-m",
            _resolve_grok_model(self._model_config.model),
            "stdio",
        ]

    def _mcp_servers(self) -> list[dict[str, Any]]:
        from core.paths import PROJECT_DIR

        return [
            {
                "name": "aw",
                "command": sys.executable,
                "args": ["-m", "core.mcp.server"],
                # ACP expects EnvVariable objects, not a plain mapping
                # (a dict fails schema validation: "did not match any variant
                # of untagged enum McpServer").
                # The MCP server runs with ONLY these variables — without the
                # embed/vector/rerank URLs it silently falls back to loading
                # SentenceTransformer / CrossEncoder models in-process
                # (2026-07-17 OOM; 2026-08-07 CrossEncoder load storm).
                "env": [
                    {"name": "ANIMAWORKS_ANIMA_DIR", "value": str(self._anima_dir)},
                    {"name": "ANIMAWORKS_PROJECT_DIR", "value": str(PROJECT_DIR)},
                    {"name": "PYTHONPATH", "value": str(PROJECT_DIR)},
                    {"name": "PATH", "value": os.environ.get("PATH", "/usr/bin:/bin")},
                    *(
                        {"name": key, "value": os.environ[key]}
                        for key in (
                            "ANIMAWORKS_EMBED_URL",
                            "ANIMAWORKS_VECTOR_URL",
                            "ANIMAWORKS_RERANK_URL",
                        )
                        if os.environ.get(key)
                    ),
                ],
            }
        ]

    @staticmethod
    def _parse_ndjson_event(line: str | bytes) -> dict[str, Any] | None:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        stripped = line.strip()
        if not stripped:
            return None
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON Grok ACP output: %s", stripped[:200])
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _tool_name(update: dict[str, Any]) -> str:
        meta = update.get("_meta")
        xai_tool = meta.get("x.ai/tool") if isinstance(meta, dict) else None
        name = xai_tool.get("name") if isinstance(xai_tool, dict) else None
        name = str(name or update.get("title") or update.get("toolName") or "unknown")
        for prefix in ("mcp__aw__", "mcp_aw_"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        return name

    @staticmethod
    def _summarize(value: Any, max_len: int = 500) -> str:
        if value in (None, "", {}, []):
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(value)
        return _truncate_for_record(text, max_len)

    @classmethod
    def _tool_record(
        cls,
        start: dict[str, Any],
        update: dict[str, Any],
    ) -> ToolCallRecord:
        # Some Grok versions repeat x.ai/tool metadata only on the terminal
        # update, while others include it only on the initial tool_call.
        name = cls._tool_name(update)
        if name == "unknown":
            name = cls._tool_name(start)
        raw_output = update.get("rawOutput")
        if raw_output in (None, ""):
            raw_output = update.get("content")
        return ToolCallRecord(
            tool_name=name,
            tool_id=str(update.get("toolCallId") or start.get("toolCallId") or ""),
            input_summary=cls._summarize(start.get("rawInput")),
            result_summary=cls._summarize(raw_output),
            is_error=update.get("status") == "failed",
        )

    @staticmethod
    def _usage_from_result(result: Any) -> TokenUsage:
        if not isinstance(result, dict):
            return TokenUsage()
        meta = result.get("_meta")
        meta = meta if isinstance(meta, dict) else {}
        usage = meta.get("usage")
        usage = usage if isinstance(usage, dict) else meta
        return TokenUsage(
            input_tokens=int(usage.get("inputTokens", 0) or 0),
            output_tokens=int(usage.get("outputTokens", 0) or 0),
            cache_read_tokens=int(usage.get("cachedReadTokens", 0) or 0),
            cache_write_tokens=0,
        )

    @staticmethod
    def _response_usage_from_update(update: dict[str, Any]) -> dict[str, int]:
        """Extract per-response usage from an ACP ``usage_update`` notification.

        Grok's wire format uses camelCase; some paths may emit snake_case.
        Accept both.  Prefer ``update["usage"]``; fall back to ``update`` itself.
        """
        payload = update.get("usage")
        if not isinstance(payload, dict):
            payload = update

        def _pick(*keys: str) -> int:
            for key in keys:
                if key not in payload or payload[key] is None:
                    continue
                try:
                    return int(payload[key] or 0)
                except (TypeError, ValueError):
                    continue
            return 0

        return {
            "input_tokens": _pick("inputTokens", "input_tokens"),
            "output_tokens": _pick("outputTokens", "output_tokens"),
            "thought_tokens": _pick("thoughtTokens", "thought_tokens"),
            "cache_read_input_tokens": _pick(
                "cachedReadTokens",
                "cache_read_input_tokens",
                "cache_read_tokens",
            ),
            "cache_creation_input_tokens": _pick(
                "cachedWriteTokens",
                "cache_creation_input_tokens",
                "cache_write_tokens",
            ),
            "total_tokens": _pick("totalTokens", "total_tokens"),
        }

    @staticmethod
    def _response_usage_from_meta(meta: dict[str, Any]) -> dict[str, int] | None:
        """Extract last-call context size from prompt result ``_meta`` top-level fields.

        Grok CLI 0.2.x (observed on 0.2.118) does not emit ``usage_update``.
        Instead, ``result._meta.inputTokens`` is the final model call's full
        prompt size (cache-inclusive = current context occupancy), while
        nested ``_meta.usage.inputTokens`` is cumulative spend across all
        internal calls.  Because top-level input is already full, cache
        fields must be stored as 0 to avoid double-counting in
        :meth:`ContextTracker.update_from_message_start`.
        """
        raw = meta.get("inputTokens")
        if raw is None:
            raw = meta.get("input_tokens")
        if raw is None:
            return None
        try:
            input_tokens = int(raw or 0)
        except (TypeError, ValueError):
            return None
        return {
            "input_tokens": input_tokens,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    @staticmethod
    async def _write_json(proc: asyncio.subprocess.Process, message: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("Grok ACP stdin is unavailable")
        proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def _send_request(
        self,
        proc: asyncio.subprocess.Process,
        request_id: int,
        method: str,
        params: dict[str, Any],
    ) -> None:
        await self._write_json(
            proc,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )

    async def _answer_server_request(
        self,
        proc: asyncio.subprocess.Process,
        message: dict[str, Any],
    ) -> None:
        """Defensively approve agent-to-client permission requests."""
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        options = params.get("options")
        options = options if isinstance(options, list) else []
        selected: dict[str, Any] | None = None
        for option in options:
            if not isinstance(option, dict):
                continue
            kind = str(option.get("kind", "")).lower()
            name = str(option.get("name", "")).lower()
            if "allow" in kind or "allow" in name or "approve" in name:
                selected = option
                break
        if selected is None and options and isinstance(options[0], dict):
            selected = options[0]

        if selected is not None:
            option_id = selected.get("optionId", selected.get("id", ""))
            result: dict[str, Any] = {"outcome": {"outcome": "selected", "optionId": option_id}}
        else:
            # ``--always-approve`` normally makes this unreachable.  Returning
            # selected still chooses the permissive path for implementations
            # that omit an options array.
            result = {"outcome": {"outcome": "selected", "optionId": "allow"}}
        await self._write_json(
            proc,
            {"jsonrpc": "2.0", "id": message["id"], "result": result},
        )

    async def _read_response(
        self,
        proc: asyncio.subprocess.Process,
        request_id: int,
        method: str,
    ) -> dict[str, Any]:
        """Read until the response for *request_id*, servicing server requests."""
        if proc.stdout is None:
            raise RuntimeError("Grok ACP stdout is unavailable")
        while True:
            if self._check_interrupted():
                raise asyncio.CancelledError
            line = await asyncio.wait_for(proc.stdout.readline(), _IDLE_TIMEOUT_SECONDS)
            if not line:
                raise _ACPError(method, "unexpected EOF")
            message = self._parse_ndjson_event(line)
            if message is None:
                continue
            if "id" in message and "method" in message:
                await self._answer_server_request(proc, message)
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise _ACPError(method, message["error"])
            result = message.get("result", {})
            return result if isinstance(result, dict) else {}

    async def _cancel_session(
        self,
        proc: asyncio.subprocess.Process,
        session_id: str,
        request_id: int,
    ) -> None:
        if not session_id or proc.returncode is not None:
            return
        try:
            await self._send_request(
                proc,
                request_id,
                "session/cancel",
                {"sessionId": session_id},
            )
        except (BrokenPipeError, ConnectionError, RuntimeError):
            logger.debug("Could not send Grok ACP session/cancel", exc_info=True)

    async def _kill_process(
        self,
        proc: asyncio.subprocess.Process,
        timeout: float = _GRACEFUL_KILL_WAIT,
    ) -> None:
        """Terminate the child, then escalate to SIGKILL after *timeout*.

        The child is spawned with ``os.setsid()``, so signalling the process
        alone leaves its own descendants (the bash tool's shells and whatever
        they spawned) running — they get reparented to ``systemd --user`` and
        can burn CPU indefinitely. Signal the whole process group instead.
        """
        if proc.returncode is not None:
            return
        # Captured before the child dies: getpgid() fails once it is reaped.
        pgid = self._process_group_of(proc)
        try:
            try:
                sent = proc.send_signal(signal.SIGTERM)
                if inspect.isawaitable(sent):  # accommodates asyncio test doubles
                    await sent
            except ProcessLookupError:
                return
            self._signal_process_group(pgid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
                if proc.returncode is not None:
                    return
            except TimeoutError:
                pass
            try:
                killed = proc.kill()
                if inspect.isawaitable(killed):  # accommodates asyncio test doubles
                    await killed
            except ProcessLookupError:
                return
            try:
                await proc.wait()
            except Exception:  # noqa: BLE001
                logger.debug("Failed waiting for killed Grok CLI", exc_info=True)
        finally:
            # Always sweep the group: the CLI exiting says nothing about the
            # shells it spawned, and orphans here have burned hours of CPU.
            self._signal_process_group(pgid, signal.SIGKILL)

    @staticmethod
    def _process_group_of(proc: asyncio.subprocess.Process) -> int | None:
        """Return the child's process group id, or None when unavailable.

        Never returns our own group — signalling that would kill the daemon.
        """
        if os.name != "posix":
            return None
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError, TypeError, OSError):
            return None
        if pgid in (os.getpgrp(), 0, 1):
            return None
        return pgid

    @staticmethod
    def _signal_process_group(pgid: int | None, sig: signal.Signals) -> None:
        """Best-effort signal to the child's whole process group."""
        if pgid is None:
            return
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    @staticmethod
    async def _drain_stderr(proc: asyncio.subprocess.Process) -> str:
        if proc.stderr is None:
            return ""
        data = await proc.stderr.read()
        if isinstance(data, str):
            return data
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _auth_error(text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in _AUTH_ERROR_WORDS)

    def _translated_error(self, detail: str, *, timed_out: bool = False) -> str:
        if timed_out:
            return t("grok_cli.timeout", timeout=_IDLE_TIMEOUT_SECONDS)
        if self._auth_error(detail):
            return t("grok_cli.not_authenticated")
        return f"[Grok CLI Error: {detail}]"

    async def _run_acp(
        self,
        prompt: str,
        system_prompt: str,
        state: _RunState,
        *,
        resume_session_id: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run one ACP connection and yield converted non-terminal events."""
        started_at = time.time()
        cmd = self._build_command()
        sandbox_enabled = self._write_sandbox_config()
        env = os.environ.copy()
        from core.execution.github_identity import resolve_github_token_env

        env.update(resolve_github_token_env(self._anima_dir))
        if sandbox_enabled:
            env["GROK_SANDBOX"] = "animaworks"
        else:
            env.pop("GROK_SANDBOX", None)
        # Stop the Grok engine from scanning host ~/.claude/skills as native
        # skills — skill presentation is unified under animaworks' own skill
        # catalog (core/skills/index.py external roots).  ~/.grok/skills has
        # no off switch; keep that host dir empty.  Applies in both sandbox paths.
        env["GROK_CLAUDE_SKILLS_ENABLED"] = "false"

        proc: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[str] | None = None
        pending_tools: dict[str, dict[str, Any]] = {}
        next_id = 1
        pty_master_fd: int | None = None
        pty_slave_fd: int | None = None
        pty_spawn_kwargs: dict[str, Any] = {}

        try:
            if sandbox_enabled:
                try:
                    import fcntl
                    import pty
                    import termios

                    pty_master_fd, pty_slave_fd = pty.openpty()
                    slave_fd = pty_slave_fd

                    def attach_controlling_tty() -> None:
                        os.setsid()
                        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

                    pty_spawn_kwargs = {
                        "preexec_fn": attach_controlling_tty,
                        "pass_fds": (slave_fd,),
                    }
                except (ImportError, AttributeError, OSError):
                    logger.warning(
                        "Could not prepare a controlling PTY for Grok sandbox; continuing without Landlock",
                        exc_info=True,
                    )
                    for fd in (pty_slave_fd, pty_master_fd):
                        if fd is not None:
                            os.close(fd)
                    pty_master_fd = None
                    pty_slave_fd = None
                    pty_spawn_kwargs = {}

            spawn_kwargs = {
                "stdin": asyncio.subprocess.PIPE,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "cwd": str(self._workspace),
                "limit": _STDOUT_LIMIT_BYTES,
                "env": env,
            }
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    **spawn_kwargs,
                    **pty_spawn_kwargs,
                )
            except (OSError, subprocess.SubprocessError, TypeError) as exc:
                if not pty_spawn_kwargs or isinstance(exc, FileNotFoundError):
                    raise
                logger.warning(
                    "Could not attach a controlling PTY for Grok sandbox; continuing without Landlock",
                    exc_info=True,
                )
                for fd in (pty_slave_fd, pty_master_fd):
                    if fd is not None:
                        os.close(fd)
                pty_master_fd = None
                pty_slave_fd = None
                pty_spawn_kwargs = {}
                proc = await asyncio.create_subprocess_exec(*cmd, **spawn_kwargs)

            if pty_slave_fd is not None:
                os.close(pty_slave_fd)
                pty_slave_fd = None
            stderr_task = asyncio.create_task(self._drain_stderr(proc))

            async with asyncio.timeout(_HARD_CAP_SECONDS):
                await self._send_request(
                    proc,
                    next_id,
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {
                            "fs": {"readTextFile": False, "writeTextFile": False},
                            "terminal": False,
                        },
                    },
                )
                await self._read_response(proc, next_id, "initialize")
                if sandbox_enabled:
                    self._log_sandbox_status()
                next_id += 1

                common_session_params = {
                    "cwd": str(self._workspace.resolve()),
                    "mcpServers": self._mcp_servers(),
                }
                if resume_session_id:
                    method = "session/load"
                    session_params = {
                        "sessionId": resume_session_id,
                        **common_session_params,
                    }
                else:
                    method = "session/new"
                    session_params = dict(common_session_params)
                    session_params["_meta"] = {"systemPromptOverride": system_prompt}

                await self._send_request(proc, next_id, method, session_params)
                session_result = await self._read_response(proc, next_id, method)
                state.session_id = str(session_result.get("sessionId") or resume_session_id or "")
                if not state.session_id:
                    raise _ACPError(method, "response did not contain sessionId")
                next_id += 1

                await self._send_request(
                    proc,
                    next_id,
                    "session/prompt",
                    {
                        "sessionId": state.session_id,
                        "prompt": [{"type": "text", "text": prompt}],
                    },
                )
                prompt_request_id = next_id
                next_id += 1

                if proc.stdout is None:
                    raise RuntimeError("Grok ACP stdout is unavailable")
                while True:
                    if self._check_interrupted():
                        state.interrupted = True
                        await self._cancel_session(proc, state.session_id, next_id)
                        break

                    line = await asyncio.wait_for(proc.stdout.readline(), _IDLE_TIMEOUT_SECONDS)
                    if not line:
                        raise _ACPError("session/prompt", "unexpected EOF")
                    message = self._parse_ndjson_event(line)
                    if message is None:
                        continue

                    if "id" in message and "method" in message:
                        await self._answer_server_request(proc, message)
                        continue

                    if message.get("id") == prompt_request_id:
                        if "error" in message:
                            raise _ACPError("session/prompt", message["error"])
                        result = message.get("result", {})
                        # Cumulative PromptUsage for cost logs (nested _meta.usage).
                        state.usage = self._usage_from_result(result)
                        if isinstance(result, dict):
                            meta = result.get("_meta")
                            if isinstance(meta, dict):
                                if meta.get("sessionId"):
                                    state.session_id = str(meta["sessionId"])
                                # Priority for context ring:
                                # (1) usage_update during stream (if CLI emits it)
                                # (2) _meta top-level inputTokens = last-call full prompt
                                # (3) cumulative usage (handled later when still None)
                                if state.last_response_usage is None:
                                    from_meta = self._response_usage_from_meta(meta)
                                    if from_meta is not None:
                                        state.last_response_usage = from_meta
                        state.completed = True
                        break

                    if message.get("method") != "session/update":
                        continue
                    params = message.get("params")
                    params = params if isinstance(params, dict) else {}
                    update = params.get("update")
                    update = update if isinstance(update, dict) else {}
                    kind = update.get("sessionUpdate")

                    if kind == "agent_thought_chunk":
                        content = update.get("content")
                        text = content.get("text", "") if isinstance(content, dict) else ""
                        if text:
                            yield {"type": "thinking_delta", "text": str(text)}
                    elif kind == "agent_message_chunk":
                        content = update.get("content")
                        text = content.get("text", "") if isinstance(content, dict) else ""
                        if text:
                            text = str(text)
                            state.full_text += text
                            yield {"type": "text_delta", "text": text}
                    elif kind == "usage_update":
                        # Per-response context size (xAI extension).  Last wins.
                        # Log the raw payload once so field names can be verified
                        # against real wire traffic without spamming info logs.
                        if state.last_response_usage is None:
                            logger.debug("Grok ACP usage_update payload: %s", update)
                        state.last_response_usage = self._response_usage_from_update(update)
                    elif kind == "tool_call":
                        tool_id = str(update.get("toolCallId") or "")
                        pending_tools[tool_id] = update
                        yield {
                            "type": "tool_start",
                            "tool_name": self._tool_name(update),
                            "tool_id": tool_id,
                            "tool_detail": self._summarize(update.get("rawInput")),
                        }
                    elif kind == "tool_call_update" and update.get("status") in {
                        "completed",
                        "failed",
                    }:
                        tool_id = str(update.get("toolCallId") or "")
                        start = pending_tools.pop(tool_id, {"toolCallId": tool_id})
                        record = self._tool_record(start, update)
                        state.tool_records.append(record)
                        yield {
                            "type": "tool_end",
                            "tool_name": record.tool_name,
                            "tool_id": record.tool_id,
                            "result": record.result_summary,
                            "is_error": record.is_error,
                        }

        except asyncio.CancelledError:
            # A pre-existing/in-flight AnimaWorks interrupt is a normal
            # terminal path.  Task cancellation from the event loop is still
            # safely cleaned up by the same finally block.
            if self._check_interrupted():
                state.interrupted = True
                if proc is not None:
                    await self._cancel_session(proc, state.session_id, next_id)
            else:
                raise
        except TimeoutError:
            state.failed = True
            state.resume_failed = resume_session_id is not None
            state.error_text = self._translated_error("", timed_out=True)
        except FileNotFoundError:
            state.failed = True
            state.error_text = t("grok_cli.not_installed")
        except _ACPError as exc:
            state.failed = True
            state.resume_failed = resume_session_id is not None and not self._auth_error(str(exc))
            state.error_text = self._translated_error(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Grok CLI ACP execution failed")
            state.failed = True
            state.resume_failed = resume_session_id is not None and not self._auth_error(str(exc))
            state.error_text = self._translated_error(str(exc))
        finally:
            if proc is not None and proc.returncode is None:
                await self._kill_process(proc)
            if stderr_task is not None:
                try:
                    state.stderr = await stderr_task
                except Exception:  # noqa: BLE001
                    logger.debug("Failed draining Grok CLI stderr", exc_info=True)

            if state.stderr and state.failed and self._auth_error(state.stderr):
                state.error_text = t("grok_cli.not_authenticated")
                state.resume_failed = False
            elif state.stderr and state.failed and "unexpected EOF" in state.error_text:
                # The child died before answering (e.g. grok 1.0.13 refusing to
                # start its sandbox); the reason is only on stderr.
                state.error_text = self._translated_error(
                    f"{state.error_text}: {state.stderr.strip()[-500:]}"
                )
            # A successful prompt response is terminal even though we then
            # terminate the long-lived stdio server ourselves.  Only classify
            # an independently failed child exit as an execution error.
            if proc is not None and proc.returncode not in (None, 0) and not state.completed and not state.interrupted:
                state.failed = True
                if not state.error_text:
                    detail = state.stderr[:500] or f"exit {proc.returncode}"
                    state.error_text = self._translated_error(f"exit {proc.returncode}: {detail}")
            if state.failed and proc is not None:
                real_error = _resolve_real_error(proc.pid, state.session_id, started_at)
                if real_error is not None and real_error.get("message"):
                    state.error_text = self._translated_error(str(real_error["message"]))
            for fd in (pty_slave_fd, pty_master_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        logger.debug("Failed closing Grok sandbox PTY fd", exc_info=True)

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run a Grok ACP turn and collect its streaming events."""
        done: dict[str, Any] | None = None
        terminal_error: dict[str, Any] | None = None
        async for event in self.execute_streaming(
            system_prompt,
            prompt,
            tracker or ContextTracker(model=self._model_config.model),
            images=images,
            prior_messages=prior_messages,
            trigger=trigger,
            thread_id=thread_id,
        ):
            if event.get("type") == "done":
                done = event
            elif event.get("type") == "error" and event.get("terminal") is True:
                terminal_error = event

        if done is None:
            done = self._done_event("", [], TokenUsage(), "", 0)
        records = [ToolCallRecord(**record) for record in done["tool_call_records"]]
        usage_dict = done["usage"]
        result_message = done["result_message"]
        return ExecutionResult(
            text=str((terminal_error or {}).get("message") or done["full_text"]),
            result_message=result_message,
            tool_call_records=records,
            usage=TokenUsage(**usage_dict),
            session_rotated=bool(done.get("session_rotated", False)),
            session_rotation_pending=bool(done.get("session_rotation_pending", False)),
            error=terminal_error is not None,
            reason=str((terminal_error or {}).get("reason") or ""),
        )

    @staticmethod
    def _done_event(
        full_text: str,
        records: list[ToolCallRecord],
        usage: TokenUsage,
        session_id: str,
        num_turns: int,
        *,
        session_rotated: bool = False,
        session_rotation_pending: bool = False,
        stop_kind: str = "normal",
    ) -> dict[str, Any]:
        usage_dict = usage.to_dict()
        return {
            "type": "done",
            "full_text": str(full_text),
            "result_message": GrokResultMessage(
                num_turns=int(num_turns),
                session_id=str(session_id),
                usage=usage_dict,
            ),
            "tool_call_records": [asdict(record) for record in records],
            "usage": usage_dict,
            "stop_kind": stop_kind,
            "session_rotated": session_rotated,
            "session_rotation_pending": session_rotation_pending,
        }

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        trigger: str = "",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream converted Grok ACP events, ending with exactly one ``done``."""
        if self._check_interrupted():
            yield self._done_event("[Session interrupted by user]", [], TokenUsage(), "", 0, stop_kind="interrupted")
            return

        if not _find_grok_binary():
            text = t("grok_cli.not_installed")
            yield {"type": "text_delta", "text": text}
            yield self._done_event(text, [], TokenUsage(), "", 0)
            return

        self._ensure_workspace()
        session_type = _resolve_session_type(trigger)
        is_resumable = session_type in _RESUMABLE_TRIGGERS
        resume_session_id: str | None = None
        turn_count = 0
        session_rotated = False
        if is_resumable:
            resume_session_id, turn_count = _load_session_id(self._anima_dir, session_type, thread_id)
            if resume_session_id and turn_count >= _MAX_RESUME_TURNS:
                _clear_session_id(self._anima_dir, session_type, thread_id)
                resume_session_id = None
                turn_count = 0
                session_rotated = True

        state = _RunState()
        async for event in self._run_acp(
            prompt,
            system_prompt,
            state,
            resume_session_id=resume_session_id,
        ):
            yield event

        if state.resume_failed and resume_session_id and not state.interrupted:
            logger.warning(
                "Grok session/load failed for %s; retrying with a fresh session",
                resume_session_id[:12],
            )
            _clear_session_id(self._anima_dir, session_type, thread_id)
            state = _RunState()
            session_rotated = True
            async for event in self._run_acp(
                prompt,
                system_prompt,
                state,
                resume_session_id=None,
            ):
                yield event

        if state.interrupted:
            if not state.full_text:
                state.full_text = "[Session interrupted by user]"
        elif state.error_text:
            error_metadata = _grok_error_metadata(
                state.error_text,
                self._model_config.model,
            )
            if error_metadata.get("terminal") is True:
                yield {
                    "type": "error",
                    "message": state.error_text,
                    **error_metadata,
                }
            else:
                if state.full_text:
                    state.full_text += "\n\n" + state.error_text
                    yield {"type": "text_delta", "text": "\n\n" + state.error_text}
                else:
                    state.full_text = state.error_text
                    yield {"type": "text_delta", "text": state.error_text}

        if state.completed and tracker is not None:
            # Context size priority:
            # (1) usage_update notification (future CLI; cache fields may apply)
            # (2) _meta top-level inputTokens (current CLI 0.2.x; already full)
            # (3) cumulative _meta.usage (legacy fallback; inflates the ring)
            if state.last_response_usage is not None:
                tracker.update_from_message_start(
                    {
                        "input_tokens": state.last_response_usage.get("input_tokens", 0),
                        "cache_read_input_tokens": state.last_response_usage.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": state.last_response_usage.get("cache_creation_input_tokens", 0),
                    }
                )
            else:
                tracker.update_from_usage(state.usage.to_dict())

        new_turn = 0
        if state.completed and state.session_id and is_resumable:
            new_turn = 1 if session_rotated or not resume_session_id else turn_count + 1
            _save_session_id(
                self._anima_dir,
                state.session_id,
                session_type,
                thread_id,
                new_turn,
            )
        elif state.completed:
            new_turn = 1

        rotation_pending = is_resumable and not session_rotated and new_turn >= _MAX_RESUME_TURNS
        yield self._done_event(
            state.full_text,
            state.tool_records,
            state.usage,
            state.session_id,
            new_turn,
            session_rotated=session_rotated,
            session_rotation_pending=rotation_pending,
            stop_kind="interrupted" if state.interrupted else "normal",
        )
