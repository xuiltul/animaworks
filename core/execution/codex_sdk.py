from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode C executor: Codex Python SDK (Codex App Server wrapper).

Runs OpenAI models via the Codex App Server as an autonomous agent.  The SDK
spawns the Codex binary and exchanges JSON-RPC notifications over stdio. Tool
safety relies on Codex's sandbox mode plus MCP integration with AnimaWorks
``core/mcp/server.py`` for permission-checked tool access.

System prompt is injected via ``model_instructions_file`` in a per-anima
CODEX_HOME directory.  Session resume uses the Codex SDK's ``thread_resume``
mechanism with thread IDs persisted to the shortterm directory.
"""

import asyncio
import inspect
import json
import logging
import os
import shutil
import sys
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.execution.base import (
    BaseExecutor,
    ExecutionResult,
    StreamDisconnectedError,
    TokenUsage,
    ToolCallRecord,
    _truncate_for_record,
)
from core.execution.session_types import is_persistent_codex_session, resolve_runtime_session_type
from core.memory.shortterm import ShortTermMemory
from core.platform.codex import default_home_dir, get_codex_executable
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger("animaworks.execution.codex_sdk")

__all__ = ["CodexSDKExecutor", "clear_codex_thread_id", "clear_codex_thread_ids", "is_codex_sdk_available"]

RESUME_TIMEOUT_SEC = 15.0
_BACKGROUND_EVENT_IDLE_TIMEOUT_SEC = 45.0
_FOREGROUND_EVENT_IDLE_TIMEOUT_SEC = 120.0

# asyncio.StreamReader default limit is 64 KB.  Codex CLI may echo the full
# context (including system prompt) in a single JSONL line during thread
# resume, triggering LimitOverrunError.  Skip resume when close to this limit.
_RESUME_PROMPT_SIZE_LIMIT = 50_000
_FATAL_STDERR_PATTERNS = ("error: stream closed",)

# Increase the asyncio.StreamReader buffer limit for Codex subprocess pipes.
# The default 64 KB (2**16) is too small for large prompts that produce JSONL
# lines exceeding 64 KB on stdout (e.g., thread resume with full context echo).
# 16 MB provides ample headroom for realistic prompt sizes.
_SUBPROCESS_STREAM_LIMIT = 16 * 1024 * 1024  # 16 MB


# ── Model name helpers ───────────────────────────────────────


def _resolve_codex_model(model: str) -> str:
    """Strip supported Codex provider prefixes to get the bare CLI model name."""
    if model.startswith("codex/"):
        return model[len("codex/") :]
    if model.startswith("openai-codex/"):
        return model[len("openai-codex/") :]
    return model


def is_codex_sdk_available() -> bool:
    """Return True when ``openai_codex`` is importable."""
    try:
        import openai_codex  # noqa: F401

        return True
    except Exception:
        return False


def _is_openai_api_key(key: str) -> bool:
    """Return True if *key* looks like a genuine OpenAI API key."""
    return bool(key) and not key.startswith("sk-ant-")


@dataclass(frozen=True)
class _CodexProviderConfig:
    model: str
    provider: str
    is_azure: bool = False
    base_url: str | None = None
    api_version: str | None = None
    env_key: str = "OPENAI_API_KEY"
    wire_api: str = "responses"


def _is_codex_azure_config(model_config: ModelConfig) -> bool:
    """Return True when the resolved credential explicitly selects Azure for Codex."""
    return model_config.credential_type == "codex_azure"


def _normalize_azure_openai_base_url(base_url: str) -> str:
    """Return the Azure OpenAI Codex provider base URL with the required /openai suffix."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/openai"):
        return normalized
    return f"{normalized}/openai"


def _resolve_codex_provider_config(model_config: ModelConfig) -> _CodexProviderConfig:
    """Resolve Codex CLI provider settings from the AnimaWorks model config."""
    extra = model_config.extra_keys or {}
    model = extra.get("codex_model") or _resolve_codex_model(model_config.model)

    if not _is_codex_azure_config(model_config):
        return _CodexProviderConfig(model=model, provider="openai")

    if not model_config.api_base_url:
        raise ValueError("Codex Azure credential requires base_url for the Azure OpenAI resource")
    api_version = extra.get("api_version")
    if not api_version:
        raise ValueError("Codex Azure credential requires keys.api_version")

    return _CodexProviderConfig(
        model=model,
        provider="azure",
        is_azure=True,
        base_url=_normalize_azure_openai_base_url(model_config.api_base_url),
        api_version=api_version,
        env_key="AZURE_OPENAI_API_KEY",
        wire_api=extra.get("codex_wire_api") or "responses",
    )


def _escape_toml_string(value: str) -> str:
    """Escape a string for safe embedding in a TOML double-quoted value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _default_home_dir() -> str:
    """Return a stable HOME value for Codex child processes across platforms."""
    return default_home_dir()


def _default_path_env() -> str:
    """Return a non-empty PATH fallback for Codex child processes."""
    path_parts: list[str] = []
    executable = get_codex_executable()
    if executable:
        path_parts.append(str(Path(executable).resolve().parent))

    # Ensure child Codex sessions can resolve helper CLIs installed into the
    # same Python environment that launched AnimaWorks.
    python_bin = str(Path(sys.executable).resolve().parent)
    if python_bin:
        path_parts.append(python_bin)

    # Editable/dev installs often keep helper entry points in the project venv
    # even when the parent PATH was started from a different shell profile.
    try:
        from core.paths import PROJECT_DIR

        project_venv_bin = PROJECT_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin")
        if project_venv_bin.is_dir():
            path_parts.append(str(project_venv_bin))
    except Exception:
        logger.debug("Failed to resolve project venv bin for Codex PATH", exc_info=True)

    existing = os.environ.get("PATH")
    if existing:
        path_parts.append(existing)
    return os.pathsep.join(dict.fromkeys(part for part in path_parts if part))


# ── Session (thread) ID persistence ──────────────────────────


def _thread_id_path(anima_dir: Path, session_type: str, chat_thread_id: str = "default") -> Path:
    base = anima_dir / "shortterm" / session_type
    if chat_thread_id != "default":
        return base / chat_thread_id / "codex_thread_id.txt"
    return base / "codex_thread_id.txt"


def _save_thread_id(anima_dir: Path, thread_id: str, session_type: str, chat_thread_id: str = "default") -> None:
    p = _thread_id_path(anima_dir, session_type, chat_thread_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(thread_id, encoding="utf-8")


def _load_thread_id(anima_dir: Path, session_type: str, chat_thread_id: str = "default") -> str | None:
    p = _thread_id_path(anima_dir, session_type, chat_thread_id)
    if p.is_file():
        tid = p.read_text(encoding="utf-8").strip()
        return tid or None
    return None


def _clear_thread_id(anima_dir: Path, session_type: str, chat_thread_id: str = "default") -> None:
    p = _thread_id_path(anima_dir, session_type, chat_thread_id)
    p.unlink(missing_ok=True)


def clear_codex_thread_id(anima_dir: Path, session_type: str, chat_thread_id: str = "default") -> None:
    """Clear one resolved Codex thread ID namespace."""
    _clear_thread_id(anima_dir, session_type, chat_thread_id)


def clear_codex_thread_ids(anima_dir: Path, chat_thread_id: str = "default") -> None:
    """Clear persisted Codex thread IDs for all known runtime session types."""
    for st in ("chat", "heartbeat", "cron", "task", "inbox"):
        _clear_thread_id(anima_dir, st, chat_thread_id)


# ── Helpers ──────────────────────────────────────────────────


def _get_thread_id(thread: Any) -> str | None:
    """Safely extract the thread ID from a Codex Thread object."""
    for attr in ("id", "thread_id"):
        val = getattr(thread, attr, None)
        if val:
            return str(val)
    return None


def _resolve_session_type(trigger: str) -> str:
    return resolve_runtime_session_type(trigger)


def _event_idle_timeout_seconds(trigger: str) -> float:
    """Return max idle time between streamed Codex events before treating it as dead."""
    if trigger == "heartbeat" or trigger.startswith("inbox") or trigger.startswith("cron:"):
        return _BACKGROUND_EVENT_IDLE_TIMEOUT_SEC
    return _FOREGROUND_EVENT_IDLE_TIMEOUT_SEC


def _is_desktop_extension_codex(executable: str | None) -> bool:
    """Return True when the Codex binary comes from a desktop-app extension bundle."""
    if not executable:
        return False
    norm = executable.replace("/", "\\").lower()
    return "\\.antigravity\\extensions\\openai.chatgpt-" in norm or "\\windowsapps\\openai.codex_" in norm


def _should_prefer_cli_exec(trigger: str) -> bool:
    """Prefer direct ``codex exec`` for unstable desktop-bundled background sessions."""
    forced = os.environ.get("ANIMAWORKS_CODEX_FORCE_CLI_EXEC", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True

    is_background = (
        trigger == "heartbeat"
        or trigger.startswith("cron:")
        or trigger.startswith("inbox")
        or trigger.startswith("task:")
    )
    if not is_background or sys.platform != "win32":
        return False
    return _is_desktop_extension_codex(get_codex_executable())


def _close_stream_transport(stream: Any, stream_name: str) -> None:
    """Best-effort close for subprocess stdio objects.

    ``asyncio`` subprocess readers expose the underlying pipe transport via a
    private ``_transport`` attribute, while writers expose ``close()``.  Close
    both when available so parent-side pipe descriptors do not linger across
    repeated background runs.
    """
    if stream is None:
        return

    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.debug("Failed to close Codex subprocess %s stream", stream_name, exc_info=True)

    transport = getattr(stream, "_transport", None) or getattr(stream, "transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            logger.debug("Failed to close Codex subprocess %s transport", stream_name, exc_info=True)


def _close_subprocess_stdio(proc: asyncio.subprocess.Process) -> None:
    """Best-effort close of parent-side subprocess stdio transports."""
    _close_stream_transport(getattr(proc, "stdin", None), "stdin")
    _close_stream_transport(getattr(proc, "stdout", None), "stdout")
    _close_stream_transport(getattr(proc, "stderr", None), "stderr")


_ITEM_TYPE_ALIASES = {
    "agentMessage": "agent_message",
    "commandExecution": "command_execution",
    "mcpToolCall": "mcp_tool_call",
    "fileChange": "file_change",
    "webSearch": "web_search",
    "dynamicToolCall": "dynamic_tool_call",
    "collabAgentToolCall": "collab_agent_tool_call",
}


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return an attribute/key from SDK models, dicts, and lightweight test objects."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = _get_attr(obj, name, default)
        if value is not default:
            return value
    return default


def _get_str(obj: Any, *names: str) -> str:
    for name in names:
        value = _get_attr(obj, name, None)
        if isinstance(value, str):
            return value
    return ""


def _get_list(obj: Any, *names: str) -> list[Any]:
    for name in names:
        value = _get_attr(obj, name, None)
        if isinstance(value, list):
            return value
    return []


def _unwrap_thread_item(item: Any) -> Any:
    """Unwrap new SDK ``ThreadItem`` RootModel values while tolerating mocks."""
    root = _get_attr(item, "root", None)
    if root is not None and (_get_str(root, "type") or _get_str(root, "id")):
        return root
    return item


def _normalise_item_type(item_type: str) -> str:
    if not item_type:
        return ""
    return _ITEM_TYPE_ALIASES.get(item_type, item_type)


def _item_type(item: Any) -> str:
    return _normalise_item_type(_get_str(_unwrap_thread_item(item), "type"))


def _item_id(item: Any) -> str:
    return _get_str(_unwrap_thread_item(item), "id")


def _event_method(event: Any) -> str:
    """Return the Codex notification method, normalising old dotted test events."""
    method = _get_str(event, "method")
    if method:
        return method
    etype = _get_str(event, "type")
    if "." in etype:
        return etype.replace(".", "/")
    return etype


def _payload_looks_real(payload: Any) -> bool:
    if payload is None:
        return False
    if _get_str(payload, "thread_id", "threadId", "turn_id", "turnId", "item_id", "itemId"):
        return True
    if _get_str(payload, "delta", "message"):
        return True
    item = _get_attr(payload, "item", None)
    if item is not None and _item_type(item):
        return True
    turn = _get_attr(payload, "turn", None)
    return bool(turn is not None and _get_str(turn, "id"))


def _event_payload(event: Any) -> Any:
    payload = _get_attr(event, "payload", None)
    if _payload_looks_real(payload):
        return payload
    return event


def _payload_item_id(payload: Any) -> str:
    return _get_str(payload, "item_id", "itemId")


def _payload_delta(payload: Any) -> str:
    return _get_str(payload, "delta")


def _extract_item_text(item: Any) -> str:
    """Extract text content from a Codex completed item."""
    item = _unwrap_thread_item(item)
    content = _get_attr(item, "content", None)
    if content is not None:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                part_text = _get_str(part, "text")
                if part_text:
                    parts.append(part_text)
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
    text = _get_str(item, "text")
    if text:
        return text
    summary = _get_list(item, "summary")
    if summary:
        return "".join(str(part) for part in summary)
    return ""


def _codex_item_tool_name(item: Any, item_type: str) -> str:
    """Derive a human-readable tool name from a Codex item."""
    item = _unwrap_thread_item(item)
    item_type = _normalise_item_type(item_type)
    if item_type == "mcp_tool_call":
        server = _get_str(item, "server")
        tool = _get_str(item, "tool")
        return f"{server}/{tool}" if server else tool or "mcp_tool"
    if item_type == "command_execution":
        cmd = _get_str(item, "command")
        return cmd[:60] if cmd else "command"
    if item_type == "file_change":
        return "file_change"
    if item_type == "web_search":
        query = _get_str(item, "query")
        return f"web_search: {query[:48]}" if query else "web_search"
    return _get_str(item, "name") or item_type or "unknown"


def _item_to_tool_record(item: Any) -> ToolCallRecord | None:
    """Convert a Codex item (command_execution / mcp_tool_call) to a ``ToolCallRecord``."""
    try:
        item = _unwrap_thread_item(item)
        item_type = _item_type(item)
        tool_id = _item_id(item)
        if item_type == "mcp_tool_call":
            name = _codex_item_tool_name(item, item_type)
            input_data = _get_attr(item, "arguments", {})
            result_obj = _get_attr(item, "result", None)
            result_data = str(_get_attr(result_obj, "content", "")) if result_obj else ""
            error_obj = _get_attr(item, "error", None)
            is_error = error_obj is not None
            return ToolCallRecord(
                tool_name=name,
                tool_id=tool_id,
                input_summary=_truncate_for_record(str(input_data), 500),
                result_summary=_truncate_for_record(result_data, 500),
                is_error=is_error,
            )
        if item_type == "command_execution":
            cmd = _get_str(item, "command")
            output = _get_str(item, "aggregated_output", "aggregatedOutput")
            exit_code = _get_first_attr(item, "exit_code", "exitCode", default=None)
            is_error = exit_code is not None and exit_code != 0
            return ToolCallRecord(
                tool_name=cmd[:80] if cmd else "command",
                tool_id=tool_id,
                input_summary=_truncate_for_record(cmd, 500),
                result_summary=_truncate_for_record(output, 500),
                is_error=is_error,
            )
        if item_type == "file_change":
            changes = _get_list(item, "changes")
            detail = _format_file_changes(changes)
            return ToolCallRecord(
                tool_name="file_change",
                tool_id=tool_id,
                input_summary=_truncate_for_record(detail, 500),
                result_summary=_truncate_for_record(_get_str(item, "status") or detail, 500),
                is_error=False,
            )
        # Legacy fallback for unknown tool-like items
        name = _get_str(item, "name") or "unknown"
        input_data = _get_attr(item, "input", {})
        result_data = _get_attr(item, "output", "")
        return ToolCallRecord(
            tool_name=name,
            tool_id=tool_id,
            input_summary=_truncate_for_record(str(input_data), 500),
            result_summary=_truncate_for_record(str(result_data), 500),
        )
    except Exception:
        return None


def _extract_tool_records(items: list[Any]) -> list[ToolCallRecord]:
    records: list[ToolCallRecord] = []
    for item in items:
        itype = _item_type(item)
        if itype in ("tool_use", "command_execution", "mcp_tool_call", "file_change"):
            rec = _item_to_tool_record(item)
            if rec:
                records.append(rec)
    return records


def _synthesise_fallback(tool_records: list[ToolCallRecord]) -> str:
    """Build a short fallback text when the model produced no text output."""
    names = [r.tool_name for r in tool_records[:5]]
    suffix = ", …" if len(tool_records) > 5 else ""
    fallback = f"(completed {len(tool_records)} tool call(s): {', '.join(names)}{suffix})"
    logger.warning(
        "Codex SDK produced no text output; synthesised fallback (tools=%d)",
        len(tool_records),
    )
    return fallback


def _usage_to_dict(usage: Any) -> dict[str, int]:
    """Normalise a Codex usage object (or dict) to a plain dict."""
    if isinstance(usage, dict):
        result: dict[str, int] = {}
        key_aliases = {
            "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens", "promptTokens"),
            "output_tokens": ("output_tokens", "outputTokens", "completion_tokens", "completionTokens"),
            "cached_input_tokens": ("cached_input_tokens", "cachedInputTokens"),
            "reasoning_output_tokens": ("reasoning_output_tokens", "reasoningOutputTokens"),
            "total_tokens": ("total_tokens", "totalTokens"),
        }
        for out_key, aliases in key_aliases.items():
            for alias in aliases:
                val = usage.get(alias)
                if val is not None:
                    result[out_key] = int(val)
                    break
        return result or usage
    total_usage = _get_attr(usage, "total", None)
    if total_usage is not None and any(
        isinstance(_get_attr(total_usage, key, None), int)
        for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")
    ):
        usage = total_usage
    d: dict[str, int] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    ):
        val = getattr(usage, key, None)
        if val is not None:
            d[key] = int(val)
    return d


def _format_file_changes(changes: list[Any]) -> str:
    parts: list[str] = []
    for change in changes:
        kind = _get_attr(change, "kind", "")
        kind_text = getattr(kind, "value", kind)
        path = _get_str(change, "path")
        if kind_text or path:
            parts.append(f"{kind_text}: {path}".strip(": "))
    return "; ".join(parts[:10])


def _cli_exec_result_text(result: Any) -> str:
    """Extract text from a Codex CLI exec result payload."""
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            if parts:
                return "".join(parts)
        if "text" in result:
            return str(result.get("text", ""))
    if isinstance(result, str):
        return result
    return ""


def _cli_exec_item_to_tool_record(item: dict[str, Any]) -> ToolCallRecord | None:
    """Convert a JSON event item from `codex exec --json` into a ToolCallRecord."""
    try:
        item_type = str(item.get("type", ""))
        tool_id = str(item.get("id", ""))
        if item_type == "mcp_tool_call":
            server = str(item.get("server", ""))
            tool = str(item.get("tool", ""))
            name = f"{server}/{tool}" if server else tool or "mcp_tool"
            result_text = _cli_exec_result_text(item.get("result"))
            return ToolCallRecord(
                tool_name=name,
                tool_id=tool_id,
                input_summary=_truncate_for_record(str(item.get("arguments", {})), 500),
                result_summary=_truncate_for_record(result_text, 500),
                is_error=item.get("error") is not None,
            )
        if item_type == "command_execution":
            cmd = str(item.get("command", ""))
            output = str(item.get("aggregated_output", "") or item.get("output", ""))
            exit_code = item.get("exit_code")
            is_error = exit_code is not None and exit_code != 0
            return ToolCallRecord(
                tool_name=cmd[:80] if cmd else "command",
                tool_id=tool_id,
                input_summary=_truncate_for_record(cmd, 500),
                result_summary=_truncate_for_record(output, 500),
                is_error=is_error,
            )
    except Exception:
        return None
    return None


def _stderr_contains_fatal_signal(text: str) -> bool:
    """Return True when Codex stderr already indicates the stream is unrecoverable."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in _FATAL_STDERR_PATTERNS)


def _should_cli_exec_fallback(exc: BaseException) -> bool:
    """Return True when the Codex SDK is failing in a way the CLI exec path can bypass."""
    cur: BaseException | None = exc
    while cur is not None:
        lowered = str(cur).lower()
        if "fatal stderr signal" in lowered:
            return True
        if "stream closed" in lowered:
            return True
        if "reading prompt from stdin" in lowered:
            return True
        cur = cur.__cause__ or cur.__context__
        if cur is exc:
            break
    return False


def _is_limit_overrun(exc: BaseException) -> bool:
    """Check whether *exc* or its cause chain contains a buffer overflow."""
    cur: BaseException | None = exc
    while cur is not None:
        name = type(cur).__name__
        if "LimitOverrunError" in name:
            return True
        msg = str(cur)
        if "chunk exceed the limit" in msg or "Separator is not found" in msg:
            return True
        cur = cur.__cause__ or cur.__context__
        if cur is exc:
            break
    return False


@dataclass
class CodexResultMessage:
    """Adapter providing the ``num_turns`` / ``session_id`` interface
    expected by ``AgentCore`` session-chaining logic."""

    num_turns: int = 0
    session_id: str = ""
    usage: dict[str, int] | None = None


def _wrap_result_message(
    turn: Any,
    thread: Any | None = None,
    completed_turns: int = 0,
) -> CodexResultMessage:
    """Wrap a Codex turn/event into a ``CodexResultMessage``."""
    usage_raw = getattr(turn, "usage", None)
    usage = _usage_to_dict(usage_raw) if usage_raw else None
    raw_num_turns = getattr(turn, "num_turns", 0)
    try:
        num_turns = int(raw_num_turns or 0)
    except (TypeError, ValueError):
        num_turns = 0
    num_turns = num_turns or completed_turns
    if num_turns <= 0 and turn is not None:
        num_turns = 1
    session_id = ""
    if thread:
        session_id = _get_thread_id(thread) or ""
    return CodexResultMessage(
        num_turns=num_turns,
        session_id=session_id,
        usage=usage,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _close_codex_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            await _maybe_await(close())
        except Exception:
            logger.debug("Failed to close Codex SDK client", exc_info=True)


# ── Executor ─────────────────────────────────────────────────


class CodexSDKExecutor(BaseExecutor):
    """Execute via Codex SDK (Mode C).

    The SDK spawns the Codex CLI as a subprocess.  Tool access is secured by
    Codex's ``sandbox_mode`` and MCP integration with ``core/mcp/server.py``.
    """

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
        self._codex_home = anima_dir / ".codex_home"

    @property
    def supports_streaming(self) -> bool:  # noqa: D102
        return True

    # ── Environment / config helpers ─────────────────────────

    def _build_env(self) -> dict[str, str]:
        """Build env dict for the Codex CLI child process."""
        from core.execution.session_context import current_runtime_session
        from core.paths import PROJECT_DIR

        env: dict[str, str] = {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PATH": _default_path_env(),
            "CODEX_HOME": str(self._codex_home),
            "HOME": _default_home_dir(),
        }
        ctx = current_runtime_session()
        if ctx is not None:
            env.update(ctx.to_env())
        # Windows requires SYSTEMROOT for Winsock/TLS initialisation and
        # TEMP/TMP for scratch files.  Without these the Codex CLI subprocess
        # fails with OS error 10106 (WSAEPROVIDERFAILEDINIT).
        if sys.platform == "win32":
            for var in ("SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "APPDATA"):
                val = os.environ.get(var)
                if val:
                    env[var] = val
        api_key = self._resolve_api_key()
        if _is_codex_azure_config(self._model_config):
            if api_key:
                env["AZURE_OPENAI_API_KEY"] = api_key
            elif os.environ.get("AZURE_OPENAI_API_KEY"):
                env["AZURE_OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
            return env

        if api_key and _is_openai_api_key(api_key):
            env["OPENAI_API_KEY"] = api_key
        elif api_key:
            logger.debug(
                "Skipping non-OpenAI API key for Codex env (prefix=%s…); relying on cached ChatGPT auth",
                api_key[:8],
            )
        # Only forward api_base_url when it is a genuine OpenAI-compatible
        # endpoint.  The default credential may point to Ollama
        # (127.0.0.1:11434) which must NOT be injected as OPENAI_BASE_URL
        # — the Codex CLI uses model_provider in config.toml for routing.
        base = self._model_config.api_base_url
        if base and ":11434" not in base:
            env["OPENAI_BASE_URL"] = base
        return env

    def _build_mcp_env(self) -> dict[str, str]:
        """Build env dict for the MCP server subprocess."""
        from core.execution.session_context import current_runtime_session
        from core.paths import PROJECT_DIR

        env = {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PYTHONPATH": str(PROJECT_DIR),
            "PATH": _default_path_env(),
        }
        ctx = current_runtime_session()
        if ctx is not None:
            env.update(ctx.to_env())
        return env

    def _propagate_auth(self) -> None:
        """Propagate ``auth.json`` from the default CODEX_HOME into per-anima CODEX_HOME.

        This lets animas share the ChatGPT subscription auth obtained via
        ``codex auth`` (or ``login_with_device_code``).  Token refreshes
        propagate automatically when a symlink or hardlink is available.
        On Windows, symlink creation may be disallowed for non-admin users,
        so we gracefully fall back to a hardlink and then to a plain file
        copy.  If the per-anima directory already has a real ``auth.json``
        (e.g. written by a prior API-key login), it is left untouched.
        """
        default_auth = Path.home() / ".codex" / "auth.json"
        target = self._codex_home / "auth.json"

        if target.exists() and not target.is_symlink():
            return

        if target.is_symlink():
            if target.resolve() == default_auth.resolve():
                return
            target.unlink()

        if default_auth.is_file():
            try:
                target.symlink_to(default_auth)
                logger.info("Symlinked auth.json -> %s", default_auth)
                return
            except OSError as exc:
                logger.debug("auth.json symlink unavailable; falling back: %s", exc)

            try:
                os.link(default_auth, target)
                logger.info("Hardlinked auth.json -> %s", default_auth)
                return
            except OSError as exc:
                logger.debug("auth.json hardlink unavailable; falling back to copy: %s", exc)

            shutil.copy2(default_auth, target)
            logger.warning(
                "Copied auth.json from %s into %s; future token refreshes may require re-sync",
                default_auth,
                target,
            )

    # Injected via config.toml ``developer_instructions`` so the Codex
    # model always produces a visible text response, even when it only
    # performed tool calls internally.  ``model_instructions_file``
    # replaces the Codex CLI's built-in system prompt (which contains its
    # own "preamble messages" guidance), so we must re-introduce the
    # requirement explicitly.
    _CODEX_DEVELOPER_INSTRUCTIONS: str = (
        "IMPORTANT: You MUST always provide a text response to the user. "
        "After performing any tool calls, write a concise text message "
        "summarising what you did or responding to the user's message. "
        "Never end a turn with only tool operations and no text output. "
        "For conversational messages (greetings, questions, casual chat), "
        "respond naturally in text before or after any tool use."
    )

    def _write_codex_config(self, system_prompt: str) -> None:
        """Write CODEX_HOME config.toml and model instructions file.

        The CODEX_HOME lives at ``{anima_dir}/.codex_home/`` and persists
        across sessions so that Codex's thread data (``sessions/``) survives.
        """
        self._codex_home.mkdir(parents=True, exist_ok=True)
        self._propagate_auth()

        instructions_file = self._codex_home / "instructions.md"
        instructions_file.write_text(system_prompt, encoding="utf-8")

        provider_config = _resolve_codex_provider_config(self._model_config)
        esc = _escape_toml_string

        from core.config.models import load_permissions

        permissions_config = load_permissions(self._anima_dir)

        if "/" in permissions_config.file_roots:
            sandbox_mode = "danger-full-access"
            sandbox_section = ""
        else:
            sandbox_mode = "workspace-write"
            writable_roots = [str(self._anima_dir)]
            for root in permissions_config.file_roots:
                resolved = str(Path(root).resolve())
                if resolved not in writable_roots:
                    writable_roots.append(resolved)
            if self._task_cwd:
                cwd_str = str(self._task_cwd)
                if cwd_str not in writable_roots:
                    writable_roots.append(cwd_str)
            roots_list = ", ".join(f'"{esc(r)}"' for r in writable_roots)
            sandbox_section = f"\n[sandbox_workspace_write]\nwritable_roots = [{roots_list}]\nnetwork_access = true\n"

        mcp_env = self._build_mcp_env()
        mcp_env_lines = "\n".join(f'{k} = "{esc(v)}"' for k, v in mcp_env.items())
        provider_section = ""
        if provider_config.is_azure:
            provider_section = (
                f"\n"
                f"[model_providers.azure]\n"
                f'name = "Azure"\n'
                f'base_url = "{esc(provider_config.base_url or "")}"\n'
                f'env_key = "{esc(provider_config.env_key)}"\n'
                f'query_params = {{ api-version = "{esc(provider_config.api_version or "")}" }}\n'
                f'wire_api = "{esc(provider_config.wire_api)}"\n'
            )

        config_toml = (
            f'model = "{esc(provider_config.model)}"\n'
            f'model_provider = "{esc(provider_config.provider)}"\n'
            f'model_instructions_file = "{esc(str(instructions_file))}"\n'
            f'developer_instructions = "{esc(self._CODEX_DEVELOPER_INSTRUCTIONS)}"\n'
            f'personality = "friendly"\n'
            f'model_verbosity = "high"\n'
            f'sandbox_mode = "{sandbox_mode}"\n'
            f'approval_policy = "never"\n'
            f"{sandbox_section}"
            f"{provider_section}"
            f"\n"
            f"[mcp_servers.aw]\n"
            f'command = "{esc(sys.executable)}"\n'
            f'args = ["-m", "core.mcp.server"]\n'
            f'default_tools_approval_mode = "approve"\n'
            f"\n"
            f"[mcp_servers.aw.env]\n"
            f"{mcp_env_lines}\n"
        )
        (self._codex_home / "config.toml").write_text(config_toml, encoding="utf-8")

    def _create_codex_client(self) -> Any:
        """Create an ``AsyncCodex`` SDK client instance."""
        try:
            from openai_codex import AsyncCodex, CodexConfig
        except ModuleNotFoundError as e:
            raise ImportError("openai_codex is required for Mode C (install openai-codex).") from e

        executable = get_codex_executable()
        config = CodexConfig(
            codex_bin=executable,
            cwd=str(self._task_cwd or self._anima_dir),
            env=self._build_env(),
            client_name="animaworks",
            client_title="AnimaWorks",
        )
        return AsyncCodex(config)

    def _sdk_approval_mode(self) -> Any:
        from openai_codex import ApprovalMode

        return ApprovalMode.deny_all

    def _sdk_sandbox(self) -> Any:
        from openai_codex import Sandbox

        from core.config.models import load_permissions

        permissions_config = load_permissions(self._anima_dir)
        if "/" in permissions_config.file_roots:
            return Sandbox.full_access
        return Sandbox.workspace_write

    def _codex_thread_kwargs(self, system_prompt: str) -> dict[str, Any]:
        provider_config = _resolve_codex_provider_config(self._model_config)
        return {
            "approval_mode": self._sdk_approval_mode(),
            "base_instructions": system_prompt or None,
            "cwd": str(self._task_cwd or self._anima_dir),
            "developer_instructions": self._CODEX_DEVELOPER_INSTRUCTIONS,
            "model": provider_config.model,
            "model_provider": provider_config.provider,
            "sandbox": self._sdk_sandbox(),
        }

    def _codex_turn_kwargs(self) -> dict[str, Any]:
        provider_config = _resolve_codex_provider_config(self._model_config)
        return {
            "approval_mode": self._sdk_approval_mode(),
            "cwd": str(self._task_cwd or self._anima_dir),
            "model": provider_config.model,
            "sandbox": self._sdk_sandbox(),
        }

    def _build_cli_exec_command(self) -> list[str]:
        """Build the `codex exec --json` command used as a runtime fallback."""
        executable = get_codex_executable()
        if not executable:
            raise RuntimeError("Codex CLI executable not available for exec fallback")
        return [
            executable,
            "exec",
            "-C",
            str(self._task_cwd or self._anima_dir),
            "--skip-git-repo-check",
            "--json",
            "-",
        ]

    async def _execute_streaming_via_cli_exec(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        trigger: str = "",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Fallback executor using `codex exec --json` when the SDK transport is unstable."""
        self._write_codex_config(system_prompt)
        cmd = self._build_cli_exec_command()
        env = self._build_env()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=_SUBPROCESS_STREAM_LIMIT,
        )
        if proc.stdin is None or proc.stdout is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            _close_subprocess_stdio(proc)
            raise RuntimeError("Codex CLI exec fallback missing stdin/stdout")

        stderr_chunks: list[bytes] = []

        async def _read_stderr() -> None:
            if proc.stderr is None:
                return
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        stderr_task = asyncio.create_task(_read_stderr())
        response_parts: list[str] = []
        tool_records: list[ToolCallRecord] = []
        usage_acc = TokenUsage()
        emitted_tool_starts: set[str] = set()
        thread_id = ""
        usage_dict: dict[str, int] | None = None

        try:
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                raw_line = line.decode("utf-8", errors="replace").rstrip("\n")
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.debug("Ignoring non-JSON codex exec output: %s", raw_line[:200])
                    continue

                ptype = str(payload.get("type", ""))
                if ptype == "thread.started":
                    thread_id = str(payload.get("thread_id", ""))
                    continue
                if ptype in ("turn.started",):
                    continue

                if ptype == "item.started":
                    item = payload.get("item") or {}
                    item_type = str(item.get("type", ""))
                    item_id = str(item.get("id", ""))
                    if item_type in ("command_execution", "mcp_tool_call") and item_id not in emitted_tool_starts:
                        emitted_tool_starts.add(item_id)
                        yield {
                            "type": "tool_start",
                            "tool_name": _codex_item_tool_name(type("Obj", (), item)(), item_type),
                            "tool_id": item_id,
                        }
                    continue

                if ptype == "item.completed":
                    item = payload.get("item") or {}
                    item_type = str(item.get("type", ""))
                    item_id = str(item.get("id", ""))

                    if item_type == "agent_message":
                        text = str(item.get("text", ""))
                        if text:
                            response_parts.append(text)
                            yield {"type": "text_delta", "text": text}
                        continue

                    if item_type in ("command_execution", "mcp_tool_call"):
                        if item_id not in emitted_tool_starts:
                            emitted_tool_starts.add(item_id)
                            yield {
                                "type": "tool_start",
                                "tool_name": _codex_item_tool_name(type("Obj", (), item)(), item_type),
                                "tool_id": item_id,
                            }
                        rec = _cli_exec_item_to_tool_record(item)
                        if rec:
                            tool_records.append(rec)
                        yield {
                            "type": "tool_end",
                            "tool_name": _codex_item_tool_name(type("Obj", (), item)(), item_type),
                            "tool_id": item_id,
                        }
                        continue

                if ptype == "turn.completed":
                    usage_dict = _usage_to_dict(payload.get("usage", {}))
                    usage_acc = TokenUsage(
                        input_tokens=usage_dict.get("input_tokens", 0) or usage_dict.get("prompt_tokens", 0) or 0,
                        output_tokens=usage_dict.get("output_tokens", 0) or usage_dict.get("completion_tokens", 0) or 0,
                    )
                    continue

            returncode = await proc.wait()
            await stderr_task
            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
            if returncode != 0:
                raise RuntimeError(stderr_text or f"codex exec exited with code {returncode}")
            if stderr_text:
                logger.debug("Codex CLI exec stderr: %s", stderr_text[:500])
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            _close_subprocess_stdio(proc)

        full_text = "\n".join(response_parts)
        if not full_text and tool_records:
            full_text = _synthesise_fallback(tool_records)
        replied_to = self._read_replied_to_file()
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": CodexResultMessage(
                num_turns=1 if (full_text or tool_records) else 0,
                session_id=thread_id,
                usage=usage_dict,
            ),
            "replied_to_from_transcript": replied_to,
            "tool_call_records": [asdict(r) for r in tool_records],
            "usage": usage_acc.to_dict(),
        }

    async def _execute_via_cli_exec(
        self,
        prompt: str,
        system_prompt: str,
        tracker: ContextTracker | None = None,
        trigger: str = "",
    ) -> ExecutionResult:
        """Blocking wrapper around the CLI exec fallback path."""
        tracker = tracker or ContextTracker(model=self._model_config.model)
        final_event: dict[str, Any] | None = None
        async for ev in self._execute_streaming_via_cli_exec(system_prompt, prompt, tracker, trigger=trigger):
            if ev.get("type") == "done":
                final_event = ev
        if final_event is None:
            return ExecutionResult(text="[Codex CLI exec fallback returned no result]")
        usage_raw = final_event.get("usage") or {}
        usage_acc = None
        if usage_raw:
            usage_acc = TokenUsage(
                input_tokens=usage_raw.get("input_tokens", 0) or usage_raw.get("prompt_tokens", 0) or 0,
                output_tokens=usage_raw.get("output_tokens", 0) or usage_raw.get("completion_tokens", 0) or 0,
            )
        return ExecutionResult(
            text=str(final_event.get("full_text", "")),
            result_message=final_event.get("result_message"),
            replied_to_from_transcript=final_event.get("replied_to_from_transcript", set()),
            tool_call_records=[ToolCallRecord(**record) for record in (final_event.get("tool_call_records") or [])],
            usage=usage_acc,
        )

    async def _start_or_resume_thread(
        self,
        codex: Any,
        thread_id: str | None,
        session_type: str,
        system_prompt: str,
        chat_thread_id: str = "default",
        persist_thread: bool = True,
    ) -> Any:
        """Start a new thread or attempt to resume an existing one."""
        thread_kwargs = self._codex_thread_kwargs(system_prompt)
        if thread_id:
            try:
                thread = await _maybe_await(codex.thread_resume(thread_id, **thread_kwargs))
                logger.info("Resumed Codex thread %s", thread_id)
                return thread
            except Exception as e:
                logger.warning(
                    "Codex thread resume failed (thread_id=%s): %s. Starting fresh thread.",
                    thread_id,
                    e,
                )
                if persist_thread:
                    _clear_thread_id(self._anima_dir, session_type, chat_thread_id)
        thread = await _maybe_await(codex.thread_start(**thread_kwargs))
        logger.info("Started fresh Codex thread")
        return thread

    def discard_thread(
        self,
        session_type: str = "chat",
        chat_thread_id: str = "default",
    ) -> None:
        """Discard Codex thread ID so next session starts fresh."""
        _clear_thread_id(self._anima_dir, session_type, chat_thread_id)
        logger.info(
            "Discarded Codex thread for %s (session=%s, thread=%s)",
            self._anima_dir.name,
            session_type,
            chat_thread_id,
        )

    # ── Blocking execution ───────────────────────────────────

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run a session via Codex SDK (blocking mode)."""
        if self._check_interrupted():
            return ExecutionResult(text="[Session interrupted by user]")

        if _should_prefer_cli_exec(trigger):
            logger.info("Using `codex exec` directly for trigger=%s", trigger)
            return await self._execute_via_cli_exec(prompt, system_prompt, tracker, trigger=trigger)

        session_type = _resolve_session_type(trigger)
        chat_thread_id = thread_id
        persist_thread = is_persistent_codex_session(trigger)
        if persist_thread:
            codex_thread_id = _load_thread_id(self._anima_dir, session_type, chat_thread_id)
        else:
            clear_codex_thread_id(self._anima_dir, session_type, chat_thread_id)
            codex_thread_id = None

        prompt_bytes = len(system_prompt.encode("utf-8"))
        if codex_thread_id and prompt_bytes > _RESUME_PROMPT_SIZE_LIMIT:
            logger.info(
                "Skipping Codex resume (prompt=%d bytes > %d limit) to avoid LimitOverrunError; using fresh thread",
                prompt_bytes,
                _RESUME_PROMPT_SIZE_LIMIT,
            )
            codex_thread_id = None

        self._write_codex_config(system_prompt)
        codex = self._create_codex_client()
        try:
            try:
                thread = await self._start_or_resume_thread(
                    codex,
                    codex_thread_id,
                    session_type,
                    system_prompt,
                    chat_thread_id,
                    persist_thread,
                )
                turn = await _maybe_await(thread.run(prompt, **self._codex_turn_kwargs()))
            except Exception as e:
                if codex_thread_id:
                    logger.warning(
                        "Codex execute failed with resume (thread=%s): %s. Retrying with fresh thread.",
                        codex_thread_id,
                        e,
                    )
                    if persist_thread:
                        _clear_thread_id(self._anima_dir, session_type, chat_thread_id)
                    try:
                        thread = await self._start_or_resume_thread(
                            codex,
                            None,
                            session_type,
                            system_prompt,
                            chat_thread_id,
                            persist_thread,
                        )
                        turn = await _maybe_await(thread.run(prompt, **self._codex_turn_kwargs()))
                    except Exception as retry_exc:
                        if _should_cli_exec_fallback(retry_exc):
                            logger.warning("Codex SDK execute failed; falling back to `codex exec`")
                            return await self._execute_via_cli_exec(prompt, system_prompt, tracker, trigger=trigger)
                        logger.exception("Codex SDK execution error (fresh retry)")
                        return ExecutionResult(
                            text=f"[Codex SDK Error: {retry_exc}]",
                        )
                else:
                    if _should_cli_exec_fallback(e):
                        logger.warning("Codex SDK execute failed; falling back to `codex exec`")
                        return await self._execute_via_cli_exec(prompt, system_prompt, tracker, trigger=trigger)
                    logger.exception("Codex SDK execution error")
                    return ExecutionResult(text=f"[Codex SDK Error: {e}]")

            if self._check_interrupted():
                logger.info("Codex SDK execute interrupted after run")
                return ExecutionResult(text="[Session interrupted by user]")

            tid = _get_thread_id(thread)
            if tid and persist_thread:
                _save_thread_id(self._anima_dir, tid, session_type, chat_thread_id)

            response_text = getattr(turn, "final_response", "") or ""
            items = getattr(turn, "items", []) or []
            tool_records = _extract_tool_records(items)

            if not response_text and tool_records:
                response_text = _synthesise_fallback(tool_records)

            usage_acc: TokenUsage | None = None
            raw_usage = getattr(turn, "usage", None)
            if raw_usage:
                ud = _usage_to_dict(raw_usage)
                usage_acc = TokenUsage(
                    input_tokens=ud.get("input_tokens", 0) or ud.get("prompt_tokens", 0) or 0,
                    output_tokens=ud.get("output_tokens", 0) or ud.get("completion_tokens", 0) or 0,
                )

            replied_to = self._read_replied_to_file()
            return ExecutionResult(
                text=response_text,
                result_message=_wrap_result_message(turn, thread, completed_turns=1),
                replied_to_from_transcript=replied_to,
                tool_call_records=tool_records,
                usage=usage_acc,
            )
        finally:
            await _close_codex_client(codex)

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream events from Codex SDK.

        Handles the full Codex event lifecycle for progressive streaming:

        - ``item.started`` / ``item.updated``: emit incremental text deltas
          by tracking per-item text length and yielding only the new portion.
        - ``item.completed``: emit any remaining text delta and tool records.
        - ``turn.completed``: update context tracker with usage stats.
        - ``turn.failed`` / ``error``: propagate as error events.

        Yields dicts:
            ``{"type": "text_delta", "text": "..."}``
            ``{"type": "tool_start", "tool_name": "...", "tool_id": "..."}``
            ``{"type": "tool_end", "tool_id": "...", "tool_name": "..."}``
            ``{"type": "tool_detail", "tool_id": "...", ...}``
            ``{"type": "done", "full_text": "...", "result_message": ...}``
        """
        if self._check_interrupted():
            yield {"type": "text_delta", "text": "[Session interrupted by user]"}
            yield {"type": "done", "full_text": "[Session interrupted by user]", "result_message": None}
            return

        if _should_prefer_cli_exec(trigger):
            logger.info("Using `codex exec` streaming directly for trigger=%s", trigger)
            async for ev in self._execute_streaming_via_cli_exec(system_prompt, prompt, tracker, trigger=trigger):
                yield ev
            return

        session_type = _resolve_session_type(trigger)
        chat_thread_id = thread_id
        persist_thread = is_persistent_codex_session(trigger)
        if persist_thread:
            codex_thread_id = _load_thread_id(self._anima_dir, session_type, chat_thread_id)
        else:
            clear_codex_thread_id(self._anima_dir, session_type, chat_thread_id)
            codex_thread_id = None

        prompt_bytes = len(system_prompt.encode("utf-8"))
        if codex_thread_id and prompt_bytes > _RESUME_PROMPT_SIZE_LIMIT:
            logger.info(
                "Skipping Codex resume (prompt=%d bytes > %d limit) to avoid LimitOverrunError; using fresh thread",
                prompt_bytes,
                _RESUME_PROMPT_SIZE_LIMIT,
            )
            codex_thread_id = None

        self._write_codex_config(system_prompt)
        codex = self._create_codex_client()

        response_item_order: list[str] = []
        response_text_by_item: dict[str, str] = {}
        all_tool_records: list[ToolCallRecord] = []
        turn_result: Any = None
        active_thread: Any = None
        usage_acc = TokenUsage()
        completed_turn_count = 0

        def _current_full_text() -> str:
            return "\n".join(response_text_by_item[item_id] for item_id in response_item_order if response_text_by_item[item_id])

        def _remember_agent_delta(item_id: str, delta: str) -> None:
            if not item_id:
                item_id = f"agent-{len(response_item_order) + 1}"
            if item_id not in response_text_by_item:
                response_item_order.append(item_id)
                response_text_by_item[item_id] = ""
            response_text_by_item[item_id] += delta

        def _set_agent_text(item_id: str, text: str) -> None:
            if not item_id:
                item_id = f"agent-{len(response_item_order) + 1}"
            if item_id not in response_text_by_item:
                response_item_order.append(item_id)
            response_text_by_item[item_id] = text

        async def _stream_turn(tid: str | None) -> AsyncGenerator[dict[str, Any], None]:
            nonlocal completed_turn_count, turn_result, active_thread
            thread = await self._start_or_resume_thread(
                codex,
                tid,
                session_type,
                system_prompt,
                chat_thread_id,
                persist_thread,
            )
            active_thread = thread
            turn = await _maybe_await(thread.turn(prompt, **self._codex_turn_kwargs()))
            stream = turn.stream()
            event_iter = stream.__aiter__()
            idle_timeout = _event_idle_timeout_seconds(trigger)

            item_text_len: dict[str, int] = {}
            agent_delta_seen: set[str] = set()
            tool_started: set[str] = set()
            tool_ended: set[str] = set()

            def _tool_start_chunk(tool_id: str, tool_name: str) -> dict[str, Any] | None:
                if not tool_id or tool_id in tool_started:
                    return None
                tool_started.add(tool_id)
                return {
                    "type": "tool_start",
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                }

            def _tool_detail_chunk(tool_id: str, tool_name: str, detail: str) -> dict[str, Any] | None:
                if not detail:
                    return None
                return {
                    "type": "tool_detail",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "detail": detail,
                }

            def _usage_from_raw(raw_usage: Any) -> None:
                if not raw_usage:
                    return
                ud = _usage_to_dict(raw_usage)
                usage_acc.input_tokens = ud.get("input_tokens", 0) or ud.get("prompt_tokens", 0) or 0
                usage_acc.output_tokens = ud.get("output_tokens", 0) or ud.get("completion_tokens", 0) or 0

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_iter.__anext__(),
                            timeout=idle_timeout,
                        )
                    except StopAsyncIteration:
                        break
                    except TimeoutError as e:
                        raise StreamDisconnectedError(
                            f"Codex SDK stream idle timeout after {idle_timeout:.0f}s",
                            partial_text=_current_full_text(),
                            immediate_retry=True,
                        ) from e

                    if self._check_interrupted():
                        logger.info("Codex SDK streaming interrupted")
                        yield {"type": "text_delta", "text": "[Session interrupted by user]"}
                        return

                    method = _event_method(event)
                    payload = _event_payload(event)

                    if method == "item/agentMessage/delta":
                        item_id = _payload_item_id(payload)
                        delta = _payload_delta(payload)
                        if delta:
                            agent_delta_seen.add(item_id)
                            _remember_agent_delta(item_id, delta)
                            yield {"type": "text_delta", "text": delta}
                        continue

                    if method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
                        delta = _payload_delta(payload)
                        if delta:
                            yield {"type": "thinking_delta", "text": delta}
                        continue

                    if method == "item/commandExecution/outputDelta":
                        tool_id = _payload_item_id(payload)
                        start = _tool_start_chunk(tool_id, "command")
                        if start:
                            yield start
                        detail = _payload_delta(payload)
                        detail_chunk = _tool_detail_chunk(tool_id, "command", detail)
                        if detail_chunk:
                            yield detail_chunk
                        continue

                    if method in ("item/fileChange/outputDelta", "item/fileChange/patchUpdated"):
                        tool_id = _payload_item_id(payload)
                        start = _tool_start_chunk(tool_id, "file_change")
                        if start:
                            yield start
                        detail = _payload_delta(payload) or _format_file_changes(_get_list(payload, "changes"))
                        detail_chunk = _tool_detail_chunk(tool_id, "file_change", detail)
                        if detail_chunk:
                            yield detail_chunk
                        continue

                    if method == "item/mcpToolCall/progress":
                        tool_id = _payload_item_id(payload)
                        start = _tool_start_chunk(tool_id, "mcp_tool")
                        if start:
                            yield start
                        detail_chunk = _tool_detail_chunk(tool_id, "mcp_tool", _get_str(payload, "message"))
                        if detail_chunk:
                            yield detail_chunk
                        continue

                    if method in ("item/started", "item/updated"):
                        item = _get_attr(payload, "item", None)
                        if item is None:
                            continue
                        item_type = _item_type(item)
                        item_id = _item_id(item)

                        if item_type == "agent_message":
                            text = _extract_item_text(item)
                            if text:
                                prev_len = item_text_len.get(item_id, 0)
                                if len(text) > prev_len:
                                    delta = text[prev_len:]
                                    item_text_len[item_id] = len(text)
                                    agent_delta_seen.add(item_id)
                                    _remember_agent_delta(item_id, delta)
                                    yield {"type": "text_delta", "text": delta}

                        elif item_type == "reasoning":
                            text = _extract_item_text(item)
                            if text:
                                prev_len = item_text_len.get(item_id, 0)
                                if len(text) > prev_len:
                                    delta = text[prev_len:]
                                    item_text_len[item_id] = len(text)
                                    yield {"type": "thinking_delta", "text": delta}

                        elif item_type in (
                            "command_execution",
                            "mcp_tool_call",
                            "file_change",
                            "web_search",
                            "dynamic_tool_call",
                            "collab_agent_tool_call",
                        ):
                            start = _tool_start_chunk(item_id, _codex_item_tool_name(item, item_type))
                            if start:
                                yield start
                        continue

                    if method == "item/completed":
                        item = _get_attr(payload, "item", None)
                        if item is None:
                            continue
                        item_type = _item_type(item)
                        item_id = _item_id(item)

                        if item_type == "agent_message":
                            text = _extract_item_text(item)
                            if text:
                                if item_id in agent_delta_seen:
                                    _set_agent_text(item_id, text)
                                else:
                                    _set_agent_text(item_id, text)
                                    yield {"type": "text_delta", "text": text}
                                item_text_len[item_id] = len(text)
                            else:
                                logger.debug(
                                    "Codex item/completed agent_message with empty text: %s",
                                    repr(item)[:300],
                                )

                        elif item_type == "reasoning":
                            text = _extract_item_text(item)
                            if text:
                                prev_len = item_text_len.get(item_id, 0)
                                if len(text) > prev_len:
                                    yield {"type": "thinking_delta", "text": text[prev_len:]}
                                item_text_len[item_id] = len(text)

                        elif item_type in (
                            "command_execution",
                            "mcp_tool_call",
                            "file_change",
                            "web_search",
                            "dynamic_tool_call",
                            "collab_agent_tool_call",
                        ):
                            tool_name = _codex_item_tool_name(item, item_type)
                            start = _tool_start_chunk(item_id, tool_name)
                            if start:
                                yield start
                            if item_type == "file_change":
                                detail = _format_file_changes(_get_list(_unwrap_thread_item(item), "changes"))
                                detail_chunk = _tool_detail_chunk(item_id, tool_name, detail)
                                if detail_chunk:
                                    yield detail_chunk
                            rec = _item_to_tool_record(item)
                            if rec:
                                all_tool_records.append(rec)
                            if item_id not in tool_ended:
                                tool_ended.add(item_id)
                                yield {
                                    "type": "tool_end",
                                    "tool_id": item_id,
                                    "tool_name": tool_name,
                                }

                        else:
                            text = _extract_item_text(item)
                            if text:
                                logger.info(
                                    "Codex item/completed type=%s has text (%d chars); emitting",
                                    item_type,
                                    len(text),
                                )
                                _set_agent_text(item_id, text)
                                yield {"type": "text_delta", "text": text}
                            else:
                                logger.debug(
                                    "Codex item/completed type=%s: %s",
                                    item_type,
                                    repr(item)[:300],
                                )
                        continue

                    if method == "thread/tokenUsage/updated":
                        _usage_from_raw(_get_attr(payload, "token_usage", None))
                        continue

                    if method == "turn/completed":
                        completed_turn_count += 1
                        turn_result = _wrap_result_message(payload, thread, completed_turns=completed_turn_count)
                        _usage_from_raw(_get_attr(payload, "usage", None) or _get_attr(payload, "token_usage", None))
                        saved_tid = _get_thread_id(thread)
                        if saved_tid and persist_thread:
                            _save_thread_id(self._anima_dir, saved_tid, session_type, chat_thread_id)
                        turn_obj = _get_attr(payload, "turn", None)
                        error_obj = _get_attr(turn_obj, "error", None)
                        error_msg = _get_str(error_obj, "message")
                        if error_msg:
                            logger.error("Codex turn/completed error: %s", error_msg)
                            yield {"type": "error", "message": f"[Codex turn failed: {error_msg}]"}
                        continue

                    if method == "turn/failed":
                        err_obj = _get_attr(payload, "error", None)
                        error_msg = _get_str(err_obj, "message") or str(err_obj or "")
                        logger.error("Codex turn.failed: %s", error_msg)
                        yield {"type": "error", "message": f"[Codex turn failed: {error_msg}]"}
                        continue

                    if method == "error":
                        error_msg = _get_str(payload, "message") or str(payload)
                        logger.error("Codex error event: %s", error_msg)
                        yield {"type": "error", "message": f"[Codex error: {error_msg}]"}
                        continue

                    if method in ("thread/started", "turn/started"):
                        logger.debug("Codex lifecycle event: %s", method)
                        continue

                    logger.debug(
                        "Codex unhandled event method=%s attrs=%s",
                        method,
                        [a for a in dir(payload) if not a.startswith("_")][:15],
                    )
            finally:
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    await aclose()

        try:
            # Try resume first, fallback to fresh thread.
            fell_back = False
            if codex_thread_id:
                try:
                    gen = _stream_turn(codex_thread_id)
                    first_event: dict[str, Any] | None = None
                    try:
                        first_event = await asyncio.wait_for(
                            gen.__anext__(),
                            timeout=RESUME_TIMEOUT_SEC,
                        )
                    except (TimeoutError, StopAsyncIteration):
                        logger.warning(
                            "Codex resume timed out or empty (thread=%s), falling back to fresh thread.",
                            codex_thread_id,
                        )
                        if persist_thread:
                            _clear_thread_id(self._anima_dir, session_type, chat_thread_id)
                        fell_back = True
                        await gen.aclose()
                    except Exception as e:
                        logger.warning(
                            "Codex resume stream failed (thread=%s): %s",
                            codex_thread_id,
                            e,
                        )
                        if persist_thread:
                            _clear_thread_id(self._anima_dir, session_type, chat_thread_id)
                        fell_back = True
                        await gen.aclose()
                    else:
                        if first_event:
                            yield first_event
                        async for ev in gen:
                            yield ev
                except Exception as e:
                    logger.warning(
                        "Codex stream resume error: %s. Fresh thread.",
                        e,
                    )
                    if persist_thread:
                        _clear_thread_id(self._anima_dir, session_type, chat_thread_id)
                    fell_back = True
            else:
                fell_back = True

            if fell_back:
                try:
                    async for ev in _stream_turn(None):
                        yield ev
                except Exception as e:
                    if _should_cli_exec_fallback(e):
                        logger.warning("Codex SDK streaming failed; falling back to `codex exec`")
                        async for ev in self._execute_streaming_via_cli_exec(
                            system_prompt, prompt, tracker, trigger=trigger
                        ):
                            yield ev
                        return
                    logger.exception("Codex SDK streaming error")
                    partial = _current_full_text()
                    is_buffer_overflow = _is_limit_overrun(e)
                    raise StreamDisconnectedError(
                        f"Codex SDK stream error: {e}",
                        partial_text=partial,
                        immediate_retry=is_buffer_overflow,
                    ) from e

            full_text = _current_full_text()
            if not full_text and all_tool_records:
                full_text = _synthesise_fallback(all_tool_records)
            if turn_result is None and (full_text or all_tool_records):
                turn_result = CodexResultMessage(
                    num_turns=max(1, completed_turn_count),
                    session_id=_get_thread_id(active_thread) or "",
                    usage=usage_acc.to_dict(),
                )

            replied_to = self._read_replied_to_file()
            yield {
                "type": "done",
                "full_text": full_text,
                "result_message": turn_result,
                "replied_to_from_transcript": replied_to,
                "tool_call_records": [asdict(r) for r in all_tool_records],
                "usage": usage_acc.to_dict(),
            }
        finally:
            await _close_codex_client(codex)
