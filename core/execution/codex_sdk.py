from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode C executor: Codex SDK (Codex CLI wrapper).

Runs OpenAI models via the Codex CLI as an autonomous agent.  The SDK spawns
the Codex CLI binary and exchanges JSONL events over stdin/stdout.  Tool
safety relies on Codex's sandbox mode (container-level isolation) plus MCP
integration with AnimaWorks ``core/mcp/server.py`` for permission-checked
tool access.

System prompt is injected via ``model_instructions_file`` in a per-anima
CODEX_HOME directory.  Session resume uses the Codex SDK's ``resume_thread``
mechanism with thread IDs persisted to the shortterm directory.
"""

import asyncio
import logging
import os
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
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory

logger = logging.getLogger("animaworks.execution.codex_sdk")

__all__ = ["CodexSDKExecutor", "clear_codex_thread_ids", "is_codex_sdk_available"]

RESUME_TIMEOUT_SEC = 15.0

# asyncio.StreamReader default limit is 64 KB.  Codex CLI may echo the full
# context (including system prompt) in a single JSONL line during thread
# resume, triggering LimitOverrunError.  Skip resume when close to this limit.
_RESUME_PROMPT_SIZE_LIMIT = 50_000


# ── Model name helpers ───────────────────────────────────────

def _resolve_codex_model(model: str) -> str:
    """Strip the ``codex/`` prefix to get the bare model name for the CLI."""
    if model.startswith("codex/"):
        return model[len("codex/"):]
    return model


def is_codex_sdk_available() -> bool:
    """Return True when ``openai_codex_sdk`` is importable."""
    try:
        import openai_codex_sdk  # noqa: F401
        return True
    except Exception:
        return False


def _is_openai_api_key(key: str) -> bool:
    """Return True if *key* looks like a genuine OpenAI API key."""
    return bool(key) and not key.startswith("sk-ant-")


def _escape_toml_string(value: str) -> str:
    """Escape a string for safe embedding in a TOML double-quoted value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


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


def clear_codex_thread_ids(anima_dir: Path, chat_thread_id: str = "default") -> None:
    """Clear all persisted Codex thread IDs (both chat and heartbeat)."""
    for st in ("chat", "heartbeat"):
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
    if trigger in ("heartbeat",) or (trigger and trigger.startswith("cron:")):
        return "heartbeat"
    return "chat"


def _extract_item_text(item: Any) -> str:
    """Extract text content from a Codex completed item."""
    if hasattr(item, "content"):
        if isinstance(item.content, str):
            return item.content
        if isinstance(item.content, list):
            parts: list[str] = []
            for part in item.content:
                if hasattr(part, "text"):
                    parts.append(part.text)
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
    if hasattr(item, "text"):
        return item.text
    return ""


def _item_to_tool_record(item: Any) -> ToolCallRecord | None:
    """Convert a Codex tool_use item to a ``ToolCallRecord``."""
    try:
        name = getattr(item, "name", "unknown")
        tool_id = getattr(item, "id", "")
        input_data = getattr(item, "input", {})
        result_data = getattr(item, "output", "")
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
        if getattr(item, "type", None) == "tool_use":
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
        "Codex SDK produced no text output; synthesised fallback "
        "(tools=%d)", len(tool_records),
    )
    return fallback


def _usage_to_dict(usage: Any) -> dict[str, int]:
    """Normalise a Codex usage object (or dict) to a plain dict."""
    if isinstance(usage, dict):
        return usage
    d: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
        val = getattr(usage, key, None)
        if val is not None:
            d[key] = int(val)
    return d


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
) -> CodexResultMessage:
    """Wrap a Codex turn/event into a ``CodexResultMessage``."""
    usage_raw = getattr(turn, "usage", None)
    usage = _usage_to_dict(usage_raw) if usage_raw else None
    num_turns = getattr(turn, "num_turns", 0) or 0
    session_id = ""
    if thread:
        session_id = _get_thread_id(thread) or ""
    return CodexResultMessage(
        num_turns=num_turns,
        session_id=session_id,
        usage=usage,
    )


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
        from core.paths import PROJECT_DIR

        env: dict[str, str] = {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "CODEX_HOME": str(self._codex_home),
            "HOME": os.environ.get("HOME", "/tmp"),
        }
        api_key = self._resolve_api_key()
        if api_key and _is_openai_api_key(api_key):
            env["OPENAI_API_KEY"] = api_key
        elif api_key:
            logger.debug(
                "Skipping non-OpenAI API key for Codex env "
                "(prefix=%s…); relying on cached ChatGPT auth",
                api_key[:8],
            )
        if self._model_config.api_base_url:
            env["OPENAI_BASE_URL"] = self._model_config.api_base_url
        return env

    def _build_mcp_env(self) -> dict[str, str]:
        """Build env dict for the MCP server subprocess."""
        from core.paths import PROJECT_DIR

        return {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PYTHONPATH": str(PROJECT_DIR),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

    def _propagate_auth(self) -> None:
        """Symlink ``auth.json`` from the default CODEX_HOME into per-anima CODEX_HOME.

        This lets animas share the ChatGPT subscription auth obtained via
        ``codex auth`` (or ``login_with_device_code``).  Token refreshes
        propagate automatically through the symlink.  If the per-anima
        directory already has a real ``auth.json`` (e.g. written by a prior
        API-key login), it is left untouched.
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
            target.symlink_to(default_auth)
            logger.info("Symlinked auth.json → %s", default_auth)

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

        bare_model = _resolve_codex_model(self._model_config.model)
        esc = _escape_toml_string

        from core.paths import PROJECT_DIR
        mcp_env = self._build_mcp_env()
        mcp_env_lines = "\n".join(
            f'{k} = "{esc(v)}"' for k, v in mcp_env.items()
        )

        config_toml = (
            f'model = "{esc(bare_model)}"\n'
            f'model_instructions_file = "{esc(str(instructions_file))}"\n'
            f'developer_instructions = "{esc(self._CODEX_DEVELOPER_INSTRUCTIONS)}"\n'
            f'personality = "friendly"\n'
            f'model_verbosity = "high"\n'
            f'sandbox_mode = "workspace-write"\n'
            f'approval_policy = "never"\n'
            f"\n"
            f"[sandbox_workspace_write]\n"
            f'writable_roots = ["{esc(str(self._anima_dir))}"]'
            f"\nnetwork_access = true\n"
            f"\n"
            f"[mcp_servers.aw]\n"
            f'command = "{esc(sys.executable)}"\n'
            f'args = ["-m", "core.mcp.server"]\n'
            f"\n"
            f"[mcp_servers.aw.env]\n"
            f"{mcp_env_lines}\n"
        )
        (self._codex_home / "config.toml").write_text(config_toml, encoding="utf-8")

    def _create_codex_client(self) -> Any:
        """Create a ``Codex`` SDK client instance."""
        try:
            from openai_codex_sdk import Codex
        except ModuleNotFoundError as e:
            raise ImportError(
                "openai_codex_sdk is required for Mode C "
                "(install openai-codex-sdk)."
            ) from e

        return Codex({"env": self._build_env()})

    def _start_or_resume_thread(
        self,
        codex: Any,
        thread_id: str | None,
        session_type: str,
    ) -> Any:
        """Start a new thread or attempt to resume an existing one."""
        if thread_id:
            try:
                thread = codex.resume_thread(thread_id)
                logger.info("Resumed Codex thread %s", thread_id)
                return thread
            except Exception as e:
                logger.warning(
                    "Codex thread resume failed (thread_id=%s): %s. "
                    "Starting fresh thread.",
                    thread_id, e,
                )
                _clear_thread_id(self._anima_dir, session_type)
        thread = codex.start_thread({
            "working_directory": str(self._anima_dir),
            "skip_git_repo_check": True,
        })
        logger.info("Started fresh Codex thread")
        return thread

    # ── Blocking execution ───────────────────────────────────

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
    ) -> ExecutionResult:
        """Run a session via Codex SDK (blocking mode)."""
        if self._check_interrupted():
            return ExecutionResult(text="[Session interrupted by user]")

        session_type = _resolve_session_type(trigger)
        thread_id = _load_thread_id(self._anima_dir, session_type)

        prompt_bytes = len(system_prompt.encode("utf-8"))
        if thread_id and prompt_bytes > _RESUME_PROMPT_SIZE_LIMIT:
            logger.info(
                "Skipping Codex resume (prompt=%d bytes > %d limit) "
                "to avoid LimitOverrunError; using fresh thread",
                prompt_bytes, _RESUME_PROMPT_SIZE_LIMIT,
            )
            thread_id = None

        self._write_codex_config(system_prompt)
        codex = self._create_codex_client()
        thread = self._start_or_resume_thread(codex, thread_id, session_type)

        try:
            turn = await thread.run(prompt)
        except Exception as e:
            if thread_id:
                logger.warning(
                    "Codex execute failed with resume (thread=%s): %s. "
                    "Retrying with fresh thread.",
                    thread_id, e,
                )
                _clear_thread_id(self._anima_dir, session_type)
                thread = codex.start_thread({
                    "working_directory": str(self._anima_dir),
                    "skip_git_repo_check": True,
                })
                try:
                    turn = await thread.run(prompt)
                except Exception as retry_exc:
                    logger.exception("Codex SDK execution error (fresh retry)")
                    return ExecutionResult(
                        text=f"[Codex SDK Error: {retry_exc}]",
                    )
            else:
                logger.exception("Codex SDK execution error")
                return ExecutionResult(text=f"[Codex SDK Error: {e}]")

        if self._check_interrupted():
            logger.info("Codex SDK execute interrupted after run")
            return ExecutionResult(text="[Session interrupted by user]")

        tid = _get_thread_id(thread)
        if tid:
            _save_thread_id(self._anima_dir, tid, session_type)

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
            if tracker:
                tracker.update(ud, include_output_in_ratio=True)

        replied_to = self._read_replied_to_file()
        return ExecutionResult(
            text=response_text,
            result_message=_wrap_result_message(turn, thread),
            replied_to_from_transcript=replied_to,
            tool_call_records=tool_records,
            usage=usage_acc,
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream events from Codex SDK.

        Yields dicts:
            ``{"type": "text_delta", "text": "..."}``
            ``{"type": "tool_start", "tool_name": "...", "tool_id": "..."}``
            ``{"type": "tool_end", "tool_id": "...", "tool_name": "..."}``
            ``{"type": "done", "full_text": "...", "result_message": ...}``
        """
        if self._check_interrupted():
            yield {"type": "text_delta", "text": "[Session interrupted by user]"}
            yield {"type": "done", "full_text": "[Session interrupted by user]", "result_message": None}
            return

        session_type = _resolve_session_type(trigger)
        thread_id = _load_thread_id(self._anima_dir, session_type)

        prompt_bytes = len(system_prompt.encode("utf-8"))
        if thread_id and prompt_bytes > _RESUME_PROMPT_SIZE_LIMIT:
            logger.info(
                "Skipping Codex resume (prompt=%d bytes > %d limit) "
                "to avoid LimitOverrunError; using fresh thread",
                prompt_bytes, _RESUME_PROMPT_SIZE_LIMIT,
            )
            thread_id = None

        self._write_codex_config(system_prompt)
        codex = self._create_codex_client()

        response_text_parts: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        turn_result: Any = None
        active_thread: Any = None
        usage_acc = TokenUsage()

        async def _stream_turn(tid: str | None) -> AsyncGenerator[dict[str, Any], None]:
            nonlocal turn_result, active_thread
            thread = self._start_or_resume_thread(codex, tid, session_type)
            active_thread = thread
            streamed = await thread.run_streamed(prompt)

            async for event in streamed.events:
                if self._check_interrupted():
                    logger.info("Codex SDK streaming interrupted")
                    yield {"type": "text_delta", "text": "[Session interrupted by user]"}
                    return
                etype = getattr(event, "type", "")
                if etype == "item.completed":
                    item = event.item
                    item_type = getattr(item, "type", "")
                    if item_type == "message":
                        text = _extract_item_text(item)
                        if text:
                            response_text_parts.append(text)
                            yield {"type": "text_delta", "text": text}
                        else:
                            logger.debug(
                                "Codex item.completed message with empty text: %s",
                                repr(item)[:300],
                            )
                    elif item_type == "tool_use":
                        tool_name = getattr(item, "name", "unknown")
                        tool_id = getattr(item, "id", "")
                        yield {
                            "type": "tool_start",
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                        }
                        rec = _item_to_tool_record(item)
                        if rec:
                            all_tool_records.append(rec)
                        yield {
                            "type": "tool_end",
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                        }
                    else:
                        text = _extract_item_text(item)
                        if text:
                            logger.info(
                                "Codex item.completed unhandled type=%s "
                                "but has text (%d chars); emitting",
                                item_type, len(text),
                            )
                            response_text_parts.append(text)
                            yield {"type": "text_delta", "text": text}
                        else:
                            logger.debug(
                                "Codex item.completed unhandled type=%s: %s",
                                item_type, repr(item)[:300],
                            )
                elif etype == "turn.completed":
                    turn_result = _wrap_result_message(event, thread)
                    raw_usage = getattr(event, "usage", None)
                    if raw_usage:
                        ud = _usage_to_dict(raw_usage)
                        usage_acc.input_tokens = ud.get("input_tokens", 0) or ud.get("prompt_tokens", 0) or 0
                        usage_acc.output_tokens = ud.get("output_tokens", 0) or ud.get("completion_tokens", 0) or 0
                        tracker.update(ud, include_output_in_ratio=True)
                    saved_tid = _get_thread_id(thread)
                    if saved_tid:
                        _save_thread_id(self._anima_dir, saved_tid, session_type)
                elif etype == "text.delta":
                    text = getattr(event, "text", "") or getattr(event, "delta", "")
                    if text:
                        response_text_parts.append(text)
                        yield {"type": "text_delta", "text": text}
                elif etype == "response.completed":
                    resp = getattr(event, "response", event)
                    text = _extract_item_text(resp)
                    if text and text not in response_text_parts:
                        response_text_parts.append(text)
                        yield {"type": "text_delta", "text": text}
                else:
                    logger.debug(
                        "Codex unhandled event type=%s attrs=%s",
                        etype,
                        [a for a in dir(event) if not a.startswith("_")][:15],
                    )

        # Try resume first, fallback to fresh thread
        fell_back = False
        if thread_id:
            try:
                gen = _stream_turn(thread_id)
                first_event: dict[str, Any] | None = None
                try:
                    first_event = await asyncio.wait_for(
                        gen.__anext__(), timeout=RESUME_TIMEOUT_SEC,
                    )
                except (asyncio.TimeoutError, StopAsyncIteration):
                    logger.warning(
                        "Codex resume timed out or empty (thread=%s), "
                        "falling back to fresh thread.",
                        thread_id,
                    )
                    _clear_thread_id(self._anima_dir, session_type)
                    fell_back = True
                    await gen.aclose()
                except Exception as e:
                    logger.warning(
                        "Codex resume stream failed (thread=%s): %s",
                        thread_id, e,
                    )
                    _clear_thread_id(self._anima_dir, session_type)
                    fell_back = True
                    await gen.aclose()
                else:
                    if first_event:
                        yield first_event
                    async for ev in gen:
                        yield ev
            except Exception as e:
                logger.warning(
                    "Codex stream resume error: %s. Fresh thread.", e,
                )
                _clear_thread_id(self._anima_dir, session_type)
                fell_back = True
        else:
            fell_back = True

        if fell_back:
            try:
                async for ev in _stream_turn(None):
                    yield ev
            except Exception as e:
                logger.exception("Codex SDK streaming error")
                partial = "\n".join(response_text_parts)
                is_buffer_overflow = _is_limit_overrun(e)
                raise StreamDisconnectedError(
                    f"Codex SDK stream error: {e}",
                    partial_text=partial,
                    immediate_retry=is_buffer_overflow,
                ) from e

        full_text = "\n".join(response_text_parts)
        if not full_text and all_tool_records:
            full_text = _synthesise_fallback(all_tool_records)

        replied_to = self._read_replied_to_file()
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": turn_result,
            "replied_to_from_transcript": replied_to,
            "tool_call_records": [asdict(r) for r in all_tool_records],
            "usage": usage_acc.to_dict(),
        }
