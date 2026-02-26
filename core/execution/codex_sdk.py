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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.execution.base import (
    BaseExecutor,
    ExecutionResult,
    StreamDisconnectedError,
    ToolCallRecord,
    _truncate_for_record,
)
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory

logger = logging.getLogger("animaworks.execution.codex_sdk")

__all__ = ["CodexSDKExecutor", "clear_codex_thread_ids"]

RESUME_TIMEOUT_SEC = 15.0


# ── Model name helpers ───────────────────────────────────────

def _resolve_codex_model(model: str) -> str:
    """Strip the ``codex/`` prefix to get the bare model name for the CLI."""
    if model.startswith("codex/"):
        return model[len("codex/"):]
    return model


def _escape_toml_string(value: str) -> str:
    """Escape a string for safe embedding in a TOML double-quoted value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


# ── Session (thread) ID persistence ──────────────────────────

def _thread_id_path(anima_dir: Path, session_type: str) -> Path:
    return anima_dir / "shortterm" / session_type / "codex_thread_id.txt"


def _save_thread_id(anima_dir: Path, thread_id: str, session_type: str) -> None:
    p = _thread_id_path(anima_dir, session_type)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(thread_id, encoding="utf-8")


def _load_thread_id(anima_dir: Path, session_type: str) -> str | None:
    p = _thread_id_path(anima_dir, session_type)
    if p.is_file():
        tid = p.read_text(encoding="utf-8").strip()
        return tid or None
    return None


def _clear_thread_id(anima_dir: Path, session_type: str) -> None:
    p = _thread_id_path(anima_dir, session_type)
    p.unlink(missing_ok=True)


def clear_codex_thread_ids(anima_dir: Path) -> None:
    """Clear all persisted Codex thread IDs (both chat and heartbeat)."""
    for st in ("chat", "heartbeat"):
        _clear_thread_id(anima_dir, st)


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
    ) -> None:
        super().__init__(model_config, anima_dir)
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
        if api_key:
            env["OPENAI_API_KEY"] = api_key
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

    def _write_codex_config(self, system_prompt: str) -> None:
        """Write CODEX_HOME config.toml and model instructions file.

        The CODEX_HOME lives at ``{anima_dir}/.codex_home/`` and persists
        across sessions so that Codex's thread data (``sessions/``) survives.
        """
        self._codex_home.mkdir(parents=True, exist_ok=True)

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
        from openai_codex_sdk import Codex

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
        session_type = _resolve_session_type(trigger)
        thread_id = _load_thread_id(self._anima_dir, session_type)

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

        tid = _get_thread_id(thread)
        if tid:
            _save_thread_id(self._anima_dir, tid, session_type)

        response_text = getattr(turn, "final_response", "") or ""
        items = getattr(turn, "items", []) or []
        tool_records = _extract_tool_records(items)

        if tracker:
            usage = getattr(turn, "usage", None)
            if usage:
                tracker.update(_usage_to_dict(usage), include_output_in_ratio=True)

        replied_to = self._read_replied_to_file()
        return ExecutionResult(
            text=response_text,
            result_message=turn,
            replied_to_from_transcript=replied_to,
            tool_call_records=tool_records,
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
        session_type = _resolve_session_type(trigger)
        thread_id = _load_thread_id(self._anima_dir, session_type)

        self._write_codex_config(system_prompt)
        codex = self._create_codex_client()

        response_text_parts: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        turn_result: Any = None
        active_thread: Any = None

        async def _stream_turn(tid: str | None) -> AsyncGenerator[dict[str, Any], None]:
            nonlocal turn_result, active_thread
            thread = self._start_or_resume_thread(codex, tid, session_type)
            active_thread = thread
            streamed = await thread.run_streamed(prompt)

            async for event in streamed.events:
                etype = getattr(event, "type", "")
                if etype == "item.completed":
                    item = event.item
                    item_type = getattr(item, "type", "")
                    if item_type == "message":
                        text = _extract_item_text(item)
                        if text:
                            response_text_parts.append(text)
                            yield {"type": "text_delta", "text": text}
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
                elif etype == "turn.completed":
                    turn_result = event
                    usage = getattr(event, "usage", None)
                    if usage:
                        tracker.update(
                            _usage_to_dict(usage),
                            include_output_in_ratio=True,
                        )
                    saved_tid = _get_thread_id(thread)
                    if saved_tid:
                        _save_thread_id(self._anima_dir, saved_tid, session_type)

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
                raise StreamDisconnectedError(
                    f"Codex SDK stream error: {e}",
                    partial_text=partial,
                ) from e

        full_text = "\n".join(response_text_parts) or "(no response)"
        replied_to = self._read_replied_to_file()
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": turn_result,
            "replied_to_from_transcript": replied_to,
            "tool_call_records": [asdict(r) for r in all_tool_records],
        }
