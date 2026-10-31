from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode D executor: cursor-agent CLI wrapper.

Runs cursor-agent as a subprocess with --output-format stream-json --stream-partial-output
and parses NDJSON output. Integrates with AnimaWorks MCP server for tool access.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import Any

from core.anima.workspace_files import AGENTS_FILE, IDENTITY_FILE, SOUL_FILE
from core.execution.base import ExecutionResult, ToolCallRecord, _truncate_for_record, join_answer_parts
from core.execution.cli_stream import CLIStreamExecutor
from core.execution.engine_base import GRACEFUL_KILL_WAIT_SECONDS, engine_error_event
from core.execution.events import done_event, text_delta_event, tool_end_event, tool_start_event
from core.execution.process_runner import ProcessRunner
from core.execution.session.engine_session import MAX_RESUME_TURNS, aload_turn_limited_session, next_turn_count
from core.execution.session.session_context import _resolve_session_type
from core.execution.session.session_ids import EngineSessionIds
from core.i18n import t
from core.memory.conversation.shortterm import ShortTermMemory
from core.platform.atomic_io import atomic_write_json, atomic_write_text
from core.platform.cursor import (
    find_cursor_agent_binary as _find_cursor_agent_binary,
)
from core.platform.cursor import (
    is_cursor_agent_available,
)
from core.platform.subprocess_entries import SubprocessEntry, module_args
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger("animaworks.execution.cursor_agent")

__all__ = ["CursorAgentExecutor", "is_cursor_agent_available", "_MAX_RESUME_TURNS", "_resolve_session_type"]

# ── Constants ───────────────────────────────────────────────────

_MAX_RESUME_TURNS = MAX_RESUME_TURNS
_CURSOR_SESSION_IDS = EngineSessionIds("cursor")


def _format_current_time() -> str:
    """Return a short time-stamp string for resume-turn injection."""
    from core.time_utils import now_local

    return "[" + now_local().strftime("%Y-%m-%d %H:%M %Z") + "]"


# ── Executor ───────────────────────────────────────────────────


class CursorAgentExecutor(CLIStreamExecutor):
    """Execute via cursor-agent CLI (Mode D).

    Spawns cursor-agent as a subprocess with NDJSON streaming output.
    MCP integration with core/mcp/server.py provides AnimaWorks tools.
    """

    engine_mode = "D"
    session_engine = "cursor"

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
        self._workspace = anima_dir / ".cursor-workspace"

    # ── Helpers ─────────────────────────────────────────────────

    def _find_binary(self) -> str | None:
        """Return path to cursor-agent binary, or None."""
        return _find_cursor_agent_binary()

    def _ensure_workspace(self) -> None:
        """Create workspace and .cursor directories."""
        self._workspace.mkdir(parents=True, exist_ok=True)
        (self._workspace / ".cursor").mkdir(parents=True, exist_ok=True)
        self._ignore_root_agents_md()

    def _ignore_root_agents_md(self) -> None:
        """Stop cursor-agent from auto-loading the anima-root AGENTS.md.

        cursor-agent walks from the workspace to its ancestors and loads
        ``AGENTS.md`` as project rules.  The workspace lives under the anima
        directory, so the root ``AGENTS.md`` would be injected a second time
        on top of the prompt.  A root-relative ``.cursorignore`` entry
        suppresses only that file while preserving other project rules.
        """
        ignore_path = self._anima_dir / ".cursorignore"
        pattern = f"/{AGENTS_FILE}"
        try:
            existing = ignore_path.read_text(encoding="utf-8") if ignore_path.exists() else ""
        except OSError:
            logger.debug("Could not read %s", ignore_path, exc_info=True)
            return
        if pattern in existing.splitlines():
            return
        prefix = existing.rstrip("\n") + "\n" if existing.strip() else ""
        try:
            atomic_write_text(ignore_path, f"{prefix}{pattern}\n")
        except OSError:
            logger.warning("Could not update %s to ignore %s", ignore_path, AGENTS_FILE, exc_info=True)

    def _write_mcp_config(self) -> None:
        """Write .cursor/mcp.json with AnimaWorks MCP server config."""
        import sys

        from core.execution.mcp_env import build_mcp_env

        mcp_dir = self._workspace / ".cursor"
        mcp_dir.mkdir(parents=True, exist_ok=True)
        mcp_path = mcp_dir / "mcp.json"
        mcp_env = build_mcp_env(self._anima_dir)
        config = {
            "mcpServers": {
                "aw": {
                    "command": sys.executable,
                    "args": list(module_args(SubprocessEntry.MCP_SERVER)),
                    "env": mcp_env,
                }
            }
        }
        atomic_write_json(mcp_path, config, indent=2, ensure_ascii=True, trailing_newline=False)

    def _write_cursor_rules(self) -> None:
        """Write static identity/rules to .cursor/rules/ for compaction resilience.

        Writes IDENTITY.md, SOUL.md, behavior_rules.md, and permissions.md
        into the cursor-agent workspace so they are loaded as persistent
        rules that survive context compaction.  Skips writing when the file
        content has not changed (I/O reduction).
        """
        rules_dir = self._workspace / ".cursor" / "rules"
        try:
            rules_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Cannot create .cursor/rules/ directory")
            return

        file_sources: dict[str, Path] = {
            IDENTITY_FILE: self._anima_dir / IDENTITY_FILE,
            SOUL_FILE: self._anima_dir / SOUL_FILE,
            "permissions.md": self._anima_dir / "permissions.md",
        }
        for name, src in file_sources.items():
            if not src.is_file():
                continue
            try:
                content = src.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not content:
                continue
            dest = rules_dir / name
            if dest.exists() and dest.read_text(encoding="utf-8") == content:
                continue
            dest.write_text(content, encoding="utf-8")

        try:
            from core.paths import load_prompt

            br_content = load_prompt("behavior_rules")
            if br_content:
                dest = rules_dir / "behavior_rules.md"
                if not dest.exists() or dest.read_text(encoding="utf-8") != br_content:
                    dest.write_text(br_content, encoding="utf-8")
        except Exception:
            logger.debug("Could not write behavior_rules to .cursor/rules", exc_info=True)

    def _resolve_cursor_model(self) -> str:
        """Strip cursor/ prefix from model name."""
        model = self._model_config.model
        if model.startswith("cursor/"):
            return model[len("cursor/") :]
        return model

    def _build_command(self, prompt: str, *, resume_chat_id: str | None = None) -> list[str]:
        """Build CLI command for cursor-agent."""
        binary = self._find_binary()
        if not binary:
            return []
        cmd = [
            binary,
            "-p",
            "--force",
            "--trust",
            "--approve-mcps",
            "--workspace",
            str(self._workspace),
            "--model",
            self._resolve_cursor_model(),
            "--output-format",
            "stream-json",
            "--stream-partial-output",
        ]
        if resume_chat_id:
            cmd.extend(["--resume", resume_chat_id])
        cmd.append(prompt)
        return cmd

    def _build_env(self) -> dict[str, str]:
        """Build environment for subprocess.

        cursor-agent authenticates via ``agent login`` (stored in
        ``~/.cursor-agent/``).  We do NOT inject ``CURSOR_API_KEY`` from
        AnimaWorks credentials — only pass it through if it's already
        set in the host environment.
        """
        env = dict(os.environ)
        from core.execution.workspace_env import update_workspace_env

        update_workspace_env(env, self._anima_dir)  # Includes ANIMAWORKS_SCRATCH_DIR and shared tool caches.
        for key in ("PATH", "HOME", "LANG"):
            if key not in env:
                val = os.environ.get(key)
                if val:
                    env[key] = val
        from core.execution.github_identity import resolve_github_token_env

        env.update(resolve_github_token_env(self._anima_dir))
        return env

    def _extract_tool_record(self, tc: dict[str, Any]) -> ToolCallRecord | None:
        """Parse tool_call event data into ToolCallRecord."""
        tool_name = ""
        tool_id = tc.get("id", "") or tc.get("tool_call_id", "")
        input_summary = ""
        result_summary = ""
        is_error = False

        for key in ("readToolCall", "writeToolCall", "editToolCall"):
            if key in tc:
                tool_name = key.replace("ToolCall", "").lower()
                if tool_name == "read":
                    tool_name = "Read"
                elif tool_name == "write":
                    tool_name = "Write"
                elif tool_name == "edit":
                    tool_name = "Edit"
                args = tc.get(key, {})
                if isinstance(args, dict):
                    input_summary = _truncate_for_record(str(args), 500)
                break

        if not tool_name and "function" in tc:
            fn = tc["function"]
            if isinstance(fn, dict):
                tool_name = fn.get("name", "") or "unknown"
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    input_summary = _truncate_for_record(args, 500)
                else:
                    input_summary = _truncate_for_record(str(args), 500)
            else:
                tool_name = str(fn)

        if not tool_name:
            tool_name = tc.get("name", "unknown")

        if tool_name.startswith("mcp__aw__"):
            tool_name = tool_name[len("mcp__aw__") :]

        result = tc.get("result") or tc.get("output") or tc.get("content")
        if result is not None:
            result_summary = _truncate_for_record(str(result), 500)
        if tc.get("is_error") or tc.get("error"):
            is_error = True

        return ToolCallRecord(
            tool_name=tool_name,
            tool_id=str(tool_id),
            input_summary=input_summary,
            result_summary=result_summary,
            is_error=is_error,
        )

    # ── Execution ───────────────────────────────────────────────

    async def _execute_cli_turn(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        thread_id: str = "default",
        _event_sink: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> ExecutionResult:
        """Run cursor-agent subprocess and parse NDJSON output.

        For resumable triggers (currently ``chat`` only), persists the
        cursor-agent session ID to disk and passes ``--resume <chatId>``
        on subsequent calls.

        Turn-based session rotation (A+C hybrid):

        * Turns 2–N (resume): only the current time is injected; the full
          system prompt is skipped to reduce context bloat.
        * When ``turn_count >= _MAX_RESUME_TURNS``, the chatId is cleared
          and a fresh session starts with the full system prompt.
        * Static identity/rules are persisted in ``.cursor/rules/`` so
          they survive cursor-agent's internal context compaction.
        """
        if self._check_interrupted():
            return ExecutionResult(text="[Session interrupted by user]")

        binary = self._find_binary()
        if not binary:
            text = t("cursor_agent.not_installed")
            self._error_metadata(text)
            return ExecutionResult(text=text, error=True, reason="unknown")

        self._ensure_workspace()
        self._write_mcp_config()
        self._write_cursor_rules()

        session_type = _resolve_session_type(trigger)
        decision = await aload_turn_limited_session(self._anima_dir, "cursor", trigger, thread_id)
        is_resumable = decision.resumable
        turn_count = decision.turn_count
        resume_chat_id = decision.session_id

        # ── Build combined prompt ──────────────────────────
        time_prefix = _format_current_time()
        if resume_chat_id:
            combined_prompt = time_prefix + "\n\n" + prompt
        else:
            if system_prompt:
                combined_prompt = (
                    "<system_context>\n" + system_prompt + "\n</system_context>\n\n" + time_prefix + "\n\n" + prompt
                )
            else:
                combined_prompt = time_prefix + "\n\n" + prompt

        if resume_chat_id:
            logger.info(
                "Resuming cursor-agent session %s (turn=%d, type=%s, thread=%s)",
                resume_chat_id[:12],
                turn_count + 1,
                session_type,
                thread_id,
            )

        rotated_during_run = False
        result, session_id, failed = await self._run_subprocess(
            combined_prompt,
            resume_chat_id=resume_chat_id,
            event_sink=_event_sink,
        )

        if failed and resume_chat_id:
            logger.warning(
                "Session resume failed (chat_id=%s), retrying with fresh session",
                resume_chat_id[:12],
            )
            await _CURSOR_SESSION_IDS.aclear(self._anima_dir, session_type, thread_id)
            if system_prompt:
                fresh_prompt = (
                    "<system_context>\n" + system_prompt + "\n</system_context>\n\n" + time_prefix + "\n\n" + prompt
                )
            else:
                fresh_prompt = time_prefix + "\n\n" + prompt
            result, session_id, _failed = await self._run_subprocess(
                fresh_prompt,
                resume_chat_id=None,
                event_sink=_event_sink,
            )
            rotated_during_run = True

        # ── Persist session state ──────────────────────────
        new_turn, rotation_pending = next_turn_count(
            decision,
            resumed=resume_chat_id is not None,
            rotated_during_run=rotated_during_run,
        )

        if session_id and is_resumable:
            await _CURSOR_SESSION_IDS.asave(self._anima_dir, session_id, session_type, thread_id, new_turn)
            logger.debug(
                "Saved cursor-agent chat_id %s turn=%d for %s/%s",
                session_id[:12],
                new_turn,
                session_type,
                thread_id,
            )

        result.session_rotation_pending = rotation_pending

        return result

    async def _stream_events(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        trigger: str = "",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream normalized Cursor CLI events while the child is running."""
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def _run_turn() -> ExecutionResult:
            try:
                return await self._execute_cli_turn(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    tracker=tracker,
                    trigger=trigger,
                    images=images,
                    prior_messages=prior_messages,
                    thread_id=thread_id,
                    _event_sink=queue.put,
                )
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_run_turn())
        streamed_text = False
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                streamed_text = streamed_text or event.get("type") == "text_delta"
                yield event
            result = await task
            if result.text and not streamed_text:
                yield text_delta_event(result.text)
            if result.error:
                yield engine_error_event(result.text, {"terminal": True, "reason": result.reason})
            yield done_event(
                result.text,
                **({"error": True, "reason": result.reason} if result.error else {}),
                result_message=result.result_message,
                replied_to_from_transcript=result.replied_to_from_transcript,
                tool_call_records=[record.__dict__ for record in result.tool_call_records],
                usage=result.usage.to_dict() if result.usage else None,
                session_rotation_pending=result.session_rotation_pending,
                truncated=result.truncated,
                stop_kind="interrupted" if self._check_interrupted() else "normal",
            )
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run_subprocess(
        self,
        combined_prompt: str,
        *,
        resume_chat_id: str | None = None,
        event_sink: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> tuple[ExecutionResult, str | None, bool]:
        """Spawn cursor-agent and parse its NDJSON output.

        Returns ``(result, session_id, failed)`` where *failed* is True
        when the process exited with a non-zero code for a non-auth reason.
        """
        cmd = self._build_command(combined_prompt, resume_chat_id=resume_chat_id)
        env = self._build_env()

        answer_parts: list[str] = []
        current_turn_chunks: list[str] = []
        tool_records: list[ToolCallRecord] = []
        session_id: str | None = None
        failed = False
        error_reason = ""
        started_tools: set[str] = set()

        async def _emit(event: dict[str, Any]) -> None:
            if event_sink is not None:
                await event_sink(event)

        def _flush_current_turn() -> None:
            if current_turn_chunks:
                answer_parts.append("".join(current_turn_chunks))
                current_turn_chunks.clear()

        def _full_text() -> str:
            return join_answer_parts([*answer_parts, "".join(current_turn_chunks)])

        proc: asyncio.subprocess.Process | None = None
        try:
            process_runner = ProcessRunner(graceful_timeout=GRACEFUL_KILL_WAIT_SECONDS)
            proc = await process_runner.start(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                async with asyncio.timeout(None):
                    assert proc.stdout is not None
                    async for line in self.iter_lines(proc.stdout):
                        if self._check_interrupted():
                            await self._kill_process(proc)
                            return (
                                ExecutionResult(
                                    text=_full_text() or "[Session interrupted by user]",
                                    tool_call_records=tool_records,
                                ),
                                session_id,
                                False,
                            )

                        event = self._parse_ndjson_event(line.decode("utf-8", errors="replace").strip())
                        if event is None:
                            continue

                        etype = event.get("type", "")

                        if etype == "system" and event.get("subtype") == "init":
                            sid = event.get("session_id")
                            if sid:
                                session_id = str(sid)

                        elif etype == "assistant":
                            content_list = event.get("message", {}).get("content", [])
                            parts: list[str] = []
                            for item in content_list:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    parts.append(item.get("text", ""))
                                elif isinstance(item, str):
                                    parts.append(item)
                            if parts:
                                current_turn_chunks.extend(parts)
                                for text_part in parts:
                                    await _emit(text_delta_event(text_part))

                        elif etype == "tool_call":
                            _flush_current_turn()
                            subtype = event.get("subtype", "")
                            tc = event.get("tool_call", {})
                            record = self._extract_tool_record(tc)
                            if record:
                                tool_key = record.tool_id or f"{record.tool_name}:{len(started_tools)}"
                                if subtype == "started" and tool_key not in started_tools:
                                    started_tools.add(tool_key)
                                    await _emit(
                                        tool_start_event(
                                            record.tool_name,
                                            record.tool_id,
                                            input=record.input_summary,
                                        )
                                    )
                                elif subtype == "completed":
                                    if tool_key not in started_tools:
                                        started_tools.add(tool_key)
                                        await _emit(
                                            tool_start_event(
                                                record.tool_name,
                                                record.tool_id,
                                                input=record.input_summary,
                                            )
                                        )
                                    tool_records.append(record)
                                    await _emit(
                                        tool_end_event(
                                            record.tool_name,
                                            record.tool_id,
                                            result=record.result_summary,
                                            is_error=record.is_error,
                                        )
                                    )

                        elif etype == "result":
                            result_text = event.get("result", "")
                            if result_text and not _full_text():
                                current_turn_chunks.append(result_text)
                                await _emit(text_delta_event(result_text))

            except TimeoutError:
                logger.warning("Cursor agent timed out after %ds", self.event_idle_timeout_seconds)
                await self._kill_process(proc)
                timeout_msg = t("cursor_agent.timeout", timeout=self.event_idle_timeout_seconds)
                full_text = _full_text()
                timeout_text = f"\n\n{timeout_msg}" if full_text else timeout_msg
                await _emit(text_delta_event(timeout_text))
                self._error_metadata(timeout_msg)
                return (
                    ExecutionResult(
                        text=full_text + f"\n\n{timeout_msg}" if full_text else timeout_msg,
                        tool_call_records=tool_records,
                        error=True,
                        reason="timeout",
                    ),
                    session_id,
                    True,
                )

            await proc.wait()
            await process_runner.close()
            stderr_bytes = await process_runner.stderr()

            if proc.returncode != 0:
                failed = True
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                logger.warning(
                    "Cursor agent exited with code %d: %s",
                    proc.returncode,
                    stderr_text[:500],
                )
                metadata = self._error_metadata(stderr_text)
                is_auth_error = any(word in stderr_text.lower() for word in ("auth", "login", "unauthorized"))
                error_reason = "auth" if is_auth_error else str(metadata.get("reason") or "unknown")
                if is_auth_error:
                    error_text = t("cursor_agent.not_authenticated")
                    await _emit(text_delta_event(error_text))
                    return (
                        ExecutionResult(text=error_text, error=True, reason=error_reason),
                        session_id,
                        False,
                    )
                if not _full_text():
                    error_text = f"[Cursor Agent Error (exit {proc.returncode}): {stderr_text[:500]}]"
                    current_turn_chunks.append(error_text)
                    await _emit(text_delta_event(error_text))

        except FileNotFoundError:
            error_text = t("cursor_agent.not_installed")
            await _emit(text_delta_event(error_text))
            self._error_metadata(error_text)
            return (ExecutionResult(text=error_text, error=True, reason="unknown"), None, True)
        except Exception as e:
            logger.exception("Cursor agent execution error")
            metadata = self._error_metadata(str(e))
            error_text = f"[Cursor Agent Error: {e}]"
            await _emit(text_delta_event(error_text))
            return (
                ExecutionResult(text=error_text, error=True, reason=str(metadata.get("reason") or "unknown")),
                None,
                True,
            )
        finally:
            # Ensure the subprocess is killed on CancelledError or any
            # other exception that bypasses the normal exit path.
            if proc is not None:
                logger.debug("Closing cursor-agent process runner (PID %s)", proc.pid)
                await process_runner.close()

        replied_to = self._read_replied_to_file()
        return (
            ExecutionResult(
                text=_full_text(),
                replied_to_from_transcript=replied_to,
                tool_call_records=tool_records,
                error=bool(error_reason),
                reason=error_reason,
            ),
            session_id,
            failed,
        )
