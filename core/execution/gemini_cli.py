from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode G executor: Gemini CLI wrapper.

Runs Google's Gemini CLI as a subprocess with ``--output-format stream-json``
and parses NDJSON output.  Integrates with AnimaWorks MCP server for tool
access via ``.gemini/settings.json``.

System prompt is injected via ``GEMINI_SYSTEM_MD`` environment variable
pointing to a temporary file, completely replacing the CLI's default system
prompt.  Authentication is delegated to the CLI itself (``GEMINI_API_KEY``
passthrough or ``gemini auth login``).
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from core.execution.base import (
    BaseExecutor,
    ExecutionResult,
    TokenUsage,
    ToolCallRecord,
    _truncate_for_record,
)
from core.i18n import t
from core.memory.shortterm import ShortTermMemory
from core.prompt.context import ContextTracker
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger("animaworks.execution.gemini_cli")

__all__ = ["GeminiCLIExecutor", "is_gemini_cli_available"]

# ── Constants ───────────────────────────────────────────────────

_GEMINI_BINARY_NAMES = ("gemini",)
_DEFAULT_TIMEOUT_SECONDS = 600
_GRACEFUL_KILL_WAIT = 3.0


# ── Binary discovery ───────────────────────────────────────────


def _find_gemini_binary() -> str | None:
    """Return path to gemini CLI binary, or None if not found."""
    for name in _GEMINI_BINARY_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def is_gemini_cli_available() -> bool:
    """Return True when gemini CLI is available on PATH."""
    return _find_gemini_binary() is not None


def _resolve_gemini_model(model: str) -> str:
    """Strip the ``gemini/`` prefix to get the bare model name for the CLI.

    AnimaWorks uses ``gemini/2.5-pro`` but the CLI expects ``gemini-2.5-pro``.
    """
    if model.startswith("gemini/"):
        bare = model[len("gemini/") :]
        if not bare.startswith("gemini-"):
            return f"gemini-{bare}"
        return bare
    return model


# ── Executor ───────────────────────────────────────────────────


class GeminiCLIExecutor(BaseExecutor):
    """Execute via Gemini CLI (Mode G).

    Spawns gemini CLI as a subprocess with stream-json NDJSON output.
    MCP integration with core/mcp/server.py provides AnimaWorks tools.
    """

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
        self._workspace = anima_dir / ".gemini-workspace"
        self._prompt_files: list[Path] = []

    # ── Helpers ─────────────────────────────────────────────────

    def _ensure_workspace(self) -> None:
        """Create workspace and .gemini directories.

        Copies auth credential files from the default ``~/.gemini``
        directory so that CLI-delegated OAuth login works even when
        ``GEMINI_CLI_HOME`` points to a per-Anima workspace.
        """
        gemini_dir = self._workspace / ".gemini"
        self._workspace.mkdir(parents=True, exist_ok=True)
        gemini_dir.mkdir(parents=True, exist_ok=True)

        default_home = Path.home() / ".gemini"
        if default_home.is_dir():
            for name in (
                "oauth_creds.json",
                "google_accounts.json",
                "installation_id",
                "state.json",
            ):
                src = default_home / name
                dst = gemini_dir / name
                if src.is_file() and not dst.exists():
                    try:
                        import shutil as _shutil

                        _shutil.copy2(src, dst)
                    except OSError:
                        pass

    def _write_settings(self) -> None:
        """Write .gemini/settings.json merging MCP config with existing auth."""
        import sys

        from core.paths import PROJECT_DIR

        settings_path = self._workspace / ".gemini" / "settings.json"

        existing: dict = {}
        if settings_path.is_file():
            try:
                existing = json.loads(settings_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        if not existing.get("security", {}).get("auth"):
            default_settings = Path.home() / ".gemini" / "settings.json"
            if default_settings.is_file():
                try:
                    default = json.loads(default_settings.read_text(encoding="utf-8"))
                    if "security" in default:
                        existing.setdefault("security", {}).update(default["security"])
                except (json.JSONDecodeError, OSError):
                    pass

        existing["mcpServers"] = {
            "aw": {
                "command": sys.executable,
                "args": ["-m", "core.mcp.server"],
                "env": {
                    "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
                    "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
                    "PYTHONPATH": str(PROJECT_DIR),
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                },
            }
        }

        settings_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def _write_system_prompt(self, system_prompt: str) -> Path:
        """Write system prompt to a temporary file and return the path."""
        fd, path = tempfile.mkstemp(suffix=".md", prefix="aw_gemini_sys_")
        p = Path(path)
        p.write_text(system_prompt, encoding="utf-8")
        os.close(fd)
        self._prompt_files.append(p)
        return p

    def _cleanup_prompt_files(self) -> None:
        """Remove temporary system prompt files."""
        for p in self._prompt_files:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temp prompt file: %s", p)
        self._prompt_files.clear()

    def _build_command(self, prompt: str) -> list[str]:
        """Build CLI command for gemini."""
        binary = _find_gemini_binary()
        if not binary:
            return []
        model = _resolve_gemini_model(self._model_config.model)
        return [
            binary,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--approval-mode",
            "yolo",
            "-m",
            model,
        ]

    def _build_env(self, system_prompt_path: Path | None = None) -> dict[str, str]:
        """Build environment for subprocess.

        Gemini CLI authenticates via ``gemini auth login`` or
        ``GEMINI_API_KEY``.  We pass through the host environment and
        optionally inject ``GEMINI_SYSTEM_MD`` for system prompt override.
        """
        env = dict(os.environ)
        env["GEMINI_CLI_HOME"] = str(self._workspace)
        if system_prompt_path is not None:
            env["GEMINI_SYSTEM_MD"] = str(system_prompt_path)
        api_key = self._resolve_api_key()
        if api_key:
            env["GEMINI_API_KEY"] = api_key
        return env

    def _resolve_api_key(self) -> str | None:
        """Resolve GEMINI_API_KEY from config or environment."""
        if self._model_config.api_key:
            return self._model_config.api_key
        env_name = self._model_config.api_key_env
        if env_name:
            return os.environ.get(env_name)
        return os.environ.get("GEMINI_API_KEY")

    def _parse_ndjson_event(self, stdout_line: str) -> dict[str, Any] | None:
        """Parse a single NDJSON line. Return dict or None on parse error."""
        line = stdout_line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse NDJSON line: %s — %s", line[:200], e)
            return None

    async def _kill_process(self, proc: asyncio.subprocess.Process, timeout: float = _GRACEFUL_KILL_WAIT) -> None:
        """Graceful kill: SIGTERM → wait → SIGKILL."""
        if proc.returncode is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
            await asyncio.sleep(timeout)
        except ProcessLookupError:
            return
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def _extract_tool_record(self, event: dict[str, Any], result_event: dict[str, Any] | None = None) -> ToolCallRecord:
        """Build a ToolCallRecord from a tool_use event, optionally paired with tool_result."""
        tool_name = event.get("tool_name", "unknown")
        tool_id = event.get("tool_id", "")
        params = event.get("parameters", {})
        input_summary = _truncate_for_record(json.dumps(params, ensure_ascii=False), 500) if params else ""

        if tool_name.startswith("mcp_aw_"):
            tool_name = tool_name[len("mcp_aw_") :]

        result_summary = ""
        is_error = False
        if result_event:
            status = result_event.get("status", "")
            is_error = status == "error"
            output = result_event.get("output") or result_event.get("error", {}).get("message", "")
            if output:
                result_summary = _truncate_for_record(str(output), 500)

        return ToolCallRecord(
            tool_name=tool_name,
            tool_id=str(tool_id),
            input_summary=input_summary,
            result_summary=result_summary,
            is_error=is_error,
        )

    def _parse_stats(self, stats: dict[str, Any] | None) -> TokenUsage | None:
        """Extract TokenUsage from the result stats object."""
        if stats is None:
            return None
        return TokenUsage(
            input_tokens=stats.get("input_tokens", 0),
            output_tokens=stats.get("output_tokens", 0),
            cache_read_tokens=stats.get("cached", 0),
        )

    # ── Execution ───────────────────────────────────────────────

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
        """Run gemini CLI subprocess and parse stream-json output."""
        if self._check_interrupted():
            return ExecutionResult(text="[Session interrupted by user]")

        binary = _find_gemini_binary()
        if not binary:
            return ExecutionResult(text=t("gemini_cli.not_installed"))

        self._ensure_workspace()
        self._write_settings()

        sys_prompt_path: Path | None = None
        if system_prompt:
            sys_prompt_path = self._write_system_prompt(system_prompt)

        cmd = self._build_command(prompt)
        env = self._build_env(sys_prompt_path)

        accumulated_text = ""
        tool_records: list[ToolCallRecord] = []
        pending_tools: dict[str, dict[str, Any]] = {}
        usage: TokenUsage | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self._workspace),
            )

            try:
                async with asyncio.timeout(_DEFAULT_TIMEOUT_SECONDS):
                    assert proc.stdout is not None
                    while True:
                        line = await proc.stdout.readline()
                        if not line:
                            break
                        if self._check_interrupted():
                            await self._kill_process(proc)
                            return ExecutionResult(
                                text=accumulated_text or "[Session interrupted by user]",
                                tool_call_records=tool_records,
                            )

                        event = self._parse_ndjson_event(line.decode("utf-8", errors="replace"))
                        if event is None:
                            continue

                        etype = event.get("type", "")

                        if etype == "message" and event.get("role") == "assistant":
                            content = event.get("content", "")
                            if event.get("delta"):
                                accumulated_text += content
                            else:
                                accumulated_text = content

                        elif etype == "tool_use":
                            tid = event.get("tool_id", "")
                            pending_tools[tid] = event

                        elif etype == "tool_result":
                            tid = event.get("tool_id", "")
                            tool_use_evt = pending_tools.pop(tid, None)
                            if tool_use_evt:
                                record = self._extract_tool_record(tool_use_evt, event)
                                tool_records.append(record)

                        elif etype == "result":
                            usage = self._parse_stats(event.get("stats"))
                            if event.get("status") == "error":
                                err = event.get("error", {})
                                err_msg = err.get("message", "") if isinstance(err, dict) else str(err)
                                if err_msg and not accumulated_text:
                                    accumulated_text = f"[Gemini CLI Error: {err_msg}]"

                        elif etype == "error":
                            severity = event.get("severity", "warning")
                            msg = event.get("message", "")
                            if severity == "error":
                                logger.warning("Gemini CLI error event: %s", msg)
                            else:
                                logger.debug("Gemini CLI warning: %s", msg)

            except TimeoutError:
                logger.warning("Gemini CLI timed out after %ds", _DEFAULT_TIMEOUT_SECONDS)
                await self._kill_process(proc)
                timeout_msg = t("gemini_cli.timeout", timeout=_DEFAULT_TIMEOUT_SECONDS)
                return ExecutionResult(
                    text=accumulated_text + f"\n\n{timeout_msg}" if accumulated_text else timeout_msg,
                    tool_call_records=tool_records,
                    usage=usage,
                )

            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            await proc.wait()

            if proc.returncode != 0:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                logger.warning(
                    "Gemini CLI exited with code %d: %s",
                    proc.returncode,
                    stderr_text[:500],
                )
                if any(kw in stderr_text.lower() for kw in ("auth", "login", "unauthorized", "unauthenticated")):
                    return ExecutionResult(text=t("gemini_cli.not_authenticated"))
                if not accumulated_text:
                    accumulated_text = f"[Gemini CLI Error (exit {proc.returncode}): {stderr_text[:500]}]"

        except FileNotFoundError:
            return ExecutionResult(text=t("gemini_cli.not_installed"))
        except Exception as e:
            logger.exception("Gemini CLI execution error")
            return ExecutionResult(text=f"[Gemini CLI Error: {e}]")
        finally:
            self._cleanup_prompt_files()

        return ExecutionResult(
            text=accumulated_text,
            tool_call_records=tool_records,
            usage=usage,
        )

    # ── Streaming ───────────────────────────────────────────────

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
        """Streaming execution yielding events from gemini CLI stream-json."""
        if self._check_interrupted():
            yield {
                "type": "done",
                "full_text": "[Session interrupted by user]",
                "result_message": None,
                "tool_call_records": [],
            }
            return

        binary = _find_gemini_binary()
        if not binary:
            text = t("gemini_cli.not_installed")
            yield {"type": "text_delta", "text": text}
            yield {"type": "done", "full_text": text, "result_message": None, "tool_call_records": []}
            return

        self._ensure_workspace()
        self._write_settings()

        sys_prompt_path: Path | None = None
        if system_prompt:
            sys_prompt_path = self._write_system_prompt(system_prompt)

        cmd = self._build_command(prompt)
        env = self._build_env(sys_prompt_path)

        accumulated_text = ""
        tool_records: list[ToolCallRecord] = []
        pending_tools: dict[str, dict[str, Any]] = {}
        usage: TokenUsage | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self._workspace),
            )

            try:
                async with asyncio.timeout(_DEFAULT_TIMEOUT_SECONDS):
                    assert proc.stdout is not None
                    while True:
                        line = await proc.stdout.readline()
                        if not line:
                            break
                        if self._check_interrupted():
                            await self._kill_process(proc)
                            yield {
                                "type": "done",
                                "full_text": accumulated_text or "[Session interrupted by user]",
                                "result_message": None,
                                "tool_call_records": [r.__dict__ for r in tool_records],
                            }
                            return

                        event = self._parse_ndjson_event(line.decode("utf-8", errors="replace"))
                        if event is None:
                            continue

                        etype = event.get("type", "")

                        if etype == "message" and event.get("role") == "assistant":
                            content = event.get("content", "")
                            if event.get("delta"):
                                accumulated_text += content
                                yield {"type": "text_delta", "text": content}
                            elif content:
                                accumulated_text = content
                                yield {"type": "text_delta", "text": content}

                        elif etype == "tool_use":
                            tid = event.get("tool_id", "")
                            tool_name = event.get("tool_name", "unknown")
                            if tool_name.startswith("mcp_aw_"):
                                tool_name = tool_name[len("mcp_aw_") :]
                            pending_tools[tid] = event
                            yield {
                                "type": "tool_start",
                                "tool_name": tool_name,
                                "tool_id": tid,
                                "input": event.get("parameters", {}),
                            }

                        elif etype == "tool_result":
                            tid = event.get("tool_id", "")
                            tool_use_evt = pending_tools.pop(tid, None)
                            if tool_use_evt:
                                record = self._extract_tool_record(tool_use_evt, event)
                                tool_records.append(record)
                                yield {
                                    "type": "tool_end",
                                    "tool_id": tid,
                                    "tool_name": record.tool_name,
                                    "result": record.result_summary,
                                    "is_error": record.is_error,
                                }

                        elif etype == "result":
                            usage = self._parse_stats(event.get("stats"))

                        elif etype == "error":
                            severity = event.get("severity", "warning")
                            msg = event.get("message", "")
                            if severity == "error":
                                logger.warning("Gemini CLI error event: %s", msg)

            except TimeoutError:
                logger.warning("Gemini CLI timed out after %ds", _DEFAULT_TIMEOUT_SECONDS)
                await self._kill_process(proc)
                timeout_msg = t("gemini_cli.timeout", timeout=_DEFAULT_TIMEOUT_SECONDS)
                yield {"type": "text_delta", "text": f"\n\n{timeout_msg}"}

            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            await proc.wait()

            if proc.returncode != 0:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                logger.warning("Gemini CLI exited with code %d: %s", proc.returncode, stderr_text[:500])
                if any(kw in stderr_text.lower() for kw in ("auth", "login", "unauthorized", "unauthenticated")):
                    err_text = t("gemini_cli.not_authenticated")
                    if not accumulated_text:
                        accumulated_text = err_text
                    yield {"type": "text_delta", "text": err_text}

        except FileNotFoundError:
            text = t("gemini_cli.not_installed")
            yield {"type": "text_delta", "text": text}
            accumulated_text = text
        except Exception as e:
            logger.exception("Gemini CLI streaming error")
            err = f"[Gemini CLI Error: {e}]"
            yield {"type": "text_delta", "text": err}
            accumulated_text = err
        finally:
            self._cleanup_prompt_files()

        yield {
            "type": "done",
            "full_text": accumulated_text,
            "result_message": None,
            "tool_call_records": [r.__dict__ for r in tool_records],
            "usage": usage.to_dict() if usage else None,
        }
