from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""SDK option building for AgentSDKExecutor (Mixin).

The methods here access ``BaseExecutor`` attributes (``_model_config``,
``_anima_dir``, ``_task_cwd``, etc.) via ``self``.  They are mixed into
``AgentSDKExecutor`` alongside ``BaseExecutor``.
"""

import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError:
        pass

from core.execution._sdk_hooks import (
    _build_post_tool_hook,
    _build_pre_compact_hook,
    _build_pre_tool_hook,
    _build_stop_hook,
)
from core.execution._sdk_session import (
    _PROMPT_FILE_THRESHOLD,
    _SDK_MAX_BUFFER_SIZE,
    _is_debug_superuser,
)

logger = logging.getLogger("animaworks.execution.agent_sdk")

# ── Cached CLI path resolution ────────────────────────────────────
# The SDK bundled ``claude.exe`` can intermittently fail to launch on
# Windows (antivirus hold, file locking, etc.) even though the file
# exists.  By resolving a *verified* CLI path once and passing it as
# ``cli_path`` we bypass the SDK's own bundled-first logic and prefer
# the npm-installed wrapper which is far more reliable.
_cached_cli_path: str | None = None
_cli_path_resolved: bool = False


def _build_sdk_path_env(anima_dir: Path, project_dir: Path) -> str:
    """Build PATH for Claude Code subprocesses.

    Ensure repo-local CLI entry points such as ``animaworks-tool`` are
    resolvable even when the parent server process was started without the
    venv's Scripts/bin directory on PATH.
    """
    sep = os.pathsep
    existing = os.environ.get("PATH", "/usr/bin:/bin")
    candidates: list[str] = [str(anima_dir)]

    launcher_dir = str(Path(sys.executable).resolve().parent)
    candidates.append(launcher_dir)

    venv_bin = project_dir / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    candidates.append(str(venv_bin))

    merged: list[str] = []
    for part in [*candidates, *existing.split(sep)]:
        part = part.strip()
        if part and part not in merged:
            merged.append(part)
    return sep.join(merged)


def _resolve_sdk_cli_path() -> str | None:
    """Return a verified Claude Code CLI path, cached after first call."""
    global _cached_cli_path, _cli_path_resolved
    if _cli_path_resolved:
        return _cached_cli_path

    from core.platform.claude_code import get_claude_executable

    _cached_cli_path = get_claude_executable()
    _cli_path_resolved = True
    if _cached_cli_path:
        logger.info("Resolved Claude Code CLI: %s", _cached_cli_path)
    else:
        logger.warning("Could not resolve a verified Claude Code CLI; SDK will fall back to its own discovery")
    return _cached_cli_path


class SDKOptionsMixin:
    """SDK option building methods mixed into AgentSDKExecutor.

    Requires ``BaseExecutor`` attributes:
    ``_model_config``, ``_anima_dir``, ``_task_cwd``,
    ``_has_subordinates()``, ``_resolve_api_key()``,
    ``_hb_soft_timeout_s``.
    """

    @property
    def _extra_mcp_servers(self) -> dict[str, dict]:
        return self._model_config.extra_mcp_servers or {}

    def _resolve_agent_sdk_model(self) -> str:
        """Return the model name suitable for Agent SDK (strip provider prefix)."""
        import re

        m = self._model_config.model
        m = re.sub(
            r"^(anthropic|bedrock|vertex_ai)/"
            r"([a-z]{2}\.anthropic\.)?",
            "",
            m,
        )
        return m

    def _build_env(self) -> dict[str, str]:
        """Build env dict for the Claude Code child process.

        Authentication mode is determined by ``mode_s_auth`` (per-Anima
        setting from status.json / anima_defaults):

        * ``"api"`` — credential has ``api_key`` →
          ``ANTHROPIC_API_KEY`` is set to that key.
        * ``"bedrock"`` — credential ``keys`` contain ``aws_access_key_id``
          → ``CLAUDE_CODE_USE_BEDROCK=1`` plus AWS env vars.
        * ``"vertex"`` — credential ``keys`` contain ``vertex_project`` →
          ``CLAUDE_CODE_USE_VERTEX=1`` plus GCP env vars.
        * ``"max"`` / ``None`` (default) — subscription auth →
          ``ANTHROPIC_API_KEY=""`` (Max plan).
        """
        from core.paths import PROJECT_DIR

        env: dict[str, str] = {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PATH": _build_sdk_path_env(self._anima_dir, PROJECT_DIR),
            "CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT": "true",
            "ENABLE_TOOL_SEARCH": "false",
            "CLAUDECODE": "",
        }

        # Claude Code on Windows requires Git Bash; auto-detect if not in
        # the default location so the CLI subprocess doesn't crash on init.
        if sys.platform == "win32" and not os.environ.get("CLAUDE_CODE_GIT_BASH_PATH"):
            from core.platform.claude_code import _find_git_bash

            git_bash = _find_git_bash()
            if git_bash:
                env["CLAUDE_CODE_GIT_BASH_PATH"] = git_bash
                logger.info("Auto-detected Git Bash: %s", git_bash)

        auth = self._model_config.mode_s_auth
        extra = self._model_config.extra_keys
        api_key = self._resolve_api_key()

        if auth == "api":
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
                logger.info("Mode S auth: API direct (mode_s_auth=api)")
            else:
                logger.warning("Mode S auth: mode_s_auth=api but no api_key found; falling back to Max plan")
                env["ANTHROPIC_API_KEY"] = ""
        elif auth == "bedrock":
            env["ANTHROPIC_API_KEY"] = ""
            env["CLAUDE_CODE_USE_BEDROCK"] = "1"
            for env_key, extra_key in (
                ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
                ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
                ("AWS_SESSION_TOKEN", "aws_session_token"),
                ("AWS_REGION", "aws_region_name"),
                ("AWS_PROFILE", "aws_profile"),
            ):
                val = extra.get(extra_key) or os.environ.get(env_key)
                if val:
                    env[env_key] = val
            logger.info("Mode S auth: Bedrock (mode_s_auth=bedrock)")
        elif auth == "vertex":
            env["ANTHROPIC_API_KEY"] = ""
            env["CLAUDE_CODE_USE_VERTEX"] = "1"
            for env_key, extra_key in (
                ("CLOUD_ML_PROJECT_ID", "vertex_project"),
                ("CLOUD_ML_REGION", "vertex_location"),
                ("GOOGLE_APPLICATION_CREDENTIALS", "vertex_credentials"),
            ):
                val = extra.get(extra_key) or os.environ.get(env_key)
                if val:
                    env[env_key] = val
            logger.info("Mode S auth: Vertex AI (mode_s_auth=vertex)")
        else:
            env["ANTHROPIC_API_KEY"] = ""
            logger.info("Mode S auth: Max plan (mode_s_auth=%s)", auth)

        if self._model_config.api_base_url:
            env["ANTHROPIC_BASE_URL"] = self._model_config.api_base_url
        return env

    def _build_mcp_env(self) -> dict[str, str]:
        """Build env dict for the MCP server subprocess.

        The MCP server needs ANIMAWORKS_ANIMA_DIR and ANIMAWORKS_PROJECT_DIR
        to initialize ToolHandler, plus PYTHONPATH so it can import core modules.
        """
        from core.paths import PROJECT_DIR

        return {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PYTHONPATH": str(PROJECT_DIR),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

    def _make_pending_executor_wake_callback(self) -> Callable[[], None] | None:
        """Create a callback that writes a .wake file for PendingTaskExecutor.

        The wake file signals the pending executor (running in the runner
        subprocess) to check for new tasks immediately rather than waiting
        for the next poll interval.
        """
        wake_path = self._anima_dir / "state" / "pending" / ".wake"

        def _wake() -> None:
            try:
                wake_path.parent.mkdir(parents=True, exist_ok=True)
                wake_path.write_text("1", encoding="utf-8")
            except Exception:
                pass

        return _wake

    def _build_mcp_servers(self) -> dict[str, Any]:
        """Build MCP server configuration dict.

        Returns the raw dict mapping server names to their config.
        """
        return {
            "aw": {
                "command": sys.executable,
                "args": ["-m", "core.mcp.server"],
                "env": self._build_mcp_env(),
            },
            **self._extra_mcp_servers,
        }

    def _build_mcp_servers_option(self) -> tuple[dict[str, Any] | str, Path | None]:
        """Return MCP servers value for ClaudeAgentOptions and optional temp file.

        On Windows the ``--mcp-config`` CLI argument is processed through
        three escaping layers — Python ``list2cmdline``, ``cmd.exe``
        (because ``claude.cmd`` is a batch wrapper), and Node.js arg
        parsing.  JSON with embedded backslash-escaped Windows paths and
        quotes is routinely corrupted by this chain, causing the CLI to
        receive a malformed MCP config.  Tool listing may still work but
        ``tools/call`` hangs because the server env is wrong.

        The fix: on Windows, write the MCP config JSON to a temp file and
        pass the **file path** to ``--mcp-config`` instead of inline JSON.
        This sidesteps all command-line escaping issues.
        """
        servers = self._build_mcp_servers()

        if sys.platform != "win32":
            return servers, None

        # Write MCP config to a temp file on Windows
        mcp_config = {"mcpServers": servers}
        fd, mcp_tmp_path = tempfile.mkstemp(suffix=".json", prefix="aw-mcpcfg-")
        try:
            os.write(fd, json.dumps(mcp_config, ensure_ascii=False).encode("utf-8"))
        finally:
            os.close(fd)
        logger.info("MCP config written to temp file: %s", mcp_tmp_path)
        return mcp_tmp_path, Path(mcp_tmp_path)

    def _build_sdk_options(
        self,
        system_prompt: str,
        max_turns: int,
        context_window: int,
        session_stats: dict[str, Any],
        *,
        resume: str | None = None,
        include_partial_messages: bool = False,
    ) -> tuple[ClaudeAgentOptions, list[Path]]:
        """Construct ``ClaudeAgentOptions`` for the Agent SDK client.

        Shared by both ``execute()`` and ``execute_streaming()`` (initial
        and retry attempts).  All SDK-specific lazy imports live here so
        callers need not repeat them.

        Returns:
            A tuple of ``(options, temp_files)``.  *temp_files* contains
            ``Path`` objects for any temp files created (system-prompt
            file, MCP config file) that the **caller must delete** after
            the SDK client has been closed.
        """
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        _cw = context_window
        temp_files: list[Path] = []

        extra_args: dict[str, str | None] = {}
        # On Windows, enable CLI debug output to stderr for diagnostics.
        # The CLI normally writes nothing to stderr; this flag makes it
        # dump its internal debug log so we can see what happens during
        # freezes (e.g., ToolSearch hang).
        if sys.platform == "win32":
            extra_args["debug-to-stderr"] = None  # boolean flag, no value
        if len(system_prompt.encode("utf-8")) > _PROMPT_FILE_THRESHOLD:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".txt",
                prefix="aw-sysprompt-",
            )
            try:
                os.write(fd, system_prompt.encode("utf-8"))
            finally:
                os.close(fd)
            temp_files.append(Path(tmp_path))
            prompt_kwarg: str | None = None
            extra_args["system-prompt-file"] = tmp_path
            logger.info(
                "System prompt too large for CLI arg (%d bytes > %d); using --system-prompt-file %s",
                len(system_prompt.encode("utf-8")),
                _PROMPT_FILE_THRESHOLD,
                tmp_path,
            )
        else:
            prompt_kwarg = system_prompt

        from core.config.models import resolve_max_tokens

        _effective_max_tokens = resolve_max_tokens(
            self._model_config.model,
            self._model_config.max_tokens,
            self._model_config.thinking,
        )

        _has_subs = self._has_subordinates()

        # Prefer a verified CLI path over SDK bundled discovery
        verified_cli = _resolve_sdk_cli_path()

        # Build MCP servers config (file-based on Windows to avoid
        # command-line escaping corruption)
        mcp_servers_val, mcp_tmp = self._build_mcp_servers_option()
        if mcp_tmp:
            temp_files.append(mcp_tmp)

        # Capture CLI stderr on Windows for diagnostics.  The CLI writes
        # important error/warning messages to stderr that are invisible
        # when not piped.  On Windows (where sessions frequently freeze),
        # having stderr output is critical for post-mortem analysis.
        _stderr_log: Path | None = None
        if sys.platform == "win32":
            _stderr_log = self._anima_dir / "state" / "sdk_stderr.log"
            _stderr_log.parent.mkdir(parents=True, exist_ok=True)
            # Truncate at session start so the log only contains the current session
            try:
                _stderr_log.write_text(
                    f"--- session start (resume={resume}) ---\n",
                    encoding="utf-8",
                )
            except Exception:
                pass

            def _stderr_handler(line: str, _path: Path = _stderr_log) -> None:
                try:
                    with open(_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass

        # In stream-json mode the CLI routes tool permission decisions
        # through the control protocol.  Even with bypassPermissions the
        # CLI requires --permission-prompt-tool stdio to know that
        # permissions flow over stdin/stdout.  Without it the CLI silently
        # stalls before executing ANY tool (built-in or MCP) — which is
        # the root cause of the "tool executing / frozen" issue on Windows.
        #
        # We provide a trivial can_use_tool callback that always approves
        # (matching bypassPermissions semantics).  The SDK automatically
        # sets --permission-prompt-tool stdio when can_use_tool is present.
        async def _auto_approve_tool(
            tool_name: str,
            tool_input: dict[str, Any],
            context: Any,
        ) -> Any:
            from claude_agent_sdk import PermissionResultAllow

            return PermissionResultAllow()

        kwargs: dict[str, Any] = dict(
            system_prompt=prompt_kwarg,
            permission_mode="bypassPermissions",
            can_use_tool=_auto_approve_tool,
            cwd=str(self._task_cwd or self._anima_dir),
            max_turns=max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            max_buffer_size=_SDK_MAX_BUFFER_SIZE,
            resume=resume,
            setting_sources=[],
            extra_args=extra_args,
            **({"cli_path": verified_cli} if verified_cli else {}),
            mcp_servers=mcp_servers_val,
            **({"stderr": _stderr_handler} if _stderr_log else {}),
            # NOTE: Hooks disabled on Windows due to Agent SDK stream-json
            # control protocol instability — the CLI's hook_callback requests
            # fail with "Stream closed" (stdin pipe breaks), crashing the
            # entire session.  The hooks provide tool logging, trust tracking,
            # and context budget observation — all non-essential for core chat.
            # TODO: Re-enable once the SDK ships a fix for Windows pipe handling.
            **(
                {
                    "hooks": {
                        "PreToolUse": [
                            HookMatcher(
                                matcher=".*",
                                hooks=[
                                    _build_pre_tool_hook(
                                        self._anima_dir,
                                        max_tokens=_effective_max_tokens,
                                        context_window=_cw,
                                        session_stats=session_stats,
                                        superuser=_is_debug_superuser(self._anima_dir),
                                        on_task_intercepted=self._make_pending_executor_wake_callback(),
                                        has_subordinates=_has_subs,
                                    )
                                ],
                            )
                        ],
                        "PreCompact": [
                            HookMatcher(
                                matcher="auto",
                                hooks=[
                                    _build_pre_compact_hook(
                                        self._anima_dir,
                                        session_stats=session_stats,
                                        context_window=_cw,
                                    )
                                ],
                            )
                        ],
                        "PostToolUse": [
                            HookMatcher(
                                matcher="Write|Edit",
                                hooks=[_build_post_tool_hook(self._anima_dir)],
                            )
                        ],
                        "Stop": [
                            HookMatcher(
                                hooks=[
                                    _build_stop_hook(
                                        self._anima_dir,
                                        session_stats=session_stats,
                                    )
                                ],
                            )
                        ],
                    },
                }
                if sys.platform != "win32"
                else {}
            ),
        )
        if self._model_config.thinking:
            from core.execution.base import is_adaptive_model, resolve_thinking_effort

            if is_adaptive_model(self._model_config.model):
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["effort"] = resolve_thinking_effort(
                    self._model_config.model,
                    self._model_config.thinking_effort,
                )

        if include_partial_messages:
            kwargs["include_partial_messages"] = True
        return ClaudeAgentOptions(**kwargs), temp_files
