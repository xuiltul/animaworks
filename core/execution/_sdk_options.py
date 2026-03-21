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
)
from core.execution._sdk_session import (
    _PROMPT_FILE_THRESHOLD,
    _SDK_MAX_BUFFER_SIZE,
    _is_debug_superuser,
)

logger = logging.getLogger("animaworks.execution.agent_sdk")


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
            "PATH": f"{self._anima_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT": "true",
            "CLAUDECODE": "",
        }

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

    def _build_sdk_options(
        self,
        system_prompt: str,
        max_turns: int,
        context_window: int,
        session_stats: dict[str, Any],
        *,
        resume: str | None = None,
        include_partial_messages: bool = False,
    ) -> tuple[ClaudeAgentOptions, Path | None]:
        """Construct ``ClaudeAgentOptions`` for the Agent SDK client.

        Shared by both ``execute()`` and ``execute_streaming()`` (initial
        and retry attempts).  All SDK-specific lazy imports live here so
        callers need not repeat them.

        Returns:
            A tuple of (options, prompt_file).  *prompt_file* is ``None``
            when the prompt fits in a CLI argument; otherwise it is a
            ``Path`` to a temp file that the **caller must delete** after
            the SDK client has been closed.
        """
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        _cw = context_window

        prompt_file: Path | None = None
        extra_args: dict[str, str | None] = {}
        if len(system_prompt.encode("utf-8")) > _PROMPT_FILE_THRESHOLD:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".txt",
                prefix="aw-sysprompt-",
            )
            try:
                os.write(fd, system_prompt.encode("utf-8"))
            finally:
                os.close(fd)
            prompt_file = Path(tmp_path)
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

        kwargs: dict[str, Any] = dict(
            system_prompt=prompt_kwarg,
            permission_mode="bypassPermissions",
            cwd=str(self._task_cwd or self._anima_dir),
            max_turns=max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            max_buffer_size=_SDK_MAX_BUFFER_SIZE,
            resume=resume,
            setting_sources=[],
            extra_args=extra_args,
            mcp_servers={
                "aw": {
                    "command": sys.executable,
                    "args": ["-m", "core.mcp.server"],
                    "env": self._build_mcp_env(),
                },
                **self._extra_mcp_servers,
            },
            hooks={
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
                        matcher=".*",
                        hooks=[_build_pre_compact_hook(self._anima_dir)],
                    )
                ],
                "PostToolUse": [
                    HookMatcher(
                        matcher="Write|Edit",
                        hooks=[_build_post_tool_hook(self._anima_dir)],
                    )
                ],
            },
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
        return ClaudeAgentOptions(**kwargs), prompt_file
