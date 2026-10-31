from __future__ import annotations

from core.platform.env import get_env

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Codex CLI/SDK startup, environment, and configuration helpers."""

import asyncio
import inspect
import logging
import os
import shutil
import stat
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.execution.process_runner import ProcessRunner
from core.platform.atomic_io import atomic_write_json
from core.platform.codex import default_home_dir, get_codex_executable
from core.platform.codex import patch_reasoning_effort_enum as _patch_reasoning_effort_enum
from core.platform.subprocess_entries import SubprocessEntry, module_args
from core.schemas import ModelConfig

logger = logging.getLogger("animaworks.execution.codex_sdk")

_CODEX_REASONING_SUMMARY_DEFAULT = "concise"
_CODEX_REASONING_SUMMARY_VALUES = {"auto", "concise", "detailed", "none"}
_CODEX_CLIENT_PROCESS_WAIT_TIMEOUT_SEC = 2.0
_CODEX_CLIENT_READER_JOIN_TIMEOUT_SEC = 2.0
_SUBPROCESS_STREAM_LIMIT = 16 * 1024 * 1024  # 16 MB


def resolve_codex_model(model: str) -> str:
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
    model = extra.get("codex_model") or resolve_codex_model(model_config.model)

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
    escapes = {
        "\\": "\\\\",
        '"': '\\"',
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
    }
    result: list[str] = []
    for char in value:
        escaped = escapes.get(char)
        if escaped is not None:
            result.append(escaped)
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            result.append(f"\\u{ord(char):04X}")
        else:
            result.append(char)
    return "".join(result)


def _git_metadata_write_paths(root: Path, *, forbidden_ancestors: list[Path] | None = None) -> list[Path]:
    """Return git metadata directories that must be writable for *root*.

    Codex remounts a writable root's ``.git`` read-only as a built-in
    safeguard, which breaks ``git worktree``/``commit`` operations for
    repository workspaces.  Explicit more-specific rules (or standalone
    writable roots) override that protection.

    Handles both layouts:
      - ``root/.git`` is a directory → the repository's own metadata.
      - ``root/.git`` is a file (linked worktree) → resolve the ``gitdir:``
        pointer and its ``commondir`` so the primary repository's metadata
        is writable too.

    ``forbidden_ancestors`` guards the resolved-pointer path: ``root/.git``
    file contents are writable by the sandboxed model itself, so a
    hand-crafted ``gitdir:`` must never widen access into runtime or
    explicitly denied trees.
    """
    git_path = root / ".git"
    forbidden = [p.resolve() for p in (forbidden_ancestors or [])]

    def _allowed(path: Path) -> bool:
        return not any(path == anc or path.is_relative_to(anc) for anc in forbidden)

    results: list[Path] = []
    try:
        if git_path.is_dir():
            results.append(git_path.resolve())
        elif git_path.is_file():
            content = git_path.read_text(encoding="utf-8", errors="replace").strip()
            if content.startswith("gitdir:"):
                gitdir = Path(content[len("gitdir:") :].strip())
                if not gitdir.is_absolute():
                    gitdir = root / gitdir
                gitdir = gitdir.resolve()
                if gitdir.is_dir() and _allowed(gitdir):
                    results.append(gitdir)
                    commondir_file = gitdir / "commondir"
                    if commondir_file.is_file():
                        common = Path(commondir_file.read_text(encoding="utf-8").strip())
                        if not common.is_absolute():
                            common = gitdir / common
                        common = common.resolve()
                        if common.is_dir() and _allowed(common):
                            results.append(common)
    except OSError:
        return []
    # Drop paths already inside a returned ancestor to keep rules minimal.
    deduped: list[Path] = []
    for path in results:
        if not any(path == kept or path.is_relative_to(kept) for kept in deduped):
            deduped.append(path)
    return deduped


def _default_home_dir() -> str:
    """Return a stable HOME value for Codex child processes across platforms."""
    return default_home_dir()


def default_path_env() -> str:
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


def _is_desktop_extension_codex(executable: str | None) -> bool:
    """Return True when the Codex binary comes from a desktop-app extension bundle."""
    if not executable:
        return False
    norm = executable.replace("/", "\\").lower()
    return "\\.antigravity\\extensions\\openai.chatgpt-" in norm or "\\windowsapps\\openai.codex_" in norm


def _should_prefer_cli_exec(trigger: str) -> bool:
    """Prefer direct ``codex exec`` for unstable desktop-bundled background sessions."""
    forced = get_env("ANIMAWORKS_CODEX_FORCE_CLI_EXEC", "").strip().lower()
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


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _declared_codex_private_attr(owner: Any, attr_name: str) -> Any:
    """Read a declared SDK private attribute without triggering dynamic mocks."""
    try:
        attributes = vars(owner)
        if attr_name not in attributes:
            return None
        return getattr(owner, attr_name, None)
    except Exception:
        logger.warning(
            "Failed to inspect Codex SDK cleanup resource %s",
            attr_name,
            exc_info=True,
        )
        return None


def _codex_client_transport_resources(client: Any) -> tuple[Any, threading.Thread | None, threading.Thread | None]:
    """Snapshot SDK transport resources before ``close()`` clears them.

    Current ``AsyncCodex`` nests the transport at ``_client._sync``;
    older/test clients may expose it at ``_sync`` or directly.  These are
    private SDK details, so every ``getattr`` is deliberately best-effort.
    """
    owners = [client]
    for link_name in ("_client", "_sync"):
        for owner in tuple(owners):
            nested = _declared_codex_private_attr(owner, link_name)
            if nested is not None and all(nested is not existing for existing in owners):
                owners.append(nested)

    resources: list[Any] = [None, None, None]
    for attr_name in ("_proc", "_reader_thread", "_stderr_thread"):
        resource_index = ("_proc", "_reader_thread", "_stderr_thread").index(attr_name)
        for owner in owners:
            value = _declared_codex_private_attr(owner, attr_name)
            if value is not None:
                resources[resource_index] = value
                break

    proc, reader_thread, stderr_thread = resources
    return (
        proc,
        reader_thread if isinstance(reader_thread, threading.Thread) else None,
        stderr_thread if isinstance(stderr_thread, threading.Thread) else None,
    )


def _finish_codex_client_transport(
    proc: Any,
    reader_thread: threading.Thread | None,
    stderr_thread: threading.Thread | None,
) -> None:
    """Force-close a Codex SDK transport after the SDK's own cleanup."""
    if proc is not None:
        ProcessRunner.terminate_popen_sync(proc, timeout=_CODEX_CLIENT_PROCESS_WAIT_TIMEOUT_SEC)

        # Close pipes before joining readers so a blocked readline receives EOF.
        try:
            _close_subprocess_stdio(proc)
        except Exception:
            logger.warning("Failed to close Codex SDK subprocess pipes", exc_info=True)

    for thread_name, thread in (
        ("reader", reader_thread),
        ("stderr", stderr_thread),
    ):
        if thread is None:
            continue
        try:
            if thread.is_alive():
                thread.join(timeout=_CODEX_CLIENT_READER_JOIN_TIMEOUT_SEC)
            if thread.is_alive():
                logger.warning("Codex SDK %s thread is still alive after cleanup", thread_name)
        except Exception:
            logger.warning("Failed to join Codex SDK %s thread", thread_name, exc_info=True)


async def _close_codex_client(client: Any) -> None:
    proc, reader_thread, stderr_thread = _codex_client_transport_resources(client)

    try:
        close = getattr(client, "close", None)
    except Exception:
        logger.warning("Failed to inspect Codex SDK close method", exc_info=True)
        close = None
    if callable(close):
        try:
            await _maybe_await(close())
        except Exception:
            logger.warning("Failed to close Codex SDK client", exc_info=True)

    try:
        await asyncio.to_thread(
            _finish_codex_client_transport,
            proc,
            reader_thread,
            stderr_thread,
        )
    except Exception:
        logger.warning("Failed to finish Codex SDK transport cleanup", exc_info=True)


def _repair_codex_daemon_dir() -> None:
    """Restore 0700 on Codex's per-user app-server socket directory.

    Codex refuses to start a thread ("app-server socket directory must be a
    user-owned directory with mode 0700") when ``/tmp/codex-daemon-<uid>`` was
    created with a looser mode by another Codex process on the host; every Codex
    anima then fails until it is fixed (2026-10-01, sora/sumire/mio).
    """
    if os.name != "posix":
        return
    path = Path(tempfile.gettempdir()) / f"codex-daemon-{os.getuid()}"
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        logger.debug("Could not stat %s", path, exc_info=True)
        return
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        return
    if stat.S_IMODE(info.st_mode) != 0o700:
        try:
            path.chmod(0o700)
            logger.warning("Reset %s to mode 0700 (was %o)", path, stat.S_IMODE(info.st_mode))
        except OSError:
            logger.warning("Could not reset %s to mode 0700", path, exc_info=True)


class CodexSetupMixin:
    # ── Environment / config helpers ─────────────────────────

    def _build_env(self) -> dict[str, str]:
        """Build env dict for the Codex CLI child process."""
        from core.execution.session.session_context import current_runtime_session

        _repair_codex_daemon_dir()
        from core.paths import PROJECT_DIR

        env: dict[str, str] = {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PATH": default_path_env(),
            "CODEX_HOME": str(self._codex_home),
            "HOME": _default_home_dir(),
        }
        from core.execution.workspace_env import update_workspace_env

        update_workspace_env(env, self._anima_dir)  # Includes ANIMAWORKS_SCRATCH_DIR and shared tool caches.
        ctx = current_runtime_session()
        if ctx is not None:
            env.update(ctx.to_env())
        if internal_auth := get_env("ANIMAWORKS_INTERNAL_AUTH"):
            env["ANIMAWORKS_INTERNAL_AUTH"] = internal_auth
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
            from core.execution.github_identity import resolve_github_token_env

            env.update(resolve_github_token_env(self._anima_dir))
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
        from core.execution.github_identity import resolve_github_token_env

        env.update(resolve_github_token_env(self._anima_dir))
        return env

    def _build_mcp_env(self) -> dict[str, str]:
        """Build env dict for the MCP server subprocess."""
        from core.execution.mcp_env import build_mcp_env

        env = build_mcp_env(self._anima_dir)
        # Codex's MCP server needs the embedded CLI and launcher Python on PATH.
        env["PATH"] = default_path_env()
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

        from core.config.file_access_policy import (
            effective_write_roots,
            resolve_effective_denied_roots,
            shared_tool_cache_write_root,
        )
        from core.config.models import load_permissions

        permissions_config = load_permissions(self._anima_dir)
        write_roots = effective_write_roots(
            self._anima_dir,
            permissions_config.file_roots,
            self._task_cwd,
        )
        tool_cache_root = shared_tool_cache_write_root(self._anima_dir)

        denied_roots = list(
            resolve_effective_denied_roots(
                self._anima_dir,
                getattr(permissions_config, "file_roots_denied", []),
            )
        )
        if denied_roots:
            from core.config.file_access_policy import foreign_owned_ssh_config_dirs, shell_internal_deny_paths

            # Permission profiles and the legacy sandbox settings are mutually
            # exclusive.  Start with broad read access, retain the charter
            # writable roots (including temp for workspace-write parity), and
            # carve denied subtrees out with more-specific ``deny`` rules.
            root_is_writable = "/" in permissions_config.file_roots
            data_dir = self._anima_dir.resolve().parent.parent

            # The model-facing shell must not be able to replace trusted
            # runtime inputs with symlinks that a later host-side prompt
            # assembly would follow.  Runtime-data writes go through the
            # constrained MCP APIs instead.
            shell_filesystem_rules: dict[str, str] = {
                ":root": "read",
                ":tmpdir": "write",
                ":slash_tmp": "write",
                str(self._anima_dir.resolve()): "write",
            }
            git_forbidden = [data_dir, *(Path(r) for r in denied_roots)]

            # The MCP server needs the same writable roots for constrained
            # memory and messaging tools.  It uses a separate profile and
            # does not expose arbitrary machine execution while deny is on.
            mcp_filesystem_rules: dict[str, str] = {
                ":root": "write" if root_is_writable else "read",
            }
            if not root_is_writable:
                mcp_filesystem_rules[":tmpdir"] = "write"
                mcp_filesystem_rules[":slash_tmp"] = "write"
                mcp_filesystem_rules[str(self._anima_dir.resolve())] = "write"
            for root in write_roots:
                root_str = str(root)
                # Charter: only companies/<own>/shared is writable under data_dir.
                # Pin the company root itself as read so siblings (knowledge/,
                # skills/, …) cannot inherit write from a looser parent rule.
                if root.parent.name and root.name == "shared":
                    company_root_str = str(root.parent)
                    shell_filesystem_rules.setdefault(company_root_str, "read")
                    mcp_filesystem_rules.setdefault(company_root_str, "read")
                shell_filesystem_rules[root_str] = "write"
                mcp_filesystem_rules[root_str] = "write"
                for git_path in _git_metadata_write_paths(root, forbidden_ancestors=git_forbidden):
                    git_path_str = str(git_path)
                    shell_filesystem_rules[git_path_str] = "write"
                    mcp_filesystem_rules[git_path_str] = "write"

            for root in denied_roots:
                resolved_root = str(Path(root).resolve())
                shell_filesystem_rules[resolved_root] = "deny"
                mcp_filesystem_rules[resolved_root] = "deny"

            # External-tool caches (Chatwork/Slack message DBs and the
            # identity map) live outside the Anima directory.  Without write
            # access even a plain inbox read fails with EROFS.
            if tool_cache_root is not None:
                cache_root_str = str(tool_cache_root)
                shell_filesystem_rules[cache_root_str] = "write"
                mcp_filesystem_rules[cache_root_str] = "write"

            # Authentication and all runtime state/cache copies must never be
            # directly readable from the model shell.  The MCP profile keeps
            # cache access for trusted, source-filtered search services.
            for internal_path in shell_internal_deny_paths(self._anima_dir):
                shell_filesystem_rules[str(internal_path)] = "deny"

            # bwrap's user namespace maps root to nobody, so ssh rejects every
            # root-owned drop-in that /etc/ssh/ssh_config includes ("Bad owner
            # or permissions on /etc/ssh/ssh_config.d/…", exit 255).  Hiding
            # the directory makes the Include glob match nothing.
            for ssh_dropin_dir in foreign_owned_ssh_config_dirs():
                shell_filesystem_rules[ssh_dropin_dir] = "deny"

            # The sandboxed Anima must not be able to remove or weaken the
            # policy that will be used to build its next session's profile.
            # A file-specific read rule is more specific than the writable
            # Anima root (and remains read-only even when ``:root`` is write).
            permissions_path = str((self._anima_dir / "permissions.json").resolve())
            shell_filesystem_rules[permissions_path] = "read"
            mcp_filesystem_rules[permissions_path] = "read"

            shell_filesystem_lines = "\n".join(
                f'"{esc(path)}" = "{access}"' for path, access in shell_filesystem_rules.items()
            )
            mcp_filesystem_lines = "\n".join(
                f'"{esc(path)}" = "{access}"' for path, access in mcp_filesystem_rules.items()
            )
            sandbox_lines = (
                'default_permissions = "animaworks"\n'
                'approval_policy = "never"\n'
                "\n"
                "[permissions.animaworks.filesystem]\n"
                f"{shell_filesystem_lines}\n"
                "\n"
                "[permissions.animaworks.network]\n"
                "enabled = true\n"
                "\n"
                "[permissions.animaworks_mcp.filesystem]\n"
                f"{mcp_filesystem_lines}\n"
                "\n"
                "[permissions.animaworks_mcp.network]\n"
                "enabled = true\n"
            )
        elif "/" in permissions_config.file_roots:
            sandbox_lines = 'sandbox_mode = "danger-full-access"\napproval_policy = "never"\n'
        else:
            writable_roots = [str(self._anima_dir), *(str(root) for root in write_roots)]
            if tool_cache_root is not None:
                writable_roots.append(str(tool_cache_root))
            # Standalone entries for git metadata escape Codex's built-in
            # read-only remount of each writable root's ``.git``.
            data_dir = self._anima_dir.resolve().parent.parent
            for root_str in list(writable_roots):
                for git_path in _git_metadata_write_paths(Path(root_str), forbidden_ancestors=[data_dir]):
                    if str(git_path) not in writable_roots:
                        writable_roots.append(str(git_path))
            roots_list = ", ".join(f'"{esc(r)}"' for r in writable_roots)
            sandbox_lines = (
                'sandbox_mode = "workspace-write"\n'
                'approval_policy = "never"\n'
                "\n"
                "[sandbox_workspace_write]\n"
                f"writable_roots = [{roots_list}]\n"
                "network_access = true\n"
            )

        mcp_env = self._build_mcp_env()
        if denied_roots:
            # The nested ``codex sandbox`` resolves the named profile from
            # this per-Anima CODEX_HOME.  Set it explicitly rather than
            # relying on the MCP launcher inheriting the parent environment.
            mcp_env["CODEX_HOME"] = str(self._codex_home)
            mcp_env["ANIMAWORKS_FILE_DENY_ACTIVE"] = "1"
        mcp_env_lines = "\n".join(f'{k} = "{esc(v)}"' for k, v in mcp_env.items())
        if denied_roots:
            mcp_command = get_codex_executable()
            if not mcp_command:
                raise RuntimeError(
                    "Codex CLI executable is required to sandbox the MCP server when file_roots_denied is configured"
                )
            mcp_args = [
                "sandbox",
                "-P",
                "animaworks_mcp",
                "--",
                sys.executable,
                *module_args(SubprocessEntry.MCP_SERVER),
            ]
        else:
            # Preserve the pre-profile MCP command exactly for Animas that do
            # not opt in to read-deny enforcement.
            mcp_command = sys.executable
            mcp_args = list(module_args(SubprocessEntry.MCP_SERVER))
        mcp_args_toml = ", ".join(f'"{esc(arg)}"' for arg in mcp_args)
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

        # Codex CLI側のeffort語彙（gpt-5.6系: low〜ultra）をそのまま渡す。
        # Claude系のresolve_thinking_effort（maxクランプ）は適用しない。
        # models.json の ``reasoning_effort`` はモデル単位の固定値で、anima の
        # thinking_effort より優先する（フォールバック先は元 anima の effort を
        # 引き継ぐため、モデル側で揃えたい場合はここで指定する）。
        from core.config.model_mode import _match_models_json

        model_entry = _match_models_json(self._model_config.model) or {}
        reasoning_effort = (
            (self._model_config.extra_keys or {}).get("codex_reasoning_effort")
            or model_entry.get("reasoning_effort")
            or self._model_config.thinking_effort
        )
        effort_line = f'model_reasoning_effort = "{esc(reasoning_effort)}"\n' if reasoning_effort else ""
        task_compaction_tokens = self._model_config.task_compaction_tokens
        task_compaction_line = (
            f"model_auto_compact_token_limit = {task_compaction_tokens}\n" if task_compaction_tokens > 0 else ""
        )
        # Codex auto-loads AGENTS.md from its cwd as a project document.  When
        # no task cwd is configured, cwd is the anima directory and AGENTS.md
        # is already injected by AnimaWorks; let the explicit prompt win.
        project_doc_line = "project_doc_max_bytes = 0\n" if self._task_cwd is None else ""

        config_toml = (
            f'model = "{esc(provider_config.model)}"\n'
            f"{effort_line}"
            f"{task_compaction_line}"
            f"{project_doc_line}"
            f'model_provider = "{esc(provider_config.provider)}"\n'
            f'model_instructions_file = "{esc(str(instructions_file))}"\n'
            f'developer_instructions = "{esc(self._CODEX_DEVELOPER_INSTRUCTIONS)}"\n'
            f'personality = "friendly"\n'
            f'model_verbosity = "high"\n'
            f"{sandbox_lines}"
            f"{provider_section}"
            f"\n"
            f"[mcp_servers.aw]\n"
            f'command = "{esc(mcp_command)}"\n'
            f"args = [{mcp_args_toml}]\n"
            f'default_tools_approval_mode = "approve"\n'
            f"\n"
            f"[mcp_servers.aw.env]\n"
            f"{mcp_env_lines}\n"
        )
        (self._codex_home / "config.toml").write_text(config_toml, encoding="utf-8")
        self._write_hooks()

    def _write_hooks(self) -> None:
        """Point Codex's PreToolUse hook at ``cli.codex_command_hook``.

        The hook runs on the host (outside the sandbox) and denies commands by the
        global/per-anima deny lists plus the recursive-search guard.  ``-m`` works
        from any cwd because the venv has an editable install of this repo.
        """
        import shlex

        from core.paths import get_global_permissions_path

        hook_cmd = " ".join(
            shlex.quote(part)
            for part in (
                sys.executable,
                *module_args(SubprocessEntry.CODEX_COMMAND_HOOK),
                "--anima-dir",
                str(self._anima_dir.resolve()),
                "--global-permissions",
                str(get_global_permissions_path()),
            )
        )
        hooks = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [{"type": "command", "command": hook_cmd, "timeout": 30}],
                    }
                ]
            }
        }
        atomic_write_json(self._codex_home / "hooks.json", hooks, indent=2, ensure_ascii=True)

    def _create_codex_client(self) -> Any:
        """Create an ``AsyncCodex`` SDK client instance."""
        try:
            from openai_codex import AsyncCodex, CodexConfig
        except ModuleNotFoundError as e:
            raise ImportError("openai_codex is required for Mode C (install openai-codex).") from e

        _patch_reasoning_effort_enum()

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

    def _sdk_reasoning_summary(self) -> Any | None:
        raw_value = (self._model_config.extra_keys or {}).get(
            "codex_reasoning_summary",
            _CODEX_REASONING_SUMMARY_DEFAULT,
        )
        value = str(raw_value or _CODEX_REASONING_SUMMARY_DEFAULT).strip().lower()
        if value in {"default", "true", "yes", "on"}:
            value = _CODEX_REASONING_SUMMARY_DEFAULT
        if value == "none":
            return None
        if value not in _CODEX_REASONING_SUMMARY_VALUES:
            logger.warning(
                "Invalid codex_reasoning_summary=%r; using %s",
                raw_value,
                _CODEX_REASONING_SUMMARY_DEFAULT,
            )
            value = _CODEX_REASONING_SUMMARY_DEFAULT

        from openai_codex.generated.v2_all import ReasoningSummary, ReasoningSummaryValue

        return ReasoningSummary(root=getattr(ReasoningSummaryValue, value))

    def _codex_thread_kwargs(self, system_prompt: str) -> dict[str, Any]:
        provider_config = _resolve_codex_provider_config(self._model_config)
        codex_config: dict[str, Any] = {"bypass_hook_trust": True}
        if self._task_cwd is None:
            # The anima directory's AGENTS.md is already part of the explicit
            # system prompt; disable Codex's automatic project-doc loading.
            codex_config["project_doc_max_bytes"] = 0
        kwargs: dict[str, Any] = {
            "approval_mode": self._sdk_approval_mode(),
            "base_instructions": system_prompt or None,
            "cwd": str(self._task_cwd or self._anima_dir),
            "developer_instructions": self._CODEX_DEVELOPER_INSTRUCTIONS,
            "model": provider_config.model,
            "model_provider": provider_config.provider,
            # hooks.json (written by _write_hooks) only runs with persisted hook
            # trust, which Codex grants via a TUI prompt we never see.  The hook
            # source is our own module, so bypass the trust gate.  Verified on
            # codex 0.151: config.toml keys / -c overrides do NOT enable it.
            "config": codex_config,
        }
        from core.config.file_access_policy import resolve_effective_denied_roots
        from core.config.models import load_permissions

        permissions_config = load_permissions(self._anima_dir)
        denied_roots = resolve_effective_denied_roots(
            self._anima_dir,
            getattr(permissions_config, "file_roots_denied", []),
        )
        if not denied_roots:
            kwargs["sandbox"] = self._sdk_sandbox()
        return kwargs

    def _codex_turn_kwargs(self) -> dict[str, Any]:
        provider_config = _resolve_codex_provider_config(self._model_config)
        kwargs: dict[str, Any] = {
            "approval_mode": self._sdk_approval_mode(),
            "cwd": str(self._task_cwd or self._anima_dir),
            "model": provider_config.model,
        }
        # Sandbox is set at thread/config level. Passing the SDK enum per turn
        # can drop config.toml details such as workspace network_access=true.
        summary = self._sdk_reasoning_summary()
        if summary is not None:
            kwargs["summary"] = summary
        return kwargs

    def _build_cli_exec_command(self) -> list[str]:
        """Build the `codex exec --json` command used as a runtime fallback."""
        executable = get_codex_executable()
        if not executable:
            raise RuntimeError("Codex CLI executable not available for exec fallback")
        command = [
            executable,
            "exec",
            "-C",
            str(self._task_cwd or self._anima_dir),
        ]
        if self._task_cwd is None:
            # See _write_codex_config: AGENTS.md is supplied explicitly.
            command.extend(["-c", "project_doc_max_bytes=0"])
        command.extend(
            [
                "--skip-git-repo-check",
                "--dangerously-bypass-hook-trust",  # see _codex_thread_kwargs
                "--json",
                "-",
            ]
        )
        return command
