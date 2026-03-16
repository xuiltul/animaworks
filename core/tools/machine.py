# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Machine tool — external agent CLI as stateless power tools for Animas.

Design philosophy: "Craftsperson and Machine Tools"
====================================================

An Anima is a craftsperson (棟梁). External agent CLIs invoked through this
module are machine tools — CNC routers, laser cutters, 3D printers.

A machine tool can cut wood with incredible precision, but it:
- Does NOT decide what to build (no autonomy)
- Does NOT remember yesterday's work (no memory)
- Does NOT talk to other craftspeople (no relationships)
- Does NOT know what it is (no identity)
- Does NOT show up tomorrow (no persistence)

**Tools extend capability. They don't replicate existence.**

The machine has NO access to AnimaWorks infrastructure:
- No memory (episodes/, knowledge/, procedures/)
- No messaging (send_message, post_channel)
- No org tree awareness
- No animaworks-tool CLI

Environment variables are sanitized to an allowlist; only API keys for
the selected engine are forwarded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import threading as _threading
import time
from pathlib import Path
from typing import Any

_machine_counter = 0
_machine_counter_lock = _threading.Lock()


def _next_machine_id() -> str:
    global _machine_counter
    with _machine_counter_lock:
        _machine_counter += 1
        return f"machine_{_machine_counter}"

from core.i18n import t
from core.tools._base import ToolResult

logger = logging.getLogger("animaworks.tools.machine")

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "run": {"expected_seconds": 600, "background_eligible": True},
}

# ── Constants ──────────────────────────────────────────────

_DEFAULT_TIMEOUT_SYNC = 600
_DEFAULT_TIMEOUT_ASYNC = 1800
_MAX_CALLS_PER_SESSION = 5
_MAX_CALLS_PER_HEARTBEAT = 2
_MAX_OUTPUT_CHARS = 50_000

_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TERM",
        "SHELL",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    }
)

_ANIMAWORKS_ENV_BLOCKLIST: frozenset[str] = frozenset(
    {
        "ANIMAWORKS_HOME",
        "ANIMAWORKS_ANIMA_DIR",
        "ANIMAWORKS_SOCKET",
        "ANIMAWORKS_DATA_DIR",
        "ANIMAWORKS_ANIMA_NAME",
    }
)

# ── Engine Command Templates ──────────────────────────────

_ENGINE_COMMANDS: dict[str, list[str]] = {
    "claude": [
        "claude",
        "-p",
        "--output-format",
        "text",
        "--no-session-persistence",
    ],
    "codex": [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "-s",
        "workspace-write",
        "--ephemeral",
    ],
    "gemini": [
        "gemini",
        "-o",
        "text",
        "--approval-mode",
        "yolo",
    ],
    "cursor-agent": [
        "cursor-agent",
        "-p",
        "--trust",
        "--force",
    ],
}

_ENGINE_MODEL_FLAGS: dict[str, str] = {
    "claude": "--model",
    "codex": "-m",
    "gemini": "-m",
    "cursor-agent": "--model",
}

_ENGINE_WORKDIR_FLAGS: dict[str, list[str]] = {
    "claude": [],
    "codex": ["-C"],
    "gemini": [],
    "cursor-agent": ["--workspace"],
}

_ENGINE_PERMISSION_FLAGS: dict[str, list[str]] = {
    "claude": ["--dangerously-skip-permissions"],
    "codex": [],
    "gemini": [],
    "cursor-agent": [],
}

_VALID_ENGINES = frozenset(_ENGINE_COMMANDS.keys())

_DEFAULT_ENGINE_PRIORITY: list[str] = [
    "cursor-agent",
    "claude",
    "codex",
    "gemini",
]

_LIST_SENTINEL = "__list__"

# ── Engine Availability ───────────────────────────────────


def _get_engine_priority() -> list[str]:
    """Return engine priority order from config, falling back to default."""
    try:
        from core.config.models import load_config

        config = load_config()
        if config.machine.engine_priority:
            return list(config.machine.engine_priority)
    except Exception as exc:
        logger.debug("Failed to load engine priority from config, using default: %s", exc)
    return list(_DEFAULT_ENGINE_PRIORITY)


def _get_available_engines() -> list[str]:
    """Return engines whose CLI binary is found in PATH, ordered by priority.

    Priority is determined by ``config.json`` ``machine.engine_priority``
    (if set) or :data:`_DEFAULT_ENGINE_PRIORITY`.  Engines not listed in
    the priority table are appended alphabetically.
    """
    priority = _get_engine_priority()
    available = {e for e in _VALID_ENGINES if shutil.which(_ENGINE_COMMANDS[e][0])}
    result = [e for e in priority if e in available]
    for e in sorted(available):
        if e not in result:
            result.append(e)
    return result


# ── Rate Limiting ──────────────────────────────────────────

_session_call_counts: dict[str, int] = {}


def _check_rate_limit(anima_dir: str, trigger: str) -> str | None:
    """Check per-session and per-heartbeat call limits."""
    key = anima_dir
    count = _session_call_counts.get(key, 0)

    if trigger.startswith("heartbeat"):
        if count >= _MAX_CALLS_PER_HEARTBEAT:
            return t("machine.rate_limit_exceeded", limit=_MAX_CALLS_PER_HEARTBEAT, period="heartbeat")
    else:
        if count >= _MAX_CALLS_PER_SESSION:
            return t("machine.rate_limit_exceeded", limit=_MAX_CALLS_PER_SESSION, period="session")
    return None


def _increment_call_count(anima_dir: str) -> None:
    _session_call_counts[anima_dir] = _session_call_counts.get(anima_dir, 0) + 1


def reset_call_counts(anima_dir: str | None = None) -> None:
    """Reset rate limit counters. Called at session boundaries."""
    if anima_dir:
        _session_call_counts.pop(anima_dir, None)
    else:
        _session_call_counts.clear()


# ── Credential Injection ──────────────────────────────────

_ENGINE_CREDENTIAL_MAP: dict[str, list[tuple[str, str, str]]] = {
    "claude": [("anthropic", "ANTHROPIC_API_KEY", "api_key")],
    "codex": [("openai", "OPENAI_API_KEY", "api_key")],
    "gemini": [
        ("google", "GEMINI_API_KEY", "api_key"),
        ("google", "GOOGLE_API_KEY", "api_key"),
    ],
    "cursor-agent": [],
}


def _resolve_engine_credentials(engine: str) -> dict[str, str]:
    """Resolve API keys for the engine from AnimaWorks credential system."""
    entries = _ENGINE_CREDENTIAL_MAP.get(engine, [])
    if not entries:
        return {}

    result: dict[str, str] = {}
    for cred_name, env_var, key_name in entries:
        if env_var in result:
            continue
        try:
            from core.tools._base import get_credential

            val = get_credential(cred_name, f"machine/{engine}", key_name, env_var)
            result[env_var] = val
        except Exception as exc:
            logger.debug("Failed to resolve credential %s for engine %s: %s", cred_name, engine, exc)
    return result


# ── Environment Sanitization ──────────────────────────────


def _build_env(engine: str) -> dict[str, str]:
    """Build a sanitized environment dict for the machine subprocess.

    Copies allowlisted env vars from the current process, then injects
    API keys resolved from AnimaWorks credential system (config.json,
    vault.json, credentials.json).
    """
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _ENV_ALLOWLIST:
            env[k] = v

    for blocked in _ANIMAWORKS_ENV_BLOCKLIST:
        env.pop(blocked, None)

    creds = _resolve_engine_credentials(engine)
    for k, v in creds.items():
        if k not in env:
            env[k] = v

    return env


# ── Instruction Prefix ─────────────────────────────────────


def _build_instruction(instruction: str, working_directory: str) -> str:
    """Prepend workspace scope constraint to the user instruction."""
    prefix = (
        f"WORKSPACE CONSTRAINT: You MUST only modify files within {working_directory}. "
        f"You may read files outside this directory, but all writes, edits, and file "
        f"creation must be within the workspace directory. "
        f"Do NOT access or modify any files in ~/.animaworks/ or any AnimaWorks "
        f"data directories.\n\n"
    )
    return prefix + instruction


# ── Command Builder ────────────────────────────────────────


def _build_command(
    engine: str,
    working_directory: str,
    model: str | None = None,
) -> list[str]:
    """Build the CLI command for the selected engine.

    The instruction is NOT included in the command — it is passed via stdin
    to avoid shell escaping issues and hangs with some CLIs.
    """
    base = list(_ENGINE_COMMANDS[engine])

    perm_flags = _ENGINE_PERMISSION_FLAGS.get(engine, [])
    base.extend(perm_flags)

    if model:
        flag = _ENGINE_MODEL_FLAGS.get(engine)
        if flag:
            base.extend([flag, model])

    workdir_flags = _ENGINE_WORKDIR_FLAGS.get(engine, [])
    if workdir_flags:
        base.extend([*workdir_flags, working_directory])

    return base


# ── Execution ──────────────────────────────────────────────


def _stream_to_file(
    proc: subprocess.Popen,
    output_path: Path,
    engine: str,
    instruction_preview: str,
    working_directory: str,
    timeout: int,
) -> tuple[int, float, bool, str]:
    """Stream process output to file. Returns (exit_code, elapsed, timed_out, output_path_str)."""
    from core.time_utils import now_local

    start = time.monotonic()
    cmd_id = output_path.stem

    # Write header
    header = (
        f"--- {cmd_id} ---\n"
        f"pid: {proc.pid}\n"
        f"engine: {engine}\n"
        f"instruction: {instruction_preview}\n"
        f"working_directory: {working_directory}\n"
        f"started_at: {now_local().isoformat()}\n"
        f"status: running\n"
        f"---\n"
    )
    output_path.write_text(header, encoding="utf-8")

    max_bytes = 10 * 1024 * 1024  # 10 MB
    total_bytes = 0
    truncated = False

    def _drain(pipe, prefix=""):
        nonlocal total_bytes, truncated
        if pipe is None:
            return
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                for line in pipe:
                    if truncated:
                        break
                    total_bytes += len(line.encode("utf-8", errors="replace"))
                    if total_bytes > max_bytes:
                        truncated = True
                        f.write(
                            f"\n... (output truncated at {max_bytes // (1024 * 1024)} MB) ...\n"
                        )
                        f.flush()
                        break
                    f.write(f"{prefix}{line}")
                    f.flush()
        except (ValueError, OSError):
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    stdout_thread = _threading.Thread(target=_drain, args=(proc.stdout,), daemon=True)
    stderr_thread = _threading.Thread(
        target=_drain, args=(proc.stderr, "[stderr] "), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    elapsed = time.monotonic() - start
    exit_code = proc.returncode if proc.returncode is not None else -1

    # Write footer
    footer = f"\n--- FINISHED ---\nexit_code: {exit_code}\nelapsed_seconds: {round(elapsed, 1)}\n"
    if timed_out:
        footer += "timed_out: true\n"
    footer += "---\n"
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(footer)

    return exit_code, elapsed, timed_out, str(output_path)


def _execute(
    engine: str,
    instruction: str,
    working_directory: str,
    model: str | None = None,
    timeout: int | None = None,
    anima_dir: str | None = None,
) -> ToolResult:
    """Execute a machine tool synchronously."""
    exe = shutil.which(_ENGINE_COMMANDS[engine][0])
    if exe is None:
        return ToolResult(
            success=False,
            error=t("machine.engine_not_found", engine=engine),
        )

    wd = Path(working_directory)
    if not wd.is_dir():
        return ToolResult(
            success=False,
            error=t("machine.working_directory_not_found", path=working_directory),
        )

    full_instruction = _build_instruction(instruction, working_directory)
    cmd = _build_command(engine, working_directory, model)
    env = _build_env(engine)

    effective_timeout = timeout or _DEFAULT_TIMEOUT_SYNC

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_directory,
            env=env,
            start_new_session=True,
        )

        # Write instruction to stdin and close
        try:
            proc.stdin.write(full_instruction)
            proc.stdin.flush()
            proc.stdin.close()
        except (OSError, BrokenPipeError):
            pass

        # Determine output directory
        if anima_dir:
            output_dir = Path(anima_dir) / "state" / "cmd_output"
        else:
            output_dir = Path(working_directory) / ".cmd_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd_id = _next_machine_id()
        output_path = output_dir / f"{cmd_id}.txt"

        exit_code, elapsed, timed_out, output_file = _stream_to_file(
            proc,
            output_path,
            engine,
            instruction[:100],
            working_directory,
            effective_timeout,
        )

        # Read final output from file for the result
        try:
            raw_output = output_path.read_text(encoding="utf-8")
        except OSError:
            raw_output = ""

        if len(raw_output) > _MAX_OUTPUT_CHARS:
            raw_output = (
                raw_output[:_MAX_OUTPUT_CHARS]
                + f"\n\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"
            )

        if timed_out:
            return ToolResult(
                success=False,
                text=raw_output,
                error=t("machine.timeout", engine=engine, seconds=effective_timeout),
                data={
                    "engine": engine,
                    "elapsed_seconds": round(elapsed, 1),
                    "timed_out": True,
                    "output_file": output_file,
                },
            )

        if exit_code == 0:
            return ToolResult(
                success=True,
                text=raw_output,
                data={
                    "engine": engine,
                    "exit_code": 0,
                    "elapsed_seconds": round(elapsed, 1),
                    "output_file": output_file,
                },
            )
        else:
            return ToolResult(
                success=False,
                text=raw_output,
                error=t("machine.engine_failed", engine=engine, code=exit_code),
                data={
                    "engine": engine,
                    "exit_code": exit_code,
                    "elapsed_seconds": round(elapsed, 1),
                    "output_file": output_file,
                },
            )
    except Exception as exc:
        return ToolResult(
            success=False,
            error=t("machine.unexpected_error", engine=engine, error=str(exc)),
        )


# ── Validation ─────────────────────────────────────────────


def _validate_working_directory(working_directory: str, anima_dir: str | None) -> str | None:
    """Check that working_directory is safe (not inside Anima memory dirs)."""
    if not anima_dir:
        return None
    wd = Path(working_directory).resolve()
    ad = Path(anima_dir).resolve()
    protected = ["memory", "episodes", "knowledge", "procedures", "shortterm", "activity_log", "state"]
    for dirname in protected:
        forbidden = ad / dirname
        if wd == forbidden or forbidden in wd.parents or wd in forbidden.parents:
            return t("machine.forbidden_directory", path=working_directory)
    return None


# ── Tool Schema ────────────────────────────────────────────


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return tool schemas for the machine tool.

    Dynamically probes PATH for available engine CLIs.  If no engines are
    found, returns an empty list — the tool is effectively hidden from the
    Anima.

    Only the top-priority engine is shown in the description.  Animas can
    pass ``engine="__list__"`` to discover all available engines.
    """
    available = _get_available_engines()
    if not available:
        return []

    top = available[0]
    others = len(available) - 1

    if others > 0:
        description = t("machine.schema.description_multi", top=top, others=others)
        engine_desc = t("machine.schema.engine_multi", top=top)
    else:
        description = t("machine.schema.description_single", top=top)
        engine_desc = t("machine.schema.engine_single", top=top)

    return [
        {
            "name": "machine_run",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string",
                        "description": engine_desc,
                    },
                    "instruction": {
                        "type": "string",
                        "description": t("machine.schema.instruction"),
                    },
                    "working_directory": {
                        "type": "string",
                        "description": t("machine.schema.working_directory_with_alias"),
                    },
                    "background": {
                        "type": "boolean",
                        "description": t("machine.schema.background"),
                        "default": False,
                    },
                    "model": {
                        "type": "string",
                        "description": t("machine.schema.model"),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": t("machine.schema.timeout"),
                    },
                },
                "required": ["engine", "instruction", "working_directory"],
            },
        }
    ]


# ── Engine List ────────────────────────────────────────────


def _handle_list_engines() -> str:
    """Return a priority-ordered list of available engines with descriptions."""
    available = _get_available_engines()
    engines = []
    for i, e in enumerate(available):
        desc_key = f"machine.engine_desc_{e.replace('-', '_')}"
        engines.append(
            {
                "rank": i + 1,
                "name": e,
                "description": t(desc_key),
                "recommended": i == 0,
            }
        )
    return json.dumps(
        {
            "engines": engines,
            "recommended": available[0] if available else None,
            "total": len(available),
            "hint": t("machine.list_hint"),
        },
        ensure_ascii=False,
        indent=2,
    )


# ── Dispatch ───────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any]) -> str:
    """Dispatch a machine tool call."""
    if name not in ("machine_run", "run"):
        return json.dumps({"error": f"Unknown action: {name}"}, ensure_ascii=False)

    engine = args.get("engine", "")

    if engine == _LIST_SENTINEL:
        return _handle_list_engines()

    instruction = args.get("instruction", "")
    working_directory_raw = args.get("working_directory", "")
    if working_directory_raw:
        try:
            from core.workspace import resolve_workspace

            working_directory = str(resolve_workspace(working_directory_raw))
        except ValueError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    else:
        working_directory = ""
    background = args.get("background", False)
    model = args.get("model")
    timeout = args.get("timeout")
    anima_dir = args.get("anima_dir", "")
    trigger = args.get("trigger", "chat")

    if engine not in _VALID_ENGINES:
        return json.dumps(
            {"error": t("machine.invalid_engine", engine=engine, valid=", ".join(sorted(_VALID_ENGINES)))},
            ensure_ascii=False,
        )

    if not instruction.strip():
        return json.dumps({"error": t("machine.empty_instruction")}, ensure_ascii=False)

    if not working_directory:
        return json.dumps({"error": t("machine.missing_working_directory")}, ensure_ascii=False)

    dir_err = _validate_working_directory(working_directory, anima_dir)
    if dir_err:
        return json.dumps({"error": dir_err}, ensure_ascii=False)

    rate_err = _check_rate_limit(anima_dir, trigger)
    if rate_err:
        return json.dumps({"error": rate_err}, ensure_ascii=False)

    _increment_call_count(anima_dir)

    if background:
        effective_timeout = timeout or _DEFAULT_TIMEOUT_ASYNC
    else:
        effective_timeout = timeout or _DEFAULT_TIMEOUT_SYNC

    result = _execute(
        engine,
        instruction,
        working_directory,
        model=model,
        timeout=effective_timeout,
        anima_dir=anima_dir or None,
    )

    output: dict[str, Any] = {
        "success": result.success,
        "engine": engine,
    }
    if result.text:
        output["output"] = result.text
    if result.error:
        output["error"] = result.error
    if result.data:
        output.update({k: v for k, v in result.data.items() if v is not None})

    return json.dumps(output, ensure_ascii=False, indent=2)


# ── CLI ────────────────────────────────────────────────────


def cli_main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``animaworks-tool machine run``."""
    parser = argparse.ArgumentParser(
        description="Run an external agent CLI as a machine tool",
    )
    sub = parser.add_subparsers(dest="subcommand")

    run_parser = sub.add_parser("run", help="Execute a machine tool")
    run_parser.add_argument("instruction", help="Task instruction for the machine")
    _cli_default_engine = (_get_available_engines() or ["claude"])[0]
    run_parser.add_argument(
        "-e",
        "--engine",
        choices=sorted(_VALID_ENGINES),
        default=_cli_default_engine,
        help=f"Engine to use (default: {_cli_default_engine})",
    )
    run_parser.add_argument(
        "-d",
        "--working-directory",
        default=os.getcwd(),
        help="Working directory (default: cwd)",
    )
    run_parser.add_argument("-m", "--model", help="Model override")
    run_parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        help="Timeout in seconds",
    )
    run_parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    parsed = parser.parse_args(argv)
    if not parsed.subcommand:
        parser.print_help()
        return

    result = _execute(
        engine=parsed.engine,
        instruction=parsed.instruction,
        working_directory=parsed.working_directory,
        model=parsed.model,
        timeout=parsed.timeout,
    )

    if parsed.json:
        out: dict[str, Any] = {
            "success": result.success,
            "engine": parsed.engine,
        }
        if result.text:
            out["output"] = result.text
        if result.error:
            out["error"] = result.error
        if result.data:
            out.update({k: v for k, v in result.data.items() if v is not None})
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if result.success:
            if result.text and result.text.strip():
                print(result.text)
            else:
                elapsed = result.data.get("elapsed_seconds", "?") if result.data else "?"
                print(f"[machine/{parsed.engine}] Completed in {elapsed}s (no output)")
        else:
            import sys

            print(f"Error: {result.error}", file=sys.stderr)
            if result.text:
                print(result.text, file=sys.stderr)
            sys.exit(1)
