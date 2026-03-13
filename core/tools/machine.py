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
import time
from pathlib import Path
from typing import Any

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


# ── Environment Sanitization ──────────────────────────────


def _build_env(engine: str) -> dict[str, str]:
    """Build a sanitized environment dict for the machine subprocess."""
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _ENV_ALLOWLIST:
            env[k] = v

    for blocked in _ANIMAWORKS_ENV_BLOCKLIST:
        env.pop(blocked, None)

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


def _execute(
    engine: str,
    instruction: str,
    working_directory: str,
    model: str | None = None,
    timeout: int | None = None,
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

    start = time.monotonic()
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
        stdout, stderr = proc.communicate(input=full_instruction, timeout=effective_timeout)

        elapsed = time.monotonic() - start
        output = stdout or ""

        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + f"\n\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"

        if proc.returncode == 0:
            return ToolResult(
                success=True,
                text=output,
                data={
                    "engine": engine,
                    "exit_code": 0,
                    "elapsed_seconds": round(elapsed, 1),
                    "stderr_excerpt": stderr[:500] if stderr else None,
                },
            )
        else:
            combined = output
            if stderr:
                combined += f"\n\n--- stderr ---\n{stderr[:2000]}"
            return ToolResult(
                success=False,
                text=combined,
                error=t("machine.engine_failed", engine=engine, code=proc.returncode),
                data={
                    "engine": engine,
                    "exit_code": proc.returncode,
                    "elapsed_seconds": round(elapsed, 1),
                },
            )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        partial = ""
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                remaining_out, _ = proc.communicate(timeout=5)
                if remaining_out:
                    partial = remaining_out
            except Exception:
                proc.kill()
        return ToolResult(
            success=False,
            text=partial[:_MAX_OUTPUT_CHARS] if partial else "",
            error=t("machine.timeout", engine=engine, seconds=effective_timeout),
            data={
                "engine": engine,
                "elapsed_seconds": round(elapsed, 1),
                "timed_out": True,
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
    """Return tool schemas for the machine tool."""
    return [
        {
            "name": "machine_run",
            "description": (
                "外部エージェントCLI（工作機械）にタスクを委託する。"
                "工作機械は指示されたタスクのみを実行するステートレスな道具であり、"
                "Animaの記憶・通信・組織情報にはアクセスできない。\n\n"
                "【重要】instruction には以下を必ず含めること:\n"
                "- 達成すべきゴールの具体的な記述\n"
                "- 対象ファイル・モジュールの明示\n"
                "- 制約条件（コーディング規約、既存APIとの整合性等）\n"
                "- 期待する出力形式\n"
                "曖昧な指示は低品質な結果につながる。職人が工作機械に渡す設計図のように、"
                "正確かつ詳細に記述すること。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string",
                        "enum": sorted(_VALID_ENGINES),
                        "description": "使用する工作機械（外部エージェントCLI）",
                    },
                    "instruction": {
                        "type": "string",
                        "description": ("工作機械への詳細な作業指示。ゴール・対象・制約・期待出力を明記する"),
                    },
                    "working_directory": {
                        "type": "string",
                        "description": ("作業ディレクトリの絶対パス。工作機械はこのディレクトリ内でのみ書き込み可能"),
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "true: 非同期実行（結果は次回heartbeatで取得）。false: 同期実行（結果を直接返す）"
                        ),
                        "default": False,
                    },
                    "model": {
                        "type": "string",
                        "description": "使用モデル（省略時はengineのデフォルト）",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "タイムアウト秒数。同期時デフォルト600、非同期時デフォルト1800",
                    },
                },
                "required": ["engine", "instruction", "working_directory"],
            },
        }
    ]


# ── Dispatch ───────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any]) -> str:
    """Dispatch a machine tool call."""
    if name not in ("machine_run", "run"):
        return json.dumps({"error": f"Unknown action: {name}"}, ensure_ascii=False)

    engine = args.get("engine", "")
    instruction = args.get("instruction", "")
    working_directory = args.get("working_directory", "")
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

    result = _execute(engine, instruction, working_directory, model, effective_timeout)

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
    run_parser.add_argument(
        "-e",
        "--engine",
        choices=sorted(_VALID_ENGINES),
        default="claude",
        help="Engine to use (default: claude)",
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
            print(result.text)
        else:
            import sys

            print(f"Error: {result.error}", file=sys.stderr)
            if result.text:
                print(result.text, file=sys.stderr)
            sys.exit(1)
