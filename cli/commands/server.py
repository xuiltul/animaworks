# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from core.platform.process import (
    find_first_matching_pid,
    subprocess_session_kwargs,
    terminate_matching_processes,
    terminate_pid,
)
from core.platform.process import (
    is_process_alive as is_pid_alive,
)

logger = logging.getLogger("animaworks")

# Command patterns used to identify the animaworks server process.
# Matches both direct invocation (main.py start) and entry point (animaworks start).
_SERVER_CMD_MARKERS = ("main.py start", "animaworks start", "-m cli start")

_DAEMON_STARTUP_TIMEOUT = 10
_DAEMON_POLL_INTERVAL = 0.3
# systemd unit templates set RestartPreventExitStatus=3 so an
# already-running process does not trigger Restart=on-failure loops.
EXIT_ALREADY_RUNNING = 3


# ── PID helpers ───────────────────────────────────────────


def _get_pid_file() -> Path:
    """Return the path to the server PID file."""
    from core.paths import get_data_dir

    return get_data_dir() / "server.pid"


def _write_pid_file() -> None:
    """Write the current process PID to the PID file."""
    pid_file = _get_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    logger.info("PID file written: %s (pid=%d)", pid_file, os.getpid())


def _remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    pid_file = _get_pid_file()
    try:
        pid_file.unlink(missing_ok=True)
        logger.debug("PID file removed: %s", pid_file)
    except OSError as exc:
        logger.warning("Failed to remove PID file %s: %s", pid_file, exc)


def _read_pid() -> int | None:
    """Read and validate the PID from the PID file.

    Returns the PID if the file exists and contains a valid integer,
    or None if the file is missing or contains invalid data.
    """
    pid_file = _get_pid_file()
    if not pid_file.exists():
        return None
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
        return int(text)
    except (ValueError, OSError) as exc:
        logger.warning("Invalid PID file %s: %s", pid_file, exc)
        return None


def _is_process_alive(pid: int) -> bool:
    """Check whether a process with the given PID is currently running."""
    return is_pid_alive(pid)


def _find_server_pid_by_process(
    extra_exclude_pids: set[int] | None = None,
) -> int | None:
    """Find the animaworks server process by command pattern."""
    excluded = {os.getpid(), os.getppid()}
    if extra_exclude_pids:
        excluded |= extra_exclude_pids
    helper_pid_str = os.environ.get("_ANIMAWORKS_RESTART_HELPER_PID")
    if helper_pid_str:
        try:
            excluded.add(int(helper_pid_str))
        except ValueError:
            pass
    return find_first_matching_pid(
        _SERVER_CMD_MARKERS,
        exclude_pids=excluded,
        require_python=True,
    )


def _stop_server(
    timeout: int = 10,
    *,
    force: bool = False,
    extra_exclude_pids: set[int] | None = None,
) -> bool:
    """Send SIGTERM to the running server and wait for it to exit.

    First tries the PID file.  If the file is missing, falls back to
    scanning running processes by command pattern so that the server can
    still be stopped even when the PID file was lost.

    Args:
        timeout: Maximum seconds to wait before reporting failure.
        force: If True, escalate to SIGKILL after SIGTERM timeout.
        extra_exclude_pids: Additional PIDs to exclude from process
            scanning (e.g. the restart helper).

    Returns:
        True if the server was stopped (or was not running), False if
        it failed to stop within the timeout.
    """

    def _cleanup_orphans() -> int:
        orphans = _kill_orphan_runners()
        if orphans:
            print(f"Killed {orphans} orphan runner process(es).")
        return orphans

    pid = _read_pid()

    if pid is None:
        pid = _find_server_pid_by_process(extra_exclude_pids=extra_exclude_pids)
        if pid is None:
            _cleanup_orphans()
            print("No PID file found and no server process detected. Server is not running.")
            return True
        print(f"PID file missing — found server process by scanning (pid={pid}).")
    else:
        if not _is_process_alive(pid):
            print(f"Stale PID file (pid={pid}). Server is not running. Cleaning up.")
            _remove_pid_file()
            _cleanup_orphans()
            return True

    print(f"Stopping server (pid={pid})...")
    try:
        terminate_pid(pid, force=False, include_children=False)
    except ProcessLookupError:
        print("Server already exited.")
        _remove_pid_file()
        _cleanup_orphans()
        return True
    except Exception:
        print(f"Error: Permission denied sending signal to pid={pid}.")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            print("Server stopped.")
            _remove_pid_file()
            _cleanup_orphans()
            return True
        time.sleep(0.2)

    if not force:
        print(f"Error: Server (pid={pid}) did not stop within {timeout}s.")
        return False

    # Force mode: escalate to SIGKILL
    print(f"Server (pid={pid}) did not stop within {timeout}s. Sending SIGKILL...")
    try:
        terminate_pid(pid, force=True, include_children=True)
    except ProcessLookupError:
        pass
    except Exception:
        print(f"Error: Permission denied sending SIGKILL to pid={pid}.")
        return False

    kill_deadline = time.monotonic() + 3
    while time.monotonic() < kill_deadline:
        if not _is_process_alive(pid):
            break
        time.sleep(0.1)

    if _is_process_alive(pid):
        print(f"Error: Server (pid={pid}) still alive after SIGKILL.")
        return False

    print("Server force-killed.")
    _remove_pid_file()
    _cleanup_orphans()

    return True


# ── Orphan runner cleanup ─────────────────────────────────

_RUNNER_CMD_MARKER = "core.supervisor.runner"


def _kill_orphan_runners() -> int:
    """Kill orphaned Anima runner processes from previous server instances.

    Uses psutil to find processes whose command line contains the runner module
    marker and references the ~/.animaworks/ data directory.

    Returns the number of processes targeted.
    """
    from core.paths import get_data_dir

    data_prefix = str(get_data_dir())
    killed = terminate_matching_processes(
        (_RUNNER_CMD_MARKER,),
        path_contains=data_prefix,
        exclude_pids={os.getpid(), os.getppid()},
        force=False,
        include_children=False,
        require_python=True,
    )
    if killed:
        time.sleep(1)

    return killed


# ── Daemon helpers ────────────────────────────────────────


def _is_port_listening(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def _get_daemon_log_path() -> Path:
    """Return path for daemon stdout/stderr redirect."""
    from core.paths import get_data_dir

    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "server-daemon.log"


def _spawn_daemon(args: argparse.Namespace) -> None:
    """Spawn the server as a background process and verify startup."""
    from core.paths import get_data_dir

    existing_pid = _read_pid()
    if existing_pid is not None and _is_process_alive(existing_pid):
        print(f"Error: Server is already running (pid={existing_pid}).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(EXIT_ALREADY_RUNNING)
    elif existing_pid is not None:
        _remove_pid_file()

    orphan_pid = _find_server_pid_by_process()
    if orphan_pid is not None and _is_process_alive(orphan_pid):
        print(f"Error: Server is already running (pid={orphan_pid}, PID file was missing).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(EXIT_ALREADY_RUNNING)

    cmd = [sys.executable, "-m", "cli", "start", "--foreground", "--host", args.host, "--port", str(args.port)]

    log_path = _get_daemon_log_path()
    try:
        from core.memory.housekeeping import _rotate_daemon_log

        _rotate_daemon_log(log_path, max_size_mb=50, keep_generations=5)
    except Exception:
        logger.debug("Failed to rotate daemon log before spawn: %s", log_path, exc_info=True)
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).resolve().parent.parent.parent,
        **subprocess_session_kwargs(),
    )
    log_file.close()

    check_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    deadline = time.monotonic() + _DAEMON_STARTUP_TIMEOUT
    started = False

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"Error: Server exited immediately (exit code {proc.returncode}).")
            print(f"Check logs: {log_path}")
            sys.exit(1)
        if _is_port_listening(check_host, args.port):
            started = True
            break
        time.sleep(_DAEMON_POLL_INTERVAL)

    if not started:
        print(f"Warning: Server process started (pid={proc.pid}) but port {args.port} not yet listening.")
        print(f"Check logs: {log_path}")
        return

    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    config_data_dir = get_data_dir()
    print(f"Server started (pid={proc.pid}).")
    print(f"  Dashboard: http://{display_host}:{args.port}/")
    print(f"  Logs:      {log_path}")
    print(f"  Data:      {config_data_dir}")
    print("  Stop:      animaworks stop")


# ── Server commands ───────────────────────────────────────


def _start_pid_watchdog() -> None:
    """Start a background thread that re-creates the PID file if it vanishes.

    Checks every 30 seconds.  If the file is missing or contains a stale PID,
    it is rewritten with the current process PID.  This guards against
    accidental deletion (e.g. by init --force, manual cleanup, etc.).
    """
    import threading

    def _watchdog() -> None:
        my_pid = os.getpid()
        while True:
            time.sleep(30)
            try:
                current = _read_pid()
                if current == my_pid:
                    continue
                # PID file missing, empty, or pointing at a different/dead process
                logger.warning(
                    "PID file watchdog: file missing or stale (read=%s, expected=%d). Re-creating.",
                    current,
                    my_pid,
                )
                _write_pid_file()
            except Exception:
                # Don't let the watchdog crash; just log and retry next cycle
                logger.debug("PID watchdog error", exc_info=True)

    t = threading.Thread(target=_watchdog, daemon=True, name="pid-watchdog")
    t.start()


def cmd_start(args: argparse.Namespace) -> None:
    """Start the AnimaWorks server.

    Default: daemonize (background).  With --foreground: run in the
    current terminal with log output (old behaviour).
    """
    if not getattr(args, "foreground", False):
        _spawn_daemon(args)
        return

    _start_foreground(args)


def _pin_native_threads() -> None:
    """Mitigate tokenizer thread race before importing torch.

    ``TOKENIZERS_PARALLELISM=false``: prevents HuggingFace tokenizer
    Rust threads from racing with Python threading in multi-thread
    embedding scenarios.
    """
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _run_rag_startup_preflight(*, force_all_vectordb: bool = False) -> None:
    """Repair suspected corrupt RAG DBs before the server imports Chroma."""
    try:
        from core import startup_progress
        from core.config import load_config

        startup_progress.set_phase("preflight", detail="Checking RAG vector databases", reset_counts=True)
        startup_progress.raise_if_cancelled()
        config = load_config()
        rag = config.rag
        if not config.setup_complete:
            startup_progress.update_progress(detail="Setup is not complete", done_count=0, total_count=0)
            return
        if not bool(getattr(rag, "repair_enabled", True)):
            startup_progress.update_progress(detail="RAG repair is disabled", done_count=0, total_count=0)
            return
        if not bool(getattr(rag, "startup_repair_preflight_enabled", True)):
            startup_progress.update_progress(detail="RAG startup preflight is disabled", done_count=0, total_count=0)
            return

        from core.memory.rag.repair import get_repair_service

        service = get_repair_service()
        window_minutes = int(getattr(rag, "startup_repair_window_minutes", 1440))
        quick_check_timeout = float(getattr(rag, "quick_check_timeout_seconds", 10.0))
        suspects = service.discover_suspect_animas(
            window_minutes=window_minutes,
            quick_check_timeout_seconds=quick_check_timeout,
            quick_check_source="startup_quick_check",
        )
        startup_progress.raise_if_cancelled()
        reason = "startup_chroma_crash_preflight"
        if force_all_vectordb:
            logger.info(
                "Ignoring startup full repair request from unclean previous exit; using corruption suspects only"
            )
        if not suspects:
            logger.info("RAG startup preflight: no suspect DBs found")
            startup_progress.update_progress(detail="No suspect vector databases found", done_count=0, total_count=0)
            return

        joined = ", ".join(suspects)
        print(f"RAG startup preflight: repairing suspected vector DB(s): {joined}")
        startup_progress.set_phase("repairing", detail=joined, done_count=0, total_count=len(suspects))
        results = service.repair_animas_if_allowed(
            suspects,
            reason=reason,
            source="startup_preflight",
            include_shared=True,
        )
        for result in results.values():
            if result.ok:
                logger.warning(
                    "RAG startup preflight repaired %s: chunks=%s quarantine=%s",
                    result.anima_name,
                    result.chunks_indexed,
                    result.quarantine_path,
                )
            else:
                logger.error(
                    "RAG startup preflight failed for %s: status=%s stage=%s error=%s",
                    result.anima_name,
                    result.status,
                    result.stage,
                    result.error,
                )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("RAG startup preflight failed unexpectedly; continuing server startup")


def _run_rag_startup_preflight_via_worker(*, force_all_vectordb: bool = False) -> None:
    """Run startup RAG repair with ChromaDB isolated in a temporary worker."""
    try:
        from core.config import load_config

        config = load_config()
        if not config.setup_complete:
            _run_rag_startup_preflight(force_all_vectordb=force_all_vectordb)
            return

        rag = config.rag
        if not bool(getattr(rag, "repair_enabled", True)):
            return
        if not bool(getattr(rag, "startup_repair_preflight_enabled", True)):
            return

        from core.memory.rag.vector_worker_client import start_temporary_vector_worker
        from core.paths import get_data_dir

        worker = start_temporary_vector_worker(
            config=config,
            log_dir=get_data_dir() / "logs",
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("RAG startup preflight vector worker unavailable; continuing server startup")
        return

    try:
        _run_rag_startup_preflight(force_all_vectordb=force_all_vectordb)
    except asyncio.CancelledError:
        raise
    finally:
        worker.stop()


def _start_foreground(args: argparse.Namespace) -> None:
    """Run the server in the foreground (blocking, with log output)."""
    _pin_native_threads()

    import uvicorn

    from core.init import ensure_runtime_dir
    from core.paths import get_animas_dir, get_shared_dir
    from core.platform.fd_limits import raise_fd_soft_limit
    from server.app import create_app

    raise_fd_soft_limit(logger=logger, process_label="server")

    existing_pid = _read_pid()
    if existing_pid is not None and _is_process_alive(existing_pid):
        print(f"Error: Server is already running (pid={existing_pid}).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(EXIT_ALREADY_RUNNING)
    elif existing_pid is not None:
        logger.info("Stale PID file found (pid=%d). Cleaning up.", existing_pid)
        _remove_pid_file()

    orphan_pid = _find_server_pid_by_process()
    if orphan_pid is not None and _is_process_alive(orphan_pid):
        print(f"Error: Server is already running (pid={orphan_pid}, PID file was missing).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(EXIT_ALREADY_RUNNING)

    orphan_count = _kill_orphan_runners()
    if orphan_count:
        print(f"Killed {orphan_count} orphan runner process(es) from previous server.")

    ensure_runtime_dir()
    _write_pid_file()
    atexit.register(_remove_pid_file)
    _start_pid_watchdog()

    from core.config import load_config
    from core.time_utils import configure_timezone

    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    config = load_config()
    configure_timezone(config.system.timezone)
    if not config.setup_complete:
        print(f"Open http://{display_host}:{args.port}/setup/ to configure your animas and settings.")
    else:
        print(f"Dashboard starting at http://{display_host}:{args.port}/")

    try:
        app = create_app(get_animas_dir(), get_shared_dir())
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            timeout_keep_alive=65,
            ws_ping_interval=25,
            ws_ping_timeout=5,
        )
    finally:
        _remove_pid_file()


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the server (alias for 'start')."""
    cmd_start(args)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the running AnimaWorks server."""
    force = getattr(args, "force", False)
    if not _stop_server(force=force):
        sys.exit(1)


def _clear_pycache() -> int:
    """Remove all __pycache__ directories under the project root.

    Returns the number of directories removed.
    """
    import shutil

    project_root = Path(__file__).resolve().parent.parent.parent
    count = 0
    for cache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            count += 1
        except OSError as exc:
            logger.warning("Failed to remove %s: %s", cache_dir, exc)
    return count


def _get_restart_status_path() -> Path:
    """Return path for the restart helper result file."""
    from core.paths import get_data_dir

    return get_data_dir() / "run" / "restart_helper_result.json"


def _spawn_restart_helper(args: argparse.Namespace, old_pid: int | None) -> int:
    """Spawn a fully detached restart helper that survives caller death.

    The helper is a new Python process in its own session.  It waits for
    the old server to exit (if *old_pid* is given), then starts a fresh
    daemon with retry logic.  Because it runs in a separate session &
    process group, it is immune to SIGTERM cascading through the caller's
    process tree — which is critical when an Anima triggers
    ``animaworks restart`` from inside the server.

    The helper writes its progress to ``restart-helper.log`` and a
    machine-readable result to ``run/restart_helper_result.json``.

    Returns the helper PID.
    """
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 18500)
    project_root = str(Path(__file__).resolve().parent.parent.parent)

    from core.paths import get_data_dir

    data_dir = str(get_data_dir())

    helper_code = f"""
import json, os, socket, sys, time, subprocess, traceback
from datetime import datetime, timezone
from pathlib import Path
from core.platform.process import (
    find_first_matching_pid,
    is_process_alive,
    subprocess_session_kwargs,
    terminate_pid,
)

os.chdir({project_root!r})
DATA_DIR = Path({data_dir!r})
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "restart-helper.log"
STATUS_FILE = DATA_DIR / "run" / "restart_helper_result.json"
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

old_pid = {old_pid!r}
host = {host!r}
port = {port!r}
_SERVER_CMD_MARKERS = {_SERVER_CMD_MARKERS!r}
MAX_RETRIES = 3
RETRY_DELAY = 5
PORT_WAIT_TIMEOUT = 15

def _log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    line = f"[{{ts}}] [restart-helper] {{msg}}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\\n")
    except OSError:
        pass

def _write_status(success, detail=""):
    try:
        STATUS_FILE.write_text(json.dumps({{
            "success": success,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
        }}), encoding="utf-8")
    except OSError:
        pass

def _alive(pid):
    return is_process_alive(pid)

def _find_server_process():
    return find_first_matching_pid(
        _SERVER_CMD_MARKERS,
        exclude_pids={{os.getpid(), os.getppid()}},
        require_python=True,
    )

def _is_port_listening(h, p):
    try:
        with socket.create_connection((h, p), timeout=1):
            return True
    except OSError:
        return False

_log(f"Started (pid={{os.getpid()}}, old_pid={{old_pid}})")

# Phase 1: Wait for old server to exit
if old_pid is not None:
    _log(f"Waiting for old server (pid={{old_pid}}) to exit...")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _alive(old_pid):
        time.sleep(0.3)
    if _alive(old_pid):
        _log(f"Old server still alive after 30s, force-killing pid={{old_pid}}")
        try:
            terminate_pid(old_pid, force=True, include_children=True)
        except (OSError, ProcessLookupError):
            pass
        time.sleep(1)
    _log("Old server exited")

# Phase 2: Wait for any lingering server process to disappear
scan_deadline = time.monotonic() + 15
while time.monotonic() < scan_deadline:
    if _find_server_process() is None:
        break
    time.sleep(0.5)

lingering_pid = _find_server_process()
if lingering_pid is not None:
    _log(f"Lingering server process still detected (pid={{lingering_pid}}); force-stopping before restart")
    try:
        terminate_pid(lingering_pid, force=True, include_children=True)
    except (OSError, ProcessLookupError):
        pass
    kill_deadline = time.monotonic() + 5
    while time.monotonic() < kill_deadline and _alive(lingering_pid):
        time.sleep(0.2)
    if _alive(lingering_pid):
        _write_status(False, f"Lingering server process still alive after force-stop: pid={{lingering_pid}}")
        _log("FAILED: lingering server process survived force-stop")
        sys.exit(1)

time.sleep(0.5)

# Phase 3: Start new server with retries
check_host = "127.0.0.1" if host == "0.0.0.0" else host
cmd = [sys.executable, "-m", "cli", "start", "--host", host, "--port", str(port)]
os.environ["_ANIMAWORKS_RESTART_HELPER_PID"] = str(os.getpid())

for attempt in range(1, MAX_RETRIES + 1):
    _log(f"Starting server (attempt {{attempt}}/{{MAX_RETRIES}})...")
    try:
        proc = subprocess.Popen(cmd, cwd={project_root!r}, **subprocess_session_kwargs())
        _log(f"Spawned server process pid={{proc.pid}}")
    except Exception as e:
        _log(f"Failed to spawn server: {{e}}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            continue
        _write_status(False, f"All {{MAX_RETRIES}} spawn attempts failed: {{e}}")
        _log("FAILED: all spawn attempts exhausted")
        sys.exit(1)

    # Wait for port to become available
    port_deadline = time.monotonic() + PORT_WAIT_TIMEOUT
    started = False
    while time.monotonic() < port_deadline:
        if proc.poll() is not None:
            _log(f"Server exited immediately (code={{proc.returncode}})")
            break
        if _is_port_listening(check_host, port):
            started = True
            break
        time.sleep(0.5)

    if started:
        _write_status(True, f"Server started (pid={{proc.pid}}, attempt={{attempt}})")
        _log(f"SUCCESS: Server listening on {{check_host}}:{{port}} (pid={{proc.pid}})")
        sys.exit(0)

    _log(f"Attempt {{attempt}} failed: port not listening after {{PORT_WAIT_TIMEOUT}}s")
    if attempt < MAX_RETRIES:
        _log(f"Retrying in {{RETRY_DELAY}}s...")
        time.sleep(RETRY_DELAY)

_write_status(False, f"Server did not start after {{MAX_RETRIES}} attempts")
_log(f"FAILED: Server did not start after {{MAX_RETRIES}} attempts")
sys.exit(1)
"""

    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    helper_log = log_dir / "restart-helper.log"
    log_file = open(helper_log, "a", encoding="utf-8")  # noqa: SIM115

    proc = subprocess.Popen(
        [sys.executable, "-c", helper_code],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=project_root,
        **subprocess_session_kwargs(),
    )
    log_file.close()
    return proc.pid


def cmd_restart(args: argparse.Namespace) -> None:
    """Restart the AnimaWorks server (stop then start).

    Spawns a detached restart-helper process **before** stopping the
    current server.  The helper runs in its own session so it survives
    even when the caller is killed during server shutdown (e.g. when an
    Anima triggers restart from inside the server).

    After stopping, waits for the helper to bring the new server up and
    reports success or failure with log path.
    """
    old_pid = _read_pid()
    if old_pid is not None and not _is_process_alive(old_pid):
        old_pid = None
    if old_pid is None:
        old_pid = _find_server_pid_by_process()

    status_path = _get_restart_status_path()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        status_path.unlink()

    helper_pid = _spawn_restart_helper(args, old_pid)
    print(f"Restart helper spawned (pid={helper_pid}). Stopping server...")

    force = getattr(args, "force", False)
    _stop_server(force=force, extra_exclude_pids={helper_pid})

    removed = _clear_pycache()
    if removed:
        print(f"Cleared {removed} __pycache__ directories.")

    port = getattr(args, "port", 18500)
    host = getattr(args, "host", "0.0.0.0")
    check_host = "127.0.0.1" if host == "0.0.0.0" else host

    from core.paths import get_data_dir

    helper_log = get_data_dir() / "logs" / "restart-helper.log"
    daemon_log = _get_daemon_log_path()

    print("Waiting for server to start...")
    deadline = time.monotonic() + 30
    started = False
    while time.monotonic() < deadline:
        if _is_port_listening(check_host, port):
            started = True
            break
        if status_path.exists():
            import json

            try:
                result = json.loads(status_path.read_text(encoding="utf-8"))
                if not result.get("success"):
                    detail = result.get("detail", "unknown error")
                    print(f"Error: Restart helper reported failure: {detail}")
                    print(f"  Helper log: {helper_log}")
                    sys.exit(1)
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.5)

    if started:
        new_pid = _read_pid()
        pid_info = f" (pid={new_pid})" if new_pid else ""
        display_host = "localhost" if host == "0.0.0.0" else host
        print(f"Server restarted successfully{pid_info}.")
        print(f"  Dashboard: http://{display_host}:{port}/")
        print(f"  Logs:      {daemon_log}")
    else:
        print("Error: Server did not start within 30 seconds.")
        print(f"  Helper log: {helper_log}")
        print(f"  Daemon log: {daemon_log}")
        sys.exit(1)


# ── Deprecated modes ──────────────────────────────────────


def cmd_gateway(args: argparse.Namespace) -> None:
    print("Error: 'gateway' mode has been deprecated. Use 'animaworks start' instead.")
    sys.exit(1)


def cmd_worker(args: argparse.Namespace) -> None:
    print("Error: 'worker' mode has been deprecated. Use 'animaworks start' instead.")
    sys.exit(1)
