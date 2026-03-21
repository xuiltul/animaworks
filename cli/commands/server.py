# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
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
        force: If True, escalate to SIGKILL after SIGTERM timeout and
            also kill orphan runner processes.
        extra_exclude_pids: Additional PIDs to exclude from process
            scanning (e.g. the restart helper).

    Returns:
        True if the server was stopped (or was not running), False if
        it failed to stop within the timeout.
    """
    pid = _read_pid()

    if pid is None:
        pid = _find_server_pid_by_process(extra_exclude_pids=extra_exclude_pids)
        if pid is None:
            if force:
                orphans = _kill_orphan_runners()
                if orphans:
                    print(f"Killed {orphans} orphan runner process(es).")
                else:
                    print("No PID file found and no server process detected. Server is not running.")
            else:
                print("No PID file found and no server process detected. Server is not running.")
            return True
        print(f"PID file missing — found server process by scanning (pid={pid}).")
    else:
        if not _is_process_alive(pid):
            print(f"Stale PID file (pid={pid}). Server is not running. Cleaning up.")
            _remove_pid_file()
            if force:
                orphans = _kill_orphan_runners()
                if orphans:
                    print(f"Killed {orphans} orphan runner process(es).")
            return True

    print(f"Stopping server (pid={pid})...")
    try:
        terminate_pid(pid, force=False, include_children=False)
    except ProcessLookupError:
        print("Server already exited.")
        _remove_pid_file()
        if force:
            orphans = _kill_orphan_runners()
            if orphans:
                print(f"Killed {orphans} orphan runner process(es).")
        return True
    except Exception:
        print(f"Error: Permission denied sending signal to pid={pid}.")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            print("Server stopped.")
            _remove_pid_file()
            if force:
                orphans = _kill_orphan_runners()
                if orphans:
                    print(f"Killed {orphans} orphan runner process(es).")
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

    orphans = _kill_orphan_runners()
    if orphans:
        print(f"Killed {orphans} orphan runner process(es).")

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
        sys.exit(1)
    elif existing_pid is not None:
        _remove_pid_file()

    orphan_pid = _find_server_pid_by_process()
    if orphan_pid is not None and _is_process_alive(orphan_pid):
        print(f"Error: Server is already running (pid={orphan_pid}, PID file was missing).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(1)

    cmd = [sys.executable, "-m", "cli", "start", "--foreground", "--host", args.host, "--port", str(args.port)]

    log_path = _get_daemon_log_path()
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


def _start_foreground(args: argparse.Namespace) -> None:
    """Run the server in the foreground (blocking, with log output)."""
    import uvicorn

    from core.init import ensure_runtime_dir
    from core.paths import get_animas_dir, get_shared_dir
    from server.app import create_app

    existing_pid = _read_pid()
    if existing_pid is not None and _is_process_alive(existing_pid):
        print(f"Error: Server is already running (pid={existing_pid}).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(1)
    elif existing_pid is not None:
        logger.info("Stale PID file found (pid=%d). Cleaning up.", existing_pid)
        _remove_pid_file()

    orphan_pid = _find_server_pid_by_process()
    if orphan_pid is not None and _is_process_alive(orphan_pid):
        print(f"Error: Server is already running (pid={orphan_pid}, PID file was missing).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(1)

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
        print(f"Dashboard ready at http://{display_host}:{args.port}/")

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


def _spawn_restart_helper(args: argparse.Namespace, old_pid: int | None) -> int:
    """Spawn a fully detached restart helper that survives caller death.

    The helper is a new Python process in its own session.  It waits for
    the old server to exit (if *old_pid* is given), then starts a fresh
    daemon.  Because it runs in a separate session & process group, it
    is immune to SIGTERM cascading through the caller's process tree —
    which is critical when an Anima triggers ``animaworks restart`` from
    inside the server.

    Returns the helper PID.
    """
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 8000)
    project_root = str(Path(__file__).resolve().parent.parent.parent)

    helper_code = f"""
import os, sys, time, subprocess
from pathlib import Path
from core.platform.process import (
    find_first_matching_pid,
    is_process_alive,
    subprocess_session_kwargs,
    terminate_pid,
)

os.chdir({project_root!r})
old_pid = {old_pid!r}
host = {host!r}
port = {port!r}
_SERVER_CMD_MARKERS = {_SERVER_CMD_MARKERS!r}

def _alive(pid):
    return is_process_alive(pid)

def _find_server_process():
    return find_first_matching_pid(
        _SERVER_CMD_MARKERS,
        exclude_pids={{os.getpid(), os.getppid()}},
        require_python=True,
    )

if old_pid is not None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _alive(old_pid):
        time.sleep(0.3)
    if _alive(old_pid):
        try:
            terminate_pid(old_pid, force=True, include_children=True)
        except (OSError, ProcessLookupError):
            pass
        time.sleep(1)

scan_deadline = time.monotonic() + 15
while time.monotonic() < scan_deadline:
    if _find_server_process() is None:
        break
    time.sleep(0.5)

time.sleep(0.5)
cmd = [sys.executable, "-m", "cli", "start", "--host", host, "--port", str(port)]
subprocess.Popen(cmd, cwd={project_root!r}, **subprocess_session_kwargs())
"""

    log_path = _get_daemon_log_path()
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

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
    """
    old_pid = _read_pid()
    if old_pid is not None and not _is_process_alive(old_pid):
        old_pid = None
    if old_pid is None:
        old_pid = _find_server_pid_by_process()

    helper_pid = _spawn_restart_helper(args, old_pid)
    print(f"Restart helper spawned (pid={helper_pid}). Stopping server...")

    force = getattr(args, "force", False)
    _stop_server(force=force, extra_exclude_pids={helper_pid})

    removed = _clear_pycache()
    if removed:
        print(f"Cleared {removed} __pycache__ directories.")

    print("Server stopped. Restart helper will start the new server.")


# ── Deprecated modes ──────────────────────────────────────


def cmd_gateway(args: argparse.Namespace) -> None:
    print("Error: 'gateway' mode has been deprecated. Use 'animaworks start' instead.")
    sys.exit(1)


def cmd_worker(args: argparse.Namespace) -> None:
    print("Error: 'worker' mode has been deprecated. Use 'animaworks start' instead.")
    sys.exit(1)
