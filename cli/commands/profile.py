# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# Multi-instance profile management.
# Allows running multiple AnimaWorks instances simultaneously,
# each with its own data directory and port.

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

_PROFILES_FILE = Path.home() / ".animaworks-profiles.json"
_DEFAULT_PORT_START = 18500
_DEFAULT_PORT_STEP = 10


# ── Profile storage ──────────────────────────────────────


def _load_profiles() -> dict[str, dict]:
    """Load profiles from the global profiles file."""
    if not _PROFILES_FILE.exists():
        return {}
    try:
        data = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
        return data.get("profiles", {})
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: Failed to read {_PROFILES_FILE}: {exc}", file=sys.stderr)
        return {}


def _save_profiles(profiles: dict[str, dict]) -> None:
    """Save profiles to the global profiles file."""
    data = {"version": 1, "profiles": profiles}
    tmp = _PROFILES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_PROFILES_FILE)


def _next_available_port(profiles: dict[str, dict]) -> int:
    """Find the next available port based on existing profiles."""
    used_ports = {p.get("port", 0) for p in profiles.values()}
    port = _DEFAULT_PORT_START
    while port in used_ports:
        port += _DEFAULT_PORT_STEP
    return port


# ── PID helpers (per data-dir) ───────────────────────────


def _read_pid_for(data_dir: Path) -> int | None:
    """Read PID from a specific data directory's server.pid."""
    pid_file = data_dir / "server.pid"
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _is_process_alive(pid: int) -> bool:
    """Check whether a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _profile_status(profile: dict) -> str:
    """Return human-readable status for a profile."""
    data_dir = Path(profile["data_dir"]).expanduser()
    pid = _read_pid_for(data_dir)
    if pid is not None and _is_process_alive(pid):
        return f"running (pid={pid})"
    if pid is not None:
        return "stopped (stale pid)"
    return "stopped"


def _is_profile_running(profile: dict) -> bool:
    """Check if a profile's server is currently running."""
    data_dir = Path(profile["data_dir"]).expanduser()
    pid = _read_pid_for(data_dir)
    return pid is not None and _is_process_alive(pid)


# ── Commands ─────────────────────────────────────────────


def cmd_profile_list(args: argparse.Namespace) -> None:
    """List all registered profiles with their status."""
    profiles = _load_profiles()
    if not profiles:
        print("No profiles registered.")
        print("Use 'animaworks profile add <name>' to create one.")
        return

    # Header
    print(f"{'NAME':<20} {'PORT':<8} {'STATUS':<22} {'DATA DIR'}")
    print("-" * 80)

    for name, prof in sorted(profiles.items()):
        status = _profile_status(prof)
        data_dir = prof.get("data_dir", "")
        port = prof.get("port", "?")
        print(f"{name:<20} {port:<8} {status:<22} {data_dir}")


def cmd_profile_add(args: argparse.Namespace) -> None:
    """Register a new profile."""
    profiles = _load_profiles()
    name = args.name

    if name in profiles:
        print(f"Error: Profile '{name}' already exists.")
        print(f"  Data dir: {profiles[name]['data_dir']}")
        print(f"  Port:     {profiles[name]['port']}")
        sys.exit(1)

    data_dir = args.data_dir or f"~/.animaworks-{name}"
    port = args.port or _next_available_port(profiles)

    profiles[name] = {
        "data_dir": data_dir,
        "port": port,
    }
    _save_profiles(profiles)

    resolved = Path(data_dir).expanduser()
    print(f"Profile '{name}' registered.")
    print(f"  Data dir: {data_dir} ({resolved})")
    print(f"  Port:     {port}")
    print(f"  Dashboard: http://localhost:{port}/")

    if not resolved.exists():
        print(f"\nData directory does not exist yet. Initialize with:")
        print(f"  animaworks --data-dir {data_dir} init")


def cmd_profile_remove(args: argparse.Namespace) -> None:
    """Remove a profile registration (does not delete data)."""
    profiles = _load_profiles()
    name = args.name

    if name not in profiles:
        print(f"Error: Profile '{name}' not found.")
        sys.exit(1)

    if _is_profile_running(profiles[name]):
        print(f"Error: Profile '{name}' is currently running. Stop it first:")
        print(f"  animaworks profile stop {name}")
        sys.exit(1)

    prof = profiles.pop(name)
    _save_profiles(profiles)
    print(f"Profile '{name}' removed (registration only).")
    print(f"Data directory preserved at: {prof['data_dir']}")


def cmd_profile_start(args: argparse.Namespace) -> None:
    """Start the server for a specific profile."""
    profiles = _load_profiles()
    name = args.name

    if name not in profiles:
        print(f"Error: Profile '{name}' not found.")
        print("Available profiles:")
        for n in sorted(profiles):
            print(f"  - {n}")
        sys.exit(1)

    prof = profiles[name]

    if _is_profile_running(prof):
        data_dir = Path(prof["data_dir"]).expanduser()
        pid = _read_pid_for(data_dir)
        print(f"Profile '{name}' is already running (pid={pid}).")
        return

    data_dir = prof["data_dir"]
    port = prof.get("port", _DEFAULT_PORT_START)
    host = args.host or "0.0.0.0"

    # Build the command — delegate to animaworks start
    cmd = [
        sys.executable, "-m", "cli",
        "--data-dir", str(Path(data_dir).expanduser()),
        "start",
        "--host", host,
        "--port", str(port),
    ]

    log_dir = Path(data_dir).expanduser() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "server-daemon.log"
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    import subprocess
    from pathlib import Path as _P

    project_root = _P(__file__).resolve().parent.parent.parent

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=project_root,
    )
    log_file.close()

    # Wait for startup
    import socket

    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    deadline = time.monotonic() + 10
    started = False

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"Error: Server for '{name}' exited immediately (exit code {proc.returncode}).")
            print(f"Check logs: {log_path}")
            sys.exit(1)
        try:
            with socket.create_connection((check_host, port), timeout=1):
                started = True
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)

    if not started:
        print(f"Warning: Server process started (pid={proc.pid}) but port {port} not yet listening.")
        print(f"Check logs: {log_path}")
        return

    print(f"Profile '{name}' started (pid={proc.pid}).")
    print(f"  Dashboard: http://localhost:{port}/")
    print(f"  Data:      {data_dir}")
    print(f"  Logs:      {log_path}")


def cmd_profile_stop(args: argparse.Namespace) -> None:
    """Stop the server for a specific profile."""
    profiles = _load_profiles()
    name = args.name

    if name not in profiles:
        print(f"Error: Profile '{name}' not found.")
        sys.exit(1)

    prof = profiles[name]
    data_dir = Path(prof["data_dir"]).expanduser()
    pid = _read_pid_for(data_dir)

    if pid is None or not _is_process_alive(pid):
        print(f"Profile '{name}' is not running.")
        # Clean stale PID file
        pid_file = data_dir / "server.pid"
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)
        return

    print(f"Stopping '{name}' (pid={pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"Profile '{name}' already stopped.")
        return
    except PermissionError:
        print(f"Error: Permission denied sending signal to pid={pid}.")
        sys.exit(1)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            print(f"Profile '{name}' stopped.")
            pid_file = data_dir / "server.pid"
            pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.2)

    print(f"Error: Profile '{name}' (pid={pid}) did not stop within 10s.")
    sys.exit(1)


def cmd_profile_start_all(args: argparse.Namespace) -> None:
    """Start all registered profiles."""
    profiles = _load_profiles()
    if not profiles:
        print("No profiles registered.")
        return

    for name in sorted(profiles):
        if _is_profile_running(profiles[name]):
            print(f"  {name}: already running")
            continue
        print(f"Starting {name}...")
        # Create a minimal args namespace for cmd_profile_start
        start_args = argparse.Namespace(name=name, host=None)
        cmd_profile_start(start_args)
        print()


def cmd_profile_stop_all(args: argparse.Namespace) -> None:
    """Stop all running profiles."""
    profiles = _load_profiles()
    if not profiles:
        print("No profiles registered.")
        return

    for name in sorted(profiles):
        if not _is_profile_running(profiles[name]):
            continue
        stop_args = argparse.Namespace(name=name)
        cmd_profile_stop(stop_args)


def cmd_profile_status(args: argparse.Namespace) -> None:
    """Show status of all profiles (alias for list)."""
    cmd_profile_list(args)


# ── Parser registration ──────────────────────────────────


def register_profile_command(sub: argparse._SubParsersAction) -> None:
    """Register the 'profile' subcommand group."""
    p_profile = sub.add_parser(
        "profile",
        help="Manage multiple AnimaWorks instances (multi-tenant)",
    )
    profile_sub = p_profile.add_subparsers(dest="profile_command")

    # profile list
    p_list = profile_sub.add_parser("list", help="List all profiles with status")
    p_list.set_defaults(func=cmd_profile_list)

    # profile add
    p_add = profile_sub.add_parser("add", help="Register a new profile")
    p_add.add_argument("name", help="Profile name (e.g. 'my-project')")
    p_add.add_argument(
        "--data-dir",
        default=None,
        help="Data directory (default: ~/.animaworks-<name>)",
    )
    p_add.add_argument(
        "--port", type=int, default=None,
        help="Server port (default: auto-assign)",
    )
    p_add.set_defaults(func=cmd_profile_add)

    # profile remove
    p_remove = profile_sub.add_parser(
        "remove", help="Remove a profile (keeps data)",
    )
    p_remove.add_argument("name", help="Profile name to remove")
    p_remove.set_defaults(func=cmd_profile_remove)

    # profile start
    p_start = profile_sub.add_parser("start", help="Start a profile's server")
    p_start.add_argument("name", help="Profile name")
    p_start.add_argument("--host", default=None, help="Bind host")
    p_start.set_defaults(func=cmd_profile_start)

    # profile stop
    p_stop = profile_sub.add_parser("stop", help="Stop a profile's server")
    p_stop.add_argument("name", help="Profile name")
    p_stop.set_defaults(func=cmd_profile_stop)

    # profile start-all
    p_start_all = profile_sub.add_parser(
        "start-all", help="Start all registered profiles",
    )
    p_start_all.set_defaults(func=cmd_profile_start_all)

    # profile stop-all
    p_stop_all = profile_sub.add_parser(
        "stop-all", help="Stop all running profiles",
    )
    p_stop_all.set_defaults(func=cmd_profile_stop_all)

    # profile status (alias for list)
    p_status = profile_sub.add_parser("status", help="Show status of all profiles")
    p_status.set_defaults(func=cmd_profile_status)

    # Default: show help when no subcommand given
    p_profile.set_defaults(func=lambda _: p_profile.print_help())
