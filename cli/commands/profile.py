# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from core.i18n import t

logger = logging.getLogger(__name__)

# ── Storage constants ─────────────────────────────────────

_PROFILES_FILE = Path.home() / ".animaworks-profiles.json"
_DEFAULT_PORT_START = 18500
_DEFAULT_PORT_STEP = 10

# ── Storage helpers ───────────────────────────────────────


def _load_profiles() -> dict[str, dict]:
    """Load profiles from JSON file. Return empty dict on corruption."""
    if not _PROFILES_FILE.exists():
        return {}
    try:
        data = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
        profiles = data.get("profiles", {})
        if isinstance(profiles, dict):
            return profiles
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load profiles: %s", exc)
        print(t("cli.profile_corrupt_file"), file=sys.stderr)
        return {}


def _save_profiles(profiles: dict[str, dict]) -> None:
    """Save profiles atomically via tmp file + rename."""
    data = {"version": 1, "profiles": profiles}
    tmp = _PROFILES_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(_PROFILES_FILE)


def _next_available_port(profiles: dict[str, dict]) -> int:
    """Find next unused port starting at _DEFAULT_PORT_START, step _DEFAULT_PORT_STEP."""
    used = {p.get("port") for p in profiles.values() if isinstance(p.get("port"), int)}
    port = _DEFAULT_PORT_START
    while port in used:
        port += _DEFAULT_PORT_STEP
    return port


# ── PID/status helpers ─────────────────────────────────────


def _read_pid_for(data_dir: Path) -> int | None:
    """Read server.pid from a specific data dir."""
    pid_file = data_dir / "server.pid"
    if not pid_file.exists():
        return None
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
        return int(text)
    except (ValueError, OSError):
        return None


def _is_process_alive(pid: int) -> bool:
    """Check if process exists (os.kill(pid, 0))."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _profile_status(profile: dict) -> str:
    """Return i18n status string for a profile."""
    data_dir = Path(profile.get("data_dir", "")).expanduser().resolve()
    pid = _read_pid_for(data_dir)
    if pid is None:
        return t("cli.profile_stopped")
    if _is_process_alive(pid):
        return t("cli.profile_running", pid=pid)
    return t("cli.profile_stopped_stale")


def _is_profile_running(profile: dict) -> bool:
    """Check if profile's server is running."""
    data_dir = Path(profile.get("data_dir", "")).expanduser().resolve()
    pid = _read_pid_for(data_dir)
    return pid is not None and _is_process_alive(pid)


# ── Commands ───────────────────────────────────────────────


def cmd_profile_list(args: argparse.Namespace) -> None:
    """List all profiles with status table."""
    profiles = _load_profiles()
    if not profiles:
        print(t("cli.profile_no_profiles"))
        print(t("cli.profile_add_hint"))
        return
    # Table: name | data_dir | port | status
    max_name = max(len(n) for n in profiles) if profiles else 4
    max_dir = 40
    for name in sorted(profiles.keys()):
        p = profiles[name]
        data_dir = str(p.get("data_dir", ""))
        if len(data_dir) > max_dir:
            data_dir = "..." + data_dir[-(max_dir - 3) :]
        port = p.get("port", "—")
        status = _profile_status(p)
        print(f"{name:<{max_name}}  {data_dir:<{max_dir}}  {port}  {status}")


def cmd_profile_add(args: argparse.Namespace) -> None:
    """Register new profile."""
    profiles = _load_profiles()
    name = args.name
    if name in profiles:
        print(t("cli.profile_already_exists", name=name), file=sys.stderr)
        sys.exit(1)
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = str(Path.home() / ".animaworks" / name)
    else:
        data_dir = str(Path(data_dir).expanduser().resolve())
    port = args.port
    if port is None:
        port = _next_available_port(profiles)
    profiles[name] = {"data_dir": data_dir, "port": port}
    _save_profiles(profiles)
    print(t("cli.profile_registered", name=name))
    data_path = Path(data_dir)
    if not data_path.exists():
        print(t("cli.profile_init_hint"))
        print(f"  animaworks init --data-dir {data_dir}")


def cmd_profile_remove(args: argparse.Namespace) -> None:
    """Remove profile registration. Reject if running. Data preserved."""
    profiles = _load_profiles()
    name = args.name
    if name not in profiles:
        print(t("cli.profile_not_found", name=name), file=sys.stderr)
        sys.exit(1)
    profile = profiles[name]
    if _is_profile_running(profile):
        print(t("cli.profile_stop_running", name=name), file=sys.stderr)
        sys.exit(1)
    data_dir = profile["data_dir"]
    del profiles[name]
    _save_profiles(profiles)
    print(t("cli.profile_removed", name=name))
    print(t("cli.profile_data_preserved", path=data_dir))


def cmd_profile_start(args: argparse.Namespace) -> None:
    """Start server for profile."""
    profiles = _load_profiles()
    name = args.name
    if name not in profiles:
        print(t("cli.profile_not_found", name=name), file=sys.stderr)
        sys.exit(1)
    profile = profiles[name]
    if _is_profile_running(profile):
        pid = _read_pid_for(Path(profile["data_dir"]).expanduser().resolve())
        print(t("cli.profile_already_running", name=name, pid=pid), file=sys.stderr)
        sys.exit(1)
    data_dir = str(Path(profile["data_dir"]).expanduser().resolve())
    port = profile.get("port", _DEFAULT_PORT_START)
    os.environ["ANIMAWORKS_DATA_DIR"] = data_dir
    print(t("cli.profile_starting", name=name))
    from cli.commands.server import cmd_start

    start_args = argparse.Namespace(
        host=args.host or "0.0.0.0",
        port=port,
        foreground=False,
        force=False,
    )
    cmd_start(start_args)
    print(t("cli.profile_started", name=name))


def cmd_profile_stop(args: argparse.Namespace) -> None:
    """Stop server for profile."""
    profiles = _load_profiles()
    name = args.name
    if name not in profiles:
        print(t("cli.profile_not_found", name=name), file=sys.stderr)
        sys.exit(1)
    profile = profiles[name]
    if not _is_profile_running(profile):
        print(t("cli.profile_not_running", name=name), file=sys.stderr)
        sys.exit(1)
    data_dir = str(Path(profile["data_dir"]).expanduser().resolve())
    pid = _read_pid_for(Path(data_dir))
    os.environ["ANIMAWORKS_DATA_DIR"] = data_dir
    print(t("cli.profile_stopping", name=name, pid=pid or "?"))
    from cli.commands.server import cmd_stop

    stop_args = argparse.Namespace(force=getattr(args, "force", False))
    cmd_stop(stop_args)
    print(t("cli.profile_stopped_ok", name=name))


def cmd_profile_start_all(args: argparse.Namespace) -> None:
    """Start all profiles that are not already running."""
    profiles = _load_profiles()
    if not profiles:
        print(t("cli.profile_no_profiles"))
        print(t("cli.profile_add_hint"))
        return
    for name in sorted(profiles.keys()):
        profile = profiles[name]
        if _is_profile_running(profile):
            continue
        data_dir = str(Path(profile["data_dir"]).expanduser().resolve())
        port = profile.get("port", _DEFAULT_PORT_START)
        os.environ["ANIMAWORKS_DATA_DIR"] = data_dir
        print(t("cli.profile_starting", name=name))
        from cli.commands.server import cmd_start

        start_args = argparse.Namespace(
            host=getattr(args, "host", None) or "0.0.0.0",
            port=port,
            foreground=False,
            force=False,
        )
        cmd_start(start_args)
        print(t("cli.profile_started", name=name))


def cmd_profile_stop_all(args: argparse.Namespace) -> None:
    """Stop all running profiles."""
    profiles = _load_profiles()
    if not profiles:
        print(t("cli.profile_no_profiles"))
        print(t("cli.profile_add_hint"))
        return
    for name in sorted(profiles.keys()):
        profile = profiles[name]
        if not _is_profile_running(profile):
            continue
        data_dir = str(Path(profile["data_dir"]).expanduser().resolve())
        pid = _read_pid_for(Path(data_dir))
        os.environ["ANIMAWORKS_DATA_DIR"] = data_dir
        print(t("cli.profile_stopping", name=name, pid=pid or "?"))
        from cli.commands.server import cmd_stop

        stop_args = argparse.Namespace(force=getattr(args, "force", False))
        try:
            cmd_stop(stop_args)
        except SystemExit:
            pass
        print(t("cli.profile_stopped_ok", name=name))


# ── Parser registration ────────────────────────────────────


def register_profile_command(sub: argparse._SubParsersAction) -> None:
    """Register the profile subcommand group with all subcommands."""
    p_profile = sub.add_parser("profile", help=t("cli.profile_help"))
    profile_sub = p_profile.add_subparsers(dest="profile_command")

    # profile list
    p_list = profile_sub.add_parser("list", help="List all profiles with status")
    p_list.set_defaults(func=cmd_profile_list)

    # profile add
    p_add = profile_sub.add_parser("add", help="Register a new profile")
    p_add.add_argument("name", help="Profile name")
    p_add.add_argument(
        "--data-dir",
        dest="data_dir",
        default=None,
        help="Data directory (default: ~/.animaworks/<name>)",
    )
    p_add.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port (default: auto-assign from 18500, step 10)",
    )
    p_add.set_defaults(func=cmd_profile_add)

    # profile remove
    p_remove = profile_sub.add_parser("remove", help="Remove profile registration")
    p_remove.add_argument("name", help="Profile name")
    p_remove.set_defaults(func=cmd_profile_remove)

    # profile start
    p_start = profile_sub.add_parser("start", help="Start server for profile")
    p_start.add_argument("name", help="Profile name")
    p_start.add_argument("--host", default=None, help="Host (default: 0.0.0.0)")
    p_start.set_defaults(func=cmd_profile_start)

    # profile stop
    p_stop = profile_sub.add_parser("stop", help="Stop server for profile")
    p_stop.add_argument("name", help="Profile name")
    p_stop.add_argument(
        "--force",
        action="store_true",
        help="Force stop (SIGKILL after timeout)",
    )
    p_stop.set_defaults(func=cmd_profile_stop)

    # profile start-all
    p_start_all = profile_sub.add_parser("start-all", help="Start all profiles")
    p_start_all.add_argument("--host", default=None, help="Host (default: 0.0.0.0)")
    p_start_all.set_defaults(func=cmd_profile_start_all)

    # profile stop-all
    p_stop_all = profile_sub.add_parser("stop-all", help="Stop all running profiles")
    p_stop_all.add_argument(
        "--force",
        action="store_true",
        help="Force stop (SIGKILL after timeout)",
    )
    p_stop_all.set_defaults(func=cmd_profile_stop_all)
