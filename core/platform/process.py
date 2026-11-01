from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform process helpers used by supervisor and CLI layers."""

import os
import signal
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psutil

from core.platform.subprocess_entries import SubprocessEntry


def subprocess_session_kwargs() -> dict[str, Any]:
    """Return Popen kwargs for launching an isolated subprocess session."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def is_process_alive(pid: int) -> bool:
    """Return True when ``pid`` exists and is not a zombie."""
    if pid <= 0 or not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


_PYTHON_IMAGE_TAGS = ("python", "pypy")


def _running_as_root() -> bool:
    return os.name == "posix" and os.geteuid() == 0


def _process_image_name(proc: psutil.Process) -> str:
    """Return the lower-cased executable name of *proc*, or "" when unreadable."""
    for getter in (proc.exe, proc.name):
        try:
            value = getter()
        except psutil.Error:
            continue
        if value:
            return Path(value).name.lower()
    return ""


def is_unrelated_process(pid: int) -> bool:
    """Return True when ``pid`` is provably not a Python process of this user.

    A live PID alone does not prove that a PID file is current: PIDs are
    reused, so a ``server.pid`` left over from before a reboot can name an
    unrelated process (on macOS a system daemon took the old PID and every
    start was refused as "already running").  The server writes its own PID
    file, so its process is a Python interpreter owned by the same user.
    A process owned by another user (unless the caller is root) or running a
    non-Python executable is therefore something else.  The command line is
    not used: it is unreadable for other users' processes on macOS and varies
    with how the server was launched.  When the owner or the executable cannot
    be read this returns False, so callers keep treating the PID as a possible
    server.

    Args:
        pid: Process ID read from a PID file.

    Returns:
        True only when the process is known to be something else.
    """
    try:
        proc = psutil.Process(pid)
        if not _running_as_root() and proc.username() != psutil.Process().username():
            return True
    except psutil.Error:
        return False
    image = _process_image_name(proc)
    if not image:
        return False
    return not any(tag in image for tag in _PYTHON_IMAGE_TAGS)


def _terminate_psutil_process(proc: psutil.Process, *, force: bool) -> None:
    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
    except psutil.Error:
        return


def snapshot_descendants(pid: int) -> list[psutil.Process]:
    """Capture all descendants while their parent tree is still intact."""
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []


def order_deepest_first(procs: Iterable[psutil.Process]) -> list[psutil.Process]:
    """Return processes ordered from deepest descendant to shallowest."""

    def _depth(proc: psutil.Process) -> int:
        try:
            return len(proc.parents())
        except psutil.Error:
            return 0

    return sorted(procs, key=_depth, reverse=True)


def _send_signal(proc: psutil.Process, sig: int) -> None:
    """Send a signal, mapping the portable termination signals on Windows."""
    try:
        if os.name == "nt" and sig == signal.SIGTERM:
            proc.terminate()
        elif os.name == "nt" and sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.send_signal(sig)
    except psutil.Error:
        return


def signal_tree(
    pid: int | None,
    sig: int,
    *,
    pgid: int | None = None,
    descendants: list[psutil.Process] | None = None,
    include_root: bool = True,
) -> None:
    """Signal a process group/root and every snapshotted descendant."""
    captured = snapshot_descendants(pid) if descendants is None and pid is not None else (descendants or [])
    used_group = os.name != "nt" and pgid is not None and pgid > 0
    if used_group:
        try:
            os.killpg(pgid, sig)
        except OSError:
            pass
    elif include_root and pid is not None:
        try:
            _send_signal(psutil.Process(pid), sig)
        except psutil.Error:
            pass

    for child in captured:
        _send_signal(child, sig)


def kill_tree(
    pid: int,
    *,
    descendants: list[psutil.Process] | None = None,
    include_root: bool = True,
    deepest_first: bool = True,
) -> int:
    """Kill descendants and optionally the root, returning successful sends."""
    captured = snapshot_descendants(pid) if descendants is None else descendants
    ordered = order_deepest_first(captured) if deepest_first else list(captured)
    killed = 0
    for proc in ordered:
        try:
            proc.kill()
            killed += 1
        except psutil.Error:
            continue

    if include_root:
        try:
            psutil.Process(pid).kill()
            killed += 1
        except psutil.Error:
            pass
    return killed


def terminate_tree(
    pid: int | None,
    *,
    descendants: list[psutil.Process] | None = None,
    grace_sec: float,
    include_root: bool = True,
) -> list[psutil.Process]:
    """Terminate a tree gracefully, then kill survivors after the grace period."""
    captured = snapshot_descendants(pid) if descendants is None and pid is not None else (descendants or [])
    targets = list(captured)
    if include_root and pid is not None:
        try:
            targets.append(psutil.Process(pid))
        except psutil.Error:
            pass

    for proc in targets:
        try:
            proc.terminate()
        except psutil.Error:
            continue

    if not targets:
        return []
    try:
        _, alive = psutil.wait_procs(targets, timeout=grace_sec)
    except psutil.Error:
        alive = targets
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            continue
    return alive


def process_group_exists(pgid: int | None, fallback_alive: bool) -> bool:
    """Check whether a POSIX process group exists, using a caller fallback."""
    if os.name != "posix" or pgid is None or pgid <= 0:
        return fallback_alive
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return fallback_alive
    return True


_TASK_RUNNER_CMDLINE_MARKERS = (
    SubprocessEntry.TASK_RUNNER.value,
    "core.supervisor.task_runner",  # legacy name until 2026-11 (S3a)
)


def task_runner_subtree_pids(
    root: psutil.Process,
    job_pids: set[int],
    cmd_marker: str | tuple[str, ...] = _TASK_RUNNER_CMDLINE_MARKERS,
) -> set[int]:
    """Return task-runner roots and all descendants for orphan-cleanup exclusion."""
    excluded: set[int] = set()
    cmd_markers = (cmd_marker,) if isinstance(cmd_marker, str) else cmd_marker

    def _add_subtree(pid: int) -> None:
        if pid in excluded:
            return
        excluded.add(pid)
        try:
            excluded.update(proc.pid for proc in psutil.Process(pid).children(recursive=True))
        except (psutil.Error, AttributeError):
            pass

    for pid in job_pids:
        _add_subtree(pid)
    try:
        descendants = root.children(recursive=True)
    except (psutil.Error, AttributeError):
        return excluded
    for proc in descendants:
        try:
            if any(marker in token for marker in cmd_markers for token in proc.cmdline()):
                _add_subtree(proc.pid)
        except (psutil.Error, TypeError, AttributeError):
            continue
    return excluded


def terminate_pid(pid: int, *, force: bool = False, include_children: bool = False) -> None:
    """Terminate ``pid`` and optionally its descendant processes."""
    try:
        proc = psutil.Process(pid)
    except psutil.Error:
        return

    # Snapshot before killing the group: members in a separate session survive
    # killpg and become untraceable once their parent exits.
    descendants = snapshot_descendants(pid) if include_children else []
    sig = signal.SIGKILL if force else signal.SIGTERM
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            _terminate_psutil_process(proc, force=force)
        for child in descendants:
            _terminate_psutil_process(child, force=force)
        return

    # Preserve Windows' previous child-before-root shutdown ordering.
    for child in descendants:
        _terminate_psutil_process(child, force=force)
    _terminate_psutil_process(proc, force=force)


def terminate_subprocess(proc: subprocess.Popen[Any], *, force: bool = False, include_children: bool = True) -> None:
    """Terminate a ``subprocess.Popen`` instance."""
    terminate_pid(proc.pid, force=force, include_children=include_children)


def find_matching_pids(
    markers: Iterable[str],
    *,
    path_contains: str | None = None,
    exclude_pids: Iterable[int] = (),
    require_python: bool = True,
) -> list[int]:
    """Return running PIDs whose command line contains any marker."""
    marker_list = tuple(markers)
    excluded = set(exclude_pids)
    current_user = psutil.Process().username()
    matches: list[int] = []

    for proc in psutil.process_iter(["pid", "cmdline", "exe", "name", "username"]):
        try:
            pid = int(proc.info["pid"])
            if pid in excluded:
                continue
            if proc.info.get("username") != current_user:
                continue
            cmdline_parts = proc.info.get("cmdline") or []
            cmdline = " ".join(cmdline_parts)
            if not cmdline or not any(marker in cmdline for marker in marker_list):
                continue
            if path_contains and path_contains not in cmdline:
                continue
            if require_python:
                exe_name = Path(proc.info.get("exe") or proc.info.get("name") or "").name.lower()
                if "python" not in exe_name:
                    continue
            matches.append(pid)
        except (psutil.Error, TypeError, ValueError):
            continue
    return matches


def find_first_matching_pid(
    markers: Iterable[str],
    *,
    path_contains: str | None = None,
    exclude_pids: Iterable[int] = (),
    require_python: bool = True,
) -> int | None:
    """Return the first PID whose command line matches the given markers."""
    matches = find_matching_pids(
        markers,
        path_contains=path_contains,
        exclude_pids=exclude_pids,
        require_python=require_python,
    )
    return matches[0] if matches else None


def terminate_matching_processes(
    markers: Iterable[str],
    *,
    path_contains: str | None = None,
    exclude_pids: Iterable[int] = (),
    force: bool = False,
    include_children: bool = False,
    require_python: bool = True,
) -> int:
    """Terminate all matching processes and return the number targeted."""
    matches = find_matching_pids(
        markers,
        path_contains=path_contains,
        exclude_pids=exclude_pids,
        require_python=require_python,
    )
    for pid in matches:
        terminate_pid(pid, force=force, include_children=include_children)
    return len(matches)
