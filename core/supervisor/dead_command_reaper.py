"""Reaper for "dead" external-CLI commands whose results no one can consume.

Every anima's external CLI (codex / grok / deepseek) runs the OS shell inside
a ``codex-linux-sandbox``.  When the CLI is killed or crashes, its sandbox
and any runaway shell it spawned survive as orphans for hours; a ``grep -R``
over the runtime data tree (~/.animaworks activity_log is >1GB per anima)
saturates disk IO and hang-kills every TaskExec (2026-09-01: 26 orphan
sessions, longest ~3h, IO stall 41%).

This module runs as a periodic sweep (see ``server.github_gateway``) and
reaps two classes of dead work:

* orphan sandbox sessions --- sandbox whose parent is no longer the codex
  CLI (re-parented to user systemd / pid 1), running for >5 minutes;
* broad recursive searches --- ``grep -r`` / ``rg`` / ``find`` / ``du``
  sweeping the runtime data tree, running for >10 minutes, whose caller is
  gone or just doesn't know better (read-only, safe to kill) --- the caller
  agent retries with a narrower search on the single failed tool call.

Candidates are enumerated (*find_*) as a pure, side-effect-free list, and
*kill* happens in a separate function so it can be dry-run or unit tested.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from core.paths import get_data_dir
from core.tooling.codex_command_hook import _is_broad_data_root, _search_targets

logger = logging.getLogger("animaworks.dead_command_reaper")

SANDBOX_MARKER = "codex-linux-sandbox"
SYSTEMD_MARKER = "systemd --user"

# Age thresholds (seconds) before a candidate is considered dead.
ORPHAN_SANDBOX_MIN_AGE = 5 * 60
BROAD_SEARCH_MIN_AGE = 10 * 60

LogFn = Callable[[str], None]


def _parse_cmdline(proc: Any) -> list[str] | None:
    try:
        return list(proc.cmdline() or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
        return None


def _age_seconds(proc: Any) -> float | None:
    try:
        created = float(proc.create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
        return None
    if created <= 0:
        return None
    return max(0.0, time.time() - created)


def _parent_is_orphaning(proc: Any) -> bool:
    """True when the process was re-parented to user systemd (its CLI died).

    No parent at all (ppid 1) or a ``systemd --user`` parent both mean the
    original codex CLI is gone and the sandbox became an orphan.
    """
    try:
        parent = proc.parent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        parent = None
    if parent is None:
        return True
    par_cmd = _parse_cmdline(parent)
    if par_cmd is None:
        par_name = ""
        try:
            par_name = parent.name() or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            par_name = ""
        return "systemd" in par_name
    joined = " ".join(par_cmd)
    return SYSTEMD_MARKER in joined or joined.strip().endswith("systemd")


def _is_orphan_sandbox(proc: Any) -> bool:
    cmd = _parse_cmdline(proc)
    if not cmd:
        return False
    if not any(SANDBOX_MARKER in part for part in cmd):
        return False
    if not _parent_is_orphaning(proc):
        return False
    age = _age_seconds(proc)
    return age is not None and age > ORPHAN_SANDBOX_MIN_AGE


def _is_broad_search(proc: Any, data_dir: Path) -> bool:
    cmd = _parse_cmdline(proc)
    if not cmd:
        return False
    targets = _search_targets(cmd)
    if not targets:
        return False
    try:
        cwd = Path(str(proc.cwd()))
    except Exception:
        # Relative paths are undecidable without cwd; prefer a miss over a kill.
        return False
    for raw in targets:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = cwd / path
        try:
            if _is_broad_data_root(path, data_dir):
                return True
        except (OSError, ValueError):
            continue
    return False


def _describe(proc: Any, *, kind: str) -> dict[str, Any] | None:
    cmd = _parse_cmdline(proc)
    age = _age_seconds(proc)
    if cmd is None or age is None:
        return None
    try:
        pid = int(proc.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
        return None
    return {
        "pid": pid,
        "age": int(age),
        "kind": kind,
        "cmd": " ".join(cmd)[:120],
    }


def collect_orphan_sandboxes() -> list[dict[str, Any]]:
    """Return orphan sandbox session leaders (side-effect free)."""
    targets: list[dict[str, Any]] = []
    seen_sessions: set[int] = set()
    for proc in psutil.process_iter():
        try:
            if not _is_orphan_sandbox(proc):
                continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        try:
            pid = int(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            continue
        if pid in (os.getpid(), os.getppid()):
            continue
        try:
            sid = os.getsid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            sid = pid
        if sid in seen_sessions:
            continue
        seen_sessions.add(sid)
        t = _describe(proc, kind="orphan-sandbox")
        if t:
            targets.append(t)
    return targets


def collect_broad_searches() -> list[dict[str, Any]]:
    """Return long-running recursive searches over the data tree."""
    data_dir = get_data_dir()
    targets: list[dict[str, Any]] = []
    for proc in psutil.process_iter():
        try:
            if not _is_broad_search(proc, data_dir):
                continue
        except psutil.Error:
            continue
        try:
            pid = int(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            continue
        if pid in (os.getpid(), os.getppid()):
            continue
        age = _age_seconds(proc)
        if age is None or age <= BROAD_SEARCH_MIN_AGE:
            continue
        t = _describe(proc, kind="broad-search")
        if t:
            targets.append(t)
    return targets


def collect_dead_commands() -> list[dict[str, Any]]:
    """Return every candidate dead command (orphan sandbox + broad search)."""
    return collect_orphan_sandboxes() + collect_broad_searches()


def _target_pids(pid: int) -> list[int]:
    pids = [pid]
    try:
        children = psutil.Process(pid).children(recursive=True)
        pids.extend(p.pid for p in children)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        pass
    return pids


def kill_target(target: dict[str, Any]) -> bool:
    """SIGKILL one target (and its descendants); returns True if any pid died.

    The kill strategy depends on the target *kind*:

    * ``orphan-sandbox`` — the sandbox's whole session (process group) is
      dead work re-parented to systemd, so kill the session with ``killpg``
      plus any descendants to be sure the whole runaway tree stops.
    * ``broad-search`` — the caller (parent agent / task runner) is alive and
      just ran a too-broad read-only search.  It may share the caller's
      session (a non-codex runner that didn't call ``setsid``), so only the
      target pid and its own descendants are killed; a ``killpg`` here could
      take the healthy runner down with it.
    """
    pid = int(target["pid"])
    kind = target.get("kind")
    killed = False
    if kind == "orphan-sandbox":
        # Whole slippery session tree stop: the codex CLI is gone, so every
        # member of this session is orphaned dead work.
        try:
            sid = os.getsid(pid)
            os.killpg(sid, signal.SIGKILL)
            killed = True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for child_pid in _target_pids(pid):
        try:
            os.kill(child_pid, signal.SIGKILL)
            killed = True
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return killed


def kill_targets(targets: list[dict[str, Any]], log: LogFn | None = None) -> int:
    """Kill every target and log one fixed line per target."""
    log = log or (lambda _msg: None)
    reaped = 0
    for target in targets:
        if kill_target(target):
            reaped += 1
            log(
                f"reaped dead command: pid={target['pid']} age={target['age']} "
                f"kind={target['kind']} cmd={target['cmd']}"
            )
    return reaped


def reap_dead_commands(log: LogFn | None = None) -> int:
    """Enumerate and kill every reaper candidate; returns how many were killed."""
    targets = collect_dead_commands()
    if not targets:
        return 0
    return kill_targets(targets, log=log or logger.info)
