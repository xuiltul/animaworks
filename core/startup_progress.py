from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe startup progress registry."""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Literal

StartupPhase = Literal[
    "starting",
    "preflight",
    "repairing",
    "indexing",
    "spawning_animas",
    "ready",
    "failed",
]

_IN_PROGRESS_PHASES = frozenset({"starting", "preflight", "repairing", "indexing", "spawning_animas"})
_TERMINAL_PHASES = frozenset({"ready", "failed"})
_VALID_PHASES = _IN_PROGRESS_PHASES | _TERMINAL_PHASES


@dataclass
class _StartupProgressState:
    phase: StartupPhase = "ready"
    detail: str = ""
    done_count: int | None = None
    total_count: int | None = None
    started_at: float = 0.0
    error: str | None = None
    tracking: bool = False
    cancel_requested: bool = False


_state = _StartupProgressState(started_at=time.time())
_lock = threading.RLock()


def begin_startup(detail: str = "") -> None:
    """Start a new tracked startup window."""
    now = time.time()
    with _lock:
        _state.phase = "starting"
        _state.detail = detail
        _state.done_count = None
        _state.total_count = None
        _state.started_at = now
        _state.error = None
        _state.tracking = True
        _state.cancel_requested = False


def set_phase(
    phase: StartupPhase,
    *,
    detail: str | None = None,
    done_count: int | None = None,
    total_count: int | None = None,
    error: str | None = None,
    reset_counts: bool = False,
) -> None:
    """Set the current startup phase.

    Non-terminal phase updates are ignored after startup has reached
    ``ready`` or ``failed`` so runtime repairs cannot accidentally close
    readiness after the server is live.
    """
    if phase not in _VALID_PHASES:
        raise ValueError(f"invalid startup phase: {phase}")

    if phase == "starting":
        begin_startup(detail or "")
        if done_count is not None or total_count is not None:
            update_progress(done_count=done_count, total_count=total_count)
        return

    with _lock:
        if phase in _IN_PROGRESS_PHASES and not _state.tracking:
            return

        _state.phase = phase
        if detail is not None:
            _state.detail = detail
        if reset_counts:
            _state.done_count = None
            _state.total_count = None
        if done_count is not None:
            _state.done_count = max(0, int(done_count))
        if total_count is not None:
            _state.total_count = max(0, int(total_count))
        if error is not None:
            _state.error = str(error)
        elif phase != "failed":
            _state.error = None
        if phase in _TERMINAL_PHASES:
            _state.tracking = False
            _state.cancel_requested = False


def update_progress(
    *,
    detail: str | None = None,
    done_count: int | None = None,
    total_count: int | None = None,
    done_increment: int = 0,
) -> None:
    """Update counts/details for the active startup window."""
    with _lock:
        if not _state.tracking or _state.phase in _TERMINAL_PHASES:
            return
        if detail is not None:
            _state.detail = detail
        if total_count is not None:
            _state.total_count = max(0, int(total_count))
        if done_count is not None:
            _state.done_count = max(0, int(done_count))
        elif done_increment:
            _state.done_count = max(0, int(_state.done_count or 0) + int(done_increment))


def request_cancel() -> None:
    """Ask startup loops to stop at their next cooperative checkpoint."""
    with _lock:
        if _state.tracking and _state.phase in _IN_PROGRESS_PHASES:
            _state.cancel_requested = True


def cancel_requested() -> bool:
    with _lock:
        return bool(_state.tracking and _state.cancel_requested)


def raise_if_cancelled() -> None:
    if cancel_requested():
        raise asyncio.CancelledError("startup initialization cancelled")


def is_active() -> bool:
    """Return True while a startup window is accepting progress updates."""
    with _lock:
        return bool(_state.tracking and _state.phase in _IN_PROGRESS_PHASES)


def snapshot() -> dict[str, object]:
    """Return a JSON-serializable startup progress snapshot."""
    with _lock:
        elapsed = max(0.0, time.time() - _state.started_at)
        if _state.phase == "ready":
            status = "ready"
        elif _state.phase == "failed":
            status = "failed"
        else:
            status = "starting"
        return {
            "status": status,
            "phase": _state.phase,
            "detail": _state.detail,
            "done_count": _state.done_count,
            "total_count": _state.total_count,
            "started_at": _state.started_at,
            "elapsed_seconds": elapsed,
            "error": _state.error,
        }


def _reset_for_testing(phase: StartupPhase = "ready") -> None:
    if phase == "starting":
        begin_startup()
        return
    with _lock:
        _state.phase = phase
        _state.detail = ""
        _state.done_count = None
        _state.total_count = None
        _state.started_at = time.time()
        _state.error = None
        _state.tracking = phase in _IN_PROGRESS_PHASES
        _state.cancel_requested = False
