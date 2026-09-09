from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Lease sidecars for claimed pending-task descriptors.

Schema:
  * v1 — ``pid/anima/leased_at/task_id`` (legacy runner in-process claims)
  * v2 — adds ``schema_version/job_id/task_pid/pgid/root_epoch/attempt/process_start_time``
    for task-runner isolation (see ``docs/specs/20260807_state-mutation-inventory.md`` §5)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

LeaseLiveness = Literal["live", "dead", "unknown"]

_RUNNER_CMDLINE_MARKERS = ("core.supervisor.runner", "core.supervisor.task_runner")


def processing_lease_path(descriptor_path: Path) -> Path:
    """Return the lease sidecar path for a processing descriptor."""
    return descriptor_path.with_name(f"{descriptor_path.name}.lease")


def _centisecond(value: float) -> int:
    """Round a Unix timestamp to centisecond precision for PID-reuse fences."""
    return int(round(float(value) * 100.0))


def _process_create_time(pid: int) -> float | None:
    """Return ``psutil.Process(pid).create_time()`` or ``None`` when unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except Exception:
        # psutil.Error hierarchy plus OS-level failures — treat as unavailable.
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via temp+fsync+replace (+ directory fsync when possible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, ensure_ascii=False)
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                logger.debug("fsync failed for lease temp file %s", tmp_path, exc_info=True)
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            logger.debug("directory fsync failed for %s", path.parent, exc_info=True)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_processing_lease(
    descriptor_path: Path,
    *,
    anima: str,
    task_id: str,
    pid: int | None = None,
    leased_at: float | None = None,
    job_id: str | None = None,
    task_pid: int | None = None,
    pgid: int | None = None,
    root_epoch: str | None = None,
    attempt: int | None = None,
    process_start_time: float | None = None,
) -> Path:
    """Write and return the lease sidecar for a claimed descriptor.

    When all v2 identity fields (``job_id``, ``task_pid``, ``pgid``,
    ``root_epoch``, ``attempt``) are provided the lease is written as
    schema version 2.  Otherwise a v1 payload is written for backward
    compatibility with in-process claims.
    """
    lease_path = processing_lease_path(descriptor_path)
    root_pid = os.getpid() if pid is None else pid
    payload: dict[str, Any] = {
        "pid": root_pid,
        "anima": anima,
        "leased_at": time.time() if leased_at is None else leased_at,
        "task_id": task_id,
    }
    v2_ready = (
        job_id is not None
        and task_pid is not None
        and pgid is not None
        and root_epoch is not None
        and attempt is not None
    )
    if v2_ready:
        start_time = process_start_time
        if start_time is None and task_pid is not None:
            start_time = _process_create_time(int(task_pid))
        if start_time is None:
            start_time = time.time()
        payload.update(
            {
                "schema_version": 2,
                "job_id": job_id,
                "task_pid": int(task_pid),
                "pgid": int(pgid),
                "root_epoch": root_epoch,
                "attempt": int(attempt),
                "process_start_time": float(start_time),
            }
        )
        _atomic_write_json(lease_path, payload)
    else:
        lease_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return lease_path


def _read_proc_cmdline(pid: int) -> str:
    """Read a process command line on Linux and non-/proc platforms.

    ``/proc`` is the preferred source because it does not add a runtime
    dependency to the lease fast path.  macOS has no procfs, however, and
    treating every existing PID there as live defeats the PID-reuse fence and
    can leave an interrupted TaskExec descriptor stuck forever.  psutil is an
    existing optional dependency used by the stronger process-start-time
    fence, so use it as the portable fallback.
    """
    proc_path = Path("/proc") / str(pid) / "cmdline"
    if proc_path.exists():
        raw = proc_path.read_bytes()
        return raw.replace(b"\0", b" ").decode(errors="replace")

    try:
        import psutil
    except ImportError as exc:
        raise OSError("process command line is unavailable") from exc
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except Exception as exc:
        raise OSError("process command line is unavailable") from exc


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _validate_common_fields(payload: dict[str, Any], expected_anima: str | None) -> bool:
    pid = payload.get("pid")
    anima = payload.get("anima")
    leased_at = payload.get("leased_at")
    task_id = payload.get("task_id")
    if not _is_positive_int(pid):
        return False
    if not isinstance(anima, str) or not anima:
        return False
    if not _is_finite_number(leased_at):
        return False
    if not isinstance(task_id, str) or not task_id:
        return False
    return expected_anima is None or anima == expected_anima


def _pid_exists(pid: int) -> bool | None:
    """Return True/False when known, ``None`` when existence is uncertain.

    Overflow/ValueError (e.g. absurdly large PIDs) are treated as dead so
    malformed leases do not block recovery.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, ValueError):
        return False
    except OSError:
        return None


def _cmdline_matches_v1(cmdline: str, anima: str) -> bool:
    return "core.supervisor.runner" in cmdline and anima in cmdline


def _cmdline_matches_v2(cmdline: str, anima: str, job_id: str) -> bool:
    has_marker = any(marker in cmdline for marker in _RUNNER_CMDLINE_MARKERS)
    return has_marker and anima in cmdline and job_id in cmdline


def classify_processing_lease(
    descriptor_path: Path,
    *,
    expected_anima: str | None = None,
    expected_root_epoch: str | None = None,
) -> LeaseLiveness:
    """Classify a lease as ``live``, ``dead``, or ``unknown``.

    ``unknown`` means the process may still be running but platform signals
    were inconclusive — callers must not kill or recover those descriptors.
    """
    lease_path = processing_lease_path(descriptor_path)
    try:
        payload = json.loads(lease_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return "dead"
    if not isinstance(payload, dict):
        return "dead"
    if not _validate_common_fields(payload, expected_anima):
        return "dead"

    schema_version = payload.get("schema_version")
    if schema_version == 2:
        return _classify_v2(payload, expected_root_epoch=expected_root_epoch)
    if schema_version is not None and schema_version != 1:
        return "dead"
    return _classify_v1(payload)


def _classify_v1(payload: dict[str, Any]) -> LeaseLiveness:
    pid = int(payload["pid"])
    anima = str(payload["anima"])
    exists = _pid_exists(pid)
    if exists is False:
        return "dead"
    if exists is None:
        return "unknown"

    try:
        cmdline = _read_proc_cmdline(pid)
    except OSError:
        return "unknown"
    return "live" if _cmdline_matches_v1(cmdline, anima) else "dead"


def _classify_v2(
    payload: dict[str, Any],
    *,
    expected_root_epoch: str | None,
) -> LeaseLiveness:
    task_pid = payload.get("task_pid")
    pgid = payload.get("pgid")
    root_epoch = payload.get("root_epoch")
    attempt = payload.get("attempt")
    job_id = payload.get("job_id")
    process_start_time = payload.get("process_start_time")
    anima = str(payload["anima"])

    if not _is_positive_int(task_pid):
        return "dead"
    if not _is_positive_int(pgid):
        return "dead"
    if not isinstance(root_epoch, str) or not root_epoch:
        return "dead"
    if not isinstance(job_id, str) or not job_id:
        return "dead"
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        return "dead"
    if not _is_finite_number(process_start_time):
        return "dead"

    if expected_root_epoch is not None and root_epoch != expected_root_epoch:
        # Stale generation: still verify the recorded child identity before
        # callers decide whether to reap that specific process group.
        pass

    exists = _pid_exists(int(task_pid))
    if exists is False:
        return "dead"
    if exists is None:
        return "unknown"

    current_start = _process_create_time(int(task_pid))
    if current_start is not None:
        if _centisecond(current_start) != _centisecond(float(process_start_time)):
            return "dead"
    # If create_time is unavailable, fall through to cmdline checks.

    try:
        cmdline = _read_proc_cmdline(int(task_pid))
    except OSError:
        return "unknown"
    return "live" if _cmdline_matches_v2(cmdline, anima, job_id) else "dead"


def is_processing_lease_live(
    descriptor_path: Path,
    *,
    expected_anima: str | None = None,
    expected_root_epoch: str | None = None,
) -> bool:
    """Return whether a descriptor's lease belongs to a live executor.

    Missing, malformed, dead, and PID-reused leases return ``False``.  When
    the platform cannot conclusively classify the process (``unknown``), the
    result is conservatively ``True`` so callers do not reclaim live work.
    """
    status = classify_processing_lease(
        descriptor_path,
        expected_anima=expected_anima,
        expected_root_epoch=expected_root_epoch,
    )
    return status in {"live", "unknown"}


def read_processing_lease(descriptor_path: Path) -> dict[str, Any] | None:
    """Return the parsed lease payload, or ``None`` when missing/malformed."""
    lease_path = processing_lease_path(descriptor_path)
    try:
        payload = json.loads(lease_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


__all__ = [
    "LeaseLiveness",
    "classify_processing_lease",
    "is_processing_lease_live",
    "processing_lease_path",
    "read_processing_lease",
    "write_processing_lease",
]
