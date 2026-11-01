from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Platform-level access to the server PID file."""

import logging
from pathlib import Path

logger = logging.getLogger("animaworks")


def is_server_running(data_dir: Path | None = None) -> bool:
    """Return whether a valid server PID file points to a live server process.

    A PID that a different process reused (for example after a reboot) does
    not count; see :func:`core.platform.process.is_unrelated_process`.
    """
    pid = read_server_pid(data_dir)
    if pid is None:
        return False
    from core.platform.process import is_process_alive, is_unrelated_process

    return is_process_alive(pid) and not is_unrelated_process(pid)


def read_server_pid(data_dir: Path | None = None) -> int | None:
    """Read and validate the server PID file for *data_dir* or the active runtime.

    Returns ``None`` when the PID file is absent, unreadable, or malformed.
    """
    if data_dir is None:
        from core.paths import get_data_dir

        data_dir = get_data_dir()

    pid_file = data_dir / "server.pid"
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        logger.warning("Invalid PID file %s: %s", pid_file, exc)
        return None
