from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform advisory file locking helpers."""

import contextlib
import os
from collections.abc import Iterator
from typing import IO

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def _prepare_windows_lock(file_obj: IO[str]) -> None:
    if not file_obj.writable():
        file_obj.seek(0)
        return
    current = file_obj.tell()
    file_obj.seek(0, os.SEEK_END)
    if file_obj.tell() == 0:
        file_obj.write("\0")
        file_obj.flush()
    file_obj.seek(current)


def acquire_file_lock(
    file_obj: IO[str],
    *,
    exclusive: bool,
    blocking: bool = True,
) -> None:
    """Acquire an advisory file lock on ``file_obj``."""
    if os.name == "nt":
        _prepare_windows_lock(file_obj)
        file_obj.seek(0)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(file_obj.fileno(), mode, 1)
        return

    flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        flags |= fcntl.LOCK_NB
    fcntl.flock(file_obj, flags)


def release_file_lock(file_obj: IO[str]) -> None:
    """Release a previously acquired advisory file lock."""
    if os.name == "nt":
        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(file_obj, fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(
    file_obj: IO[str],
    *,
    exclusive: bool,
    blocking: bool = True,
) -> Iterator[IO[str]]:
    """Context manager wrapper around :func:`acquire_file_lock`."""
    acquire_file_lock(file_obj, exclusive=exclusive, blocking=blocking)
    try:
        yield file_obj
    finally:
        release_file_lock(file_obj)
