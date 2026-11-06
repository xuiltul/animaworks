from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Crash-safe atomic file writing helpers shared across the application."""

import copy
import fcntl
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.platform.locks import locked_path

logger = logging.getLogger(__name__)


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    mode: int | None = None,
    fsync_dir: bool = False,
) -> None:
    """Atomically write bytes using a unique sibling temporary file."""
    _atomic_write_data(path, data, mode=mode, fsync_dir=fsync_dir)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
    fsync_dir: bool = False,
) -> None:
    """Atomically write text using a unique sibling temporary file."""
    _atomic_write_data(path, text, encoding=encoding, mode=mode, fsync_dir=fsync_dir)


def atomic_write_json(
    path: Path,
    obj: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    trailing_newline: bool = True,
    sort_keys: bool = False,
    mode: int | None = None,
    fsync_dir: bool = False,
) -> None:
    """Atomically serialize JSON with explicitly configurable formatting."""
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    if trailing_newline:
        text += "\n"
    atomic_write_text(path, text, mode=mode, fsync_dir=fsync_dir)


def append_jsonl_locked(
    path: Path,
    record: Any,
    *,
    raw_line: bool = False,
    fsync: bool = False,
    lock_held: bool = False,
) -> None:
    """Append one JSONL record while holding the adjacent persistent lock.

    Writers serialize the complete encoded record before opening the target,
    then hold a ``flock`` on ``<name>.lock`` for the full append so concurrent
    processes cannot interleave large lines. ``lock_held`` lets a caller append
    within a larger transaction that already holds that same lock. ``raw_line``
    accepts an existing single-line JSON representation when an operation must
    preserve its bytes. Durability is opt-in to preserve each caller's existing
    fsync policy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_line:
        if not isinstance(record, str):
            raise TypeError("raw_line requires a pre-serialized string")
        line = record.rstrip("\r\n")
        if "\n" in line or "\r" in line:
            raise ValueError("raw JSONL records must contain exactly one line")
        line += "\n"
    else:
        line = json.dumps(record, ensure_ascii=False) + "\n"
    lock_path = path.with_name(f"{path.name}.lock")
    if lock_held:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            if fsync:
                stream.flush()
                os.fsync(stream.fileno())
    else:
        with locked_path(lock_path, exclusive=True, thread_lock=True), path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            if fsync:
                stream.flush()
                os.fsync(stream.fileno())


def update_json(
    path: Path,
    fn: Callable[[dict[str, Any]], dict[str, Any] | None],
    *,
    mode: int | None = None,
    always_write: bool = False,
    write_if_missing: bool = True,
    after_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Lock, read, modify, and atomically write an object-valued JSON file.

    The sibling ``.lock`` file is intentionally persistent: unlinking a lock
    file can let concurrent processes lock different inodes for the same path.
    Missing files are treated as empty objects; malformed or non-object JSON
    is rejected without modifying the file. Unchanged objects are left alone
    unless ``always_write`` is requested. ``after_update``, when supplied, runs
    under the lock after a successful update.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                with path.open(encoding="utf-8") as data_file:
                    current = json.load(data_file)
                if not isinstance(current, dict):
                    raise ValueError(f"Expected a JSON object in {path}")
            else:
                current = {}

            original = copy.deepcopy(current)
            updated = fn(current)
            if updated is None:
                updated = current
            if not isinstance(updated, dict):
                raise TypeError("JSON update callback must return a dict or None")
            if always_write or updated != original or (write_if_missing and not path.exists()):
                atomic_write_json(path, updated, mode=mode)
            if after_update is not None:
                after_update(updated)
            return updated
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_data(
    path: Path,
    data: bytes | str,
    *,
    encoding: str | None = None,
    mode: int | None = None,
    fsync_dir: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        if isinstance(data, bytes):
            stream = os.fdopen(fd, "wb")
        else:
            stream = os.fdopen(fd, "w", encoding=encoding)
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to unlink atomic-write temp file %s", temp_path, exc_info=True)
        raise

    if fsync_dir:
        _fsync_directory(path.parent)


def cleanup_tmp_files(directory: Path, prefix: str = ".") -> int:
    """Remove stale ``.tmp`` files from *directory* and return the count."""
    removed = 0
    if not directory.exists():
        return removed
    for tmp in directory.glob(f"{prefix}*.tmp"):
        try:
            tmp.unlink()
            removed += 1
        except OSError:
            logger.debug("Failed to remove stale tmp file %s", tmp, exc_info=True)
    return removed


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory after replacing its entry."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        logger.debug("Failed to open directory for fsync: %s", directory, exc_info=True)
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        logger.debug("Failed to fsync directory %s", directory, exc_info=True)
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            logger.debug("Failed to close directory after fsync: %s", directory, exc_info=True)
