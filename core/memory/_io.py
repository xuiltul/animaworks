from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Crash-safe I/O utilities for the memory subsystem.

Provides atomic file writes (temp + rename) and fsync-backed appends
to protect against data loss on process crash or power failure.
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("animaworks.memory._io")


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to file atomically using temp + rename pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=f".{path.name}.",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.debug("Failed to unlink temp file %s", tmp_path, exc_info=True)
        raise


def cleanup_tmp_files(directory: Path, prefix: str = ".") -> int:
    """Remove stale .tmp files from directory. Returns count removed."""
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
