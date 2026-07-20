from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared path exclusions for long-term memory indexing and priming."""

import fnmatch
from collections.abc import Iterable
from pathlib import Path


def is_archive_path(file_path: Path, *, root: Path) -> bool:
    """Return whether *file_path* is below a directory named ``archive``.

    ``root`` is stripped first so a data directory whose ancestors happen to
    be named ``archive`` does not make every memory file look archived.
    """
    try:
        relative = file_path.relative_to(root)
    except ValueError:
        relative = file_path
    return "archive" in relative.parts[:-1]


def is_rag_excluded(
    file_path: Path,
    *,
    root: Path,
    ragignore_patterns: Iterable[str] = (),
) -> bool:
    """Apply built-in archive and configured ``.ragignore`` exclusions."""
    if is_archive_path(file_path, root=root):
        return True
    name = file_path.name
    path_text = file_path.as_posix()
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(path_text, pattern) for pattern in ragignore_patterns)
