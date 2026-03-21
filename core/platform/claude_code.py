from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform helpers for Claude Code CLI discovery."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _iter_claude_candidates() -> list[str]:
    """Return plausible Claude Code CLI locations."""
    seen: set[str] = set()
    candidates: list[str] = []

    def _add(p: str) -> None:
        if p and p not in seen:
            seen.add(p)
            candidates.append(p)

    # npm global bin (Windows: %APPDATA%/npm, Unix: varies)
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            npm_bin = Path(appdata) / "npm"
            for name in ("claude.cmd", "claude.exe", "claude"):
                candidate = npm_bin / name
                if candidate.is_file():
                    _add(str(candidate))
    else:
        # Common Unix npm global locations
        home = Path.home()
        for d in (
            home / ".npm-global" / "bin",
            Path("/usr/local/bin"),
            home / ".local" / "bin",
        ):
            candidate = d / "claude"
            if candidate.is_file():
                _add(str(candidate))

    # shutil.which as fallback
    direct = shutil.which("claude")
    if direct:
        _add(direct)

    return candidates


def _is_usable_claude_executable(candidate: str) -> bool:
    """Return True if the candidate responds to --version."""
    try:
        result = subprocess.run(
            [candidate, "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def get_claude_executable() -> str | None:
    """Return the best available Claude Code CLI executable path."""
    for candidate in _iter_claude_candidates():
        if _is_usable_claude_executable(candidate):
            return candidate
    return None


def is_claude_code_available() -> bool:
    """Return True when a Claude Code CLI executable is available."""
    return get_claude_executable() is not None


__all__ = [
    "get_claude_executable",
    "is_claude_code_available",
]
