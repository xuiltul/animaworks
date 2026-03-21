from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Cursor Agent CLI detection and authentication status helpers."""

import json
import os
import shutil
from pathlib import Path

_CURSOR_AGENT_BINARY_NAMES = ("agent", "cursor-agent", "cursor")


def cursor_agent_auth_dir() -> Path:
    """Return the default cursor-agent configuration directory (``~/.cursor-agent``)."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or str(Path.home())
    return Path(home) / ".cursor-agent"


def is_cursor_agent_available() -> bool:
    """Return True when any known cursor-agent CLI binary is available on PATH."""
    return any(shutil.which(name) is not None for name in _CURSOR_AGENT_BINARY_NAMES)


def is_cursor_agent_authenticated() -> bool:
    """Return True when credential JSON under ``~/.cursor-agent`` exists and is non-empty."""
    auth_dir = cursor_agent_auth_dir()
    if not auth_dir.is_dir():
        return False
    primary = auth_dir / "auth.json"
    if primary.is_file():
        try:
            data = json.loads(primary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        else:
            if bool(data):
                return True
    for path in auth_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".json" or path.name == "auth.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if bool(data):
            return True
    return False
