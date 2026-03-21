from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini CLI detection and authentication status helpers."""

import json
import os
import shutil
from pathlib import Path


def gemini_config_dir() -> Path:
    """Return the default Gemini CLI configuration directory (``~/.gemini``)."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or str(Path.home())
    return Path(home) / ".gemini"


def is_gemini_cli_available() -> bool:
    """Return True when the ``gemini`` CLI is available on PATH."""
    return shutil.which("gemini") is not None


def is_gemini_authenticated() -> bool:
    """Return True when ``GEMINI_API_KEY`` is set or OAuth credentials are present."""
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return True
    oauth_path = gemini_config_dir() / "oauth_creds.json"
    if not oauth_path.is_file():
        return False
    try:
        data = json.loads(oauth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data)
