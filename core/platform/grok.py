from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Grok CLI detection and authentication status helpers."""

import json
import os
import shutil
from pathlib import Path


def grok_config_dir() -> Path:
    """Return the default Grok CLI configuration directory (``~/.grok``)."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or str(Path.home())
    return Path(home) / ".grok"


def get_grok_executable() -> str | None:
    """Return the path to the ``grok`` CLI executable, if available."""
    return shutil.which("grok")


def is_grok_cli_available() -> bool:
    """Return True when the ``grok`` CLI is available on PATH."""
    return get_grok_executable() is not None


def is_grok_authenticated() -> bool:
    """Return True when Grok CLI credentials contain valid, non-empty JSON."""
    auth_path = grok_config_dir() / "auth.json"
    if not auth_path.is_file():
        return False
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data)
